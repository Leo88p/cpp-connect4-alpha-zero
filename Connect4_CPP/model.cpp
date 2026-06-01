#include "model.h"
#include <torch/torch.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <execution>

namespace Connect4 {

    ResidualBlockImpl::ResidualBlockImpl(int in_channels) {
        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1)),
            torch::nn::BatchNorm2d(in_channels),
            torch::nn::LeakyReLU()
        );

        conv2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1)),
            torch::nn::BatchNorm2d(in_channels)
        );

        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
        torch::Tensor identity = x;
        x = conv1->forward(x);
        x = conv2->forward(x);
        x = x + identity;
        return torch::leaky_relu(x);
    }

    Connect4NetImpl::Connect4NetImpl(int num_blocks) {
        // Input shape: (2, 6, 7)
        conv_in = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(2, NUM_FILTERS, 3).padding(1)),
            torch::nn::BatchNorm2d(NUM_FILTERS),
            torch::nn::LeakyReLU()
        );

        // Initialize residual blocks
        for (int i = 0; i < num_blocks; i++) {
            residual_blocks->push_back(ResidualBlock(NUM_FILTERS));
        } 

        // Value head
        conv_val = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_FILTERS, 32, 1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::LeakyReLU(),
            torch::nn::Flatten()
        );

        // Policy head
        conv_policy = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_FILTERS, 2, 1)),
            torch::nn::BatchNorm2d(2),
            torch::nn::LeakyReLU(),
            torch::nn::Flatten()
        );

        // Register all modules first
        register_module("conv_in", conv_in);
        register_module("residual_blocks", residual_blocks);
        register_module("conv_val", conv_val);
        register_module("conv_policy", conv_policy);

        // Create dummy input to determine sizes
        torch::Tensor dummy = torch::zeros({ 1, NUM_FILTERS, GAME_ROWS, GAME_COLS });

        {
            torch::NoGradGuard no_grad;

            // Get value head size
            torch::Tensor dummy_val = conv_val->forward(dummy.clone());
            int val_size = dummy_val.size(1);
            val_linear1 = torch::nn::Linear(val_size, 128);
            val_linear2 = torch::nn::Linear(128, 1);
            //val_linear3 = torch::nn::Linear(256, 1);

            // Get policy head size
            torch::Tensor dummy_policy = conv_policy->forward(dummy.clone());
            int policy_size = dummy_policy.size(1);
            policy_linear = torch::nn::Linear(policy_size, GAME_COLS);
        }

        // Register the linear layers
        register_module("val_linear1", val_linear1);
        register_module("val_linear2", val_linear2);
        //register_module("val_linear3", val_linear3);
        register_module("policy_linear", policy_linear);
    }

    std::pair<torch::Tensor, torch::Tensor> Connect4NetImpl::forward(torch::Tensor x) {
        x = conv_in->forward(x);
        
        for (size_t i = 0; i < residual_blocks->size(); ++i) {
            auto block = residual_blocks[i]->as<ResidualBlockImpl>();
            x = block->forward(x);
        } 

        // Value head
        torch::Tensor val = conv_val->forward(x);
        val = torch::leaky_relu(val_linear1->forward(val));
        //val = torch::leaky_relu(val_linear2->forward(val));
        val = torch::tanh(val_linear2->forward(val));

        // Policy head
        torch::Tensor pol = conv_policy->forward(x);
        pol = policy_linear->forward(pol);

        return { pol, val };
    }

    torch::Tensor state_lists_to_batch(const std::vector<GameState>& states,
        const std::vector<Player>& who_moves,
        const torch::Device& device) {

        size_t batch_size = states.size();

        // Pre-allocate on CPU first (faster than direct CUDA allocation)
        torch::Tensor batch = torch::zeros({ static_cast<int64_t>(batch_size), 2, GAME_ROWS, GAME_COLS },
            torch::dtype(torch::kFloat32).device(torch::kCPU));
        auto batch_accessor = batch.accessor<float, 4>();

        for (size_t idx = 0; idx < batch_size; ++idx) {
            const auto& state = states[idx];
            Player who_move = who_moves[idx];

            // Determine which channel is "us" vs "them"
            int our_channel = (who_move == Player::BLACK) ? 0 : 1;
            int their_channel = 1 - our_channel;

            // Direct bit iteration (no nested row/col loops!)
            uint64_t our_pieces = (who_move == Player::BLACK) ? state.black_pieces : state.white_pieces;
            uint64_t their_pieces = (who_move == Player::BLACK) ? state.white_pieces : state.black_pieces;

            // Unroll the 42 positions directly
            for (int pos = 0; pos < 42; ++pos) {
                int row = pos / 7;
                int col = pos % 7;
                int tensor_row = GAME_ROWS - 1 - row;  // Flip vertically

                if (our_pieces & (1ULL << pos)) {
                    batch_accessor[idx][our_channel][tensor_row][col] = 1.0f;
                }
                if (their_pieces & (1ULL << pos)) {
                    batch_accessor[idx][their_channel][tensor_row][col] = 1.0f;
                }
            }
        }

        // Copy to GPU only once at the end (if needed)
        if (device.is_cuda()) {
            batch = batch.to(device);
        }

        return batch;
    }

} // namespace Connect4