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

    Connect4NetImpl::Connect4NetImpl(int num_blocks, int num_filters) {
        // Input shape: (2, 6, 7)
        conv_in = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(2, num_filters, 3).padding(1)),
            torch::nn::BatchNorm2d(num_filters),
            torch::nn::LeakyReLU()
        );

        // Initialize residual blocks
        for (int i = 0; i < num_blocks; i++) {
            residual_blocks->push_back(ResidualBlock(num_filters));
        } 

        // Value head
        conv_val = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 32, 1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::LeakyReLU(),
            torch::nn::Flatten()
        );

        // Policy head
        conv_policy = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 2, 1)),
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
        torch::Tensor dummy = torch::zeros({ 1, num_filters, GAME_ROWS, GAME_COLS });

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

    void state_lists_to_batches(torch::Tensor& cpu_buffer,
        const std::vector<GameState>& states,
        const std::vector<Player>& who_moves) {
        TORCH_CHECK(cpu_buffer.device().is_cpu(), "fill_batch_buffer requires a CPU tensor");
        TORCH_CHECK(cpu_buffer.is_contiguous(), "Buffer must be contiguous");

        float* ptr = cpu_buffer.data_ptr<float>();
        const int channel_stride = GAME_ROWS * GAME_COLS; // 6*7 = 42
        const int sample_stride = 2 * channel_stride;

        for (size_t idx = 0; idx < states.size(); ++idx) {
            const auto& state = states[idx];
            Player who_move = who_moves[idx];

            int our_channel = 0;
            int their_channel = 1;

            uint64_t our_pieces = (who_move == Player::BLACK) ? state.black_pieces : state.white_pieces;
            uint64_t their_pieces = (who_move == Player::BLACK) ? state.white_pieces : state.black_pieces;

            float* our_plane = ptr + idx * sample_stride + our_channel * channel_stride;
            float* their_plane = ptr + idx * sample_stride + their_channel * channel_stride;

            // Zero out planes (CPU-safe)
            std::memset(our_plane, 0, channel_stride * sizeof(float));
            std::memset(their_plane, 0, channel_stride * sizeof(float));

            for (int pos = 0; pos < 42; ++pos) {
                int row = pos / 7;
                int col = pos % 7;
                int tensor_idx = (GAME_ROWS - 1 - row) * GAME_COLS + col; // Flip vertically

                if (our_pieces & (1ULL << pos))   our_plane[tensor_idx] = 1.0f;
                if (their_pieces & (1ULL << pos)) their_plane[tensor_idx] = 1.0f;
            }
        }
    }

} // namespace Connect4