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
            torch::nn::BatchNorm2d(in_channels),
            torch::nn::LeakyReLU()
        );

        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
        torch::Tensor identity = x;
        x = conv1->forward(x);
        x = conv2->forward(x);
        return x + identity;
    }

    Connect4NetImpl::Connect4NetImpl() {
        // Input shape: (2, 6, 7)
        conv_in = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(2, NUM_FILTERS, 3).padding(1)),
            torch::nn::BatchNorm2d(NUM_FILTERS),
            torch::nn::LeakyReLU()
        );

        // Initialize residual blocks
        block1 = ResidualBlock(NUM_FILTERS);
        block2 = ResidualBlock(NUM_FILTERS);
        block3 = ResidualBlock(NUM_FILTERS);
        block4 = ResidualBlock(NUM_FILTERS);
        block5 = ResidualBlock(NUM_FILTERS);

        // Value head
        conv_val = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_FILTERS, 1, 1)),
            torch::nn::BatchNorm2d(1),
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
        register_module("block1", block1);
        register_module("block2", block2);
        register_module("block3", block3);
        register_module("block4", block4);
        register_module("block5", block5);
        register_module("conv_val", conv_val);
        register_module("conv_policy", conv_policy);

        // Create dummy input to determine sizes
        torch::Tensor dummy = torch::zeros({ 1, NUM_FILTERS, GAME_ROWS, GAME_COLS });

        {
            torch::NoGradGuard no_grad;

            // Get value head size
            torch::Tensor dummy_val = conv_val->forward(dummy.clone());
            int val_size = dummy_val.size(1);
            val_linear1 = torch::nn::Linear(val_size, 20);
            val_linear2 = torch::nn::Linear(20, 1);

            // Get policy head size
            torch::Tensor dummy_policy = conv_policy->forward(dummy.clone());
            int policy_size = dummy_policy.size(1);
            policy_linear = torch::nn::Linear(policy_size, GAME_COLS);
        }

        // Register the linear layers
        register_module("val_linear1", val_linear1);
        register_module("val_linear2", val_linear2);
        register_module("policy_linear", policy_linear);
    }

    std::pair<torch::Tensor, torch::Tensor> Connect4NetImpl::forward(torch::Tensor x) {
        x = conv_in->forward(x);
        x = block1->forward(x);
        x = block2->forward(x);
        x = block3->forward(x);
        x = block4->forward(x);
        x = block5->forward(x);

        // Value head
        torch::Tensor val = conv_val->forward(x);
        val = torch::leaky_relu(val_linear1->forward(val));
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
        torch::Tensor batch = torch::zeros({ static_cast<int64_t>(batch_size), 2, GAME_ROWS, GAME_COLS },
            torch::dtype(torch::kFloat32).device(device));

        auto batch_accessor = batch.accessor<float, 4>();

        for (size_t idx = 0; idx < batch_size; ++idx) {
            const auto& state = states[idx];
            Player who_move = who_moves[idx];

            // Fill the tensor based on game state
            for (int col = 0; col < GAME_COLS; ++col) {
                for (int row = 0; row < state.heights[col]; ++row) {
                    int tensor_row = GAME_ROWS - 1 - row;
                    int pos = row * GAME_COLS + col;
                    uint64_t bit = 1ULL << pos;

                    bool is_black = (state.black_pieces & bit) != 0;
                    bool is_white = (state.white_pieces & bit) != 0;

                    if (is_black) {
                        if (who_move == Player::BLACK) {
                            batch_accessor[idx][0][tensor_row][col] = 1.0f;
                        }
                        else {
                            batch_accessor[idx][1][tensor_row][col] = 1.0f;
                        }
                    }
                    else if (is_white) {
                        if (who_move == Player::WHITE) {
                            batch_accessor[idx][0][tensor_row][col] = 1.0f;
                        }
                        else {
                            batch_accessor[idx][1][tensor_row][col] = 1.0f;
                        }
                    }
                }
            }
        }

        return batch;
    }

} // namespace Connect4