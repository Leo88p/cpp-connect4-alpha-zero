// tictactoe_model.cpp
#include "model.h"
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

namespace TicTacToe {

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
        return x + identity;  // Skip connection
    }

    GameNetImpl::GameNetImpl(int num_residual_blocks) : num_blocks_(num_residual_blocks) {
        // Input: (batch, 2, 3, 3) - [current player pieces, opponent pieces]
        conv_in = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(2, NUM_FILTERS, 3).padding(1)),
            torch::nn::BatchNorm2d(NUM_FILTERS),
            torch::nn::LeakyReLU()
        );

        // Create configurable number of residual blocks
        for (int i = 0; i < num_residual_blocks; ++i) {
            residual_blocks_.push_back(ResidualBlock(NUM_FILTERS));
            register_module("block_" + std::to_string(i), residual_blocks_.back());
        }

        // Value head: scalar evaluation [-1, 1]
        conv_val = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_FILTERS, 1, 1)),
            torch::nn::BatchNorm2d(1),
            torch::nn::LeakyReLU(),
            torch::nn::Flatten()
        );

        // Policy head: 9 moves (one per cell)
        conv_policy = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NUM_FILTERS, 2, 1)),
            torch::nn::BatchNorm2d(2),
            torch::nn::LeakyReLU(),
            torch::nn::Flatten()
        );

        // Register core modules
        register_module("conv_in", conv_in);
        register_module("conv_val", conv_val);
        register_module("conv_policy", conv_policy);

        // Determine linear layer sizes using dummy input
        torch::Tensor dummy = torch::zeros({ 1, NUM_FILTERS, BOARD_SIZE, BOARD_SIZE });

        {
            torch::NoGradGuard no_grad;

            // Value head size
            torch::Tensor dummy_val = conv_val->forward(dummy.clone());
            int val_size = dummy_val.size(1);
            val_linear1 = torch::nn::Linear(val_size, 16);  // Reduced from 20 (smaller board)
            val_linear2 = torch::nn::Linear(16, 1);

            // Policy head size
            torch::Tensor dummy_policy = conv_policy->forward(dummy.clone());
            int policy_size = dummy_policy.size(1);
            policy_linear = torch::nn::Linear(policy_size, TOTAL_CELLS);  // 9 outputs
        }

        // Register linear layers
        register_module("val_linear1", val_linear1);
        register_module("val_linear2", val_linear2);
        register_module("policy_linear", policy_linear);
    }

    std::pair<torch::Tensor, torch::Tensor> GameNetImpl::forward(torch::Tensor x) {
        x = conv_in->forward(x);

        // Pass through all residual blocks
        for (auto& block : residual_blocks_) {
            x = block->forward(x);
        }

        // Value head
        torch::Tensor val = conv_val->forward(x);
        val = torch::leaky_relu(val_linear1->forward(val));
        val = torch::tanh(val_linear2->forward(val));  // Output in [-1, 1]

        // Policy head
        torch::Tensor pol = conv_policy->forward(x);
        pol = policy_linear->forward(pol);  // Raw logits for 9 moves

        return { pol, val };
    }

    torch::Tensor state_lists_to_batch(const std::vector<GameState>& states,
        const std::vector<Player>& who_moves,
        const torch::Device& device) {
        size_t batch_size = states.size();
        torch::Tensor batch = torch::zeros({ static_cast<int64_t>(batch_size), 2, BOARD_SIZE, BOARD_SIZE },
            torch::dtype(torch::kFloat32).device(device));

        // Work on CPU first if target is CUDA (for safety/accessor speed)
        torch::Tensor batch_cpu = batch.is_cuda() ? batch.cpu() : batch;
        auto batch_accessor = batch_cpu.accessor<float, 4>();

        for (size_t idx = 0; idx < batch_size; ++idx) {
            const auto& state = states[idx];
            Player current_player = who_moves[idx];

            // Fill channels: [0] = current player pieces, [1] = opponent pieces
            for (int pos = 0; pos < TOTAL_CELLS; ++pos) {
                int row = pos / BOARD_SIZE;
                int col = pos % BOARD_SIZE;
                uint16_t bit = 1 << pos;

                bool is_x = (state.x_pieces & bit) != 0;
                bool is_o = (state.o_pieces & bit) != 0;

                if (is_x || is_o) {
                    bool belongs_to_current =
                        (current_player == Player::X && is_x) ||
                        (current_player == Player::O && is_o);

                    int channel = belongs_to_current ? 0 : 1;
                    batch_accessor[idx][channel][row][col] = 1.0f;
                }
            }
        }

        // Copy back to CUDA if needed
        if (batch.is_cuda()) {
            batch.copy_(batch_cpu);
        }

        return batch;
    }

} // namespace TicTacToe