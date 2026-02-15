// tictactoe_model.h
#pragma once
#include <torch/torch.h>
#include <vector>
#include "game.h"

namespace TicTacToe {

    constexpr int NUM_FILTERS = 32;  // Reduced from Connect4 (sufficient for 3x3 board)

    class ResidualBlockImpl : public torch::nn::Module {
    public:
        ResidualBlockImpl(int in_channels);
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Sequential conv1{ nullptr };
        torch::nn::Sequential conv2{ nullptr };
    };

    TORCH_MODULE(ResidualBlock);

    class GameNetImpl : public torch::nn::Module {
    public:
        // Configurable residual block count
        explicit GameNetImpl(int num_residual_blocks = 2);
        std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

        // Accessor for block count (useful for logging/checkpointing)
        int get_num_blocks() const { return num_blocks_; }

    private:
        int num_blocks_;
        torch::nn::Sequential conv_in{ nullptr };
        std::vector<ResidualBlock> residual_blocks_;  // Dynamic block storage
        torch::nn::Sequential conv_val{ nullptr };
        torch::nn::Sequential conv_policy{ nullptr };
        torch::nn::Linear val_linear1{ nullptr };
        torch::nn::Linear val_linear2{ nullptr };
        torch::nn::Linear policy_linear{ nullptr };
    };

    TORCH_MODULE(GameNet);

    // Convert game states to batch tensor (2 channels: current player pieces, opponent pieces)
    torch::Tensor state_lists_to_batch(const std::vector<GameState>& states,
        const std::vector<Player>& who_moves,
        const torch::Device& device);

} // namespace TicTacToe