#pragma once
#include <torch/torch.h>
#include <vector>
#include <array>
#include <deque>
#include <optional>
#include <unordered_map>

#include "connect4_game.h"

namespace Connect4 {

    constexpr int NUM_FILTERS = 64;

    class ResidualBlockImpl : public torch::nn::Module {
    public:
        ResidualBlockImpl(int in_channels);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Sequential conv1 = nullptr;
        torch::nn::Sequential conv2 = nullptr;
    };

    TORCH_MODULE(ResidualBlock);

    class Connect4NetImpl : public torch::nn::Module {
    public:
        Connect4NetImpl(int num_blocks = 5);

        std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    private:
        torch::nn::Sequential conv_in = nullptr;
        torch::nn::ModuleList residual_blocks;

        torch::nn::Sequential conv_val = nullptr;
        torch::nn::Linear val_linear1 = nullptr;
        torch::nn::Linear val_linear2 = nullptr;

        torch::nn::Sequential conv_policy = nullptr;
        torch::nn::Linear policy_linear = nullptr;
    };

    TORCH_MODULE(Connect4Net);

    // Function to convert game states to batch tensor
    torch::Tensor state_lists_to_batch(const std::vector<GameState>& states,
        const std::vector<Player>& who_moves,
        const torch::Device& device = torch::kCPU);

} // namespace Connect4