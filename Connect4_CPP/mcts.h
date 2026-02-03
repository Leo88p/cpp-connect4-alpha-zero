#pragma once
#include <unordered_map>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <memory>
#include <torch/torch.h>

#include "connect4_game.h"
#include "model.h"

namespace Connect4 {

    class MCTSNode {
    public:
        std::array<int, GAME_COLS> visit_count = { 0 };
        std::array<float, GAME_COLS> value = { 0.0f };
        std::array<float, GAME_COLS> value_avg = { 0.0f };
        std::array<float, GAME_COLS> probs = { 0.0f };

        MCTSNode() = default;
        MCTSNode(const MCTSNode& other) = default; // Use default copy constructor

        void reset();
    };

    class MCTS {
    public:
        explicit MCTS(float c_puct = 1.0f);

        void clear();
        size_t size() const;

        std::tuple<float, GameState, Player, std::vector<GameState>, std::vector<int>>
            find_leaf(const GameState& root_state, Player player);

        bool is_leaf(const GameState& state) const;

        void search_batch(int count, int batch_size, const GameState& state,
            Player player, Connect4Net& net, const torch::Device& device);

        void search_minibatch(int count, const GameState& state, Player player,
            Connect4Net& net, const torch::Device& device);

        std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
            get_policy_value(const GameState& state, float tau = 1.0f) const;

    private:
        float c_puct_;
        std::unordered_map<uint64_t, MCTSNode> tree_;

        // Random number generation for Dirichlet noise
        std::mt19937 rng_;
        std::gamma_distribution<float> dirichlet_dist_;

        std::array<float, GAME_COLS> generate_dirichlet_noise();
    };

} // namespace Connect4