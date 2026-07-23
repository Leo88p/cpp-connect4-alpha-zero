#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <queue>
#include <memory_resource>
#include <torch/torch.h>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <bit>

#include "connect4_game.h" // Assumes this includes the updated GameState
#include "model.h"
#include "neural_worker.h"

namespace Connect4 {

    struct ChildStats {
        uint64_t move = 0;          // The actual bitboard move
        int visit_count = 0;
        float value = 0.0f;
        float value_avg = 0.0f;
        float prob = 0.0f;
    };

    class MCTSNode {
    public:
        std::array<ChildStats, GAME_COLS> children;
        int num_children = 0; // Only iterate up to this number

        MCTSNode() = default;

        void reset() {
            for (int i = 0; i < num_children; ++i) {
                children[i].visit_count = 0;
                children[i].value = 0.0f;
                children[i].value_avg = 0.0f;
                children[i].prob = 0.0f;
            }
        }
    };

    class MCTS {
    public:
        explicit MCTS(float c_puct = 1.0f, float c_fpu = 0.25f, float virtual_loss = 2.0f);
        MCTS(const MCTS&) = delete;
        MCTS& operator=(const MCTS&) = delete;
        MCTS(MCTS&&) = default;
        MCTS& operator=(MCTS&&) = default;

        bool use_noise = true;

        void clear();
        size_t size() const;

        // Note: actions vector now stores uint64_t move bitboards, not int indices
        std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<uint64_t>>
            find_leaf(const GameState& root_state, Player player,
                std::pmr::vector<std::pair<uint64_t, uint64_t>>* virtual_loss_path,
                std::pmr::polymorphic_allocator<void> alloc);

        bool is_leaf(const GameState& state) const;

        void search_batch(int count, int batch_size, const GameState& state,
            Player player, Connect4Net& net, const torch::Device& device);

        void search_minibatch(int count, const GameState& state, Player player,
            Connect4Net& net, const torch::Device& device);

        // Returns arrays indexed by column (0-6) for compatibility with training pipeline
        std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
            get_policy_value(const GameState& state, float tau = 1.0f) const;

        NeuralWorker* neural_worker_ = nullptr;
        void set_neural_worker(NeuralWorker* worker) {
            neural_worker_ = worker;
        }

    private:
        std::pmr::unsynchronized_pool_resource pool_resource_;
        std::pmr::unordered_map<uint64_t, MCTSNode> tree_;
        float c_puct_;
        float c_fpu_;
        float virtual_loss_;

        std::mt19937 rng_;
        std::gamma_distribution<float> dirichlet_dist_;
        std::array<float, GAME_COLS> dirichlet_noise;

        std::array<float, GAME_COLS> generate_dirichlet_noise();
    };
}