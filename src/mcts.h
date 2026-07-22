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
#include <unordered_map>   // CRITICAL: Required for std::pmr::unordered_map
#include <unordered_set> 

#include "connect4_game.h"
#include "model.h"
#include "neural_worker.h"

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
        explicit MCTS(float c_puct = 1.0f, float c_fpu = 0.25f, float virtual_loss = 2.0f);
        MCTS(const MCTS&) = delete;
        MCTS& operator=(const MCTS&) = delete;
        MCTS(MCTS&&) = default;
        MCTS& operator=(MCTS&&) = default;
        
        bool use_noise = true;

        void clear();
        size_t size() const;

        std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<int>>
            find_leaf(const GameState& root_state, Player player,
                std::pmr::vector<std::pair<uint64_t, int>>* virtual_loss_path,
                std::pmr::polymorphic_allocator<void> alloc);

        bool is_leaf(const GameState& state) const;

        void search_batch(int count, int batch_size, const GameState& state,
            Player player, Connect4Net& net, const torch::Device& device);

        void search_minibatch(int count, const GameState& state, Player player,
            Connect4Net& net, const torch::Device& device);

        std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
            get_policy_value(const GameState& state, float tau = 1.0f) const;
        NeuralWorker* neural_worker_ = nullptr;
        void set_neural_worker(NeuralWorker* worker) {
            neural_worker_ = worker;
        }

    private:
        // Lock-free, thread-safe (when used by a single thread) memory pool for the tree
        std::pmr::unsynchronized_pool_resource pool_resource_;

        // PMR-enabled unordered map
        std::pmr::unordered_map<uint64_t, MCTSNode> tree_;
        float c_puct_;
        float c_fpu_;
        float virtual_loss_;

        // Random number generation for Dirichlet noise
        std::mt19937 rng_;
        std::gamma_distribution<float> dirichlet_dist_;

        std::array<float, GAME_COLS> generate_dirichlet_noise();
        std::array<float, GAME_COLS> dirichlet_noise;
    };
}