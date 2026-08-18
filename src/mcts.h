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

    // PMR-enabled helper struct to ensure internal vectors also use the fast allocator
    struct BackupEntry {
        std::pmr::vector<GameState> states;
        std::pmr::vector<int> actions;
        std::pmr::vector<std::pair<uint64_t, int>> virtual_loss_path;

        explicit BackupEntry(std::pmr::polymorphic_allocator<void> alloc)
            : states(alloc), actions(alloc), virtual_loss_path(alloc) {
        }
    };

    class MCTSNode {
    public:
        std::array<int, GAME_COLS> visit_count = { 0 };
        std::array<float, GAME_COLS> value = { 0.0f };
        std::array<float, GAME_COLS> probs = { 0.0f };
        float value_avg(int i) const {
            return visit_count[i] > 0 ? value[i] / visit_count[i] : 0;
        }

        MCTSNode() = default;
        MCTSNode(const MCTSNode& other) = default; // Use default copy constructor

        virtual void reset();
    };

    class MCTS {
    public:
        explicit MCTS(float c_puct = 1.0f, float dirichlet_alpha = 1.0f, float dirichlet_epsilon = 0.25f, float virtual_loss = 2.0f, int tree_size = 10000);
        virtual ~MCTS() = default;
        MCTS(const MCTS&) = delete;
        MCTS& operator=(const MCTS&) = delete;
        MCTS(MCTS&&) = default;
        MCTS& operator=(MCTS&&) = default;

        bool use_noise = true;

        virtual void clear();
        virtual size_t size() const;

        virtual std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<int>>
            find_leaf(const GameState& root_state, Player player,
                std::pmr::vector<std::pair<uint64_t, int>>* virtual_loss_path,
                std::pmr::polymorphic_allocator<void> alloc);

        virtual bool is_leaf(const GameState& state) const;

        virtual void search_batch(int count, int batch_size, const GameState& state, Player player);
        virtual void search_minibatch(int count, const GameState& state, Player player);

        virtual std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
            get_policy_value(const GameState& state, float tau = 1.0f) const;

        NeuralWorker* neural_worker_ = nullptr;
        void set_neural_worker(NeuralWorker* worker) { neural_worker_ = worker; }
        std::array<bool, Connect4::GAME_COLS> get_valid_mask(const GameState& state) const;
        void set_c_puct(float new_c_puct) { c_puct_ = new_c_puct; }

    protected:
        std::pmr::unsynchronized_pool_resource pool_resource_;
        std::pmr::unordered_map<uint64_t, MCTSNode> tree_;
        float c_puct_;
        float dirichlet_alpha;
        float dirichlet_epsilon;
        float virtual_loss_;

        std::mt19937 rng_;
        std::gamma_distribution<float> dirichlet_dist_;

        std::array<float, GAME_COLS> generate_dirichlet_noise();
        std::array<float, GAME_COLS> dirichlet_noise;
    };
}