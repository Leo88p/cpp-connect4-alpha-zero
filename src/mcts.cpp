#include "mcts.h"
#include <numeric>
#include <execution>
#include <thread>
#include <memory_resource>
#include <unordered_set>

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

    void MCTSNode::reset() {
        std::fill(visit_count.begin(), visit_count.end(), 0);
        std::fill(value.begin(), value.end(), 0.0f);
        std::fill(probs.begin(), probs.end(), 0.0f);
    }

    MCTS::MCTS(float c_puct, float c_fpu, float virtual_loss)
        : c_puct_(c_puct), c_fpu_(c_fpu), virtual_loss_(virtual_loss),
        tree_(&pool_resource_) { // Bind the map to the lock-free pool resource
        std::random_device rd;
        rng_.seed(rd());
        dirichlet_dist_ = std::gamma_distribution<float>(1.4f, 1.0f);
        tree_.reserve(10000);
    }

    void MCTS::clear() {
        tree_.clear();
    }

    size_t MCTS::size() const {
        return tree_.size();
    }

    std::array<float, GAME_COLS> MCTS::generate_dirichlet_noise() {
        std::array<float, GAME_COLS> noise;
        float sum = 0.0f;
        for (int i = 0; i < GAME_COLS; ++i) {
            noise[i] = dirichlet_dist_(rng_);
            sum += noise[i];
        }
        if (sum > 0.0f) {
            for (float& n : noise) n /= sum;
        }
        return noise;
    }

    std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<int>>
        MCTS::find_leaf(const GameState& root_state, Player player,
            std::pmr::vector<std::pair<uint64_t, int>>* virtual_loss_path,
            std::pmr::polymorphic_allocator<void> alloc) {

        std::pmr::vector<GameState> states(alloc);
        std::pmr::vector<int> actions(alloc);
        GameState cur_state = root_state;
        Player cur_player = player;
        float value = std::numeric_limits<float>::quiet_NaN();

        while (!is_leaf(cur_state)) {
            states.push_back(cur_state);
            uint64_t cur_key = cur_state.key();

            const auto& node = tree_.at(cur_key);

            float total_sqrt = std::sqrt(static_cast<float>(std::accumulate(
                node.visit_count.begin(), node.visit_count.end(), 0)));

            std::array<float, GAME_COLS> score;
            const auto& probs = node.probs;
            const auto& counts = node.visit_count;

            if (cur_state.key() == root_state.key() && use_noise) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = node.value_avg(i) + c_puct_ * (0.75f * probs[i] + 0.25f * dirichlet_noise[i]) *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }
            else {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = node.value_avg(i) + c_puct_ * probs[i] *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }

            std::vector<bool> valid_mask(Connect4::GAME_COLS, false);
            for (int col = 0; col < Connect4::GAME_COLS; ++col) {
                valid_mask[col] = cur_state.canPlay(col);
            }

            int best_action = -1;
            float best_score = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < GAME_COLS; ++i) {
                if (!valid_mask[i]) score[i] = -std::numeric_limits<float>::infinity();
                if (score[i] > best_score) {
                    best_score = score[i];
                    best_action = i;
                }
            }

            if (best_action == -1) {
                throw std::runtime_error("No valid moves found in MCTS");
            }

            actions.push_back(best_action);

            if (virtual_loss_path != nullptr) {
                auto& mutable_node = tree_[cur_key];
                mutable_node.value[best_action] -= virtual_loss_;
                mutable_node.visit_count[best_action]++;
                virtual_loss_path->emplace_back(cur_key, best_action);
            }

            if (cur_state.isWinningMove(best_action)) {
                value = -1.0f;
                cur_state.playCol(best_action);
                return { value, cur_state, static_cast<Player>(1 - static_cast<int>(cur_player)),
                         std::move(states), std::move(actions) };
            }
            cur_state.playCol(best_action);
            cur_player = static_cast<Player>(1 - static_cast<int>(cur_player));

            if (cur_state.possible() == 0) {
                value = 0.0f;
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }
        }

        return { std::numeric_limits<float>::quiet_NaN(), cur_state, cur_player,
                 std::move(states), std::move(actions) };
    }

    bool MCTS::is_leaf(const GameState& state) const {
        uint64_t key = state.key();
        return tree_.find(key) == tree_.end();
    }

    void MCTS::search_batch(int count, int batch_size, const GameState& state,
        Player player) {
        if (use_noise) dirichlet_noise = generate_dirichlet_noise();
        for (int i = 0; i < count; ++i) {
            search_minibatch(batch_size, state, player);
        }
    }

    void MCTS::search_minibatch(int count, const GameState& state, Player player) {

        // PMR OPTIMIZATION: Stack-allocated monotonic buffer for per-batch temporary allocations.
        // 8KB is generous for a single minibatch search, preventing ANY heap allocation contention.
        alignas(64) std::byte temp_buffer[8192];
        std::pmr::monotonic_buffer_resource mbr{ temp_buffer, sizeof(temp_buffer), std::pmr::new_delete_resource() };
        std::pmr::polymorphic_allocator<void> alloc(&mbr);

        std::pmr::unordered_map<uint64_t, std::pmr::vector<BackupEntry>> pending_backups{ alloc };
        std::pmr::vector<GameState> expand_states{ alloc };
        std::pmr::vector<Player> expand_players{ alloc };
        std::pmr::unordered_set<uint64_t> planned{ alloc };

        for (int i = 0; i < count; ++i) {
            try {
                std::pmr::vector<std::pair<uint64_t, int>> virtual_loss_path{ alloc };

                auto [value, leaf_state, leaf_player, states, actions] =
                    find_leaf(state, player, &virtual_loss_path, alloc);

                if (!std::isnan(value)) {
                    float terminal_value = value;
                    float cur_value = -terminal_value;

                    for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
                        if (j < static_cast<int>(actions.size()) &&
                            actions[j] >= 0 && actions[j] < GAME_COLS) {

                            uint64_t state_key = states[j].key();
                            if (tree_.find(state_key) == tree_.end()) {
                                tree_[state_key] = MCTSNode();
                            }
                            auto& node = tree_[state_key];

                            if (!virtual_loss_path.empty() &&
                                virtual_loss_path.back().first == state_key &&
                                virtual_loss_path.back().second == actions[j]) {
                                node.value[actions[j]] += virtual_loss_;
                                node.visit_count[actions[j]]--;
                                virtual_loss_path.pop_back();
                            }

                            node.visit_count[actions[j]]++;
                            node.value[actions[j]] += cur_value;

                            cur_value = -cur_value;
                        }
                    }
                }
                else {
                    uint64_t leaf_key = leaf_state.key();

                    // Construct BackupEntry with the monotonic allocator
                    pending_backups[leaf_key].emplace_back(alloc);
                    auto& new_entry = pending_backups[leaf_key].back();

                    // Move assignment is O(1) pointer swap because allocators match
                    new_entry.states = std::move(states);
                    new_entry.actions = std::move(actions);
                    new_entry.virtual_loss_path = std::move(virtual_loss_path);

                    if (planned.find(leaf_key) == planned.end()) {
                        planned.insert(leaf_key);
                        expand_states.push_back(leaf_state);
                        expand_players.push_back(leaf_player);
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "MCTS search error: " << e.what() << std::endl;
                continue;
            }
        }

        if (!expand_states.empty()) {
            try {
                std::pmr::vector<std::future<std::pair<std::array<float, GAME_COLS>, float>>> futures{ alloc };
                futures.reserve(expand_states.size());

                if (neural_worker_) {
                    for (size_t i = 0; i < expand_states.size(); ++i) {
                        futures.push_back(neural_worker_->submit_query(
                            expand_states[i], expand_players[i]));
                    }
                }

                for (size_t i = 0; i < expand_states.size(); ++i) {
                    const auto& leaf_state = expand_states[i];
                    uint64_t leaf_key = leaf_state.key();

                    std::array<float, 7> probs;
                    float value;

                    if (neural_worker_) {
                        auto [probs_, value_] = futures[i].get();
                        probs = probs_;
                        value = value_;
                    }
                    else {
                        int count = 0;
                        for (int col = 0; col < Connect4::GAME_COLS; ++col) {
                            probs[col] = leaf_state.canPlay(col) ? 1 : 0;
                            count++;
                        }
                        for (int col = 0; col < Connect4::GAME_COLS; ++col) {
                            if (count > 1e-8f) {
                                probs[col] /= count;
                            }
                        }
                        value = 0;
                    }

                    if (tree_.find(leaf_key) == tree_.end()) {
                        MCTSNode node;
                        std::vector<bool> valid_mask(Connect4::GAME_COLS, false);
                        for (int col = 0; col < Connect4::GAME_COLS; ++col) {
                            valid_mask[col] = leaf_state.canPlay(col);
                        }

                        for (int j = 0; j < GAME_COLS; ++j) {
                            node.probs[j] = valid_mask[j] ? probs[j] : 0.0f;
                            node.visit_count[j] = 0;
                            node.value[j] = 0.0f;
                        }

                        float sum = std::accumulate(node.probs.begin(), node.probs.end(), 0.0f);
                        if (sum > 1e-8f) {
                            for (float& p : node.probs) p /= sum;
                        }
                        tree_[leaf_key] = std::move(node);
                    }

                    auto it = pending_backups.find(leaf_key);
                    if (it != pending_backups.end()) {
                        for (auto& entry : it->second) {
                            float cur_value = -value;

                            for (int j = static_cast<int>(entry.states.size()) - 1; j >= 0; --j) {
                                if (j < static_cast<int>(entry.actions.size()) &&
                                    entry.actions[j] >= 0 && entry.actions[j] < GAME_COLS) {

                                    uint64_t state_key = entry.states[j].key();
                                    if (tree_.find(state_key) == tree_.end()) {
                                        tree_[state_key] = MCTSNode();
                                    }
                                    auto& node = tree_[state_key];

                                    if (!entry.virtual_loss_path.empty() &&
                                        entry.virtual_loss_path.back().first == state_key &&
                                        entry.virtual_loss_path.back().second == entry.actions[j]) {
                                        node.value[entry.actions[j]] += virtual_loss_;
                                        node.visit_count[entry.actions[j]]--;
                                        entry.virtual_loss_path.pop_back();
                                    }

                                    node.visit_count[entry.actions[j]]++;
                                    node.value[entry.actions[j]] += cur_value;

                                    cur_value = -cur_value;
                                }
                            }
                        }
                        pending_backups.erase(it);
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Neural network expansion error: " << e.what() << std::endl;
                for (auto& [leaf_key, backups] : pending_backups) {
                    for (auto& entry : backups) {
                        for (auto& [key, action] : entry.virtual_loss_path) {
                            if (tree_.find(key) != tree_.end()) {
                                tree_[key].value[action] += virtual_loss_;
                                tree_[key].visit_count[action]--;
                            }
                        }
                    }
                }
                return;
            }
        }
    }

    std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
        MCTS::get_policy_value(const GameState& state, float tau) const {
        uint64_t state_key = state.key();
        const auto& node = tree_.at(state_key);

        std::array<float, GAME_COLS> probs;
        std::array<float, GAME_COLS> values;

        if (tau == 0.0f) {
            std::fill(probs.begin(), probs.end(), 0.0f);
            int best_action = std::distance(node.visit_count.begin(),
                std::max_element(node.visit_count.begin(), node.visit_count.end()));
            probs[best_action] = 1.0f;
        }
        else {
            std::array<float, GAME_COLS> counts_pow;
            float sum = 0.0f;
            for (int i = 0; i < GAME_COLS; ++i) {
                counts_pow[i] = std::pow(static_cast<float>(node.visit_count[i]), 1.0f / tau);
                sum += counts_pow[i];
            }
            if (sum > 0.0f) {
                for (int i = 0; i < GAME_COLS; ++i) probs[i] = counts_pow[i] / sum;
            }
            else {
                float uniform_prob = 1.0f / GAME_COLS;
                std::fill(probs.begin(), probs.end(), uniform_prob);
            }
        }

        for (int i = 0; i < GAME_COLS; ++i) values[i] = node.value_avg(i);
        return { probs, values };
    }
}