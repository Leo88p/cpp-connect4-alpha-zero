#include "mcts.h"
#include <numeric>
#include <execution>
#include <thread>
#include <memory_resource>
#include <unordered_set>

namespace Connect4 {

    // PMR-enabled helper struct for batched backups
    struct BackupEntry {
        std::pmr::vector<GameState> states;
        std::pmr::vector<uint64_t> actions; // Changed from int to uint64_t
        std::pmr::vector<std::pair<uint64_t, uint64_t>> virtual_loss_path; // state_key, move_bitboard

        explicit BackupEntry(std::pmr::polymorphic_allocator<void> alloc)
            : states(alloc), actions(alloc), virtual_loss_path(alloc) {
        }
    };

    MCTS::MCTS(float c_puct, float c_fpu, float virtual_loss)
        : c_puct_(c_puct), c_fpu_(c_fpu), virtual_loss_(virtual_loss),
        tree_(&pool_resource_) {
        std::random_device rd;
        rng_.seed(rd());
        dirichlet_dist_ = std::gamma_distribution<float>(0.3f, 1.0f);
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

    std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<uint64_t>>
        MCTS::find_leaf(const GameState& root_state, Player player,
            std::pmr::vector<std::pair<uint64_t, uint64_t>>* virtual_loss_path,
            std::pmr::polymorphic_allocator<void> alloc) {

        std::pmr::vector<GameState> states(alloc);
        std::pmr::vector<uint64_t> actions(alloc); // Now stores bitboards
        GameState cur_state = root_state;
        Player cur_player = player;
        float value = std::numeric_limits<float>::quiet_NaN();

        while (!is_leaf(cur_state)) {
            states.push_back(cur_state);
            uint64_t cur_key = cur_state.key();
            const auto& node = tree_.at(cur_key);

            int total_visits = 0;
            for (int i = 0; i < node.num_children; ++i) {
                total_visits += node.children[i].visit_count;
            }
            float total_sqrt = std::sqrt(static_cast<float>(total_visits));

            int best_child_idx = -1;
            float best_score = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < node.num_children; ++i) {
                const auto& child = node.children[i];
                float score = child.value_avg + c_puct_ * child.prob * total_sqrt / (1.0f + static_cast<float>(child.visit_count));

                if (cur_key == root_state.key() && use_noise) {
                    int col = std::countr_zero(child.move) / (GAME_ROWS + 1);
                    score = child.value_avg + c_puct_ * (0.75f * child.prob + 0.25f * dirichlet_noise[col]) *
                        total_sqrt / (1.0f + static_cast<float>(child.visit_count));
                }

                if (score > best_score) {
                    best_score = score;
                    best_child_idx = i;
                }
            }

            if (best_child_idx == -1) {
                throw std::runtime_error("No valid moves found in MCTS node");
            }

            const auto& best_child = node.children[best_child_idx];
            uint64_t best_move = best_child.move;
            actions.push_back(best_move);

            if (virtual_loss_path != nullptr) {
                auto& mutable_node = tree_.at(cur_key);
                mutable_node.children[best_child_idx].value -= virtual_loss_;
                virtual_loss_path->emplace_back(cur_key, best_move);
            }

            // Check win using the bitboard directly!
            if (cur_state.isWinningMove(best_move)) {
                value = 1.0f;
                cur_state.play(best_move);
                return { value, cur_state, static_cast<Player>(1 - static_cast<int>(cur_player)),
                         std::move(states), std::move(actions) };
            }

            cur_state.play(best_move);
            cur_player = static_cast<Player>(1 - static_cast<int>(cur_player));

            // Check draw using bitboard (no possible moves left)
            if (cur_state.possible() == 0) {
                value = 0.0f;
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }
        }

        return { std::numeric_limits<float>::quiet_NaN(), cur_state, cur_player,
                 std::move(states), std::move(actions) };
    }

    bool MCTS::is_leaf(const GameState& state) const {
        return tree_.find(state.key()) == tree_.end();
    }

    void MCTS::search_batch(int count, int batch_size, const GameState& state,
        Player player, Connect4Net& net, const torch::Device& device) {
        if (use_noise) dirichlet_noise = generate_dirichlet_noise();
        for (int i = 0; i < count; ++i) {
            search_minibatch(batch_size, state, player, net, device);
        }
    }

    void MCTS::search_minibatch(int count, const GameState& state, Player player,
        Connect4Net& net, const torch::Device& device) {

        alignas(64) std::byte temp_buffer[8192];
        std::pmr::monotonic_buffer_resource mbr{ temp_buffer, sizeof(temp_buffer), std::pmr::new_delete_resource() };
        std::pmr::polymorphic_allocator<void> alloc(&mbr);

        std::pmr::unordered_map<uint64_t, std::pmr::vector<BackupEntry>> pending_backups{ alloc };
        std::pmr::vector<GameState> expand_states{ alloc };
        std::pmr::vector<Player> expand_players{ alloc };
        std::pmr::unordered_set<uint64_t> planned{ alloc };

        for (int i = 0; i < count; ++i) {
            try {
                std::pmr::vector<std::pair<uint64_t, uint64_t>> virtual_loss_path{ alloc };

                auto [value, leaf_state, leaf_player, states, actions] =
                    find_leaf(state, player, &virtual_loss_path, alloc);

                if (!std::isnan(value)) {
                    float terminal_value = value;
                    float cur_value = -terminal_value;

                    for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
                        if (j < static_cast<int>(actions.size())) {
                            uint64_t state_key = states[j].key();
                            uint64_t move = actions[j];

                            auto& node = tree_.at(state_key);

                            // Find the child matching this move (max 7 iterations, extremely fast)
                            for (int k = 0; k < node.num_children; ++k) {
                                if (node.children[k].move == move) {
                                    if (!virtual_loss_path.empty() &&
                                        virtual_loss_path.back().first == state_key &&
                                        virtual_loss_path.back().second == move) {
                                        node.children[k].value += virtual_loss_;
                                        virtual_loss_path.pop_back();
                                    }

                                    node.children[k].visit_count++;
                                    node.children[k].value += cur_value;
                                    node.children[k].value_avg = node.children[k].value /
                                        static_cast<float>(node.children[k].visit_count);
                                    break;
                                }
                            }
                            cur_value = -cur_value;
                        }
                    }
                }
                else {
                    uint64_t leaf_key = leaf_state.key();
                    pending_backups[leaf_key].emplace_back(alloc);
                    auto& new_entry = pending_backups[leaf_key].back();

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

                for (size_t i = 0; i < expand_states.size(); ++i) {
                    futures.push_back(neural_worker_->submit_query(expand_states[i], expand_players[i]));
                }

                for (size_t i = 0; i < expand_states.size(); ++i) {
                    const auto& leaf_state = expand_states[i];
                    uint64_t leaf_key = leaf_state.key();

                    auto [probs, value] = futures[i].get();

                    if (tree_.find(leaf_key) == tree_.end()) {
                        MCTSNode new_node;
                        new_node.num_children = 0;

                        uint64_t possible_moves = leaf_state.possible();
                        float prob_sum = 0.0f;

                        // Expand ONLY valid moves, storing the bitboard directly
                        while (possible_moves) {
                            uint64_t move = possible_moves & -possible_moves;
                            int col = std::countr_zero(move) / (GAME_ROWS + 1); // Only translation point

                            auto& child = new_node.children[new_node.num_children++];
                            child.move = move;
                            child.prob = probs[col];
                            child.visit_count = 0;
                            child.value = 0.0f;
                            child.value_avg = 0.0f;

                            prob_sum += child.prob;
                            possible_moves ^= move;
                        }

                        // Normalize probabilities
                        if (prob_sum > 1e-8f) {
                            for (int k = 0; k < new_node.num_children; ++k) {
                                new_node.children[k].prob /= prob_sum;
                            }
                        }
                        else {
                            float uniform_prob = 1.0f / new_node.num_children;
                            for (int k = 0; k < new_node.num_children; ++k) {
                                new_node.children[k].prob = uniform_prob;
                            }
                        }

                        tree_[leaf_key] = std::move(new_node);
                    }

                    auto it = pending_backups.find(leaf_key);
                    if (it != pending_backups.end()) {
                        for (auto& entry : it->second) {
                            float cur_value = -value;

                            for (int j = static_cast<int>(entry.states.size()) - 1; j >= 0; --j) {
                                if (j < static_cast<int>(entry.actions.size())) {
                                    uint64_t state_key = entry.states[j].key();
                                    uint64_t move = entry.actions[j];

                                    auto& node = tree_.at(state_key);

                                    for (int k = 0; k < node.num_children; ++k) {
                                        if (node.children[k].move == move) {
                                            if (!entry.virtual_loss_path.empty() &&
                                                entry.virtual_loss_path.back().first == state_key &&
                                                entry.virtual_loss_path.back().second == move) {
                                                node.children[k].value += virtual_loss_;
                                                entry.virtual_loss_path.pop_back();
                                            }

                                            node.children[k].visit_count++;
                                            node.children[k].value += cur_value;
                                            node.children[k].value_avg = node.children[k].value /
                                                static_cast<float>(node.children[k].visit_count);
                                            break;
                                        }
                                    }
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
                        for (auto& [key, move] : entry.virtual_loss_path) {
                            if (tree_.find(key) != tree_.end()) {
                                auto& node = tree_.at(key);
                                for (int k = 0; k < node.num_children; ++k) {
                                    if (node.children[k].move == move) {
                                        node.children[k].value += virtual_loss_;
                                        break;
                                    }
                                }
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

        std::array<float, GAME_COLS> probs = { 0.0f };
        std::array<float, GAME_COLS> values = { 0.0f };

        if (tau == 0.0f) {
            int best_child_idx = 0;
            int max_visits = -1;
            for (int i = 0; i < node.num_children; ++i) {
                if (node.children[i].visit_count > max_visits) {
                    max_visits = node.children[i].visit_count;
                    best_child_idx = i;
                }
            }
            if (max_visits > 0) {
                int col = std::countr_zero(node.children[best_child_idx].move) / (GAME_ROWS + 1);
                probs[col] = 1.0f;
                values[col] = node.children[best_child_idx].value_avg;
            }
        }
        else {
            float sum_counts_pow = 0.0f;
            std::array<float, GAME_COLS> counts_pow = { 0.0f };

            for (int i = 0; i < node.num_children; ++i) {
                int col = std::countr_zero(node.children[i].move) / (GAME_ROWS + 1);
                counts_pow[col] = std::pow(static_cast<float>(node.children[i].visit_count), 1.0f / tau);
                sum_counts_pow += counts_pow[col];
                values[col] = node.children[i].value_avg;
            }

            if (sum_counts_pow > 0.0f) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    probs[i] = counts_pow[i] / sum_counts_pow;
                }
            }
            else {
                int valid_count = node.num_children > 0 ? node.num_children : GAME_COLS;
                float uniform_prob = 1.0f / valid_count;
                for (int i = 0; i < node.num_children; ++i) {
                    int col = std::countr_zero(node.children[i].move) / (GAME_ROWS + 1);
                    probs[col] = uniform_prob;
                }
            }
        }
        return { probs, values };
    }
}