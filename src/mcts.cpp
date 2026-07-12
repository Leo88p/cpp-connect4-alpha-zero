#include "mcts.h"
#include <numeric>
#include <execution>
#include <thread>

namespace Connect4 {

    // === VIRTUAL LOSS CONFIGURATION ===
    constexpr float VIRTUAL_LOSS = 2.0f;  // Typical range: 1.0–3.0

    // Helper struct to bundle backup data + virtual loss path
    struct BackupEntry {
        std::vector<GameState> states;
        std::vector<int> actions;
        std::vector<std::pair<uint64_t, int>> virtual_loss_path;  // Track penalized edges

        // Constructor for easy emplace_back
        BackupEntry(std::vector<GameState> s, std::vector<int> a,
            std::vector<std::pair<uint64_t, int>> vlp)
            : states(std::move(s)), actions(std::move(a)), virtual_loss_path(std::move(vlp)) {
        }
    };

    MCTS::MCTS(float c_puct) : c_puct_(c_puct) {
        std::random_device rd;
        rng_.seed(rd());
        dirichlet_dist_ = std::gamma_distribution<float>(0.3f, 1.0f);
        tree_.reserve(10000);
    }

    MCTS::MCTS(const MCTS& other)
        : c_puct_(other.c_puct_),
        tree_(other.tree_),
        rng_(other.rng_),
        dirichlet_dist_(other.dirichlet_dist_) {
        std::random_device rd;
        rng_.seed(rd());
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

    // === ENHANCED: find_leaf with virtual loss tracking ===
    std::tuple<float, GameState, Player, std::vector<GameState>, std::vector<int>>
        MCTS::find_leaf(const GameState& root_state, Player player,
            std::vector<std::pair<uint64_t, int>>* virtual_loss_path) {

        std::vector<GameState> states;
        std::vector<int> actions;
        GameState cur_state = root_state;
        Player cur_player = player;
        float value = std::numeric_limits<float>::quiet_NaN();

        while (!is_leaf(cur_state)) {
            states.push_back(cur_state);
            uint64_t cur_key = cur_state.hash_key;

            // Safe access: node must exist if !is_leaf()
            const auto& node = tree_.at(cur_key);

            float total_sqrt = std::sqrt(static_cast<float>(std::accumulate(
                node.visit_count.begin(), node.visit_count.end(), 0)));

            std::array<float, GAME_COLS> score;
            const auto& probs = node.probs;
            const auto& counts = node.visit_count;

            // Choose action via UCB (with optional Dirichlet noise at root)
            if (cur_state.hash_key == root_state.hash_key && use_noise) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = node.get_value_avg(i) + c_puct_ * (0.75f * probs[i] + 0.25f * dirichlet_noise[i]) *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }
            else {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = node.get_value_avg(i) + c_puct_ * probs[i] *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }

            // Mask invalid actions
            auto valid_moves = cur_state.get_possible_moves();
            std::vector<bool> valid_mask(GAME_COLS, false);
            for (int move = 0; move < valid_moves.count; move++) valid_mask[valid_moves.columns[move]] = true;

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

            // === VIRTUAL LOSS: Apply penalty BEFORE recursing ===
            if (virtual_loss_path != nullptr) {
                // Penalize the edge we're about to traverse
                // Note: node is non-const reference because we need to modify it
                auto& mutable_node = tree_[cur_key];  // Safe: node exists
                mutable_node.value_sum[best_action] -= VIRTUAL_LOSS;
                virtual_loss_path->emplace_back(cur_key, best_action);
            }

            bool won = cur_state.make_move(best_action);

            if (won) {
                value = -1.0f;  // Terminal win: -1 from opponent's perspective
                return { value, cur_state, static_cast<Player>(1 - static_cast<int>(cur_player)),
                         std::move(states), std::move(actions) };
            }

            cur_player = static_cast<Player>(1 - static_cast<int>(cur_player));

            if (cur_state.get_possible_moves().count == 0) {
                value = 0.0f;  // Draw
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }
        }

        uint64_t leaf_key = cur_state.hash_key;
        return { std::numeric_limits<float>::quiet_NaN(), cur_state, cur_player,
                 std::move(states), std::move(actions) };
    }

    bool MCTS::is_leaf(const GameState& state) const {
        uint64_t key = state.hash_key;
        return tree_.find(key) == tree_.end();
    }

    void MCTS::search_batch(int count, int batch_size, const GameState& state,
        Player player, Connect4Net& net, const torch::Device& device) {
        for (int i = 0; i < count; ++i) {
            search_minibatch(batch_size, state, player, net, device);
        }
    }

    void MCTS::search_minibatch(int count, const GameState& state, Player player,
        Connect4Net& net, const torch::Device& device) {

        std::unordered_map<uint64_t, std::vector<BackupEntry>> pending_backups;
        std::vector<GameState> expand_states;
        std::vector<Player> expand_players;
        std::unordered_set<uint64_t> planned;

        // === Phase 1: Find leaves with virtual loss ===
        for (int i = 0; i < count; ++i) {
            try {
                dirichlet_noise = generate_dirichlet_noise();

                // Track virtual loss path for this search
                std::vector<std::pair<uint64_t, int>> virtual_loss_path;

                auto [value, leaf_state, leaf_player, states, actions] =
                    find_leaf(state, player, &virtual_loss_path);

                if (!std::isnan(value)) {
                    // === Terminal backup: reverse virtual loss THEN apply real update ===
                    float terminal_value = value;
                    float cur_value = -terminal_value;

                    for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
                        if (j < static_cast<int>(actions.size()) &&
                            actions[j] >= 0 && actions[j] < GAME_COLS) {

                            uint64_t state_key = states[j].hash_key;
                            if (tree_.find(state_key) == tree_.end()) {
                                tree_[state_key] = MCTSNode();
                            }
                            auto& node = tree_[state_key];

                            // === Reverse virtual loss if this edge was penalized ===
                            if (!virtual_loss_path.empty() &&
                                virtual_loss_path.back().first == state_key &&
                                virtual_loss_path.back().second == actions[j]) {
                                node.value_sum[actions[j]] += VIRTUAL_LOSS;  // Undo penalty
                                virtual_loss_path.pop_back();
                            }

                            // Normal backup
                            node.visit_count[actions[j]]++;
                            node.value_sum[actions[j]] += cur_value;

                            cur_value = -cur_value;
                        }
                    }
                }
                else {
                    // === Non-terminal: queue for NN expansion ===
                    uint64_t leaf_key = leaf_state.hash_key;
                    pending_backups[leaf_key].emplace_back(
                        std::move(states), std::move(actions), std::move(virtual_loss_path));

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

        // === Phase 2: Expand nodes with batched NN query ===
        if (!expand_states.empty()) {
            try {
                std::vector<std::future<std::pair<std::array<float, GAME_COLS>, float>>> futures;
                futures.reserve(expand_states.size());

                for (size_t i = 0; i < expand_states.size(); ++i) {
                    futures.push_back(neural_worker_->submit_query(
                        expand_states[i], expand_players[i]));
                }

                // === Phase 3: Create nodes and backup ALL pending searches ===
                for (size_t i = 0; i < expand_states.size(); ++i) {
                    const auto& leaf_state = expand_states[i];
                    uint64_t leaf_key = leaf_state.hash_key;

                    auto [probs, value] = futures[i].get();

                    // Create node if needed
                    if (tree_.find(leaf_key) == tree_.end()) {
                        MCTSNode node;
                        auto valid_moves = leaf_state.get_possible_moves();
                        std::vector<bool> valid_mask(GAME_COLS, false);
                        for (int move = 0; move < valid_moves.count; move++) valid_mask[valid_moves.columns[move]] = true;

                        for (int j = 0; j < GAME_COLS; ++j) {
                            node.probs[j] = valid_mask[j] ? probs[j] : 0.0f;
                            node.visit_count[j] = 0;
                            node.value_sum[j] = 0.0f;
                        }

                        float sum = std::accumulate(node.probs.begin(), node.probs.end(), 0.0f);
                        if (sum > 1e-8f) {
                            for (float& p : node.probs) p /= sum;
                        }
                        tree_[leaf_key] = std::move(node);
                    }

                    // Backup ALL paths that reached this leaf
                    auto it = pending_backups.find(leaf_key);
                    if (it != pending_backups.end()) {
                        for (auto& [states, actions, vloss_path] : it->second) {
                            float cur_value = -value;  // Flip perspective

                            for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
                                if (j < static_cast<int>(actions.size()) &&
                                    actions[j] >= 0 && actions[j] < GAME_COLS) {

                                    uint64_t state_key = states[j].hash_key;
                                    if (tree_.find(state_key) == tree_.end()) {
                                        tree_[state_key] = MCTSNode();
                                    }
                                    auto& node = tree_[state_key];

                                    // === Reverse virtual loss before real update ===
                                    if (!vloss_path.empty() &&
                                        vloss_path.back().first == state_key &&
                                        vloss_path.back().second == actions[j]) {
                                        node.value_sum[actions[j]] += VIRTUAL_LOSS;
                                        vloss_path.pop_back();
                                    }

                                    // Normal backup
                                    node.visit_count[actions[j]]++;
                                    node.value_sum[actions[j]] += cur_value;

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
                // === CRITICAL: Clean up virtual loss on NN failure ===
                for (auto& [leaf_key, backups] : pending_backups) {
                    for (auto& [states, actions, vloss_path] : backups) {
                        for (auto& [key, action] : vloss_path) {
                            if (tree_.find(key) != tree_.end()) {
                                tree_[key].value_sum[action] += VIRTUAL_LOSS;
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
        uint64_t state_key = state.hash_key;
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

        for (int i = 0; i < GAME_COLS; ++i) values[i] = node.get_value_avg(i);
        return { probs, values };
    }

} // namespace Connect4