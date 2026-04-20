#include "mcts.h"
#include <numeric>
#include <execution>
#include <thread>a

namespace Connect4 {

    void MCTSNode::reset() {
        std::fill(visit_count.begin(), visit_count.end(), 0);
        std::fill(value.begin(), value.end(), 0.0f);
        std::fill(value_avg.begin(), value_avg.end(), 0.0f);
        std::fill(probs.begin(), probs.end(), 0.0f);
    }

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
        // Reset the random number generator state to avoid identical sequences
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

        // Generate gamma distributed samples
        for (int i = 0; i < GAME_COLS; ++i) {
            noise[i] = dirichlet_dist_(rng_);
            sum += noise[i];
        }

        // Normalize to get Dirichlet distribution
        if (sum > 0.0f) {
            for (float& n : noise) {
                n /= sum;
            }
        }

        return noise;
    }

    std::tuple<float, GameState, Player, std::vector<GameState>, std::vector<int>>
        MCTS::find_leaf(const GameState& root_state, Player player) {
        std::vector<GameState> states;
        std::vector<int> actions;
        GameState cur_state = root_state;
        Player cur_player = player;
        float value = std::numeric_limits<float>::quiet_NaN();

        while (!is_leaf(cur_state)) {
            states.push_back(cur_state);

            uint64_t cur_key = cur_state.to_key();
            const auto& node = tree_.at(cur_key);

            float total_sqrt = std::sqrt(static_cast<float>(std::accumulate(
                node.visit_count.begin(), node.visit_count.end(), 0)));

            std::array<float, GAME_COLS> score;
            const auto& probs = node.probs;
            const auto& values_avg = node.value_avg;
            const auto& counts = node.visit_count;

            // Choose action to take
            if (cur_state.to_key() == root_state.to_key() && use_noise) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = values_avg[i] + c_puct_ * (0.75f * probs[i] + 0.25f * dirichlet_noise[i]) *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }
            else {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = values_avg[i] + c_puct_ * probs[i] *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }

            // Mask invalid actions
            auto valid_moves = GameLogic::get_possible_moves(cur_state);
            std::vector<bool> valid_mask(GAME_COLS, false);
            for (int move : valid_moves) {
                valid_mask[move] = true;
            }

            int best_action = -1;
            float best_score = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < GAME_COLS; ++i) {
                if (!valid_mask[i]) {
                    score[i] = -std::numeric_limits<float>::infinity();
                }
                if (score[i] > best_score) {
                    best_score = score[i];
                    best_action = i;
                }
            }

            if (best_action == -1) {
                throw std::runtime_error("No valid moves found in MCTS");
            }

            actions.push_back(best_action);

            auto [new_state, won] = GameLogic::make_move(cur_state, best_action);

            if (won) {
                value = -1.0f; // Game won, value is -1 for opponent's turn
                return { value, new_state, static_cast<Player>(1 - static_cast<int>(cur_player)),
                       std::move(states), std::move(actions) };
            }

            cur_state = new_state;
            cur_player = static_cast<Player>(1 - static_cast<int>(cur_player));

            // Check for draw
            if (GameLogic::get_possible_moves(cur_state).empty()) {
                value = 0.0f;
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }
        }

        uint64_t leaf_key = cur_state.to_key();

        return { std::numeric_limits<float>::quiet_NaN(), cur_state, cur_player, std::move(states), std::move(actions) };
    }

    bool MCTS::is_leaf(const GameState& state) const {
        uint64_t key = state.to_key();
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

        // Track pending backups: leaf_key -> list of (states, actions) paths to backup
        // This ensures EVERY search contributes to learning, not just unique leaves
        std::unordered_map<uint64_t, std::vector<std::pair<std::vector<GameState>, std::vector<int>>>> pending_backups;

        std::vector<GameState> expand_states;
        std::vector<Player> expand_players;
        std::unordered_set<uint64_t> planned;

        // === Phase 1: Find leaves and queue for expansion ===
        for (int i = 0; i < count; ++i) {
            try {
                dirichlet_noise = generate_dirichlet_noise();
                auto [value, leaf_state, leaf_player, states, actions] = find_leaf(state, player);

                if (!std::isnan(value)) {
                    // Terminal state: backup immediately with terminal value
                    float terminal_value = value;  // From perspective of player who just moved (lost)
                    float cur_value = -terminal_value;  // Flip: from perspective of parent (who won)

                    for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
                        if (j < static_cast<int>(actions.size()) &&
                            actions[j] >= 0 && actions[j] < GAME_COLS) {

                            uint64_t state_key = states[j].to_key();
                            if (tree_.find(state_key) == tree_.end()) {
                                tree_[state_key] = MCTSNode();
                            }
                            auto& node = tree_[state_key];

                            node.visit_count[actions[j]]++;
                            node.value[actions[j]] += cur_value;
                            node.value_avg[actions[j]] = node.value[actions[j]] /
                                static_cast<float>(node.visit_count[actions[j]]);

                            cur_value = -cur_value;  // Flip perspective for next parent up
                        }
                    }
                }
                else {
                    // Non-terminal leaf: queue for backup after NN expansion
                    uint64_t leaf_key = leaf_state.to_key();

                    // ALWAYS store this search path for later backup (critical fix)
                    pending_backups[leaf_key].emplace_back(std::move(states), std::move(actions));

                    // Plan NN expansion only once per unique leaf (for efficiency)
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

        // === Phase 2: Expand nodes with single batched forward pass ===
        if (!expand_states.empty()) {
            try {
                // Convert states to batch tensor
                std::vector<std::future<std::pair<std::array<float, GAME_COLS>, float>>> futures;
                futures.reserve(expand_states.size());

                for (size_t i = 0; i < expand_states.size(); ++i) {
                    futures.push_back(neural_worker_->submit_query(
                        expand_states[i], expand_players[i]));
                }

                // === Phase 3: Create nodes and backup ALL pending searches ===
                for (size_t i = 0; i < expand_states.size(); ++i) {
                    const auto& leaf_state = expand_states[i];
                    uint64_t leaf_key = leaf_state.to_key();

                    // Get result from worker
                    auto [probs, value] = futures[i].get();

                    // Create node if it doesn't exist yet
                    if (tree_.find(leaf_key) == tree_.end()) {
                        MCTSNode node;

                        auto valid_moves = GameLogic::get_possible_moves(leaf_state);
                        std::vector<bool> valid_mask(GAME_COLS, false);
                        for (int move : valid_moves) {
                            valid_mask[move] = true;
                        }

                        for (int j = 0; j < GAME_COLS; ++j) {
                            if (valid_mask[j]) {
                                node.probs[j] = probs[j];
                            }
                            else {
                                node.probs[j] = 0.0f;  // Mask invalid moves
                            }
                            node.visit_count[j] = 0;
                            node.value[j] = 0.0f;
                            node.value_avg[j] = 0.0f;
                        }

                        float sum = std::accumulate(node.probs.begin(), node.probs.end(), 0.0f);
                        if (sum > 1e-8f) {
                            for (float& p : node.probs) {
                                p /= sum;
                            }
                        }
                        tree_[leaf_key] = std::move(node);
                    }

                    // Backup ALL search paths that reached this leaf with the correct value
                    auto it = pending_backups.find(leaf_key);
                    if (it != pending_backups.end()) {
                        for (auto& [states, actions] : it->second) {
                            // Value from network is from leaf player's perspective
                            // We need value from parent's perspective, so flip sign
                            float cur_value = -value;

                            for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
                                if (j < static_cast<int>(actions.size()) &&
                                    actions[j] >= 0 && actions[j] < GAME_COLS) {

                                    uint64_t state_key = states[j].to_key();

                                    // Ensure node exists (should always exist after expansion, but be safe)
                                    if (tree_.find(state_key) == tree_.end()) {
                                        tree_[state_key] = MCTSNode();
                                    }

                                    auto& node = tree_[state_key];

                                    // Update statistics
                                    node.visit_count[actions[j]]++;
                                    node.value[actions[j]] += cur_value;
                                    node.value_avg[actions[j]] = node.value[actions[j]] /
                                        static_cast<float>(node.visit_count[actions[j]]);

                                    // Flip perspective for next parent up the tree
                                    cur_value = -cur_value;
                                }
                            }
                        }
                        // Clean up to save memory
                        pending_backups.erase(it);
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Neural network expansion error: " << e.what() << std::endl;
                return;
            }
        }

        // Note: Any remaining entries in pending_backups are for leaves that weren't expanded
        // (shouldn't happen in normal operation, but if it does, they're silently dropped)
        // This is acceptable because expand_states should contain all non-terminal unique leaves
    }

    std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
        MCTS::get_policy_value(const GameState& state, float tau) const {
        uint64_t state_key = state.to_key();
        const auto& node = tree_.at(state_key);

        std::array<float, GAME_COLS> probs;
        std::array<float, GAME_COLS> values;

        if (tau == 0.0f) {
            // Deterministic policy
            std::fill(probs.begin(), probs.end(), 0.0f);
            int best_action = std::distance(node.visit_count.begin(),
                std::max_element(node.visit_count.begin(), node.visit_count.end()));
            probs[best_action] = 1.0f;
        }
        else {
            // Stochastic policy with temperature
            std::array<float, GAME_COLS> counts_pow;
            float sum = 0.0f;

            for (int i = 0; i < GAME_COLS; ++i) {
                counts_pow[i] = std::pow(static_cast<float>(node.visit_count[i]), 1.0f / tau);
                sum += counts_pow[i];
            }

            if (sum > 0.0f) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    probs[i] = counts_pow[i] / sum;
                }
            }
            else {
                // Fallback to uniform distribution
                float uniform_prob = 1.0f / GAME_COLS;
                std::fill(probs.begin(), probs.end(), uniform_prob);
            }
        }

        // Copy values
        for (int i = 0; i < GAME_COLS; ++i) {
            values[i] = node.value_avg[i];
        }

        return { probs, values };
    }
}