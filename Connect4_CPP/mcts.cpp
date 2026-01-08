#include "mcts.h"
#include <numeric>
#include <execution>
#include <thread>

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
        dirichlet_dist_ = std::gamma_distribution<float>(0.03f, 1.0f);
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
            if (cur_state.to_key() == root_state.to_key()) {
                auto noises = generate_dirichlet_noise();
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = values_avg[i] + c_puct_ * (0.75f * probs[i] + 0.25f * noises[i]) *
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
        std::vector<std::tuple<float, std::vector<GameState>, std::vector<int>>> backup_queue;
        std::vector<GameState> expand_states;
        std::vector<Player> expand_players;
        std::vector<std::tuple<GameState, std::vector<GameState>, std::vector<int>>> expand_queue;
        std::unordered_set<uint64_t> planned;

        // Phase 1: Find leaf nodes for all searches
        backup_queue.reserve(count);
        expand_queue.reserve(count);

        for (int i = 0; i < count; ++i) {
            auto [value, leaf_state, leaf_player, states, actions] =
                find_leaf(state, player);

            if (!std::isnan(value)) {
                backup_queue.emplace_back(value, std::move(states), std::move(actions));
            }
            else {
                uint64_t leaf_key = leaf_state.to_key();
                if (planned.find(leaf_key) == planned.end()) {
                    planned.insert(leaf_key);
                    expand_states.push_back(leaf_state);
                    expand_players.push_back(leaf_player);
                    expand_queue.emplace_back(leaf_state, std::move(states), std::move(actions));
                }
            }
        }

        // Phase 2: Expand nodes
        if (!expand_queue.empty()) {
            // Convert states to batch tensor
            auto batch_v = state_lists_to_batch(expand_states, expand_players, device);

            // Forward pass through network
            torch::NoGradGuard no_grad;
            auto output = net->forward(batch_v);
            auto logits_v = std::get<0>(output);
            auto values_v = std::get<1>(output);

            auto probs_v = torch::softmax(logits_v, 1);
            auto values = values_v.squeeze(1).cpu();
            auto probs = probs_v.cpu();

            // Create nodes and add to backup queue
            for (size_t i = 0; i < expand_queue.size(); ++i) {
                auto& [leaf_state, states, actions] = expand_queue[i];
                uint64_t leaf_key = leaf_state.to_key();

                if (tree_.find(leaf_key) == tree_.end()) {

                    MCTSNode node;
                    auto prob_row = probs[i];
                    auto prob_accessor = prob_row.accessor<float, 1>();
                    for (int j = 0; j < GAME_COLS; ++j) {
                        node.probs[j] = prob_accessor[j];
                        // Initialize with small values to avoid division by zero
                        node.visit_count[j] = 1; // Start with 1 visit to avoid NaNs
                        node.value_avg[j] = 0.0f;
                    }
                    tree_[leaf_key] = std::move(node);
                }

                float value = values[i].item<float>();
                backup_queue.emplace_back(value, std::move(states), std::move(actions));
            }
        }

        // Phase 3: Backup values
        for (auto& [value, states, actions] : backup_queue) {
            float cur_value = -value;

            for (int i = static_cast<int>(states.size()) - 1; i >= 0; --i) {
                const auto& state = states[i];
                int action = actions[i];
                uint64_t state_key = state.to_key();

                auto& node = tree_[state_key];
                node.visit_count[action]++;
                node.value[action] += cur_value;
                node.value_avg[action] = node.value[action] / static_cast<float>(node.visit_count[action]);

                cur_value = -cur_value;
            }
        }
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

} // namespace Connect4