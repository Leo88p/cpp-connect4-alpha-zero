#pragma once
#include "mcts.h"
#include <numeric>
#include <execution>
#include <thread>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <iostream>

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

            int total_visits = std::accumulate(node.visit_count.begin(), node.visit_count.end(), 0);
            float total_sqrt = std::sqrt(static_cast<float>(total_visits));

            std::array<float, GAME_COLS> score;
            const auto& probs = node.probs;
            const auto& values_avg = node.value_avg;
            const auto& counts = node.visit_count;

            // Choose action to take - check if this is the root node
            bool is_root = (cur_state.to_key() == root_state.to_key());

            if (is_root) {
                auto noises = generate_dirichlet_noise();
                const float epsilon = 0.25f; // Standard AlphaZero epsilon

                for (int i = 0; i < GAME_COLS; ++i) {
                    float mixed_prob = (1.0f - epsilon) * probs[i] + epsilon * noises[i];
                    score[i] = values_avg[i] + c_puct_ * mixed_prob *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }
            else {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = values_avg[i] + c_puct_ * probs[i] *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }

            // Get valid moves
            auto valid_moves = GameLogic::get_possible_moves(cur_state);
            std::vector<bool> valid_mask(GAME_COLS, false);

            for (int move : valid_moves) {
                if (move >= 0 && move < GAME_COLS) {
                    valid_mask[move] = true;
                }
            }

            if (valid_moves.empty()) {
                // Draw position
                value = 0.0f;
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }

            int best_action = -1;
            float best_score = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < GAME_COLS; ++i) {
                if (!valid_mask[i]) continue;
                if (score[i] > best_score) {
                    best_score = score[i];
                    best_action = i;
                }
            }

            if (best_action < 0 || best_action >= GAME_COLS) {
                throw std::runtime_error("MCTS: No valid action selected. Best action: " + std::to_string(best_action));
            }

            actions.push_back(best_action);

            auto [new_state, won] = GameLogic::make_move(cur_state, best_action);

            // Check win FIRST - critical fix
            if (won) {
                value = -1.0f; // From perspective of next player (who lost)
                return { value, new_state, static_cast<Player>(1 - static_cast<int>(cur_player)),
                       std::move(states), std::move(actions) };
            }

            cur_state = new_state;
            cur_player = static_cast<Player>(1 - static_cast<int>(cur_player));
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
        std::unordered_set<uint64_t> planned_expansions;

        // Phase 1: Find leaf nodes for all searches
        for (int i = 0; i < count; ++i) {
            try {
                auto [value, leaf_state, leaf_player, states, actions] = find_leaf(state, player);

                if (!std::isnan(value)) {
                    // Terminal state - backup immediately
                    backup_queue.emplace_back(value, std::move(states), std::move(actions));
                }
                else {
                    uint64_t leaf_key = leaf_state.to_key();

                    if (planned_expansions.find(leaf_key) == planned_expansions.end()) {
                        // First time seeing this leaf - plan for expansion
                        planned_expansions.insert(leaf_key);
                        expand_states.push_back(leaf_state);
                        expand_players.push_back(leaf_player);
                        expand_queue.emplace_back(leaf_state, std::move(states), std::move(actions));
                    }
                    else {
                        // This leaf will be expanded - add to backup queue later
                        backup_queue.emplace_back(0.0f, std::move(states), std::move(actions)); // Placeholder, will be replaced
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "MCTS search error: " << e.what() << std::endl;
                continue;
            }
        }

        // Phase 2: Expand nodes
        if (!expand_queue.empty()) {
            try {
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

                // Create nodes and fill backup queue
                for (size_t i = 0; i < expand_queue.size(); ++i) {
                    auto& [leaf_state, states, actions] = expand_queue[i];
                    uint64_t leaf_key = leaf_state.to_key();
                    float value = values[i].item<float>();

                    // Create node if it doesn't exist
                    if (tree_.find(leaf_key) == tree_.end()) {
                        MCTSNode node;
                        auto prob_row = probs[i];
                        auto prob_accessor = prob_row.accessor<float, 1>();
                        for (int j = 0; j < GAME_COLS; ++j) {
                            node.probs[j] = (j < prob_accessor.size(0)) ? prob_accessor[j] : 0.0f;
                            node.visit_count[j] = 0;
                            node.value[j] = 0.0f;
                            node.value_avg[j] = 0.0f;
                        }
                        tree_[leaf_key] = std::move(node);
                    }

                    // Backup with neural network value
                    backup_queue.emplace_back(value, std::move(states), std::move(actions));
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Neural network expansion error: " << e.what() << std::endl;
                return;
            }
        }

        // Phase 3: Backup values - with strict bounds checking
        for (auto& [value, states, actions] : backup_queue) {
            if (std::isnan(value)) continue;

            float cur_value = -value; // Start from leaf perspective
            for (int i = static_cast<int>(states.size()) - 1; i >= 0; --i) {
                if (i >= static_cast<int>(actions.size()) || i < 0) {
                    continue; // Skip if action index is invalid
                }

                int action = actions[i];
                if (action < 0 || action >= GAME_COLS) {
                    continue; // Skip invalid actions
                }

                const auto& state = states[i];
                uint64_t state_key = state.to_key();

                // Ensure node exists
                if (tree_.find(state_key) == tree_.end()) {
                    tree_[state_key] = MCTSNode();
                }

                auto& node = tree_[state_key];

                // SAFETY CHECK: Validate action bounds before accessing
                if (action < 0 || action >= static_cast<int>(node.visit_count.size())) {
                    continue;
                }

                node.visit_count[action]++;
                node.value[action] += cur_value;
                node.value_avg[action] = node.value[action] / static_cast<float>(node.visit_count[action]);

                cur_value = -cur_value; // Flip perspective for parent
            }
        }
    }

    std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
        MCTS::get_policy_value(const GameState& state, float tau) const {
        uint64_t state_key = state.to_key();

        // Handle case where state hasn't been searched yet
        auto it = tree_.find(state_key);
        if (it == tree_.end()) {
            std::array<float, GAME_COLS> uniform_probs;
            std::array<float, GAME_COLS> zero_values;
            float uniform_prob = 1.0f / GAME_COLS;
            std::fill(uniform_probs.begin(), uniform_probs.end(), uniform_prob);
            std::fill(zero_values.begin(), zero_values.end(), 0.0f);
            return { uniform_probs, zero_values };
        }

        const auto& node = it->second;

        std::array<float, GAME_COLS> probs;
        std::array<float, GAME_COLS> values;

        if (tau == 0.0f || tau < 1e-6f) {
            // Deterministic policy - most visited action
            std::fill(probs.begin(), probs.end(), 0.0f);
            int best_action = 0;
            int max_visits = -1;

            for (int i = 0; i < GAME_COLS; ++i) {
                if (node.visit_count[i] > max_visits) {
                    max_visits = node.visit_count[i];
                    best_action = i;
                }
            }

            if (max_visits > 0 && best_action >= 0 && best_action < GAME_COLS) {
                probs[best_action] = 1.0f;
            }
            else {
                // Fallback to uniform if no visits
                float uniform_prob = 1.0f / GAME_COLS;
                std::fill(probs.begin(), probs.end(), uniform_prob);
            }
        }
        else {
            // Stochastic policy with temperature
            std::array<float, GAME_COLS> counts_pow;
            float sum = 0.0f;

            for (int i = 0; i < GAME_COLS; ++i) {
                if (node.visit_count[i] > 0) {
                    counts_pow[i] = std::pow(static_cast<float>(node.visit_count[i]), 1.0f / tau);
                    sum += counts_pow[i];
                }
                else {
                    counts_pow[i] = 0.0f;
                }
            }

            if (sum > 1e-8f) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    probs[i] = counts_pow[i] / sum;
                }
            }
            else {
                // Fallback to uniform distribution among valid moves
                auto valid_moves = GameLogic::get_possible_moves(state);
                if (!valid_moves.empty()) {
                    float valid_prob = 1.0f / static_cast<float>(valid_moves.size());
                    std::fill(probs.begin(), probs.end(), 0.0f);
                    for (int move : valid_moves) {
                        if (move >= 0 && move < GAME_COLS) {
                            probs[move] = valid_prob;
                        }
                    }
                }
                else {
                    float uniform_prob = 1.0f / GAME_COLS;
                    std::fill(probs.begin(), probs.end(), uniform_prob);
                }
            }
        }

        // Copy values - handle unvisited actions safely
        for (int i = 0; i < GAME_COLS; ++i) {
            if (node.visit_count[i] > 0) {
                values[i] = node.value_avg[i];
            }
            else {
                // For unvisited actions, use the prior probability as a heuristic
                values[i] = node.probs[i] * 2.0f - 1.0f; // Scale to [-1, 1] range
            }
        }

        return { probs, values };
    }

} // namespace Connect4