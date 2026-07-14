#include "game_play.h"
#include "mcts.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

namespace Connect4 {

    std::pair<float, int> play_game(
        std::vector<MCTS>& mcts_stores,
        std::vector<ReplayBuffer::value_type>* local_buffer,
        Connect4Net& net1,
        Connect4Net& net2,
        int steps_before_tau_0,
        int mcts_searches,
        int mcts_batch_size,
        std::optional<bool> net1_plays_first,
        const torch::Device& device) {

        GameState state = GameLogic::INITIAL_STATE;
        std::vector<Connect4Net> nets = { net1, net2 };

        int cur_player_int;
        if (net1_plays_first.has_value()) {
            cur_player_int = net1_plays_first.value() ? 0 : 1;
        }
        else {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 1);
            cur_player_int = dis(gen);
        }

        Player cur_player = static_cast<Player>(cur_player_int);

        int step = 0;
        float tau = steps_before_tau_0 > 0 ? 1.0f : 0.0f;
        std::vector<std::tuple<GameState, Player, std::array<float, GAME_COLS>>> game_history;

        float result = std::numeric_limits<float>::quiet_NaN();
        float net1_result = std::numeric_limits<float>::quiet_NaN();
        std::random_device rd;
        std::mt19937 gen(rd());

        while (std::isnan(result)) {
            if (cur_player_int >= static_cast<int>(mcts_stores.size())) {
                throw std::runtime_error("Invalid player index for MCTS stores");
            }

            auto& mcts = mcts_stores[cur_player_int];
            mcts.search_batch(mcts_searches, mcts_batch_size, state,
                cur_player, nets[cur_player_int], device);

            auto [probs, _] = mcts.get_policy_value(state, tau);
            auto [h_probs, __] = mcts.get_policy_value(state, 1);
            game_history.emplace_back(state, cur_player, h_probs);

            // Select action based on probabilities
            std::vector<float> probs_vec(probs.begin(), probs.end());
            std::discrete_distribution<int> dist(probs_vec.begin(), probs_vec.end());
            int action = dist(gen);

            if (!state.is_valid_move(action)) {
                // Fallback to first valid move if selected action is invalid
                auto valid_moves = GameLogic::get_possible_moves(state);
                if (!valid_moves.empty()) {
                    action = valid_moves[0];
                }
                else {
                    throw std::runtime_error("No valid moves available");
                }
            }

            auto [new_state, won] = GameLogic::make_move(state, action);

            if (won) {
                result = 1.0f;
                net1_result = (cur_player_int == 0) ? 1.0f : -1.0f;
                state = new_state;
                break;
            }
            state = new_state;
            cur_player_int = 1 - cur_player_int;
            cur_player = static_cast<Player>(cur_player_int);

            // Check for draw
            if (GameLogic::get_possible_moves(state).empty()) {
                result = 0.0f;
                net1_result = 0.0f;
                break;
            }

            step++;
            if (step >= steps_before_tau_0) {
                tau = 0.0f;
            }
        }

        // Update replay buffer if provided
        if (local_buffer) {
            float current_result = result;
            for (auto it = game_history.rbegin(); it != game_history.rend(); ++it) {
                const auto& [s, p, pr] = *it;
                local_buffer->emplace_back(std::make_tuple(s, p, pr, current_result));

                GameState flipped = s;
                uint64_t new_black = 0, new_white = 0;

                // Simple bitboard flip (row by row)
                for (int row = 0; row < 6; ++row) {
                    for (int col = 0; col < 7; ++col) {
                        int orig_pos = row * 7 + col;
                        int flip_pos = row * 7 + (6 - col);

                        if (s.black_pieces & (1ULL << orig_pos))
                            new_black |= (1ULL << flip_pos);
                        if (s.white_pieces & (1ULL << orig_pos))
                            new_white |= (1ULL << flip_pos);
                    }

                }


                flipped.black_pieces = new_black;
                flipped.white_pieces = new_white;
                flipped.occupied = new_white | new_black;

                for (int col = 0; col < 7; ++col) {
                    int height = 0;
                    for (int row = 0; row < 6; ++row) {
                        int pos = row * 7 + col;
                        if (flipped.occupied & (1ULL << pos))
                            height = row + 1;
                    }
                    flipped.heights[col] = height;
                }
                // Flip policy
                std::array<float, 7> flipped_probs;
                for (int i = 0; i < 7; ++i)
                    flipped_probs[i] = pr[6 - i];

                local_buffer->emplace_back(std::make_tuple(flipped, p, flipped_probs, current_result));
                current_result = -current_result;
            }
        }

        return { net1_result, step };
    }
}