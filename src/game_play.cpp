#include "game_play.h"
#include "mcts.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory_resource> // Required for std::pmr

namespace Connect4 {

    std::pair<float, int> play_game(
        const std::unique_ptr<MCTS>* mcts_stores,
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

        // PMR OPTIMIZATION: Stack-allocated monotonic buffer for game history.
        // Connect 4 max moves is 42. 4096 bytes is more than enough to avoid ANY heap allocation.
        alignas(64) std::byte game_history_buffer[4096];
        std::pmr::monotonic_buffer_resource gh_mbr{ game_history_buffer, sizeof(game_history_buffer), std::pmr::new_delete_resource() };
        std::pmr::polymorphic_allocator<void> gh_alloc(&gh_mbr);
        std::pmr::vector<std::tuple<GameState, Player, std::array<float, GAME_COLS>>> game_history{ gh_alloc };

        float result = std::numeric_limits<float>::quiet_NaN();
        float net1_result = std::numeric_limits<float>::quiet_NaN();
        thread_local std::mt19937 gen(std::random_device{}());

        while (std::isnan(result)) {
            if (cur_player_int < 0 || cur_player_int > 1) {
                throw std::runtime_error("Invalid player index for MCTS stores");
            }

            auto& mcts = mcts_stores[cur_player_int];
            mcts->search_batch(mcts_searches, mcts_batch_size, state,
                cur_player, nets[cur_player_int], device);

            auto [probs, _] = mcts->get_policy_value(state, tau);
            auto [h_probs, __] = mcts->get_policy_value(state, 1);
            game_history.emplace_back(state, cur_player, h_probs);

            // BONUS OPTIMIZATION: discrete_distribution accepts iterators directly. 
            // This completely eliminates the need for the intermediate std::vector<float> allocation.
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float random_value = dist(gen);

            int action = 0;
            float cumulative_sum = 0.0f;

            // Линейное сканирование по 7 колонкам Connect4
            for (int col = 0; col < GAME_COLS; ++col) {
                cumulative_sum += probs[col];
                if (random_value <= cumulative_sum) {
                    action = col;
                    break;
                }
                // На случай микроскопических погрешностей float (например, сумма 0.99999)
                if (col == GAME_COLS - 1) {
                    action = col;
                }
            }

            if (!state.is_valid_move(action)) {
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

        if (local_buffer) {
            float current_result = result;
            for (auto it = game_history.rbegin(); it != game_history.rend(); ++it) {
                const auto& [s, p, pr] = *it;
                local_buffer->emplace_back(std::make_tuple(s, p, pr, current_result));

                GameState flipped = s;
                uint64_t new_black = 0, new_white = 0;

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