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

        GameState state = GameState();
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
                auto valid_moves = state.get_possible_moves();
                if (valid_moves.count != 0) {
                    action = valid_moves.columns[0];
                }
                else {
                    throw std::runtime_error("No valid moves available");
                }
            }

            bool won = state.make_move(action);

            if (won) {
                result = 1.0f;
                net1_result = (cur_player_int == 0) ? 1.0f : -1.0f;
                break;
            }
            cur_player_int = 1 - cur_player_int;
            cur_player = static_cast<Player>(cur_player_int);

            // Check for draw
            if (state.get_possible_moves().count == 0) {
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

                uint64_t new_black = 0, new_white = 0;

                for (int col = 0; col < 7; ++col) {
                    int flip_col = 6 - col;

                    // Extract the 7 bits of the current column (including the 0 sentinel bit)
                    uint64_t mask = 0x7FULL << (col * 7);

                    // Move the column to its mirrored position using fast bitwise shifts
                    new_black |= ((s.black_pieces & mask) >> (col * 7)) << (flip_col * 7);
                    new_white |= ((s.white_pieces & mask) >> (col * 7)) << (flip_col * 7);
                }

                GameState flipped = s;
                flipped.black_pieces = new_black;
                flipped.white_pieces = new_white;

                // 2. Recompute the Zobrist hash for the flipped state
                // (Required so MCTS and TT recognize this state correctly)
                flipped.hash_key = 0;
                for (int i = 0; i < 42; ++i) {
                    if (new_black & (1ULL << i)) flipped.hash_key ^= ZOBRIST.table[i][0];
                    if (new_white & (1ULL << i)) flipped.hash_key ^= ZOBRIST.table[i][1];
                }
                // Add the turn key if it's Black's turn (based on your make_move logic)
                if (flipped.current_player == Player::BLACK) {
                    flipped.hash_key ^= ZOBRIST.black_turn_key;
                }

                // 3. Flip the policy (Column 0 becomes Column 6, etc.)
                std::array<float, 7> flipped_probs;
                for (int i = 0; i < 7; ++i) {
                    flipped_probs[i] = pr[6 - i];
                }
                local_buffer->emplace_back(std::make_tuple(flipped, p, flipped_probs, current_result));
                current_result = -current_result;
            }
        }

        return { net1_result, step };
    }

} // namespace Connect4