#include "game_play.h"
#include "mcts.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <bit>       // Required for std::countr_zero
#include <tuple>

namespace Connect4 {

    // Helper function to horizontally flip a Pascal Pons bitboard state
    // Each column is exactly 7 bits (6 rows + 1 guard bit). 
    // We simply swap the 7-bit chunks: col 0 <-> 6, col 1 <-> 5, col 2 <-> 4.
    static GameState flip_state_horizontally(const GameState& s) {
        GameState flipped;
        flipped.moves = s.moves;

        uint64_t new_current = 0;
        uint64_t new_mask = 0;

        for (int col = 0; col < GAME_COLS; ++col) {
            int src_col = 6 - col;
            int shift = src_col * (GAME_ROWS + 1); // 7 bits per column

            // Extract exactly 7 bits (0x7F = 1111111 in binary)
            uint64_t col_bits_current = (s.current_position >> shift) & 0x7F;
            uint64_t col_bits_mask = (s.mask >> shift) & 0x7F;

            // Place them in the mirrored column position
            new_current |= (col_bits_current << (col * (GAME_ROWS + 1)));
            new_mask |= (col_bits_mask << (col * (GAME_ROWS + 1)));
        }

        flipped.current_position = new_current;
        flipped.mask = new_mask;
        return flipped;
    }

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

        // Default constructor creates a clean, empty initial state
        GameState state;
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
            mcts->search_batch(mcts_searches, mcts_batch_size, state, cur_player, nets[cur_player_int], device);

            auto [probs, _] = mcts->get_policy_value(state, tau);
            // History always records tau=1.0 policy for training targets
            auto [h_probs, __] = mcts->get_policy_value(state, 1.0f);
            game_history.emplace_back(state, cur_player, h_probs);

            // Sample action from policy
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float random_value = dist(gen);

            int action = 0;
            float cumulative_sum = 0.0f;

            for (int col = 0; col < GAME_COLS; ++col) {
                cumulative_sum += probs[col];
                if (random_value <= cumulative_sum) {
                    action = col;
                    break;
                }
                // Fallback for microscopic float inaccuracies (e.g., sum = 0.99999)
                if (col == GAME_COLS - 1) {
                    action = col;
                }
            }

            // Safety fallback: If the NN somehow assigned probability to a full column, 
            // redirect to the first genuinely available column.
            if (!state.canPlay(action)) {
                uint64_t possible = state.possible();
                if (possible != 0) {
                    action = std::countr_zero(possible) / (GAME_ROWS + 1);
                }
                else {
                    throw std::runtime_error("No valid moves available");
                }
            }

            // Check win BEFORE playing the move, using native bitboard logic
            bool won = state.isWinningMove(action);
            state.playCol(action);

            if (won) {
                result = 1.0f;
                net1_result = (cur_player_int == 0) ? 1.0f : -1.0f;
                break;
            }

            cur_player_int = 1 - cur_player_int;
            cur_player = static_cast<Player>(cur_player_int);

            // Check for draw (board is full)
            if (state.possible() == 0) {
                result = 0.0f;
                net1_result = 0.0f;
                break;
            }

            step++;
            if (step >= steps_before_tau_0) {
                tau = 0.0f;
            }
        }

        // Data Augmentation: Save original and horizontally flipped states
        if (local_buffer) {
            float current_result = result;
            for (auto it = game_history.rbegin(); it != game_history.rend(); ++it) {
                const auto& [s, p, pr] = *it;

                // 1. Save original state
                local_buffer->emplace_back(std::make_tuple(s, p, pr, current_result));

                // 2. Generate and save flipped state
                GameState flipped = flip_state_horizontally(s);

                std::array<float, GAME_COLS> flipped_probs;
                for (int i = 0; i < GAME_COLS; ++i) {
                    flipped_probs[i] = pr[6 - i];
                }

                // Note: The perspective (player 'p') remains the same, but the board is mirrored.
                // The result from the perspective of the flipped state's player is inverted.
                local_buffer->emplace_back(std::make_tuple(flipped, p, flipped_probs, current_result));

                current_result = -current_result;
            }
        }

        return { net1_result, step };
    }
}