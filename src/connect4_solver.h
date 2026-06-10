#include <cstdint>
#include <algorithm>
#include <iostream>
#include "connect4_game.h"
#include "transportation_table.h"
#include <bit> // Required for std::popcount

namespace Connect4 {
    class Connect4Solver {
    private:
        TranspositionTable tt;
        uint64_t nodes_visited = 0;

        // Changed 'ply' to 'depth_remaining' to correctly limit search from current position
        int8_t alpha_beta(GameState& state, int8_t alpha, int8_t beta, uint8_t depth_remaining) {
            nodes_visited++;

            // 1. Depth limit reached: Return 0 (Unknown/Draw) 
            if (depth_remaining == 0) return 0;

            // 2. Board is full
            if (std::popcount(state.get_occupied()) >= 42) return 0;

            TTEntry cached_entry;
            int tt_move = -1;

            if (tt.lookup(state.hash_key, cached_entry)) {
                // Only use TT if it was searched to at least the current remaining depth
                if (cached_entry.depth >= depth_remaining) {
                    if (cached_entry.type == EntryType::EXACT) return cached_entry.score;
                    if (cached_entry.type == EntryType::LOWER_BOUND) alpha = std::max(alpha, cached_entry.score);
                    else if (cached_entry.type == EntryType::UPPER_BOUND) beta = std::min(beta, cached_entry.score);
                    if (alpha >= beta) return cached_entry.score;
                }
                tt_move = cached_entry.best_move;
            }

            auto moves = state.get_possible_moves();
            int8_t best_score = -44;
            EntryType entry_type = EntryType::UPPER_BOUND;
            int best_move_col = -1;

            auto process_move = [&](int col) -> bool {
                bool is_win = state.make_move(col);
                int8_t score;
                if (is_win) {
                    // Use absolute ply to correctly prefer faster wins
                    uint8_t absolute_ply = std::popcount(state.get_occupied());
                    score = 43 - absolute_ply;
                }
                else {
                    score = -alpha_beta(state, -beta, -alpha, depth_remaining - 1);
                }
                state.unmake_move(col);

                if (score > best_score) {
                    best_score = score;
                    best_move_col = col;
                }
                if (score > alpha) {
                    alpha = score;
                    entry_type = EntryType::EXACT;
                }
                if (alpha >= beta) {
                    entry_type = EntryType::LOWER_BOUND;
                    return true; // Beta cutoff
                }
                return false;
                };

            // Validate TT move to ensure it's actually legal in the current state
            bool is_valid_tt_move = false;
            if (tt_move != -1) {
                for (int i = 0; i < moves.count; ++i) {
                    if (moves.columns[i] == tt_move) {
                        is_valid_tt_move = true;
                        break;
                    }
                }
            }

            // Try TT move first for massive speedup
            if (is_valid_tt_move) {
                if (process_move(tt_move)) {
                    tt.store(state.hash_key, best_score, depth_remaining, entry_type, best_move_col);
                    return best_score;
                }
            }

            // Try remaining moves
            for (int i = 0; i < moves.count; ++i) {
                int col = moves.columns[i];
                if (col == tt_move) continue;
                if (process_move(col)) {
                    break;
                }
            }

            tt.store(state.hash_key, best_score, depth_remaining, entry_type, best_move_col);
            return best_score;
        }

    public:
        // Increased default TT size to 1<<25 (32M entries, ~512MB)
        explicit Connect4Solver(size_t tt_size = 1 << 25) : tt(tt_size) {}

        std::pair<int, int8_t> solve(GameState& state, uint8_t max_depth = 42) {
            nodes_visited = 0;

            // Safeguard for 0 depth
            if (max_depth == 0) return { -1, 0 };

            auto moves = state.get_possible_moves();
            if (moves.count == 0) return { -1, 0 }; // Board full

            int best_move = -1;
            int8_t best_score = -44;
            int8_t alpha = -44;
            int8_t beta = 44;

            for (int i = 0; i < moves.count; ++i) {
                int col = moves.columns[i];
                bool is_win = state.make_move(col);
                int8_t score;

                if (is_win) {
                    uint8_t absolute_ply = std::popcount(state.get_occupied());
                    score = 43 - absolute_ply;
                }
                else {
                    // Pass max_depth - 1 because we just made 1 ply in this loop
                    score = -alpha_beta(state, -beta, -alpha, max_depth - 1);
                }

                state.unmake_move(col);

                if (score > best_score) {
                    best_score = score;
                    best_move = col;
                }
                alpha = std::max(alpha, score);
            }
            return { best_move, best_score };
        }

        void clear_cache() {
            tt.clear();
        }
    };
}