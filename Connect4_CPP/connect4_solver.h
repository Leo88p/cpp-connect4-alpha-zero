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

        int8_t alpha_beta(GameState& state, int8_t alpha, int8_t beta, uint8_t ply, uint8_t max_depth) {
            nodes_visited++;

            // 1. Depth limit reached: Return 0 (Unknown/Draw) 
            // Note: You could replace this with a simple heuristic evaluation 
            // (e.g., counting open 3-in-a-rows) to make the imperfect minimax smarter.
            if (ply >= max_depth) return 0;

            // 2. Board is full
            if (ply >= 42) return 0;

            TTEntry cached_entry;
            int tt_move = -1;

            if (tt.lookup(state.hash_key, cached_entry)) {
                // Only use TT if it was searched to at least the current ply
                // AND the TT depth is >= the remaining depth we care about
                if (cached_entry.depth >= ply) {
                    if (cached_entry.type == EntryType::EXACT) return cached_entry.score;
                    if (cached_entry.type == EntryType::LOWER_BOUND) alpha = std::max(alpha, cached_entry.score);
                    else if (cached_entry.type == EntryType::UPPER_BOUND) beta = std::min(beta, cached_entry.score);
                    if (alpha >= beta) return cached_entry.score;
                }
                tt_move = cached_entry.best_move;
            }

            auto moves = state.get_possible_moves();
            int8_t best_score = -42;
            EntryType entry_type = EntryType::UPPER_BOUND;
            int best_move_col = -1;

            auto process_move = [&](int col) -> bool {
                bool is_win = state.make_move(col);
                int8_t score;
                if (is_win) {
                    score = 42 - ply; // Prefer faster wins
                }
                else {
                    score = -alpha_beta(state, -beta, -alpha, ply + 1, max_depth);
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

            // Try TT move first for massive speedup
            if (tt_move != -1) {
                if (process_move(tt_move)) {
                    tt.store(state.hash_key, best_score, ply, entry_type, best_move_col);
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

            tt.store(state.hash_key, best_score, ply, entry_type, best_move_col);
            return best_score;
        }

    public:
        // Increased default TT size to 1<<25 (32M entries, ~512MB)
        explicit Connect4Solver(size_t tt_size = 1 << 25) : tt(tt_size) {}

        std::pair<int, int8_t> solve(GameState& state, uint8_t max_depth = 42) {
            nodes_visited = 0;
            auto moves = state.get_possible_moves();

            int best_move = -1;
            int8_t best_score = -42;
            int8_t alpha = -42;
            int8_t beta = 42;

            uint8_t current_ply = std::popcount(state.get_occupied());

            for (int i = 0; i < moves.count; ++i) {
                int col = moves.columns[i];
                bool is_win = state.make_move(col);
                int8_t score;

                if (is_win) {
                    score = 42 - current_ply;
                }
                else {
                    score = -alpha_beta(state, -beta, -alpha, current_ply + 1, max_depth);
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