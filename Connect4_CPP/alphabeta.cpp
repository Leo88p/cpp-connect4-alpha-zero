#pragma once
#include "connect4_game.h"
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <random>
#include <intrin.h>
#include <limits>

#pragma warning(disable: 4146) // Negation of unsigned warning

namespace Connect4 {

    // MSVC-compatible ctzll
    inline int ctzll(uint64_t x) {
        unsigned long index;
        return _BitScanForward64(&index, x) ? static_cast<int>(index) : 64;
    }

    class MinimaxPlayer {
    private:
        // Zobrist hashing
        struct ZobristKeys {
            std::array<std::array<uint64_t, 2>, 42> piece; // [pos][player]
            uint64_t player_to_move;
        } zobrist_;

        // Compact 16-byte TT entry
        struct alignas(16) TTEntry {
            uint64_t hash;
            int16_t score;
            int8_t depth;
            int8_t best_move; // -1 = invalid
            uint8_t flag;     // 0=exact, 1=lower, 2=upper
            uint8_t age;

            TTEntry() : hash(0), score(0), depth(-1), best_move(-1), flag(0), age(0) {}
        };

        std::vector<TTEntry> tt_;
        size_t tt_mask_;
        uint8_t tt_age_ = 0;

        // Killer moves (2 per ply)
        static constexpr int MAX_PLY = 42;
        std::array<std::array<int8_t, 2>, MAX_PLY> killers_;

        // History heuristic (column-based)
        std::array<int16_t, 7> history_;

        // Center-first ordering
        static constexpr std::array<int8_t, 7> COLUMN_ORDER = { 3, 2, 4, 1, 5, 0, 6 };

        // Search state
        uint64_t nodes_searched_ = 0;
        uint64_t tt_hits_ = 0;
        bool timeout_ = false;
        std::chrono::steady_clock::time_point start_time_;
        int64_t time_limit_ns_ = 0; // Use nanoseconds for precision

        // Constants for scoring
        static constexpr int WIN_SCORE = 1000000;
        static constexpr int DRAW_SCORE = 0;
        static constexpr int LOSS_SCORE = -1000000;

        void init_zobrist() {
            std::mt19937_64 rng(12345);
            for (int pos = 0; pos < 42; ++pos) {
                for (int p = 0; p < 2; ++p) {
                    zobrist_.piece[pos][p] = rng();
                }
            }
            zobrist_.player_to_move = rng();
        }

        uint64_t compute_hash(const GameState& state) const {
            uint64_t hash = 0;
            uint64_t bits = state.black_pieces;
            while (bits) {
                int pos = ctzll(bits);
                hash ^= zobrist_.piece[pos][0];
                bits &= bits - 1;
            }
            bits = state.white_pieces;
            while (bits) {
                int pos = ctzll(bits);
                hash ^= zobrist_.piece[pos][1];
                bits &= bits - 1;
            }
            if (state.current_player == Player::WHITE) {
                hash ^= zobrist_.player_to_move;
            }
            return hash;
        }

        // Detect if current player can win immediately
        int detect_immediate_win(const GameState& state, std::vector<int>& moves) {
            for (int col : moves) {
                auto [_, won] = GameLogic::make_move(state, col);
                if (won) return col;
            }
            return -1;
        }

        // Detect forced block (opponent wins next move if we don't block)
        int detect_forced_block(const GameState& state, const std::vector<int>& moves) {
            Player opponent = (state.current_player == Player::BLACK) ? Player::WHITE : Player::BLACK;
            uint64_t opp_pieces = (opponent == Player::BLACK) ? state.black_pieces : state.white_pieces;

            for (int col : moves) {
                if (state.heights[col] >= GAME_ROWS) continue;
                int row = state.heights[col];
                int pos = row * GAME_COLS + col;
                uint64_t bit = 1ULL << pos;
                uint64_t new_pieces = opp_pieces | bit;

                for (uint64_t pattern : GameLogic::WIN_PATTERNS) {
                    if ((pattern & bit) && ((new_pieces & pattern) == pattern)) {
                        return col; // Must block here
                    }
                }
            }
            return -1;
        }

        void order_moves(const GameState& state, std::vector<int>& moves, int ply, int tt_move) {
            // TT move first
            if (tt_move != -1) {
                auto it = std::find(moves.begin(), moves.end(), tt_move);
                if (it != moves.end() && it != moves.begin()) {
                    std::rotate(moves.begin(), it, it + 1);
                }
            }

            // Killer moves
            if (ply < MAX_PLY) {
                for (int k = 0; k < 2; ++k) {
                    int killer = killers_[ply][k];
                    if (killer != -1 && killer != tt_move) {
                        auto it = std::find(moves.begin(), moves.end(), killer);
                        if (it != moves.end()) {
                            int insert_pos = (tt_move != -1) ? 1 : 0;
                            if (it - moves.begin() > insert_pos) {
                                std::rotate(moves.begin() + insert_pos, it, it + 1);
                            }
                        }
                    }
                }
            }

            // Column preference + history
            std::stable_sort(moves.begin(), moves.end(), [this](int a, int b) {
                int idx_a = static_cast<int>(std::find(COLUMN_ORDER.begin(), COLUMN_ORDER.end(), a) - COLUMN_ORDER.begin());
                int idx_b = static_cast<int>(std::find(COLUMN_ORDER.begin(), COLUMN_ORDER.end(), b) - COLUMN_ORDER.begin());
                return (history_[a] - idx_a * 100) > (history_[b] - idx_b * 100);
                });
        }

        // Advanced evaluation with threat detection
        int evaluate(const GameState& state) {
            int score = 0;
            Player player = state.current_player;
            uint64_t us = (player == Player::BLACK) ? state.black_pieces : state.white_pieces;
            uint64_t them = (player == Player::BLACK) ? state.white_pieces : state.black_pieces;
            uint64_t empty = ~state.occupied;

            // Check all win patterns for threats
            for (uint64_t pattern : GameLogic::WIN_PATTERNS) {
                uint64_t our_pieces = us & pattern;
                uint64_t their_pieces = them & pattern;
                uint64_t empty_sq = empty & pattern;
                int our_count = __popcnt64(our_pieces);
                int their_count = __popcnt64(their_pieces);
                int empty_count = __popcnt64(empty_sq);

                // Our threats
                if (our_count == 3 && empty_count == 1) {
                    // Check if the empty square is playable (has support below)
                    uint64_t empty_bit = empty_sq & -empty_sq; // Isolate lowest bit
                    int pos = ctzll(empty_bit);
                    int col = pos % GAME_COLS;
                    int row = pos / GAME_COLS;
                    if (row == 0 || (state.occupied & (1ULL << (pos - GAME_COLS)))) {
                        score += 50000; // Winning threat
                    }
                }
                else if (our_count == 2 && empty_count == 2) {
                    score += 1000;
                }

                // Opponent threats (more urgent)
                if (their_count == 3 && empty_count == 1) {
                    uint64_t empty_bit = empty_sq & -empty_sq;
                    int pos = ctzll(empty_bit);
                    int col = pos % GAME_COLS;
                    int row = pos / GAME_COLS;
                    if (row == 0 || (state.occupied & (1ULL << (pos - GAME_COLS)))) {
                        score -= 100000; // Must block immediately
                    }
                }
                else if (their_count == 2 && empty_count == 2) {
                    score -= 2000;
                }
            }

            // Center control bonus
            for (int row = 0; row < GAME_ROWS; ++row) {
                int pos = row * GAME_COLS + 3; // Center column
                uint64_t bit = 1ULL << pos;
                if (us & bit) score += 50;
                else if (them & bit) score -= 50;
            }

            return score;
        }

        // TRUE negamax with proper terminal state handling
        int negamax(GameState state, int depth, int alpha, int beta, int ply) {
            // Timeout check (every 1024 nodes)
            if ((nodes_searched_ & 0x3FF) == 0) {
                if (timeout_ || (time_limit_ns_ > 0 &&
                    (std::chrono::steady_clock::now() - start_time_).count() >= time_limit_ns_)) {
                    timeout_ = true;
                    return 0;
                }
            }

            nodes_searched_++;

            // Check for draw (board full)
            auto moves = GameLogic::get_possible_moves(state);
            if (moves.empty()) {
                return DRAW_SCORE;
            }

            // Transposition table lookup
            uint64_t hash = compute_hash(state);
            size_t idx = hash & tt_mask_;
            TTEntry& entry = tt_[idx];

            if (entry.hash == hash && entry.depth >= depth) {
                tt_hits_++;
                switch (entry.flag) {
                case 0: return entry.score;                          // Exact
                case 1: alpha = std::max(alpha, (int)entry.score); break; // Lower bound
                case 2: beta = std::min(beta, (int)entry.score);   break; // Upper bound
                }
                if (alpha >= beta) return entry.score;
            }

            // Check for immediate wins BEFORE search
            int win_move = detect_immediate_win(state, moves);
            if (win_move != -1) {
                entry.hash = hash;
                entry.depth = (int8_t)depth;
                entry.score = (int16_t)(WIN_SCORE - ply);
                entry.best_move = (int8_t)win_move;
                entry.flag = 0;
                entry.age = tt_age_;
                return WIN_SCORE - ply;
            }

            // Depth limit reached - evaluate
            if (depth == 0) {
                return evaluate(state);
            }

            // Forced block detection (critical for Connect4)
            int block_move = detect_forced_block(state, moves);
            if (block_move != -1) {
                auto [new_state, _] = GameLogic::make_move(state, block_move);
                int score = -negamax(new_state, depth - 1, -beta, -alpha, ply + 1);
                entry.hash = hash;
                entry.depth = (int8_t)depth;
                entry.score = (int16_t)score;
                entry.best_move = (int8_t)block_move;
                entry.flag = 0;
                entry.age = tt_age_;
                return score;
            }

            // Move ordering
            int tt_move = (entry.hash == hash && entry.best_move != -1) ? entry.best_move : -1;
            order_moves(state, moves, ply, tt_move);

            int best_score = LOSS_SCORE;
            int best_move = moves[0];
            bool full_window = true;

            for (size_t i = 0; i < moves.size(); ++i) {
                int col = moves[i];
                auto [new_state, won] = GameLogic::make_move(state, col);

                // If this move loses immediately (opponent wins next ply), skip or penalize
                // But we already checked for our wins above, so "won" here would be opponent win
                // Actually: won==true means CURRENT player (us) won with this move - already handled

                int score;
                if (full_window || i == 0) {
                    score = -negamax(new_state, depth - 1, -beta, -alpha, ply + 1);
                }
                else {
                    // Null window search (PVS)
                    score = -negamax(new_state, depth - 1, -alpha - 1, -alpha, ply + 1);
                    if (score > alpha && score < beta) {
                        score = -negamax(new_state, depth - 1, -beta, -alpha, ply + 1);
                    }
                }

                if (timeout_) return 0;

                if (score > best_score) {
                    best_score = score;
                    best_move = col;
                    if (score > alpha) {
                        alpha = score;
                        history_[col] = std::min(32767, history_[col] + depth * depth);
                        if (ply < MAX_PLY && col != tt_move) {
                            if (killers_[ply][0] != col) {
                                killers_[ply][1] = killers_[ply][0];
                                killers_[ply][0] = (int8_t)col;
                            }
                        }
                    }
                }

                if (alpha >= beta) {
                    entry.hash = hash;
                    entry.depth = (int8_t)depth;
                    entry.score = (int16_t)beta;
                    entry.best_move = (int8_t)best_move;
                    entry.flag = 2; // Upper bound
                    entry.age = tt_age_;
                    return beta;
                }
                full_window = false;
            }

            // Store exact result
            entry.hash = hash;
            entry.depth = (int8_t)depth;
            entry.score = (int16_t)best_score;
            entry.best_move = (int8_t)best_move;
            entry.flag = 0; // Exact
            entry.age = tt_age_;

            return best_score;
        }

    public:
        explicit MinimaxPlayer(size_t tt_size_mb = 128) {
            init_zobrist();

            // Power-of-two TT sizing
            size_t tt_entries = (tt_size_mb * 1024 * 1024) / sizeof(TTEntry);
            size_t tt_size = 1;
            while (tt_size < tt_entries) tt_size <<= 1;
            tt_size >>= 1;

            tt_.resize(tt_size);
            tt_mask_ = tt_size - 1;

            for (auto& k : killers_) {
                k[0] = k[1] = -1;
            }
            history_.fill(0);
        }

        // time_limit_ms = 0 means UNLIMITED search
        int find_best_move(const GameState& state, int max_depth = 42, int time_limit_ms = 0) {
            timeout_ = false;
            nodes_searched_ = 0;
            tt_hits_ = 0;
            start_time_ = std::chrono::steady_clock::now();
            time_limit_ns_ = (time_limit_ms <= 0) ? 0 : time_limit_ms * 1'000'000LL;
            tt_age_ = (tt_age_ + 1) & 0x3F;

            auto moves = GameLogic::get_possible_moves(state);
            if (moves.empty()) return -1;

            // Immediate win check
            int win_move = detect_immediate_win(state, moves);
            if (win_move != -1) return win_move;

            // Forced block check
            int block_move = detect_forced_block(state, moves);
            if (block_move != -1) return block_move;

            // Iterative deepening
            int best_move = moves[0];
            int best_score = LOSS_SCORE;

            for (int depth = 1; depth <= max_depth; ++depth) {
                if (timeout_) break;

                int score = negamax(state, depth, LOSS_SCORE + 1, WIN_SCORE - 1, 0);
                if (timeout_) break;

                // Update best move from TT
                uint64_t hash = compute_hash(state);
                size_t idx = hash & tt_mask_;
                if (tt_[idx].hash == hash && tt_[idx].best_move != -1) {
                    best_move = tt_[idx].best_move;
                    best_score = score;
                }

                // Stop if we found a forced win
                if (best_score >= WIN_SCORE - 42) break;
            }

            // Safety fallback
            if (best_move < 0 || best_move >= 7 || state.heights[best_move] >= GAME_ROWS) {
                for (int col : COLUMN_ORDER) {
                    if (state.heights[col] < GAME_ROWS) return col;
                }
                return moves[0];
            }

            return best_move;
        }

        void clear() {
            std::fill(tt_.begin(), tt_.end(), TTEntry());
            for (auto& k : killers_) {
                k[0] = k[1] = -1;
            }
            history_.fill(0);
            tt_age_ = 0;
        }

        uint64_t get_nodes_searched() const { return nodes_searched_; }
        uint64_t get_tt_hits() const { return tt_hits_; }
        double get_tt_hit_rate() const {
            return nodes_searched_ ? (tt_hits_ * 100.0 / nodes_searched_) : 0.0;
        }
    };

} // namespace Connect4