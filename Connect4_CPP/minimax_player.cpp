#include "minimax_player.h"
#include <algorithm>
#include <cmath>

namespace Connect4 {

    MinimaxPlayer::MinimaxPlayer(size_t tt_size_mb)
        : tt_age_(0)
        , nodes_searched_(0)
        , tt_hits_(0)
        , timeout_(false)
        , time_limit_ns_(0)
    {
        init_zobrist();

        // Power-of-two TT sizing for efficient masking
        size_t tt_entries = (tt_size_mb * 1024 * 1024) / sizeof(TTEntry);
        size_t tt_size = 1;
        while (tt_size < tt_entries) tt_size <<= 1;
        tt_size >>= 1;  // Use half to reduce collisions

        tt_.resize(tt_size);
        tt_mask_ = tt_size - 1;

        // Initialize killer moves and history
        for (auto& k : killers_) {
            k[0] = k[1] = -1;
        }
        history_.fill(0);
    }

    void MinimaxPlayer::init_zobrist() {
        std::mt19937_64 rng(12345);  // Fixed seed for reproducibility
        for (int pos = 0; pos < 42; ++pos) {
            for (int p = 0; p < 2; ++p) {
                zobrist_.piece[pos][p] = rng();
            }
        }
        zobrist_.player_to_move = rng();
    }

    uint64_t MinimaxPlayer::compute_hash(const GameState& state) const {
        uint64_t hash = 0;

        // XOR black pieces
        uint64_t bits = state.black_pieces;
        while (bits) {
            int pos = ctzll(bits);
            hash ^= zobrist_.piece[pos][0];
            bits &= bits - 1;  // Clear lowest set bit
        }

        // XOR white pieces
        bits = state.white_pieces;
        while (bits) {
            int pos = ctzll(bits);
            hash ^= zobrist_.piece[pos][1];
            bits &= bits - 1;
        }

        // XOR side to move
        if (state.current_player == Player::WHITE) {
            hash ^= zobrist_.player_to_move;
        }

        return hash;
    }

    int MinimaxPlayer::detect_immediate_win(const GameState& state, const std::vector<int>& moves) {
        for (int col : moves) {
            auto [_, won] = GameLogic::make_move(state, col);
            if (won) return col;
        }
        return -1;
    }

    int MinimaxPlayer::detect_forced_block(const GameState& state, const std::vector<int>& moves) {
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
                    return col;  // Must block this column
                }
            }
        }
        return -1;
    }

    void MinimaxPlayer::order_moves(const GameState& state, std::vector<int>& moves, int ply, int tt_move) {
        // TT move first (most important)
        if (tt_move != -1) {
            auto it = std::find(moves.begin(), moves.end(), tt_move);
            if (it != moves.end() && it != moves.begin()) {
                std::rotate(moves.begin(), it, it + 1);
            }
        }

        // Killer moves (second priority)
        if (ply < MAX_PLY) {
            for (int k = 0; k < 2; ++k) {
                int killer = killers_[ply][k];
                if (killer != -1 && killer != tt_move) {
                    auto it = std::find(moves.begin(), moves.end(), killer);
                    if (it != moves.end()) {
                        int insert_pos = (tt_move != -1) ? 1 : 0;
                        if (static_cast<int>(it - moves.begin()) > insert_pos) {
                            std::rotate(moves.begin() + insert_pos, it, it + 1);
                        }
                    }
                }
            }
        }

        // Column preference + history heuristic (tertiary)
        std::stable_sort(moves.begin(), moves.end(), [this](int a, int b) {
            int idx_a = static_cast<int>(std::find(COLUMN_ORDER.begin(), COLUMN_ORDER.end(), a) - COLUMN_ORDER.begin());
            int idx_b = static_cast<int>(std::find(COLUMN_ORDER.begin(), COLUMN_ORDER.end(), b) - COLUMN_ORDER.begin());
            // Higher history score + better column position = higher priority
            return (history_[a] - idx_a * 100) > (history_[b] - idx_b * 100);
            });
    }

    int MinimaxPlayer::evaluate(const GameState& state) {
        int score = 0;
        Player player = state.current_player;
        uint64_t us = (player == Player::BLACK) ? state.black_pieces : state.white_pieces;
        uint64_t them = (player == Player::BLACK) ? state.white_pieces : state.black_pieces;
        uint64_t empty = ~state.occupied;

        // Evaluate all win patterns for threats and opportunities
        for (uint64_t pattern : GameLogic::WIN_PATTERNS) {
            uint64_t our_pieces = us & pattern;
            uint64_t their_pieces = them & pattern;
            uint64_t empty_sq = empty & pattern;
            int our_count = __popcnt64(our_pieces);
            int their_count = __popcnt64(their_pieces);
            int empty_count = __popcnt64(empty_sq);

            // Our winning threats (3 in a row + 1 playable empty)
            if (our_count == 3 && empty_count == 1) {
                uint64_t empty_bit = empty_sq & -empty_sq;  // Isolate lowest bit
                int pos = ctzll(empty_bit);
                int col = pos % GAME_COLS;
                int row = pos / GAME_COLS;
                // Check if empty square is playable (has support below or is bottom row)
                if (row == 0 || (state.occupied & (1ULL << (pos - GAME_COLS)))) {
                    score += 50000;  // Strong winning threat
                }
            }
            else if (our_count == 2 && empty_count == 2) {
                score += 1000;  // Potential threat
            }

            // Opponent threats (more urgent to block)
            if (their_count == 3 && empty_count == 1) {
                uint64_t empty_bit = empty_sq & -empty_sq;
                int pos = ctzll(empty_bit);
                int col = pos % GAME_COLS;
                int row = pos / GAME_COLS;
                if (row == 0 || (state.occupied & (1ULL << (pos - GAME_COLS)))) {
                    score -= 100000;  // Critical: must block immediately
                }
            }
            else if (their_count == 2 && empty_count == 2) {
                score -= 2000;  // Opponent potential
            }
        }

        // Center control bonus (column 3 is most valuable)
        for (int row = 0; row < GAME_ROWS; ++row) {
            int pos = row * GAME_COLS + 3;
            uint64_t bit = 1ULL << pos;
            if (us & bit) score += 50;
            else if (them & bit) score -= 50;
        }

        return score;
    }

    int MinimaxPlayer::negamax(GameState state, int depth, int alpha, int beta, int ply) {
        // Periodic timeout check (every 1024 nodes to minimize overhead)
        if ((nodes_searched_ & 0x3FF) == 0) {
            if (timeout_ || (time_limit_ns_ > 0 &&
                (std::chrono::steady_clock::now() - start_time_).count() >= time_limit_ns_)) {
                timeout_ = true;
                return 0;  // Return neutral score on timeout
            }
        }

        ++nodes_searched_;

        // Check for draw (no legal moves)
        auto moves = GameLogic::get_possible_moves(state);
        if (moves.empty()) {
            return DRAW_SCORE;
        }

        // Transposition table probe
        uint64_t hash = compute_hash(state);
        size_t idx = hash & tt_mask_;
        TTEntry& entry = tt_[idx];

        if (entry.hash == hash && entry.depth >= depth) {
            ++tt_hits_;
            switch (entry.flag) {
            case 0: return entry.score;  // Exact value
            case 1: alpha = std::max(alpha, static_cast<int>(entry.score)); break;  // Lower bound
            case 2: beta = std::min(beta, static_cast<int>(entry.score)); break;   // Upper bound
            }
            if (alpha >= beta) return entry.score;
        }

        // Check for immediate winning move (before deeper search)
        int win_move = detect_immediate_win(state, moves);
        if (win_move != -1) {
            entry.hash = hash;
            entry.depth = static_cast<int8_t>(depth);
            entry.score = static_cast<int16_t>(WIN_SCORE - ply);  // Prefer faster wins
            entry.best_move = static_cast<int8_t>(win_move);
            entry.flag = 0;  // Exact
            entry.age = tt_age_;
            return WIN_SCORE - ply;
        }

        // Depth limit: return heuristic evaluation
        if (depth == 0) {
            return evaluate(state);
        }

        // Forced block detection (critical tactical pattern in Connect4)
        int block_move = detect_forced_block(state, moves);
        if (block_move != -1) {
            auto [new_state, _] = GameLogic::make_move(state, block_move);
            int score = -negamax(new_state, depth - 1, -beta, -alpha, ply + 1);
            entry.hash = hash;
            entry.depth = static_cast<int8_t>(depth);
            entry.score = static_cast<int16_t>(score);
            entry.best_move = static_cast<int8_t>(block_move);
            entry.flag = 0;
            entry.age = tt_age_;
            return score;
        }

        // Move ordering using TT, killers, and history
        int tt_move = (entry.hash == hash && entry.best_move != -1) ? entry.best_move : -1;
        order_moves(state, moves, ply, tt_move);

        int best_score = LOSS_SCORE;
        int best_move = moves[0];
        bool full_window = true;

        for (size_t i = 0; i < moves.size(); ++i) {
            int col = moves[i];
            auto [new_state, won] = GameLogic::make_move(state, col);

            // Note: 'won' here means current player won with this move (already handled above)
            // So we proceed with recursive search

            int score;
            if (full_window || i == 0) {
                // Full window search for first move or when needed
                score = -negamax(new_state, depth - 1, -beta, -alpha, ply + 1);
            }
            else {
                // Principal Variation Search (null window)
                score = -negamax(new_state, depth - 1, -alpha - 1, -alpha, ply + 1);
                // Re-search if null window failed high
                if (score > alpha && score < beta) {
                    score = -negamax(new_state, depth - 1, -beta, -alpha, ply + 1);
                }
            }

            if (timeout_) return 0;

            // Update best move and alpha
            if (score > best_score) {
                best_score = score;
                best_move = col;
                if (score > alpha) {
                    alpha = score;
                    // Update history heuristic (boost successful moves)
                    history_[col] = std::min(32767, history_[col] + depth * depth);
                    // Update killer moves (if not from TT)
                    if (ply < MAX_PLY && col != tt_move) {
                        if (killers_[ply][0] != col) {
                            killers_[ply][1] = killers_[ply][0];
                            killers_[ply][0] = static_cast<int8_t>(col);
                        }
                    }
                }
            }

            // Beta cutoff: prune remaining moves
            if (alpha >= beta) {
                entry.hash = hash;
                entry.depth = static_cast<int8_t>(depth);
                entry.score = static_cast<int16_t>(beta);
                entry.best_move = static_cast<int8_t>(best_move);
                entry.flag = 2;  // Upper bound
                entry.age = tt_age_;
                return beta;
            }
            full_window = false;
        }

        // Store exact result in TT
        entry.hash = hash;
        entry.depth = static_cast<int8_t>(depth);
        entry.score = static_cast<int16_t>(best_score);
        entry.best_move = static_cast<int8_t>(best_move);
        entry.flag = 0;  // Exact
        entry.age = tt_age_;

        return best_score;
    }

    int MinimaxPlayer::find_best_move(const GameState& state, int max_depth, int time_limit_ms) {
        // Reset search state
        timeout_ = false;
        nodes_searched_ = 0;
        tt_hits_ = 0;
        start_time_ = std::chrono::steady_clock::now();
        time_limit_ns_ = (time_limit_ms <= 0) ? 0 : static_cast<int64_t>(time_limit_ms) * 1'000'000LL;
        tt_age_ = (tt_age_ + 1) & 0x3F;  // Increment age with wraparound

        auto moves = GameLogic::get_possible_moves(state);
        if (moves.empty()) return -1;

        // Quick tactical checks before search
        int win_move = detect_immediate_win(state, moves);
        if (win_move != -1) return win_move;

        int block_move = detect_forced_block(state, moves);
        if (block_move != -1) return block_move;

        // Iterative deepening with early termination on forced wins
        int best_move = moves[0];
        int best_score = LOSS_SCORE;

        for (int depth = 1; depth <= max_depth; ++depth) {
            if (timeout_) break;

            int score = negamax(state, depth, LOSS_SCORE + 1, WIN_SCORE - 1, 0);
            if (timeout_) break;

            // Extract best move from transposition table
            uint64_t hash = compute_hash(state);
            size_t idx = hash & tt_mask_;
            if (tt_[idx].hash == hash && tt_[idx].best_move != -1) {
                best_move = tt_[idx].best_move;
                best_score = score;
            }

            // Early exit on forced win (score close to WIN_SCORE)
            if (best_score >= WIN_SCORE - MAX_PLY) break;
        }

        // Safety fallback: return first legal move in preferred order
        if (best_move < 0 || best_move >= 7 || state.heights[best_move] >= GAME_ROWS) {
            for (int col : COLUMN_ORDER) {
                if (state.heights[col] < GAME_ROWS) return col;
            }
            return moves[0];  // Ultimate fallback
        }

        return best_move;
    }

    void MinimaxPlayer::clear() {
        std::fill(tt_.begin(), tt_.end(), TTEntry());
        for (auto& k : killers_) {
            k[0] = k[1] = -1;
        }
        history_.fill(0);
        tt_age_ = 0;
        nodes_searched_ = 0;
        tt_hits_ = 0;
    }

} // namespace Connect4