#pragma once
#include "connect4_game.h"
#include <array>
#include <vector>
#include <cstdint>
#include <chrono>
#include <random>
#include <intrin.h>

#pragma warning(disable: 4146) // Negation of unsigned warning

namespace Connect4 {

    // MSVC-compatible count trailing zeros for uint64_t
    inline int ctzll(uint64_t x) {
        unsigned long index;
        return _BitScanForward64(&index, x) ? static_cast<int>(index) : 64;
    }

    class MinimaxPlayer {
    public:
        explicit MinimaxPlayer(size_t tt_size_mb = 128);
        ~MinimaxPlayer() = default;

        // Find best move with optional depth/time limits (time_limit_ms=0 for unlimited)
        int find_best_move(const GameState& state, int max_depth = 42, int time_limit_ms = 0);

        // Clear all internal state (TT, killers, history)
        void clear();

        // Statistics getters
        uint64_t get_nodes_searched() const { return nodes_searched_; }
        uint64_t get_tt_hits() const { return tt_hits_; }
        double get_tt_hit_rate() const {
            return nodes_searched_ ? (tt_hits_ * 100.0 / nodes_searched_) : 0.0;
        }

    private:
        // Constants
        static constexpr int MAX_PLY = 42;
        static constexpr int WIN_SCORE = 1000000;
        static constexpr int DRAW_SCORE = 0;
        static constexpr int LOSS_SCORE = -1000000;
        static constexpr std::array<int8_t, 7> COLUMN_ORDER = { 3, 2, 4, 1, 5, 0, 6 };

        // Zobrist hashing keys
        struct ZobristKeys {
            std::array<std::array<uint64_t, 2>, 42> piece; // [pos][player]
            uint64_t player_to_move;
        };

        // Compact 16-byte transposition table entry
        struct alignas(16) TTEntry {
            uint64_t hash;
            int16_t score;
            int8_t depth;
            int8_t best_move;   // -1 = invalid
            uint8_t flag;       // 0=exact, 1=lower bound, 2=upper bound
            uint8_t age;

            TTEntry() : hash(0), score(0), depth(-1), best_move(-1), flag(0), age(0) {}
        };

        // Private methods
        void init_zobrist();
        uint64_t compute_hash(const GameState& state) const;
        int detect_immediate_win(const GameState& state, const std::vector<int>& moves);
        int detect_forced_block(const GameState& state, const std::vector<int>& moves);
        void order_moves(const GameState& state, std::vector<int>& moves, int ply, int tt_move);
        int evaluate(const GameState& state);
        int negamax(GameState state, int depth, int alpha, int beta, int ply);

        // Member variables
        ZobristKeys zobrist_;
        std::vector<TTEntry> tt_;
        size_t tt_mask_;
        uint8_t tt_age_;
        std::array<std::array<int8_t, 2>, MAX_PLY> killers_;
        std::array<int16_t, 7> history_;

        // Search state
        uint64_t nodes_searched_;
        uint64_t tt_hits_;
        bool timeout_;
        std::chrono::steady_clock::time_point start_time_;
        int64_t time_limit_ns_;
    };

} // namespace Connect4