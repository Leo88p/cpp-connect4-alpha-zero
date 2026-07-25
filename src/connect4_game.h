#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <numeric>
#include <random>
#include <bit>
#include <functional>

constexpr static uint64_t bottom(int width, int height) {
    return width == 0 ? 0 : bottom(width - 1, height) | 1LL << (width - 1) * (height + 1);
}
namespace Connect4 {

    constexpr int GAME_ROWS = 6;
    constexpr int GAME_COLS = 7;
    constexpr int MIN_SCORE = -(GAME_COLS * GAME_ROWS) / 2 + 3;
    constexpr int MAX_SCORE = (GAME_COLS * GAME_ROWS + 1) / 2 - 3;

    enum class Player : uint8_t {
        WHITE = 0,
        BLACK = 1
    };

    class GameState {
    public:
        uint64_t current_position;
        uint64_t mask;
        unsigned int moves;
        GameState(): current_position{ 0 }, mask{ 0 }, moves{ 0 } {}
        uint64_t key() const {
            return current_position + mask;
        }
        void playCol(int col) {
            play((mask + bottom_mask_col(col)) & column_mask(col));
        }
        void play(uint64_t move) {
            current_position ^= mask;
            mask |= move;
            moves++;
        }
        bool canPlay(int col) const {
            return (mask & top_mask_col(col)) == 0;
        }
        uint64_t possible() const {
            return (mask + bottom_mask) & board_mask;
        }
        bool isWinningMove(int col) const {
            return winning_position() & possible() & column_mask(col);
        }
        int nbMoves() const {
            return moves;
        }
        uint64_t key3() const {
            uint64_t key_forward = 0;
            for (int i = 0; i < GAME_COLS; i++) partialKey3(key_forward, i);  // compute key in increasing order of columns

            uint64_t key_reverse = 0;
            for (int i = GAME_COLS; i--;) partialKey3(key_reverse, i);  // compute key in decreasing order of columns

            return key_forward < key_reverse ? key_forward / 3 : key_reverse / 3; // take the smallest key and divide per 3 as the last base3 digit is always 0
        }
        void partialKey3(uint64_t& key, int col) const {
            for (uint64_t pos = UINT64_C(1) << (col * (GAME_ROWS + 1)); pos & mask; pos <<= 1) {
                key *= 3;
                if (pos & current_position) key += 1;
                else key += 2;
            }
            key *= 3;
        }
        bool canWinNext() const {
            return winning_position() & possible();
        }
        uint64_t possibleNonLosingMoves() const {
            assert(!canWinNext());
            uint64_t possible_mask = possible();
            uint64_t opponent_win = opponent_winning_position();
            uint64_t forced_moves = possible_mask & opponent_win;
            if (forced_moves) {
                if (forced_moves & (forced_moves - 1)) // check if there is more than one forced move
                    return 0;                           // the opponnent has two winning moves and you cannot stop him
                else possible_mask = forced_moves;    // enforce to play the single forced move
            }
            return possible_mask & ~(opponent_win >> 1);  // avoid to play below an opponent winning spot
        }
        uint64_t opponent_winning_position() const {
            return compute_winning_position(current_position ^ mask, mask);
        }
        static constexpr uint64_t bottom_mask_col(int col) {
            return UINT64_C(1) << col * (GAME_ROWS + 1);
        }
        static constexpr uint64_t column_mask(int col) {
            return ((UINT64_C(1) << GAME_ROWS) - 1) << col * (GAME_ROWS + 1);
        }
        static constexpr uint64_t top_mask_col(int col) {
            return UINT64_C(1) << ((GAME_ROWS - 1) + col * (GAME_ROWS + 1));
        }
        const static uint64_t bottom_mask = bottom(GAME_COLS, GAME_ROWS);
        const static uint64_t board_mask = bottom_mask * ((1LL << GAME_ROWS) - 1);
        int moveScore(uint64_t move) const {
            return popcount(compute_winning_position(current_position | move, mask));
        }

    private:
        uint64_t winning_position() const {
            return compute_winning_position(current_position, mask);
        }
        static uint64_t compute_winning_position(uint64_t position, uint64_t mask) {
            // vertical;
            uint64_t r = (position << 1) & (position << 2) & (position << 3);

            //horizontal
            uint64_t p = (position << (GAME_ROWS + 1)) & (position << 2 * (GAME_ROWS + 1));
            r |= p & (position << 3 * (GAME_ROWS + 1));
            r |= p & (position >> (GAME_ROWS + 1));
            p = (position >> (GAME_ROWS + 1)) & (position >> 2 * (GAME_ROWS + 1));
            r |= p & (position << (GAME_ROWS + 1));
            r |= p & (position >> 3 * (GAME_ROWS + 1));

            //diagonal 1
            p = (position << GAME_ROWS) & (position << 2 * GAME_ROWS);
            r |= p & (position << 3 * GAME_ROWS);
            r |= p & (position >> GAME_ROWS);
            p = (position >> GAME_ROWS) & (position >> 2 * GAME_ROWS);
            r |= p & (position << GAME_ROWS);
            r |= p & (position >> 3 * GAME_ROWS);

            //diagonal 2
            p = (position << (GAME_ROWS + 2)) & (position << 2 * (GAME_ROWS + 2));
            r |= p & (position << 3 * (GAME_ROWS + 2));
            r |= p & (position >> (GAME_ROWS + 2));
            p = (position >> (GAME_ROWS + 2)) & (position >> 2 * (GAME_ROWS + 2));
            r |= p & (position << (GAME_ROWS + 2));
            r |= p & (position >> 3 * (GAME_ROWS + 2));

            return r & (board_mask ^ mask);
        }
        static unsigned int popcount(uint64_t m) {
            unsigned int c = 0;
            for (c = 0; m; c++) m &= m - 1;
            return c;
        }
    };

} // namespace Connect4