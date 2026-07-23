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

    private:
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
    };

} // namespace Connect4