#pragma once
#include <array>
#include "zorbist_table.h"

namespace Connect4 {

    constexpr int GAME_ROWS = 6;
    constexpr int GAME_COLS = 7;
    constexpr int COUNT_TO_WIN = 4;
    constexpr int BOARD_SIZE = GAME_ROWS * GAME_COLS;

    enum class Player : uint8_t {
        WHITE = 0,
        BLACK = 1
    };

    struct GameState {
        static constexpr int GAME_COLS = 7;
        static constexpr int GAME_ROWS = 6;

        uint64_t black_pieces = 0;
        uint64_t white_pieces = 0;
        Player current_player = Player::BLACK;
        uint64_t hash_key = ZOBRIST.black_turn_key;

        bool operator==(const GameState& other) const {
            return black_pieces == other.black_pieces &&
                white_pieces == other.white_pieces &&
                current_player == other.current_player;
        }

        uint64_t get_occupied() const { return black_pieces | white_pieces; }

        bool is_valid_move(int col) const {
            uint64_t top_row_mask = 1ULL << (col * 7 + 5);
            return (get_occupied() & top_row_mask) == 0;
        }

        bool make_move(int col) {
            uint64_t column_mask = 1ULL << (col * 7);
            uint64_t move_bit = (get_occupied() + column_mask) & (column_mask * 0x7F);

            unsigned long bit_index;
            _BitScanForward64(&bit_index, move_bit);

            Player moving_player = current_player;
            if (moving_player == Player::BLACK) {
                black_pieces |= move_bit;
                hash_key ^= ZOBRIST.table[bit_index][0];
                current_player = Player::WHITE;
            }
            else {
                white_pieces |= move_bit;
                hash_key ^= ZOBRIST.table[bit_index][1];
                current_player = Player::BLACK;
            }
            hash_key ^= ZOBRIST.black_turn_key;
            return check_win(moving_player);
        }

        void unmake_move(int col) {
            uint64_t column_mask = 0x7FULL << (col * 7);
            uint64_t occupied_in_col = get_occupied() & column_mask;
            if (occupied_in_col == 0) return; // Safety check

            unsigned long bit_index;
            _BitScanReverse64(&bit_index, occupied_in_col); // Find the TOP-MOST piece
            uint64_t move_bit = 1ULL << bit_index;

            // Flip player back to the one who made the move
            current_player = (current_player == Player::BLACK) ? Player::WHITE : Player::BLACK;

            if (current_player == Player::BLACK) {
                black_pieces &= ~move_bit;
                hash_key ^= ZOBRIST.table[bit_index][0];
            }
            else {
                white_pieces &= ~move_bit;
                hash_key ^= ZOBRIST.table[bit_index][1];
            }
            hash_key ^= ZOBRIST.black_turn_key;
        }

        bool check_win(Player player) const {
            uint64_t board = (player == Player::BLACK) ? black_pieces : white_pieces;
            uint64_t temp;
            temp = board & (board >> 7); if (temp & (temp >> 14)) return true;
            temp = board & (board >> 6); if (temp & (temp >> 12)) return true;
            temp = board & (board >> 8); if (temp & (temp >> 16)) return true;
            temp = board & (board >> 1); if (temp & (temp >> 2)) return true;
            return false;
        }

        struct PossibleMoves {
            std::array<int, GAME_COLS> columns;
            int count = 0;
        };

        PossibleMoves get_possible_moves() const {
            PossibleMoves moves;
            static constexpr std::array<int, GAME_COLS> COL_ORDER = { 3, 4, 2, 5, 1, 6, 0 };
            for (int col : COL_ORDER) {
                if (is_valid_move(col)) {
                    moves.columns[moves.count++] = col;
                }
            }
            return moves;
        }
    };

    inline constexpr GameState get_initial_state() {
        return GameState{ 0, 0, Player::BLACK };
    }

} // namespace Connect4