#include "connect4_game.h"
#include <array>
#include <numeric>
#include <random>

namespace Connect4 {

    // Define static members here - only one definition across entire project
    const std::array<uint64_t, 69> GameLogic::WIN_PATTERNS = [] {
        std::array<uint64_t, 69> patterns{};
        size_t idx = 0;

        // Horizontal wins (4 in a row)
        for (int row = 0; row < GAME_ROWS; ++row) {
            for (int col = 0; col <= GAME_COLS - COUNT_TO_WIN; ++col) {
                uint64_t pattern = 0;
                for (int i = 0; i < COUNT_TO_WIN; ++i) {
                    int pos = row * GAME_COLS + (col + i);
                    pattern |= (1ULL << pos);
                }
                patterns[idx++] = pattern;
            }
        }

        // Vertical wins (4 in a column)
        for (int col = 0; col < GAME_COLS; ++col) {
            for (int row = 0; row <= GAME_ROWS - COUNT_TO_WIN; ++row) {
                uint64_t pattern = 0;
                for (int i = 0; i < COUNT_TO_WIN; ++i) {
                    int pos = (row + i) * GAME_COLS + col;
                    pattern |= (1ULL << pos);
                }
                patterns[idx++] = pattern;
            }
        }

        // Diagonal wins (up-right)
        for (int row = 0; row <= GAME_ROWS - COUNT_TO_WIN; ++row) {
            for (int col = 0; col <= GAME_COLS - COUNT_TO_WIN; ++col) {
                uint64_t pattern = 0;
                for (int i = 0; i < COUNT_TO_WIN; ++i) {
                    int pos = (row + i) * GAME_COLS + (col + i);
                    pattern |= (1ULL << pos);
                }
                patterns[idx++] = pattern;
            }
        }

        // Diagonal wins (down-right)
        for (int row = COUNT_TO_WIN - 1; row < GAME_ROWS; ++row) {
            for (int col = 0; col <= GAME_COLS - COUNT_TO_WIN; ++col) {
                uint64_t pattern = 0;
                for (int i = 0; i < COUNT_TO_WIN; ++i) {
                    int pos = (row - i) * GAME_COLS + (col + i);
                    pattern |= (1ULL << pos);
                }
                patterns[idx++] = pattern;
            }
        }

        return patterns;
        }();

    const GameState GameLogic::INITIAL_STATE = [] {
        GameState state;
        state.black_pieces = 0;
        state.white_pieces = 0;
        state.occupied = 0;
        for (int i = 0; i < GAME_COLS; ++i) {
            state.heights[i] = 0;
        }
        state.current_player = Player::BLACK;
        return state;
        }();

    // Implement all static methods here
    std::vector<int> GameLogic::get_possible_moves(const GameState& state) {
        std::vector<int> moves;
        moves.reserve(GAME_COLS);
        for (int col = 0; col < GAME_COLS; ++col) {
            if (state.heights[col] < GAME_ROWS) {
                moves.push_back(col);
            }
        }
        return moves;
    }

    std::pair<GameState, bool> GameLogic::make_move(const GameState& state, int col) {
        GameState new_state = state;

        // Calculate position: row = height[col], col = col
        int row = state.heights[col];
        int pos = row * GAME_COLS + col;
        uint64_t bit = 1ULL << pos;

        // Update board state
        if (state.current_player == Player::BLACK) {
            new_state.black_pieces |= bit;
        }
        else {
            new_state.white_pieces |= bit;
        }
        new_state.occupied |= bit;
        new_state.heights[col]++;
        new_state.current_player = (state.current_player == Player::BLACK) ? Player::WHITE : Player::BLACK;

        // Check for win
        bool won = check_win(new_state, state.current_player, pos);

        return { new_state, won };
    }

    bool GameLogic::check_win(const GameState& state, Player player, int pos) {
        uint64_t pieces = state.get_player_pieces(player);

        // Check all win patterns that include this position
        for (uint64_t pattern : WIN_PATTERNS) {
            if ((pattern & (1ULL << pos)) && ((pieces & pattern) == pattern)) {
                return true;
            }
        }
        return false;
    }

    std::vector<std::string> GameLogic::render(const GameState& state) {
        std::vector<std::string> board(GAME_ROWS, std::string(GAME_COLS, ' '));

        for (int col = 0; col < GAME_COLS; ++col) {
            for (int row = 0; row < state.heights[col]; ++row) {
                int tensor_row = GAME_ROWS - 1 - row;
                int pos = row * GAME_COLS + col;
                uint64_t bit = 1ULL << pos;
                char piece = ' ';
                if (state.black_pieces & bit) {
                    piece = '1';
                }
                else if (state.white_pieces & bit) {
                    piece = '0';
                }
                board[tensor_row][col] = piece;
            }
        }

        return board;
    }

    GameState GameLogic::from_list_representation(const std::vector<std::vector<int>>& field_lists) {
        GameState state = INITIAL_STATE;

        for (int col = 0; col < GAME_COLS; ++col) {
            const auto& col_pieces = field_lists[col];
            state.heights[col] = static_cast<uint8_t>(col_pieces.size());

            for (int row = 0; row < col_pieces.size(); ++row) {
                int pos = row * GAME_COLS + col;
                uint64_t bit = 1ULL << pos;
                if (col_pieces[row] == 1) { // BLACK
                    state.black_pieces |= bit;
                }
                else { // WHITE
                    state.white_pieces |= bit;
                }
            }
        }

        state.occupied = state.black_pieces | state.white_pieces;
        return state;
    }

    std::vector<std::vector<int>> GameLogic::to_list_representation(const GameState& state) {
        std::vector<std::vector<int>> field(GAME_COLS);

        for (int col = 0; col < GAME_COLS; ++col) {
            field[col].reserve(state.heights[col]);
            for (int row = 0; row < state.heights[col]; ++row) {
                int pos = row * GAME_COLS + col;
                uint64_t bit = 1ULL << pos;
                if (state.black_pieces & bit) {
                    field[col].push_back(1);
                }
                else {
                    field[col].push_back(0);
                }
            }
        }

        return field;
    }

    void WinStats::update(const std::string& key, const Counts& counts) {
        auto& v = stats_[key];
        v.wins += counts.wins;
        v.losses += counts.losses;
        v.draws += counts.draws;
    }

    const WinStats::Counts& WinStats::get(const std::string& key) const {
        auto it = stats_.find(key);
        static const Counts empty{ 0, 0, 0 };
        return it != stats_.end() ? it->second : empty;
    }

} // namespace Connect4