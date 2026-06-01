// tictactoe_game.cpp
#include "game.h"
#include <array>
#include <numeric>
#include <random>

namespace TicTacToe {

    // Define static members
    const std::array<uint16_t, 8> GameLogic::WIN_PATTERNS = [] {
        std::array<uint16_t, 8> patterns{};
        size_t idx = 0;

        // Horizontal wins (3 rows)
        for (int row = 0; row < BOARD_SIZE; ++row) {
            uint16_t pattern = 0;
            for (int col = 0; col < BOARD_SIZE; ++col) {
                int pos = row * BOARD_SIZE + col;
                pattern |= (1 << pos);
            }
            patterns[idx++] = pattern;
        }

        // Vertical wins (3 columns)
        for (int col = 0; col < BOARD_SIZE; ++col) {
            uint16_t pattern = 0;
            for (int row = 0; row < BOARD_SIZE; ++row) {
                int pos = row * BOARD_SIZE + col;
                pattern |= (1 << pos);
            }
            patterns[idx++] = pattern;
        }

        // Main diagonal (top-left to bottom-right)
        {
            uint16_t pattern = 0;
            for (int i = 0; i < BOARD_SIZE; ++i) {
                int pos = i * BOARD_SIZE + i;
                pattern |= (1 << pos);
            }
            patterns[idx++] = pattern;
        }

        // Anti-diagonal (top-right to bottom-left)
        {
            uint16_t pattern = 0;
            for (int i = 0; i < BOARD_SIZE; ++i) {
                int pos = i * BOARD_SIZE + (BOARD_SIZE - 1 - i);
                pattern |= (1 << pos);
            }
            patterns[idx++] = pattern;
        }

        return patterns;
        }();

    const GameState GameLogic::INITIAL_STATE = [] {
        GameState state;
        state.x_pieces = 0;
        state.o_pieces = 0;
        state.occupied = 0;
        state.current_player = Player::X;
        return state;
        }();

    std::vector<int> GameLogic::get_possible_moves(const GameState& state) {
        std::vector<int> moves;
        moves.reserve(TOTAL_CELLS);
        for (int pos = 0; pos < TOTAL_CELLS; ++pos) {
            if (!(state.occupied & (1 << pos))) {
                moves.push_back(pos);
            }
        }
        return moves;
    }

    std::pair<GameState, bool> GameLogic::make_move(const GameState& state, int move) {
        GameState new_state = state;
        uint16_t bit = 1 << move;

        // Place piece for current player
        if (state.current_player == Player::X) {
            new_state.x_pieces |= bit;
        }
        else {
            new_state.o_pieces |= bit;
        }
        new_state.occupied |= bit;
        new_state.current_player = (state.current_player == Player::X) ? Player::O : Player::X;

        // Check if this move created a win
        bool won = check_win(new_state, state.current_player);

        return { new_state, won };
    }

    bool GameLogic::check_win(const GameState& state, Player player) {
        uint16_t pieces = state.get_player_pieces(player);
        for (uint16_t pattern : WIN_PATTERNS) {
            if ((pieces & pattern) == pattern) {
                return true;
            }
        }
        return false;
    }

    std::vector<std::string> GameLogic::render(const GameState& state) {
        std::vector<std::string> board(BOARD_SIZE, std::string(BOARD_SIZE, ' '));

        for (int pos = 0; pos < TOTAL_CELLS; ++pos) {
            int row = pos / BOARD_SIZE;
            int col = pos % BOARD_SIZE;
            uint16_t bit = 1 << pos;
            if (state.x_pieces & bit) {
                board[row][col] = 'X';
            }
            else if (state.o_pieces & bit) {
                board[row][col] = 'O';
            }
        }

        return board;
    }

    GameState GameLogic::from_list_representation(const std::vector<std::vector<int>>& board) {
        // board[row][col]: 1=X, -1=O, 0=empty
        GameState state = INITIAL_STATE;
        int x_count = 0, o_count = 0;

        for (int row = 0; row < BOARD_SIZE; ++row) {
            for (int col = 0; col < BOARD_SIZE; ++col) {
                int val = board[row][col];
                int pos = row * BOARD_SIZE + col;
                uint16_t bit = 1 << pos;

                if (val == 1) {
                    state.x_pieces |= bit;
                    state.occupied |= bit;
                    x_count++;
                }
                else if (val == -1) {
                    state.o_pieces |= bit;
                    state.occupied |= bit;
                    o_count++;
                }
            }
        }

        // Determine current player based on piece counts (X starts)
        state.current_player = (x_count == o_count) ? Player::X : Player::O;
        return state;
    }

    std::vector<std::vector<int>> GameLogic::to_list_representation(const GameState& state) {
        std::vector<std::vector<int>> board(BOARD_SIZE, std::vector<int>(BOARD_SIZE, 0));

        for (int pos = 0; pos < TOTAL_CELLS; ++pos) {
            int row = pos / BOARD_SIZE;
            int col = pos % BOARD_SIZE;
            uint16_t bit = 1 << pos;
            if (state.x_pieces & bit) {
                board[row][col] = 1;   // X
            }
            else if (state.o_pieces & bit) {
                board[row][col] = -1;  // O
            }
        }

        return board;
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

} // namespace TicTacToe