// tictactoe_game.h
#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace TicTacToe {

    constexpr int BOARD_SIZE = 3;
    constexpr int TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE;
    constexpr int WIN_LENGTH = 3;

    enum class Player : uint8_t {
        X = 0,  // First player (traditionally X)
        O = 1   // Second player (traditionally O)
    };

    struct GameState {
        uint16_t x_pieces = 0;      // Bitmask for X pieces (player 0)
        uint16_t o_pieces = 0;      // Bitmask for O pieces (player 1)
        uint16_t occupied = 0;      // Bitmask of all occupied cells
        Player current_player = Player::X;

        // Helper to get current player's pieces
        uint16_t get_player_pieces(Player p) const {
            return (p == Player::X) ? x_pieces : o_pieces;
        }

        // Check if game is terminal (win or draw)
        bool is_terminal() const {
            return is_win(Player::X) || is_win(Player::O) || (occupied == (1 << TOTAL_CELLS) - 1);
        }

        // Check win for specific player
        bool is_win(Player p) const {
            uint16_t pieces = get_player_pieces(p);
            for (uint16_t pattern : GameLogic::WIN_PATTERNS) {
                if ((pieces & pattern) == pattern) {
                    return true;
                }
            }
            return false;
        }

        // Check for draw (board full, no winner)
        bool is_draw() const {
            return (occupied == (1 << TOTAL_CELLS) - 1) && !is_win(Player::X) && !is_win(Player::O);
        }
    };

    class GameLogic {
    public:
        // 8 win patterns for 3x3 board (3 rows, 3 cols, 2 diagonals)
        static const std::array<uint16_t, 8> WIN_PATTERNS;
        static const GameState INITIAL_STATE;

        static std::vector<int> get_possible_moves(const GameState& state);
        static std::pair<GameState, bool> make_move(const GameState& state, int move); // move: 0-8 cell index
        static bool check_win(const GameState& state, Player player);
        static std::vector<std::string> render(const GameState& state);

        // Conversion helpers for Python integration
        static GameState from_list_representation(const std::vector<std::vector<int>>& board);
        static std::vector<std::vector<int>> to_list_representation(const GameState& state);

        // Utility: convert move index to (row, col)
        static std::pair<int, int> move_to_coord(int move) {
            return { move / BOARD_SIZE, move % BOARD_SIZE };
        }

        // Utility: convert (row, col) to move index
        static int coord_to_move(int row, int col) {
            return row * BOARD_SIZE + col;
        }
    };

    struct WinStats {
        struct Counts {
            int wins = 0;
            int losses = 0;
            int draws = 0;
        };

        void update(const std::string& key, const Counts& counts);
        const Counts& get(const std::string& key) const;

    private:
        std::unordered_map<std::string, Counts> stats_;
    };

} // namespace TicTacToe