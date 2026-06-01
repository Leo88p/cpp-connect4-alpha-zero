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
        uint64_t black_pieces = 0;
        uint64_t white_pieces = 0;
        uint64_t occupied = 0; // black_pieces | white_pieces
        uint8_t heights[GAME_COLS] = { 0 }; // Current height of each column (0-6)
        Player current_player = Player::BLACK;

        bool operator==(const GameState& other) const {
            return black_pieces == other.black_pieces &&
                white_pieces == other.white_pieces &&
                std::equal(std::begin(heights), std::end(heights), std::begin(other.heights)) &&
                current_player == other.current_player;
        }

        uint64_t get_player_pieces(Player player) const {
            return player == Player::BLACK ? black_pieces : white_pieces;
        }

        uint64_t get_opponent_pieces(Player player) const {
            return player == Player::BLACK ? white_pieces : black_pieces;
        }

        bool is_valid_move(int col) const {
            return col >= 0 && col < GAME_COLS && heights[col] < GAME_ROWS;
        }

        // Convert to 64-bit key for hashing/transposition tables
        uint64_t to_key() const {
            // Use a standard hash combiner
            size_t seed = 0;

            // Hash black pieces
            seed ^= std::hash<uint64_t>()(black_pieces) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash white pieces
            seed ^= std::hash<uint64_t>()(white_pieces) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash heights
            for (int i = 0; i < GAME_COLS; ++i) {
                seed ^= std::hash<uint8_t>()(heights[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            // Hash current player
            seed ^= std::hash<uint8_t>()(static_cast<uint8_t>(current_player)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return static_cast<uint64_t>(seed);
        }
    };

    class GameLogic {
    public:
        // Declare static members without definitions
        static const std::array<uint64_t, 69> WIN_PATTERNS;
        static const GameState INITIAL_STATE;

        // Get all possible moves for current state
        static std::vector<int> get_possible_moves(const GameState& state);

        // Make a move and return new state with win information
        static std::pair<GameState, bool> make_move(const GameState& state, int col);

        // Check if the last move at position 'pos' by 'player' resulted in a win
        static bool check_win(const GameState& state, Player player, int pos);

        // Render the board as strings for display
        static std::vector<std::string> render(const GameState& state);

        // Convert to/from the Python-style list representation (for compatibility)
        static GameState from_list_representation(const std::vector<std::vector<int>>& field_lists);

        static std::vector<std::vector<int>> to_list_representation(const GameState& state);
    };

    // Statistics tracking (equivalent to update_counts in Python)
    class WinStats {
    public:
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

} // namespace Connect4