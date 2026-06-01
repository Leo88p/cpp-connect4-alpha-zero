// connect4_oracle.h
#pragma once
#include "connect4_game.h"
#include "minimax_player.h"  // Your existing MinimaxPlayer
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <optional>

namespace Connect4 {
    struct AnalysisResult {
        int forced = 0;      // Single available move
        int best_move = 0;   // Perfect + tie-break preferred
        int good_move = 0;   // Perfect but not tie-break preferred  
        int mistake = 0;     // win-to-draw or draw-to-loss
        int blunder = 0;     // win-to-loss
        int total = 0;

        void clear() {
            forced = best_move = good_move = mistake = blunder = total = 0;
        }

        // Convenience: get percentage of optimal moves (best+good)
        double optimal_rate() const {
            return total > 0 ? 100.0 * (best_move + good_move) / total : 100.0;
        }
    };

    class Oracle {
    public:
        explicit Oracle(int minimax_time_limit_ms = 500);  // Default 500ms per position for analysis
        ~Oracle() = default;

        // Get best move for current state (uses UCI cache + minimax fallback)
        int get_best_move(const GameState& state);

        // Analyze completed game: moves are column indices from initial state
        // opponent_player: which player's moves to evaluate
        AnalysisResult analyze_game(const std::vector<int>& moves, Player opponent_player);

        // Load UCI dataset to pre-seed perfect values
        // Format: 42 comma-separated values (a1..g6: x/o/b), then class (win/loss/draw)
        bool load_uci_dataset(const std::string& filepath);

        // Clear all caches (UCI + minimax TT)
        void clear_cache();

        // Configure minimax search depth/time for analysis
        void set_analysis_params(int time_limit_ms, int max_depth = 42);

        // Statistics
        size_t uci_cache_size() const { return uci_values_.size(); }
        uint64_t minimax_nodes_searched() const { return minimax_.get_nodes_searched(); }

    private:
        // Game-theoretic evaluation: +1 win, 0 draw, -1 loss for CURRENT player
        // Uses UCI cache first, then deep minimax with early termination
        int8_t evaluate_perfect(const GameState& state);

        // Check if state is terminal (win/draw) - returns {is_terminal, value_for_current_player}
        std::pair<bool, int8_t> is_terminal(const GameState& state) const;

        // Encode state as UCI dataset string: "a1,a2,...,g6" with x/o/b
        std::string encode_for_uci(const GameState& state) const;

        // Among optimal moves, select tie-broken "best" move (lowest column index)
        int select_tiebroken_best(const GameState& state, const std::vector<int>& optimal_moves);

        // Convert UCI class string to value for FIRST PLAYER (x)
        static int8_t uci_class_to_value(const std::string& cls);

        // Helper: get all moves that achieve the optimal outcome
        std::vector<int> get_optimal_moves(const GameState& state);

        MinimaxPlayer minimax_;                    // Your existing minimax for search
        std::unordered_map<std::string, int8_t> uci_values_;  // UCI dataset cache
        int analysis_time_limit_ms_;               // Time limit for deep analysis searches
        int analysis_max_depth_;                   // Max depth for analysis searches

        // Cache for minimax evaluations during a single analysis session (cleared per game)
        std::unordered_map<uint64_t, int8_t> eval_cache_;
    };

} // namespace Connect4