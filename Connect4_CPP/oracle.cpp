// connect4_oracle.cpp
#include "oracle.h"
#include <cmath>
#include <chrono>

namespace Connect4 {

    Oracle::Oracle(int minimax_time_limit_ms)
        : minimax_(128)  // 128MB transposition table
        , analysis_time_limit_ms_(minimax_time_limit_ms)
        , analysis_max_depth_(42)
    {
        // Initialize minimax player with reasonable defaults
    }

    std::pair<bool, int8_t> Oracle::is_terminal(const GameState& state) const {
        // Check wins using existing WIN_PATTERNS
        uint64_t black = state.black_pieces;
        uint64_t white = state.white_pieces;

        for (uint64_t pattern : GameLogic::WIN_PATTERNS) {
            if ((black & pattern) == pattern) {
                // Black won: if current player is Black, they already won (impossible state)
                // So the player who just moved won -> current player lost
                return { true, -1 };
            }
            if ((white & pattern) == pattern) {
                return { true, -1 };
            }
        }

        // Check draw (board full)
        if (state.occupied == 0xFFFFFFFFFFFULL) {  // 42 bits set
            return { true, 0 };
        }

        return { false, 0 };
    }

    std::string Oracle::encode_for_uci(const GameState& state) const {
        // UCI format: iterate columns a-g (0-6), rows 1-6 (our row 0-5 = bottom)
        // Output: "a1,a2,...,a6,b1,...,g6" with values x=BLACK, o=WHITE, b=blank
        std::ostringstream oss;

        for (int col = 0; col < GAME_COLS; ++col) {
            for (int row = 0; row < GAME_ROWS; ++row) {
                if (row > 0 || col > 0) oss << ',';
                int pos = row * GAME_COLS + col;
                uint64_t bit = 1ULL << pos;
                char val;
                if (state.occupied & bit) {
                    val = (state.black_pieces & bit) ? 'x' : 'o';
                }
                else {
                    val = 'b';
                }
                oss << val;
            }
        }
        return oss.str();
    }

    int8_t Oracle::uci_class_to_value(const std::string& cls) {
        if (cls == "win") return +1;
        if (cls == "loss") return -1;
        return 0;  // draw
    }

    int8_t Oracle::evaluate_perfect(const GameState& state) {
        // Check terminal state first
        auto [terminal, term_val] = is_terminal(state);
        if (terminal) return term_val;

        // Check UCI dataset cache
        std::string uci_key = encode_for_uci(state);
        auto uci_it = uci_values_.find(uci_key);
        if (uci_it != uci_values_.end()) {
            // UCI value is for FIRST PLAYER (x). Convert to current player perspective.
            int8_t uci_val = uci_it->second;
            return (state.current_player == Player::BLACK) ? uci_val : -uci_val;
        }

        // Check local eval cache (for repeated positions during analysis)
        // Use simple hash: XOR of piece bitboards + player
        uint64_t hash = state.black_pieces ^ state.white_pieces ^
            (static_cast<uint64_t>(state.current_player) << 63);
        auto cache_it = eval_cache_.find(hash);
        if (cache_it != eval_cache_.end()) {
            return cache_it->second;
        }

        // Fallback: deep minimax search with game-theoretic interpretation
        // We use your MinimaxPlayer but interpret scores as win/draw/loss
        int score = minimax_.find_best_move(state, analysis_max_depth_, analysis_time_limit_ms_);

        // Convert minimax score to game-theoretic value
        // Your MinimaxPlayer uses WIN_SCORE=1000000, so we check thresholds
        int8_t result;
        if (score >= 1000000 - 42) {  // Forced win within depth
            result = +1;
        }
        else if (score <= -1000000 + 42) {  // Forced loss
            result = -1;
        }
        else {
            // Heuristic score: treat as draw for classification purposes
            // (Could add depth-to-win estimation for finer grading)
            result = 0;
        }

        // Cache result
        eval_cache_[hash] = result;
        return result;
    }

    std::vector<int> Oracle::get_optimal_moves(const GameState& state) {
        auto moves = GameLogic::get_possible_moves(state);
        if (moves.empty()) return {};

        // Find the best achievable value
        int8_t best_value = -2;
        std::vector<int> optimal;

        for (int col : moves) {
            auto [new_state, won] = GameLogic::make_move(state, col);
            int8_t value = won ? +1 : -evaluate_perfect(new_state);

            if (value > best_value) {
                best_value = value;
                optimal.clear();
                optimal.push_back(col);
            }
            else if (value == best_value) {
                optimal.push_back(col);
            }
        }

        return optimal;
    }

    int Oracle::select_tiebroken_best(const GameState& state, const std::vector<int>& optimal_moves) {
        if (optimal_moves.empty()) return -1;
        // Deterministic tie-break: lowest column index (matches your COLUMN_ORDER preference)
        return *std::min_element(optimal_moves.begin(), optimal_moves.end());
    }

    int Oracle::get_best_move(const GameState& state) {
        auto moves = GameLogic::get_possible_moves(state);
        if (moves.empty()) return -1;
        if (moves.size() == 1) return moves[0];

        // Quick checks first
        auto [terminal, _] = is_terminal(state);
        if (terminal) return -1;  // Game over

        // Try UCI cache for instant lookup
        std::string uci_key = encode_for_uci(state);
        auto uci_it = uci_values_.find(uci_key);
        if (uci_it != uci_values_.end()) {
            // For known positions, use minimax with shallow search to pick move
            return minimax_.find_best_move(state, 10, 100);  // Fast lookup
        }

        // Full minimax search for best move
        return minimax_.find_best_move(state, analysis_max_depth_, analysis_time_limit_ms_);
    }

    bool Oracle::load_uci_dataset(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;

        std::string line;
        int loaded = 0;

        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;

            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }

            // Expect 43 tokens: 42 board cells + 1 class label
            if (tokens.size() != 43) continue;

            // Build UCI key (already in a1..g6 order from dataset)
            std::string uci_key;
            for (int i = 0; i < 42; ++i) {
                if (i > 0) uci_key += ',';
                uci_key += tokens[i];
            }

            // Store value for FIRST PLAYER (x)
            uci_values_[uci_key] = uci_class_to_value(tokens[42]);
            ++loaded;
        }

        return loaded > 0;
    }

    void Oracle::set_analysis_params(int time_limit_ms, int max_depth) {
        analysis_time_limit_ms_ = time_limit_ms;
        analysis_max_depth_ = max_depth;
    }

    void Oracle::clear_cache() {
        uci_values_.clear();
        eval_cache_.clear();
        minimax_.clear();  // Clear minimax transposition table
    }

    AnalysisResult Oracle::analyze_game(const std::vector<int>& moves, Player opponent_player) {
        AnalysisResult result;
        eval_cache_.clear();  // Fresh cache for this analysis

        GameState state = GameLogic::INITIAL_STATE;

        for (size_t move_idx = 0; move_idx < moves.size(); ++move_idx) {
            int played_col = moves[move_idx];
            Player current_player = state.current_player;

            // Only analyze moves by specified opponent
            if (current_player == opponent_player) {
                auto possible_moves = GameLogic::get_possible_moves(state);

                // Classification: forced move?
                if (possible_moves.size() == 1) {
                    result.forced++;
                    result.total++;
                }
                else {
                    // Evaluate position BEFORE move (from current player's perspective)
                    int8_t pre_value = evaluate_perfect(state);

                    // Find all moves that achieve optimal outcome
                    std::vector<int> optimal_moves = get_optimal_moves(state);
                    int8_t optimal_value = evaluate_perfect(state);  // Best achievable

                    // Evaluate the ACTUAL move played
                    auto [actual_state, actual_won] = GameLogic::make_move(state, played_col);
                    int8_t actual_value = actual_won ? +1 : -evaluate_perfect(actual_state);

                    // Classify the move
                    if (actual_value == optimal_value) {
                        // Optimal move: check tie-break preference
                        int tiebroken = select_tiebroken_best(state, optimal_moves);
                        if (played_col == tiebroken) {
                            result.best_move++;
                        }
                        else {
                            result.good_move++;
                        }
                    }
                    else {
                        // Suboptimal: determine severity based on outcome change
                        if (optimal_value == +1) {
                            if (actual_value == 0) {
                                result.mistake++;  // Win to Draw
                            }
                            else if (actual_value == -1) {
                                result.blunder++;  // Win to Loss
                            }
                        }
                        else if (optimal_value == 0 && actual_value == -1) {
                            result.mistake++;  // Draw to Loss
                        }
                        // Note: if optimal_value == -1, position was already lost;
                        // all moves are theoretically equivalent (could add "delay loss" logic)
                    }
                    result.total++;
                }
            }

            // Apply move to advance state (for all players)
            auto [new_state, won] = GameLogic::make_move(state, played_col);
            state = new_state;

            // Stop if game ended
            if (won || state.occupied == 0xFFFFFFFFFFFULL) {
                break;
            }
        }

        return result;
    }

} // namespace Connect4