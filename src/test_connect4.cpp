#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "connect4_game.h" 
#include "connect4_solver.h"
#include "transportation_table.h" // Your TT
#include <iostream>

using namespace Connect4;

// Helper to count pieces (for debugging)
int count_pieces(uint64_t board) {
    return std::popcount(board);
}
void apply_moves(GameState& state, const std::vector<int>& moves) {
    for (int col : moves) {
        state.make_move(col);
    }
}

TEST_CASE("GameState: Basic Move and Unmove Symmetry", "[gamestate]") {
    GameState state = get_initial_state();
    uint64_t initial_hash = state.hash_key;

    REQUIRE(state.get_possible_moves().count == 7);
    REQUIRE(count_pieces(state.get_occupied()) == 0);

    // 1. Make a move
    bool win = state.make_move(3);
    REQUIRE_FALSE(win);
    REQUIRE(count_pieces(state.get_occupied()) == 1);
    REQUIRE(state.current_player == Player::WHITE);
    REQUIRE(state.hash_key != initial_hash); // Hash MUST change

    // 2. Unmake the move
    state.unmake_move(3);
    REQUIRE(count_pieces(state.get_occupied()) == 0);
    REQUIRE(state.current_player == Player::BLACK);

    // CRITICAL: Hash MUST be exactly restored. If this fails, your TT is poisoned!
    REQUIRE(state.hash_key == initial_hash);
}

TEST_CASE("GameState: Hash Consistency Over Multiple Moves", "[gamestate]") {
    GameState state = get_initial_state();
    uint64_t initial_hash = state.hash_key;

    // Play a sequence of moves
    std::vector<int> moves = { 3, 4, 3, 4, 3, 2, 5 };
    apply_moves(state, moves);

    uint64_t mid_hash = state.hash_key;
    REQUIRE(count_pieces(state.get_occupied()) == moves.size());

    // Unmake them in reverse order
    for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
        state.unmake_move(*it);
    }

    // Hash must perfectly match the start
    REQUIRE(state.hash_key == initial_hash);
    REQUIRE(count_pieces(state.get_occupied()) == 0);
}

TEST_CASE("GameState: Win Detection (Vertical)", "[win]") {
    GameState state = get_initial_state();
    // Black plays 4 times in column 0
    apply_moves(state, { 0, 1, 0, 2, 0, 3 });
    bool win = state.make_move(0); // Black wins

    REQUIRE(win == true);
    REQUIRE(state.check_win(Player::BLACK) == true);
    REQUIRE(state.check_win(Player::WHITE) == false);
}

TEST_CASE("GameState: Win Detection (Horizontal)", "[win]") {
    GameState state = get_initial_state();
    // Black plays in cols 0, 1, 2, 3 at row 0
    apply_moves(state, { 0, 4, 1, 4, 2, 4 });
    bool win = state.make_move(3); // Black wins

    REQUIRE(win == true);
    REQUIRE(state.check_win(Player::BLACK) == true);
}

TEST_CASE("GameState: Win Detection (Diagonal)", "[win]") {
    GameState state = get_initial_state();
    // Black plays diagonal down-right: (0,0), (1,1), (2,2), (3,3)
    apply_moves(state, { 0, 0, 1, 1, 2, 2 });
    // Col 3, Row 3
    bool win = state.make_move(3); // Black wins

    REQUIRE(win == true);
    REQUIRE(state.check_win(Player::BLACK) == true);
}
TEST_CASE("GameState: Deep Search Imitation", "[gamestate]") {
    GameState state = get_initial_state();

    std::vector<int> sequence = { 3, 2, 4, 3, 3, 4, 4, 2, 2, 3, 5, 1, 0, 0 };
    for (int c : sequence) state.make_move(c);

    uint64_t deep_hash = state.hash_key;
    uint64_t deep_black = state.black_pieces;
    uint64_t deep_white = state.white_pieces;

    state.make_move(3);
    state.make_move(3);

    state.unmake_move(3);
    state.unmake_move(3);

    REQUIRE(state.hash_key == deep_hash);
    REQUIRE(state.black_pieces == deep_black);
    REQUIRE(state.white_pieces == deep_white);
}

TEST_CASE("Solver finds immediate winning move and calculates correct score", "[solver][win]") {
    GameState state;
    // Setup: P1 has 3 in a row on the bottom (cols 0, 1, 2)
    // P2 wastes turns in col 6 so that col 3 remains empty for P1 to win!
    apply_moves(state, { 0, 6, 1, 6, 2, 6 });

    Connect4Solver solver;
    // Depth 1 is enough to find an immediate win
    auto [best_move, score] = solver.solve(state, 1);

    REQUIRE(best_move == 3);

    // 6 pieces are on the board. P1 makes the 7th move to win.
    // The score formula is 43 - absolute_ply. 
    // 43 - 7 = 36.
    REQUIRE(score == 36);
}

TEST_CASE("Solver blocks opponent's immediate win", "[solver][block]") {
    GameState state;
    // Setup: P1 has 3 in a row (0, 1, 2). P2 wastes turns in 6.
    // It is P2's turn. P2 MUST play 3 to block P1 from winning at 3.
    apply_moves(state, { 0, 6, 1, 6, 2 });

    Connect4Solver solver;
    // Depth 2 is enough to see the immediate threat (P1 winning next turn) and block it
    auto [best_move, score] = solver.solve(state, 2);

    REQUIRE(best_move == 3);

    // Since P2 successfully blocks and no one wins within depth 2, 
    // the score should evaluate to a draw (0) from P2's perspective.
    REQUIRE(score == 0);
}

TEST_CASE("Depth limit strictly restricts search horizon", "[solver][depth]") {
    GameState state;
    // Setup a board with no immediate wins (vertical stacks)
    apply_moves(state, { 3, 3, 4, 4, 3, 4 }); // P1: 3,3,3 P2: 4,4,4

    Connect4Solver solver;

    SECTION("Depth 1 returns 0 when no immediate win exists") {
        auto [best_move, score] = solver.solve(state, 1);
        REQUIRE(score == 0);
    }

    SECTION("Depth 2 returns 0 when no win exists within 2 plies") {
        auto [best_move, score] = solver.solve(state, 2);
        REQUIRE(score == 0);
    }
}

TEST_CASE("Transposition table correctly identifies identical board states", "[solver][tt]") {
    Connect4Solver solver;

    GameState state1;
    // Sequence 1: Fill col 0, then col 1
    apply_moves(state1, { 0, 0, 1, 1 });
    auto [move1, score1] = solver.solve(state1, 5);

    GameState state2;
    // Sequence 2: Fill col 1, then col 0
    // This results in the EXACT same physical board state as state1
    apply_moves(state2, { 1, 1, 0, 0 });

    // The solver should hit the Transposition Table for state2 and return 
    // the exact same evaluation and best move as state1
    auto [move2, score2] = solver.solve(state2, 5);

    REQUIRE(score1 == score2);
    REQUIRE(move1 == move2);
}

TEST_CASE("Clear cache resets transposition table without crashing", "[solver][tt]") {
    Connect4Solver solver;
    GameState state;
    apply_moves(state, { 3, 3, 4, 4 });

    // Populate the cache
    solver.solve(state, 5);

    // Clear it
    solver.clear_cache();

    // Ensure it can still solve correctly after clearing
    auto [move, score] = solver.solve(state, 5);
    REQUIRE(move >= 0);
}