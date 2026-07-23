#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "connect4_game.h" 
#include <iostream>

using namespace Connect4;

// Helper to count pieces (for debugging)
/*int count_pieces(uint64_t board) {
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
} */