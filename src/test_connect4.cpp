#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "connect4_game.h" 
#include <iostream>
#include "mcts.h"

using namespace Connect4;

void apply_moves(GameState& state, const std::vector<int>& moves) {
    for (int col : moves) {
        state.playCol(col);
    }
}

TEST_CASE("GameState: Win Detection (Vertical)", "[win]") {
    GameState state;
    // Black plays 4 times in column 0
    apply_moves(state, { 0, 1, 0, 2, 0, 3 });
    bool win = state.isWinningMove(0); // Black wins

    REQUIRE(win == true);
}

TEST_CASE("GameState: Win Detection (Horizontal)", "[win]") {
    GameState state;
    // Black plays in cols 0, 1, 2, 3 at row 0
    apply_moves(state, { 0, 4, 1, 4, 2, 4 });
    bool win = state.isWinningMove(3); // Black wins

    REQUIRE(win == true);
}

TEST_CASE("GameState: Win Detection (Diagonal)", "[win]") {
    GameState state;
    // Black plays diagonal down-right: (0,0), (1,1), (2,2), (3,3)
    apply_moves(state, { 0, 0, 1, 1, 2, 2 });
    // Col 3, Row 3
    bool win = state.isWinningMove(3); // Black wins

    REQUIRE(win == true);
}

TEST_CASE("Basic MCTS test") {
    GameState state;
    MCTS mcts(1, 0, 1);
    mcts.search_batch(7, 1, state, Player::BLACK);
    REQUIRE(mcts.size() == 7);
    mcts.clear();
    mcts.search_batch(1, 7, state, Player::BLACK);
    REQUIRE(mcts.size() == 7);
}

TEST_CASE("MCTS win in 1 test") {
    GameState state;
    apply_moves(state, { 1, 2, 1, 2, 1, 2 });
    MCTS mcts(1, 0, 1);
    mcts.search_batch(8, 1, state, Player::BLACK);
    auto policy = mcts.get_policy_value(state, 1).first;
    auto max_it = std::max_element(policy.begin(), policy.end());
    auto argmax = std::distance(policy.begin(), max_it);
    REQUIRE(argmax == 1);
}