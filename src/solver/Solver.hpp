/*
 * This file is part of Connect4 Game Solver <http://connect4.gamesolver.org>
 * Copyright (C) 2017-2019 Pascal Pons <contact@gamesolver.org>
 *
 * Connect4 Game Solver is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Connect4 Game Solver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Connect4 Game Solver. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <string>
#include "../connect4_game.h"
#include "TranspositionTable.hpp"
#include "OpeningBook.hpp"

namespace Connect4 {
namespace GameSolver {

class Solver {
 private:
  static constexpr int TABLE_SIZE = 24; // store 2^TABLE_SIZE elements in the transpositiontbale
  TranspositionTable < uint_t < GAME_COLS*(GAME_ROWS + 1) - TABLE_SIZE >, uint64_t, uint8_t, TABLE_SIZE > transTable;
  OpeningBook book{GAME_COLS, GAME_ROWS}; // opening book
  unsigned long long nodeCount; // counter of explored nodes.
  int columnOrder[GAME_COLS]; // column exploration order

  /**
   * Reccursively score connect 4 position using negamax variant of alpha-beta algorithm.
   * @param: position to evaluate, this function assumes nobody already won and
   *         current player cannot win next move. This has to be checked before
   * @param: alpha < beta, a score window within which we are evaluating the position.
   *
   * @return the exact score, an upper or lower bound score depending of the case:
   * - if actual score of position <= alpha then actual score <= return value <= alpha
   * - if actual score of position >= beta then beta <= return value <= actual score
   * - if alpha <= actual score <= beta then return value = actual score
   */
  int negamax(const GameState &P, int alpha, int beta);
  int negamax_depth(const GameState& P, int alpha, int beta, int depth);

 public:
  static const int INVALID_MOVE = -1000;

  // Returns the score of a position
  int solve(const GameState &P, bool weak = false);

  // Returns the score off all possible moves of a position as an array.
  // Returns INVALID_MOVE for unplayable columns
  std::vector<int> analyze(const GameState &P, bool weak = false);

  unsigned long long getNodeCount() const {
    return nodeCount;
  }

  void reset() {
    nodeCount = 0;
    transTable.reset();
  }

  void loadBook(std::string book_file) {
    book.load(book_file);
  }
  int solve_depth_limited(const GameState& P, int max_depth);
  std::vector<int> analyze_depth(const GameState& P, int max_depth);

  Solver(); // Constructor
};

} // namespace Connect4
} // namespace GameSolver
#endif
