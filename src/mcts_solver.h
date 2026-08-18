#pragma once
#include "mcts.h"
#include "solver/Solver.hpp"

namespace Connect4 {

	enum NodeStatus {UNKNOWN = 0, LOSE = 1, DRAW = 2, WIN = 3};

	class MCTSSolverNode : public MCTSNode {
	public:
		static const uint64_t invalidValue = std::numeric_limits<uint64_t>::max();
		static const uint64_t unexploredValue = std::numeric_limits<uint64_t>::max() - 1;
		NodeStatus status = NodeStatus::UNKNOWN;
		std::array<uint64_t, GAME_COLS> children;
		std::array<uint64_t, GAME_COLS> parents;
		void reset() override;
		MCTSSolverNode() {
			reset();
		}
	};

	class MCTSSolver : public MCTS {
	public:
		explicit MCTSSolver(float c_puct = 1.0f, float dirichlet_alpha = 1.0f, 
			float dirichlet_epsilon = 0.25f, float virtual_loss = 2.0f, int solver_depth = 5);
		std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<int>>
			find_leaf(const GameState& root_state, Player player,
				std::pmr::vector<std::pair<uint64_t, int>>* virtual_loss_path,
				std::pmr::polymorphic_allocator<void> alloc) override;

		bool is_leaf(const GameState& state) const override;

		void clear() override;

		void search_minibatch(int count, const GameState& state, Player player) override;

		void update_node_statuses(uint64_t root);
		std::array<bool, GAME_COLS> get_explorable_moves(MCTSSolverNode node, GameState state);
		// Add const GameState& leaf_state to the signature
		void backup_values(float value, const GameState& leaf_state, std::pmr::vector<GameState> states, std::pmr::vector<int> actions);
		void solve_node(MCTSSolverNode& node);
		std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
			get_policy_value(const GameState& state, float tau = 1.0f) const override;

	private:
		GameSolver::Solver solver_;
		int solver_depth_;
		std::pmr::unordered_map<uint64_t, MCTSSolverNode> solver_tree_;
	};
}