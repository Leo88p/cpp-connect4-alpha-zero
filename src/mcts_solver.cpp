#include "mcts_solver.h"

namespace Connect4 {

	void MCTSSolverNode::reset() {
		MCTSNode::reset();
        status = NodeStatus::UNKNOWN;
        parents.fill(unexploredValue);
        children.fill(unexploredValue);
	}

    void MCTSSolver::clear() {
        tree_.clear();
        solver_tree_.clear();
    }

	MCTSSolver::MCTSSolver(float c_puct, float dirichlet_alpha,
		float dirichlet_epsilon, float virtual_loss, int solver_depth) :
		MCTS::MCTS(c_puct, dirichlet_alpha, dirichlet_epsilon, virtual_loss, 0), 
		solver_depth_(solver_depth), solver_tree_(tree_.get_allocator()) {
		
		solver_tree_.reserve(10000);
        MCTSSolverNode invalid;
        invalid.status = NodeStatus::WIN;
        solver_tree_[MCTSSolverNode::invalidValue] = invalid;
	}

    std::array<bool, GAME_COLS> MCTSSolver::get_explorable_moves(MCTSSolverNode node, GameState state) {
        std::array<bool, GAME_COLS> explorableMoves = {};
        for (int col = 0; col < GAME_COLS; col++) {
            if (node.children[col] != MCTSSolverNode::unexploredValue) {
                explorableMoves[col] = state.canPlay(col) && solver_tree_[node.children[col]].status == NodeStatus::UNKNOWN;
            }
            else {
                explorableMoves[col] = true;
            }
        }
        return explorableMoves;
    }

    std::tuple<float, GameState, Player, std::pmr::vector<GameState>, std::pmr::vector<int>>
        MCTSSolver::find_leaf(const GameState& root_state, Player player,
            std::pmr::vector<std::pair<uint64_t, int>>* virtual_loss_path,
            std::pmr::polymorphic_allocator<void> alloc) {

        std::pmr::vector<GameState> states(alloc);
        std::pmr::vector<int> actions(alloc);
        GameState cur_state = root_state;
        Player cur_player = player;
        float value = std::numeric_limits<float>::quiet_NaN();

        while (!is_leaf(cur_state)) {
            states.push_back(cur_state);
            uint64_t cur_key = cur_state.key();

            auto& node = solver_tree_[cur_key];

            int result = solver_.solve_depth_limited(cur_state, solver_depth_);

            if (result != GameSolver::Solver::UNKNOWN_MOVE && result != -GameSolver::Solver::UNKNOWN_MOVE) {
                if (result > 0) {
                    node.status = NodeStatus::WIN;
                }
                else if (result < 0) {
                    node.status = NodeStatus::LOSE;
                }
                else {
                    node.status = NodeStatus::DRAW;
                }
            } 
            for (int col = 0; col < GAME_COLS; col++) {
                if (node.children[col] == MCTSSolverNode::unexploredValue) {
                    if (!cur_state.canPlay(col)) {
                        node.children[col] = MCTSSolverNode::invalidValue;
                    }
                    else if (!cur_state.canPlayNonLosingMove(col)) {
                        GameState NewState = cur_state;
                        NewState.playCol(col);
                        MCTSSolverNode newNode;
                        newNode.parents[col] = cur_key;
                        node.children[col] = NewState.key();
                        newNode.status = NodeStatus::WIN;
                        solver_tree_[NewState.key()] = newNode;
                    }
                }
            }

            solve_node(node);

            if (node.status != NodeStatus::UNKNOWN) {
                value = node.status == NodeStatus::WIN ? 1 : node.status == NodeStatus::LOSE ? -1 : 0;
                return { value, cur_state, cur_player,
                 std::move(states), std::move(actions) };
            }

            auto explorableMoves = get_explorable_moves(node, cur_state);

            float total_sqrt = std::sqrt(static_cast<float>(std::accumulate(
                node.visit_count.begin(), node.visit_count.end(), 0)) + 1.0f);

            std::array<float, GAME_COLS> score;
            const auto& probs = node.probs;
            const auto& counts = node.visit_count;

            if (cur_state.key() == root_state.key() && use_noise) {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = node.value_avg(i) + c_puct_ * ((1 - dirichlet_epsilon) * probs[i] + dirichlet_epsilon * dirichlet_noise[i]) *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }
            else {
                for (int i = 0; i < GAME_COLS; ++i) {
                    score[i] = node.value_avg(i) + c_puct_ * probs[i] *
                        total_sqrt / (1.0f + static_cast<float>(counts[i]));
                }
            }

            int best_action = -1;
            float best_score = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < GAME_COLS; ++i) {
                if (!explorableMoves[i]) score[i] = -std::numeric_limits<float>::infinity();
                if (score[i] > best_score) {
                    best_score = score[i];
                    best_action = i;
                }
            }

            if (best_action == -1) {
                throw std::runtime_error("No valid moves found in MCTS");
            }

            actions.push_back(best_action);

            if (virtual_loss_path != nullptr) {
                auto& mutable_node = solver_tree_[cur_key];
                mutable_node.value[best_action] -= virtual_loss_;
                mutable_node.visit_count[best_action]++;
                for (int i = 0; i < GAME_COLS; ++i) {
                    if (node.children[i] == MCTSSolverNode::unexploredValue) {
                        if (!cur_state.canPlay(i)) mutable_node.children[i] = MCTSSolverNode::invalidValue;
                        else if (!cur_state.canPlayNonLosingMove(i)) {
                            GameState NewState = cur_state;
                            NewState.playCol(i);
                            MCTSSolverNode newNode;
                            newNode.parents[i] = cur_key;
                            mutable_node.children[i] = NewState.key();
                            newNode.status = NodeStatus::WIN;
                            solver_tree_[NewState.key()] = newNode;
                        }
                    }
                }
                virtual_loss_path->emplace_back(cur_key, best_action);
            }

            if (cur_state.isWinningMove(best_action)) {
                value = -1.0f;
                cur_state.playCol(best_action);
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }

            cur_state.playCol(best_action);
            cur_player = static_cast<Player>(1 - static_cast<int>(cur_player));

            if (cur_state.possible() == 0) {
                value = 0.0f;
                return { value, cur_state, cur_player, std::move(states), std::move(actions) };
            }
        }

        return { std::numeric_limits<float>::quiet_NaN(), cur_state, cur_player,
                 std::move(states), std::move(actions) };
    }

    bool MCTSSolver::is_leaf(const GameState& state) const {
        uint64_t key = state.key();
        return solver_tree_.find(key) == solver_tree_.end();
    }

    void MCTSSolver::solve_node(MCTSSolverNode& node) {
        int winCount = 0;
        int drawCount = 0;
        if (node.status == NodeStatus::UNKNOWN) {
            for (int col = 0; col < GAME_COLS; col++) {
                if (node.children[col] == MCTSSolverNode::unexploredValue) {
                    continue;
                }
                if (solver_tree_[node.children[col]].status == NodeStatus::LOSE) {
                    node.status = NodeStatus::WIN;
                    break;
                }
                else if (solver_tree_[node.children[col]].status == NodeStatus::WIN) {
                    winCount++;
                }
                else if (solver_tree_[node.children[col]].status == NodeStatus::DRAW) {
                    drawCount++;
                }
            }
            if (winCount == GAME_COLS) {
                node.status = NodeStatus::LOSE;
            }
            else if (drawCount + winCount == GAME_COLS) {
                node.status = NodeStatus::DRAW;
            }
        }
    }

    void MCTSSolver::update_node_statuses(uint64_t root) {
        std::queue<uint64_t> nodes;
        nodes.push(root);
        while (!nodes.empty()) {
            auto& node = solver_tree_[nodes.front()];
            nodes.pop();
            solve_node(node);
            if (node.status != NodeStatus::UNKNOWN) {
                for (int col = 0; col < GAME_COLS; col++) {
                    if (node.parents[col] != MCTSSolverNode::unexploredValue) {
                        nodes.push(node.parents[col]);
                    }
                }
            }
        }
    }

    void MCTSSolver::backup_values(float value, const GameState& leaf_state, std::pmr::vector<GameState> states, std::pmr::vector<int> actions) {
        float cur_value = -value;
        for (int j = static_cast<int>(states.size()) - 1; j >= 0; --j) {
            if (j < static_cast<int>(actions.size()) &&
                actions[j] >= 0 && actions[j] < GAME_COLS) {

                uint64_t state_key = states[j].key();
                auto& node = solver_tree_[state_key];

                // FIX 1: Correctly link the child pointer, including the newly expanded leaf
                if (j == static_cast<int>(states.size()) - 1) {
                    node.children[actions[j]] = leaf_state.key();
                }
                else {
                    node.children[actions[j]] = states[j + 1].key();
                }

                // FIX 2: Correctly link the parent pointer (was pointing to itself before)
                if (j > 0) {
                    node.parents[actions[j - 1]] = states[j - 1].key();
                }

                node.visit_count[actions[j]]++;
                node.value[actions[j]] += cur_value;

                cur_value = -cur_value;
            }
        }
        if (states.size() > 0) {
            update_node_statuses(states[static_cast<int>(states.size()) - 1].key());
        }
    }

    void MCTSSolver::search_minibatch(int count, const GameState& state, Player player) {
        alignas(64) std::byte temp_buffer[8196];
        std::pmr::monotonic_buffer_resource mbr{ temp_buffer, sizeof(temp_buffer), std::pmr::new_delete_resource() };
        std::pmr::polymorphic_allocator<void> alloc(&mbr);

        // Track pending NN queries: state_key -> (leaf_state, player, backup_entries)
        std::pmr::unordered_map<uint64_t, std::tuple<GameState, Player, std::pmr::vector<BackupEntry>>> pending_queries{ alloc };

        for (int i = 0; i < count; ++i) {
            try {
                std::pmr::vector<std::pair<uint64_t, int>> virtual_loss_path{ alloc };

                auto [value, leaf_state, leaf_player, states, actions] =
                    find_leaf(state, player, &virtual_loss_path, alloc);

                if (!std::isnan(value)) {
                    // Terminal state: back up immediately

                    // FIX 2: Decouple virtual loss reversion from the states loop.
                    // Revert ALL virtual losses first (including the leaf node if it was added to the path).
                    auto& mutable_node = solver_tree_[leaf_state.key()];
                    mutable_node.status = value == -1.0f ? NodeStatus::LOSE : 
                        value == 1.0f ? NodeStatus::WIN : NodeStatus::DRAW;
                    for (auto& [key, action] : virtual_loss_path) {
                        if (solver_tree_.find(key) != solver_tree_.end()) {
                            solver_tree_[key].value[action] += virtual_loss_;
                            solver_tree_[key].visit_count[action]--;
                        }
                    }

                    // Now back up the true terminal value
                    backup_values(value, leaf_state, states, actions);
                }
                else {
                    uint64_t leaf_key = leaf_state.key();

                    // CRITICAL: Immediately add leaf to tree with default policy
                    if (solver_tree_.find(leaf_key) == solver_tree_.end()) {
                        MCTSSolverNode node;
                        auto valid_mask = get_explorable_moves(node, state);
                        int validCount = std::count(valid_mask.begin(), valid_mask.end(), true);

                        for (int j = 0; j < GAME_COLS; ++j) {
                            node.probs[j] = valid_mask[j] ? (1.0f / validCount) : 0.0f;
                            node.visit_count[j] = 0;
                            node.value[j] = 0.0f;
                        }
                        solver_tree_[leaf_key] = std::move(node);
                    }

                    // Queue this leaf for NN evaluation (if not already queued)
                    if (pending_queries.find(leaf_key) == pending_queries.end()) {
                        pending_queries[leaf_key] = std::make_tuple(
                            leaf_state, leaf_player, std::pmr::vector<BackupEntry>{alloc}
                        );
                    }

                    // Add backup entry to the pending query
                    auto& [_, __, backups] = pending_queries[leaf_key];
                    backups.emplace_back(alloc);
                    auto& new_entry = backups.back();
                    new_entry.states = std::move(states);
                    new_entry.actions = std::move(actions);
                    new_entry.virtual_loss_path = std::move(virtual_loss_path);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "MCTS search error: " << e.what() << std::endl;
                continue;
            }
        }

        // Now send all unique leaves to the neural network in one batch
        if (!pending_queries.empty()) {
            try {
                std::pmr::vector<std::future<std::pair<std::array<float, GAME_COLS>, float>>> futures{ alloc };
                std::pmr::vector<uint64_t> query_keys{ alloc };
                futures.reserve(pending_queries.size());

                for (auto& [leaf_key, data] : pending_queries) {
                    auto& [leaf_state, leaf_player, _] = data;
                    if (neural_worker_) {
                        futures.push_back(neural_worker_->submit_query(leaf_state, leaf_player));
                    }
                    query_keys.push_back(leaf_key);
                }

                // Process NN responses
                for (size_t i = 0; i < query_keys.size(); ++i) {
                    uint64_t leaf_key = query_keys[i];
                    auto& [leaf_state, leaf_player, backups] = pending_queries[leaf_key];

                    std::array<float, GAME_COLS> probs;
                    float value;

                    if (neural_worker_) {
                        auto [probs_, value_] = futures[i].get();
                        probs = probs_;
                        value = value_;
                    }
                    else {
                        // FIX 4: Fallback uniform policy now correctly counts ONLY valid moves
                        auto valid_mask = get_explorable_moves(solver_tree_[leaf_state.key()], leaf_state);

                        int validCount = std::count(valid_mask.begin(), valid_mask.end(), true);

                        for (int j = 0; j < GAME_COLS; ++j) {
                            probs[j] = valid_mask[j] ? (1.0f / validCount) : 0.0f;
                        }
                        value = 0.0f;
                    }

                    // Update the node with real NN values
                    auto& node = solver_tree_[leaf_key];
                    auto valid_mask = get_explorable_moves(node, leaf_state);

                    int validCount = std::count(valid_mask.begin(), valid_mask.end(), true);

                    for (int j = 0; j < GAME_COLS; ++j) {
                        node.probs[j] = valid_mask[j] ? probs[j] : 0.0f;
                    }

                    // Normalize probabilities
                    float sum = std::accumulate(node.probs.begin(), node.probs.end(), 0.0f);
                    if (sum > 1e-8f) {
                        for (float& p : node.probs) p /= sum;
                    }
                    else {
                        // FALLBACK: If NN policy is masked out entirely, fall back to uniform over valid moves
                        int valid_count = std::count(valid_mask.begin(), valid_mask.end(), true);
                        float uniform_prob = (valid_count > 0) ? (1.0f / valid_count) : 0.0f;
                        for (int j = 0; j < GAME_COLS; ++j) {
                            node.probs[j] = valid_mask[j] ? uniform_prob : 0.0f;
                        }
                    }

                    // Back up values through all backup entries for this leaf
                    for (auto& entry : backups) {
                        // FIX 6: Decouple virtual loss reversion from the backup loop.
                        // Revert ALL virtual losses first, guaranteeing the leaf node is cleaned up.
                        for (auto& [key, action] : entry.virtual_loss_path) {
                            if (solver_tree_.find(key) != solver_tree_.end()) {
                                solver_tree_[key].value[action] += virtual_loss_;
                                solver_tree_[key].visit_count[action]--;
                            }
                        }

                        // Now back up the true value
                        backup_values(value, leaf_state, entry.states, entry.actions);
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Neural network expansion error: " << e.what() << std::endl;
                // Revert virtual loss on error
                for (auto& [leaf_key, data] : pending_queries) {
                    auto& [_, __, backups] = data;
                    for (auto& entry : backups) {
                        for (auto& [key, action] : entry.virtual_loss_path) {
                            if (solver_tree_.find(key) != solver_tree_.end()) {
                                solver_tree_[key].value[action] += virtual_loss_;
                                solver_tree_[key].visit_count[action]--;
                            }
                        }
                    }
                }
                return;
            }
        }
    }
    std::pair<std::array<float, GAME_COLS>, std::array<float, GAME_COLS>>
        MCTSSolver::get_policy_value(const GameState& state, float tau) const {
        uint64_t state_key = state.key();
        const auto& node = solver_tree_.at(state_key);

        std::array<float, GAME_COLS> probs;
        std::array<float, GAME_COLS> values;

        // FIX 3: Get valid mask to avoid picking the sentinel node
        if (node.status != NodeStatus::UNKNOWN) {
            std::fill(probs.begin(), probs.end(), 0.0f);
            for (int i = 0; i < GAME_COLS; i++) {
                // FIX 3: Skip invalid moves
                if (!state.canPlay(i)) continue;

                if (node.children[i] == MCTSSolverNode::unexploredValue) continue;
                if (node.status == NodeStatus::WIN && solver_tree_.at(node.children[i]).status == NodeStatus::LOSE) {
                    probs[i] = 1.0f;
                    break;
                }
                else if (node.status == NodeStatus::DRAW && solver_tree_.at(node.children[i]).status == NodeStatus::DRAW) {
                    probs[i] = 1.0f;
                    break;
                }
                else if (node.status == NodeStatus::LOSE && solver_tree_.at(node.children[i]).status == NodeStatus::WIN) {
                    probs[i] = 1.0f;
                    break;
                }
            }
        }
        else {
            MCTSSolverNode mutable_node = node;
            for (int i = 0; i < GAME_COLS; i++) {
                if (mutable_node.children[i] == MCTSSolverNode::unexploredValue) continue;
                if (solver_tree_.at(node.children[i]).status == NodeStatus::WIN) {
                    mutable_node.visit_count[i] = 0;
                }
            }
            if (tau == 0.0f) {
                std::fill(probs.begin(), probs.end(), 0.0f);
                int best_action = std::distance(mutable_node.visit_count.begin(),
                    std::max_element(mutable_node.visit_count.begin(), mutable_node.visit_count.end()));
                probs[best_action] = 1.0f;
            }
            else {
                std::array<float, GAME_COLS> counts_pow;
                float sum = 0.0f;
                for (int i = 0; i < GAME_COLS; ++i) {
                    counts_pow[i] = std::pow(static_cast<float>(mutable_node.visit_count[i]), 1.0f / tau);
                    sum += counts_pow[i];
                }
                if (sum > 0.0f) {
                    for (int i = 0; i < GAME_COLS; ++i) probs[i] = counts_pow[i] / sum;
                }
                else {
                    float uniform_prob = 1.0f / GAME_COLS;
                    std::fill(probs.begin(), probs.end(), uniform_prob);
                }
            }
        }

        for (int i = 0; i < GAME_COLS; ++i) values[i] = node.value_avg(i);
        return { probs, values };
    }
}