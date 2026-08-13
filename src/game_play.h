#pragma once
#include <vector>
#include <optional>
#include <tuple>
#include <array>

#include <torch/torch.h>

#include "mcts.h"

namespace Connect4 {

    using ReplayBuffer = std::deque<std::tuple<GameState, Player, std::array<float, GAME_COLS>, float>>;

    // Play game function
    std::pair<float, int> play_game(
        MCTS& mcts_store,
        std::vector<ReplayBuffer::value_type>* local_buffer,
        Connect4Net& net1,
        Connect4Net& net2,
        int steps_before_tau_0,
        int mcts_searches,
        int mcts_batch_size,
        float c_discount,
        const torch::Device& device);

} // namespace Connect4