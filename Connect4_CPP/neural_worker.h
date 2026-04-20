// neural_worker.h
#pragma once
#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <array>
#include "connect4_game.h"
#include "model.h"

namespace Connect4 {

    struct NeuralQuery {
        GameState state;
        Player player;
        std::promise<std::pair<std::array<float, GAME_COLS>, float>> promise;
    };

    class NeuralWorker {
        std::queue<NeuralQuery> queue_;
        std::mutex mutex_;
        std::condition_variable cv_;
        Connect4Net net_;
        torch::Device device_;
        std::thread worker_;
        std::atomic<bool> running_{ true };
        int batch_size_;

    public:
        NeuralWorker(Connect4Net net, torch::Device device, int batch_size = 256)
            : net_(net), device_(device), batch_size_(batch_size) {
            net_->eval();
            worker_ = std::thread([this]() { worker_loop(); });
        }

        std::future<std::pair<std::array<float, GAME_COLS>, float>>
            submit_query(const GameState& state, Player player) {
            NeuralQuery q{ state, player, {} };
            auto future = q.promise.get_future();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(std::move(q));
            }
            cv_.notify_one();
            return future;
        }

        void worker_loop() {
            std::vector<NeuralQuery> batch;
            batch.reserve(batch_size_);

            while (running_) {
                batch.clear();

                // Collect batch with timeout
                {
                    std::unique_lock<std::mutex> lock(mutex_);

                    // Wait for first query
                    cv_.wait(lock, [this] { return !queue_.empty() || !running_; });

                    if (!running_ && queue_.empty()) break;

                    // Grab first query
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();

                    // Collect more queries (up to batch_size, with short timeout)
                    auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::microseconds(500); // 0.5ms max wait

                    while (batch.size() < static_cast<size_t>(batch_size_) &&
                        !queue_.empty()) {

                        auto remaining = deadline - std::chrono::steady_clock::now();
                        if (remaining <= std::chrono::milliseconds::zero()) break;

                        if (cv_.wait_for(lock, remaining,
                            [this] { return !queue_.empty(); })) {
                            batch.push_back(std::move(queue_.front()));
                            queue_.pop();
                        }
                    }
                }

                if (batch.empty()) continue;

                // Build batch tensors
                std::vector<GameState> states;
                std::vector<Player> players;
                states.reserve(batch.size());
                players.reserve(batch.size());

                for (const auto& q : batch) {
                    states.push_back(q.state);
                    players.push_back(q.player);
                }

                // Single forward pass for ALL queries
                auto states_v = state_lists_to_batch(states, players, device_);

                torch::NoGradGuard no_grad;
                auto output = net_->forward(states_v);

                auto probs_v = torch::softmax(std::get<0>(output), 1).cpu();
                auto values_v = std::get<1>(output).cpu();

                // Distribute results back to waiting MCTS instances
                auto probs_accessor = probs_v.accessor<float, 2>();
                auto values_accessor = values_v.accessor<float, 2>();

                for (size_t i = 0; i < batch.size(); ++i) {
                    std::array<float, GAME_COLS> probs;
                    for (int j = 0; j < GAME_COLS; ++j) {
                        probs[j] = probs_accessor[i][j];
                    }
                    batch[i].promise.set_value({ probs, values_accessor[i][0] });
                }
            }
        }

        ~NeuralWorker() {
            running_ = false;
            cv_.notify_all();
            if (worker_.joinable()) {
                worker_.join();
            }
        }

        // Prevent copying
        NeuralWorker(const NeuralWorker&) = delete;
        NeuralWorker& operator=(const NeuralWorker&) = delete;
    };

} // namespace Connect4