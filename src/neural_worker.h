// neural_worker.h
#pragma once
#pragma warning(disable: 4996) 
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>                 // c10::cuda::CUDAStream, CUDAStreamGuard
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <array>
#include <cstring>  // for std::memcpy
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

        // === CUDA OPTIMIZATIONS ===
        torch::Tensor cpu_staging_buffer_;
        torch::Tensor gpu_input_buffer_;          // Reusable [B, 2, R, C] on GPU
        torch::Tensor pinned_probs_host_;         // Pinned [B, GAME_COLS] on CPU
        torch::Tensor pinned_values_host_;        // Pinned [B, 1] on CPU
        c10::cuda::CUDAStream stream_ = c10::cuda::getCurrentCUDAStream(); // Safe default
        at::cuda::CUDAEvent sync_event_;                                  // Has default ctor

    public:
        NeuralWorker(Connect4Net net, torch::Device device, int batch_size = 256)
            : net_(std::move(net)), device_(device), batch_size_(batch_size) {

            net_->eval();

            // Initialize CUDA resources if using GPU
            if (device_.is_cuda()) {
                stream_ = c10::cuda::getStreamFromPool();

                // 1. PINNED CPU buffer (zero-copy async H2D compatible)
                cpu_staging_buffer_ = torch::empty(
                    { batch_size_, 2, GAME_ROWS, GAME_COLS },
                    torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true)
                );

                // 2. GPU buffer for inference
                gpu_input_buffer_ = torch::empty(
                    { batch_size_, 2, GAME_ROWS, GAME_COLS },
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_)
                );

                // 3. Pinned host buffers for D2H results
                pinned_probs_host_ = torch::empty({ batch_size_, GAME_COLS },
                    torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                pinned_values_host_ = torch::empty({ batch_size_, 1 },
                    torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
            }

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

            // CPU-side staging buffers for input encoding (avoid repeated allocs)
            std::vector<GameState> states;
            std::vector<Player> players;
            states.reserve(batch_size_);
            players.reserve(batch_size_);

            while (running_) {
                batch.clear();
                states.clear();
                players.clear();

                // === Batch Collection (unchanged logic) ===
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
                    if (!running_ && queue_.empty()) break;

                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();

                    auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::microseconds(500);
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

                // === Prepare Input Data ===
                for (const auto& q : batch) {
                    states.push_back(q.state);
                    players.push_back(q.player);
                }

                // === GPU EXECUTION PATH (async, explicit stream) ===
                if (device_.is_cuda()) {
                    c10::cuda::CUDAStreamGuard guard(stream_);

                    // 1. Fill CPU staging buffer (fast, no GPU access violations)
                    state_lists_to_batches(cpu_staging_buffer_, states, players);

                    // 2. Async Host-to-Device copy
                    cudaMemcpyAsync(gpu_input_buffer_.data_ptr<float>(),
                        cpu_staging_buffer_.data_ptr<float>(),
                        cpu_staging_buffer_.nbytes(),
                        cudaMemcpyHostToDevice,
                        stream_.stream());

                    // 3. Run inference on GPU buffer
                    auto input_view = gpu_input_buffer_.narrow(0, 0, static_cast<int64_t>(batch.size()));

                    torch::NoGradGuard no_grad;
                    auto output = net_->forward(input_view);

                    auto probs_gpu = torch::softmax(std::get<0>(output), 1);
                    auto values_gpu = std::get<1>(output);

                    // 4. Async Device-to-Host copy for results
                    auto probs_view = pinned_probs_host_.narrow(0, 0, static_cast<int64_t>(batch.size()));
                    auto values_view = pinned_values_host_.narrow(0, 0, static_cast<int64_t>(batch.size()));

                    cudaMemcpyAsync(probs_view.data_ptr<float>(), probs_gpu.data_ptr<float>(),
                        probs_view.nbytes(), cudaMemcpyDeviceToHost, stream_.stream());
                    cudaMemcpyAsync(values_view.data_ptr<float>(), values_gpu.data_ptr<float>(),
                        values_view.nbytes(), cudaMemcpyDeviceToHost, stream_.stream());

                    // 5. Wait for GPU work + D2H copies to complete
                    sync_event_.record(stream_);
                    sync_event_.synchronize(); // CPU blocks here

                    // 6. Distribute results
                    const float* probs_ptr = probs_view.data_ptr<float>();
                    const float* values_ptr = values_view.data_ptr<float>();

                    for (size_t i = 0; i < batch.size(); ++i) {
                        std::array<float, GAME_COLS> probs;
                        std::memcpy(probs.data(), probs_ptr + i * GAME_COLS, GAME_COLS * sizeof(float));
                        batch[i].promise.set_value({ probs, values_ptr[i] });
                    }
                }
            }
        }

        ~NeuralWorker() {
            running_ = false;
            cv_.notify_all();

            // Ensure GPU work completes before destruction
            if (device_.is_cuda()) {
                torch::cuda::synchronize(device_.index());
            }

            if (worker_.joinable()) {
                worker_.join();
            }
        }

        // Prevent copying
        NeuralWorker(const NeuralWorker&) = delete;
        NeuralWorker& operator=(const NeuralWorker&) = delete;
    };

} // namespace Connect4