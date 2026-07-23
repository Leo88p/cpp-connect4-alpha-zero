#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sstream>
#include <optional>
#include <numeric>
#include <unordered_map>
#include <csignal>
#include <atomic>
#include <future>
#include <mutex>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/optim.h>

#include "connect4_game.h"
#include "mcts.h"
#include "model.h"
#include "utils.h"
#include "game_play.h"
#include "config.h"

namespace fs = std::filesystem;
using namespace Connect4;

// Global flag for termination handling
std::atomic<bool> terminate_requested(false);
std::atomic<bool> saving_checkpoint(false);

// Add this class (minimal thread pool):
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
                });
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using return_type = decltype(f());
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(queue_mutex); stop = true; }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n\nTermination signal received. Gracefully shutting down..." << std::endl;

        if (saving_checkpoint) {
            std::cout << "Currently saving checkpoint, please wait..." << std::endl;
            return;
        }

        terminate_requested = true;
    }
}

void save_model(Connect4Net& net, const std::string& path) {
    try {
        saving_checkpoint = true;
        torch::save(net, path);
        std::cout << "Model saved to: " << path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }
    saving_checkpoint = false;
}

void load_model(Connect4Net& net, const std::string& path, const torch::Device& device) {
    if (fs::exists(path)) {
        try {
            torch::load(net, path);
            net->to(device);
            std::cout << "Model loaded successfully from: " << path << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading model from " << path << ": " << e.what() << std::endl;
        }
    }
    else {
        std::cerr << "Model file not found: " << path << std::endl;
    }
}

// Function to save final checkpoint and exit
void save_and_exit(Connect4Net& net, const fs::path& saves_path, int step_idx) {
    std::cout << "\nSaving final checkpoint before exit..." << std::endl;

    // Save final checkpoint
    std::ostringstream final_filename;
    final_filename << "final_checkpoint_" << std::setw(5) << std::setfill('0') << step_idx << ".pt";
    fs::path final_save_path = saves_path / final_filename.str();
    save_model(net, final_save_path.string());

    // Save training progress info including MCTS state
    std::ostringstream progress_filename;
    progress_filename << "training_progress_" << std::setw(5) << std::setfill('0') << step_idx << ".txt";
    fs::path progress_path = saves_path / progress_filename.str();

    std::ofstream progress_file(progress_path);
    if (progress_file.is_open()) {
        progress_file << "step_idx=" << step_idx << std::endl;
        progress_file << "timestamp=" << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
        progress_file.close();
        std::cout << "Training progress saved to: " << progress_path.string() << std::endl;
    }

    std::cout << "Graceful shutdown complete. Exiting..." << std::endl;
}

// Clear replay buffer but optionally keep recent data for warm start
void clear_replay_buffer(ReplayBuffer& buffer, float keep_ratio = 0.0f) {
    if (keep_ratio <= 0.0f || buffer.empty()) {
        buffer.clear();
        std::cout << "Replay buffer cleared completely." << std::endl;
        return;
    }
}

int main(int argc, char** argv) {
    // In main(), create ONCE:
    torch::set_num_threads(1);        // Limit intra-op parallelism per forward
    torch::set_num_interop_threads(1); // Limit inter-op parallelism

    // Setup signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);  // Ctrl+C
    std::signal(SIGTERM, signal_handler); // Termination request

    std::string config_path = "config.txt";
    if (argc > 1) {
        config_path = argv[1]; // Usage: ./train my_custom_config.txt
    }

    // 2. Load configuration
    Config cfg;
    cfg.loadFromFile(config_path);

    ThreadPool pool(cfg.parallel_games);

    if (cfg.run_name.empty()) {
        std::cerr << "Error: --name argument is required" << std::endl;
        return 1;
    }

    std::cout << "Starting training with name: " << cfg.run_name << std::endl;
    std::cout << "Device: " << cfg.device_str << std::endl;

    if (!cfg.net_to_load.empty()) {
        std::cout << "Loading network from: " << cfg.net_to_load << std::endl;
    }

    torch::Device device(cfg.device_str);
    fs::path saves_path = fs::path("saves") / cfg.run_name;
    fs::create_directories(saves_path);

    // Setup CSV logging
    fs::path logs_path = saves_path / "logs";
    fs::create_directories(logs_path);
    CSVLogger csv_logger((logs_path / "metrics.csv").string());

    // Initialize network
    Connect4Net net = std::make_shared<Connect4NetImpl>(cfg.num_blocks, cfg.num_filters);
    net->to(device);  // Ensure network is on the correct device from the start

    if (!cfg.net_to_load.empty()) {
        fs::path net_path = saves_path / cfg.net_to_load;
        if (fs::exists(net_path)) {
            std::cout << "Loading network weights from " << net_path.string() << std::endl;
            try {
                torch::load(net, net_path.string());
                std::cout << "Network loaded successfully" << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "Error loading network: " << e.what() << std::endl;
                std::cerr << "Creating new network instead" << std::endl;
                net = std::make_shared<Connect4NetImpl>();
                net->to(device);
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading network: " << e.what() << std::endl;
                std::cerr << "Creating new network instead" << std::endl;
                net = std::make_shared<Connect4NetImpl>();
                net->to(device);
            }
        }
        else {
            std::cerr << "Warning: Network file not found at " << net_path.string() << std::endl;
        }
    }
    net->to(device);

    // Print network architecture
    std::cout << *net << std::endl;

    torch::optim::SGDOptions sgd_opts = torch::optim::SGDOptions(cfg.learning_rate).momentum(0.9).weight_decay(1e-4);
    torch::optim::SGD optimizer(net->parameters(), sgd_opts);

    // Replay buffer
    ReplayBuffer replay_buffer;

    int step_idx = 0;

    std::deque<float> loss_history;
    int steps_since_last_improvement = 0;
    float best_loss = std::numeric_limits<float>::max();

    if (!cfg.net_to_load.empty()) {
        try {
            // Parse step_idx and potentially mcts state from filename
            std::string filename = cfg.net_to_load;
            size_t underscore = filename.find('_');
            size_t dot = filename.find('.');

            if (underscore != std::string::npos && dot != std::string::npos) {
                step_idx = std::stoi(filename.substr(underscore + 1, dot - 1));
                std::cout << "Resuming from step " << step_idx << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing filename: " << e.what() << std::endl;
        }
    }

    SimpleTracker tracker(csv_logger, 1);
    std::mt19937 rng(std::random_device{}());

    // Training loop with termination handling
    while (!terminate_requested) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int game_steps = 0;
        int total_leaves = 0; // Track leaves properly

        // Check for termination before starting episodes
        if (terminate_requested) break;

        auto neural_worker = std::make_unique<Connect4::NeuralWorker>(net, device, cfg.parallel_games);
        net->eval(); 
        {
            torch::NoGradGuard no_grad;
            std::vector<std::unique_ptr<MCTS>> all_mcts;
            all_mcts.reserve(cfg.parallel_games * 2);
            for (int i = 0; i < cfg.parallel_games * 2; ++i) {
                all_mcts.emplace_back(std::make_unique<MCTS>(cfg.c_puct, cfg.c_fpu, cfg.virtual_loss));
            }
            for (auto& mcts_ptr : all_mcts) {
                mcts_ptr->set_neural_worker(neural_worker.get());
            }

            std::mutex replay_mutex;

            for (int episode_start = 0; episode_start < cfg.play_episodes && !terminate_requested; episode_start += cfg.parallel_games) {
                int batch_games = std::min(cfg.parallel_games, cfg.play_episodes - episode_start);

                std::vector<std::future<std::pair<int, int>>> futures;
                std::vector<std::vector<ReplayBuffer::value_type>> local_buffers(batch_games);

                for (int i = 0; i < batch_games; ++i) {
                    futures.push_back(pool.enqueue(
                        [&, i, episode_idx = episode_start + i]() -> std::pair<int, int> {

                            // Запускаем игру, передавая локальный буфер
                            auto [game_result, steps] = play_game(
                                &all_mcts[i * 2], &local_buffers[i],
                                net, net,
                                cfg.steps_before_tau_0, cfg.mcts_batches, cfg.mcts_batch_size,
                                std::nullopt, device
                            );

                            int leaves = static_cast<int>(all_mcts[i * 2]->size());
                            return { steps, leaves };
                        }
                    ));
                }

                for (int i = 0; i < batch_games; ++i) {
                    auto [steps, leaves] = futures[i].get();
                    game_steps += steps;
                    total_leaves += leaves;
                }

                {
                    std::lock_guard<std::mutex> lock(replay_mutex);
                    for (const auto& local_buffer : local_buffers) {
                        for (const auto& exp : local_buffer) {
                            replay_buffer.push_back(exp);
                            if (replay_buffer.size() > cfg.replay_buffer_size) {
                                replay_buffer.pop_front();
                            }
                        }
                    }
                }
            }
        }
        net->train();

        size_t game_nodes = total_leaves;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        float dt = duration.count() / 1000.0f;

        float speed_steps = static_cast<float>(game_steps) / dt;
        float speed_nodes = static_cast<float>(game_nodes) / dt;

        // Print the same format as Python version + MCTS info
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Step " << step_idx
            << ", steps " << std::setw(3) << game_steps
            << ", leaves " << std::setw(4) << game_nodes
            << ", steps/s " << std::setw(5) << speed_steps
            << ", leaves/s " << std::setw(6) << speed_nodes
            << ", replay " << replay_buffer.size() << std::endl;

        step_idx++;
        auto update_optimizer_lr = [&](float new_lr) {
            for (auto& param_group : optimizer.param_groups()) {
                static_cast<torch::optim::SGDOptions&>(param_group.options()).lr(new_lr);
            }
            std::cout << "[LR Decay] Learning rate adjusted to: " << new_lr << std::endl;
            };

        if (step_idx == cfg.adjusted_idx_1) {
            cfg.learning_rate = cfg.learning_rate_adjusted_1;
            update_optimizer_lr(cfg.learning_rate);
        }
        if (step_idx == cfg.adjusted_idx_2) {
            cfg.learning_rate = cfg.learning_rate_adjusted_2;
            update_optimizer_lr(cfg.learning_rate);
        }

        // Check for termination after episodes
        if (terminate_requested) break;

        // Check if we have enough data to train
        if (replay_buffer.size() < cfg.min_replay_to_train) {
            std::cout << "Not enough samples in replay buffer. Waiting... ("
                << replay_buffer.size() << "/" << cfg.min_replay_to_train << ")" << std::endl;
            continue;
        }

        // Training phase
        float sum_loss = 0.0f;
        float sum_value_loss = 0.0f;
        float sum_policy_loss = 0.0f;

        for (int train_round = 0; train_round < cfg.train_rounds && !terminate_requested; ++train_round) {
            // Sample batch from replay buffer with prioritized sampling
            std::vector<std::tuple<GameState, Player, std::array<float, GAME_COLS>, float>> batch;
            batch.reserve(cfg.batch_size);

            if (static_cast<int>(replay_buffer.size()) < cfg.batch_size) {
                std::cerr << "Warning: Replay buffer too small for batch size" << std::endl;
                continue;
            }

            std::uniform_int_distribution<> dist(0, replay_buffer.size() - 1);

            for (int i = 0; i < cfg.batch_size; ++i) {
                size_t idx = dist(rng);
                batch.push_back(replay_buffer[idx]);
            }

            // Prepare batch data
            std::vector<GameState> batch_states;
            std::vector<Player> batch_who_moves;
            std::vector<std::array<float, GAME_COLS>> batch_probs;
            std::vector<float> batch_values;

            batch_states.reserve(cfg.batch_size);
            batch_who_moves.reserve(cfg.batch_size);
            batch_probs.reserve(cfg.batch_size);
            batch_values.reserve(cfg.batch_size);

            for (const auto& [state, player, probs, value] : batch) {
                batch_states.push_back(state);
                batch_who_moves.push_back(player);
                batch_probs.push_back(probs);
                batch_values.push_back(value);
            }
            if (train_round == 0) {

                float avg_entropy = 0;
                int batch_count = 0;

                for (const auto& entry : batch) {
                    const auto& probs = std::get<2>(entry);  // Policy targets from MCTS
                    float entropy = 0;
                    for (float p : probs) {
                        if (p > 1e-6) {
                            entropy -= p * std::log(p + 1e-10);
                        }
                    }
                    avg_entropy += entropy;
                    batch_count++;
                }

                avg_entropy /= batch_count;
                std::cout << "Avg policy target entropy: " << avg_entropy << std::endl;
            }
            // Convert to tensors
            auto states_v = torch::empty({ cfg.batch_size, 3, GAME_ROWS, GAME_COLS }, torch::kFloat32);
            state_lists_to_batches(states_v, batch_states, batch_who_moves);
            // Convert probs and values to tensors
            torch::Tensor probs_v = torch::zeros({ static_cast<int64_t>(cfg.batch_size), GAME_COLS }, torch::kFloat32);
            torch::Tensor values_v = torch::zeros({ static_cast<int64_t>(cfg.batch_size) }, torch::kFloat32);

            for (int i = 0; i < cfg.batch_size; ++i) {
                for (int j = 0; j < GAME_COLS; ++j) {
                    probs_v[i][j] = batch_probs[i][j];
                }
                values_v[i] = batch_values[i];
            }

            states_v = states_v.to(device);
            probs_v = probs_v.to(device);
            values_v = values_v.to(device);

            // Forward pass
            optimizer.zero_grad();
            auto output = net->forward(states_v);
            auto out_logits_v = std::get<0>(output);
            auto out_values_v = std::get<1>(output);

            // Calculate losses
            auto loss_value_v = torch::mse_loss(out_values_v.squeeze(-1), values_v);

            auto log_probs = torch::log_softmax(out_logits_v, 1);
            auto loss_policy_v = -log_probs * probs_v;
            loss_policy_v = loss_policy_v.sum(1).mean();


            auto loss_v = loss_policy_v + loss_value_v;
            loss_v.backward();

            torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
            optimizer.step();

            sum_loss += loss_v.item<float>();
            sum_value_loss += loss_value_v.item<float>();
            sum_policy_loss += loss_policy_v.item<float>();
        }

        tracker.track("loss_total", sum_loss / cfg.train_rounds, step_idx);
        tracker.track("loss_value", sum_value_loss / cfg.train_rounds, step_idx);
        tracker.track("loss_policy", sum_policy_loss / cfg.train_rounds, step_idx);

        // Check for termination after training phase
        if (terminate_requested) break;

        // Check for termination after evaluation
        if (terminate_requested) break;

        Connect4Net current_net = net;

        std::ostringstream filename;
        filename << "checkpoint_" << std::setw(5) << std::setfill('0') << step_idx
            << ".pt";
        fs::path save_path = saves_path / filename.str();
        save_model(net, save_path.string());
        std::cout << "Saved periodic checkpoint to: " << save_path.string() << std::endl;
    }

    // Handle graceful shutdown
    if (terminate_requested) {
        save_and_exit(net, saves_path, step_idx);
        return 0;
    }

    return 0;
}