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

namespace fs = std::filesystem;
using namespace Connect4;

// Hyperparameters - fixed for Connect4 AlphaZero
int PLAY_EPISODES = 1024;
int PARALLEL_GAMES = 256;
int MCTS_SEARCHES = 10;
int MCTS_BATCH_SIZE = 32;
constexpr size_t REPLAY_BUFFER_SIZE = 1000000;
float LEARNING_RATE = 0.2f;
constexpr float LEARNING_RATE_ADJUSTED_1 = 0.02f;
constexpr float LEARNING_RATE_ADJUSTED_2 = 0.002f;
constexpr int ADJUSTED_IDX_1 = 20;
constexpr int ADJUSTED_IDX_2 = 100;
constexpr int BATCH_SIZE = 2048;
constexpr int TRAIN_ROUNDS = 30;
constexpr size_t MIN_REPLAY_TO_TRAIN = 10;
constexpr int STEPS_BEFORE_TAU_0 = 10;
constexpr int NUM_BLOCKS = 5;
constexpr int NUM_FILTERS = 64;
//constexpr int EVALUATION_ROUNDS = 500;
//constexpr float WIN_RATIO = 0.52f;

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

void print_help() {
    std::cout << "Usage: connect4_az -n <run_name> [--dev <device>] [--net <filename>]" << std::endl;
    std::cout << "  -n, --name    Name of the run (required)" << std::endl;
    std::cout << "  --dev         Device to use (cpu or cuda, default: cpu)" << std::endl;
    std::cout << "  --net         Network file to load (optional)" << std::endl;
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

    std::mt19937_64 rng(42);  // Fixed seed for reproducibility, or use random_device
    //Connect4::ZobristHash::init(rng);
    // In main(), create ONCE:
    torch::set_num_threads(1);        // Limit intra-op parallelism per forward
    torch::set_num_interop_threads(1); // Limit inter-op parallelism

    // Setup signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);  // Ctrl+C
    std::signal(SIGTERM, signal_handler); // Termination request

    // Parse command line arguments
    std::string run_name;
    std::string device_str = "cpu";
    std::string net_to_load;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--name") {
            if (i + 1 < argc) run_name = argv[++i];
        }
        else if (arg == "--dev") {
            if (i + 1 < argc) device_str = argv[++i];
        }
        else if (arg == "--net") {
            if (i + 1 < argc) net_to_load = argv[++i];
        }
        else if (arg == "--mcts") {
            if (i + 1 < argc) MCTS_SEARCHES = std::stoi(argv[++i]);
        }
        else if (arg == "--batch") {
            if (i + 1 < argc) MCTS_BATCH_SIZE = std::stoi(argv[++i]);
        }
        else if (arg == "--games") {
            if (i + 1 < argc) PLAY_EPISODES = std::stoi(argv[++i]);
        }
        else if (arg == "--parallel") {
            if (i + 1 < argc) PARALLEL_GAMES = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        }
    }
    ThreadPool pool(PARALLEL_GAMES);  // Match your 12 cores

    if (run_name.empty()) {
        std::cerr << "Error: --name argument is required" << std::endl;
        print_help();
        return 1;
    }

    std::cout << "Starting training with name: " << run_name << std::endl;
    std::cout << "Device: " << device_str << std::endl;

    if (!net_to_load.empty()) {
        std::cout << "Loading network from: " << net_to_load << std::endl;
    }

    torch::Device device(device_str);
    fs::path saves_path = fs::path("saves") / run_name;
    fs::create_directories(saves_path);

    // Setup CSV logging
    fs::path logs_path = saves_path / "logs";
    fs::create_directories(logs_path);
    CSVLogger csv_logger((logs_path / "metrics.csv").string());

    // Initialize network
    Connect4Net net = std::make_shared<Connect4NetImpl>(NUM_BLOCKS);
    net->to(device);  // Ensure network is on the correct device from the start

    if (!net_to_load.empty()) {
        fs::path net_path = saves_path / net_to_load;
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

    torch::optim::SGDOptions sgd_opts = torch::optim::SGDOptions(LEARNING_RATE).momentum(0.9).weight_decay(1e-4);
    torch::optim::SGD optimizer(net->parameters(), sgd_opts);

    // Replay buffer
    ReplayBuffer replay_buffer;

    int step_idx = 0;

    std::deque<float> loss_history;
    int steps_since_last_improvement = 0;
    float best_loss = std::numeric_limits<float>::max();

    if (!net_to_load.empty()) {
        try {
            // Parse step_idx and potentially mcts state from filename
            std::string filename = net_to_load;
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
    //std::mt19937 rng(std::random_device{}());

    // Training loop with termination handling
    while (!terminate_requested) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int game_steps = 0;
        int total_leaves = 0; // Track leaves properly

        // Check for termination before starting episodes
        if (terminate_requested) break;

        auto neural_worker = std::make_unique<Connect4::NeuralWorker>(net, device, PARALLEL_GAMES);
        net->eval(); 
        {
            torch::NoGradGuard no_grad;
            std::vector<std::vector<MCTS>> all_mcts(PARALLEL_GAMES,
                std::vector<MCTS>(2, MCTS(1.0f)));

            for (auto& game_mcts : all_mcts) {
                for (auto& mcts : game_mcts) {
                    mcts.set_neural_worker(neural_worker.get());
                }
            }

            // Мьютекс для защиты общего replay_buffer
            std::mutex replay_mutex;

            for (int episode_start = 0; episode_start < PLAY_EPISODES && !terminate_requested; episode_start += PARALLEL_GAMES) {
                int batch_games = std::min(PARALLEL_GAMES, PLAY_EPISODES - episode_start);

                // Вектор фьючерсов для асинхронных игр
                std::vector<std::future<std::pair<int, int>>> futures;
                // Вектор локальных буферов (каждый для своей игры)
                std::vector<std::vector<ReplayBuffer::value_type>> local_buffers(batch_games);

                for (int i = 0; i < batch_games; ++i) {
                    futures.push_back(pool.enqueue(
                        [&, i, episode_idx = episode_start + i]() -> std::pair<int, int> {

                            // Запускаем игру, передавая локальный буфер
                            auto [game_result, steps] = play_game(
                                all_mcts[i], &local_buffers[i],
                                net, net,
                                STEPS_BEFORE_TAU_0, MCTS_SEARCHES, MCTS_BATCH_SIZE,
                                std::nullopt, device
                            );

                            int leaves = static_cast<int>(all_mcts[i][0].size());
                            return { steps, leaves };
                        }
                    ));
                }

                // Собираем результаты
                for (int i = 0; i < batch_games; ++i) {
                    auto [steps, leaves] = futures[i].get();
                    game_steps += steps;
                    total_leaves += leaves;
                }

                // Сливаем локальные буферы в общий (под мьютексом)
                {
                    std::lock_guard<std::mutex> lock(replay_mutex);
                    for (const auto& local_buffer : local_buffers) {
                        for (const auto& exp : local_buffer) {
                            replay_buffer.push_back(exp);
                            if (replay_buffer.size() > REPLAY_BUFFER_SIZE) {
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
        if (step_idx == ADJUSTED_IDX_1) {
            LEARNING_RATE = LEARNING_RATE_ADJUSTED_1;
        }
        if (step_idx == ADJUSTED_IDX_2) {
            LEARNING_RATE = LEARNING_RATE_ADJUSTED_2;
        }

        // Check for termination after episodes
        if (terminate_requested) break;

        // Check if we have enough data to train
        if (replay_buffer.size() < MIN_REPLAY_TO_TRAIN) {
            std::cout << "Not enough samples in replay buffer. Waiting... ("
                << replay_buffer.size() << "/" << MIN_REPLAY_TO_TRAIN << ")" << std::endl;
            continue;
        }

        // Training phase
        float sum_loss = 0.0f;
        float sum_value_loss = 0.0f;
        float sum_policy_loss = 0.0f;

        for (int train_round = 0; train_round < TRAIN_ROUNDS && !terminate_requested; ++train_round) {
            // Sample batch from replay buffer with prioritized sampling
            std::vector<std::tuple<GameState, Player, std::array<float, GAME_COLS>, float>> batch;
            batch.reserve(BATCH_SIZE);

            if (static_cast<int>(replay_buffer.size()) < BATCH_SIZE) {
                std::cerr << "Warning: Replay buffer too small for batch size" << std::endl;
                continue;
            }

            std::uniform_int_distribution<> dist(0, replay_buffer.size() - 1);

            for (int i = 0; i < BATCH_SIZE; ++i) {
                size_t idx = dist(rng);
                batch.push_back(replay_buffer[idx]);
            }

            // Prepare batch data
            std::vector<GameState> batch_states;
            std::vector<Player> batch_who_moves;
            std::vector<std::array<float, GAME_COLS>> batch_probs;
            std::vector<float> batch_values;

            batch_states.reserve(BATCH_SIZE);
            batch_who_moves.reserve(BATCH_SIZE);
            batch_probs.reserve(BATCH_SIZE);
            batch_values.reserve(BATCH_SIZE);

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
            auto states_v = state_lists_to_batch(batch_states, batch_who_moves, device);
            //Test 1
            /*std::cout << "Batch tensor stats: "
                << "mean=" << states_v.mean().item<float>() << ", "
                << "std=" << states_v.std().item<float>() << ", "
                << "non-zero=" << (states_v != 0).sum().item<int64_t>() << std::endl;*/

            // Convert probs and values to tensors
            torch::Tensor probs_v = torch::zeros({ static_cast<int64_t>(BATCH_SIZE), GAME_COLS }, torch::kFloat32);
            torch::Tensor values_v = torch::zeros({ static_cast<int64_t>(BATCH_SIZE) }, torch::kFloat32);

            for (int i = 0; i < BATCH_SIZE; ++i) {
                for (int j = 0; j < GAME_COLS; ++j) {
                    probs_v[i][j] = batch_probs[i][j];
                }
                values_v[i] = batch_values[i];
            }

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
            // Backward pass
            loss_v.backward();
            //Test 2
            /*float total_norm = 0.0f;
            for (const auto& p : net->parameters()) {
                if (p.grad().defined()) {
                    total_norm += p.grad().norm().item<float>();
                }
            }
            std::cout << "Gradient norm: " << total_norm << std::endl; */

            torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
            optimizer.step();

            sum_loss += loss_v.item<float>();
            sum_value_loss += loss_value_v.item<float>();
            sum_policy_loss += loss_policy_v.item<float>();
        }

        tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx);
        tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx);
        tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx);

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