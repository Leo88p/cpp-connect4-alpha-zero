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
constexpr int PLAY_EPISODES = 1;
constexpr int INITIAL_MCTS_SEARCHES = 8;  // Starting MCTS searches
constexpr int MCTS_BATCH_SIZE = 64;
constexpr size_t REPLAY_BUFFER_SIZE = 200000; // Critical fix: much larger buffer
constexpr float INITIAL_LEARNING_RATE = 0.2f; // AlphaZero standard
constexpr float MOMENTUM = 0.9f;
constexpr float WEIGHT_DECAY = 1e-4f; // L2 regularization
constexpr int BATCH_SIZE = 256;
constexpr int TRAIN_ROUNDS = 10;
constexpr size_t MIN_REPLAY_TO_TRAIN = 2000;
constexpr float CONSERVATIVE_WIN_RATIO = 0.55f; // More conservative threshold
constexpr int EVALUATE_EVERY_STEP = 200;
constexpr int EVALUATION_ROUNDS = 100;
constexpr int STEPS_BEFORE_TAU_0 = 10;
constexpr int ROLLING_EVAL_WINDOW = 1; // Increased for more stable evaluation
constexpr float MCTS_INCREASE_THRESHOLD = 0.8f; // Win ratio threshold to increase MCTS
constexpr bool CLEAR_BUFFER_ON_MCTS_INCREASE = true; // Whether to clear replay buffer when MCTS increases
constexpr float WARM_START_RATIO = 0.0f; // Keep 20% of recent data when clearing buffer

// MCTS progression schedule - start low, increase gradually
const std::vector<int> MCTS_SCHEDULE = { 8, 16, 32, 64, 128, 256, 512 };

// Global flag for termination handling
std::atomic<bool> terminate_requested(false);
std::atomic<bool> saving_checkpoint(false);

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

class TargetNet {
public:
    explicit TargetNet(Connect4Net& net) : model_(std::make_shared<Connect4NetImpl>()) {
        sync(net);
    }

    void sync(Connect4Net& net) {
        torch::Device device = net->parameters().empty() ? torch::kCPU : net->parameters().front().device();
        model_->to(device);
        net->to(device);

        auto target_params = model_->named_parameters();
        auto source_params = net->named_parameters();

        {
            torch::NoGradGuard no_grad;
            for (const auto& param_pair : source_params) {
                const std::string& name = param_pair.key();
                const torch::Tensor& source_param = param_pair.value();

                if (target_params.contains(name)) {
                    torch::Tensor target_param = target_params[name];

                    // Ensure shapes match
                    if (target_param.sizes() != source_param.sizes()) {
                        std::cerr << "Warning: Parameter shapes don't match for '" << name << "'" << std::endl;
                        continue;
                    }

                    target_param.copy_(source_param.detach());
                }
            }
        }
    }

    Connect4Net get_model() const {
        return model_;
    }

private:
    Connect4Net model_;
};

float evaluate(Connect4Net& net1, Connect4Net& net2, int rounds, const torch::Device& device, int mcts_searches) {
    int n1_win = 0, n2_win = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int r_idx = 0; r_idx < rounds; ++r_idx) {
        std::vector<MCTS> mcts_stores;
        mcts_stores.push_back(MCTS());  // Player 1's MCTS store
        mcts_stores.push_back(MCTS());  // Player 2's MCTS store
        auto [result, _] = play_game(
            mcts_stores, nullptr, net1, net2,
            0, mcts_searches, MCTS_BATCH_SIZE, std::nullopt, device, REPLAY_BUFFER_SIZE
        );

        if (result < -0.5f) {
            n2_win++;
        }
        else if (result > 0.5f) {
            n1_win++;
        }
    }

    return static_cast<float>(n1_win) / (n1_win + n2_win + 1e-8f);
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
void save_and_exit(Connect4Net& net, const fs::path& saves_path, int step_idx, int best_idx, int current_mcts_idx) {
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
        progress_file << "best_idx=" << best_idx << std::endl;
        progress_file << "current_mcts_idx=" << current_mcts_idx << std::endl;
        progress_file << "current_mcts_searches=" << MCTS_SCHEDULE[current_mcts_idx] << std::endl;
        progress_file << "timestamp=" << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
        progress_file.close();
        std::cout << "Training progress saved to: " << progress_path.string() << std::endl;
    }

    std::cout << "Graceful shutdown complete. Exiting..." << std::endl;
}

// Learning rate schedule - match AlphaZero paper
float get_learning_rate(int step_idx) {
    if (step_idx < 400000) {
        return INITIAL_LEARNING_RATE;
    }
    else if (step_idx < 600000) {
        return INITIAL_LEARNING_RATE * 0.1f; // 10x decay
    }
    else {
        return INITIAL_LEARNING_RATE * 0.01f; // Another 10x decay
    }
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
        else if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        }
    }

    if (run_name.empty()) {
        std::cerr << "Error: --name argument is required" << std::endl;
        print_help();
        return 1;
    }

    std::cout << "Starting training with name: " << run_name << std::endl;
    std::cout << "Device: " << device_str << std::endl;
    std::cout << "Dynamic MCTS schedule: ";
    for (size_t i = 0; i < MCTS_SCHEDULE.size(); ++i) {
        std::cout << MCTS_SCHEDULE[i];
        if (i < MCTS_SCHEDULE.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    std::cout << "MCTS increase threshold: " << MCTS_INCREASE_THRESHOLD
        << ", Clear buffer on increase: " << (CLEAR_BUFFER_ON_MCTS_INCREASE ? "true" : "false")
        << ", Warm start ratio: " << WARM_START_RATIO << std::endl;

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
    Connect4Net net = std::make_shared<Connect4NetImpl>();
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

    // Initialize best_net and etalon_net
    TargetNet best_net(net);
    TargetNet etalon_net(net);  // Etalon model starts as current model

    // Print network architecture
    std::cout << *net << std::endl;

    // CORRECT AlphaZero optimizer: SGD with momentum + weight decay
    torch::optim::SGDOptions sgd_opts(INITIAL_LEARNING_RATE);
    sgd_opts.momentum(MOMENTUM).weight_decay(WEIGHT_DECAY);
    torch::optim::SGD optimizer(net->parameters(), sgd_opts);

    // Replay buffer
    ReplayBuffer replay_buffer;

    // Training state
    int step_idx = 0;
    int best_idx = 0;
    int etalon_idx = 0;  // Index for etalon model
    int current_mcts_idx = 0;  // Current index in MCTS_SCHEDULE
    int current_mcts_searches = MCTS_SCHEDULE[current_mcts_idx];

    std::cout << "Starting with MCTS searches: " << current_mcts_searches << std::endl;

    std::deque<float> rolling_win_ratios;
    std::deque<float> rolling_etalon_ratios;  // For tracking win rate against etalon

    if (!net_to_load.empty()) {
        try {
            // Parse step_idx, best_idx, and potentially mcts state from filename
            std::string filename = net_to_load;
            size_t underscore1 = filename.find('_');
            size_t underscore2 = filename.find('_', underscore1 + 1);
            size_t dot = filename.find('.');

            if (underscore1 != std::string::npos && underscore2 != std::string::npos && dot != std::string::npos) {
                best_idx = std::stoi(filename.substr(underscore1 + 1, underscore2 - underscore1 - 1));
                step_idx = std::stoi(filename.substr(underscore2 + 1, dot - underscore2 - 1));
                std::cout << "Resuming from step " << step_idx << ", best index " << best_idx << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing filename: " << e.what() << std::endl;
        }
    }

    SimpleTracker tracker(csv_logger, 10);
    std::mt19937 rng(std::random_device{}());

    // Training loop with termination handling
    while (!terminate_requested) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int game_steps = 0;
        int total_leaves = 0; // Track leaves properly

        // Check for termination before starting episodes
        if (terminate_requested) break;

        // Play episodes using current MCTS searches
        for (int episode_idx = 0; episode_idx < PLAY_EPISODES && !terminate_requested; ++episode_idx) {
            std::vector<MCTS> mcts_stores;
            mcts_stores.push_back(MCTS(1.0f));  // Player 0
            mcts_stores.push_back(MCTS(1.0f));  // Player 1

            // Use best_net for self-play generation
            auto best_model = best_net.get_model();

            auto [game_result, steps] = play_game(
                mcts_stores, &replay_buffer,
                best_model, best_model,
                STEPS_BEFORE_TAU_0, current_mcts_searches, MCTS_BATCH_SIZE,
                std::nullopt, device, REPLAY_BUFFER_SIZE
            );

            // Calculate leaves created DURING THIS GAME
            size_t new_nodes = mcts_stores[0].size();
            total_leaves += static_cast<int>(new_nodes);
            game_steps += steps;
        }

        size_t game_nodes = total_leaves;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        float dt = duration.count() / 1000.0f;

        float speed_steps = static_cast<float>(game_steps) / dt;
        float speed_nodes = static_cast<float>(game_nodes) / dt;

        tracker.track("speed_steps", speed_steps, step_idx);
        tracker.track("speed_nodes", speed_nodes, step_idx);
        tracker.track("mcts_searches", static_cast<float>(current_mcts_searches), step_idx);

        // Print the same format as Python version + MCTS info
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Step " << step_idx
            << ", steps " << std::setw(3) << game_steps
            << ", leaves " << std::setw(4) << game_nodes
            << ", steps/s " << std::setw(5) << speed_steps
            << ", leaves/s " << std::setw(6) << speed_nodes
            << ", best_idx " << best_idx
            << ", etalon_idx " << etalon_idx
            << ", MCTS " << current_mcts_searches
            << ", replay " << replay_buffer.size() << std::endl;

        step_idx++;

        // Check for termination after episodes
        if (terminate_requested) break;

        // Check if we have enough data to train
        if (replay_buffer.size() < MIN_REPLAY_TO_TRAIN) {
            std::cout << "Not enough samples in replay buffer. Waiting... ("
                << replay_buffer.size() << "/" << MIN_REPLAY_TO_TRAIN << ")" << std::endl;
            continue;
        }

        // Update learning rate according to AlphaZero schedule
        float current_lr = get_learning_rate(step_idx);
        for (auto& group : optimizer.param_groups()) {
            auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
            options.lr(current_lr);
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

            // Prioritized sampling based on data age (recent data has higher priority)
            size_t buffer_size = std::min(replay_buffer.size(), static_cast<size_t>(200000));
            std::vector<float> priorities(buffer_size);
            float total_priority = 0.0f;

            // Calculate priorities - exponential decay based on age
            for (size_t i = 0; i < buffer_size; ++i) {
                float age = static_cast<float>(buffer_size - 1 - i);
                float priority = std::exp(-age / 50000.0f); // Adjust decay constant based on buffer size
                priorities[i] = priority;
                total_priority += priority;
            }

            // Normalize priorities to create discrete distribution
            if (total_priority > 0.0f) {
                for (float& p : priorities) {
                    p /= total_priority;
                }
            }

            std::discrete_distribution<size_t> dist(priorities.begin(), priorities.end());

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

            // Convert to tensors
            auto states_v = state_lists_to_batch(batch_states, batch_who_moves, device);

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

        // Evaluation phase - with rolling average
        if (step_idx % EVALUATE_EVERY_STEP == 0) {
            std::cout << "\n=== Evaluation Phase ===" << std::endl;
            std::cout << "Current MCTS searches: " << current_mcts_searches
                << " (level " << current_mcts_idx + 1 << "/" << MCTS_SCHEDULE.size() << ")" << std::endl;

            // First evaluate against best model (for best_net updates)
            std::cout << "Evaluating against best model (roll " << rolling_win_ratios.size() + 1
                << "/" << ROLLING_EVAL_WINDOW << ")..." << std::endl;

            // Get models first to create lvalues
            Connect4Net current_net = net;
            Connect4Net best_net_model = best_net.get_model();

            float win_ratio = evaluate(current_net, best_net_model, EVALUATION_ROUNDS, device, current_mcts_searches);
            rolling_win_ratios.push_back(win_ratio);

            if (rolling_win_ratios.size() > ROLLING_EVAL_WINDOW) {
                rolling_win_ratios.pop_front();
            }

            std::cout << "Win ratio vs best model: " << std::fixed << std::setprecision(3)
                << win_ratio << std::endl;

            // Log evaluation result
            if (csv_logger.is_open()) {
                csv_logger.log(step_idx, "eval_win_ratio_vs_best", win_ratio);
            }

            // Update best_net if we have enough evaluations AND average is good
            if (rolling_win_ratios.size() >= ROLLING_EVAL_WINDOW) {
                float avg_win_ratio = std::accumulate(rolling_win_ratios.begin(),
                    rolling_win_ratios.end(), 0.0f) /
                    rolling_win_ratios.size();

                std::cout << "Rolling average vs best: " << std::fixed << std::setprecision(3)
                    << avg_win_ratio << std::endl;

                if (avg_win_ratio > CONSERVATIVE_WIN_RATIO) {
                    std::cout << "Net is better than current best (avg=" << avg_win_ratio
                        << " > " << CONSERVATIVE_WIN_RATIO << "), updating best model..." << std::endl;
                    best_net.sync(net);
                    best_idx++;

                    // Clear rolling window after successful update
                    rolling_win_ratios.clear();

                    // Save best model
                    std::ostringstream filename;
                    filename << "best_" << std::setw(3) << std::setfill('0') << best_idx
                        << "_" << std::setw(5) << std::setfill('0') << step_idx << ".pt";

                    fs::path save_path = saves_path / filename.str();
                    save_model(net, save_path.string());
                    std::cout << "Saved best model to: " << save_path.string() << std::endl;
                }
            }

            // Now evaluate against etalon model (for MCTS progression)
            std::cout << "\nEvaluating against etalon model (MCTS=" << current_mcts_searches << ")..." << std::endl;

            Connect4Net etalon_net_model = etalon_net.get_model();
            float etalon_win_ratio = evaluate(current_net, etalon_net_model, EVALUATION_ROUNDS, device, current_mcts_searches);
            rolling_etalon_ratios.push_back(etalon_win_ratio);

            if (rolling_etalon_ratios.size() > ROLLING_EVAL_WINDOW) {
                rolling_etalon_ratios.pop_front();
            }

            std::cout << "Win ratio vs etalon: " << std::fixed << std::setprecision(3)
                << etalon_win_ratio << std::endl;

            if (csv_logger.is_open()) {
                csv_logger.log(step_idx, "eval_win_ratio_vs_etalon", etalon_win_ratio);
                csv_logger.log(step_idx, "current_mcts_searches", static_cast<float>(current_mcts_searches));
            }

            // Check if we should increase MCTS
            if (rolling_etalon_ratios.size() >= ROLLING_EVAL_WINDOW) {
                float avg_etalon_ratio = std::accumulate(rolling_etalon_ratios.begin(),
                    rolling_etalon_ratios.end(), 0.0f) /
                    rolling_etalon_ratios.size();

                std::cout << "Rolling average vs etalon: " << std::fixed << std::setprecision(3)
                    << avg_etalon_ratio << std::endl;

                // Check if we should increase MCTS
                if (avg_etalon_ratio > MCTS_INCREASE_THRESHOLD &&
                    current_mcts_idx < static_cast<int>(MCTS_SCHEDULE.size()) - 1) {

                    std::cout << "\MCTS PROGRESSION TRIGGERED!" << std::endl;
                    std::cout << "Average win ratio vs etalon (" << avg_etalon_ratio
                        << ") > threshold (" << MCTS_INCREASE_THRESHOLD << ")" << std::endl;

                    // Update etalon model to current best model
                    std::cout << "Updating etalon model..." << std::endl;
                    auto best_model = best_net.get_model();
                    etalon_net.sync(best_model);
                    etalon_idx = best_idx;

                    // Clear rolling window
                    rolling_etalon_ratios.clear();

                    // Save etalon model
                    std::ostringstream etalon_filename;
                    etalon_filename << "etalon_" << std::setw(3) << std::setfill('0') << etalon_idx
                        << "_" << std::setw(5) << std::setfill('0') << step_idx
                        << "_mcts" << current_mcts_searches << ".pt";
                    fs::path etalon_save_path = saves_path / etalon_filename.str();
                    save_model(best_model, etalon_save_path.string());
                    std::cout << "Saved etalon model to: " << etalon_save_path.string() << std::endl;

                    // Clear replay buffer if configured
                    if (CLEAR_BUFFER_ON_MCTS_INCREASE) {
                        std::cout << "Clearing replay buffer for MCTS progression..." << std::endl;
                        clear_replay_buffer(replay_buffer, WARM_START_RATIO);
                    }
                    else {
                        std::cout << "Keeping replay buffer (CLEAR_BUFFER_ON_MCTS_INCREASE=false)" << std::endl;
                    }

                    // Increase MCTS
                    current_mcts_idx++;
                    current_mcts_searches = MCTS_SCHEDULE[current_mcts_idx];
                    std::cout << "Increased MCTS searches to: " << current_mcts_searches
                        << " (level " << current_mcts_idx + 1 << "/" << MCTS_SCHEDULE.size() << ")" << std::endl;

                    // Log the MCTS increase
                    if (csv_logger.is_open()) {
                        csv_logger.log(step_idx, "mcts_increased", static_cast<float>(current_mcts_searches));
                    }
                }
            }
        }

        // Check for termination after evaluation
        if (terminate_requested) break;

        // Save checkpoint every 1000 steps
        if (step_idx % 1000 == 0) {
            std::ostringstream filename;
            filename << "checkpoint_" << std::setw(5) << std::setfill('0') << step_idx
                << "_mcts" << current_mcts_searches << ".pt";
            fs::path save_path = saves_path / filename.str();
            save_model(net, save_path.string());
            std::cout << "Saved checkpoint to: " << save_path.string() << std::endl;
        }
    }

    // Handle graceful shutdown
    if (terminate_requested) {
        save_and_exit(net, saves_path, step_idx, best_idx, current_mcts_idx);
        return 0;
    }

    return 0;
}