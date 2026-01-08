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

// Hyperparameters (same as Python version)
constexpr int PLAY_EPISODES = 1;
constexpr int MCTS_SEARCHES = 40;
constexpr int MCTS_BATCH_SIZE = 32;
constexpr size_t REPLAY_BUFFER_SIZE = 5000;
constexpr float LEARNING_RATE = 0.001f;
constexpr int BATCH_SIZE = 256;
constexpr int TRAIN_ROUNDS = 10;
constexpr size_t MIN_REPLAY_TO_TRAIN = 2000;
constexpr float BEST_NET_WIN_RATIO = 0.60f;
constexpr int EVALUATE_EVERY_STEP = 100;
constexpr int EVALUATION_ROUNDS = 20;
constexpr int STEPS_BEFORE_TAU_0 = 10;

class TargetNet {
public:
    explicit TargetNet(Connect4Net& net) {
        sync(net);
    }

    void sync(Connect4Net& net) {
        // Create a new model with the same architecture
        model_ = std::make_shared<Connect4NetImpl>();

        // Ensure both models are on the same device
        torch::Device device = net->parameters().empty() ? torch::kCPU : net->parameters().front().device();
        model_->to(device);
        net->to(device);

        // Get parameters from both models
        auto source_params = net->named_parameters();
        auto target_params = model_->named_parameters();

        {
            // Disable gradient tracking during parameter copying
            torch::NoGradGuard no_grad;

            // Copy parameters with proper error handling
            for (const auto& param_pair : source_params) {
                const std::string& name = param_pair.key();
                const torch::Tensor& source_param = param_pair.value();

                // Check if target has this parameter using contains() method
                if (target_params.contains(name)) {
                    torch::Tensor target_param = target_params[name];

                    // Ensure shapes match before copying
                    if (target_param.sizes() != source_param.sizes()) {
                        std::cerr << "Warning: Parameter shapes don't match for '" << name << "'" << std::endl;
                        std::cerr << "Source shape: " << source_param.sizes() << ", Target shape: " << target_param.sizes() << std::endl;
                        continue;
                    }

                    // Create a detached copy of the source parameter to avoid gradient issues
                    torch::Tensor source_detached = source_param.detach();

                    // Ensure same device and dtype
                    if (target_param.device() != source_detached.device()) {
                        target_param = target_param.to(source_detached.device());
                    }

                    if (target_param.dtype() != source_detached.dtype()) {
                        target_param = target_param.to(source_detached.dtype());
                    }

                    try {
                        target_param.copy_(source_detached);
                    }
                    catch (const c10::Error& e) {
                        std::cerr << "Error copying parameter '" << name << "': " << e.what() << std::endl;
                        throw;
                    }
                }
                else {
                    std::cerr << "Warning: Parameter '" << name << "' exists in source but not in target model" << std::endl;
                }
            }
        }

        // Verify the copy was successful
        verify_copy(net);
    }

    Connect4Net get_model() const {
        return model_;
    }

private:
    Connect4Net model_;

    void verify_copy(Connect4Net& net) {
        auto source_params = net->named_parameters();
        auto target_params = model_->named_parameters();

        for (const auto& param_pair : source_params) {
            const std::string& name = param_pair.key();
            const torch::Tensor& source_param = param_pair.value();

            if (target_params.contains(name)) {
                torch::Tensor target_param = target_params[name];

                // Check if the copy was successful (allow small numerical differences)
                torch::Tensor diff = (target_param - source_param).abs().max();
                if (diff.item<float>() > 1e-6) {
                    std::cerr << "Warning: Parameter copy verification failed for '" << name
                        << "', max difference: " << diff.item<float>() << std::endl;
                }
            }
        }
    }
};

float evaluate(Connect4Net& net1, Connect4Net& net2, int rounds, const torch::Device& device) {
    int n1_win = 0, n2_win = 0;
    std::vector<MCTS> mcts_stores;
    mcts_stores.push_back(MCTS());  // Player 1's MCTS store
    mcts_stores.push_back(MCTS());  // Player 2's MCTS store

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int r_idx = 0; r_idx < rounds; ++r_idx) {
        auto [result, _] = play_game(
            mcts_stores, nullptr, net1, net2,
            0, MCTS_SEARCHES, MCTS_BATCH_SIZE, std::nullopt, device, REPLAY_BUFFER_SIZE
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
        torch::save(net, path);
        std::cout << "Model saved to: " << path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }
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

int main(int argc, char** argv) {
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

    TargetNet best_net(net);

    // Print network architecture
    std::cout << *net << std::endl;

    // Setup optimizer
    torch::optim::SGD optimizer(
        net->parameters(),
        torch::optim::SGDOptions(LEARNING_RATE).momentum(0.9)
    );

    // Replay buffer
    ReplayBuffer replay_buffer;

    // MCTS store
    MCTS mcts_store(1.0f);

    // Training state
    int step_idx = 0;
    int best_idx = 0;

    if (!net_to_load.empty()) {
        try {
            // Parse step_idx and best_idx from filename: best_{best_idx}_{step_idx}.pt
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

    // Training loop
    while (true) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int game_steps = 0;
        int total_leaves = 0; // Track leaves properly

        // Play episodes
        for (int episode_idx = 0; episode_idx < PLAY_EPISODES; ++episode_idx) {
            // Create FRESH MCTS stores for this game
            MCTS mcts_store(1.0f); // Start with empty store for each game
            std::vector<MCTS> mcts_stores;
            mcts_stores.push_back(mcts_store);  // Player 0
            mcts_stores.push_back(mcts_store);  // Player 1

            // Track size before game
            size_t prev_nodes = mcts_store.size();
            auto best_model = best_net.get_model();

            auto [game_result, steps] = play_game(
                mcts_stores, &replay_buffer,
                best_model, best_model,
                STEPS_BEFORE_TAU_0, MCTS_SEARCHES, MCTS_BATCH_SIZE,
                std::nullopt, device, REPLAY_BUFFER_SIZE
            );

            // Calculate leaves created DURING THIS GAME
            size_t new_nodes = mcts_stores[0].size() - prev_nodes;
            total_leaves += static_cast<int>(new_nodes);
            game_steps += steps;

            // No need to update shared store - we use fresh stores each time
        }

        size_t game_nodes = total_leaves;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        float dt = duration.count() / 1000.0f;

        float speed_steps = static_cast<float>(game_steps) / dt;
        float speed_nodes = static_cast<float>(game_nodes) / dt;

        tracker.track("speed_steps", speed_steps, step_idx);
        tracker.track("speed_nodes", speed_nodes, step_idx);

        // Print the same format as Python version
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Step " << step_idx
            << ", steps " << std::setw(3) << game_steps
            << ", leaves " << std::setw(4) << game_nodes
            << ", steps/s " << std::setw(5) << speed_steps
            << ", leaves/s " << std::setw(6) << speed_nodes
            << ", best_idx " << best_idx
            << ", replay " << replay_buffer.size() << std::endl;

        step_idx++;

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

        for (int train_round = 0; train_round < TRAIN_ROUNDS; ++train_round) {
            // Sample batch from replay buffer
            std::vector<std::tuple<GameState, Player, std::array<float, GAME_COLS>, float>> batch;
            batch.reserve(BATCH_SIZE);

            if (static_cast<int>(replay_buffer.size()) < BATCH_SIZE) {
                std::cerr << "Warning: Replay buffer too small for batch size" << std::endl;
                continue;
            }

            for (int i = 0; i < BATCH_SIZE; ++i) {
                std::uniform_int_distribution<size_t> dist(0, replay_buffer.size() - 1);
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

        // Evaluation phase
        // Evaluation phase
        if (step_idx % EVALUATE_EVERY_STEP == 0) {
            std::cout << "Evaluating network..." << std::endl;

            // Get models first to create lvalues
            Connect4Net current_net = net;
            Connect4Net best_net_model = best_net.get_model();

            // Create new MCTS stores for evaluation
            std::vector<MCTS> eval_mcts_stores = { MCTS(), MCTS() };
            float win_ratio = evaluate(current_net, best_net_model, EVALUATION_ROUNDS, device);

            std::cout << "Net evaluated, win ratio = " << std::fixed << std::setprecision(2)
                << win_ratio << std::endl;

            // Log evaluation result
            if (csv_logger.is_open()) {
                csv_logger.log(step_idx, "eval_win_ratio", win_ratio);
            }

            if (win_ratio > BEST_NET_WIN_RATIO) {
                std::cout << "Net is better than cur best, sync" << std::endl;
                best_net.sync(net);
                best_idx++;

                // Save model
                std::ostringstream filename;
                filename << "best_" << std::setw(3) << std::setfill('0') << best_idx
                    << "_" << std::setw(5) << std::setfill('0') << step_idx << ".pt";

                fs::path save_path = saves_path / filename.str();
                save_model(net, save_path.string());
                std::cout << "Saved best model to: " << save_path.string() << std::endl;

                // Clear MCTS store
                mcts_store.clear();
            }
        }

        // Save checkpoint every 1000 steps
        if (step_idx % 1000 == 0) {
            std::ostringstream filename;
            filename << "checkpoint_" << std::setw(5) << std::setfill('0') << step_idx << ".pt";
            fs::path save_path = saves_path / filename.str();
            save_model(net, save_path.string());
            std::cout << "Saved checkpoint to: " << save_path.string() << std::endl;
        }
    }

    return 0;
}