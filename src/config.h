#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <locale>
#include <stdexcept>
#include <signal.h>
#include <torch/torch.h>

namespace Connect4 {
    // --- Helper Functions ---

// Trim whitespace from both ends of a string
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (std::string::npos == first) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    // Safely parse floats regardless of system locale (forces '.' as decimal separator)
    static float safe_stof(const std::string& str) {
        std::stringstream ss(str);
        ss.imbue(std::locale::classic());
        float val;
        ss >> val;
        if (ss.fail()) throw std::invalid_argument("Invalid float format");
        return val;
    }

    // --- Configuration Structure ---

    struct Config {
        // String parameters
        std::string run_name = "default_run";
        std::string device_str = "cpu";
        std::string net_to_load = "";

        // Integer parameters
        int play_episodes = 1024;
        int parallel_games = 256;
        int mcts_batches = 10;
        int mcts_batch_size = 32;
        int adjusted_idx_1 = 20;
        int adjusted_idx_2 = 100;
        int batch_size = 2048;
        int train_rounds = 30;
        int steps_before_tau_0 = 10;
        int num_blocks = 5;
        int num_filters = 64;

        // Size parameters
        size_t replay_buffer_size = 1000000;
        size_t min_replay_to_train = 10;

        // Float parameters
        float learning_rate = 0.2f;
        float learning_rate_adjusted_1 = 0.02f;
        float learning_rate_adjusted_2 = 0.002f;
        float virtual_loss = 2.0f;
        float c_puct = 1.0f;
        float c_fpu = 0.25f;

        void loadFromFile(const std::string& filename);
    };

    void Config::loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Config file '" << filename << "' not found. Using default values.\n";
            return;
        }

        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            line_num++;
            line = trim(line);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            size_t delimiterPos = line.find('=');
            if (delimiterPos == std::string::npos) {
                std::cerr << "Warning: Invalid syntax at line " << line_num << ": " << line << "\n";
                continue;
            }

            std::string key = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));

            try {
                if (key == "run_name") run_name = value;
                else if (key == "device_str") device_str = value;
                else if (key == "net_to_load") net_to_load = value;

                else if (key == "play_episodes") play_episodes = std::stoi(value);
                else if (key == "parallel_games") parallel_games = std::stoi(value);
                else if (key == "mcts_batches") mcts_batches = std::stoi(value);
                else if (key == "mcts_batch_size") mcts_batch_size = std::stoi(value);
                else if (key == "adjusted_idx_1") adjusted_idx_1 = std::stoi(value);
                else if (key == "adjusted_idx_2") adjusted_idx_2 = std::stoi(value);
                else if (key == "batch_size") batch_size = std::stoi(value);
                else if (key == "train_rounds") train_rounds = std::stoi(value);
                else if (key == "steps_before_tau_0") steps_before_tau_0 = std::stoi(value);
                else if (key == "num_blocks") num_blocks = std::stoi(value);
                else if (key == "num_filters") num_filters = std::stoi(value);

                else if (key == "replay_buffer_size") replay_buffer_size = std::stoull(value);
                else if (key == "min_replay_to_train") min_replay_to_train = std::stoull(value);

                else if (key == "learning_rate") learning_rate = safe_stof(value);
                else if (key == "learning_rate_adjusted_1") learning_rate_adjusted_1 = safe_stof(value);
                else if (key == "learning_rate_adjusted_2") learning_rate_adjusted_2 = safe_stof(value);
                else if (key == "virtual_loss") virtual_loss = safe_stof(value);
                else if (key == "c_puct") c_puct = safe_stof(value);
                else if (key == "c_fpu") c_fpu = safe_stof(value);

                else {
                    std::cerr << "Warning: Unknown key '" << key << "' at line " << line_num << "\n";
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing value for '" << key << "' at line " << line_num << ": " << e.what() << "\n";
            }
        }
    }
}
