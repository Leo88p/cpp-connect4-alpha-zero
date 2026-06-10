#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <numeric>

namespace Connect4 {

    class CSVLogger {
    public:
        explicit CSVLogger(const std::string& filename) {
            file_.open(filename);
            if (file_.is_open()) {
                file_ << "step,metric,value" << std::endl;
            }
            else {
                std::cerr << "Warning: Could not open CSV log file: " << filename << std::endl;
            }
        }

        ~CSVLogger() {
            if (file_.is_open()) {
                file_.close();
            }
        }

        void log(int step, const std::string& metric, float value) {
            if (file_.is_open()) {
                file_ << step << "," << metric << "," << value << std::endl;
                file_.flush();
            }
        }

        bool is_open() const {
            return file_.is_open();
        }

    private:
        std::ofstream file_;
    };

    class SimpleTracker {
    public:
        SimpleTracker(CSVLogger& logger, size_t batch_size)
            : logger_(logger), batch_size_(batch_size) {
        }

        void track(const std::string& name, float value, int64_t step) {
            buffer_[name].push_back(value);
            if (buffer_[name].size() >= batch_size_) {
                float mean = std::accumulate(buffer_[name].begin(), buffer_[name].end(), 0.0f) /
                    static_cast<float>(buffer_[name].size());

                // Print to console with same format as Python version
                std::cout << "Step " << step << ", " << name << ": "
                    << std::fixed << std::setprecision(4) << mean << std::endl;

                // Log to CSV
                logger_.log(static_cast<int>(step), name, mean);
                buffer_[name].clear();
            }
        }

    private:
        CSVLogger& logger_;
        size_t batch_size_;
        std::unordered_map<std::string, std::vector<float>> buffer_;
    };

} // namespace Connect4