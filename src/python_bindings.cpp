#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <bit> // Required for std::countr_zero

#include "connect4_game.h"
#include "mcts.h"
#include "model.h"
#include "game_play.h"
#include "neural_worker.h"
#include "solver/Solver.hpp"

namespace py = pybind11;
using namespace Connect4;
using namespace Connect4::GameSolver;

// Wrapper class for GameState to expose to Python
class PyGameState {
public:
    PyGameState() : state() {} // Default constructor creates an empty initial state
    PyGameState(const GameState& s) : state(s) {}

    // Convert bitboard to a 2D list (6 rows x 7 cols) for Python visualization
    // 0 = empty, 1 = current player to move, 2 = opponent
    std::vector<std::vector<int>> to_list_representation() const {
        std::vector<std::vector<int>> board(GAME_ROWS, std::vector<int>(GAME_COLS, 0));
        uint64_t our_pieces = state.current_position;
        uint64_t their_pieces = state.mask ^ state.current_position;

        for (int col = 0; col < GAME_COLS; ++col) {
            for (int row = 0; row < GAME_ROWS; ++row) {
                int pos = col * (GAME_ROWS + 1) + row;
                int visual_row = (GAME_ROWS - 1) - row; // Flip so row 0 is top of the board

                if (our_pieces & (1ULL << pos)) {
                    board[visual_row][col] = 1;
                }
                else if (their_pieces & (1ULL << pos)) {
                    board[visual_row][col] = 2;
                }
            }
        }
        return board;
    }

    static PyGameState initial_state() {
        return PyGameState(); // Default constructor is the initial state
    }

    GameState get_state() const { return state; }

    // Helper to get valid moves as a list of ints for Python
    std::vector<int> get_possible_moves() const {
        std::vector<int> moves;
        uint64_t possible = state.possible();
        while (possible) {
            uint64_t move = possible & -possible; // Isolate lowest set bit
            moves.push_back(std::countr_zero(move) / (GAME_ROWS + 1));
            possible ^= move; // Clear lowest set bit
        }
        return moves;
    }

    bool can_play(int col) const {
        return state.canPlay(col);
    }

    bool is_winning_move(int col) const {
        return state.isWinningMove(col);
    }

    uint64_t key() const {
        return state.key();
    }

private:
    GameState state;
};

// Wrapper for Neural Network
class PyNet {
public:
    PyNet() : net(std::make_shared<Connect4NetImpl>()) {}
    PyNet(int blocks, int filters, const std::string& device_str = "cpu", float lr = 0.001f)
        : net(std::make_shared<Connect4NetImpl>(blocks, filters)) {
        torch::Device device(device_str);
        net->to(device);
        net->train();

        // Initialize Adam optimizer
        optimizer_ = std::make_unique<torch::optim::AdamW>(
            net->parameters(),
            torch::optim::AdamWOptions(lr).weight_decay(1e-4)
        );
    }

    // 2. INFERENCE CONSTRUCTOR (Loads weights & sets eval mode)
    PyNet(const std::string& path, const int blocks, const int filters, const std::string& device_str = "cpu")
        : net(std::make_shared<Connect4NetImpl>(blocks, filters)) {
        load(path, device_str);
        net->eval();
        torch::NoGradGuard no_grad;
    }

    void load(const std::string& path, const std::string& device_str = "cpu") {
        torch::Device device(device_str);
        try {
            torch::load(net, path);
            net->to(device);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }

    void save(const std::string& path) {
        torch::save(net, path);
    }

    std::pair<py::array_t<float>, py::array_t<float>> forward(py::array_t<float> input_array) {
        py::buffer_info buf = input_array.request();

        torch::Tensor tensor;
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(net->parameters().empty() ? torch::kCPU : net->parameters().front().device());

        if (buf.ndim == 4) {
            tensor = torch::from_blob(buf.ptr,
                { static_cast<int64_t>(buf.shape[0]), static_cast<int64_t>(buf.shape[1]),
                 static_cast<int64_t>(buf.shape[2]), static_cast<int64_t>(buf.shape[3]) },
                options).clone();
        }
        else if (buf.ndim == 3) {
            tensor = torch::from_blob(buf.ptr,
                { 1, static_cast<int64_t>(buf.shape[0]), static_cast<int64_t>(buf.shape[1]),
                 static_cast<int64_t>(buf.shape[2]) },
                options).clone();
        }
        else {
            throw std::runtime_error("Invalid input dimensions: " + std::to_string(buf.ndim));
        }

        auto [logits, values] = net->forward(tensor);

        auto logits_cpu = logits.to(torch::kCPU).detach();
        auto values_cpu = values.to(torch::kCPU).detach();

        auto logits_array = py::array_t<float>({
            static_cast<py::ssize_t>(logits_cpu.size(0)),
            static_cast<py::ssize_t>(logits_cpu.size(1))
            });

        auto values_array = py::array_t<float>({
            static_cast<py::ssize_t>(values_cpu.size(0))
            });

        std::memcpy(logits_array.mutable_data(), logits_cpu.data_ptr<float>(), logits_cpu.numel() * sizeof(float));
        std::memcpy(values_array.mutable_data(), values_cpu.data_ptr<float>(), values_cpu.numel() * sizeof(float));

        return { logits_array, values_array };
    }

    Connect4Net get_net() const { return net; }

    // 3. CORRECTED TRAINING STEP
    std::tuple<float, float, float> train_step(
        py::array_t<float> states_array,
        py::array_t<float> policies_array,
        py::array_t<float> values_array,
        float c_policy = 1.0f,
        float c_value = 1.0f,
        float max_grad_norm = 1.0f
    ) {
        if (!optimizer_) {
            throw std::runtime_error("Optimizer not initialized. Use the training constructor.");
        }

        // ==========================================
        // PHASE 1: EXTRACT RAW POINTERS (Holds GIL)
        // ==========================================
        // We must do this while the GIL is held because we are accessing Python objects.
        auto states_buf = states_array.request();
        auto policies_buf = policies_array.request();
        auto values_buf = values_array.request();

        float* states_ptr = static_cast<float*>(states_buf.ptr);
        std::vector<int64_t> states_shape(states_buf.shape.begin(), states_buf.shape.end());

        float* policies_ptr = static_cast<float*>(policies_buf.ptr);
        std::vector<int64_t> policies_shape(policies_buf.shape.begin(), policies_buf.shape.end());

        float* values_ptr = static_cast<float*>(values_buf.ptr);
        std::vector<int64_t> values_shape(values_buf.shape.begin(), values_buf.shape.end());

        // ==========================================
        // PHASE 2: RELEASE GIL (Autograd requires this!)
        // ==========================================
        py::gil_scoped_release release;

        net->train();
        auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        // Create tensors from raw pointers. 
        // .clone() is critical: it copies the memory so PyTorch owns it and doesn't 
        // rely on Python's NumPy buffers which could be garbage collected or moved.
        torch::Tensor states_v = torch::from_blob(states_ptr, states_shape, cpu_options).clone();
        torch::Tensor policies_v = torch::from_blob(policies_ptr, policies_shape, cpu_options).clone();
        torch::Tensor values_v = torch::from_blob(values_ptr, values_shape, cpu_options).clone();

        // Move to GPU if necessary
        torch::Device device = net->parameters().front().device();
        if (device.is_cuda()) {
            states_v = states_v.to(device, /*non_blocking=*/true);
            policies_v = policies_v.to(device, /*non_blocking=*/true);
            values_v = values_v.to(device, /*non_blocking=*/true);
        }

        // Forward & Loss
        optimizer_->zero_grad();
        auto [logits, pred_values] = net->forward(states_v);

        auto loss_value_v = torch::mse_loss(pred_values.squeeze(-1), values_v);
        auto log_probs = torch::log_softmax(logits, 1);
        auto loss_policy_v = -log_probs * policies_v;
        loss_policy_v = loss_policy_v.sum(1).mean();

        auto loss_v = c_policy * loss_policy_v + c_value * loss_value_v;

        // Autograd backward pass (Now safely runs without GIL!)
        loss_v.backward();
        torch::nn::utils::clip_grad_norm_(net->parameters(), max_grad_norm);
        optimizer_->step();

        return std::make_tuple(
            loss_v.item<float>(),
            loss_policy_v.item<float>(),
            loss_value_v.item<float>()
        );
    }

private:
    Connect4Net net;
    std::unique_ptr<torch::optim::AdamW> optimizer_ = nullptr;
};

class PyNeuralWorker {
public:
    PyNeuralWorker(PyNet& net_wrapper, const std::string& device_str, int max_batch_size = 256) {
        torch::Device device(device_str);
        Connect4Net net = net_wrapper.get_net();
        worker_ = std::make_shared<Connect4::NeuralWorker>(net, device, max_batch_size);
    }

    std::shared_ptr<Connect4::NeuralWorker> get() const { return worker_; }
private:
    std::shared_ptr<Connect4::NeuralWorker> worker_;
};

// Wrapper for MCTS
class PyMCTS {
public:
    PyMCTS(float c_puct = 1.0f, float virtual_loss = 2.0f)
        : mcts(c_puct, 0, 0, virtual_loss) {
        mcts.use_noise = false;
    }

    void set_neural_worker(PyNeuralWorker& worker) {
        shared_worker_ = worker.get();
        mcts.set_neural_worker(shared_worker_.get());
    }

    void search_batch(int count, int batch_size, const PyGameState& state,
        int player, PyNet& net_wrapper, const std::string& device_str) {
        // Fallback to internal worker if set_neural_worker wasn't called
        if (!shared_worker_) {
            torch::Device device(device_str);
            Connect4Net net = net_wrapper.get_net();
            if (!neural_worker_) {
                neural_worker_ = std::make_unique<Connect4::NeuralWorker>(net, device, 256);
                mcts.set_neural_worker(neural_worker_.get());
            }
        }

        mcts.search_batch(count, batch_size, state.get_state(), static_cast<Player>(player));
    }

    std::pair<std::vector<float>, std::vector<float>> get_policy_value(
        const PyGameState& state, float tau = 1.0f) {

        auto [probs, values] = mcts.get_policy_value(state.get_state(), tau);

        std::vector<float> probs_vec(GAME_COLS);
        std::vector<float> values_vec(GAME_COLS);

        for (int i = 0; i < GAME_COLS; ++i) {
            probs_vec[i] = probs[i];
            values_vec[i] = values[i];
        }

        return { probs_vec, values_vec };
    }

    void clear() { mcts.clear(); }
    size_t size() const { return mcts.size(); }

private:
    MCTS mcts;
    std::unique_ptr<Connect4::NeuralWorker> neural_worker_ = nullptr;
    std::shared_ptr<Connect4::NeuralWorker> shared_worker_ = nullptr; // New member
};

class PySolver {
public:
    PySolver() {
        weak = false;
    }
    PySolver(std::string bookPath) {
        weak = false;
        solver.loadBook(bookPath);
    }
    int solve(const PyGameState& state) {
        return solver.solve(state.get_state(), weak);
    }
    std::vector<int> analyze(const PyGameState& state) {
        return solver.analyze(state.get_state(), weak);
    }
private:
    Solver solver;
    bool weak;
};

// Module definition
PYBIND11_MODULE(_C, m) {
    m.doc() = "Connect4 AlphaZero C++ core with high-performance bitboard simulation";

    m.attr("GAME_ROWS") = GAME_ROWS;
    m.attr("GAME_COLS") = GAME_COLS;

    py::class_<PyGameState>(m, "GameState")
        .def(py::init<>())
        .def("to_list_representation", &PyGameState::to_list_representation)
        .def_static("initial_state", &PyGameState::initial_state)
        .def("get_possible_moves", &PyGameState::get_possible_moves)
        .def("can_play", &PyGameState::can_play)
        .def("is_winning_move", &PyGameState::is_winning_move)
        .def("__hash__", &PyGameState::key)
        .def("__eq__", [](const PyGameState& self, const PyGameState& other) {
        return self.get_state().key() == other.get_state().key();
            })
        .def_property_readonly("heights", [](const PyGameState& self) {
        std::array<int, GAME_COLS> heights;
        const GameState& state = self.get_state();
        uint64_t mask = state.mask;
        for (int col = 0; col < GAME_COLS; ++col) {
            // Extract the 7-bit chunk for this column (6 rows + 1 guard bit)
            int shift = col * (GAME_ROWS + 1);
            uint64_t col_mask = (mask >> shift) & 0x7F;
            // Trailing zeros perfectly equal the number of pieces in the column
            heights[col] = std::countr_zero(col_mask);
        }
        return heights;
            });

    py::class_<PyNeuralWorker, std::shared_ptr<PyNeuralWorker>>(m, "NeuralWorker")
        .def(py::init<PyNet&, const std::string&, int>(),
            py::arg("net"), py::arg("device"), py::arg("max_batch_size") = 256);

    py::class_<PyMCTS>(m, "MCTS")
        .def(py::init<float, float>(),
            py::arg("c_puct") = 1.0f, py::arg("virtual_loss") = 2.0f)
        .def("set_neural_worker", &PyMCTS::set_neural_worker, py::arg("worker"))
        .def("search_batch", &PyMCTS::search_batch,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("count"), py::arg("batch_size"), py::arg("state"),
            py::arg("player"), py::arg("net"), py::arg("device") = "cpu")
        .def("get_policy_value", &PyMCTS::get_policy_value,
            py::arg("state"), py::arg("tau") = 1.0f)
        .def("clear", &PyMCTS::clear)
        .def("size", &PyMCTS::size);

    py::class_<PyNet>(m, "Net")
        .def(py::init<>())
        // Training Constructor
        .def(py::init<int, int, const std::string&, float>(),
            py::arg("blocks"), py::arg("filters"), py::arg("device") = "cpu", py::arg("lr") = 0.001f)
        // Inference Constructor
        .def(py::init<const std::string&, const int, const int, const std::string&>(),
            py::arg("path"), py::arg("blocks"), py::arg("filters"), py::arg("device") = "cpu")
        .def("load", &PyNet::load, py::arg("path"), py::arg("device") = "cpu")
        .def("save", &PyNet::save, py::arg("path"))
        .def("forward", &PyNet::forward)
        .def("train_step", &PyNet::train_step,
            py::arg("states"), py::arg("policies"), py::arg("values"),
            py::arg("c_policy") = 1.0f, py::arg("c_value") = 1.0f, py::arg("max_grad_norm") = 1.0f);

    py::class_<PySolver>(m, "Solver")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("solve", &PySolver::solve, py::arg("state"))
        .def("analyze", &PySolver::analyze, 
            py::call_guard<py::gil_scoped_release>(),
            py::arg("state"));

    // make_move no longer needs the 'player' argument because the bitboard inherently knows whose turn it is!
    m.def("make_move", [](const PyGameState& state, int col) {
        GameState new_state = state.get_state();
        bool won = new_state.isWinningMove(col);
        new_state.playCol(col); // playCol automatically swaps current_position and mask
        return std::make_pair(PyGameState(new_state), won);
        }, py::arg("state"), py::arg("col"));

    m.def("possible_moves", [](const PyGameState& state) {
        return state.get_possible_moves();
        }, py::arg("state"));
}