#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <bit> // Required for std::countr_zero

#include "connect4_game.h"
#include "mcts.h"
#include "model.h"
#include "game_play.h"
#include "neural_worker.h"

namespace py = pybind11;
using namespace Connect4;

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

private:
    GameState state;
};

// Wrapper for Neural Network
class PyNet {
public:
    PyNet() : net(std::make_shared<Connect4NetImpl>()) {}
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
            std::cout << "Model loaded successfully from: " << path << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
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

private:
    Connect4Net net;
};

// Wrapper for MCTS
class PyMCTS {
public:
    PyMCTS(float c_puct = 1.0f, float c_fpu = 0.25f, float virtual_loss = 2.0f)
        : mcts(c_puct, c_fpu, virtual_loss) {
        mcts.use_noise = false;
    }

    void search_batch(int count, int batch_size, const PyGameState& state,
        int player, PyNet& net_wrapper, const std::string& device_str) {
        torch::Device device(device_str);
        Connect4Net net = net_wrapper.get_net();
        if (!neural_worker_) {
            neural_worker_ = std::make_unique<Connect4::NeuralWorker>(net, device, 256);
            mcts.set_neural_worker(neural_worker_.get());
        }
        mcts.search_batch(count, batch_size, state.get_state(),
            static_cast<Player>(player), net, device);
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

    py::class_<PyMCTS>(m, "MCTS")
        .def(py::init<float, float, float>(),
            py::arg("c_puct") = 1.0f, py::arg("c_fpu") = 0.25f, py::arg("virtual_loss") = 2.0f)
        .def("search_batch", &PyMCTS::search_batch,
            py::arg("count"), py::arg("batch_size"), py::arg("state"),
            py::arg("player"), py::arg("net"), py::arg("device") = "cpu")
        .def("get_policy_value", &PyMCTS::get_policy_value,
            py::arg("state"), py::arg("tau") = 1.0f)
        .def("clear", &PyMCTS::clear)
        .def("size", &PyMCTS::size);

    py::class_<PyNet>(m, "Net")
        .def(py::init<>())
        .def(py::init<const std::string&, const int, const int, const std::string&>(),
            py::arg("path"), py::arg("blocks"), py::arg("filters"), py::arg("device") = "cpu")
        .def("load", &PyNet::load, py::arg("path"), py::arg("device") = "cpu")
        .def("forward", &PyNet::forward);

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