#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

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
    PyGameState() : state(GameLogic::INITIAL_STATE) {}
    PyGameState(const GameState& s) : state(s) {}

    std::vector<std::vector<int>> to_list_representation() const {
        return GameLogic::to_list_representation(state);
    }

    static PyGameState initial_state() {
        return PyGameState(GameLogic::INITIAL_STATE);
    }

    GameState get_state() const { return state; }
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
        // Convert numpy array to torch tensor
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

        // Convert to numpy arrays
        auto logits_cpu = logits.to(torch::kCPU).detach();
        auto values_cpu = values.to(torch::kCPU).detach();

        auto logits_array = py::array_t<float>({
            static_cast<py::ssize_t>(logits_cpu.size(0)),
            static_cast<py::ssize_t>(logits_cpu.size(1))
            });

        auto values_array = py::array_t<float>({
            static_cast<py::ssize_t>(values_cpu.size(0))
            });

        // Copy data
        std::memcpy(logits_array.mutable_data(),
            logits_cpu.data_ptr<float>(),
            logits_cpu.numel() * sizeof(float));

        std::memcpy(values_array.mutable_data(),
            values_cpu.data_ptr<float>(),
            values_cpu.numel() * sizeof(float));

        return { logits_array, values_array };
    }

    Connect4Net get_net() const { return net; }

private:
    Connect4Net net;
};

// Wrapper for MCTS
class PyMCTS {
public:
    PyMCTS(float c_puct = 1.0f) : mcts(c_puct) {
        mcts.use_noise = false;
    }

    void search_batch(int count, int batch_size, const PyGameState& state,
        int player, PyNet& net_wrapper, const std::string& device_str) {
        torch::Device device(device_str);
        // Get the actual network from the wrapper
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

// Play game function wrapper
std::tuple<std::vector<std::tuple<int, int, float>>, float, int, bool>
py_play_game(PyMCTS& mcts1, PyMCTS& mcts2, PyNet& net1, PyNet& net2,
    int steps_before_tau_0, int mcts_searches, int mcts_batch_size,
    std::optional<bool> net1_plays_first, const std::string& device_str,
    size_t max_replay_size) {

    torch::Device device(device_str);


    std::vector<MCTS> mcts_stores;
    mcts_stores.push_back(MCTS(1.0f)); // Copy from PyMCTS
    mcts_stores.push_back(MCTS(1.0f)); // Copy from PyMCTS

    auto n1 = net1.get_net();
    auto n2 = net2.get_net();
    auto [result, steps] = play_game(
        mcts_stores, nullptr,
        n1, n2,
        steps_before_tau_0, mcts_searches, mcts_batch_size,
        net1_plays_first, device
    );

    // For visualization, we just need the result and steps
    std::vector<std::tuple<int, int, float>> empty_moves; // Placeholder
    return { empty_moves, result, steps, true }; // won flag not used
}

// Module definition
// Module definition
PYBIND11_MODULE(_C, m) {
    m.doc() = "Connect4 AlphaZero C++ core with high-performance simulation";

    m.attr("GAME_ROWS") = GAME_ROWS;
    m.attr("GAME_COLS") = GAME_COLS;
    m.attr("COUNT_TO_WIN") = COUNT_TO_WIN;

    py::class_<PyGameState>(m, "GameState")
        .def(py::init<>())
        .def("to_list_representation", &PyGameState::to_list_representation)
        .def_static("initial_state", &PyGameState::initial_state)
        .def_property_readonly("heights", [](const PyGameState& self) {
        std::array<int, GAME_COLS> heights;
        const GameState& state = self.get_state();
        for (int i = 0; i < GAME_COLS; ++i) {
            heights[i] = state.heights[i];
        }
        return heights;
            });

    py::class_<PyMCTS>(m, "MCTS")
        .def(py::init<float>(), py::arg("c_puct") = 1.0f)
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
        .def("load", &PyNet::load,
            py::arg("path"), py::arg("device") = "cpu")
        .def("forward", &PyNet::forward)
        .def("get_net", &PyNet::get_net);

    m.def("play_game", &py_play_game,
        py::arg("mcts1"), py::arg("mcts2"), py::arg("net1"), py::arg("net2"),
        py::arg("steps_before_tau_0") = 10, py::arg("mcts_searches") = 40,
        py::arg("mcts_batch_size") = 32, py::arg("net1_plays_first") = std::nullopt,
        py::arg("device") = "cpu", py::arg("max_replay_size") = 5000);

    m.def("make_move", [](const PyGameState& state, int col, int player) {
        auto [new_state, won] = GameLogic::make_move(state.get_state(), col);
        return std::make_pair(PyGameState(new_state), won);
        }, py::arg("state"), py::arg("col"), py::arg("player"));

    m.def("possible_moves", [](const PyGameState& state) {
        return GameLogic::get_possible_moves(state.get_state());
        }, py::arg("state"));
}