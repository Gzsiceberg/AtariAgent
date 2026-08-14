#include <torch/extension.h>

#include "models/state_dict.h"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace py = pybind11;

namespace atariagent::native {

void load_state_dict(torch::nn::Module& module, const py::dict& state) {
    auto parameters = module.named_parameters(/*recurse=*/true);
    auto buffers = module.named_buffers(/*recurse=*/true);
    std::unordered_set<std::string> expected;
    expected.reserve(parameters.size() + buffers.size());
    for (const auto& item : parameters) {
        expected.insert(item.key());
    }
    for (const auto& item : buffers) {
        expected.insert(item.key());
    }
    for (const auto& key : expected) {
        if (!state.contains(py::str(key))) {
            throw std::invalid_argument("state_dict is missing key: " + key);
        }
    }
    for (const auto& item : state) {
        const std::string key = py::cast<std::string>(item.first);
        if (!expected.contains(key)) {
            throw std::invalid_argument("state_dict has unexpected key: " + key);
        }
    }
    torch::NoGradGuard no_grad;
    for (auto& item : parameters) {
        item.value().copy_(
            py::cast<torch::Tensor>(state[py::str(item.key())])
        );
    }
    for (auto& item : buffers) {
        item.value().copy_(
            py::cast<torch::Tensor>(state[py::str(item.key())])
        );
    }
}

py::dict state_dict(torch::nn::Module& module) {
    py::dict state;
    for (const auto& item : module.named_parameters(/*recurse=*/true)) {
        state[py::str(item.key())] = item.value();
    }
    for (const auto& item : module.named_buffers(/*recurse=*/true)) {
        state[py::str(item.key())] = item.value();
    }
    return state;
}

}  // namespace atariagent::native
