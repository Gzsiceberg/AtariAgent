#pragma once

#include <torch/torch.h>

#include <pybind11/pybind11.h>

#include <string>
#include <unordered_map>

namespace atariagent::native {

using TensorState = std::unordered_map<std::string, torch::Tensor>;

TensorState tensor_state_from_dict(const pybind11::dict& state);
void load_state_dict(torch::nn::Module& module, const TensorState& state);
void load_state_dict(torch::nn::Module& module, const pybind11::dict& state);
pybind11::dict state_dict(torch::nn::Module& module);

}  // namespace atariagent::native
