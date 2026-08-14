#pragma once

#include <torch/torch.h>

#include <pybind11/pybind11.h>

namespace atariagent::native {

void load_state_dict(torch::nn::Module& module, const pybind11::dict& state);
pybind11::dict state_dict(torch::nn::Module& module);

}  // namespace atariagent::native
