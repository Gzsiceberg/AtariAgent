#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>

namespace atariagent::native {

using FloatArray = pybind11::array_t<
    float,
    pybind11::array::c_style | pybind11::array::forcecast
>;

class BatchTree {
public:
    BatchTree(
        FloatArray root_priors,
        FloatArray root_values,
        FloatArray root_value_prefixes,
        int simulations,
        float discount,
        int value_prefix_horizon,
        float minimum_delta,
        std::uint64_t seed,
        bool deterministic_ties
    );
    ~BatchTree();

    BatchTree(const BatchTree&) = delete;
    BatchTree& operator=(const BatchTree&) = delete;

    pybind11::tuple traverse_arrays(float pb_c_base, float pb_c_init);
    void expand_and_back_up_arrays(
        int state_slot,
        FloatArray value_prefixes,
        FloatArray values,
        FloatArray policy_logits
    );
    pybind11::array_t<std::int32_t> visit_counts_array() const;
    pybind11::array_t<float> root_values_array() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

int set_search_num_threads(int count);

}  // namespace atariagent::native
