#pragma once

#include "inference/mcts.h"
#include "models/representation.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

namespace atariagent::native {

class ValueTargetNetwork {
public:
    ValueTargetNetwork(
        std::shared_ptr<RepresentationNetworkImpl> representation,
        std::shared_ptr<PredictionNetworkImpl> prediction,
        std::shared_ptr<DynamicsNetworkImpl> dynamics,
        std::int64_t action_space_size,
        std::int64_t support_min = -300,
        std::int64_t support_max = 300,
        std::int64_t chunk_size = 768,
        const std::string& precision = "fp32",
        std::int64_t num_simulations = 50,
        double discount = 0.997,
        double pb_c_init = 1.25,
        double pb_c_base = 19652.0,
        double value_delta_max = 0.01,
        double dirichlet_alpha = 0.3,
        double root_exploration_fraction = 0.25,
        std::int64_t value_prefix_horizon = 5,
        std::uint64_t seed = 0
    );

    void eval();
    void to(const std::string& device);
    void synchronize(
        const pybind11::dict& representation,
        const pybind11::dict& prediction,
        const pybind11::dict& dynamics
    );
    torch::Tensor decoded_values(const torch::Tensor& observations);
    torch::Tensor reanalyze_values(
        const torch::Tensor& bootstrap_frames,
        const torch::Tensor& bootstrap_mask,
        const torch::Tensor& stored_bootstrap_values,
        const torch::Tensor& bootstrap_discounts,
        const torch::Tensor& value_targets,
        std::int64_t stack_size
    );
    torch::Tensor reanalyze_policies(
        const torch::Tensor& frames,
        const torch::Tensor& policy_mask,
        const torch::Tensor& policy_targets,
        std::int64_t stack_size,
        bool add_exploration_noise = true,
        bool deterministic_ties = false
    );
    pybind11::object reanalyze_batch(
        const pybind11::object& batch,
        bool add_exploration_noise = true,
        bool deterministic_ties = false
    );

private:
    static torch::Tensor stacked_observations(
        const torch::Tensor& frames,
        const torch::Tensor& positions,
        std::int64_t stack_size
    );

    std::shared_ptr<RepresentationNetworkImpl> representation_;
    std::shared_ptr<PredictionNetworkImpl> prediction_;
    std::shared_ptr<DynamicsNetworkImpl> dynamics_;
    std::shared_ptr<BatchedNetworkEvaluator> evaluator_;
    std::shared_ptr<MCTS> mcts_;
    std::int64_t support_min_;
    std::int64_t support_max_;
    std::int64_t chunk_size_;
    bool use_bfloat16_;
};

}  // namespace atariagent::native
