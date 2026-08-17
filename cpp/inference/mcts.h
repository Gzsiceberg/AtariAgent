#pragma once

#include "inference/evaluator.h"

#include <cstdint>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace atariagent::native {

int set_mcts_num_threads(int count);

class MCTS {
public:
    MCTS(
        std::shared_ptr<BatchedNetworkEvaluator> evaluator,
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

    std::tuple<torch::Tensor, torch::Tensor> search_batch(
        const torch::Tensor& root_states,
        const torch::Tensor& root_values,
        const torch::Tensor& root_policy_logits,
        bool add_exploration_noise = false,
        bool deterministic_ties = false
    );

private:
    void add_root_noise(std::vector<float>& priors);

    std::shared_ptr<BatchedNetworkEvaluator> evaluator_;
    std::int64_t num_simulations_;
    double discount_;
    double pb_c_init_;
    double pb_c_base_;
    double value_delta_max_;
    double dirichlet_alpha_;
    double root_exploration_fraction_;
    std::int64_t value_prefix_horizon_;
    std::mt19937_64 rng_;
};

}  // namespace atariagent::native
