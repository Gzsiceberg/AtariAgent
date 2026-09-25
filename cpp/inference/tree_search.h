#pragma once

#include "inference/evaluator.h"
#include "native_search.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace atariagent::native {

int set_tree_search_num_threads(int count);

class TreeSearch {
public:
    TreeSearch(
        std::shared_ptr<BatchedNetworkEvaluator> evaluator,
        std::int64_t num_simulations = 50,
        double discount = 0.997,
        double pb_c_init = 1.25,
        double pb_c_base = 19652.0,
        double value_delta_max = 0.01,
        double dirichlet_alpha = 0.3,
        double root_exploration_fraction = 0.25,
        std::int64_t value_prefix_horizon = 5,
        std::uint64_t seed = 0,
        const std::string& search_algorithm = "puct",
        std::int64_t num_top_actions = 4,
        double c_visit = 50.0,
        double c_scale = 0.1,
        const std::string& search_value_mode = "improved_policy"
    );

    std::tuple<torch::Tensor, torch::Tensor> search_batch(
        const torch::Tensor& root_states,
        const torch::Tensor& root_values,
        const torch::Tensor& root_policy_logits,
        double root_noise_temperature = 0.0,
        bool gumbel_sampling = false,
        bool deterministic_ties = false
    );

    bool uses_gumbel() const;

private:
    void add_root_noise(
        std::vector<float>& priors,
        double temperature
    );

    std::shared_ptr<BatchedNetworkEvaluator> evaluator_;
    std::int64_t num_simulations_;
    double discount_;
    double pb_c_init_;
    double pb_c_base_;
    double value_delta_max_;
    double dirichlet_alpha_;
    double root_exploration_fraction_;
    std::int64_t value_prefix_horizon_;
    SearchAlgorithm search_algorithm_;
    std::int64_t num_top_actions_;
    double c_visit_;
    double c_scale_;
    bool simulation_average_;
    std::mt19937_64 rng_;
};

}  // namespace atariagent::native
