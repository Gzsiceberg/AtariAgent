#include "inference/tree_search.h"

#include "native_search.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <span>
#include <utility>
#include <vector>

#ifdef ATARIAGENT_HAS_OPENMP
#include <omp.h>
#endif

namespace atariagent::native {
namespace {

std::atomic<int> tree_search_num_threads{1};

int configured_tree_search_num_threads() {
    return tree_search_num_threads.load(std::memory_order_relaxed);
}

}  // namespace

int set_tree_search_num_threads(int count) {
    if (count <= 0) {
        throw std::invalid_argument("thread count must be positive");
    }
    tree_search_num_threads.store(count, std::memory_order_relaxed);
#ifdef ATARIAGENT_HAS_OPENMP
    omp_set_num_threads(count);
    return count;
#else
    return 1;
#endif
}

TreeSearch::TreeSearch(
    std::shared_ptr<BatchedNetworkEvaluator> evaluator,
    std::int64_t num_simulations,
    double discount,
    double pb_c_init,
    double pb_c_base,
    double value_delta_max,
    double dirichlet_alpha,
    double root_exploration_fraction,
    std::int64_t value_prefix_horizon,
    std::uint64_t seed,
    const std::string& search_algorithm,
    std::int64_t num_top_actions,
    double c_visit,
    double c_scale,
    const std::string& search_value_mode
)
    : evaluator_(std::move(evaluator)),
      num_simulations_(num_simulations),
      discount_(discount),
      pb_c_init_(pb_c_init),
      pb_c_base_(pb_c_base),
      value_delta_max_(value_delta_max),
      dirichlet_alpha_(dirichlet_alpha),
      root_exploration_fraction_(root_exploration_fraction),
      value_prefix_horizon_(value_prefix_horizon),
      search_algorithm_(
          search_algorithm == "gumbel"
              ? SearchAlgorithm::Gumbel
              : SearchAlgorithm::Puct
      ),
      num_top_actions_(num_top_actions),
      c_visit_(c_visit),
      c_scale_(c_scale),
      simulation_average_(search_value_mode == "simulation_average"),
      rng_(seed) {
    if (search_value_mode != "improved_policy"
        && search_value_mode != "simulation_average") {
        throw std::invalid_argument(
            "search_value_mode must be improved_policy or simulation_average"
        );
    }
    if (!evaluator_) {
        throw std::invalid_argument("evaluator must not be null");
    }
    if (search_algorithm != "puct" && search_algorithm != "mcts"
        && search_algorithm != "gumbel") {
        throw std::invalid_argument("search_algorithm must be puct or gumbel");
    }
    if (num_simulations_ <= 0 || value_prefix_horizon_ <= 0) {
        throw std::invalid_argument(
            "simulation count and value-prefix horizon must be positive"
        );
    }
    if (discount_ < 0.0 || discount_ > 1.0 || pb_c_init_ < 0.0
        || pb_c_base_ <= 0.0 || value_delta_max_ <= 0.0
        || dirichlet_alpha_ <= 0.0
        || root_exploration_fraction_ < 0.0
        || root_exploration_fraction_ > 1.0) {
        throw std::invalid_argument("invalid tree-search configuration");
    }
    if (search_algorithm_ == SearchAlgorithm::Gumbel
        && (num_top_actions_ < 2 || num_top_actions_ > evaluator_->action_space_size()
            || (num_top_actions_ & (num_top_actions_ - 1)) != 0
            || num_simulations_ < num_top_actions_
            || c_visit_ < 0.0 || c_scale_ <= 0.0)) {
        throw std::invalid_argument("invalid Gumbel search configuration");
    }
}

bool TreeSearch::uses_gumbel() const {
    return search_algorithm_ == SearchAlgorithm::Gumbel;
}

std::tuple<torch::Tensor, torch::Tensor> TreeSearch::search_batch(
    const torch::Tensor& root_states,
    const torch::Tensor& root_values,
    const torch::Tensor& root_policy_logits,
    double root_noise_temperature,
    bool gumbel_sampling,
    bool deterministic_ties
) {
    c10::InferenceMode inference_guard;
    if (!std::isfinite(root_noise_temperature)
        || root_noise_temperature < 0.0
        || root_noise_temperature > 1.0) {
        throw std::invalid_argument(
            "root_noise_temperature must be in [0, 1]"
        );
    }
    if (root_states.dim() < 2) {
        throw std::invalid_argument(
            "root_states must contain a batch dimension"
        );
    }
    const auto root_count = root_states.size(0);
    if (root_values.sizes() != torch::IntArrayRef({root_count})) {
        throw std::invalid_argument(
            "root_values must have shape (batch_size,)"
        );
    }
    if (root_policy_logits.dim() != 2
        || root_policy_logits.size(0) != root_count) {
        throw std::invalid_argument(
            "root_policy_logits must have shape (batch_size, actions)"
        );
    }
    if (root_states.device() != root_values.device()
        || root_states.device() != root_policy_logits.device()) {
        throw std::invalid_argument("root tensors must be on the same device");
    }
    const auto action_count = root_policy_logits.size(1);
    evaluator_->validate_policy(root_policy_logits, root_count);
    if (root_count == 0) {
        return {
            torch::empty({0, action_count}, torch::kFloat32),
            torch::empty({0}, torch::kFloat32),
        };
    }

    torch::Tensor root_priors = torch::softmax(
        root_policy_logits.to(torch::kFloat32), -1
    ).cpu().contiguous();
    torch::Tensor cpu_values = root_values.to(torch::kFloat32)
        .cpu().contiguous();
    if (!torch::isfinite(root_priors).all().item<bool>()
        || !torch::isfinite(cpu_values).all().item<bool>()) {
        throw std::invalid_argument(
            "root values and policy logits must be finite"
        );
    }
    auto prior_data = root_priors.accessor<float, 2>();
    auto value_data = cpu_values.accessor<float, 1>();
    std::vector<RootTree> trees;
    trees.reserve(root_count);
    for (std::int64_t root = 0; root < root_count; ++root) {
        std::vector<float> priors(action_count);
        for (std::int64_t action = 0; action < action_count; ++action) {
            priors[action] = prior_data[root][action];
        }
        if (root_noise_temperature > 0.0 && !uses_gumbel()) {
            add_root_noise(priors, root_noise_temperature);
        }
        trees.emplace_back(
            priors,
            value_data[root],
            0.0F,
            static_cast<int>(num_simulations_),
            static_cast<float>(discount_),
            static_cast<int>(value_prefix_horizon_),
            static_cast<float>(value_delta_max_),
            rng_(),
            deterministic_ties,
            search_algorithm_,
            static_cast<int>(num_top_actions_),
            static_cast<float>(c_visit_),
            static_cast<float>(c_scale_),
            gumbel_sampling
        );
    }

    const auto device = root_states.device();
    LSTMHidden initial_hidden = evaluator_->initial_hidden(
        root_count, root_states
    );
    std::vector<std::int64_t> state_shape{
        root_count, num_simulations_ + 1
    };
    state_shape.insert(
        state_shape.end(),
        root_states.sizes().begin() + 1,
        root_states.sizes().end()
    );
    torch::Tensor state_store = root_states.new_empty(state_shape);
    state_store.select(1, 0).copy_(root_states);
    torch::Tensor hidden_store = root_states.new_empty(
        {root_count, num_simulations_ + 1, 512}
    );
    torch::Tensor cell_store = torch::empty_like(hidden_store);
    hidden_store.select(1, 0).copy_(
        std::get<0>(initial_hidden).select(0, 0)
    );
    cell_store.select(1, 0).copy_(
        std::get<1>(initial_hidden).select(0, 0)
    );
    torch::Tensor root_indices = torch::arange(
        root_count,
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );

    const bool pin_staging = device.is_cuda();
    const auto cpu_long_options = torch::TensorOptions()
        .dtype(torch::kLong).pinned_memory(pin_staging);
    const auto cpu_bool_options = torch::TensorOptions()
        .dtype(torch::kBool).pinned_memory(pin_staging);
    torch::Tensor cpu_state_slots = torch::empty({root_count}, cpu_long_options);
    torch::Tensor cpu_actions = torch::empty({root_count, 1}, cpu_long_options);
    torch::Tensor cpu_resets = torch::empty({root_count}, cpu_bool_options);
    torch::Tensor state_slots = torch::empty(
        {root_count},
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );
    torch::Tensor action_tensor = torch::empty(
        {root_count, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );
    torch::Tensor reset_tensor = torch::empty(
        {root_count},
        torch::TensorOptions().dtype(torch::kBool).device(device)
    );
    auto* slot_data = cpu_state_slots.data_ptr<std::int64_t>();
    auto* action_data = cpu_actions.data_ptr<std::int64_t>();
    auto* reset_data = cpu_resets.data_ptr<bool>();
    using namespace torch::indexing;
    for (std::int64_t simulation = 0; simulation < num_simulations_; ++simulation) {
#pragma omp parallel for if(root_count >= 32) schedule(static) \
    num_threads(configured_tree_search_num_threads())
        for (std::int64_t root = 0; root < root_count; ++root) {
            auto [slot, action, reset] = trees[root].traverse(
                static_cast<float>(pb_c_base_),
                static_cast<float>(pb_c_init_)
            );
            slot_data[root] = slot;
            action_data[root] = action;
            reset_data[root] = reset;
        }
        state_slots.copy_(cpu_state_slots, /*non_blocking=*/pin_staging);
        action_tensor.copy_(cpu_actions, /*non_blocking=*/pin_staging);
        reset_tensor.copy_(cpu_resets, /*non_blocking=*/pin_staging);
        torch::Tensor states = state_store.index({root_indices, state_slots});
        LSTMHidden hidden{
            hidden_store.index({root_indices, state_slots}).unsqueeze(0),
            cell_store.index({root_indices, state_slots}).unsqueeze(0),
        };
        auto [next_states, next_hidden, prefixes, values, policy_logits] =
            evaluator_->evaluate_tensors(
                states, action_tensor, hidden, reset_tensor
            );
        const auto next_slot = simulation + 1;
        state_store.select(1, next_slot).copy_(next_states);
        hidden_store.select(1, next_slot).copy_(
            std::get<0>(next_hidden).select(0, 0)
        );
        cell_store.select(1, next_slot).copy_(
            std::get<1>(next_hidden).select(0, 0)
        );
        torch::Tensor output = torch::cat(
            {prefixes.unsqueeze(1), values.unsqueeze(1), policy_logits}, 1
        ).to(torch::kFloat32).cpu().contiguous();
        const auto output_stride = output.size(1);
        const auto* output_data = output.data_ptr<float>();
        if (!std::all_of(
                output_data,
                output_data + output.numel(),
                [](float value) { return std::isfinite(value); }
            )) {
            throw std::invalid_argument(
                "recurrent inference output must be finite"
            );
        }
#pragma omp parallel for if(root_count >= 32) schedule(static) \
    num_threads(configured_tree_search_num_threads())
        for (std::int64_t root = 0; root < root_count; ++root) {
            const float* row = output_data + root * output_stride;
            trees[root].expand_and_back_up(
                static_cast<int>(next_slot),
                row[0],
                row[1],
                std::span<const float>(row + 2, action_count)
            );
        }
    }

    torch::Tensor policies = torch::empty(
        {root_count, action_count},
        torch::TensorOptions().dtype(torch::kFloat32)
    );
    torch::Tensor values = torch::empty(
        {root_count}, torch::TensorOptions().dtype(torch::kFloat32)
    );
    auto* policy_data = policies.data_ptr<float>();
    auto* roots_data = values.data_ptr<float>();
#pragma omp parallel for if(root_count >= 32) schedule(static) \
    num_threads(configured_tree_search_num_threads())
    for (std::int64_t root = 0; root < root_count; ++root) {
        roots_data[root] = trees[root].write_policy_and_root_value(
            policy_data + root * action_count, simulation_average_
        );
    }
    return {policies, values};
}

void TreeSearch::add_root_noise(
    std::vector<float>& priors,
    double temperature
) {
    std::gamma_distribution<double> distribution(dirichlet_alpha_, 1.0);
    std::vector<double> samples(priors.size());
    double total = 0.0;
    for (double& sample : samples) {
        sample = distribution(rng_);
        total += sample;
    }
    if (total <= 0.0) {
        std::fill(samples.begin(), samples.end(), 1.0 / samples.size());
    } else {
        for (double& sample : samples) {
            sample /= total;
        }
    }
    const double fraction = root_exploration_fraction_ * temperature;
    for (std::size_t action = 0; action < priors.size(); ++action) {
        priors[action] = static_cast<float>(
            (1.0 - fraction) * priors[action]
            + fraction * samples[action]
        );
    }
}

}  // namespace atariagent::native
