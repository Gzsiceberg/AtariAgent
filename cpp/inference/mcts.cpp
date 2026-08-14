#include "inference/mcts.h"

#include "native_search.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace atariagent::native {

MCTS::MCTS(
    std::shared_ptr<BatchedNetworkEvaluator> evaluator,
    std::int64_t num_simulations,
    double discount,
    double pb_c_init,
    double pb_c_base,
    double value_delta_max,
    double dirichlet_alpha,
    double root_exploration_fraction,
    std::int64_t value_prefix_horizon,
    std::uint64_t seed
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
      rng_(seed) {
    if (!evaluator_) {
        throw std::invalid_argument("evaluator must not be null");
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
        throw std::invalid_argument("invalid MCTS configuration");
    }
}

std::tuple<torch::Tensor, torch::Tensor> MCTS::search_batch(
    const torch::Tensor& root_states,
    const torch::Tensor& root_values,
    const torch::Tensor& root_policy_logits,
    bool add_exploration_noise,
    bool deterministic_ties
) {
    c10::InferenceMode inference_guard;
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
            torch::empty({0, action_count}, torch::kInt32),
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
        if (add_exploration_noise) {
            add_root_noise(priors);
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
            deterministic_ties
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

    using namespace torch::indexing;
    for (std::int64_t simulation = 0; simulation < num_simulations_; ++simulation) {
        std::vector<std::int64_t> slots(root_count);
        std::vector<std::int64_t> actions(root_count);
        std::vector<std::int64_t> resets(root_count);
        for (std::int64_t root = 0; root < root_count; ++root) {
            auto [slot, action, reset] = trees[root].traverse(
                static_cast<float>(pb_c_base_),
                static_cast<float>(pb_c_init_)
            );
            slots[root] = slot;
            actions[root] = action;
            resets[root] = reset ? 1 : 0;
        }
        torch::Tensor state_slots = torch::tensor(
            slots,
            torch::TensorOptions().dtype(torch::kLong).device(device)
        );
        torch::Tensor action_tensor = torch::tensor(
            actions,
            torch::TensorOptions().dtype(torch::kLong).device(device)
        ).reshape({root_count, 1});
        torch::Tensor reset_tensor = torch::tensor(
            resets,
            torch::TensorOptions().dtype(torch::kBool).device(device)
        );
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
        auto output_data = output.accessor<float, 2>();
        for (std::int64_t root = 0; root < root_count; ++root) {
            std::vector<float> logits(action_count);
            for (std::int64_t action = 0; action < action_count; ++action) {
                logits[action] = output_data[root][action + 2];
            }
            trees[root].expand_and_back_up(
                static_cast<int>(next_slot),
                output_data[root][0],
                output_data[root][1],
                logits
            );
        }
    }

    torch::Tensor visits = torch::empty(
        {root_count, action_count},
        torch::TensorOptions().dtype(torch::kInt32)
    );
    torch::Tensor values = torch::empty(
        {root_count}, torch::TensorOptions().dtype(torch::kFloat32)
    );
    auto visits_data = visits.accessor<std::int32_t, 2>();
    auto roots_data = values.accessor<float, 1>();
    for (std::int64_t root = 0; root < root_count; ++root) {
        const std::vector<int> counts = trees[root].visit_counts();
        for (std::int64_t action = 0; action < action_count; ++action) {
            visits_data[root][action] = counts[action];
        }
        roots_data[root] = trees[root].root_value();
    }
    return {visits, values};
}

void MCTS::add_root_noise(std::vector<float>& priors) {
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
    for (std::size_t action = 0; action < priors.size(); ++action) {
        priors[action] = static_cast<float>(
            (1.0 - root_exploration_fraction_) * priors[action]
            + root_exploration_fraction_ * samples[action]
        );
    }
}

}  // namespace atariagent::native
