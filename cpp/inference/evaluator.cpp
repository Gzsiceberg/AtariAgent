#include "inference/evaluator.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace atariagent::native {

torch::Tensor categorical_to_scalar(
    const torch::Tensor& logits,
    std::int64_t support_min,
    std::int64_t support_max,
    double epsilon
) {
    if (support_min >= support_max) {
        throw std::invalid_argument("support_min must be less than support_max");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be positive");
    }
    const auto support_size = support_max - support_min + 1;
    if (logits.dim() == 0 || logits.size(-1) != support_size) {
        throw std::invalid_argument(
            "categorical logits have an invalid support size"
        );
    }
    torch::Tensor probabilities = torch::softmax(logits, -1);
    torch::Tensor support = torch::arange(
        support_min, support_max + 1, logits.options()
    );
    torch::Tensor transformed = (probabilities * support).sum(-1);
    torch::Tensor magnitude = (
        (
            torch::sqrt(
                1 + 4 * epsilon * (transformed.abs() + 1 + epsilon)
            ) - 1
        ) / (2 * epsilon)
    ).square() - 1;
    torch::Tensor scalar = torch::nan_to_num(transformed.sign() * magnitude);
    return torch::where(
        scalar.abs() < epsilon, torch::zeros_like(scalar), scalar
    );
}

BatchedNetworkEvaluator::BatchedNetworkEvaluator(
    std::shared_ptr<DynamicsNetworkImpl> dynamics,
    std::shared_ptr<PredictionNetworkImpl> prediction,
    std::int64_t action_space_size,
    std::int64_t support_min,
    std::int64_t support_max
)
    : dynamics_(std::move(dynamics)),
      prediction_(std::move(prediction)),
      action_space_size_(action_space_size),
      support_min_(support_min),
      support_max_(support_max) {
    if (!dynamics_ || !prediction_) {
        throw std::invalid_argument("evaluator networks must not be null");
    }
    if (action_space_size_ <= 0) {
        throw std::invalid_argument("action_space_size must be positive");
    }
    if (support_max_ - support_min_ + 1 != 601) {
        throw std::invalid_argument(
            "native dynamics requires a categorical support size of 601"
        );
    }
}

LSTMHidden BatchedNetworkEvaluator::initial_hidden(
    std::int64_t batch_size,
    const torch::Tensor& state_template
) const {
    const auto shape = std::vector<std::int64_t>{1, batch_size, 512};
    return {
        torch::zeros(shape, state_template.options()),
        torch::zeros(shape, state_template.options()),
    };
}

PackedEvaluation BatchedNetworkEvaluator::evaluate_tensors(
    const torch::Tensor& states,
    const torch::Tensor& actions,
    LSTMHidden hidden,
    std::optional<torch::Tensor> reset_value_prefix
) {
    const auto batch_size = states.size(0);
    if (actions.sizes() != torch::IntArrayRef({batch_size, 1})) {
        throw std::invalid_argument(
            "actions must have shape (batch_size, 1)"
        );
    }
    auto [hidden_state, cell_state] = std::move(hidden);
    if (reset_value_prefix) {
        if (reset_value_prefix->sizes() != torch::IntArrayRef({batch_size})) {
            throw std::invalid_argument(
                "reset mask must have shape (batch_size,)"
            );
        }
        torch::Tensor reset = reset_value_prefix->reshape({1, batch_size, 1});
        hidden_state = hidden_state.masked_fill(reset, 0.0);
        cell_state = cell_state.masked_fill(reset, 0.0);
    }
    auto [next_states, next_hidden, prefix_logits] = dynamics_->forward(
        states, actions, LSTMHidden{hidden_state, cell_state}
    );
    auto [policy_logits, value_logits] = prediction_->forward(next_states);
    validate_policy(policy_logits, batch_size);
    torch::Tensor values = categorical_to_scalar(
        value_logits.to(torch::kFloat32), support_min_, support_max_
    ).reshape({batch_size});
    torch::Tensor value_prefixes = categorical_to_scalar(
        prefix_logits.to(torch::kFloat32), support_min_, support_max_
    ).reshape({batch_size});
    return {
        next_states,
        next_hidden,
        value_prefixes,
        values,
        policy_logits,
    };
}

void BatchedNetworkEvaluator::validate_policy(
    const torch::Tensor& policy_logits,
    std::int64_t batch_size
) const {
    if (policy_logits.sizes() != torch::IntArrayRef(
        {batch_size, action_space_size_}
    )) {
        throw std::invalid_argument(
            "prediction network returned an invalid policy shape"
        );
    }
}

std::int64_t BatchedNetworkEvaluator::action_space_size() const {
    return action_space_size_;
}

}  // namespace atariagent::native
