#pragma once

#include "models/dynamics.h"
#include "models/prediction.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

namespace atariagent::native {

torch::Tensor categorical_to_scalar(
    const torch::Tensor& logits,
    std::int64_t support_min,
    std::int64_t support_max,
    double epsilon = 0.001
);

using PackedEvaluation = std::tuple<
    torch::Tensor,
    LSTMHidden,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>;

class BatchedNetworkEvaluator {
public:
    BatchedNetworkEvaluator(
        std::shared_ptr<DynamicsNetworkImpl> dynamics,
        std::shared_ptr<PredictionNetworkImpl> prediction,
        std::int64_t action_space_size,
        std::int64_t support_min = -300,
        std::int64_t support_max = 300
    );

    LSTMHidden initial_hidden(
        std::int64_t batch_size,
        const torch::Tensor& state_template
    ) const;
    PackedEvaluation evaluate_tensors(
        const torch::Tensor& states,
        const torch::Tensor& actions,
        LSTMHidden hidden,
        std::optional<torch::Tensor> reset_value_prefix = std::nullopt
    );
    void validate_policy(
        const torch::Tensor& policy_logits,
        std::int64_t batch_size
    ) const;
    std::int64_t action_space_size() const;

private:
    std::shared_ptr<DynamicsNetworkImpl> dynamics_;
    std::shared_ptr<PredictionNetworkImpl> prediction_;
    std::int64_t action_space_size_;
    std::int64_t support_min_;
    std::int64_t support_max_;
};

}  // namespace atariagent::native
