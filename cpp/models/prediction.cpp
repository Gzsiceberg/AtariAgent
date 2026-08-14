#include "models/prediction.h"

#include <stdexcept>

namespace atariagent::native {

PredictionHeadImpl::PredictionHeadImpl(
    std::int64_t output_size,
    double batch_norm_momentum
)
    : output_size_(output_size) {
    if (output_size_ <= 0) {
        throw std::invalid_argument("output_size must be positive");
    }
    features = register_module(
        "features",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 16, 1)),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(16)
                    .momentum(batch_norm_momentum)
            ),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))
        )
    );
    torch::nn::Linear output(32, output_size_);
    {
        torch::NoGradGuard no_grad;
        output->weight.zero_();
        output->bias.zero_();
    }
    projection = register_module(
        "projection",
        torch::nn::Sequential(
            torch::nn::Linear(16 * 6 * 6, 32),
            torch::nn::BatchNorm1d(
                torch::nn::BatchNorm1dOptions(32)
                    .momentum(batch_norm_momentum)
            ),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            output
        )
    );
}

torch::Tensor PredictionHeadImpl::forward(const torch::Tensor& state) {
    return projection->forward(features->forward(state).flatten(1));
}

PolicyNetworkImpl::PolicyNetworkImpl(
    std::int64_t action_space_size,
    double batch_norm_momentum
)
    : PredictionHeadImpl(action_space_size, batch_norm_momentum),
      action_space_size_(action_space_size) {}

torch::Tensor PolicyNetworkImpl::forward(const torch::Tensor& state) {
    return PredictionHeadImpl::forward(state);
}

ValueNetworkImpl::ValueNetworkImpl(
    std::int64_t support_size,
    double batch_norm_momentum
)
    : PredictionHeadImpl(support_size, batch_norm_momentum),
      support_size_(support_size) {}

torch::Tensor ValueNetworkImpl::forward(const torch::Tensor& state) {
    return PredictionHeadImpl::forward(state);
}

PredictionNetworkImpl::PredictionNetworkImpl(
    std::int64_t action_space_size,
    std::int64_t value_support_size,
    double batch_norm_momentum
) {
    residual = register_module(
        "residual", ResidualBlock(64, std::nullopt, 1, batch_norm_momentum)
    );
    policy = register_module(
        "policy",
        std::make_shared<PolicyNetworkImpl>(
            action_space_size, batch_norm_momentum
        )
    );
    value = register_module(
        "value",
        std::make_shared<ValueNetworkImpl>(
            value_support_size, batch_norm_momentum
        )
    );
}

std::tuple<torch::Tensor, torch::Tensor> PredictionNetworkImpl::forward(
    const torch::Tensor& state
) {
    torch::Tensor prediction_state = residual->forward(state);
    return {
        policy->forward(prediction_state),
        value->forward(prediction_state),
    };
}

}  // namespace atariagent::native
