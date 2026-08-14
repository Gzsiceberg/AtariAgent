#include "models/dynamics.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace atariagent::native {

RewardPredictionNetworkImpl::RewardPredictionNetworkImpl(
    double batch_norm_momentum
) {
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
    lstm = register_module("lstm", torch::nn::LSTM(16 * 6 * 6, hidden_size));
    lstm_output = register_module(
        "lstm_output",
        torch::nn::Sequential(
            torch::nn::BatchNorm1d(
                torch::nn::BatchNorm1dOptions(hidden_size)
                    .momentum(batch_norm_momentum)
            ),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))
        )
    );
    torch::nn::Linear output(32, support_size);
    {
        torch::NoGradGuard no_grad;
        output->weight.zero_();
        output->bias.zero_();
    }
    projection = register_module(
        "projection",
        torch::nn::Sequential(
            torch::nn::Linear(hidden_size, 32),
            torch::nn::BatchNorm1d(
                torch::nn::BatchNorm1dOptions(32)
                    .momentum(batch_norm_momentum)
            ),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            output
        )
    );
}

LSTMHidden RewardPredictionNetworkImpl::initial_hidden(
    const torch::Tensor& state
) {
    const auto shape = std::vector<std::int64_t>{
        1, state.size(0), hidden_size
    };
    return {
        torch::zeros(shape, state.options()),
        torch::zeros(shape, state.options()),
    };
}

std::tuple<torch::Tensor, LSTMHidden> RewardPredictionNetworkImpl::forward(
    const torch::Tensor& state,
    std::optional<LSTMHidden> hidden
) {
    if (!hidden) {
        hidden = initial_hidden(state);
    }
    torch::Tensor input = features->forward(state).flatten(1).unsqueeze(0);
    auto [recurrent_output, next_hidden] = lstm->forward(input, hidden);
    recurrent_output = lstm_output->forward(recurrent_output.squeeze(0));
    return {projection->forward(recurrent_output), next_hidden};
}

DynamicsNetworkImpl::DynamicsNetworkImpl(
    std::int64_t action_space_size,
    double batch_norm_momentum,
    bool scale_state_gradient
)
    : action_space_size_(action_space_size),
      scale_state_gradient_(scale_state_gradient) {
    if (action_space_size_ <= 0) {
        throw std::invalid_argument("action_space_size must be positive");
    }
    transition = register_module(
        "transition",
        torch::nn::Sequential(
            conv3x3(65, 64),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(64)
                    .momentum(batch_norm_momentum)
            )
        )
    );
    relu = register_module(
        "relu", torch::nn::ReLU(torch::nn::ReLUOptions(true))
    );
    residual = register_module(
        "residual", ResidualBlock(64, std::nullopt, 1, batch_norm_momentum)
    );
    reward_prediction = register_module(
        "reward_prediction",
        std::make_shared<RewardPredictionNetworkImpl>(batch_norm_momentum)
    );
}

std::tuple<torch::Tensor, LSTMHidden, torch::Tensor>
DynamicsNetworkImpl::forward(
    const torch::Tensor& state,
    const torch::Tensor& action,
    std::optional<LSTMHidden> reward_hidden
) {
    torch::Tensor action_plane = action.reshape({action.size(0), 1, 1, 1})
        .expand({-1, 1, 6, 6})
        .to(state.options()) / static_cast<double>(action_space_size_);
    torch::Tensor transition_state = transition->forward(
        torch::cat({state, action_plane}, 1)
    );
    torch::Tensor next_state = residual->forward(
        relu->forward(transition_state + state)
    );
    if (scale_state_gradient_) {
        next_state = next_state * 0.5 + next_state.detach() * 0.5;
    }
    auto [value_prefix, next_hidden] = reward_prediction->forward(
        next_state, std::move(reward_hidden)
    );
    return {next_state, next_hidden, value_prefix};
}

}  // namespace atariagent::native
