#pragma once

#include "models/representation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

namespace atariagent::native {

using LSTMHidden = std::tuple<torch::Tensor, torch::Tensor>;

class RewardPredictionNetworkImpl : public torch::nn::Module {
public:
    static constexpr std::int64_t hidden_size = 512;
    static constexpr std::int64_t support_size = 601;

    explicit RewardPredictionNetworkImpl(double batch_norm_momentum = 0.1);

    static LSTMHidden initial_hidden(const torch::Tensor& state);
    std::tuple<torch::Tensor, LSTMHidden> forward(
        const torch::Tensor& state,
        std::optional<LSTMHidden> hidden = std::nullopt
    );

private:
    torch::nn::Sequential features{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Sequential lstm_output{nullptr};
    torch::nn::Sequential projection{nullptr};
};

class DynamicsNetworkImpl : public torch::nn::Module {
public:
    DynamicsNetworkImpl(
        std::int64_t action_space_size,
        double batch_norm_momentum = 0.1,
        bool scale_state_gradient = true,
        std::int64_t action_embedding_dim = 16,
        bool action_embedding = true
    );

    std::tuple<torch::Tensor, LSTMHidden, torch::Tensor> forward(
        const torch::Tensor& state,
        const torch::Tensor& action,
        std::optional<LSTMHidden> reward_hidden = std::nullopt
    );

private:
    std::int64_t action_space_size_;
    bool scale_state_gradient_;
    std::int64_t action_embedding_dim_;
    bool action_embedding_;
    torch::nn::Conv2d action_projection{nullptr};
    torch::nn::LayerNorm action_normalization{nullptr};
    torch::nn::Sequential transition{nullptr};
    torch::nn::ReLU relu{nullptr};
    ResidualBlock residual{nullptr};
    std::shared_ptr<RewardPredictionNetworkImpl> reward_prediction{nullptr};
};

}  // namespace atariagent::native
