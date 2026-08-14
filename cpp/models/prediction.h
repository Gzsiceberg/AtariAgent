#pragma once

#include "models/representation.h"

#include <cstdint>
#include <memory>
#include <tuple>

namespace atariagent::native {

class PredictionHeadImpl : public torch::nn::Module {
public:
    PredictionHeadImpl(
        std::int64_t output_size,
        double batch_norm_momentum = 0.1
    );

    torch::Tensor forward(const torch::Tensor& state);

private:
    std::int64_t output_size_;
    torch::nn::Sequential features{nullptr};
    torch::nn::Sequential projection{nullptr};
};

class PolicyNetworkImpl : public PredictionHeadImpl {
public:
    PolicyNetworkImpl(
        std::int64_t action_space_size,
        double batch_norm_momentum = 0.1
    );

    torch::Tensor forward(const torch::Tensor& state);

private:
    std::int64_t action_space_size_;
};

class ValueNetworkImpl : public PredictionHeadImpl {
public:
    static constexpr std::int64_t default_support_size = 601;

    ValueNetworkImpl(
        std::int64_t support_size = default_support_size,
        double batch_norm_momentum = 0.1
    );

    torch::Tensor forward(const torch::Tensor& state);

private:
    std::int64_t support_size_;
};

class PredictionNetworkImpl : public torch::nn::Module {
public:
    PredictionNetworkImpl(
        std::int64_t action_space_size,
        std::int64_t value_support_size = ValueNetworkImpl::default_support_size,
        double batch_norm_momentum = 0.1
    );

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& state
    );

private:
    ResidualBlock residual{nullptr};
    std::shared_ptr<PolicyNetworkImpl> policy{nullptr};
    std::shared_ptr<ValueNetworkImpl> value{nullptr};
};

}  // namespace atariagent::native
