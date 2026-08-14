#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace atariagent::native {

torch::nn::Conv2d conv3x3(
    std::int64_t in_channels,
    std::int64_t out_channels,
    std::int64_t stride = 1
);

class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(
        std::int64_t in_channels,
        std::optional<std::int64_t> out_channels = std::nullopt,
        std::int64_t stride = 1,
        double batch_norm_momentum = 0.1
    );

    torch::Tensor forward(const torch::Tensor& input);

private:
    std::int64_t in_channels_;
    std::int64_t out_channels_;
    std::int64_t stride_;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Conv2d skip_conv{nullptr};
    torch::nn::Identity skip_identity{nullptr};
};
TORCH_MODULE(ResidualBlock);

class RepresentationNetworkImpl : public torch::nn::Module {
public:
    explicit RepresentationNetworkImpl(
        std::int64_t in_channels,
        double batch_norm_momentum = 0.1
    );

    torch::Tensor forward(const torch::Tensor& observation);

private:
    std::int64_t in_channels_;
    torch::nn::Sequential stem{nullptr};
    ResidualBlock residual_48{nullptr};
    ResidualBlock downsample_24{nullptr};
    ResidualBlock residual_24{nullptr};
    torch::nn::AvgPool2d pool_12{nullptr};
    ResidualBlock residual_12{nullptr};
    torch::nn::AvgPool2d pool_6{nullptr};
    ResidualBlock residual_6{nullptr};
};

}  // namespace atariagent::native
