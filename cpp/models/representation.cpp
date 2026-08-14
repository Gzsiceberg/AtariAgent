#include "models/representation.h"

#include <stdexcept>

namespace atariagent::native {

torch::nn::Conv2d conv3x3(
    std::int64_t in_channels,
    std::int64_t out_channels,
    std::int64_t stride
) {
    return torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 3)
            .stride(stride)
            .padding(1)
            .bias(false)
    );
}

ResidualBlockImpl::ResidualBlockImpl(
    std::int64_t in_channels,
    std::optional<std::int64_t> out_channels,
    std::int64_t stride,
    double batch_norm_momentum
)
    : in_channels_(in_channels),
      out_channels_(out_channels.value_or(in_channels)),
      stride_(stride),
      conv1(register_module(
          "conv1", conv3x3(in_channels_, out_channels_, stride_)
      )),
      bn1(register_module(
          "bn1",
          torch::nn::BatchNorm2d(
              torch::nn::BatchNorm2dOptions(out_channels_)
                  .momentum(batch_norm_momentum)
          )
      )),
      conv2(register_module("conv2", conv3x3(out_channels_, out_channels_))),
      bn2(register_module(
          "bn2",
          torch::nn::BatchNorm2d(
              torch::nn::BatchNorm2dOptions(out_channels_)
                  .momentum(batch_norm_momentum)
          )
      )),
      relu(register_module(
          "relu", torch::nn::ReLU(torch::nn::ReLUOptions(true))
      )) {
    if (stride_ != 1 || in_channels_ != out_channels_) {
        skip_conv = register_module(
            "skip", conv3x3(in_channels_, out_channels_, stride_)
        );
    } else {
        skip_identity = register_module("skip", torch::nn::Identity());
    }
}

torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& input) {
    torch::Tensor identity = skip_conv ? skip_conv->forward(input) : input;
    torch::Tensor output = relu->forward(
        bn1->forward(conv1->forward(input))
    );
    output = bn2->forward(conv2->forward(output));
    return relu->forward(output + identity);
}

RepresentationNetworkImpl::RepresentationNetworkImpl(
    std::int64_t in_channels,
    double batch_norm_momentum
)
    : in_channels_(in_channels) {
    if (in_channels_ <= 0) {
        throw std::invalid_argument("in_channels must be positive");
    }
    stem = register_module(
        "stem",
        torch::nn::Sequential(
            conv3x3(in_channels_, 32, 2),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(32)
                    .momentum(batch_norm_momentum)
            ),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))
        )
    );
    residual_48 = register_module(
        "residual_48", ResidualBlock(32, std::nullopt, 1, batch_norm_momentum)
    );
    downsample_24 = register_module(
        "downsample_24", ResidualBlock(32, 64, 2, batch_norm_momentum)
    );
    residual_24 = register_module(
        "residual_24", ResidualBlock(64, std::nullopt, 1, batch_norm_momentum)
    );
    pool_12 = register_module(
        "pool_12",
        torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions(3).stride(2).padding(1)
        )
    );
    residual_12 = register_module(
        "residual_12", ResidualBlock(64, std::nullopt, 1, batch_norm_momentum)
    );
    pool_6 = register_module(
        "pool_6",
        torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions(3).stride(2).padding(1)
        )
    );
    residual_6 = register_module(
        "residual_6", ResidualBlock(64, std::nullopt, 1, batch_norm_momentum)
    );
}

torch::Tensor RepresentationNetworkImpl::forward(
    const torch::Tensor& observation
) {
    if (observation.dim() != 4 || observation.size(1) != in_channels_) {
        throw std::invalid_argument(
            "observation must have shape (batch, in_channels, 96, 96)"
        );
    }
    if (observation.size(2) != 96 || observation.size(3) != 96) {
        throw std::invalid_argument(
            "representation network expects 96x96 observations"
        );
    }
    torch::Tensor state = stem->forward(observation);
    state = residual_48->forward(state);
    state = downsample_24->forward(state);
    state = residual_24->forward(state);
    state = pool_12->forward(state);
    state = residual_12->forward(state);
    state = pool_6->forward(state);
    return residual_6->forward(state);
}

}  // namespace atariagent::native
