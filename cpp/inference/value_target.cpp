#include <torch/extension.h>

#include "inference/value_target.h"

#include "models/state_dict.h"

#include <ATen/autocast_mode.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace atariagent::native {
namespace {

class BFloat16AutocastGuard {
public:
    BFloat16AutocastGuard(const torch::Tensor& input, bool enabled)
        : enabled_(enabled), device_type_(input.device().type()) {
        if (!enabled_) {
            return;
        }
        if (!at::autocast::is_autocast_available(device_type_)) {
            throw std::invalid_argument(
                "bfloat16 autocast is unavailable on the target device"
            );
        }
        previous_enabled_ = at::autocast::is_autocast_enabled(device_type_);
        previous_dtype_ = at::autocast::get_autocast_dtype(device_type_);
        at::autocast::set_autocast_dtype(device_type_, torch::kBFloat16);
        at::autocast::set_autocast_enabled(device_type_, true);
    }

    ~BFloat16AutocastGuard() {
        if (enabled_) {
            at::autocast::set_autocast_enabled(
                device_type_, previous_enabled_
            );
            at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
        }
    }

private:
    bool enabled_;
    at::DeviceType device_type_;
    bool previous_enabled_ = false;
    at::ScalarType previous_dtype_ = torch::kFloat32;
};

}  // namespace

ValueTargetNetwork::ValueTargetNetwork(
    std::shared_ptr<RepresentationNetworkImpl> representation,
    std::shared_ptr<PredictionNetworkImpl> prediction,
    std::shared_ptr<DynamicsNetworkImpl> dynamics,
    std::int64_t action_space_size,
    std::int64_t support_min,
    std::int64_t support_max,
    std::int64_t chunk_size,
    const std::string& precision,
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
    : representation_(std::move(representation)),
      prediction_(std::move(prediction)),
      dynamics_(std::move(dynamics)),
      support_min_(support_min),
      support_max_(support_max),
      chunk_size_(chunk_size),
      use_bfloat16_(precision == "bf16") {
    if (!representation_ || !prediction_ || !dynamics_) {
        throw std::invalid_argument("target networks must not be null");
    }
    if (chunk_size_ <= 0) {
        throw std::invalid_argument("chunk_size must be positive");
    }
    if (precision != "fp32" && precision != "bf16") {
        throw std::invalid_argument("precision must be fp32 or bf16");
    }
    evaluator_ = std::make_shared<BatchedNetworkEvaluator>(
        dynamics_, prediction_, action_space_size, support_min_, support_max_
    );
    search_ = std::make_shared<TreeSearch>(
        evaluator_,
        num_simulations,
        discount,
        pb_c_init,
        pb_c_base,
        value_delta_max,
        dirichlet_alpha,
        root_exploration_fraction,
        value_prefix_horizon,
        seed + 1,
        search_algorithm,
        num_top_actions,
        c_visit,
        c_scale,
        search_value_mode
    );
    eval();
}

void ValueTargetNetwork::eval() {
    representation_->eval();
    prediction_->eval();
    dynamics_->eval();
}

void ValueTargetNetwork::to(const std::string& device) {
    const torch::Device resolved(device);
    representation_->to(resolved);
    prediction_->to(resolved);
    dynamics_->to(resolved);
}

void ValueTargetNetwork::synchronize(
    const TensorState& representation,
    const TensorState& prediction,
    const TensorState& dynamics
) {
    load_state_dict(*representation_, representation);
    load_state_dict(*prediction_, prediction);
    load_state_dict(*dynamics_, dynamics);
    eval();
}

void ValueTargetNetwork::synchronize(
    const py::dict& representation,
    const py::dict& prediction,
    const py::dict& dynamics
) {
    synchronize(
        tensor_state_from_dict(representation),
        tensor_state_from_dict(prediction),
        tensor_state_from_dict(dynamics)
    );
}

std::tuple<torch::Tensor, torch::Tensor>
ValueTargetNetwork::value_predictions(
    const torch::Tensor& bootstrap_frames,
    const torch::Tensor& bootstrap_mask,
    std::int64_t stack_size
) {
    c10::InferenceMode inference_guard;
    BFloat16AutocastGuard autocast_guard(
        bootstrap_frames, use_bfloat16_
    );
    torch::Tensor positions = torch::nonzero(bootstrap_mask).contiguous();
    if (positions.size(0) == 0) {
        return {
            positions.cpu(),
            torch::empty({0}, torch::kFloat32),
        };
    }
    std::vector<torch::Tensor> value_chunks;
    value_chunks.reserve(
        static_cast<std::size_t>(
            (positions.size(0) + chunk_size_ - 1) / chunk_size_
        )
    );
    using namespace torch::indexing;
    for (std::int64_t start = 0; start < positions.size(0);
         start += chunk_size_) {
        torch::Tensor chunk = positions.index({Slice(
            start, std::min(start + chunk_size_, positions.size(0))
        )});
        torch::Tensor observations = stacked_observations(
            bootstrap_frames, chunk, stack_size
        );
        torch::Tensor states = representation_->forward(observations);
        auto prediction = prediction_->forward(states);
        value_chunks.push_back(categorical_to_scalar(
            std::get<1>(prediction).to(torch::kFloat32),
            support_min_,
            support_max_
        ));
    }
    return {
        positions.cpu().contiguous(),
        torch::cat(value_chunks, 0).cpu().contiguous(),
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ValueTargetNetwork::policy_reanalysis_outputs(
    const torch::Tensor& frames,
    const torch::Tensor& policy_mask,
    std::int64_t stack_size,
    double root_noise_temperature,
    bool gumbel_sampling,
    bool deterministic_ties
) {
    c10::InferenceMode inference_guard;
    BFloat16AutocastGuard autocast_guard(frames, use_bfloat16_);
    torch::Tensor positions = torch::nonzero(policy_mask).contiguous();
    if (positions.size(0) == 0) {
        return {
            positions.cpu(),
            torch::empty({0, 0}, torch::kFloat32),
            torch::empty({0}, torch::kFloat32),
        };
    }
    std::vector<torch::Tensor> policy_chunks;
    std::vector<torch::Tensor> value_chunks;
    policy_chunks.reserve(
        static_cast<std::size_t>(
            (positions.size(0) + chunk_size_ - 1) / chunk_size_
        )
    );
    value_chunks.reserve(policy_chunks.capacity());
    using namespace torch::indexing;
    for (std::int64_t start = 0; start < positions.size(0);
         start += chunk_size_) {
        torch::Tensor chunk = positions.index({Slice(
            start, std::min(start + chunk_size_, positions.size(0))
        )});
        torch::Tensor observations = stacked_observations(
            frames, chunk, stack_size
        );
        torch::Tensor states = representation_->forward(observations);
        auto [policy_logits, value_logits] = prediction_->forward(states);
        torch::Tensor values = categorical_to_scalar(
            value_logits.to(torch::kFloat32), support_min_, support_max_
        );
        auto [search_output, root_values] = search_->search_batch(
            states,
            values,
            policy_logits,
            root_noise_temperature,
            gumbel_sampling,
            deterministic_ties
        );
        policy_chunks.push_back(std::move(search_output));
        value_chunks.push_back(std::move(root_values));
    }
    return {
        positions.cpu().contiguous(),
        torch::cat(policy_chunks, 0).contiguous(),
        torch::cat(value_chunks, 0).contiguous(),
    };
}

torch::Tensor ValueTargetNetwork::stacked_observations(
    const torch::Tensor& frames,
    const torch::Tensor& positions,
    std::int64_t stack_size
) {
    if (stack_size <= 0) {
        throw std::invalid_argument("stack_size must be positive");
    }
    torch::Tensor offsets = torch::arange(
        stack_size,
        torch::TensorOptions().dtype(torch::kLong).device(positions.device())
    );
    torch::Tensor samples = positions.select(1, 0).unsqueeze(1);
    torch::Tensor frame_indices = positions.select(1, 1).unsqueeze(1) + offsets;
    using namespace torch::indexing;
    torch::Tensor selected = frames.index({samples, frame_indices});
    return selected.reshape({
        positions.size(0),
        stack_size * frames.size(2),
        frames.size(3),
        frames.size(4),
    }).to(torch::kFloat32).div_(255.0);
}

}  // namespace atariagent::native
