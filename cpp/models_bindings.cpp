#include <torch/extension.h>

#include "inference/evaluator.h"
#include "inference/tree_search.h"
#include "inference/value_target.h"
#include "models/dynamics.h"
#include "models/prediction.h"
#include "models/representation.h"
#include "models/state_dict.h"
#include "reanalysis/engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace py = pybind11;
using namespace atariagent::native;

namespace {

template <typename ModuleType>
void add_inference_module_methods(
    py::class_<ModuleType, std::shared_ptr<ModuleType>>& binding
) {
    binding
        .def("eval", [](ModuleType& self) -> ModuleType& {
            self.eval();
            return self;
        }, py::return_value_policy::reference_internal)
        .def("train", [](ModuleType& self, bool mode) -> ModuleType& {
            self.train(mode);
            return self;
        }, py::arg("mode") = true, py::return_value_policy::reference_internal)
        .def("to", [](ModuleType& self, const std::string& device) -> ModuleType& {
            self.to(torch::Device(device));
            return self;
        }, py::arg("device"), py::return_value_policy::reference_internal)
        .def("load_state_dict", [](ModuleType& self, const py::dict& state) {
            load_state_dict(self, state);
        }, py::arg("state_dict"))
        .def("state_dict", [](ModuleType& self) {
            return state_dict(self);
        });
}

}  // namespace

PYBIND11_MODULE(_models_native, module) {
    module.doc() = "Inference-only LibTorch implementations of AtariAgent";
    module.def(
        "set_tree_search_num_threads",
        &set_tree_search_num_threads,
        py::arg("count")
    );
    module.attr("set_mcts_num_threads") =
        module.attr("set_tree_search_num_threads");

    auto residual = py::class_<ResidualBlockImpl, std::shared_ptr<ResidualBlockImpl>>(
        module, "ResidualBlock"
    );
    residual
        .def(
            py::init<
                std::int64_t,
                std::optional<std::int64_t>,
                std::int64_t,
                double
            >(),
            py::arg("in_channels"),
            py::arg("out_channels") = py::none(),
            py::arg("stride") = 1,
            py::arg("batch_norm_momentum") = 0.1
        )
        .def("forward", &ResidualBlockImpl::forward)
        .def("__call__", &ResidualBlockImpl::forward);
    add_inference_module_methods(residual);

    auto representation = py::class_<
        RepresentationNetworkImpl,
        std::shared_ptr<RepresentationNetworkImpl>
    >(module, "RepresentationNetwork");
    representation
        .def(
            py::init<std::int64_t, double>(),
            py::arg("in_channels"),
            py::arg("batch_norm_momentum") = 0.1
        )
        .def("forward", &RepresentationNetworkImpl::forward)
        .def("__call__", &RepresentationNetworkImpl::forward);
    add_inference_module_methods(representation);

    auto reward = py::class_<
        RewardPredictionNetworkImpl,
        std::shared_ptr<RewardPredictionNetworkImpl>
    >(module, "RewardPredictionNetwork");
    reward
        .def(
            py::init<double>(),
            py::arg("batch_norm_momentum") = 0.1
        )
        .def(
            "forward",
            &RewardPredictionNetworkImpl::forward,
            py::arg("state"),
            py::arg("hidden") = py::none()
        )
        .def(
            "__call__",
            &RewardPredictionNetworkImpl::forward,
            py::arg("state"),
            py::arg("hidden") = py::none()
        );
    add_inference_module_methods(reward);

    auto dynamics = py::class_<DynamicsNetworkImpl, std::shared_ptr<DynamicsNetworkImpl>>(
        module, "DynamicsNetwork"
    );
    dynamics
        .def(
            py::init<std::int64_t, double, bool, std::int64_t>(),
            py::arg("action_space_size"),
            py::arg("batch_norm_momentum") = 0.1,
            py::arg("scale_state_gradient") = true,
            py::arg("action_embedding_dim") = 16
        )
        .def(
            "forward",
            &DynamicsNetworkImpl::forward,
            py::arg("state"),
            py::arg("action"),
            py::arg("reward_hidden") = py::none()
        )
        .def(
            "__call__",
            &DynamicsNetworkImpl::forward,
            py::arg("state"),
            py::arg("action"),
            py::arg("reward_hidden") = py::none()
        );
    add_inference_module_methods(dynamics);

    auto policy = py::class_<PolicyNetworkImpl, std::shared_ptr<PolicyNetworkImpl>>(
        module, "PolicyNetwork"
    );
    policy
        .def(
            py::init<std::int64_t, double>(),
            py::arg("action_space_size"),
            py::arg("batch_norm_momentum") = 0.1
        )
        .def("forward", &PolicyNetworkImpl::forward)
        .def("__call__", &PolicyNetworkImpl::forward);
    add_inference_module_methods(policy);

    auto value = py::class_<ValueNetworkImpl, std::shared_ptr<ValueNetworkImpl>>(
        module, "ValueNetwork"
    );
    value
        .def(
            py::init<std::int64_t, double>(),
            py::arg("support_size") = ValueNetworkImpl::default_support_size,
            py::arg("batch_norm_momentum") = 0.1
        )
        .def("forward", &ValueNetworkImpl::forward)
        .def("__call__", &ValueNetworkImpl::forward);
    add_inference_module_methods(value);

    auto prediction = py::class_<
        PredictionNetworkImpl,
        std::shared_ptr<PredictionNetworkImpl>
    >(module, "PredictionNetwork");
    prediction
        .def(
            py::init<std::int64_t, std::int64_t, double>(),
            py::arg("action_space_size"),
            py::arg("value_support_size") = ValueNetworkImpl::default_support_size,
            py::arg("batch_norm_momentum") = 0.1
        )
        .def("forward", &PredictionNetworkImpl::forward)
        .def("__call__", &PredictionNetworkImpl::forward);
    add_inference_module_methods(prediction);

    auto evaluator = py::class_<
        BatchedNetworkEvaluator,
        std::shared_ptr<BatchedNetworkEvaluator>
    >(module, "BatchedNetworkEvaluator");
    evaluator
        .def(
            py::init<
                std::shared_ptr<DynamicsNetworkImpl>,
                std::shared_ptr<PredictionNetworkImpl>,
                std::int64_t,
                std::int64_t,
                std::int64_t
            >(),
            py::arg("dynamics"),
            py::arg("prediction"),
            py::arg("action_space_size"),
            py::arg("support_min") = -300,
            py::arg("support_max") = 300
        )
        .def(
            "initial_hidden",
            &BatchedNetworkEvaluator::initial_hidden,
            py::arg("batch_size"),
            py::arg("state_template")
        )
        .def(
            "evaluate_tensors",
            &BatchedNetworkEvaluator::evaluate_tensors,
            py::arg("states"),
            py::arg("actions"),
            py::arg("value_prefix_hidden"),
            py::arg("reset_value_prefix") = py::none()
        )
        .def(
            "validate_policy",
            &BatchedNetworkEvaluator::validate_policy,
            py::arg("policy_logits"),
            py::arg("batch_size")
        );

    auto tree_search = py::class_<
        TreeSearch,
        std::shared_ptr<TreeSearch>
    >(module, "TreeSearch");
    tree_search
        .def(
            py::init<
                std::shared_ptr<BatchedNetworkEvaluator>,
                std::int64_t,
                double,
                double,
                double,
                double,
                double,
                double,
                std::int64_t,
                std::uint64_t,
                const std::string&,
                std::int64_t,
                double,
                double
            >(),
            py::arg("evaluator"),
            py::arg("num_simulations") = 50,
            py::arg("discount") = 0.997,
            py::arg("pb_c_init") = 1.25,
            py::arg("pb_c_base") = 19652.0,
            py::arg("value_delta_max") = 0.01,
            py::arg("dirichlet_alpha") = 0.3,
            py::arg("root_exploration_fraction") = 0.25,
            py::arg("value_prefix_horizon") = 5,
            py::arg("seed") = 0,
            py::arg("search_algorithm") = "puct",
            py::arg("num_top_actions") = 4,
            py::arg("c_visit") = 50.0,
            py::arg("c_scale") = 0.1
        )
        .def(
            "search_batch",
            &TreeSearch::search_batch,
            py::arg("root_states"),
            py::arg("root_values"),
            py::arg("root_policy_logits"),
            py::arg("root_noise_temperature") = 0.0,
            py::arg("gumbel_sampling") = false,
            py::arg("deterministic_ties") = false
        );
    module.attr("MCTS") = module.attr("TreeSearch");

    auto target = py::class_<
        ValueTargetNetwork,
        std::shared_ptr<ValueTargetNetwork>
    >(module, "ValueTargetNetwork");
    target
        .def(
            py::init<
                std::shared_ptr<RepresentationNetworkImpl>,
                std::shared_ptr<PredictionNetworkImpl>,
                std::shared_ptr<DynamicsNetworkImpl>,
                std::int64_t,
                std::int64_t,
                std::int64_t,
                std::int64_t,
                std::string,
                std::int64_t,
                double,
                double,
                double,
                double,
                double,
                double,
                std::int64_t,
                std::uint64_t,
                const std::string&,
                std::int64_t,
                double,
                double
            >(),
            py::arg("representation"),
            py::arg("prediction"),
            py::arg("dynamics"),
            py::arg("action_space_size"),
            py::arg("support_min") = -300,
            py::arg("support_max") = 300,
            py::arg("chunk_size") = 768,
            py::arg("precision") = "fp32",
            py::arg("num_simulations") = 50,
            py::arg("discount") = 0.997,
            py::arg("pb_c_init") = 1.25,
            py::arg("pb_c_base") = 19652.0,
            py::arg("value_delta_max") = 0.01,
            py::arg("dirichlet_alpha") = 0.3,
            py::arg("root_exploration_fraction") = 0.25,
            py::arg("value_prefix_horizon") = 5,
            py::arg("seed") = 0,
            py::arg("search_algorithm") = "puct",
            py::arg("num_top_actions") = 4,
            py::arg("c_visit") = 50.0,
            py::arg("c_scale") = 0.1
        )
        .def("eval", [](ValueTargetNetwork& self) -> ValueTargetNetwork& {
            self.eval();
            return self;
        }, py::return_value_policy::reference_internal)
        .def("to", [](ValueTargetNetwork& self, const std::string& device)
            -> ValueTargetNetwork& {
            self.to(device);
            return self;
        }, py::arg("device"), py::return_value_policy::reference_internal)
        .def(
            "synchronize",
            [](ValueTargetNetwork& self,
               const py::dict& representation,
               const py::dict& prediction,
               const py::dict& dynamics) {
                self.synchronize(representation, prediction, dynamics);
            },
            py::arg("representation"),
            py::arg("prediction"),
            py::arg("dynamics")
        )
        .def(
            "reanalyze_values",
            &ValueTargetNetwork::reanalyze_values,
            py::arg("bootstrap_frames"),
            py::arg("bootstrap_mask"),
            py::arg("mcts_bootstrap_mask"),
            py::arg("stored_bootstrap_values"),
            py::arg("bootstrap_discounts"),
            py::arg("value_targets"),
            py::arg("stack_size"),
            py::arg("root_noise_temperature") = 0.0,
            py::arg("gumbel_sampling") = false
        );

    py::class_<NativeReanalysisEngine>(module, "NativeReanalysisEngine")
        .def(
            py::init<
                std::shared_ptr<ValueTargetNetwork>,
                const std::string&,
                int,
                double,
                int,
                bool,
                std::int64_t
            >(),
            py::arg("target"),
            py::arg("device"),
            py::arg("prefetch_batches"),
            py::arg("timeout_seconds"),
            py::arg("target_update_interval"),
            py::arg("cache_targets"),
            py::arg("cache_target_ttl")
        )
        .def(
            "publish_weights",
            &NativeReanalysisEngine::publish_weights,
            py::arg("version"),
            py::arg("representation"),
            py::arg("prediction"),
            py::arg("dynamics")
        )
        .def(
            "submit",
            &NativeReanalysisEngine::submit,
            py::arg("batch"),
            py::arg("root_noise_temperature"),
            py::arg("gumbel_sampling"),
            py::arg("trained_step"),
            py::arg("use_mcts_bootstrap") = false
        )
        .def("wait_next", &NativeReanalysisEngine::wait_next)
        .def("clear_cache", &NativeReanalysisEngine::clear_cache)
        .def("close", &NativeReanalysisEngine::close)
        .def_property_readonly(
            "pending_count", &NativeReanalysisEngine::pending_count
        )
        .def_property_readonly(
            "max_pending", &NativeReanalysisEngine::max_pending
        )
        .def_property_readonly(
            "cache_size", &NativeReanalysisEngine::cache_size
        )
        .def_property_readonly(
            "weight_version", &NativeReanalysisEngine::weight_version
        );

    module.def(
        "categorical_to_scalar",
        &categorical_to_scalar,
        py::arg("logits"),
        py::arg("support_min") = -300,
        py::arg("support_max") = 300,
        py::arg("epsilon") = 0.001
    );
}
