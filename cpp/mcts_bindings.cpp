#include "reanalysis/engine.h"
#include "search/batch_tree.h"

#include <pybind11/pybind11.h>

#include <cstdint>

namespace py = pybind11;
using namespace atariagent::native;

PYBIND11_MODULE(_mcts_native, module) {
    module.doc() = "Native EfficientZero search and reanalysis operations";
    module.def(
        "set_num_threads",
        &set_search_num_threads,
        py::arg("count")
    );

    py::class_<BatchTree>(module, "BatchTree")
        .def(
            py::init<
                FloatArray,
                FloatArray,
                FloatArray,
                int,
                float,
                int,
                float,
                std::uint64_t,
                bool
            >(),
            py::arg("root_priors"),
            py::arg("root_values"),
            py::arg("root_value_prefixes"),
            py::arg("simulations"),
            py::arg("discount"),
            py::arg("value_prefix_horizon"),
            py::arg("minimum_delta"),
            py::arg("seed"),
            py::arg("deterministic_ties") = false
        )
        .def(
            "traverse_arrays",
            &BatchTree::traverse_arrays,
            py::arg("pb_c_base"),
            py::arg("pb_c_init")
        )
        .def(
            "expand_and_back_up_arrays",
            &BatchTree::expand_and_back_up_arrays,
            py::arg("state_slot"),
            py::arg("value_prefixes"),
            py::arg("values"),
            py::arg("policy_logits")
        )
        .def("visit_counts_array", &BatchTree::visit_counts_array)
        .def("root_values_array", &BatchTree::root_values_array);

    py::class_<NativeReanalysisEngine>(module, "NativeReanalysisEngine")
        .def(
            py::init<py::object, int, double, int, bool>(),
            py::arg("target"),
            py::arg("prefetch_batches"),
            py::arg("timeout_seconds"),
            py::arg("target_update_interval"),
            py::arg("cache_targets")
        )
        .def(
            "publish_weights",
            &NativeReanalysisEngine::publish_weights,
            py::arg("version"),
            py::arg("representation"),
            py::arg("prediction"),
            py::arg("dynamics")
        )
        .def("submit", &NativeReanalysisEngine::submit, py::arg("batch"))
        .def("wait_next", &NativeReanalysisEngine::wait_next)
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
}
