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
                bool,
                const std::string&,
                int,
                float,
                float,
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
            py::arg("deterministic_ties") = false,
            py::arg("search_algorithm") = "puct",
            py::arg("num_top_actions") = 4,
            py::arg("c_visit") = 50.0F,
            py::arg("c_scale") = 0.1F,
            py::arg("use_gumbel_noise") = true
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
        .def("policy_array", &BatchTree::policy_array)
        .def("selected_actions_array", &BatchTree::selected_actions_array)
        .def("root_values_array", &BatchTree::root_values_array);
}
