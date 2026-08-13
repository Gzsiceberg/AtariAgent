#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#ifdef ATARIAGENT_HAS_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<
    float,
    py::array::c_style | py::array::forcecast
>;

struct MinMaxStats {
    explicit MinMaxStats(float minimum_delta)
        : minimum_delta(minimum_delta) {}

    float minimum_delta;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();

    void clear() {
        minimum = std::numeric_limits<float>::infinity();
        maximum = -std::numeric_limits<float>::infinity();
    }

    void update(float value) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    float normalize(float value) const {
        const float delta = maximum - minimum;
        if (delta > 0.0F) {
            value = (value - minimum) / std::max(delta, minimum_delta);
        }
        return std::clamp(value, 0.0F, 1.0F);
    }
};

struct Node {
    float prior = 0.0F;
    int parent = -1;
    int action = -1;
    int depth = 0;
    int state_slot = -1;
    float value_prefix = 0.0F;
    bool reset_value_prefix = false;
    int visit_count = 0;
    float value_sum = 0.0F;
    std::vector<int> children;

    bool expanded() const { return !children.empty(); }

    float value() const {
        return visit_count == 0
            ? 0.0F
            : value_sum / static_cast<float>(visit_count);
    }
};

class RootTree {
public:
    RootTree(
        const std::vector<float>& root_priors,
        float root_value,
        float root_value_prefix,
        int simulations,
        float discount,
        int value_prefix_horizon,
        float minimum_delta,
        std::uint64_t seed,
        bool deterministic_ties
    )
        : action_count_(static_cast<int>(root_priors.size())),
          discount_(discount),
          value_prefix_horizon_(value_prefix_horizon),
          stats_(minimum_delta),
          rng_(seed),
          deterministic_ties_(deterministic_ties) {
        if (action_count_ <= 0) {
            throw std::invalid_argument("root policy must not be empty");
        }
        nodes_.reserve(
            1 + action_count_ * (std::max(simulations, 0) + 1)
        );
        Node root;
        root.prior = 1.0F;
        root.state_slot = 0;
        root.value_prefix = root_value_prefix;
        root.visit_count = 1;
        root.value_sum = root_value;
        nodes_.push_back(std::move(root));
        expand(0, 0, root_value_prefix, root_priors, false, true);
    }

    std::tuple<int, int, bool> traverse(
        float pb_c_base,
        float pb_c_init
    ) {
        path_.clear();
        int node_index = 0;
        float parent_mean_q = 0.0F;
        path_.push_back(node_index);

        while (nodes_[node_index].expanded()) {
            const float mean_q = node_mean_q(
                node_index,
                parent_mean_q,
                node_index == 0
            );
            node_index = select_child(
                node_index,
                mean_q,
                pb_c_base,
                pb_c_init
            );
            path_.push_back(node_index);
            parent_mean_q = mean_q;
        }

        const Node& leaf = nodes_[node_index];
        if (leaf.parent < 0 || leaf.action < 0) {
            throw std::runtime_error("selection did not reach a child leaf");
        }
        const Node& parent = nodes_[leaf.parent];
        return {parent.state_slot, leaf.action, parent.reset_value_prefix};
    }

    void expand_and_back_up(
        int state_slot,
        float value_prefix,
        float value,
        const std::vector<float>& policy_logits
    ) {
        if (path_.empty()) {
            throw std::runtime_error("traverse must precede expansion");
        }
        const int leaf_index = path_.back();
        const bool reset = (
            nodes_[leaf_index].depth % value_prefix_horizon_ == 0
        );
        expand(
            leaf_index,
            state_slot,
            value_prefix,
            softmax(policy_logits),
            reset,
            false
        );
        back_up(value);
        rebuild_min_max();
    }

    std::vector<int> visit_counts() const {
        std::vector<int> counts;
        counts.reserve(action_count_);
        for (const int child_index : nodes_[0].children) {
            counts.push_back(nodes_[child_index].visit_count);
        }
        return counts;
    }

    float root_value() const { return nodes_[0].value(); }

private:
    int action_count_;
    float discount_;
    int value_prefix_horizon_;
    MinMaxStats stats_;
    std::mt19937_64 rng_;
    bool deterministic_ties_;
    std::vector<Node> nodes_;
    std::vector<int> path_;

    static std::vector<float> softmax(
        const std::vector<float>& logits
    ) {
        if (logits.empty()) {
            throw std::invalid_argument("policy logits must not be empty");
        }
        const float maximum = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probabilities;
        probabilities.reserve(logits.size());
        float total = 0.0F;
        for (const float logit : logits) {
            if (!std::isfinite(logit)) {
                throw std::invalid_argument("policy logits must be finite");
            }
            const float probability = std::exp(logit - maximum);
            probabilities.push_back(probability);
            total += probability;
        }
        for (float& probability : probabilities) {
            probability /= total;
        }
        return probabilities;
    }

    void expand(
        int node_index,
        int state_slot,
        float value_prefix,
        const std::vector<float>& priors,
        bool reset,
        bool priors_are_probabilities
    ) {
        if (static_cast<int>(priors.size()) != action_count_) {
            throw std::invalid_argument("policy action count changed during search");
        }
        if (!std::isfinite(value_prefix)) {
            throw std::invalid_argument("value prefix must be finite");
        }
        if (nodes_[node_index].expanded()) {
            throw std::runtime_error("cannot expand a node more than once");
        }

        if (priors_are_probabilities) {
            float total = 0.0F;
            for (const float prior : priors) {
                if (!std::isfinite(prior) || prior < 0.0F) {
                    throw std::invalid_argument(
                        "root priors must be finite and non-negative"
                    );
                }
                total += prior;
            }
            if (!(total > 0.0F)) {
                throw std::invalid_argument("root priors must have positive mass");
            }
        }

        nodes_[node_index].state_slot = state_slot;
        nodes_[node_index].value_prefix = value_prefix;
        nodes_[node_index].reset_value_prefix = reset;
        nodes_[node_index].children.reserve(action_count_);
        const int child_depth = nodes_[node_index].depth + 1;
        for (int action = 0; action < action_count_; ++action) {
            Node child;
            child.prior = priors[action];
            child.parent = node_index;
            child.action = action;
            child.depth = child_depth;
            nodes_.push_back(std::move(child));
            nodes_[node_index].children.push_back(
                static_cast<int>(nodes_.size()) - 1
            );
        }
    }

    float reward(int node_index) const {
        const Node& node = nodes_[node_index];
        if (node.parent < 0) {
            return 0.0F;
        }
        const Node& parent = nodes_[node.parent];
        if (parent.reset_value_prefix) {
            return node.value_prefix;
        }
        return node.value_prefix - parent.value_prefix;
    }

    float q_value(int node_index) const {
        return reward(node_index) + discount_ * nodes_[node_index].value();
    }

    float node_mean_q(
        int node_index,
        float parent_mean_q,
        bool is_root
    ) const {
        float sum = 0.0F;
        int count = 0;
        for (const int child_index : nodes_[node_index].children) {
            if (nodes_[child_index].visit_count > 0) {
                sum += q_value(child_index);
                ++count;
            }
        }
        if (is_root) {
            return count == 0 ? 0.0F : sum / static_cast<float>(count);
        }
        return (parent_mean_q + sum) / static_cast<float>(1 + count);
    }

    int select_child(
        int node_index,
        float mean_q,
        float pb_c_base,
        float pb_c_init
    ) {
        int child_visits = 0;
        for (const int child_index : nodes_[node_index].children) {
            child_visits += nodes_[child_index].visit_count;
        }
        const float exploration_scale = pb_c_init + std::log(
            (static_cast<float>(child_visits) + pb_c_base + 1.0F)
            / pb_c_base
        );
        const float sqrt_visits = std::sqrt(static_cast<float>(child_visits));

        float best_score = -std::numeric_limits<float>::infinity();
        std::vector<int> best_children;
        for (const int child_index : nodes_[node_index].children) {
            const Node& child = nodes_[child_index];
            const float prior_score = child.prior * sqrt_visits
                / static_cast<float>(1 + child.visit_count)
                * exploration_scale;
            const float q = child.visit_count > 0
                ? q_value(child_index)
                : mean_q;
            const float score = stats_.normalize(q) + prior_score;
            if (score > best_score + 1.0e-12F) {
                best_score = score;
                best_children.assign(1, child_index);
            } else if (std::abs(score - best_score) <= 1.0e-12F) {
                best_children.push_back(child_index);
            }
        }
        if (best_children.empty()) {
            throw std::runtime_error("expanded node has no selectable child");
        }
        if (deterministic_ties_) {
            return best_children.front();
        }
        std::uniform_int_distribution<std::size_t> distribution(
            0,
            best_children.size() - 1
        );
        return best_children[distribution(rng_)];
    }

    void back_up(float leaf_value) {
        if (!std::isfinite(leaf_value)) {
            throw std::invalid_argument("leaf value must be finite");
        }
        float bootstrap = leaf_value;
        for (auto iterator = path_.rbegin(); iterator != path_.rend(); ++iterator) {
            Node& node = nodes_[*iterator];
            node.value_sum += bootstrap;
            ++node.visit_count;
            bootstrap = reward(*iterator) + discount_ * bootstrap;
        }
    }

    void rebuild_min_max() {
        stats_.clear();
        std::vector<int> stack{0};
        while (!stack.empty()) {
            const int node_index = stack.back();
            stack.pop_back();
            for (const int child_index : nodes_[node_index].children) {
                if (nodes_[child_index].visit_count > 0) {
                    stats_.update(q_value(child_index));
                    stack.push_back(child_index);
                }
            }
        }
    }
};

class BatchTree {
public:
    BatchTree(
        FloatArray root_priors,
        FloatArray root_values,
        FloatArray root_value_prefixes,
        int simulations,
        float discount,
        int value_prefix_horizon,
        float minimum_delta,
        std::uint64_t seed,
        bool deterministic_ties
    ) {
        if (root_priors.ndim() != 2
            || root_values.ndim() != 1
            || root_value_prefixes.ndim() != 1) {
            throw std::invalid_argument(
                "root priors must be 2D and root values must be 1D"
            );
        }
        const py::ssize_t root_count = root_priors.shape(0);
        if (root_count <= 0) {
            throw std::invalid_argument("at least one root is required");
        }
        if (root_values.shape(0) != root_count
            || root_value_prefixes.shape(0) != root_count) {
            throw std::invalid_argument("root arrays must have equal lengths");
        }
        action_count_ = static_cast<int>(root_priors.shape(1));
        if (action_count_ <= 0) {
            throw std::invalid_argument("root policy must not be empty");
        }
        if (simulations <= 0) {
            throw std::invalid_argument("simulations must be positive");
        }
        if (value_prefix_horizon <= 0) {
            throw std::invalid_argument("value prefix horizon must be positive");
        }

        const auto priors_data = root_priors.unchecked<2>();
        const auto values_data = root_values.unchecked<1>();
        const auto prefixes_data = root_value_prefixes.unchecked<1>();
        roots_.reserve(static_cast<std::size_t>(root_count));
        std::mt19937_64 seed_generator(seed);
        for (py::ssize_t index = 0; index < root_count; ++index) {
            if (!std::isfinite(values_data(index))
                || !std::isfinite(prefixes_data(index))) {
                throw std::invalid_argument("root values must be finite");
            }
            std::vector<float> priors(action_count_);
            for (int action = 0; action < action_count_; ++action) {
                const float prior = priors_data(index, action);
                if (!std::isfinite(prior) || prior < 0.0F) {
                    throw std::invalid_argument(
                        "root priors must be finite and non-negative"
                    );
                }
                priors[action] = prior;
            }
            roots_.emplace_back(
                priors,
                values_data(index),
                prefixes_data(index),
                simulations,
                discount,
                value_prefix_horizon,
                minimum_delta,
                seed_generator(),
                deterministic_ties
            );
        }
    }

    py::tuple traverse_arrays(float pb_c_base, float pb_c_init) {
        if (!std::isfinite(pb_c_base) || pb_c_base <= 0.0F
            || !std::isfinite(pb_c_init) || pb_c_init < 0.0F) {
            throw std::invalid_argument("invalid exploration constants");
        }
        const py::ssize_t root_count = static_cast<py::ssize_t>(roots_.size());
        py::array_t<std::int64_t> state_slots(root_count);
        py::array_t<std::int64_t> actions(root_count);
        py::array_t<bool> resets(root_count);
        auto slots_data = state_slots.mutable_unchecked<1>();
        auto actions_data = actions.mutable_unchecked<1>();
        auto resets_data = resets.mutable_unchecked<1>();
        #pragma omp parallel for if(root_count >= 32)
        for (py::ssize_t index = 0; index < root_count; ++index) {
            auto [state_slot, action, reset] = roots_[index].traverse(
                pb_c_base,
                pb_c_init
            );
            slots_data(index) = state_slot;
            actions_data(index) = action;
            resets_data(index) = reset;
        }
        return py::make_tuple(state_slots, actions, resets);
    }

    void expand_and_back_up_arrays(
        int state_slot,
        py::array_t<float, py::array::c_style | py::array::forcecast>
            value_prefixes,
        py::array_t<float, py::array::c_style | py::array::forcecast> values,
        py::array_t<float, py::array::c_style | py::array::forcecast>
            policy_logits
    ) {
        if (value_prefixes.ndim() != 1 || values.ndim() != 1
            || policy_logits.ndim() != 2
            || value_prefixes.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || values.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || policy_logits.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || policy_logits.shape(1) != action_count_) {
            throw std::invalid_argument(
                "evaluation arrays have invalid shapes"
            );
        }
        auto prefixes_data = value_prefixes.unchecked<1>();
        auto values_data = values.unchecked<1>();
        auto policy_data = policy_logits.unchecked<2>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            if (!std::isfinite(prefixes_data(index))
                || !std::isfinite(values_data(index))) {
                throw std::invalid_argument("evaluation values must be finite");
            }
            for (int action = 0; action < action_count_; ++action) {
                if (!std::isfinite(policy_data(index, action))) {
                    throw std::invalid_argument("policy logits must be finite");
                }
            }
        }
        #pragma omp parallel for if(roots_.size() >= 32)
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(roots_.size());
             ++index) {
            std::vector<float> policy(action_count_);
            for (int action = 0; action < action_count_; ++action) {
                policy[action] = policy_data(index, action);
            }
            roots_[index].expand_and_back_up(
                state_slot,
                prefixes_data(index),
                values_data(index),
                policy
            );
        }
    }

    py::array_t<std::int32_t> visit_counts_array() const {
        py::array_t<std::int32_t> result(
            {static_cast<py::ssize_t>(roots_.size()),
             static_cast<py::ssize_t>(action_count_)}
        );
        auto output = result.mutable_unchecked<2>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            const std::vector<int> counts = roots_[index].visit_counts();
            for (int action = 0; action < action_count_; ++action) {
                output(index, action) = counts[action];
            }
        }
        return result;
    }

    py::array_t<float> root_values_array() const {
        py::array_t<float> result(
            static_cast<py::ssize_t>(roots_.size())
        );
        auto output = result.mutable_unchecked<1>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            output(index) = roots_[index].root_value();
        }
        return result;
    }

private:
    int action_count_ = 0;
    std::vector<RootTree> roots_;
};

}  // namespace

PYBIND11_MODULE(_mcts_native, module) {
    module.doc() = "Native batched EfficientZero MCTS tree operations";
    module.def("set_num_threads", [](int count) {
        if (count <= 0) {
            throw std::invalid_argument("thread count must be positive");
        }
#ifdef ATARIAGENT_HAS_OPENMP
        omp_set_num_threads(count);
        return omp_get_max_threads();
#else
        return 1;
#endif
    });
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
}
