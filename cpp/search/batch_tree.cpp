#include "search/batch_tree.h"

#include "native_search.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#ifdef ATARIAGENT_HAS_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace atariagent::native {

class BatchTree::Impl {
public:
    Impl(
        FloatArray root_priors,
        FloatArray root_values,
        FloatArray root_value_prefixes,
        int simulations,
        float discount,
        int value_prefix_horizon,
        float minimum_delta,
        std::uint64_t seed,
        bool deterministic_ties,
        const std::string& search_algorithm,
        int num_top_actions,
        float c_visit,
        float c_scale,
        bool use_gumbel_noise
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

        SearchAlgorithm algorithm;
        if (search_algorithm == "puct" || search_algorithm == "mcts") {
            algorithm = SearchAlgorithm::Puct;
        } else if (search_algorithm == "gumbel") {
            algorithm = SearchAlgorithm::Gumbel;
        } else {
            throw std::invalid_argument(
                "search_algorithm must be puct or gumbel"
            );
        }

        const auto priors_data = root_priors.unchecked<2>();
        const auto values_data = root_values.unchecked<1>();
        const auto prefixes_data = root_value_prefixes.unchecked<1>();
        roots_.reserve(static_cast<std::size_t>(root_count));
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
                seed + static_cast<std::uint64_t>(index),
                deterministic_ties,
                algorithm,
                num_top_actions,
                c_visit,
                c_scale,
                use_gumbel_noise
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
                pb_c_base, pb_c_init
            );
            slots_data(index) = state_slot;
            actions_data(index) = action;
            resets_data(index) = reset;
        }
        return py::make_tuple(state_slots, actions, resets);
    }

    void expand_and_back_up_arrays(
        int state_slot,
        FloatArray value_prefixes,
        FloatArray values,
        FloatArray policy_logits
    ) {
        if (value_prefixes.ndim() != 1 || values.ndim() != 1
            || policy_logits.ndim() != 2
            || value_prefixes.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || values.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || policy_logits.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || policy_logits.shape(1) != action_count_) {
            throw std::invalid_argument("evaluation arrays have invalid shapes");
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
            roots_[index].expand_and_back_up(
                state_slot,
                prefixes_data(index),
                values_data(index),
                std::span<const float>(
                    &policy_data(index, 0),
                    static_cast<std::size_t>(action_count_)
                )
            );
        }
    }

    py::array_t<std::int32_t> visit_counts_array() const {
        py::array_t<std::int32_t> result({
            static_cast<py::ssize_t>(roots_.size()),
            static_cast<py::ssize_t>(action_count_),
        });
        auto output = result.mutable_unchecked<2>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            roots_[index].write_visit_counts(&output(index, 0));
        }
        return result;
    }

    py::array_t<float> policy_array() const {
        py::array_t<float> result({
            static_cast<py::ssize_t>(roots_.size()),
            static_cast<py::ssize_t>(action_count_),
        });
        auto output = result.mutable_unchecked<2>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            roots_[index].write_policy(&output(index, 0));
        }
        return result;
    }

    py::array_t<std::int64_t> selected_actions_array() const {
        py::array_t<std::int64_t> result(
            static_cast<py::ssize_t>(roots_.size())
        );
        auto output = result.mutable_unchecked<1>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            output(index) = roots_[index].selected_action();
        }
        return result;
    }

    py::array_t<float> root_values_array() const {
        py::array_t<float> result(static_cast<py::ssize_t>(roots_.size()));
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

BatchTree::BatchTree(
    FloatArray root_priors,
    FloatArray root_values,
    FloatArray root_value_prefixes,
    int simulations,
    float discount,
    int value_prefix_horizon,
    float minimum_delta,
    std::uint64_t seed,
    bool deterministic_ties,
    const std::string& search_algorithm,
    int num_top_actions,
    float c_visit,
    float c_scale,
    bool use_gumbel_noise
)
    : impl_(std::make_unique<Impl>(
          std::move(root_priors),
          std::move(root_values),
          std::move(root_value_prefixes),
          simulations,
          discount,
          value_prefix_horizon,
          minimum_delta,
          seed,
          deterministic_ties,
          search_algorithm,
          num_top_actions,
          c_visit,
          c_scale,
          use_gumbel_noise
      )) {}

BatchTree::~BatchTree() = default;

py::tuple BatchTree::traverse_arrays(float pb_c_base, float pb_c_init) {
    return impl_->traverse_arrays(pb_c_base, pb_c_init);
}

void BatchTree::expand_and_back_up_arrays(
    int state_slot,
    FloatArray value_prefixes,
    FloatArray values,
    FloatArray policy_logits
) {
    impl_->expand_and_back_up_arrays(
        state_slot,
        std::move(value_prefixes),
        std::move(values),
        std::move(policy_logits)
    );
}

py::array_t<std::int32_t> BatchTree::visit_counts_array() const {
    return impl_->visit_counts_array();
}

py::array_t<float> BatchTree::policy_array() const {
    return impl_->policy_array();
}

py::array_t<std::int64_t> BatchTree::selected_actions_array() const {
    return impl_->selected_actions_array();
}

py::array_t<float> BatchTree::root_values_array() const {
    return impl_->root_values_array();
}

int set_search_num_threads(int count) {
    if (count <= 0) {
        throw std::invalid_argument("thread count must be positive");
    }
#ifdef ATARIAGENT_HAS_OPENMP
    omp_set_num_threads(count);
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace atariagent::native
