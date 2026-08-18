#include "native_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace atariagent::native {

MinMaxStats::MinMaxStats(float minimum_delta)
    : minimum_delta_(minimum_delta),
      minimum_(std::numeric_limits<float>::infinity()),
      maximum_(-std::numeric_limits<float>::infinity()) {}

void MinMaxStats::clear() {
    minimum_ = std::numeric_limits<float>::infinity();
    maximum_ = -std::numeric_limits<float>::infinity();
}

void MinMaxStats::update(float value) {
    minimum_ = std::min(minimum_, value);
    maximum_ = std::max(maximum_, value);
}

float MinMaxStats::normalize(float value) const {
    const float delta = maximum_ - minimum_;
    if (delta > 0.0F) {
        value = (value - minimum_) / std::max(delta, minimum_delta_);
    }
    return std::clamp(value, 0.0F, 1.0F);
}

bool SearchNode::expanded() const { return first_child >= 0; }

float SearchNode::value() const {
    return visit_count == 0
        ? 0.0F
        : value_sum / static_cast<float>(visit_count);
}

RootTree::RootTree(
    const std::vector<float>& root_priors,
    float root_value,
    float root_value_prefix,
    int simulations,
    float discount,
    int value_prefix_horizon,
    float minimum_delta,
    std::uint64_t seed,
    bool deterministic_ties,
    SearchAlgorithm algorithm,
    int num_top_actions,
    float c_visit,
    float c_scale,
    bool use_gumbel_noise
)
    : action_count_(static_cast<int>(root_priors.size())),
      discount_(discount),
      value_prefix_horizon_(value_prefix_horizon),
      stats_(minimum_delta),
      rng_(seed),
      deterministic_ties_(deterministic_ties),
      algorithm_(algorithm),
      num_simulations_(simulations),
      num_top_actions_(num_top_actions),
      c_visit_(c_visit),
      c_scale_(c_scale) {
    if (action_count_ <= 0) {
        throw std::invalid_argument("root policy must not be empty");
    }
    if (algorithm_ == SearchAlgorithm::Gumbel) {
        if (num_top_actions_ < 2 || num_top_actions_ > action_count_
            || (num_top_actions_ & (num_top_actions_ - 1)) != 0) {
            throw std::invalid_argument(
                "Gumbel num_top_actions must be a power of two in [2, actions]"
            );
        }
        if (simulations < num_top_actions_ || !std::isfinite(c_visit_)
            || c_visit_ < 0.0F || !std::isfinite(c_scale_)
            || c_scale_ <= 0.0F) {
            throw std::invalid_argument("invalid Gumbel search configuration");
        }
    }
    nodes_.reserve(1 + action_count_ * (std::max(simulations, 0) + 1));
    const auto scratch_capacity = static_cast<std::size_t>(
        std::max(simulations, 0) + 2
    );
    path_.reserve(scratch_capacity);
    min_max_stack_.reserve(scratch_capacity);
    SearchNode root;
    root.prior = 1.0F;
    root.state_slot = 0;
    root.value_prefix = root_value_prefix;
    root.visit_count = 1;
    root.value_sum = root_value;
    nodes_.push_back(std::move(root));
    expand_probabilities(0, 0, root_value_prefix, root_priors, false);

    if (algorithm_ == SearchAlgorithm::Gumbel) {
        gumbels_.resize(action_count_, 0.0F);
        if (use_gumbel_noise) {
            std::uint64_t state = seed;
            for (float& gumbel : gumbels_) {
                state += 0x9e3779b97f4a7c15ULL;
                std::uint64_t value = state;
                value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
                value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
                value ^= value >> 31U;
                const double uniform = (
                    static_cast<double>(value >> 11U) + 0.5
                ) * (1.0 / 9007199254740992.0);
                gumbel = static_cast<float>(-std::log(-std::log(uniform)));
            }
        }
        initialize_gumbel_candidates();
        current_num_top_actions_ = num_top_actions_;
        const double phases = std::log2(static_cast<double>(num_top_actions_));
        const int visits_per_action = std::max(
            static_cast<int>(std::floor(
                simulations / (phases * num_top_actions_)
            )),
            1
        );
        phase_visit_threshold_ = std::min(
            visits_per_action * num_top_actions_, simulations
        );
    }
}

std::tuple<int, int, bool> RootTree::traverse(
    float pb_c_base,
    float pb_c_init
) {
    path_.clear();
    int node_index = 0;
    float parent_mean_q = 0.0F;
    path_.push_back(node_index);
    while (nodes_[node_index].expanded()) {
        const float mean_q = node_mean_q(
            node_index, parent_mean_q, node_index == 0
        );
        node_index = algorithm_ == SearchAlgorithm::Gumbel
            ? select_gumbel_child(node_index)
            : select_child(node_index, mean_q, pb_c_base, pb_c_init);
        path_.push_back(node_index);
        parent_mean_q = mean_q;
    }
    const SearchNode& leaf = nodes_[node_index];
    if (leaf.parent < 0 || leaf.action < 0) {
        throw std::runtime_error("selection did not reach a child leaf");
    }
    const SearchNode& parent = nodes_[leaf.parent];
    return {parent.state_slot, leaf.action, parent.reset_value_prefix};
}

void RootTree::expand_and_back_up(
    int state_slot,
    float value_prefix,
    float value,
    std::span<const float> policy_logits
) {
    if (path_.empty()) {
        throw std::runtime_error("traverse must precede expansion");
    }
    const int leaf_index = path_.back();
    const bool reset = nodes_[leaf_index].depth % value_prefix_horizon_ == 0;
    expand_logits(leaf_index, state_slot, value_prefix, policy_logits, reset);
    back_up(value);
    rebuild_min_max();
    if (algorithm_ == SearchAlgorithm::Gumbel) {
        ++completed_simulations_;
        advance_gumbel_phase();
    }
}

void RootTree::write_visit_counts(std::int32_t* output) const {
    const int first_child = nodes_[0].first_child;
    for (int action = 0; action < action_count_; ++action) {
        output[action] = nodes_[first_child + action].visit_count;
    }
}

void RootTree::write_policy(float* output) const {
    if (algorithm_ == SearchAlgorithm::Gumbel) {
        const std::vector<float> policy = improved_policy(0);
        std::copy(policy.begin(), policy.end(), output);
        return;
    }
    const int first_child = nodes_[0].first_child;
    float total = 0.0F;
    for (int action = 0; action < action_count_; ++action) {
        total += static_cast<float>(nodes_[first_child + action].visit_count);
    }
    for (int action = 0; action < action_count_; ++action) {
        output[action] = static_cast<float>(nodes_[first_child + action].visit_count)
            / total;
    }
}

int RootTree::selected_action() const {
    if (algorithm_ == SearchAlgorithm::Gumbel) {
        if (selected_root_actions_.empty()) {
            throw std::runtime_error("Gumbel search has no selected action");
        }
        return selected_root_actions_.front();
    }
    const int first_child = nodes_[0].first_child;
    int best_action = 0;
    for (int action = 1; action < action_count_; ++action) {
        if (nodes_[first_child + action].visit_count
            > nodes_[first_child + best_action].visit_count) {
            best_action = action;
        }
    }
    return best_action;
}

float RootTree::root_value() const { return nodes_[0].value(); }

void RootTree::expand_probabilities(
    int node_index,
    int state_slot,
    float value_prefix,
    std::span<const float> priors,
    bool reset
) {
    if (static_cast<int>(priors.size()) != action_count_) {
        throw std::invalid_argument("policy action count changed during search");
    }
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
    initialize_children(node_index, state_slot, value_prefix, reset);
    const int child_depth = nodes_[node_index].depth + 1;
    for (int action = 0; action < action_count_; ++action) {
        append_child(node_index, action, priors[action], child_depth);
    }
}

void RootTree::expand_logits(
    int node_index,
    int state_slot,
    float value_prefix,
    std::span<const float> logits,
    bool reset
) {
    if (static_cast<int>(logits.size()) != action_count_) {
        throw std::invalid_argument("policy action count changed during search");
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("policy logits must be finite");
        }
        maximum = std::max(maximum, logit);
    }
    float total = 0.0F;
    for (const float logit : logits) {
        total += std::exp(logit - maximum);
    }
    initialize_children(node_index, state_slot, value_prefix, reset);
    const int child_depth = nodes_[node_index].depth + 1;
    for (int action = 0; action < action_count_; ++action) {
        append_child(
            node_index,
            action,
            std::exp(logits[action] - maximum) / total,
            child_depth
        );
    }
}

void RootTree::initialize_children(
    int node_index,
    int state_slot,
    float value_prefix,
    bool reset
) {
    if (!std::isfinite(value_prefix)) {
        throw std::invalid_argument("value prefix must be finite");
    }
    if (nodes_[node_index].expanded()) {
        throw std::runtime_error("cannot expand a node more than once");
    }
    nodes_[node_index].state_slot = state_slot;
    nodes_[node_index].value_prefix = value_prefix;
    nodes_[node_index].reset_value_prefix = reset;
    nodes_[node_index].first_child = static_cast<int>(nodes_.size());
}

void RootTree::append_child(
    int node_index,
    int action,
    float prior,
    int depth
) {
    SearchNode child;
    child.prior = prior;
    child.parent = node_index;
    child.action = action;
    child.depth = depth;
    nodes_.push_back(std::move(child));
}

float RootTree::reward(int node_index) const {
    const SearchNode& node = nodes_[node_index];
    if (node.parent < 0) {
        return 0.0F;
    }
    const SearchNode& parent = nodes_[node.parent];
    return parent.reset_value_prefix
        ? node.value_prefix
        : node.value_prefix - parent.value_prefix;
}

float RootTree::q_value(int node_index) const {
    return reward(node_index) + discount_ * nodes_[node_index].value();
}

float RootTree::node_mean_q(
    int node_index,
    float parent_mean_q,
    bool is_root
) const {
    float sum = 0.0F;
    int count = 0;
    const int first_child = nodes_[node_index].first_child;
    for (int action = 0; action < action_count_; ++action) {
        const int child_index = first_child + action;
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

int RootTree::select_child(
    int node_index,
    float mean_q,
    float pb_c_base,
    float pb_c_init
) {
    const int first_child = nodes_[node_index].first_child;
    int child_visits = 0;
    for (int action = 0; action < action_count_; ++action) {
        child_visits += nodes_[first_child + action].visit_count;
    }
    const float exploration_scale = pb_c_init + std::log(
        (static_cast<float>(child_visits) + pb_c_base + 1.0F) / pb_c_base
    );
    const float sqrt_visits = std::sqrt(static_cast<float>(child_visits));
    float best_score = -std::numeric_limits<float>::infinity();
    int selected_child = -1;
    int tie_count = 0;
    for (int action = 0; action < action_count_; ++action) {
        const int child_index = first_child + action;
        const SearchNode& child = nodes_[child_index];
        const float prior_score = child.prior * sqrt_visits
            / static_cast<float>(1 + child.visit_count)
            * exploration_scale;
        const float q = child.visit_count > 0 ? q_value(child_index) : mean_q;
        const float score = stats_.normalize(q) + prior_score;
        if (score > best_score + 1.0e-12F) {
            best_score = score;
            selected_child = child_index;
            tie_count = 1;
        } else if (std::abs(score - best_score) <= 1.0e-12F) {
            ++tie_count;
            if (!deterministic_ties_) {
                std::uniform_int_distribution<int> distribution(1, tie_count);
                if (distribution(rng_) == 1) {
                    selected_child = child_index;
                }
            }
        }
    }
    if (selected_child < 0) {
        throw std::runtime_error("expanded node has no selectable child");
    }
    return selected_child;
}

int RootTree::select_gumbel_child(int node_index) {
    const int first_child = nodes_[node_index].first_child;
    if (node_index == 0) {
        int selected_action = -1;
        int minimum_visits = std::numeric_limits<int>::max();
        for (const int action : selected_root_actions_) {
            const int visits = nodes_[first_child + action].visit_count;
            if (visits < minimum_visits) {
                minimum_visits = visits;
                selected_action = action;
            }
        }
        if (selected_action < 0) {
            throw std::runtime_error("Gumbel root has no candidate action");
        }
        return first_child + selected_action;
    }

    const std::vector<float> policy = improved_policy(node_index);
    const float denominator = 1.0F + static_cast<float>(nodes_[node_index].visit_count);
    int best_action = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < action_count_; ++action) {
        const float score = policy[action]
            - static_cast<float>(nodes_[first_child + action].visit_count)
                / denominator;
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return first_child + best_action;
}

float RootTree::mixed_value(int node_index) const {
    const SearchNode& node = nodes_[node_index];
    const int first_child = node.first_child;
    float prior_sum = 0.0F;
    float weighted_q_sum = 0.0F;
    for (int action = 0; action < action_count_; ++action) {
        const SearchNode& child = nodes_[first_child + action];
        if (child.visit_count > 0) {
            prior_sum += child.prior;
            weighted_q_sum += child.prior * q_value(first_child + action);
        }
    }
    if (!(prior_sum > 0.0F)) {
        return node.value();
    }
    return (node.value() + static_cast<float>(node.visit_count)
        * weighted_q_sum / prior_sum)
        / (1.0F + static_cast<float>(node.visit_count));
}

std::vector<float> RootTree::transformed_completed_q(int node_index) const {
    const int first_child = nodes_[node_index].first_child;
    const float v_mix = mixed_value(node_index);
    int maximum_visits = 0;
    std::vector<float> completed(action_count_);
    for (int action = 0; action < action_count_; ++action) {
        const SearchNode& child = nodes_[first_child + action];
        maximum_visits = std::max(maximum_visits, child.visit_count);
        const float q = child.visit_count > 0
            ? q_value(first_child + action)
            : v_mix;
        completed[action] = stats_.normalize(q);
    }
    const float scale = (c_visit_ + static_cast<float>(maximum_visits))
        * c_scale_;
    for (float& value : completed) {
        value *= scale;
    }
    return completed;
}

std::vector<float> RootTree::improved_policy(int node_index) const {
    const int first_child = nodes_[node_index].first_child;
    std::vector<float> logits = transformed_completed_q(node_index);
    float maximum = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < action_count_; ++action) {
        logits[action] += std::log(std::max(
            nodes_[first_child + action].prior,
            std::numeric_limits<float>::min()
        ));
        maximum = std::max(maximum, logits[action]);
    }
    float total = 0.0F;
    for (float& value : logits) {
        value = std::exp(value - maximum);
        total += value;
    }
    for (float& value : logits) {
        value /= total;
    }
    return logits;
}

void RootTree::initialize_gumbel_candidates() {
    const int first_child = nodes_[0].first_child;
    selected_root_actions_.resize(action_count_);
    std::iota(selected_root_actions_.begin(), selected_root_actions_.end(), 0);
    std::stable_sort(
        selected_root_actions_.begin(),
        selected_root_actions_.end(),
        [&](int left, int right) {
            const float left_score = gumbels_[left] + std::log(std::max(
                nodes_[first_child + left].prior,
                std::numeric_limits<float>::min()
            ));
            const float right_score = gumbels_[right] + std::log(std::max(
                nodes_[first_child + right].prior,
                std::numeric_limits<float>::min()
            ));
            return left_score > right_score;
        }
    );
    selected_root_actions_.resize(num_top_actions_);
}

void RootTree::advance_gumbel_phase() {
    if (completed_simulations_ < phase_visit_threshold_
        || selected_root_actions_.size() <= 1) {
        return;
    }
    const std::vector<float> transformed = transformed_completed_q(0);
    const int first_child = nodes_[0].first_child;
    std::stable_sort(
        selected_root_actions_.begin(),
        selected_root_actions_.end(),
        [&](int left, int right) {
            const float left_score = gumbels_[left] + std::log(std::max(
                nodes_[first_child + left].prior,
                std::numeric_limits<float>::min()
            )) + transformed[left];
            const float right_score = gumbels_[right] + std::log(std::max(
                nodes_[first_child + right].prior,
                std::numeric_limits<float>::min()
            )) + transformed[right];
            return left_score > right_score;
        }
    );
    current_num_top_actions_ /= 2;
    selected_root_actions_.resize(current_num_top_actions_);
    used_phase_visits_ = completed_simulations_;
    if (selected_root_actions_.size() <= 1) {
        phase_visit_threshold_ = num_simulations_;
        return;
    }
    const double phases = std::log2(static_cast<double>(num_top_actions_));
    int extra_visits;
    if (current_num_top_actions_ > 2) {
        extra_visits = static_cast<int>(std::floor(
            num_simulations_ / (phases * current_num_top_actions_)
        )) * current_num_top_actions_;
    } else {
        extra_visits = num_simulations_ - used_phase_visits_;
    }
    phase_visit_threshold_ = std::min(
        phase_visit_threshold_ + extra_visits,
        num_simulations_
    );
}

void RootTree::back_up(float leaf_value) {
    if (!std::isfinite(leaf_value)) {
        throw std::invalid_argument("leaf value must be finite");
    }
    float bootstrap = leaf_value;
    for (auto iterator = path_.rbegin(); iterator != path_.rend(); ++iterator) {
        SearchNode& node = nodes_[*iterator];
        node.value_sum += bootstrap;
        ++node.visit_count;
        bootstrap = reward(*iterator) + discount_ * bootstrap;
    }
}

void RootTree::rebuild_min_max() {
    stats_.clear();
    min_max_stack_.clear();
    min_max_stack_.push_back(0);
    while (!min_max_stack_.empty()) {
        const int node_index = min_max_stack_.back();
        min_max_stack_.pop_back();
        if (!nodes_[node_index].expanded()) {
            continue;
        }
        const int first_child = nodes_[node_index].first_child;
        for (int action = 0; action < action_count_; ++action) {
            const int child_index = first_child + action;
            if (nodes_[child_index].visit_count > 0) {
                stats_.update(q_value(child_index));
                min_max_stack_.push_back(child_index);
            }
        }
    }
}

}  // namespace atariagent::native
