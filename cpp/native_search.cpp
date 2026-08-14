#include "native_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

bool SearchNode::expanded() const { return !children.empty(); }

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
    nodes_.reserve(1 + action_count_ * (std::max(simulations, 0) + 1));
    SearchNode root;
    root.prior = 1.0F;
    root.state_slot = 0;
    root.value_prefix = root_value_prefix;
    root.visit_count = 1;
    root.value_sum = root_value;
    nodes_.push_back(std::move(root));
    expand(0, 0, root_value_prefix, root_priors, false, true);
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
        node_index = select_child(node_index, mean_q, pb_c_base, pb_c_init);
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
    const std::vector<float>& policy_logits
) {
    if (path_.empty()) {
        throw std::runtime_error("traverse must precede expansion");
    }
    const int leaf_index = path_.back();
    const bool reset = nodes_[leaf_index].depth % value_prefix_horizon_ == 0;
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

std::vector<int> RootTree::visit_counts() const {
    std::vector<int> counts;
    counts.reserve(action_count_);
    for (const int child_index : nodes_[0].children) {
        counts.push_back(nodes_[child_index].visit_count);
    }
    return counts;
}

float RootTree::root_value() const { return nodes_[0].value(); }

std::vector<float> RootTree::softmax(const std::vector<float>& logits) {
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

void RootTree::expand(
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
        SearchNode child;
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

int RootTree::select_child(
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
        (static_cast<float>(child_visits) + pb_c_base + 1.0F) / pb_c_base
    );
    const float sqrt_visits = std::sqrt(static_cast<float>(child_visits));
    float best_score = -std::numeric_limits<float>::infinity();
    std::vector<int> best_children;
    for (const int child_index : nodes_[node_index].children) {
        const SearchNode& child = nodes_[child_index];
        const float prior_score = child.prior * sqrt_visits
            / static_cast<float>(1 + child.visit_count)
            * exploration_scale;
        const float q = child.visit_count > 0 ? q_value(child_index) : mean_q;
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
        0, best_children.size() - 1
    );
    return best_children[distribution(rng_)];
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

}  // namespace atariagent::native
