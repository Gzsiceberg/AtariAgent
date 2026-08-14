#pragma once

#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

namespace atariagent::native {

class MinMaxStats {
public:
    explicit MinMaxStats(float minimum_delta);

    void clear();
    void update(float value);
    float normalize(float value) const;

private:
    float minimum_delta_;
    float minimum_;
    float maximum_;
};

struct SearchNode {
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

    bool expanded() const;
    float value() const;
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
    );

    std::tuple<int, int, bool> traverse(float pb_c_base, float pb_c_init);
    void expand_and_back_up(
        int state_slot,
        float value_prefix,
        float value,
        const std::vector<float>& policy_logits
    );
    std::vector<int> visit_counts() const;
    float root_value() const;

private:
    static std::vector<float> softmax(const std::vector<float>& logits);
    void expand(
        int node_index,
        int state_slot,
        float value_prefix,
        const std::vector<float>& priors,
        bool reset,
        bool priors_are_probabilities
    );
    float reward(int node_index) const;
    float q_value(int node_index) const;
    float node_mean_q(
        int node_index,
        float parent_mean_q,
        bool is_root
    ) const;
    int select_child(
        int node_index,
        float mean_q,
        float pb_c_base,
        float pb_c_init
    );
    void back_up(float leaf_value);
    void rebuild_min_max();

    int action_count_;
    float discount_;
    int value_prefix_horizon_;
    MinMaxStats stats_;
    std::mt19937_64 rng_;
    bool deterministic_ties_;
    std::vector<SearchNode> nodes_;
    std::vector<int> path_;
};

}  // namespace atariagent::native
