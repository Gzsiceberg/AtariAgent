#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <tuple>
#include <vector>

namespace atariagent::native {

enum class SearchAlgorithm {
    Puct,
    Gumbel,
};

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
    float log_prior = 0.0F;
    int parent = -1;
    int action = -1;
    int depth = 0;
    int state_slot = -1;
    float value_prefix = 0.0F;
    bool reset_value_prefix = false;
    int visit_count = 0;
    float value_sum = 0.0F;
    // Immutable network value captured when this node is expanded.
    float raw_value = 0.0F;
    int first_child = -1;

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
        bool deterministic_ties,
        SearchAlgorithm algorithm = SearchAlgorithm::Puct,
        int num_top_actions = 4,
        float c_visit = 50.0F,
        float c_scale = 0.1F,
        bool use_gumbel_noise = true
    );

    std::tuple<int, int, bool> traverse(float pb_c_base, float pb_c_init);
    void expand_and_back_up(
        int state_slot,
        float value_prefix,
        float value,
        std::span<const float> policy_logits
    );
    float write_policy_and_root_value(float* output) const;
    int selected_action() const;

private:
    void expand_probabilities(
        int node_index,
        int state_slot,
        float value_prefix,
        std::span<const float> priors,
        bool reset
    );
    void expand_logits(
        int node_index,
        int state_slot,
        float value_prefix,
        std::span<const float> logits,
        bool reset
    );
    void initialize_children(
        int node_index,
        int state_slot,
        float value_prefix,
        bool reset
    );
    void append_child(int node_index, int action, float prior, int depth);
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
    int select_gumbel_child(int node_index);
    float mixed_value(int node_index, int* maximum_visits = nullptr) const;
    float completed_q(int node_index, int action, float v_mix) const;
    void write_policy(float* output) const;
    void write_transformed_completed_q(int node_index, float* output) const;
    void write_improved_policy(int node_index, float* output) const;
    void initialize_gumbel_candidates();
    void advance_gumbel_phase();
    void back_up(float leaf_value);
    void rebuild_min_max();

    int action_count_;
    float discount_;
    int value_prefix_horizon_;
    MinMaxStats stats_;
    std::mt19937_64 rng_;
    bool deterministic_ties_;
    SearchAlgorithm algorithm_;
    int num_simulations_;
    int num_top_actions_;
    float c_visit_;
    float c_scale_;
    int completed_simulations_ = 0;
    int current_num_top_actions_ = 0;
    int phase_visit_threshold_ = 0;
    int used_phase_visits_ = 0;
    std::vector<float> gumbels_;
    std::vector<float> gumbel_score_scratch_;
    std::vector<int> selected_root_actions_;
    std::vector<SearchNode> nodes_;
    std::vector<int> path_;
    std::vector<int> min_max_stack_;
};

}  // namespace atariagent::native
