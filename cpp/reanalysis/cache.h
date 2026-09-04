#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace atariagent::native {

struct CacheMiss {
    std::int64_t state_id = -1;
    std::size_t source = 0;
    std::vector<std::size_t> positions;
};

struct CachePreparation {
    torch::Tensor miss_mask;
    torch::Tensor search_value_targets;
    torch::Tensor policy_targets;
    std::vector<CacheMiss> misses;
    std::int64_t roots_searched = 0;
    std::int64_t cache_hits = 0;
    double cache_target_age_sum = 0.0;
    std::int64_t cache_target_age_max = 0;
};

struct ValueCachePreparation {
    torch::Tensor values;
    torch::Tensor miss_mask;
    std::vector<CacheMiss> misses;
    std::int64_t roots_searched = 0;
    std::int64_t cache_hits = 0;
};

class ReanalysisCache {
public:
    ReanalysisCache();
    ~ReanalysisCache();

    ReanalysisCache(const ReanalysisCache&) = delete;
    ReanalysisCache& operator=(const ReanalysisCache&) = delete;

    CachePreparation prepare_policy(
        const torch::Tensor& policy_mask,
        const torch::Tensor& policy_targets,
        const torch::Tensor& state_ids,
        std::int64_t current_step,
        std::int64_t target_ttl
    );
    void resolve_policy(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& search_value_targets,
        torch::Tensor& policy_targets,
        std::int64_t current_step
    );
    ValueCachePreparation prepare_values(
        const torch::Tensor& value_mask,
        const torch::Tensor& state_ids
    );
    void resolve_values(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& values
    );
    void clear_policy();
    void clear_all();
    std::size_t policy_size() const;
    std::size_t value_size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace atariagent::native
