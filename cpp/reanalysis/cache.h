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
    torch::Tensor value_targets;
    torch::Tensor search_value_targets;
    torch::Tensor policy_targets;
    std::vector<CacheMiss> misses;
    std::int64_t roots_searched = 0;
};

class ReanalysisCache {
public:
    ReanalysisCache();
    ~ReanalysisCache();

    ReanalysisCache(const ReanalysisCache&) = delete;
    ReanalysisCache& operator=(const ReanalysisCache&) = delete;

    CachePreparation prepare(
        const torch::Tensor& policy_mask,
        const torch::Tensor& policy_targets,
        const torch::Tensor& value_targets,
        const torch::Tensor& indices
    );
    void resolve(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& value_targets,
        torch::Tensor& search_value_targets,
        torch::Tensor& policy_targets
    );
    void clear();
    std::size_t size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace atariagent::native
