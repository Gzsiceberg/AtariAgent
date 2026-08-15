#include "reanalysis/cache.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace atariagent::native {
namespace {

template <typename T>
torch::Tensor vector_tensor(
    const std::vector<T>& values,
    torch::ScalarType dtype
) {
    return torch::from_blob(
        const_cast<T*>(values.data()),
        {static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().dtype(dtype)
    ).clone();
}

struct CacheEntry {
    float value = 0.0F;
    float search_value = 0.0F;
    std::vector<float> policy;
};

}  // namespace

class ReanalysisCache::Impl {
public:
    CachePreparation prepare(
        const torch::Tensor& policy_mask,
        const torch::Tensor& policy_targets,
        const torch::Tensor& value_targets,
        const torch::Tensor& indices
    ) {
        const torch::Tensor positions = torch::nonzero(policy_mask).contiguous();
        const torch::Tensor contiguous_indices = indices.contiguous();
        const auto* position_data = positions.data_ptr<std::int64_t>();
        const auto* index_data = contiguous_indices.data_ptr<std::int64_t>();
        const auto state_count = policy_targets.size(1);
        const auto action_count = policy_targets.size(2);
        std::unordered_map<std::int64_t, std::size_t> miss_lookup;
        std::vector<std::pair<std::int64_t, const CacheEntry*>> hits;
        CachePreparation result;
        for (std::int64_t row = 0; row < positions.size(0); ++row) {
            const auto sample = position_data[row * 2];
            const auto offset = position_data[row * 2 + 1];
            const auto flat = sample * state_count + offset;
            const auto state_id = index_data[sample] + offset;
            auto cached = entries_.find(state_id);
            if (cached != entries_.end()) {
                hits.emplace_back(flat, &cached->second);
                continue;
            }
            auto existing = miss_lookup.find(state_id);
            if (existing == miss_lookup.end()) {
                const auto index = result.misses.size();
                miss_lookup[state_id] = index;
                result.misses.push_back(CacheMiss{
                    state_id,
                    static_cast<std::size_t>(flat),
                    {static_cast<std::size_t>(flat)},
                });
            } else {
                result.misses[existing->second].positions.push_back(
                    static_cast<std::size_t>(flat)
                );
            }
        }
        result.roots_searched = static_cast<std::int64_t>(
            result.misses.size()
        );
        result.value_targets = value_targets.clone();
        result.search_value_targets = torch::zeros_like(value_targets);
        result.policy_targets = policy_targets.clone();
        apply_hits(hits, action_count, result);
        result.miss_mask = make_miss_mask(policy_mask, result.misses);
        return result;
    }

    void resolve(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& value_targets,
        torch::Tensor& search_value_targets,
        torch::Tensor& policy_targets
    ) {
        const auto action_count = policy_targets.size(2);
        std::vector<std::int64_t> source_indices;
        source_indices.reserve(misses.size());
        for (const CacheMiss& miss : misses) {
            source_indices.push_back(static_cast<std::int64_t>(miss.source));
        }
        const torch::Tensor source_tensor = vector_tensor(
            source_indices, torch::kLong
        );
        const torch::Tensor miss_values = value_targets.view({-1})
            .index_select(0, source_tensor)
            .contiguous();
        const torch::Tensor miss_search_values = search_value_targets
            .view({-1}).index_select(0, source_tensor).contiguous();
        const torch::Tensor miss_policies = policy_targets
            .view({-1, action_count})
            .index_select(0, source_tensor)
            .contiguous();
        const auto* value_data = miss_values.data_ptr<float>();
        const auto* search_value_data = miss_search_values.data_ptr<float>();
        const auto* policy_data = miss_policies.data_ptr<float>();
        std::vector<std::int64_t> duplicate_indices;
        std::vector<float> duplicate_values;
        std::vector<float> duplicate_search_values;
        std::vector<float> duplicate_policies;
        for (std::size_t index = 0; index < misses.size(); ++index) {
            CacheEntry entry;
            entry.value = value_data[index];
            entry.search_value = search_value_data[index];
            entry.policy.assign(
                policy_data + index * action_count,
                policy_data + (index + 1) * action_count
            );
            append_duplicates(
                misses[index],
                entry,
                duplicate_indices,
                duplicate_values,
                duplicate_search_values,
                duplicate_policies
            );
            entries_[misses[index].state_id] = std::move(entry);
        }
        apply_duplicates(
            duplicate_indices,
            duplicate_values,
            duplicate_search_values,
            duplicate_policies,
            action_count,
            value_targets,
            search_value_targets,
            policy_targets
        );
    }

    void clear() { entries_.clear(); }
    std::size_t size() const { return entries_.size(); }

private:
    static void apply_hits(
        const std::vector<std::pair<std::int64_t, const CacheEntry*>>& hits,
        std::int64_t action_count,
        CachePreparation& result
    ) {
        if (hits.empty()) {
            return;
        }
        std::vector<std::int64_t> hit_indices;
        std::vector<float> hit_values;
        std::vector<float> hit_search_values;
        std::vector<float> hit_policies;
        hit_indices.reserve(hits.size());
        hit_values.reserve(hits.size());
        hit_search_values.reserve(hits.size());
        hit_policies.reserve(hits.size() * action_count);
        for (const auto& [flat, entry] : hits) {
            hit_indices.push_back(flat);
            hit_values.push_back(entry->value);
            hit_search_values.push_back(entry->search_value);
            hit_policies.insert(
                hit_policies.end(), entry->policy.begin(), entry->policy.end()
            );
        }
        const torch::Tensor hit_index_tensor = vector_tensor(
            hit_indices, torch::kLong
        );
        result.value_targets.view({-1}).index_put_(
            {hit_index_tensor}, vector_tensor(hit_values, torch::kFloat)
        );
        result.search_value_targets.view({-1}).index_put_(
            {hit_index_tensor},
            vector_tensor(hit_search_values, torch::kFloat)
        );
        result.policy_targets.view({-1, action_count}).index_put_(
            {hit_index_tensor},
            torch::from_blob(
                hit_policies.data(),
                {static_cast<std::int64_t>(hits.size()), action_count},
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone()
        );
    }

    static torch::Tensor make_miss_mask(
        const torch::Tensor& policy_mask,
        const std::vector<CacheMiss>& misses
    ) {
        torch::Tensor miss_mask = torch::zeros_like(policy_mask);
        if (misses.empty()) {
            return miss_mask;
        }
        std::vector<std::int64_t> miss_indices;
        miss_indices.reserve(misses.size());
        for (const CacheMiss& miss : misses) {
            miss_indices.push_back(static_cast<std::int64_t>(miss.source));
        }
        miss_mask.view({-1}).index_fill_(
            0, vector_tensor(miss_indices, torch::kLong), true
        );
        return miss_mask;
    }

    static void append_duplicates(
        const CacheMiss& miss,
        const CacheEntry& entry,
        std::vector<std::int64_t>& indices,
        std::vector<float>& values,
        std::vector<float>& search_values,
        std::vector<float>& policies
    ) {
        for (std::size_t duplicate = 1;
             duplicate < miss.positions.size();
             ++duplicate) {
            indices.push_back(static_cast<std::int64_t>(
                miss.positions[duplicate]
            ));
            values.push_back(entry.value);
            search_values.push_back(entry.search_value);
            policies.insert(
                policies.end(), entry.policy.begin(), entry.policy.end()
            );
        }
    }

    static void apply_duplicates(
        const std::vector<std::int64_t>& indices,
        const std::vector<float>& values,
        const std::vector<float>& search_values,
        const std::vector<float>& policies,
        std::int64_t action_count,
        torch::Tensor& value_targets,
        torch::Tensor& search_value_targets,
        torch::Tensor& policy_targets
    ) {
        if (indices.empty()) {
            return;
        }
        const torch::Tensor index_tensor = vector_tensor(
            indices, torch::kLong
        );
        value_targets.view({-1}).index_put_(
            {index_tensor}, vector_tensor(values, torch::kFloat)
        );
        search_value_targets.view({-1}).index_put_(
            {index_tensor}, vector_tensor(search_values, torch::kFloat)
        );
        policy_targets.view({-1, action_count}).index_put_(
            {index_tensor},
            torch::from_blob(
                const_cast<float*>(policies.data()),
                {static_cast<std::int64_t>(indices.size()), action_count},
                torch::TensorOptions().dtype(torch::kFloat)
            ).clone()
        );
    }

    std::unordered_map<std::int64_t, CacheEntry> entries_;
};

ReanalysisCache::ReanalysisCache() : impl_(std::make_unique<Impl>()) {}
ReanalysisCache::~ReanalysisCache() = default;

CachePreparation ReanalysisCache::prepare(
    const torch::Tensor& policy_mask,
    const torch::Tensor& policy_targets,
    const torch::Tensor& value_targets,
    const torch::Tensor& indices
) {
    return impl_->prepare(
        policy_mask, policy_targets, value_targets, indices
    );
}

void ReanalysisCache::resolve(
    const std::vector<CacheMiss>& misses,
    torch::Tensor& value_targets,
    torch::Tensor& search_value_targets,
    torch::Tensor& policy_targets
) {
    impl_->resolve(
        misses, value_targets, search_value_targets, policy_targets
    );
}

void ReanalysisCache::clear() { impl_->clear(); }
std::size_t ReanalysisCache::size() const { return impl_->size(); }

}  // namespace atariagent::native
