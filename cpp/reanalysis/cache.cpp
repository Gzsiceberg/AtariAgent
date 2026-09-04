#include "reanalysis/cache.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace atariagent::native {
namespace {

struct PolicyCacheEntry {
    float search_value = 0.0F;
    std::vector<float> policy;
    std::int64_t created_step = 0;
};

// Engine validation and the contiguous temporaries below establish the flat
// CPU layout before these spans are created.
template <typename T>
std::span<T> tensor_span(torch::Tensor& tensor) {
    return {
        tensor.data_ptr<T>(),
        static_cast<std::size_t>(tensor.numel()),
    };
}

template <typename T>
std::span<const T> tensor_span(const torch::Tensor& tensor) {
    return {
        tensor.data_ptr<T>(),
        static_cast<std::size_t>(tensor.numel()),
    };
}

class PolicyWriter {
public:
    PolicyWriter(
        torch::Tensor& search_values,
        torch::Tensor& policies
    )
        : search_values_(tensor_span<float>(search_values)),
          policies_(tensor_span<float>(policies)),
          action_count_(static_cast<std::size_t>(policies.size(2))) {}

    float search_value(std::size_t position) const {
        return search_values_[position];
    }

    std::span<const float> policy(std::size_t position) const {
        return policies_.subspan(position * action_count_, action_count_);
    }

    void copy_entry(
        std::size_t position,
        const PolicyCacheEntry& entry
    ) {
        if (entry.policy.size() != action_count_) {
            throw std::runtime_error(
                "cached policy has an incompatible action count"
            );
        }
        search_values_[position] = entry.search_value;
        std::copy_n(
            entry.policy.data(),
            action_count_,
            mutable_policy(position).data()
        );
    }

private:
    std::span<float> mutable_policy(std::size_t position) {
        return policies_.subspan(position * action_count_, action_count_);
    }

    std::span<float> search_values_;
    std::span<float> policies_;
    std::size_t action_count_;
};

}  // namespace

class ReanalysisCache::Impl {
public:
    CachePreparation prepare_policy(
        const torch::Tensor& policy_mask,
        const torch::Tensor& policy_targets,
        const torch::Tensor& state_ids,
        std::int64_t current_step,
        std::int64_t target_ttl
    ) {
        if (current_step < 0) {
            throw std::invalid_argument("current_step must be non-negative");
        }
        if (target_ttl < 0) {
            throw std::invalid_argument("target_ttl must be non-negative");
        }
        const torch::Tensor contiguous_mask = policy_mask.contiguous();
        const torch::Tensor contiguous_state_ids = state_ids.contiguous();
        const auto mask = tensor_span<bool>(contiguous_mask);
        const auto state_id_data = tensor_span<std::int64_t>(
            contiguous_state_ids
        );

        CachePreparation result;
        result.search_value_targets = torch::zeros(
            contiguous_mask.sizes(),
            policy_targets.options()
        );
        result.policy_targets = policy_targets.clone();
        result.miss_mask = torch::zeros_like(contiguous_mask);
        PolicyWriter targets(
            result.search_value_targets,
            result.policy_targets
        );
        auto miss_mask = tensor_span<bool>(result.miss_mask);

        std::unordered_map<std::int64_t, std::size_t> miss_lookup;
        for (std::size_t flat = 0; flat < mask.size(); ++flat) {
            if (!mask[flat]) {
                continue;
            }

            const auto state_id = state_id_data[flat];
            if (state_id < 0) {
                throw std::invalid_argument(
                    "active reanalysis state IDs must be non-negative"
                );
            }
            const auto existing = miss_lookup.find(state_id);
            if (existing != miss_lookup.end()) {
                result.misses[existing->second].positions.push_back(flat);
                continue;
            }

            const auto cached = policy_entries_.find(state_id);
            if (cached != policy_entries_.end()) {
                const auto age = current_step - cached->second.created_step;
                if (age < 0) {
                    throw std::runtime_error(
                        "cache entry was created after the current step"
                    );
                }
                const bool expired = target_ttl > 0 && age >= target_ttl;
                if (!expired) {
                    targets.copy_entry(flat, cached->second);
                    ++result.cache_hits;
                    result.cache_target_age_sum += static_cast<double>(age);
                    result.cache_target_age_max = std::max(
                        result.cache_target_age_max, age
                    );
                    continue;
                }
            }

            const auto index = result.misses.size();
            miss_lookup.emplace(state_id, index);
            result.misses.push_back(CacheMiss{
                state_id,
                flat,
                {flat},
            });
            miss_mask[flat] = true;
        }
        result.roots_searched = static_cast<std::int64_t>(
            result.misses.size()
        );
        return result;
    }

    void resolve_policy(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& search_value_targets,
        torch::Tensor& policy_targets,
        std::int64_t current_step
    ) {
        if (current_step < 0) {
            throw std::invalid_argument("current_step must be non-negative");
        }
        PolicyWriter targets(search_value_targets, policy_targets);
        for (const CacheMiss& miss : misses) {
            PolicyCacheEntry entry;
            entry.search_value = targets.search_value(miss.source);
            const auto source_policy = targets.policy(miss.source);
            entry.policy.assign(source_policy.begin(), source_policy.end());
            entry.created_step = current_step;
            policy_entries_[miss.state_id] = entry;
            for (std::size_t duplicate = 1;
                 duplicate < miss.positions.size();
                 ++duplicate) {
                targets.copy_entry(miss.positions[duplicate], entry);
            }
        }
    }

    ValueCachePreparation prepare_values(
        const torch::Tensor& value_mask,
        const torch::Tensor& state_ids
    ) {
        const torch::Tensor contiguous_mask = value_mask.contiguous();
        const torch::Tensor contiguous_state_ids = state_ids.contiguous();
        const auto mask = tensor_span<bool>(contiguous_mask);
        const auto state_id_data = tensor_span<std::int64_t>(
            contiguous_state_ids
        );

        ValueCachePreparation result;
        result.values = torch::zeros(
            contiguous_mask.sizes(),
            contiguous_mask.options().dtype(torch::kFloat)
        );
        result.miss_mask = torch::zeros_like(contiguous_mask);
        auto values = tensor_span<float>(result.values);
        auto miss_mask = tensor_span<bool>(result.miss_mask);

        std::unordered_map<std::int64_t, std::size_t> miss_lookup;
        for (std::size_t flat = 0; flat < mask.size(); ++flat) {
            if (!mask[flat]) {
                continue;
            }
            const auto state_id = state_id_data[flat];
            if (state_id < 0) {
                throw std::invalid_argument(
                    "active value-bootstrap state IDs must be non-negative"
                );
            }
            const auto existing = miss_lookup.find(state_id);
            if (existing != miss_lookup.end()) {
                result.misses[existing->second].positions.push_back(flat);
                continue;
            }
            const auto cached = value_entries_.find(state_id);
            if (cached != value_entries_.end()) {
                values[flat] = cached->second;
                ++result.cache_hits;
                continue;
            }

            const auto index = result.misses.size();
            miss_lookup.emplace(state_id, index);
            result.misses.push_back(CacheMiss{
                state_id,
                flat,
                {flat},
            });
            miss_mask[flat] = true;
        }
        result.roots_searched = static_cast<std::int64_t>(
            result.misses.size()
        );
        return result;
    }

    void resolve_values(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& values
    ) {
        auto value_data = tensor_span<float>(values);
        for (const CacheMiss& miss : misses) {
            const float value = value_data[miss.source];
            value_entries_[miss.state_id] = value;
            for (std::size_t duplicate = 1;
                 duplicate < miss.positions.size();
                 ++duplicate) {
                value_data[miss.positions[duplicate]] = value;
            }
        }
    }

    void clear_policy() { policy_entries_.clear(); }

    void clear_all() {
        policy_entries_.clear();
        value_entries_.clear();
    }

    std::size_t policy_size() const { return policy_entries_.size(); }
    std::size_t value_size() const { return value_entries_.size(); }

private:
    std::unordered_map<std::int64_t, PolicyCacheEntry> policy_entries_;
    std::unordered_map<std::int64_t, float> value_entries_;
};

ReanalysisCache::ReanalysisCache() : impl_(std::make_unique<Impl>()) {}
ReanalysisCache::~ReanalysisCache() = default;

CachePreparation ReanalysisCache::prepare_policy(
    const torch::Tensor& policy_mask,
    const torch::Tensor& policy_targets,
    const torch::Tensor& state_ids,
    std::int64_t current_step,
    std::int64_t target_ttl
) {
    return impl_->prepare_policy(
        policy_mask,
        policy_targets,
        state_ids,
        current_step,
        target_ttl
    );
}

void ReanalysisCache::resolve_policy(
    const std::vector<CacheMiss>& misses,
    torch::Tensor& search_value_targets,
    torch::Tensor& policy_targets,
    std::int64_t current_step
) {
    impl_->resolve_policy(
        misses,
        search_value_targets,
        policy_targets,
        current_step
    );
}

ValueCachePreparation ReanalysisCache::prepare_values(
    const torch::Tensor& value_mask,
    const torch::Tensor& state_ids
) {
    return impl_->prepare_values(value_mask, state_ids);
}

void ReanalysisCache::resolve_values(
    const std::vector<CacheMiss>& misses,
    torch::Tensor& values
) {
    impl_->resolve_values(misses, values);
}

void ReanalysisCache::clear_policy() { impl_->clear_policy(); }
void ReanalysisCache::clear_all() { impl_->clear_all(); }
std::size_t ReanalysisCache::policy_size() const {
    return impl_->policy_size();
}
std::size_t ReanalysisCache::value_size() const {
    return impl_->value_size();
}

}  // namespace atariagent::native
