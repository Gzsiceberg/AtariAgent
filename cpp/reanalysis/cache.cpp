#include "reanalysis/cache.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace atariagent::native {
namespace {

struct CacheEntry {
    float value = 0.0F;
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

class TargetWriter {
public:
    TargetWriter(
        torch::Tensor& values,
        torch::Tensor& search_values,
        torch::Tensor& policies
    )
        : values_(tensor_span<float>(values)),
          search_values_(tensor_span<float>(search_values)),
          policies_(tensor_span<float>(policies)),
          action_count_(static_cast<std::size_t>(policies.size(2))) {}

    float value(std::size_t position) const { return values_[position]; }

    float search_value(std::size_t position) const {
        return search_values_[position];
    }

    std::span<const float> policy(std::size_t position) const {
        return policies_.subspan(
            position * action_count_, action_count_
        );
    }

    void copy_entry(std::size_t position, const CacheEntry& entry) {
        if (entry.policy.size() != action_count_) {
            throw std::runtime_error(
                "cached policy has an incompatible action count"
            );
        }
        values_[position] = entry.value;
        search_values_[position] = entry.search_value;
        std::copy_n(
            entry.policy.data(),
            action_count_,
            mutable_policy(position).data()
        );
    }

private:
    std::span<float> mutable_policy(std::size_t position) {
        return policies_.subspan(
            position * action_count_, action_count_
        );
    }

    std::span<float> values_;
    std::span<float> search_values_;
    std::span<float> policies_;
    std::size_t action_count_;
};

class PreparationWriter {
public:
    PreparationWriter(
        const torch::Tensor& source_values,
        const torch::Tensor& source_policies,
        torch::Tensor& values,
        torch::Tensor& search_values,
        torch::Tensor& policies,
        torch::Tensor& miss_mask,
        torch::Tensor& search_value_available_mask
    )
        : source_values_(tensor_span<float>(source_values)),
          source_policies_(tensor_span<float>(source_policies)),
          values_(tensor_span<float>(values)),
          search_values_(tensor_span<float>(search_values)),
          policies_(tensor_span<float>(policies)),
          miss_mask_(tensor_span<bool>(miss_mask)),
          search_value_available_mask_(
              tensor_span<bool>(search_value_available_mask)
          ),
          action_count_(
              static_cast<std::size_t>(source_policies.size(2))
          ) {}

    void copy_original(std::size_t position) {
        values_[position] = source_values_[position];
        const auto source = source_policy(position);
        std::copy_n(
            source.data(), action_count_, policy(position).data()
        );
    }

    void copy_hit(std::size_t position, const CacheEntry& entry) {
        if (entry.policy.size() != action_count_) {
            throw std::runtime_error(
                "cached policy has an incompatible action count"
            );
        }
        values_[position] = entry.value;
        search_values_[position] = entry.search_value;
        std::copy_n(
            entry.policy.data(),
            action_count_,
            policy(position).data()
        );
        search_value_available_mask_[position] = true;
    }

    void mark_miss(std::size_t position) {
        miss_mask_[position] = true;
        search_value_available_mask_[position] = true;
    }

private:
    std::span<const float> source_policy(std::size_t position) const {
        return source_policies_.subspan(
            position * action_count_, action_count_
        );
    }

    std::span<float> policy(std::size_t position) {
        return policies_.subspan(
            position * action_count_, action_count_
        );
    }

    std::span<const float> source_values_;
    std::span<const float> source_policies_;
    std::span<float> values_;
    std::span<float> search_values_;
    std::span<float> policies_;
    std::span<bool> miss_mask_;
    std::span<bool> search_value_available_mask_;
    std::size_t action_count_;
};

}  // namespace

class ReanalysisCache::Impl {
public:
    CachePreparation prepare(
        const torch::Tensor& policy_mask,
        const torch::Tensor& policy_targets,
        const torch::Tensor& value_targets,
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
        const auto position_count = static_cast<std::size_t>(
            value_targets.numel()
        );

        // Assemble the contiguous CPU outputs directly. Materializing hit
        // vectors as temporary tensors makes cached requests much more costly
        // than the map lookups themselves.
        CachePreparation result;
        result.value_targets = torch::empty_like(value_targets);
        result.search_value_targets = torch::zeros_like(value_targets);
        result.policy_targets = torch::empty_like(policy_targets);
        result.miss_mask = torch::zeros_like(contiguous_mask);
        result.search_value_available_mask = torch::zeros_like(
            contiguous_mask
        );
        PreparationWriter targets(
            value_targets,
            policy_targets,
            result.value_targets,
            result.search_value_targets,
            result.policy_targets,
            result.miss_mask,
            result.search_value_available_mask
        );

        std::unordered_map<std::int64_t, std::size_t> miss_lookup;
        for (std::size_t flat = 0; flat < position_count; ++flat) {
            if (!mask[flat]) {
                targets.copy_original(flat);
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
                targets.copy_original(flat);
                continue;
            }

            const auto cached = entries_.find(state_id);
            if (cached != entries_.end()) {
                const auto age = current_step - cached->second.created_step;
                if (age < 0) {
                    throw std::runtime_error(
                        "cache entry was created after the current step"
                    );
                }
                const bool expired = target_ttl > 0 && age >= target_ttl;
                if (!expired) {
                    targets.copy_hit(flat, cached->second);
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
            targets.copy_original(flat);
            targets.mark_miss(flat);
        }
        result.roots_searched = static_cast<std::int64_t>(
            result.misses.size()
        );
        return result;
    }

    void resolve(
        const std::vector<CacheMiss>& misses,
        torch::Tensor& value_targets,
        torch::Tensor& search_value_targets,
        torch::Tensor& policy_targets,
        std::int64_t current_step
    ) {
        if (current_step < 0) {
            throw std::invalid_argument("current_step must be non-negative");
        }
        // Miss sources and their duplicate destinations are already flat
        // positions in these contiguous output tensors.
        TargetWriter targets(
            value_targets, search_value_targets, policy_targets
        );
        for (const CacheMiss& miss : misses) {
            auto [entry_position, inserted] = entries_.try_emplace(
                miss.state_id
            );
            (void)inserted;
            CacheEntry& entry = entry_position->second;
            entry.value = targets.value(miss.source);
            entry.search_value = targets.search_value(miss.source);
            const auto source_policy = targets.policy(miss.source);
            entry.policy.assign(
                source_policy.begin(), source_policy.end()
            );
            entry.created_step = current_step;
            for (std::size_t duplicate = 1;
                 duplicate < miss.positions.size();
                 ++duplicate) {
                targets.copy_entry(
                    miss.positions[duplicate], entry
                );
            }
        }
    }

    SearchValueLookup lookup_search_values(
        const torch::Tensor& mask,
        const torch::Tensor& state_ids,
        std::int64_t current_step,
        std::int64_t target_ttl
    ) const {
        if (current_step < 0) {
            throw std::invalid_argument("current_step must be non-negative");
        }
        if (target_ttl < 0) {
            throw std::invalid_argument("target_ttl must be non-negative");
        }
        const torch::Tensor contiguous_mask = mask.contiguous();
        const torch::Tensor contiguous_state_ids = state_ids.contiguous();
        SearchValueLookup result{
            torch::zeros(
                contiguous_mask.sizes(),
                contiguous_mask.options().dtype(torch::kFloat)
            ),
            torch::zeros_like(contiguous_mask),
        };
        const auto mask_data = tensor_span<bool>(contiguous_mask);
        const auto state_id_data = tensor_span<std::int64_t>(
            contiguous_state_ids
        );
        auto value_data = tensor_span<float>(result.values);
        auto available_data = tensor_span<bool>(result.available_mask);
        for (std::size_t flat = 0; flat < mask_data.size(); ++flat) {
            if (!mask_data[flat]) {
                continue;
            }
            const auto state_id = state_id_data[flat];
            if (state_id < 0) {
                throw std::invalid_argument(
                    "active bootstrap state IDs must be non-negative"
                );
            }
            const auto cached = entries_.find(state_id);
            if (cached == entries_.end()) {
                continue;
            }
            const auto age = current_step - cached->second.created_step;
            if (age < 0) {
                throw std::runtime_error(
                    "cache entry was created after the current step"
                );
            }
            const bool expired = target_ttl > 0 && age >= target_ttl;
            if (!expired) {
                value_data[flat] = cached->second.search_value;
                available_data[flat] = true;
            }
        }
        return result;
    }

    void clear() { entries_.clear(); }
    std::size_t size() const { return entries_.size(); }

private:
    std::unordered_map<std::int64_t, CacheEntry> entries_;
};

ReanalysisCache::ReanalysisCache() : impl_(std::make_unique<Impl>()) {}
ReanalysisCache::~ReanalysisCache() = default;

CachePreparation ReanalysisCache::prepare(
    const torch::Tensor& policy_mask,
    const torch::Tensor& policy_targets,
    const torch::Tensor& value_targets,
    const torch::Tensor& state_ids,
    std::int64_t current_step,
    std::int64_t target_ttl
) {
    return impl_->prepare(
        policy_mask,
        policy_targets,
        value_targets,
        state_ids,
        current_step,
        target_ttl
    );
}

void ReanalysisCache::resolve(
    const std::vector<CacheMiss>& misses,
    torch::Tensor& value_targets,
    torch::Tensor& search_value_targets,
    torch::Tensor& policy_targets,
    std::int64_t current_step
) {
    impl_->resolve(
        misses,
        value_targets,
        search_value_targets,
        policy_targets,
        current_step
    );
}

SearchValueLookup ReanalysisCache::lookup_search_values(
    const torch::Tensor& mask,
    const torch::Tensor& state_ids,
    std::int64_t current_step,
    std::int64_t target_ttl
) const {
    return impl_->lookup_search_values(
        mask, state_ids, current_step, target_ttl
    );
}

void ReanalysisCache::clear() { impl_->clear(); }
std::size_t ReanalysisCache::size() const { return impl_->size(); }

}  // namespace atariagent::native
