#include <torch/extension.h>

#include "reanalysis/engine.h"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace atariagent::native {

class NativeReanalysisEngine::Impl {
public:
    Impl(
        py::object target,
        int prefetch_batches,
        double timeout_seconds,
        int target_update_interval,
        bool cache_targets
    )
        : target_(std::move(target)),
          prefetch_batches_(prefetch_batches),
          timeout_seconds_(timeout_seconds),
          target_update_interval_(target_update_interval),
          cache_targets_(cache_targets) {
        if (prefetch_batches <= 0) {
            throw std::invalid_argument("prefetch_batches must be positive");
        }
        if (!std::isfinite(timeout_seconds) || timeout_seconds <= 0.0) {
            throw std::invalid_argument("timeout_seconds must be positive");
        }
        if (target_update_interval <= 0) {
            throw std::invalid_argument(
                "target_update_interval must be positive"
            );
        }
        worker_ = std::thread(&Impl::run, this);
    }

    ~Impl() { close_impl(); }

    void publish_weights(
        std::int64_t version,
        py::object representation,
        py::object prediction,
        py::object dynamics
    ) {
        auto job = std::make_shared<Job>();
        job->kind = Kind::Weights;
        job->version = version;
        job->representation = std::move(representation);
        job->prediction = std::move(prediction);
        job->dynamics = std::move(dynamics);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require_open_locked();
            if (version <= published_version_) {
                throw std::invalid_argument(
                    "target weight version must increase"
                );
            }
            published_version_ = version;
            jobs_.push_back(job);
        }
        work_ready_.notify_one();
        wait_job(job, "timed out publishing target weights");
    }

    std::int64_t submit(py::object batch) {
        auto job = std::make_shared<Job>();
        job->kind = Kind::Request;
        job->policy_mask = py::cast<at::Tensor>(batch.attr("policy_mask"));
        job->policy_targets = py::cast<at::Tensor>(
            batch.attr("policy_targets")
        );
        job->value_targets = py::cast<at::Tensor>(batch.attr("value_targets"));
        job->indices = py::cast<at::Tensor>(batch.attr("indices"));
        py::object bootstrap_mask = batch.attr("value_bootstrap_mask");
        if (!bootstrap_mask.is_none()) {
            job->value_bootstrap_mask = py::cast<at::Tensor>(bootstrap_mask);
        }
        validate_batch_tensors(
            job->policy_mask,
            job->policy_targets,
            job->value_targets,
            job->indices,
            job->value_bootstrap_mask
        );
        job->batch = std::move(batch);
        job->submitted = Clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require_open_locked();
            if (published_version_ < 0) {
                throw std::runtime_error(
                    "target weights must be published before submission"
                );
            }
            if (pending_count_locked() >= prefetch_batches_) {
                PyErr_SetString(
                    PyExc_BufferError,
                    "reanalysis prefetch limit reached"
                );
                throw py::error_already_set();
            }
            job->request_id = next_request_id_++;
            job->version = published_version_;
            pending_[job->request_id] = job;
            jobs_.push_back(job);
            max_pending_ = std::max(max_pending_, pending_count_locked());
        }
        work_ready_.notify_one();
        return job->request_id;
    }

    py::dict wait_next() {
        std::shared_ptr<Job> job;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            const auto timeout = std::chrono::duration<double>(timeout_seconds_);
            if (!result_ready_.wait_for(lock, timeout, [this] {
                    return !completed_.empty() || failure_ || closed_;
                })) {
                PyErr_SetString(
                    PyExc_TimeoutError,
                    "timed out waiting for asynchronous reanalysis"
                );
                throw py::error_already_set();
            }
            if (failure_) {
                std::rethrow_exception(failure_);
            }
            if (completed_.empty()) {
                throw std::runtime_error("reanalysis engine is closed");
            }
            job = completed_.front();
            completed_.pop_front();
            pending_.erase(job->request_id);
        }
        if (job->error) {
            std::rethrow_exception(job->error);
        }
        const auto lag = published_version_ - job->version;
        if (lag < 0 || lag > target_update_interval_) {
            throw std::runtime_error(
                "reanalysis result is older than one target update interval"
            );
        }
        py::dict result;
        result["request_id"] = job->request_id;
        result["weight_version"] = job->version;
        result["batch"] = job->result_batch;
        result["queue_wait_ms"] = job->queue_wait_ms;
        result["worker_duration_ms"] = job->worker_duration_ms;
        result["peak_memory_bytes"] = job->peak_memory_bytes;
        result["policy_roots_requested"] = job->roots_requested;
        result["policy_roots_searched"] = job->roots_searched;
        result["cache_size"] = cache_size_;
        return result;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
            for (const auto& job : jobs_) {
                if (job->kind == Kind::Request) {
                    job->error = std::make_exception_ptr(
                        std::runtime_error("reanalysis engine is closed")
                    );
                    completed_.push_back(job);
                }
            }
            jobs_.clear();
        }
        work_ready_.notify_all();
        result_ready_.notify_all();
        if (worker_.joinable()) {
            py::gil_scoped_release release;
            worker_.join();
        }
    }

    std::size_t pending_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_count_locked();
    }

    std::size_t max_pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_pending_;
    }

    std::size_t cache_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_size_;
    }

    std::int64_t weight_version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return published_version_;
    }

private:
    using Clock = std::chrono::steady_clock;
    enum class Kind { Request, Weights };

    struct CacheMiss {
        std::int64_t state_id = -1;
        std::size_t source = 0;
        std::vector<std::size_t> positions;
    };

    struct CacheEntry {
        float value = 0.0F;
        std::vector<float> policy;
    };

    struct Job {
        Kind kind = Kind::Request;
        std::int64_t request_id = -1;
        std::int64_t version = -1;
        py::object batch;
        py::object result_batch;
        py::object representation;
        py::object prediction;
        py::object dynamics;
        py::object original_batch;
        at::Tensor policy_mask;
        at::Tensor policy_targets;
        at::Tensor value_targets;
        at::Tensor indices;
        std::optional<at::Tensor> value_bootstrap_mask;
        std::vector<CacheMiss> cache_misses;
        Clock::time_point submitted;
        double queue_wait_ms = 0.0;
        double worker_duration_ms = 0.0;
        std::int64_t peak_memory_bytes = 0;
        std::int64_t roots_requested = 0;
        std::int64_t roots_searched = 0;
        std::exception_ptr error;
        bool done = false;
        std::mutex done_mutex;
        std::condition_variable done_cv;
    };

    py::object target_;
    int prefetch_batches_;
    double timeout_seconds_;
    int target_update_interval_;
    bool cache_targets_;
    mutable std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable result_ready_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::deque<std::shared_ptr<Job>> completed_;
    std::unordered_map<std::int64_t, std::shared_ptr<Job>> pending_;
    std::unordered_map<std::int64_t, CacheEntry> cache_;
    std::thread worker_;
    std::exception_ptr failure_;
    bool closed_ = false;
    std::int64_t next_request_id_ = 0;
    std::int64_t published_version_ = -1;
    std::size_t max_pending_ = 0;
    std::size_t cache_size_ = 0;

    std::size_t pending_count_locked() const { return pending_.size(); }

    template <typename T>
    static at::Tensor vector_tensor(
        const std::vector<T>& values,
        at::ScalarType dtype
    ) {
        return at::from_blob(
            const_cast<T*>(values.data()),
            {static_cast<std::int64_t>(values.size())},
            at::TensorOptions().dtype(dtype)
        ).clone();
    }

    static void validate_batch_tensors(
        const at::Tensor& policy_mask,
        const at::Tensor& policy_targets,
        const at::Tensor& value_targets,
        const at::Tensor& indices,
        const std::optional<at::Tensor>& value_bootstrap_mask
    ) {
        const auto require_cpu = [](const at::Tensor& tensor, const char* name) {
            if (!tensor.device().is_cpu()) {
                throw std::invalid_argument(
                    std::string(name) + " must be a CPU tensor"
                );
            }
        };
        require_cpu(policy_mask, "policy_mask");
        require_cpu(policy_targets, "policy_targets");
        require_cpu(value_targets, "value_targets");
        require_cpu(indices, "indices");
        if (policy_mask.scalar_type() != at::kBool || policy_mask.dim() != 2) {
            throw std::invalid_argument("policy_mask must be a 2D bool tensor");
        }
        if (policy_targets.scalar_type() != at::kFloat
            || policy_targets.dim() != 3) {
            throw std::invalid_argument(
                "policy_targets must be a 3D float32 tensor"
            );
        }
        if (value_targets.scalar_type() != at::kFloat
            || value_targets.dim() != 2) {
            throw std::invalid_argument(
                "value_targets must be a 2D float32 tensor"
            );
        }
        if (indices.scalar_type() != at::kLong || indices.dim() != 1) {
            throw std::invalid_argument("indices must be a 1D int64 tensor");
        }
        if (policy_targets.size(0) != policy_mask.size(0)
            || policy_targets.size(1) != policy_mask.size(1)
            || value_targets.sizes() != policy_mask.sizes()
            || indices.size(0) != policy_mask.size(0)) {
            throw std::invalid_argument(
                "reanalysis batch tensors have incompatible shapes"
            );
        }
        if (value_bootstrap_mask.has_value()) {
            require_cpu(*value_bootstrap_mask, "value_bootstrap_mask");
            if (value_bootstrap_mask->scalar_type() != at::kBool
                || value_bootstrap_mask->sizes() != policy_mask.sizes()) {
                throw std::invalid_argument(
                    "value_bootstrap_mask must match policy_mask"
                );
            }
        }
    }

    void require_open_locked() const {
        if (closed_) {
            throw std::runtime_error("reanalysis engine is closed");
        }
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

    void wait_job(const std::shared_ptr<Job>& job, const char* message) {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(job->done_mutex);
        const auto timeout = std::chrono::duration<double>(timeout_seconds_);
        if (!job->done_cv.wait_for(lock, timeout, [&job] { return job->done; })) {
            PyErr_SetString(PyExc_TimeoutError, message);
            throw py::error_already_set();
        }
        if (job->error) {
            std::rethrow_exception(job->error);
        }
    }

    void run() {
        py::gil_scoped_acquire acquire;
        while (true) {
            std::shared_ptr<Job> job;
            {
                py::gil_scoped_release release;
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [this] {
                    return closed_ || !jobs_.empty();
                });
                if (closed_) {
                    return;
                }
                job = jobs_.front();
                jobs_.pop_front();
            }
            try {
                if (job->kind == Kind::Weights) {
                    process_weights(job);
                } else {
                    process_request(job);
                }
            } catch (...) {
                job->error = std::current_exception();
                if (job->kind == Kind::Request) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    completed_.push_back(job);
                    result_ready_.notify_one();
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    failure_ = job->error;
                    result_ready_.notify_all();
                }
            }
            {
                std::lock_guard<std::mutex> lock(job->done_mutex);
                job->done = true;
            }
            job->done_cv.notify_all();
        }
    }

    void process_weights(const std::shared_ptr<Job>& job) {
        if (job->version <= active_version_) {
            throw std::invalid_argument(
                "target weight version must increase"
            );
        }
        target_.attr("synchronize")(
            job->representation,
            job->prediction,
            job->dynamics
        );
        cache_.clear();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_size_ = 0;
        }
        active_version_ = job->version;
        job->representation = py::none();
        job->prediction = py::none();
        job->dynamics = py::none();
    }

    void process_request(const std::shared_ptr<Job>& job) {
        if (job->version != active_version_) {
            throw std::runtime_error(
                "request target version does not match active target version"
            );
        }
        job->queue_wait_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - job->submitted
        ).count();
        py::object batch = job->batch;
        job->original_batch = batch;
        job->roots_requested = job->policy_mask.sum().item<std::int64_t>();
        const auto started = Clock::now();
        if (cache_targets_) {
            batch = prepare_cache(batch, job.get());
        } else {
            job->roots_searched = job->roots_requested;
        }
        if (job->roots_searched > 0) {
            job->result_batch = target_.attr("reanalyze_batch")(batch);
            if (cache_targets_) {
                job->value_targets = py::cast<at::Tensor>(
                    job->result_batch.attr("value_targets")
                );
                job->policy_targets = py::cast<at::Tensor>(
                    job->result_batch.attr("policy_targets")
                );
                resolve_cache(job.get());
            }
        } else {
            job->result_batch = batch;
        }
        if (cache_targets_) {
            job->result_batch = job->original_batch.attr(
                "with_reanalysis_targets"
            )(
                py::arg("value_targets") = job->value_targets,
                py::arg("policy_targets") = job->policy_targets
            );
        }
        job->worker_duration_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started
        ).count();
        try {
            py::module_ torch = py::module_::import("torch");
            if (py::cast<bool>(torch.attr("cuda").attr("is_available")())) {
                job->peak_memory_bytes = py::cast<std::int64_t>(
                    torch.attr("cuda").attr("max_memory_allocated")()
                );
            }
        } catch (...) {
            PyErr_Clear();
        }
        job->batch = py::none();
        job->original_batch = py::none();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_.push_back(job);
        }
        result_ready_.notify_one();
    }

    py::object prepare_cache(py::object batch, Job* job) {
        const at::Tensor positions = at::nonzero(job->policy_mask).contiguous();
        const at::Tensor indices = job->indices.contiguous();
        const auto* position_data = positions.data_ptr<std::int64_t>();
        const auto* index_data = indices.data_ptr<std::int64_t>();
        const auto state_count = job->policy_targets.size(1);
        const auto action_count = job->policy_targets.size(2);
        std::unordered_map<std::int64_t, std::size_t> miss_lookup;
        std::vector<std::pair<std::int64_t, const CacheEntry*>> hits;
        job->cache_misses.clear();
        for (std::int64_t row = 0; row < positions.size(0); ++row) {
            const auto sample = position_data[row * 2];
            const auto offset = position_data[row * 2 + 1];
            const auto flat = sample * state_count + offset;
            const auto state_id = index_data[sample] + offset;
            auto cached = cache_.find(state_id);
            if (cached != cache_.end()) {
                hits.emplace_back(flat, &cached->second);
                continue;
            }
            auto existing = miss_lookup.find(state_id);
            if (existing == miss_lookup.end()) {
                const auto index = job->cache_misses.size();
                miss_lookup[state_id] = index;
                job->cache_misses.push_back(CacheMiss{
                    state_id,
                    static_cast<std::size_t>(flat),
                    {static_cast<std::size_t>(flat)},
                });
            } else {
                job->cache_misses[existing->second].positions.push_back(
                    static_cast<std::size_t>(flat)
                );
            }
        }
        job->roots_searched = static_cast<std::int64_t>(
            job->cache_misses.size()
        );
        job->value_targets = job->value_targets.clone();
        job->policy_targets = job->policy_targets.clone();
        if (!hits.empty()) {
            std::vector<std::int64_t> hit_indices;
            std::vector<float> hit_values;
            std::vector<float> hit_policies;
            hit_indices.reserve(hits.size());
            hit_values.reserve(hits.size());
            hit_policies.reserve(hits.size() * action_count);
            for (const auto& [flat, entry] : hits) {
                hit_indices.push_back(flat);
                hit_values.push_back(entry->value);
                hit_policies.insert(
                    hit_policies.end(),
                    entry->policy.begin(),
                    entry->policy.end()
                );
            }
            const at::Tensor hit_index_tensor = vector_tensor(
                hit_indices, at::kLong
            );
            job->value_targets.view({-1}).index_put_(
                {hit_index_tensor}, vector_tensor(hit_values, at::kFloat)
            );
            job->policy_targets.view({-1, action_count}).index_put_(
                {hit_index_tensor},
                at::from_blob(
                    hit_policies.data(),
                    {static_cast<std::int64_t>(hits.size()), action_count},
                    at::TensorOptions().dtype(at::kFloat)
                ).clone()
            );
        }
        at::Tensor miss_mask = at::zeros_like(job->policy_mask);
        if (!job->cache_misses.empty()) {
            std::vector<std::int64_t> miss_indices;
            miss_indices.reserve(job->cache_misses.size());
            for (const CacheMiss& miss : job->cache_misses) {
                miss_indices.push_back(static_cast<std::int64_t>(miss.source));
            }
            miss_mask.view({-1}).index_fill_(
                0, vector_tensor(miss_indices, at::kLong), true
            );
        }
        std::optional<at::Tensor> bootstrap_mask;
        if (job->value_bootstrap_mask.has_value()) {
            bootstrap_mask = *job->value_bootstrap_mask & miss_mask;
        }
        py::module_ dataclasses = py::module_::import("dataclasses");
        return dataclasses.attr("replace")(
            batch,
            py::arg("value_targets") = job->value_targets,
            py::arg("policy_targets") = job->policy_targets,
            py::arg("policy_mask") = job->policy_mask & miss_mask,
            py::arg("value_bootstrap_mask") = bootstrap_mask
        );
    }

    void resolve_cache(Job* job) {
        const auto action_count = job->policy_targets.size(2);
        std::vector<std::int64_t> source_indices;
        source_indices.reserve(job->cache_misses.size());
        for (const CacheMiss& miss : job->cache_misses) {
            source_indices.push_back(static_cast<std::int64_t>(miss.source));
        }
        const at::Tensor source_tensor = vector_tensor(
            source_indices, at::kLong
        );
        const at::Tensor miss_values = job->value_targets.view({-1})
            .index_select(0, source_tensor)
            .contiguous();
        const at::Tensor miss_policies = job->policy_targets
            .view({-1, action_count})
            .index_select(0, source_tensor)
            .contiguous();
        const auto* value_data = miss_values.data_ptr<float>();
        const auto* policy_data = miss_policies.data_ptr<float>();
        std::vector<std::int64_t> duplicate_indices;
        std::vector<float> duplicate_values;
        std::vector<float> duplicate_policies;
        for (std::size_t index = 0; index < job->cache_misses.size(); ++index) {
            CacheEntry entry;
            entry.value = value_data[index];
            entry.policy.assign(
                policy_data + index * action_count,
                policy_data + (index + 1) * action_count
            );
            for (std::size_t duplicate = 1;
                 duplicate < job->cache_misses[index].positions.size();
                 ++duplicate) {
                duplicate_indices.push_back(static_cast<std::int64_t>(
                    job->cache_misses[index].positions[duplicate]
                ));
                duplicate_values.push_back(entry.value);
                duplicate_policies.insert(
                    duplicate_policies.end(),
                    entry.policy.begin(),
                    entry.policy.end()
                );
            }
            cache_[job->cache_misses[index].state_id] = std::move(entry);
        }
        if (!duplicate_indices.empty()) {
            const at::Tensor duplicate_index_tensor = vector_tensor(
                duplicate_indices, at::kLong
            );
            job->value_targets.view({-1}).index_put_(
                {duplicate_index_tensor},
                vector_tensor(duplicate_values, at::kFloat)
            );
            job->policy_targets.view({-1, action_count}).index_put_(
                {duplicate_index_tensor},
                at::from_blob(
                    duplicate_policies.data(),
                    {
                        static_cast<std::int64_t>(duplicate_indices.size()),
                        action_count,
                    },
                    at::TensorOptions().dtype(at::kFloat)
                ).clone()
            );
        }
        std::lock_guard<std::mutex> lock(mutex_);
        cache_size_ = cache_.size();
    }

    void close_impl() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
        }
        work_ready_.notify_all();
        result_ready_.notify_all();
        if (worker_.joinable()) {
            if (PyGILState_Check()) {
                py::gil_scoped_release release;
                worker_.join();
            } else {
                worker_.join();
            }
        }
        py::gil_scoped_acquire acquire;
        jobs_.clear();
        completed_.clear();
        pending_.clear();
        cache_.clear();
        target_ = py::none();
    }

    std::int64_t active_version_ = -1;
};

NativeReanalysisEngine::NativeReanalysisEngine(
    py::object target,
    int prefetch_batches,
    double timeout_seconds,
    int target_update_interval,
    bool cache_targets
)
    : impl_(std::make_unique<Impl>(
          std::move(target),
          prefetch_batches,
          timeout_seconds,
          target_update_interval,
          cache_targets
      )) {}

NativeReanalysisEngine::~NativeReanalysisEngine() = default;

void NativeReanalysisEngine::publish_weights(
    std::int64_t version,
    py::object representation,
    py::object prediction,
    py::object dynamics
) {
    impl_->publish_weights(
        version,
        std::move(representation),
        std::move(prediction),
        std::move(dynamics)
    );
}

std::int64_t NativeReanalysisEngine::submit(py::object batch) {
    return impl_->submit(std::move(batch));
}

py::dict NativeReanalysisEngine::wait_next() { return impl_->wait_next(); }

void NativeReanalysisEngine::close() { impl_->close(); }

std::size_t NativeReanalysisEngine::pending_count() const {
    return impl_->pending_count();
}

std::size_t NativeReanalysisEngine::max_pending() const {
    return impl_->max_pending();
}

std::size_t NativeReanalysisEngine::cache_size() const {
    return impl_->cache_size();
}

std::int64_t NativeReanalysisEngine::weight_version() const {
    return impl_->weight_version();
}

}  // namespace atariagent::native
