#include <torch/extension.h>

#include "reanalysis/engine.h"

#include "models/state_dict.h"
#include "reanalysis/cache.h"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef ATARIAGENT_HAS_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace py = pybind11;

namespace atariagent::native {

class NativeReanalysisEngine::Impl {
public:
    Impl(
        std::shared_ptr<ValueTargetNetwork> target,
        const std::string& device,
        int prefetch_batches,
        double timeout_seconds,
        int target_update_interval,
        bool cache_targets,
        std::int64_t cache_target_ttl,
        std::shared_ptr<ValueTargetNetwork> bootstrap_target,
        int bootstrap_update_interval
    )
        : target_(std::move(target)),
          bootstrap_target_(bootstrap_target ? std::move(bootstrap_target) : target_),
          device_(device),
          prefetch_batches_(prefetch_batches),
          timeout_seconds_(timeout_seconds),
          target_update_interval_(target_update_interval),
          bootstrap_update_interval_(bootstrap_update_interval == 0
              ? target_update_interval : bootstrap_update_interval),
          cache_targets_(cache_targets),
          cache_target_ttl_(cache_target_ttl) {
        if (!target_) {
            throw std::invalid_argument("target must not be null");
        }
        if (prefetch_batches_ <= 0) {
            throw std::invalid_argument("prefetch_batches must be positive");
        }
        if (!std::isfinite(timeout_seconds_) || timeout_seconds_ <= 0.0) {
            throw std::invalid_argument("timeout_seconds must be positive");
        }
        if (target_update_interval_ <= 0) {
            throw std::invalid_argument(
                "target_update_interval must be positive"
            );
        }
        if (bootstrap_update_interval_ <= 0) {
            throw std::invalid_argument("bootstrap_update_interval must be positive");
        }
        if (cache_target_ttl_ < 0) {
            throw std::invalid_argument(
                "cache_target_ttl must be non-negative"
            );
        }
#ifdef ATARIAGENT_HAS_CUDA
        if (device_.is_cuda() && !device_.has_index()) {
            device_ = torch::Device(
                torch::kCUDA, c10::cuda::current_device()
            );
        }
#endif
        target_->to(device_.str());
        if (bootstrap_target_ != target_) {
            bootstrap_target_->to(device_.str());
        }
        worker_ = std::thread(&Impl::run, this);
    }

    ~Impl() { close_impl(); }

    void publish_weights(
        std::int64_t version,
        TensorState representation,
        TensorState prediction,
        TensorState dynamics,
        bool policy,
        bool bootstrap
    ) {
        if (!policy && !bootstrap) {
            throw std::invalid_argument("publication must select at least one target");
        }
        if (policy != bootstrap && bootstrap_target_ == target_) {
            throw std::invalid_argument("split publication requires a separate bootstrap target");
        }
        auto job = std::make_shared<Job>();
        job->kind = Kind::Weights;
        job->publish_policy = policy;
        job->publish_bootstrap = bootstrap;
        job->version = version;
        job->representation = std::move(representation);
        job->prediction = std::move(prediction);
        job->dynamics = std::move(dynamics);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require_open_locked();
            if (version < 0
                || (policy && version <= published_version_)
                || (bootstrap && version <= published_bootstrap_version_)) {
                throw std::invalid_argument(
                    "target weight version must increase"
                );
            }
            if (policy) {
                published_version_ = version;
                ++policy_generation_;
            }
            if (bootstrap) {
                published_bootstrap_version_ = version;
                ++bootstrap_generation_;
            }
            jobs_.push_back(job);
        }
        work_ready_.notify_one();
        wait_job(job, "timed out publishing target weights");
    }

    std::int64_t submit(
        py::object batch,
        double root_noise_temperature,
        bool gumbel_sampling,
        std::int64_t trained_step
    ) {
        if (!std::isfinite(root_noise_temperature)
            || root_noise_temperature < 0.0
            || root_noise_temperature > 1.0) {
            throw std::invalid_argument(
                "root_noise_temperature must be in [0, 1]"
            );
        }
        if (trained_step < 0) {
            throw std::invalid_argument("trained_step must be non-negative");
        }
        auto job = std::make_shared<Job>();
        job->root_noise_temperature = root_noise_temperature;
        job->gumbel_sampling = gumbel_sampling;
        job->trained_step = trained_step;
        job->kind = Kind::Request;
        job->original_batch = std::move(batch);
        job->frames = tensor_attribute(job->original_batch, "frames");
        job->policy_mask = tensor_attribute(job->original_batch, "policy_mask");
        job->policy_targets = tensor_attribute(
            job->original_batch, "policy_targets"
        );
        job->value_targets = tensor_attribute(
            job->original_batch, "value_targets"
        );
        job->indices = tensor_attribute(job->original_batch, "indices");
        job->reanalysis_state_ids = optional_tensor_attribute(
            job->original_batch, "reanalysis_state_ids"
        );
        job->bootstrap = bootstrap_attributes(job->original_batch);
        job->stack_size = py::cast<std::int64_t>(
            job->original_batch.attr("stack_size")
        );
        job->reanalysis_frames = tensor_attribute(
            job->original_batch, "reanalysis_frames"
        );
        validate_batch(*job);
        if (cache_targets_ && !job->reanalysis_state_ids) {
            throw std::invalid_argument(
                "cached reanalysis requires reanalysis_state_ids"
            );
        }
        if (cache_targets_ && !job->bootstrap.state_ids) {
            throw std::invalid_argument(
                "cached reanalysis requires value_bootstrap_state_ids"
            );
        }
        job->submitted = Clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require_open_locked();
            if (published_version_ < 0 || published_bootstrap_version_ < 0) {
                throw std::runtime_error(
                    "target weights must be published before submission"
                );
            }
            if (pending_count_locked() >= prefetch_batches_) {
                PyErr_SetString(
                    PyExc_BufferError, "reanalysis prefetch limit reached"
                );
                throw py::error_already_set();
            }
            job->request_id = next_request_id_++;
            job->version = published_version_;
            job->bootstrap_version = published_bootstrap_version_;
            job->policy_generation = policy_generation_;
            job->bootstrap_generation = bootstrap_generation_;
            jobs_.push_back(job);
            ++pending_count_;
            max_pending_ = std::max(max_pending_, pending_count_);
        }
        work_ready_.notify_one();
        return job->request_id;
    }

    void clear_cache() {
        auto job = std::make_shared<Job>();
        job->kind = Kind::CacheClear;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            require_open_locked();
            jobs_.push_back(job);
        }
        work_ready_.notify_one();
    }

    py::dict wait_next() {
        std::shared_ptr<Job> job;
        bool timed_out = false;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            const auto timeout = std::chrono::duration<double>(timeout_seconds_);
            timed_out = !result_ready_.wait_for(lock, timeout, [this] {
                return !completed_.empty() || failure_ || closed_;
            });
            if (!timed_out) {
                if (failure_) {
                    std::rethrow_exception(failure_);
                }
                if (completed_.empty()) {
                    throw std::runtime_error("reanalysis engine is closed");
                }
                job = completed_.front();
                completed_.pop_front();
                --pending_count_;
            }
        }
        if (timed_out) {
            PyErr_SetString(
                PyExc_TimeoutError,
                "timed out waiting for asynchronous reanalysis"
            );
            throw py::error_already_set();
        }
        if (job->error) {
            std::rethrow_exception(job->error);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Count publications rather than subtract learner-step versions:
            // a resume with shorter intervals can legitimately skip versions
            // on its first publication. Permit one intervening copy per path.
            if (policy_generation_ - job->policy_generation > 1
                || bootstrap_generation_ - job->bootstrap_generation > 1) {
                throw std::runtime_error(
                    "reanalysis result is older than one target update interval"
                );
            }
        }
        py::object result_batch = job->original_batch.attr(
            "with_reanalysis_targets"
        )(
            py::arg("value_targets") = job->value_targets,
            py::arg("policy_targets") = job->policy_targets,
            py::arg("search_value_targets") = job->search_value_targets
        );
        py::dict result;
        result["request_id"] = job->request_id;
        result["weight_version"] = job->version;
        result["bootstrap_weight_version"] = job->bootstrap_version;
        result["batch"] = std::move(result_batch);
        result["queue_wait_ms"] = job->queue_wait_ms;
        result["worker_duration_ms"] = job->worker_duration_ms;
        result["peak_memory_bytes"] = job->peak_memory_bytes;
        result["policy_roots_requested"] = job->roots_requested;
        result["policy_roots_searched"] = job->roots_searched;
        result["cache_hits"] = job->cache_hits;
        result["value_roots_requested"] = job->value_roots_requested;
        result["value_roots_searched"] = job->value_roots_searched;
        result["value_cache_hits"] = job->value_cache_hits;
        result["cache_target_age_mean"] = job->cache_hits > 0
            ? job->cache_target_age_sum
                / static_cast<double>(job->cache_hits)
            : 0.0;
        result["cache_target_age_max"] = job->cache_target_age_max;
        result["cache_size"] = cache_size_;
        job->original_batch = py::none();
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

    std::size_t value_cache_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_cache_size_;
    }

    std::int64_t weight_version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return published_version_;
    }

    std::int64_t bootstrap_weight_version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return published_bootstrap_version_;
    }

private:
    using Clock = std::chrono::steady_clock;
    enum class Kind { Request, Weights, CacheClear };

    struct BootstrapData {
        torch::Tensor frames;
        torch::Tensor values;
        torch::Tensor discounts;
        torch::Tensor mask;
        std::optional<torch::Tensor> state_ids;
    };

    struct DeviceFrameViews {
        torch::Tensor policy_frames;
        torch::Tensor bootstrap_frames;
    };

    struct Job {
        Kind kind = Kind::Request;
        std::int64_t request_id = -1;
        std::int64_t version = -1;
        std::int64_t bootstrap_version = -1;
        std::int64_t policy_generation = 0;
        std::int64_t bootstrap_generation = 0;
        bool publish_policy = false;
        bool publish_bootstrap = false;
        py::object original_batch;
        torch::Tensor frames;
        torch::Tensor policy_mask;
        torch::Tensor policy_targets;
        torch::Tensor value_targets;
        torch::Tensor search_value_targets;
        torch::Tensor indices;
        std::optional<torch::Tensor> reanalysis_state_ids;
        BootstrapData bootstrap;
        torch::Tensor reanalysis_frames;
        std::int64_t stack_size = 0;
        TensorState representation;
        TensorState prediction;
        TensorState dynamics;
        std::vector<CacheMiss> cache_misses;
        std::vector<CacheMiss> value_cache_misses;
        torch::Tensor fresh_bootstrap_values;
        Clock::time_point submitted;
        double queue_wait_ms = 0.0;
        double worker_duration_ms = 0.0;
        std::int64_t peak_memory_bytes = 0;
        std::int64_t roots_requested = 0;
        std::int64_t roots_searched = 0;
        std::int64_t cache_hits = 0;
        std::int64_t value_roots_requested = 0;
        std::int64_t value_roots_searched = 0;
        std::int64_t value_cache_hits = 0;
        double cache_target_age_sum = 0.0;
        std::int64_t cache_target_age_max = 0;
        std::int64_t trained_step = 0;
        double root_noise_temperature = 0.0;
        bool gumbel_sampling = false;
        std::exception_ptr error;
        bool done = false;
        std::mutex done_mutex;
        std::condition_variable done_cv;
    };

    static torch::Tensor tensor_attribute(
        const py::object& object,
        const char* name
    ) {
        py::object value = object.attr(name);
        if (value.is_none()) {
            throw std::invalid_argument(
                std::string(name) + " is required for native reanalysis"
            );
        }
        return py::cast<torch::Tensor>(value);
    }

    static std::optional<torch::Tensor> optional_tensor_attribute(
        const py::object& object,
        const char* name
    ) {
        py::object value = object.attr(name);
        if (value.is_none()) {
            return std::nullopt;
        }
        return py::cast<torch::Tensor>(value);
    }

    static BootstrapData bootstrap_attributes(const py::object& object) {
        return BootstrapData{
            tensor_attribute(object, "value_bootstrap_frames"),
            tensor_attribute(object, "value_bootstrap_values"),
            tensor_attribute(object, "value_bootstrap_discounts"),
            tensor_attribute(object, "value_bootstrap_mask"),
            optional_tensor_attribute(object, "value_bootstrap_state_ids"),
        };
    }

    static void require_cpu(const torch::Tensor& tensor, const char* name) {
        if (!tensor.device().is_cpu()) {
            throw std::invalid_argument(
                std::string(name) + " must be a CPU tensor"
            );
        }
    }

    static void validate_batch(const Job& job) {
        require_cpu(job.frames, "frames");
        require_cpu(job.policy_mask, "policy_mask");
        require_cpu(job.policy_targets, "policy_targets");
        require_cpu(job.value_targets, "value_targets");
        require_cpu(job.indices, "indices");
        if (job.policy_mask.scalar_type() != torch::kBool
            || job.policy_mask.dim() != 2) {
            throw std::invalid_argument("policy_mask must be a 2D bool tensor");
        }
        if (job.policy_targets.scalar_type() != torch::kFloat
            || job.policy_targets.dim() != 3
            || !job.policy_targets.is_contiguous()) {
            throw std::invalid_argument(
                "policy_targets must be a contiguous 3D float32 tensor"
            );
        }
        if (job.value_targets.scalar_type() != torch::kFloat
            || job.value_targets.dim() != 2
            || !job.value_targets.is_contiguous()) {
            throw std::invalid_argument(
                "value_targets must be a contiguous 2D float32 tensor"
            );
        }
        if (job.indices.scalar_type() != torch::kLong
            || job.indices.dim() != 1) {
            throw std::invalid_argument("indices must be a 1D int64 tensor");
        }
        if (job.frames.dim() != 5) {
            throw std::invalid_argument("frames must be a 5D tensor");
        }
        if (job.policy_targets.size(0) != job.policy_mask.size(0)
            || job.policy_targets.size(1) != job.policy_mask.size(1)
            || job.value_targets.sizes() != job.policy_mask.sizes()
            || job.indices.size(0) != job.policy_mask.size(0)) {
            throw std::invalid_argument(
                "reanalysis batch tensors have incompatible shapes"
            );
        }
        if (job.reanalysis_state_ids) {
            require_cpu(*job.reanalysis_state_ids, "reanalysis_state_ids");
            if (job.reanalysis_state_ids->scalar_type() != torch::kLong
                || job.reanalysis_state_ids->sizes()
                    != job.policy_mask.sizes()
                || !job.reanalysis_state_ids->is_contiguous()) {
                throw std::invalid_argument(
                    "reanalysis_state_ids must be a contiguous int64 tensor "
                    "matching policy_mask"
                );
            }
        }
        const BootstrapData& bootstrap = job.bootstrap;
        require_cpu(job.reanalysis_frames, "reanalysis_frames");
        if (job.reanalysis_frames.dim() != 5
            || job.reanalysis_frames.size(0) != job.frames.size(0)
            || job.reanalysis_frames.size(1) < job.frames.size(1)
            || job.reanalysis_frames.size(1) < bootstrap.frames.size(1)) {
            throw std::invalid_argument(
                "reanalysis_frames has an invalid shape"
            );
        }
        require_cpu(bootstrap.mask, "value_bootstrap_mask");
        if (bootstrap.mask.scalar_type() != torch::kBool
            || bootstrap.mask.sizes() != job.policy_mask.sizes()) {
            throw std::invalid_argument(
                "value_bootstrap_mask must match policy_mask"
            );
        }
        if (bootstrap.state_ids) {
            require_cpu(*bootstrap.state_ids, "value_bootstrap_state_ids");
            if (bootstrap.state_ids->scalar_type() != torch::kLong
                || bootstrap.state_ids->sizes() != job.policy_mask.sizes()
                || !bootstrap.state_ids->is_contiguous()) {
                throw std::invalid_argument(
                    "value_bootstrap_state_ids must be a contiguous int64 "
                    "tensor matching policy_mask"
                );
            }
        }
        require_cpu(bootstrap.frames, "value_bootstrap_frames");
        require_cpu(bootstrap.values, "value_bootstrap_values");
        if (bootstrap.values.scalar_type() != torch::kFloat
            || bootstrap.values.dim() != 2
            || bootstrap.values.sizes() != bootstrap.mask.sizes()) {
            throw std::invalid_argument(
                "value_bootstrap_values must be a 2D float32 tensor matching "
                "value_bootstrap_mask"
            );
        }
        require_cpu(bootstrap.discounts, "value_bootstrap_discounts");
        if (bootstrap.discounts.scalar_type() != torch::kFloat
            || bootstrap.discounts.dim() != 2
            || bootstrap.discounts.sizes() != bootstrap.mask.sizes()) {
            throw std::invalid_argument(
                "value_bootstrap_discounts must be a 2D float32 tensor "
                "matching value_bootstrap_mask"
            );
        }
        if (job.stack_size <= 0) {
            throw std::invalid_argument("stack_size must be positive");
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
        bool timed_out = false;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(job->done_mutex);
            const auto timeout = std::chrono::duration<double>(timeout_seconds_);
            timed_out = !job->done_cv.wait_for(
                lock, timeout, [&job] { return job->done; }
            );
        }
        if (timed_out) {
            PyErr_SetString(PyExc_TimeoutError, message);
            throw py::error_already_set();
        }
        if (job->error) {
            std::rethrow_exception(job->error);
        }
    }

    void run() {
        while (true) {
            std::shared_ptr<Job> job;
            {
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
                } else if (job->kind == Kind::CacheClear) {
                    process_policy_cache_clear();
                } else {
                    process_request(job);
                }
            } catch (...) {
                job->error = std::current_exception();
                if (job->kind != Kind::Request) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    failure_ = job->error;
                    result_ready_.notify_all();
                }
            }
            if (job->kind == Kind::Request) {
                complete_request(job);
            }
            {
                std::lock_guard<std::mutex> lock(job->done_mutex);
                job->done = true;
            }
            job->done_cv.notify_all();
        }
    }

    void complete_request(const std::shared_ptr<Job>& job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_.push_back(job);
        }
        result_ready_.notify_one();
    }

    void process_weights(const std::shared_ptr<Job>& job) {
        if (job->publish_policy) {
            target_->synchronize(job->representation, job->prediction, job->dynamics);
            process_policy_cache_clear();
            active_version_ = job->version;
        }
        if (job->publish_bootstrap) {
            if (bootstrap_target_ != target_) {
                bootstrap_target_->synchronize(
                    job->representation, job->prediction, job->dynamics
                );
            }
            cache_.clear_values();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                value_cache_size_ = 0;
            }
            active_bootstrap_version_ = job->version;
        }
        job->representation.clear();
        job->prediction.clear();
        job->dynamics.clear();
    }

    void process_policy_cache_clear() {
        cache_.clear_policy();
        std::lock_guard<std::mutex> lock(mutex_);
        cache_size_ = 0;
    }

    void process_request(const std::shared_ptr<Job>& job) {
        if (job->version != active_version_
            || job->bootstrap_version != active_bootstrap_version_) {
            throw std::runtime_error(
                "request target version does not match active target version"
            );
        }
        job->queue_wait_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - job->submitted
        ).count();
        job->roots_requested = job->policy_mask.sum().item<std::int64_t>();
        job->value_roots_requested = job->bootstrap.mask.sum()
            .item<std::int64_t>();
        const auto started = Clock::now();
        reset_peak_memory();

        BootstrapData& bootstrap = job->bootstrap;
        torch::Tensor effective_policy_mask = job->policy_mask;
        torch::Tensor effective_value_mask = bootstrap.mask;
        job->value_targets = job->value_targets.clone();
        if (cache_targets_) {
            CachePreparation prepared = cache_.prepare_policy(
                job->policy_mask,
                job->policy_targets,
                *job->reanalysis_state_ids,
                job->trained_step,
                cache_target_ttl_
            );
            job->cache_misses = std::move(prepared.misses);
            job->roots_searched = prepared.roots_searched;
            job->cache_hits = prepared.cache_hits;
            job->cache_target_age_sum = prepared.cache_target_age_sum;
            job->cache_target_age_max = prepared.cache_target_age_max;
            job->search_value_targets = std::move(
                prepared.search_value_targets
            );
            job->policy_targets = std::move(prepared.policy_targets);
            effective_policy_mask = std::move(prepared.miss_mask);

            // Cache only the deterministic raw endpoint prediction. Corrected
            // TD targets remain occurrence-specific because masks, stored
            // bootstrap terms, and discounts can differ for the same state.
            ValueCachePreparation values = cache_.prepare_values(
                bootstrap.mask,
                *bootstrap.state_ids
            );
            job->value_cache_misses = std::move(values.misses);
            job->value_roots_searched = values.roots_searched;
            job->value_cache_hits = values.cache_hits;
            job->fresh_bootstrap_values = std::move(values.values);
            effective_value_mask = std::move(values.miss_mask);
        } else {
            job->roots_searched = job->roots_requested;
            job->value_roots_searched = job->value_roots_requested;
            job->search_value_targets = torch::zeros_like(
                job->value_targets
            );
            job->policy_targets = job->policy_targets.clone();
            job->fresh_bootstrap_values = torch::zeros(
                bootstrap.values.sizes(), bootstrap.values.options()
            );
        }

        if (job->roots_searched > 0 || job->value_roots_searched > 0) {
            run_target(
                job.get(),
                effective_policy_mask,
                effective_value_mask
            );
        }
        if (cache_targets_) {
            cache_.resolve_policy(
                job->cache_misses,
                job->search_value_targets,
                job->policy_targets,
                job->trained_step
            );
            cache_.resolve_values(
                job->value_cache_misses,
                job->fresh_bootstrap_values
            );
            std::lock_guard<std::mutex> lock(mutex_);
            cache_size_ = cache_.policy_size();
            value_cache_size_ = cache_.value_size();
        }
        if (job->value_roots_requested > 0) {
            torch::Tensor delta = (
                job->fresh_bootstrap_values - bootstrap.values
            ) * bootstrap.discounts;
            job->value_targets = torch::where(
                bootstrap.mask,
                job->value_targets + delta,
                job->value_targets
            );
        }
        job->worker_duration_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started
        ).count();
        job->peak_memory_bytes = peak_memory();
        release_request_inputs(job.get());
    }

    DeviceFrameViews transfer_frames(const Job& job) const {
        torch::Tensor combined = job.reanalysis_frames.to(device_);
        const auto bootstrap_offset = combined.size(1)
            - job.bootstrap.frames.size(1);
        return DeviceFrameViews{
            combined.narrow(1, 0, job.frames.size(1)),
            combined.narrow(
                1,
                bootstrap_offset,
                job.bootstrap.frames.size(1)
            ),
        };
    }

    void reanalyze_policy_roots(
        Job* job,
        const torch::Tensor& device_frames,
        const torch::Tensor& policy_mask
    ) {
        auto [positions, policies, search_values] =
            target_->policy_reanalysis_outputs(
                device_frames,
                policy_mask.to(device_),
                job->stack_size,
                job->root_noise_temperature,
                job->gumbel_sampling,
                false
            );
        const auto root_count = positions.size(0);
        const auto state_count = job->policy_targets.size(1);
        const auto action_count = job->policy_targets.size(2);
        if (policies.sizes() != torch::IntArrayRef({
                root_count, action_count
            })
            || search_values.sizes() != torch::IntArrayRef({root_count})) {
            throw std::runtime_error(
                "policy reanalysis returned incompatible outputs"
            );
        }
        const auto* position_data = positions.data_ptr<std::int64_t>();
        const auto* policy_data = policies.data_ptr<float>();
        const auto* search_value_data = search_values.data_ptr<float>();
        auto* target_policy_data = job->policy_targets.data_ptr<float>();
        auto* target_search_value_data =
            job->search_value_targets.data_ptr<float>();
        for (std::int64_t root = 0; root < root_count; ++root) {
            const auto flat = position_data[root * 2] * state_count
                + position_data[root * 2 + 1];
            std::copy_n(
                policy_data + root * action_count,
                action_count,
                target_policy_data + flat * action_count
            );
            target_search_value_data[flat] = search_value_data[root];
        }
    }

    void predict_bootstrap_values(
        Job* job,
        const torch::Tensor& device_bootstrap_frames,
        const torch::Tensor& bootstrap_mask
    ) {
        auto [positions, values] = bootstrap_target_->value_predictions(
            device_bootstrap_frames,
            bootstrap_mask.to(device_),
            job->stack_size
        );
        const auto root_count = positions.size(0);
        if (values.sizes() != torch::IntArrayRef({root_count})) {
            throw std::runtime_error(
                "value reanalysis returned incompatible outputs"
            );
        }
        const auto state_count = job->fresh_bootstrap_values.size(1);
        const auto* position_data = positions.data_ptr<std::int64_t>();
        const auto* value_data = values.data_ptr<float>();
        auto* fresh_value_data =
            job->fresh_bootstrap_values.data_ptr<float>();
        for (std::int64_t root = 0; root < root_count; ++root) {
            const auto flat = position_data[root * 2] * state_count
                + position_data[root * 2 + 1];
            fresh_value_data[flat] = value_data[root];
        }
    }

    void run_target(
        Job* job,
        const torch::Tensor& policy_mask,
        const torch::Tensor& value_mask
    ) {
        DeviceFrameViews frames = transfer_frames(*job);
        if (job->roots_searched > 0) {
            reanalyze_policy_roots(job, frames.policy_frames, policy_mask);
        }
        if (job->value_roots_searched > 0) {
            predict_bootstrap_values(
                job,
                frames.bootstrap_frames,
                value_mask
            );
        }
    }

    static void release_request_inputs(Job* job) {
        job->frames = torch::Tensor();
        job->policy_mask = torch::Tensor();
        job->indices = torch::Tensor();
        job->reanalysis_state_ids.reset();
        job->bootstrap = BootstrapData{};
        job->reanalysis_frames = torch::Tensor();
        job->fresh_bootstrap_values = torch::Tensor();
        job->cache_misses.clear();
        job->value_cache_misses.clear();
    }

    void reset_peak_memory() const {
#ifdef ATARIAGENT_HAS_CUDA
        if (device_.is_cuda()) {
            c10::cuda::CUDACachingAllocator::resetPeakStats(device_.index());
        }
#endif
    }

    std::int64_t peak_memory() const {
#ifdef ATARIAGENT_HAS_CUDA
        if (device_.is_cuda()) {
            const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(
                device_.index()
            );
            return stats.allocated_bytes[
                static_cast<std::size_t>(
                    c10::CachingAllocator::StatType::AGGREGATE
                )
            ].peak;
        }
#endif
        return 0;
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
        pending_count_ = 0;
        cache_.clear_all();
        target_.reset();
        bootstrap_target_.reset();
    }

    std::size_t pending_count_locked() const { return pending_count_; }

    std::shared_ptr<ValueTargetNetwork> target_;
    std::shared_ptr<ValueTargetNetwork> bootstrap_target_;
    torch::Device device_;
    int prefetch_batches_;
    double timeout_seconds_;
    int target_update_interval_;
    int bootstrap_update_interval_;
    bool cache_targets_;
    std::int64_t cache_target_ttl_;
    mutable std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable result_ready_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::deque<std::shared_ptr<Job>> completed_;
    ReanalysisCache cache_;
    std::thread worker_;
    std::exception_ptr failure_;
    bool closed_ = false;
    std::int64_t next_request_id_ = 0;
    std::int64_t published_version_ = -1;
    std::int64_t active_version_ = -1;
    std::int64_t published_bootstrap_version_ = -1;
    std::int64_t active_bootstrap_version_ = -1;
    std::int64_t policy_generation_ = 0;
    std::int64_t bootstrap_generation_ = 0;
    std::size_t pending_count_ = 0;
    std::size_t max_pending_ = 0;
    std::size_t cache_size_ = 0;
    std::size_t value_cache_size_ = 0;
};

NativeReanalysisEngine::NativeReanalysisEngine(
    std::shared_ptr<ValueTargetNetwork> target,
    const std::string& device,
    int prefetch_batches,
    double timeout_seconds,
    int target_update_interval,
    bool cache_targets,
    std::int64_t cache_target_ttl,
    std::shared_ptr<ValueTargetNetwork> bootstrap_target,
    int bootstrap_update_interval
)
    : impl_(std::make_unique<Impl>(
          std::move(target),
          device,
          prefetch_batches,
          timeout_seconds,
          target_update_interval,
          cache_targets,
          cache_target_ttl,
          std::move(bootstrap_target),
          bootstrap_update_interval
      )) {}

NativeReanalysisEngine::~NativeReanalysisEngine() = default;

void NativeReanalysisEngine::publish_weights(
    std::int64_t version,
    const py::dict& representation,
    const py::dict& prediction,
    const py::dict& dynamics,
    bool policy,
    bool bootstrap
) {
    impl_->publish_weights(
        version,
        tensor_state_from_dict(representation),
        tensor_state_from_dict(prediction),
        tensor_state_from_dict(dynamics),
        policy,
        bootstrap
    );
}

std::int64_t NativeReanalysisEngine::submit(
    py::object batch,
    double root_noise_temperature,
    bool gumbel_sampling,
    std::int64_t trained_step
) {
    return impl_->submit(
        std::move(batch),
        root_noise_temperature,
        gumbel_sampling,
        trained_step
    );
}

py::dict NativeReanalysisEngine::wait_next() { return impl_->wait_next(); }

void NativeReanalysisEngine::clear_cache() { impl_->clear_cache(); }

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

std::size_t NativeReanalysisEngine::value_cache_size() const {
    return impl_->value_cache_size();
}

std::int64_t NativeReanalysisEngine::weight_version() const {
    return impl_->weight_version();
}

std::int64_t NativeReanalysisEngine::bootstrap_weight_version() const {
    return impl_->bootstrap_weight_version();
}

}  // namespace atariagent::native
