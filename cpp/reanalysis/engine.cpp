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
#include <unordered_map>
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
        double cache_refresh_probability
    )
        : target_(std::move(target)),
          device_(device),
          prefetch_batches_(prefetch_batches),
          timeout_seconds_(timeout_seconds),
          target_update_interval_(target_update_interval),
          cache_targets_(cache_targets),
          cache_refresh_probability_(cache_refresh_probability) {
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
        if (!std::isfinite(cache_refresh_probability_)
            || cache_refresh_probability_ < 0.0
            || cache_refresh_probability_ > 1.0) {
            throw std::invalid_argument(
                "cache_refresh_probability must be in [0, 1]"
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
        worker_ = std::thread(&Impl::run, this);
    }

    ~Impl() { close_impl(); }

    void publish_weights(
        std::int64_t version,
        TensorState representation,
        TensorState prediction,
        TensorState dynamics
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

    std::int64_t submit(
        py::object batch,
        double root_noise_temperature,
        bool gumbel_sampling
    ) {
        if (!std::isfinite(root_noise_temperature)
            || root_noise_temperature < 0.0
            || root_noise_temperature > 1.0) {
            throw std::invalid_argument(
                "root_noise_temperature must be in [0, 1]"
            );
        }
        auto job = std::make_shared<Job>();
        job->root_noise_temperature = root_noise_temperature;
        job->gumbel_sampling = gumbel_sampling;
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
        job->stack_size = py::cast<std::int64_t>(
            job->original_batch.attr("stack_size")
        );
        job->value_bootstrap_frames = optional_tensor_attribute(
            job->original_batch, "value_bootstrap_frames"
        );
        job->reanalysis_frames = optional_tensor_attribute(
            job->original_batch, "reanalysis_frames"
        );
        job->value_bootstrap_values = optional_tensor_attribute(
            job->original_batch, "value_bootstrap_values"
        );
        job->value_bootstrap_discounts = optional_tensor_attribute(
            job->original_batch, "value_bootstrap_discounts"
        );
        job->value_bootstrap_mask = optional_tensor_attribute(
            job->original_batch, "value_bootstrap_mask"
        );
        validate_batch(*job);
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
                    PyExc_BufferError, "reanalysis prefetch limit reached"
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
                pending_.erase(job->request_id);
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
        const auto lag = published_version_ - job->version;
        if (lag < 0 || lag > target_update_interval_) {
            throw std::runtime_error(
                "reanalysis result is older than one target update interval"
            );
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
        result["batch"] = std::move(result_batch);
        result["queue_wait_ms"] = job->queue_wait_ms;
        result["worker_duration_ms"] = job->worker_duration_ms;
        result["peak_memory_bytes"] = job->peak_memory_bytes;
        result["policy_roots_requested"] = job->roots_requested;
        result["policy_roots_searched"] = job->roots_searched;
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

    std::int64_t weight_version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return published_version_;
    }

private:
    using Clock = std::chrono::steady_clock;
    enum class Kind { Request, Weights, CacheClear };

    struct Job {
        Kind kind = Kind::Request;
        std::int64_t request_id = -1;
        std::int64_t version = -1;
        py::object original_batch;
        torch::Tensor frames;
        torch::Tensor policy_mask;
        torch::Tensor policy_targets;
        torch::Tensor value_targets;
        torch::Tensor search_value_targets;
        torch::Tensor indices;
        std::optional<torch::Tensor> value_bootstrap_frames;
        std::optional<torch::Tensor> reanalysis_frames;
        std::optional<torch::Tensor> value_bootstrap_values;
        std::optional<torch::Tensor> value_bootstrap_discounts;
        std::optional<torch::Tensor> value_bootstrap_mask;
        std::int64_t stack_size = 0;
        TensorState representation;
        TensorState prediction;
        TensorState dynamics;
        std::vector<CacheMiss> cache_misses;
        Clock::time_point submitted;
        double queue_wait_ms = 0.0;
        double worker_duration_ms = 0.0;
        std::int64_t peak_memory_bytes = 0;
        std::int64_t roots_requested = 0;
        std::int64_t roots_searched = 0;
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
        return py::cast<torch::Tensor>(object.attr(name));
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
        if (job.reanalysis_frames) {
            require_cpu(*job.reanalysis_frames, "reanalysis_frames");
            if (job.reanalysis_frames->dim() != 5
                || job.reanalysis_frames->size(0) != job.frames.size(0)
                || job.reanalysis_frames->size(1) < job.frames.size(1)) {
                throw std::invalid_argument(
                    "reanalysis_frames has an invalid shape"
                );
            }
        }
        if (job.value_bootstrap_mask) {
            require_cpu(*job.value_bootstrap_mask, "value_bootstrap_mask");
            if (job.value_bootstrap_mask->scalar_type() != torch::kBool
                || job.value_bootstrap_mask->sizes()
                    != job.policy_mask.sizes()) {
                throw std::invalid_argument(
                    "value_bootstrap_mask must match policy_mask"
                );
            }
        }
        const bool any_bootstrap = job.value_bootstrap_frames.has_value()
            || job.value_bootstrap_values.has_value()
            || job.value_bootstrap_discounts.has_value()
            || job.value_bootstrap_mask.has_value();
        const bool all_bootstrap = job.value_bootstrap_frames.has_value()
            && job.value_bootstrap_values.has_value()
            && job.value_bootstrap_discounts.has_value()
            && job.value_bootstrap_mask.has_value();
        if (any_bootstrap != all_bootstrap) {
            throw std::invalid_argument(
                "batch has incomplete value-bootstrap metadata"
            );
        }
        if (all_bootstrap) {
            require_cpu(*job.value_bootstrap_frames, "value_bootstrap_frames");
            require_cpu(*job.value_bootstrap_values, "value_bootstrap_values");
            require_cpu(
                *job.value_bootstrap_discounts,
                "value_bootstrap_discounts"
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
                    process_cache_clear();
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
        target_->synchronize(
            job->representation,
            job->prediction,
            job->dynamics
        );
        process_cache_clear();
        active_version_ = job->version;
        job->representation.clear();
        job->prediction.clear();
        job->dynamics.clear();
    }

    void process_cache_clear() {
        cache_.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        cache_size_ = 0;
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
        job->roots_requested = job->policy_mask.sum().item<std::int64_t>();
        const auto started = Clock::now();
        reset_peak_memory();

        torch::Tensor effective_policy_mask = job->policy_mask;
        std::optional<torch::Tensor> effective_bootstrap_mask =
            job->value_bootstrap_mask;
        if (cache_targets_) {
            CachePreparation prepared = cache_.prepare(
                job->policy_mask,
                job->policy_targets,
                job->value_targets,
                job->indices,
                cache_refresh_probability_
            );
            job->cache_misses = std::move(prepared.misses);
            job->roots_searched = prepared.roots_searched;
            job->value_targets = std::move(prepared.value_targets);
            job->search_value_targets = std::move(
                prepared.search_value_targets
            );
            job->policy_targets = std::move(prepared.policy_targets);
            effective_policy_mask = job->policy_mask & prepared.miss_mask;
            if (job->value_bootstrap_mask) {
                effective_bootstrap_mask = *job->value_bootstrap_mask
                    & prepared.miss_mask;
            }
        } else {
            job->roots_searched = job->roots_requested;
            job->value_targets = job->value_targets.clone();
            job->search_value_targets = torch::zeros_like(
                job->value_targets
            );
            job->policy_targets = job->policy_targets.clone();
        }

        if (job->roots_searched > 0) {
            run_target(
                job.get(), effective_policy_mask, effective_bootstrap_mask
            );
            if (cache_targets_) {
                cache_.resolve(
                    job->cache_misses,
                    job->value_targets,
                    job->search_value_targets,
                    job->policy_targets
                );
                std::lock_guard<std::mutex> lock(mutex_);
                cache_size_ = cache_.size();
            }
        }
        job->worker_duration_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - started
        ).count();
        job->peak_memory_bytes = peak_memory();
        release_request_inputs(job.get());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_.push_back(job);
        }
        result_ready_.notify_one();
    }

    void run_target(
        Job* job,
        const torch::Tensor& policy_mask,
        const std::optional<torch::Tensor>& bootstrap_mask
    ) {
        torch::Tensor device_frames;
        std::optional<torch::Tensor> device_bootstrap_frames;
        if (job->reanalysis_frames) {
            torch::Tensor combined = job->reanalysis_frames->to(device_);
            device_frames = combined.narrow(1, 0, job->frames.size(1));
            if (job->value_bootstrap_frames) {
                const auto offset = combined.size(1)
                    - job->value_bootstrap_frames->size(1);
                device_bootstrap_frames = combined.narrow(
                    1, offset, job->value_bootstrap_frames->size(1)
                );
            }
        } else {
            device_frames = job->frames.to(device_);
            if (job->value_bootstrap_frames) {
                device_bootstrap_frames =
                    job->value_bootstrap_frames->to(device_);
            }
        }
        if (bootstrap_mask) {
            torch::Tensor reanalyzed_values = target_->reanalyze_values(
                *device_bootstrap_frames,
                bootstrap_mask->to(device_),
                job->value_bootstrap_values->to(device_),
                job->value_bootstrap_discounts->to(device_),
                job->value_targets.to(device_),
                job->stack_size
            );
            job->value_targets.copy_(
                reanalyzed_values.cpu().contiguous()
            );
        }
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

    static void release_request_inputs(Job* job) {
        job->frames = torch::Tensor();
        job->policy_mask = torch::Tensor();
        job->indices = torch::Tensor();
        job->value_bootstrap_frames.reset();
        job->reanalysis_frames.reset();
        job->value_bootstrap_values.reset();
        job->value_bootstrap_discounts.reset();
        job->value_bootstrap_mask.reset();
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
        pending_.clear();
        cache_.clear();
        target_.reset();
    }

    std::size_t pending_count_locked() const { return pending_.size(); }

    std::shared_ptr<ValueTargetNetwork> target_;
    torch::Device device_;
    int prefetch_batches_;
    double timeout_seconds_;
    int target_update_interval_;
    bool cache_targets_;
    double cache_refresh_probability_;
    mutable std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable result_ready_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::deque<std::shared_ptr<Job>> completed_;
    std::unordered_map<std::int64_t, std::shared_ptr<Job>> pending_;
    ReanalysisCache cache_;
    std::thread worker_;
    std::exception_ptr failure_;
    bool closed_ = false;
    std::int64_t next_request_id_ = 0;
    std::int64_t published_version_ = -1;
    std::int64_t active_version_ = -1;
    std::size_t max_pending_ = 0;
    std::size_t cache_size_ = 0;
};

NativeReanalysisEngine::NativeReanalysisEngine(
    std::shared_ptr<ValueTargetNetwork> target,
    const std::string& device,
    int prefetch_batches,
    double timeout_seconds,
    int target_update_interval,
    bool cache_targets,
    double cache_refresh_probability
)
    : impl_(std::make_unique<Impl>(
          std::move(target),
          device,
          prefetch_batches,
          timeout_seconds,
          target_update_interval,
          cache_targets,
          cache_refresh_probability
      )) {}

NativeReanalysisEngine::~NativeReanalysisEngine() = default;

void NativeReanalysisEngine::publish_weights(
    std::int64_t version,
    const py::dict& representation,
    const py::dict& prediction,
    const py::dict& dynamics
) {
    impl_->publish_weights(
        version,
        tensor_state_from_dict(representation),
        tensor_state_from_dict(prediction),
        tensor_state_from_dict(dynamics)
    );
}

std::int64_t NativeReanalysisEngine::submit(
    py::object batch,
    double root_noise_temperature,
    bool gumbel_sampling
) {
    return impl_->submit(
        std::move(batch), root_noise_temperature, gumbel_sampling
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

std::int64_t NativeReanalysisEngine::weight_version() const {
    return impl_->weight_version();
}

}  // namespace atariagent::native
