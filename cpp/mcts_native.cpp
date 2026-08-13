#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ATARIAGENT_HAS_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<
    float,
    py::array::c_style | py::array::forcecast
>;

struct MinMaxStats {
    explicit MinMaxStats(float minimum_delta)
        : minimum_delta(minimum_delta) {}

    float minimum_delta;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();

    void clear() {
        minimum = std::numeric_limits<float>::infinity();
        maximum = -std::numeric_limits<float>::infinity();
    }

    void update(float value) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    float normalize(float value) const {
        const float delta = maximum - minimum;
        if (delta > 0.0F) {
            value = (value - minimum) / std::max(delta, minimum_delta);
        }
        return std::clamp(value, 0.0F, 1.0F);
    }
};

struct Node {
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

    bool expanded() const { return !children.empty(); }

    float value() const {
        return visit_count == 0
            ? 0.0F
            : value_sum / static_cast<float>(visit_count);
    }
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
    )
        : action_count_(static_cast<int>(root_priors.size())),
          discount_(discount),
          value_prefix_horizon_(value_prefix_horizon),
          stats_(minimum_delta),
          rng_(seed),
          deterministic_ties_(deterministic_ties) {
        if (action_count_ <= 0) {
            throw std::invalid_argument("root policy must not be empty");
        }
        nodes_.reserve(
            1 + action_count_ * (std::max(simulations, 0) + 1)
        );
        Node root;
        root.prior = 1.0F;
        root.state_slot = 0;
        root.value_prefix = root_value_prefix;
        root.visit_count = 1;
        root.value_sum = root_value;
        nodes_.push_back(std::move(root));
        expand(0, 0, root_value_prefix, root_priors, false, true);
    }

    std::tuple<int, int, bool> traverse(
        float pb_c_base,
        float pb_c_init
    ) {
        path_.clear();
        int node_index = 0;
        float parent_mean_q = 0.0F;
        path_.push_back(node_index);

        while (nodes_[node_index].expanded()) {
            const float mean_q = node_mean_q(
                node_index,
                parent_mean_q,
                node_index == 0
            );
            node_index = select_child(
                node_index,
                mean_q,
                pb_c_base,
                pb_c_init
            );
            path_.push_back(node_index);
            parent_mean_q = mean_q;
        }

        const Node& leaf = nodes_[node_index];
        if (leaf.parent < 0 || leaf.action < 0) {
            throw std::runtime_error("selection did not reach a child leaf");
        }
        const Node& parent = nodes_[leaf.parent];
        return {parent.state_slot, leaf.action, parent.reset_value_prefix};
    }

    void expand_and_back_up(
        int state_slot,
        float value_prefix,
        float value,
        const std::vector<float>& policy_logits
    ) {
        if (path_.empty()) {
            throw std::runtime_error("traverse must precede expansion");
        }
        const int leaf_index = path_.back();
        const bool reset = (
            nodes_[leaf_index].depth % value_prefix_horizon_ == 0
        );
        expand(
            leaf_index,
            state_slot,
            value_prefix,
            softmax(policy_logits),
            reset,
            false
        );
        back_up(value);
        rebuild_min_max();
    }

    std::vector<int> visit_counts() const {
        std::vector<int> counts;
        counts.reserve(action_count_);
        for (const int child_index : nodes_[0].children) {
            counts.push_back(nodes_[child_index].visit_count);
        }
        return counts;
    }

    float root_value() const { return nodes_[0].value(); }

private:
    int action_count_;
    float discount_;
    int value_prefix_horizon_;
    MinMaxStats stats_;
    std::mt19937_64 rng_;
    bool deterministic_ties_;
    std::vector<Node> nodes_;
    std::vector<int> path_;

    static std::vector<float> softmax(
        const std::vector<float>& logits
    ) {
        if (logits.empty()) {
            throw std::invalid_argument("policy logits must not be empty");
        }
        const float maximum = *std::max_element(logits.begin(), logits.end());
        std::vector<float> probabilities;
        probabilities.reserve(logits.size());
        float total = 0.0F;
        for (const float logit : logits) {
            if (!std::isfinite(logit)) {
                throw std::invalid_argument("policy logits must be finite");
            }
            const float probability = std::exp(logit - maximum);
            probabilities.push_back(probability);
            total += probability;
        }
        for (float& probability : probabilities) {
            probability /= total;
        }
        return probabilities;
    }

    void expand(
        int node_index,
        int state_slot,
        float value_prefix,
        const std::vector<float>& priors,
        bool reset,
        bool priors_are_probabilities
    ) {
        if (static_cast<int>(priors.size()) != action_count_) {
            throw std::invalid_argument("policy action count changed during search");
        }
        if (!std::isfinite(value_prefix)) {
            throw std::invalid_argument("value prefix must be finite");
        }
        if (nodes_[node_index].expanded()) {
            throw std::runtime_error("cannot expand a node more than once");
        }

        if (priors_are_probabilities) {
            float total = 0.0F;
            for (const float prior : priors) {
                if (!std::isfinite(prior) || prior < 0.0F) {
                    throw std::invalid_argument(
                        "root priors must be finite and non-negative"
                    );
                }
                total += prior;
            }
            if (!(total > 0.0F)) {
                throw std::invalid_argument("root priors must have positive mass");
            }
        }

        nodes_[node_index].state_slot = state_slot;
        nodes_[node_index].value_prefix = value_prefix;
        nodes_[node_index].reset_value_prefix = reset;
        nodes_[node_index].children.reserve(action_count_);
        const int child_depth = nodes_[node_index].depth + 1;
        for (int action = 0; action < action_count_; ++action) {
            Node child;
            child.prior = priors[action];
            child.parent = node_index;
            child.action = action;
            child.depth = child_depth;
            nodes_.push_back(std::move(child));
            nodes_[node_index].children.push_back(
                static_cast<int>(nodes_.size()) - 1
            );
        }
    }

    float reward(int node_index) const {
        const Node& node = nodes_[node_index];
        if (node.parent < 0) {
            return 0.0F;
        }
        const Node& parent = nodes_[node.parent];
        if (parent.reset_value_prefix) {
            return node.value_prefix;
        }
        return node.value_prefix - parent.value_prefix;
    }

    float q_value(int node_index) const {
        return reward(node_index) + discount_ * nodes_[node_index].value();
    }

    float node_mean_q(
        int node_index,
        float parent_mean_q,
        bool is_root
    ) const {
        float sum = 0.0F;
        int count = 0;
        for (const int child_index : nodes_[node_index].children) {
            if (nodes_[child_index].visit_count > 0) {
                sum += q_value(child_index);
                ++count;
            }
        }
        if (is_root) {
            return count == 0 ? 0.0F : sum / static_cast<float>(count);
        }
        return (parent_mean_q + sum) / static_cast<float>(1 + count);
    }

    int select_child(
        int node_index,
        float mean_q,
        float pb_c_base,
        float pb_c_init
    ) {
        int child_visits = 0;
        for (const int child_index : nodes_[node_index].children) {
            child_visits += nodes_[child_index].visit_count;
        }
        const float exploration_scale = pb_c_init + std::log(
            (static_cast<float>(child_visits) + pb_c_base + 1.0F)
            / pb_c_base
        );
        const float sqrt_visits = std::sqrt(static_cast<float>(child_visits));

        float best_score = -std::numeric_limits<float>::infinity();
        std::vector<int> best_children;
        for (const int child_index : nodes_[node_index].children) {
            const Node& child = nodes_[child_index];
            const float prior_score = child.prior * sqrt_visits
                / static_cast<float>(1 + child.visit_count)
                * exploration_scale;
            const float q = child.visit_count > 0
                ? q_value(child_index)
                : mean_q;
            const float score = stats_.normalize(q) + prior_score;
            if (score > best_score + 1.0e-12F) {
                best_score = score;
                best_children.assign(1, child_index);
            } else if (std::abs(score - best_score) <= 1.0e-12F) {
                best_children.push_back(child_index);
            }
        }
        if (best_children.empty()) {
            throw std::runtime_error("expanded node has no selectable child");
        }
        if (deterministic_ties_) {
            return best_children.front();
        }
        std::uniform_int_distribution<std::size_t> distribution(
            0,
            best_children.size() - 1
        );
        return best_children[distribution(rng_)];
    }

    void back_up(float leaf_value) {
        if (!std::isfinite(leaf_value)) {
            throw std::invalid_argument("leaf value must be finite");
        }
        float bootstrap = leaf_value;
        for (auto iterator = path_.rbegin(); iterator != path_.rend(); ++iterator) {
            Node& node = nodes_[*iterator];
            node.value_sum += bootstrap;
            ++node.visit_count;
            bootstrap = reward(*iterator) + discount_ * bootstrap;
        }
    }

    void rebuild_min_max() {
        stats_.clear();
        std::vector<int> stack{0};
        while (!stack.empty()) {
            const int node_index = stack.back();
            stack.pop_back();
            for (const int child_index : nodes_[node_index].children) {
                if (nodes_[child_index].visit_count > 0) {
                    stats_.update(q_value(child_index));
                    stack.push_back(child_index);
                }
            }
        }
    }
};

class NativeReanalysisEngine {
public:
    NativeReanalysisEngine(
        py::object target,
        int prefetch_batches,
        double timeout_seconds,
        int max_weight_lag,
        bool cache_targets
    )
        : target_(std::move(target)),
          prefetch_batches_(prefetch_batches),
          timeout_seconds_(timeout_seconds),
          max_weight_lag_(max_weight_lag),
          cache_targets_(cache_targets) {
        if (prefetch_batches <= 0) {
            throw std::invalid_argument("prefetch_batches must be positive");
        }
        if (!std::isfinite(timeout_seconds) || timeout_seconds <= 0.0) {
            throw std::invalid_argument("timeout_seconds must be positive");
        }
        if (max_weight_lag < 0) {
            throw std::invalid_argument("max_weight_lag must be non-negative");
        }
        worker_ = std::thread(&NativeReanalysisEngine::run, this);
    }

    ~NativeReanalysisEngine() { close_impl(); }

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
        if (lag < 0 || lag > max_weight_lag_) {
            throw std::runtime_error(
                "reanalysis result target-weight lag exceeds limit"
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
    int max_weight_lag_;
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

class BatchTree {
public:
    BatchTree(
        FloatArray root_priors,
        FloatArray root_values,
        FloatArray root_value_prefixes,
        int simulations,
        float discount,
        int value_prefix_horizon,
        float minimum_delta,
        std::uint64_t seed,
        bool deterministic_ties
    ) {
        if (root_priors.ndim() != 2
            || root_values.ndim() != 1
            || root_value_prefixes.ndim() != 1) {
            throw std::invalid_argument(
                "root priors must be 2D and root values must be 1D"
            );
        }
        const py::ssize_t root_count = root_priors.shape(0);
        if (root_count <= 0) {
            throw std::invalid_argument("at least one root is required");
        }
        if (root_values.shape(0) != root_count
            || root_value_prefixes.shape(0) != root_count) {
            throw std::invalid_argument("root arrays must have equal lengths");
        }
        action_count_ = static_cast<int>(root_priors.shape(1));
        if (action_count_ <= 0) {
            throw std::invalid_argument("root policy must not be empty");
        }
        if (simulations <= 0) {
            throw std::invalid_argument("simulations must be positive");
        }
        if (value_prefix_horizon <= 0) {
            throw std::invalid_argument("value prefix horizon must be positive");
        }

        const auto priors_data = root_priors.unchecked<2>();
        const auto values_data = root_values.unchecked<1>();
        const auto prefixes_data = root_value_prefixes.unchecked<1>();
        roots_.reserve(static_cast<std::size_t>(root_count));
        std::mt19937_64 seed_generator(seed);
        for (py::ssize_t index = 0; index < root_count; ++index) {
            if (!std::isfinite(values_data(index))
                || !std::isfinite(prefixes_data(index))) {
                throw std::invalid_argument("root values must be finite");
            }
            std::vector<float> priors(action_count_);
            for (int action = 0; action < action_count_; ++action) {
                const float prior = priors_data(index, action);
                if (!std::isfinite(prior) || prior < 0.0F) {
                    throw std::invalid_argument(
                        "root priors must be finite and non-negative"
                    );
                }
                priors[action] = prior;
            }
            roots_.emplace_back(
                priors,
                values_data(index),
                prefixes_data(index),
                simulations,
                discount,
                value_prefix_horizon,
                minimum_delta,
                seed_generator(),
                deterministic_ties
            );
        }
    }

    py::tuple traverse_arrays(float pb_c_base, float pb_c_init) {
        if (!std::isfinite(pb_c_base) || pb_c_base <= 0.0F
            || !std::isfinite(pb_c_init) || pb_c_init < 0.0F) {
            throw std::invalid_argument("invalid exploration constants");
        }
        const py::ssize_t root_count = static_cast<py::ssize_t>(roots_.size());
        py::array_t<std::int64_t> state_slots(root_count);
        py::array_t<std::int64_t> actions(root_count);
        py::array_t<bool> resets(root_count);
        auto slots_data = state_slots.mutable_unchecked<1>();
        auto actions_data = actions.mutable_unchecked<1>();
        auto resets_data = resets.mutable_unchecked<1>();
        #pragma omp parallel for if(root_count >= 32)
        for (py::ssize_t index = 0; index < root_count; ++index) {
            auto [state_slot, action, reset] = roots_[index].traverse(
                pb_c_base,
                pb_c_init
            );
            slots_data(index) = state_slot;
            actions_data(index) = action;
            resets_data(index) = reset;
        }
        return py::make_tuple(state_slots, actions, resets);
    }

    void expand_and_back_up_arrays(
        int state_slot,
        py::array_t<float, py::array::c_style | py::array::forcecast>
            value_prefixes,
        py::array_t<float, py::array::c_style | py::array::forcecast> values,
        py::array_t<float, py::array::c_style | py::array::forcecast>
            policy_logits
    ) {
        if (value_prefixes.ndim() != 1 || values.ndim() != 1
            || policy_logits.ndim() != 2
            || value_prefixes.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || values.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || policy_logits.shape(0) != static_cast<py::ssize_t>(roots_.size())
            || policy_logits.shape(1) != action_count_) {
            throw std::invalid_argument(
                "evaluation arrays have invalid shapes"
            );
        }
        auto prefixes_data = value_prefixes.unchecked<1>();
        auto values_data = values.unchecked<1>();
        auto policy_data = policy_logits.unchecked<2>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            if (!std::isfinite(prefixes_data(index))
                || !std::isfinite(values_data(index))) {
                throw std::invalid_argument("evaluation values must be finite");
            }
            for (int action = 0; action < action_count_; ++action) {
                if (!std::isfinite(policy_data(index, action))) {
                    throw std::invalid_argument("policy logits must be finite");
                }
            }
        }
        #pragma omp parallel for if(roots_.size() >= 32)
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(roots_.size());
             ++index) {
            std::vector<float> policy(action_count_);
            for (int action = 0; action < action_count_; ++action) {
                policy[action] = policy_data(index, action);
            }
            roots_[index].expand_and_back_up(
                state_slot,
                prefixes_data(index),
                values_data(index),
                policy
            );
        }
    }

    py::array_t<std::int32_t> visit_counts_array() const {
        py::array_t<std::int32_t> result(
            {static_cast<py::ssize_t>(roots_.size()),
             static_cast<py::ssize_t>(action_count_)}
        );
        auto output = result.mutable_unchecked<2>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            const std::vector<int> counts = roots_[index].visit_counts();
            for (int action = 0; action < action_count_; ++action) {
                output(index, action) = counts[action];
            }
        }
        return result;
    }

    py::array_t<float> root_values_array() const {
        py::array_t<float> result(
            static_cast<py::ssize_t>(roots_.size())
        );
        auto output = result.mutable_unchecked<1>();
        for (py::ssize_t index = 0;
             index < static_cast<py::ssize_t>(roots_.size());
             ++index) {
            output(index) = roots_[index].root_value();
        }
        return result;
    }

private:
    int action_count_ = 0;
    std::vector<RootTree> roots_;
};

}  // namespace

PYBIND11_MODULE(_mcts_native, module) {
    module.doc() = "Native batched EfficientZero MCTS tree operations";
    module.def("set_num_threads", [](int count) {
        if (count <= 0) {
            throw std::invalid_argument("thread count must be positive");
        }
#ifdef ATARIAGENT_HAS_OPENMP
        omp_set_num_threads(count);
        return omp_get_max_threads();
#else
        return 1;
#endif
    });
    py::class_<BatchTree>(module, "BatchTree")
        .def(
            py::init<
                FloatArray,
                FloatArray,
                FloatArray,
                int,
                float,
                int,
                float,
                std::uint64_t,
                bool
            >(),
            py::arg("root_priors"),
            py::arg("root_values"),
            py::arg("root_value_prefixes"),
            py::arg("simulations"),
            py::arg("discount"),
            py::arg("value_prefix_horizon"),
            py::arg("minimum_delta"),
            py::arg("seed"),
            py::arg("deterministic_ties") = false
        )
        .def(
            "traverse_arrays",
            &BatchTree::traverse_arrays,
            py::arg("pb_c_base"),
            py::arg("pb_c_init")
        )
        .def(
            "expand_and_back_up_arrays",
            &BatchTree::expand_and_back_up_arrays,
            py::arg("state_slot"),
            py::arg("value_prefixes"),
            py::arg("values"),
            py::arg("policy_logits")
        )
        .def("visit_counts_array", &BatchTree::visit_counts_array)
        .def("root_values_array", &BatchTree::root_values_array);

    py::class_<NativeReanalysisEngine>(module, "NativeReanalysisEngine")
        .def(
            py::init<py::object, int, double, int, bool>(),
            py::arg("target"),
            py::arg("prefetch_batches"),
            py::arg("timeout_seconds"),
            py::arg("max_weight_lag"),
            py::arg("cache_targets")
        )
        .def(
            "publish_weights",
            &NativeReanalysisEngine::publish_weights,
            py::arg("version"),
            py::arg("representation"),
            py::arg("prediction"),
            py::arg("dynamics")
        )
        .def("submit", &NativeReanalysisEngine::submit, py::arg("batch"))
        .def("wait_next", &NativeReanalysisEngine::wait_next)
        .def("close", &NativeReanalysisEngine::close)
        .def_property_readonly(
            "pending_count",
            &NativeReanalysisEngine::pending_count
        )
        .def_property_readonly(
            "max_pending",
            &NativeReanalysisEngine::max_pending
        )
        .def_property_readonly(
            "cache_size",
            &NativeReanalysisEngine::cache_size
        )
        .def_property_readonly(
            "weight_version",
            &NativeReanalysisEngine::weight_version
        );
}
