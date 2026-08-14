#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace atariagent::native {

class NativeReanalysisEngine {
public:
    NativeReanalysisEngine(
        pybind11::object target,
        int prefetch_batches,
        double timeout_seconds,
        int target_update_interval,
        bool cache_targets
    );
    ~NativeReanalysisEngine();

    NativeReanalysisEngine(const NativeReanalysisEngine&) = delete;
    NativeReanalysisEngine& operator=(const NativeReanalysisEngine&) = delete;

    void publish_weights(
        std::int64_t version,
        pybind11::object representation,
        pybind11::object prediction,
        pybind11::object dynamics
    );
    std::int64_t submit(pybind11::object batch);
    pybind11::dict wait_next();
    void close();
    std::size_t pending_count() const;
    std::size_t max_pending() const;
    std::size_t cache_size() const;
    std::int64_t weight_version() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace atariagent::native
