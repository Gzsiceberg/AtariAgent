# Training performance measurements — 2026-09-05

## Scope

Phase 1 of `TRAINING_PERFORMANCE_PLAN.md`: short systems benchmarks, not a full Atari 100k run or an optimization acceptance test. No training defaults, search budget, precision, evaluation episode count, production code, or plan were changed. No commits were made.

Raw configurations, per-update samples, logs, and profiling artifacts are in `measurements/20260905/`. The benchmark scripts are:

- `scripts/benchmark_training_performance.py`: real replay sampling, BatchWorker, optional native reanalysis, compiled learner, immediate priority updates, target publication, and checkpoint writing.
- `scripts/benchmark_training_actor.py`: real Atari collection and the existing evaluation function.
- `scripts/summarize_training_performance.py`: aggregate saved measurements without using the GPU.

## Environment and controls

- Base commit: `1ee8ea88808515938bcb149014798a15b0cb5f9b`.
- RTX 4070 Ti, 12 GiB; Intel i9-13900KF; WSL2 Linux 6.6.87.2; approximately 15 GiB system RAM.
- Python 3.13.7, PyTorch 2.13.0+cu130, CUDA runtime 13.0, cuDNN reported version 92000. Driver output reports Linux 595.58.04 / Windows 596.21.
- Production Hydra configuration: Asterix, **9 actions**, RGB 96×96, four stacked frames; batch 256; unroll/TD/LSTM horizon 5; PUCT-50; policy chunk 768; TTL 200; target publication interval 1,000; max in flight 3; ready prefetch 2.
- SGD, LR 0.2 and the production schedule, gradient clip 1, shift/intensity augmentation, consistency weight 2, BF16 learner/reanalysis, compiled complete learner with `compile_mode=default`. Actor/evaluation follow their existing FP32 path.
- Four native search/torch CPU threads. Production non-deterministic cuDNN/TF32 backend settings were retained.
- Runs were serialized, not launched concurrently on the GPU. The user confirmed GPU availability. WSL reports desktop GPU allocations/activity without listing their owners, so complete absence of host/display interference cannot be certified.

## Method and limitations

Synthetic replay uses the production storage/sampling implementation with 400-step blocks, valid lookahead, stable state IDs, and PER. Frames and rewards are artificial; networks start randomly initialized. The late workload sets the learner schedule to update 100,000, but **does not restore a trained model or real replay distribution**. It is a shape/cache/systems workload, not a prediction of final-training speed or score.

Each main workload has three consecutive 100-update windows. These are repeated windows of an evolving process, **not independent training seeds**. Compilation plus ten learner warmup updates occurs before timing, followed by pipeline warmup. Existing compiler disk caches were retained; recorded setup time is not a clean-cache compilation benchmark. Variable cache-miss shapes can still incur first-use costs after this warmup.

`early` and `late` explicitly clear the **policy cache** before each window. Bootstrap-value cache entries remain valid until target publication. Consequently these are policy-cold windows, not identical fully cold repetitions. Their start/end/drain costs remain inside throughput timing; the explicit cache-clear call is separately recorded. `late_warm` instead retains caches across windows and uses the production TTL/clear schedule after 100 pipeline warmup updates.

`learner_fixed` removes native reanalysis and uses stored TD targets because the isolated replay batches contain no refreshed MCTS values. It retains the learner shape, optimizer, augmentation, and precision; it is a component-isolation reference, not a semantics-preserving faster training mode.

CUDA events bracket the learner stream and its ready-event dependency without synchronizing each substage. CPU timings include actual waits. **Do not sum native worker time, GPU stream spans, replay sampling, and learner wait:** they overlap. Priority “D2H” timing includes waiting for earlier learner work, not just copying 256 numbers.

## Results

### Main finding: cuDNN autotuning is an expensive part of the native path

The only diagnostic factor changed was `torch.backends.cudnn.benchmark=False`, in the benchmark process. TF32, determinism setting, BF16, learner compilation, batch size, PER, search budget, cache TTL, and target schedule were retained.

| Workload | Three windows, updates/s | Pooled updates/s | Effective policy hit* |
| --- | --- | ---: | ---: |
| 2k replay, policy-cold, baseline | 13.55 / 13.04 / 13.65 | 13.41 | 98.68% |
| 100k replay, policy-cold, baseline | 1.74 / 2.10 / 2.59 | 2.09 | 46.46% |
| 100k replay, policy-cold, autotuning off | 3.92 / 4.15 / 4.19 | **4.08** | 46.44% |
| 100k replay, cache-warmed, baseline | 3.00 / 3.01 / 3.72 | 3.21 | 77.34% |
| 100k replay, cache-warmed, autotuning off | 5.79 / 5.46 / 5.59 | **5.61** | 77.38% |
| 100k replay, learner isolation | 23.59 / 23.28 / 23.25 | 23.37 | N/A |

\* `1 − searched/requested`, including within-request deduplication, not solely persistent-cache hits. Pooled throughput is total completed updates divided by total measured wall time, not the arithmetic mean of window throughputs.

Autotuning off improved the complete sampled-batch → reanalysis → learner → immediate-priority-update path by **95.4% in policy-cold windows** and **74.8% in warmed windows**. Both comparisons have almost identical effective cache-hit fractions. Every off-window exceeded every corresponding baseline window, although the baseline's upward drift shows that its variable-shape first-use effects were not fully exhausted. These are promising short-workload results, **not a demonstrated 75% reduction in full Atari training time**.

### Latency distributions

All entries below are **CPU wall-clock p50 / p95 milliseconds**, pooled over 300 updates. Concurrent rows must not be added.

| Measurement | 2k baseline | 100k cold baseline | 100k warmed baseline | 100k warmed, autotuning off | Learner isolation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wait for ready learner batch | 0.67 / 244.24 | 383.83 / 642.08 | 270.15 / 454.48 | **102.26 / 123.65** | 0.11 / 0.15 |
| Native worker request | 0.24 / 286.73 | 477.46 / 741.67 | 317.79 / 499.21 | **177.49 / 193.57** | — |
| Replay sampling, lock excluded | 18.81 / 21.69 | 25.70 / 69.13 | 23.62 / 48.67 | 41.11 / 59.45 | 18.06 / 19.66 |
| Sampling lock acquisition | 0.004 / 8.70 | 0.005 / 0.006 | 0.006 / 0.007 | 0.006 / 0.007 | 0.004 / 0.005 |
| Priority D2H + preceding stream wait | 8.55 / 14.45 | 0.89 / 12.48 | 8.83 / 13.97 | 0.40 / 5.97 | 9.34 / 13.08 |
| CPU replay priority mutation | 0.12 / 0.19 | 0.10 / 0.26 | 0.11 / 0.18 | 0.10 / 0.24 | 0.10 / 0.13 |

Warm baseline mean learner-ready wait was **250 ms/update**, versus **104 ms** with autotuning off. Maximum warm request latency fell from **549 to 224 ms**; maximum ready-batch wait fell from **504 to 169 ms**. The native path remains the bottleneck even after this improvement; learner isolation is much faster. Sampling itself became slower under the faster native path, so an isolated component improvement must not be mistaken for an additive end-to-end gain.

Fully cached small-replay requests can complete in roughly **0.24 ms**, but a few cache misses can still create hundreds-of-milliseconds stalls. In the warmed baseline, requests with 90–99% effective policy hits averaged **297 ms** (57 requests). Root count alone therefore does not explain latency: first-use/autotuning, launches, synchronization, and interference matter.

### Setup, publication, and checkpoint costs

- Cached compilation + ten learner warmup updates: **6.7–7.2 s** across the principal workloads. This includes setup/execution, not just compiler CPU time; clean-cache compilation remains unmeasured.
- Ten-request cold late-pipeline warmup: **9.00 s baseline → 3.94 s autotuning off**.
- One hundred late-pipeline warmup requests: **60.76 s baseline → 28.18 s autotuning off**. The improvement did not merely move work into an excluded warmup phase.
- Five idle publication repetitions per pipeline run: target snapshot means **7.8–8.9 ms** and acknowledged target publication means **44.5–47.3 ms** in the original baseline workloads. These measurements are drained/idle, not queued target-boundary stalls.
- Three production-format training-checkpoint writes per workload: means **61–67 ms** in the baseline workloads. They include networks, consistency head, optimizer, delayed target, and config. No `fsync` was issued: these are page-cache write timings, not durable-storage guarantees. The roughly 3 GB replay snapshot, representative checkpoint copying, and plotting/history I/O were not benchmarked.
- Explicit idle policy-cache-clear calls are recorded separately in each cold window. Their queued effect is included in the following measured window; they do not represent cache clearing under a full queue.

### CUDA timeline and critical path

The system `nsys` is version 2023.4.4 and failed to import its trace. The already-installed **Nsight Systems CLI 2025.5.1** successfully produced `timeline_new.nsys-rep` and `timeline_no_autotune.nsys-rep`, with SQLite exports and reproducible analysis in `measurements/20260905/analyze_timeline.py`.

Both traces cover a separate **12-update policy-cold late-replay** workload after warmup. They are for attribution, not throughput acceptance:

| Trace observation | Baseline | Autotuning off |
| --- | ---: | ---: |
| CPU NVTX measured interval | 10.57 s | 5.15 s |
| CUDA device-wide synchronization calls | **2,114** | **2** |
| CUDA event synchronization calls | **1,920** | **0** |
| Kernel launches recorded | 255,304 | 241,008 |
| Union of GPU kernel/copy/memset intervals | ~4.81 s | ~2.43 s |

Of the baseline's device-wide synchronizations, **2,112 originate from the native inference thread**; the other two are the benchmark's phase-boundary synchronizations. Turning off autotuning removes the native device-wide synchronizations in this trace. This is stronger evidence than attributing all slowness to default-stream sharing alone.

The main learner thread, PyTorch autograd thread, and native inference thread all launch most kernels on **stream 7, identified by Nsight as the null/default stream**. Learner H2D uses distinct non-blocking **stream 45**. Native library auxiliary streams 173/181 also contain a small amount of work: it would be incorrect to claim literally all GPU execution is on one stream. Nevertheless, there is no explicitly separate native reanalysis compute stream in the observed baseline.

Across each trace, learner-stream H2D copied **729.94 MiB**, consistent with twelve approximately 60.75 MiB frame payloads plus metadata. Default-stream H2D copied about **1,147.8 MiB**, consistent with twelve 94.5 MiB reanalysis payloads plus search inputs. Summed GPU H2D durations were about **170 ms baseline / 176 ms autotuning off**, small compared with the 10.57 / 5.15 s intervals. These are overlapping transfer durations, not sequential costs. Raw PCIe payload movement is not the first bottleneck indicated here.

**Observed critical path:** waiting for native reanalysis, with substantial host launch/synchronization overhead and idle intervals; not the small priority mutation or raw frame-copy duration. Disabling autotuning removes major synchronization overhead, but approximately 20,000 recorded kernels per update remain in the profiled cold workload. Explicit stream ownership and native inference launch reduction remain justified *experiments*, not proven improvements.

Timing caveats: Nsight warns that this CUDA 13.2 driver is newer than its supported 13.0 tracing libraries. Also, a separate synchronized CUDA-event versus `perf_counter` check showed event durations around 38.1 ms while wall time was around 35.0 ms. GPU-event spans are retained as diagnostics but are not treated as interchangeable with CPU wall time or summed into the throughput model. Timeline activity coverage is approximate, not an occupancy measurement.

### Numerical comparison and memory

A fixed-input native BF16 PUCT-50 comparison used identical initialized weights and search RNG at **48, 768, and 1,536 roots**. With autotuning on versus off, policy targets, corrected values, and MCTS search values were **exactly equal** in all three fixtures; argmax agreement was 100%, and all outputs were finite. See `validate_autotune.py` and `.json`. This does not establish equivalence for trained networks, all shapes, learner gradients, or learning curves.

Reliable isolated PyTorch allocation peaks: learner-only **0.829 GiB** (reserved 1.111 GiB); collection **40.6 MiB**; evaluation **85.2 MiB**. Native per-request reported allocation peaks reached at least **1.79 GiB** in the warmed baseline and **1.38 GiB** with autotuning off. These latter values are lower bounds because of the counter-reset issue described below; **no reliable whole-pipeline memory improvement is claimed**. Allocator reserved counters can also grow even when observed live allocations are smaller.

## Additional findings

### Evaluation is a substantial synchronous cost

Actual collection of 100 transitions across four environments took **2.160 / 2.175 / 2.198 seconds** after a 2.976-second warmup window: approximately **45.9 transitions/s**. These are real environment/search/trajectory timings, without learner updates, replay insertion, or actor-weight refresh.

One complete evaluation using the unchanged **32 episodes / 16 environments** protocol took **256.09 seconds (4m16s)**. It used random initial weights, not a final checkpoint; the score was 470.31 and is not a learning result. The first sweep includes evaluation-specific first-use costs. Further evaluation repetition was interrupted after this completed result was saved, to keep this measurement session bounded. No partial evaluation is reported as complete.

This directly supports treating intermediate evaluation as a separate workflow cost. It does **not** establish the evaluation fraction of a complete trained run or justify changing the reporting protocol without agreement.

### Priority-lock narrowing is not the dominant opportunity in these workloads

The baseline really holds the replay lock during priority D2H. Isolated replay mutation is only about a tenth of a millisecond. The full-replay cold measurements show negligible sampling lock-acquisition wait despite several milliseconds spent completing priorities. Moving D2H outside the lock remains a reasonable targeted experiment, but these data do not support presenting it as the main source of speedup.

### Peak-memory reporting needs care

`cpp/reanalysis/engine.cpp::reset_peak_memory()` resets the **device-wide** PyTorch allocator peak before every native request. Consequently a Python end-of-window `max_memory_allocated()` or `max_memory_reserved()` is not a reliable whole-window peak when reanalysis is active. Early output files retain these raw counters, but they must not be interpreted as true phase peaks.

Later instrumentation also records every native result's reported allocation peak. These are useful observed lower bounds, not guaranteed whole-phase peaks: allocations between reset/report intervals can still be missed. Actor and learner-isolation peaks do not have this native reset interference. External desktop/driver memory is not included in PyTorch allocation counters.

### Actor and reanalysis inference are different execution paths

`AtariAgent.act()` uses Python `TreeSearch.search_batch()` with native tree traversal and a Python/PyTorch recurrent inference loop. Reanalysis uses `cpp/inference/tree_search.cpp` and LibTorch inference. Therefore native reanalysis CUDA Graphs or buffer reuse would not automatically accelerate actor/evaluation inference. Benchmark both paths before extrapolating a native-search improvement to collection or evaluation.

## Recommendation and remaining work

1. **Validate autotuning off on real replay/trained weights before changing defaults.** It is the clearest measured candidate. Repeat independent processes/seeds and include actor/evaluation, learner numerics, and full time-to-final-model. This session changed no production setting and did not run a full training/score regression experiment.
2. Keep intermediate evaluation as a separately agreed workflow experiment. The measured full sweep costs 4m16s, but changing episode count/cadence is not bitwise-neutral or automatically protocol-neutral.
3. Test explicit native stream ownership only with correctly ordered target publication/buffer lifetimes and a retained baseline. Removing repeated device-wide autotuning synchronization is especially relevant before evaluating potential overlap. No stream implementation or synchronization semantics were changed here.
4. Prioritize native launch reduction and variable-root-shape handling after these checks. Defer larger frame-transfer work and asynchronous priorities until they show meaningful critical-path cost. Do not silently change TTL, search simulations, prefetch depth, batch size, or update count.

**Phase 1 is informative but not fully complete.** Still missing: a real-replay early/final baseline; true whole-phase peak memory; fully separated native cache preparation/traversal/inference CPU ranges; separately attributed learner forward/backward/optimizer GPU spans; queued target-publication/cache-clear stalls; full replay-snapshot I/O; complete collection/training/evaluation accounting and time-to-final-model. No full-training speedup or learning-quality acceptance is claimed.

## Validation and artifact notes

- New instrumentation unit tests cover latency summaries, mutual exclusion/exception release of the timing lock, and preservation of replay-priority values/stale-ID errors.
- Final relevant pytest run: **32 passed** (`tests/training/test_performance_benchmark.py`, `test_batch_worker.py`, `test_config.py`). Existing batch-worker tests also passed alone (**8 passed**). Ruff check and format check passed for all four new source/test files.
- One earlier combined test run hit the existing timing-sensitive `test_worker_propagates_sampling_failure`: the worker reported its second-sample error before the test consumed the first ready batch. Production code was unchanged; subsequent runs passed. This race was not silently “fixed” or hidden.
- `early_pilot` failed after its throughput windows on a benchmark-only repeated publication version; the first `learner` attempt failed because mixed targets require MCTS metadata. These were corrected in the harness and rerun. Neither incomplete attempt is included in the result tables. `timeline.qdstrm` is the failed old-Nsight import attempt; use the newer `.nsys-rep` files instead.
- `aggregate.json` includes successful **profiled** runs as separate entries; exclude `trace*` from unprofiled throughput comparisons.
- `actor/results.json` contains three complete collection windows and one complete 32-episode evaluation. Its log also contains the intentional interruption of further evaluation repetition. The final actor harness now has an independent `+benchmark.evaluation_repetitions=1` control; episode count itself is unchanged.

## Reproduction

**Implementation follow-up:** after these measurements, the training default was changed at the user's request to `training.cudnn_benchmark=false`. To reproduce the historical baseline below with current code, add **`training.cudnn_benchmark=true`** to the baseline and baseline-trace commands. The stored original measurements/configurations are unchanged. The benchmark-only `+benchmark.cudnn_benchmark=false` override still reproduces the off comparison.

Run these sequentially, with the GPU otherwise available:

```bash
uv run python scripts/benchmark_training_performance.py \
  +benchmark.output=measurements/repro/early +benchmark.replay_size=2000 \
  +benchmark.updates=100 +benchmark.warmup=10 +benchmark.repetitions=3

uv run python scripts/benchmark_training_performance.py \
  +benchmark.output=measurements/repro/late +benchmark.replay_size=100000 \
  +benchmark.start_step=100000 +benchmark.updates=100 \
  +benchmark.warmup=10 +benchmark.repetitions=3

uv run python scripts/benchmark_training_performance.py \
  +benchmark.output=measurements/repro/late_warm +benchmark.replay_size=100000 \
  +benchmark.start_step=100000 +benchmark.updates=100 +benchmark.warmup=10 \
  +benchmark.pipeline_warmup=100 +benchmark.repetitions=3 +benchmark.cold_windows=false

uv run python scripts/benchmark_training_performance.py \
  +benchmark.output=measurements/repro/learner +benchmark.mode=learner \
  +benchmark.replay_size=100000 +benchmark.start_step=100000 \
  +benchmark.updates=100 +benchmark.warmup=10 +benchmark.repetitions=3

uv run python scripts/benchmark_training_actor.py \
  +benchmark.output=measurements/repro/actor \
  +benchmark.repetitions=3 +benchmark.evaluation_repetitions=1

# Repeat either late pipeline command with this additional one-factor override:
#   +benchmark.cudnn_benchmark=false

uv run python scripts/summarize_training_performance.py measurements/repro

/opt/nvidia/nsight-systems-cli/2025.5.1/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o measurements/repro/timeline \
  uv run python scripts/benchmark_training_performance.py \
  +benchmark.output=measurements/repro/trace +benchmark.replay_size=100000 \
  +benchmark.start_step=100000 +benchmark.updates=12 \
  +benchmark.warmup=10 +benchmark.repetitions=1 +benchmark.trace=true
```

Use a dedicated output directory. The final harness creates its three checkpoint-write samples inside a unique temporary subdirectory and removes that directory afterward; production checkpoint files are never touched. Stored resolved configurations capture each run's Hydra overrides. Later runs also include a source snapshot and hash.
