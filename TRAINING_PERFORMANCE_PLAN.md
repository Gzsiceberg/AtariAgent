# Training Performance Improvement Plan

## Goal

Reduce wall-clock time for Atari 100k training on the existing single-GPU setup. Prioritize systems changes that preserve the training algorithm, search budget, cache freshness, batch size, and update count.

This plan is based on code inspection and existing run logs. Proposed speedups have not been benchmarked or implemented.

## Evidence from existing runs

Source: `runs/per_experiments/20260905_033728/`.

| Completed Asterix run | Logged elapsed | Final-phase median reanalysis | Mean final-phase cache hit |
| --- | --- | --- | --- |
| `alpha-0.6-beta-1-v1` | 4h 11m | 100 ms/batch | 77% |
| `alpha-1-beta-1-threshold-20000-v1` | 3h 59m | 94 ms/batch | 85% |

Elapsed times are final progress-bar readings and include evaluation. Reanalysis and cache statistics come from sampled progress records, not every batch. These runs differ algorithmically and are not controlled systems comparisons.

Key observations:

- Earlier batches can take about 85 ms to reanalyze despite very few policy-cache misses. Fixed overhead, synchronization, and learner interference deserve investigation.
- Logged `queue` time is time waiting inside the native reanalysis queue, not necessarily learner starvation.
- Progress gaps around evaluation boundaries total approximately 36 and 29 minutes respectively, or 12–14% of elapsed time. These estimates include nearby work and are not isolated evaluation timings.
- The current configuration evaluates every 20k collection updates with 32 episodes; these completed runs evaluated every 10k with 16 episodes. Do not assume their evaluation overhead directly predicts current runs.

## Phase 1 — Establish a controlled baseline

### Tasks

- Record commit, resolved Hydra configuration, GPU/CPU, precision, and software versions.
- Benchmark with no competing GPU workload.
- Separate compilation/warmup from steady-state measurements.
- Measure collection, learner-only, evaluation, and checkpoint time separately.
- Instrument:
  - Replay sampling and replay-lock wait.
  - Learner wait for a ready batch.
  - H2D transfers and GPU event dependencies.
  - Learner forward/backward/optimizer GPU time.
  - Priority D2H transfer and CPU replay update.
  - Reanalysis cache preparation, frame transfer, inference, traversal, and result transfer.
  - Target-network publication and cache-clear stalls.
- Capture a CUDA timeline to determine whether learner and native reanalysis share the default stream and whether the GPU has exploitable idle gaps.
- Measure early/small-replay and late/full-replay workloads, including cold-cache boundaries.

### Relevant files

- `scripts/train_agent.py`
- `src/atariagent/training/batch_worker.py`
- `src/atariagent/training/learner.py`
- `cpp/reanalysis/engine.cpp`
- Existing `scripts/benchmark_*.py` tools; verify their settings match production before using their results.

### Deliverable

A baseline report with repeated-run throughput, latency distributions, peak memory, and the observed critical path. Do not add overlapping CPU/GPU timings as though they were sequential costs.

## Phase 2 — Reduce intermediate evaluation overhead

### Proposal

- Separate intermediate evaluation episode count from final evaluation episode count using Hydra configuration.
- Use fewer episodes or fewer checkpoints for intermediate diagnostics.
- Retain full final evaluation.
- Optionally defer intermediate checkpoint evaluation until after training.

### Rationale

`checkpoint_and_evaluate()` in `scripts/train_agent.py` evaluates synchronously and blocks learner progress.

### Validation

- Report time-to-final-model separately from total training-plus-evaluation compute time.
- Preserve checkpoint availability and final evaluation protocol.
- Check evaluation scheduling and RNG handling; do not assume changing evaluation cadence is bitwise neutral.
- Do not silently change the user’s benchmark reporting protocol.

## Phase 3 — Test a dedicated native reanalysis CUDA stream

### Proposal

Give the native worker in `cpp/reanalysis/engine.cpp` an explicitly owned CUDA stream and device guard.

### Rationale

The worker uses a separate CPU thread, but inspection found no explicit CUDA stream guard in the native code. CPU asynchrony alone does not establish concurrent GPU execution.

### Requirements

- Synchronize initial target setup and target-weight publication correctly.
- Preserve request ordering and target-version guarantees.
- Maintain tensor and pinned-buffer lifetimes across streams.
- Avoid replacing stream dependencies with device-wide synchronization.
- Keep a baseline execution mode for comparison.

### Validation

- Confirm actual stream separation and overlap in the CUDA timeline.
- Test target publication, cache clearing, shutdown, and error handling.
- Compare outputs within appropriate numerical tolerances.
- Compare sustained end-to-end throughput and peak memory, not just isolated reanalysis latency.

### Risk

Learner and reanalysis compete for the same GPU. Concurrency may increase contention or memory usage rather than improve throughput. Retain the change only if controlled measurements support it.

## Phase 4 — Narrow replay-lock scope around priority updates

### Proposal A: preserve immediate updates

Transfer priorities to CPU before acquiring `_replay_lock`, then acquire the lock only for replay mutation and validation that requires replay state.

### Rationale

`BatchWorker.complete()` currently holds the replay lock while `ReplayBuffer.update_priorities()` calls `priorities.detach().cpu().numpy()`. GPU synchronization can therefore block replay sampling.

### Proposal B: optional asynchronous completion

If Proposal A leaves a meaningful bottleneck, test pinned-memory asynchronous priority transfers with CUDA events and bounded completion processing.

### Requirements

- Preserve stable transition-ID validation.
- Maintain exactly-once completion and capacity-slot release.
- Keep batch and pinned-buffer ownership until GPU consumers finish.
- For asynchronous completion, define and measure priority lag explicitly; this is not identical to immediate updates.
- Drain pending completions safely at phase boundaries and shutdown.

### Relevant files

- `src/atariagent/training/batch_worker.py`
- `src/atariagent/replay.py`
- `scripts/train_agent.py`

### Validation

Use pytest coverage for completion ownership, exceptions, stale IDs, and shutdown. Measure replay-lock wait, sampling throughput, and end-to-end update throughput.

## Phase 5 — Optimize the native search inference loop

### Proposal

Investigate, one change at a time:

1. Reusable staging and output buffers.
2. CUDA Graph capture of one simulation’s GPU inference section.
3. Root-count buckets for variable cache-miss batches.
4. Inference-only convolution/BatchNorm folding where applicable.

### Rationale

`cpp/inference/tree_search.cpp` performs CPU traversal, three small H2D copies, eager LibTorch inference, and a blocking D2H result transfer for each simulation. PUCT repeats this 50 times per search. Compiling the Python learner does not compile this native path.

### Requirements

- Preserve search behavior and recurrent hidden-state resets.
- Account for target-weight updates when managing captured graphs or folded weights.
- Bound graph/buffer memory across root-count buckets.
- Validate any padding does not affect real roots.
- Keep inference transformations out of training-mode modules.

### Validation

- Compare policies and root values under fixed inputs and search RNG settings.
- Benchmark four-root self-play, small cache-miss batches, typical late-training batches, and cold-cache batches.
- Include graph warmup/capture cost in full-run accounting.
- Measure whether launch reduction improves the end-to-end critical path.

### Limitation

CPU tree traversal still requires each simulation’s output. CUDA Graphs reduce launch overhead but do not remove that dependency.

## Phase 6 — Reduce redundant frame transfers

### Proposal

Compare:

- Reusing one device-resident frame payload between reanalysis and learning.
- Transferring only frames needed by cache misses.
- An adaptive strategy based on miss density.

### Rationale

`cpp/reanalysis/engine.cpp` transfers the full reanalysis sequence whenever either policy or bootstrap-value misses require inference. The learner later transfers an overlapping sequence.

At default batch size and RGB dimensions, raw payload sizes are approximately:

- Reanalysis: 94.5 MiB per transferred sequence batch.
- Learner: 60.75 MiB per frame batch.

These are payload calculations, not measured bottleneck costs.

### Requirements

- Preserve raw observations and learner augmentation behavior.
- Handle policy and bootstrap misses independently.
- Preserve bounded memory and cross-stream ownership.
- Avoid expanding overlapping frame stacks into a larger transfer than the original payload.

### Validation

Measure transfer bytes, CPU gathering time, transfer latency, peak memory, and end-to-end throughput at several cache-hit rates.

## Deferred changes

Do not prioritize these without evidence or a separate algorithmic decision:

- Larger prefetch queues: they cannot fix serialized GPU work and increase memory and target lag.
- PER changes purely for speed: they change sampling and learning behavior.
- Longer cache TTL or fewer search simulations: these trade target quality/freshness for speed.
- Switching PUCT-50 to Gumbel-16: useful as a separate algorithm/runtime comparison, not a semantics-preserving optimization.
- Combining all consistency-target representation passes into one batch without accounting for training-mode BatchNorm: this changes statistics and running-state updates.

## Acceptance criteria

For each systems experiment:

1. Change one factor at a time against a fixed baseline.
2. Keep environment budget, learner updates, batch size, search budget, cache TTL, precision, and evaluation protocol fixed unless that factor is explicitly under study.
3. Pass relevant pytest tests and add tests for new synchronization/ownership behavior.
4. Demonstrate repeatable end-to-end improvement beyond timing noise.
5. Report warmup cost, steady-state speed, tail stalls, peak memory, and any semantic differences.
6. Validate learning quality with full runs before changing defaults; use multiple training seeds when assessing score regressions.
7. Reject changes that only move work outside the measured region.

## Execution order

1. Instrument and establish the baseline.
2. Reduce intermediate evaluation overhead as an independent workflow improvement.
3. Test explicit reanalysis stream ownership.
4. Move priority transfer outside the replay lock.
5. Optimize native inference if profiling confirms it remains critical.
6. Optimize frame movement if transfer costs remain significant.

Use `uv run` for Python/project tools, Hydra for configuration, and pytest for tests. Do not launch long training experiments or commit changes without user approval.
