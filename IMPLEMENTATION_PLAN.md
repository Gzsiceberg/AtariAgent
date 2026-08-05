# Atari EfficientZero and EfficientZero V2 Implementation Plan

## 1. Goal and agreed direction

Build a from-scratch, Atari-focused implementation in two sequential milestones:

1. **EfficientZero (NeurIPS 2021)**
2. **EfficientZero V2 (ICML 2024)**, reusing the validated EfficientZero foundation

The implementation will have two Hydra profiles from the beginning:

- **`demo`** — reduced model/search/workload for an RTX 3090 (24 GB); intended to prove that the complete system runs and learns.
- **`atari_100k`** — paper-oriented settings for serious reproduction runs on Runpod.

The code path should remain the same across profiles. A demo profile may reduce capacity, actors, simulations, replay size, and training duration, but it must not silently replace core algorithm components.

This project will implement the algorithm rather than depend on RLlib. Ray will be restricted to orchestration and distributed training.

## 2. Scope

### Included

- Gymnasium/ALE Atari environments with configurable games.
- Atari preprocessing, action repeat, frame stacking, life handling, and evaluation.
- EfficientZero representation, dynamics, prediction, value-prefix, and consistency networks.
- Batched MuZero-style MCTS for EfficientZero.
- Self-play, prioritized replay, target generation, reanalysis, learning, checkpointing, and evaluation.
- EfficientZero V2 sampled Gumbel search and its Atari-specific training changes.
- Local single-GPU execution on the RTX 3090.
- Runpod multi-GPU execution using Ray Train for learners and Ray Core actors for online RL services.
- Hydra configuration, W&B metrics, reproducible seeds, tests, profiling, and documentation.

### Initially excluded

- Continuous-action control and DM Control.
- A general-purpose RL framework.
- RLlib policies/trainers.
- Multi-node scaling before single-node multi-GPU scaling works.
- C++/CUDA MCTS before a correct, profiled Python/Torch baseline exists.

## 3. Success criteria

### Correctness

- Unit tests cover scalar transforms, support encoding, targets, replay priorities, model shape contracts, tree backup, action selection, and episode boundaries.
- Tiny deterministic tree tests agree with hand-calculated MCTS results.
- Fixed-seed integration tests can collect trajectories, sample replay, train, reanalyse, save, and resume.
- Reference settings and equations are traceable to the papers or official repositories.

### Local demo

- A complete run fits in 24 GB VRAM and remains stable for several hours.
- W&B shows environment steps, update steps, returns, losses, search statistics, throughput, replay age, and GPU utilization.
- Evaluation return improves over a random-policy baseline on the selected demo game.
- Checkpoint resume reproduces counters, optimizer/scheduler state, target network, and replay metadata.

### Scale-out

- The same experiment configuration runs with one or multiple Ray Train learner workers.
- One process is used per learner GPU with PyTorch DDP gradient synchronization.
- Self-play, replay, and reanalysis scale independently from learner GPUs.
- Multi-node runs use shared object/checkpoint storage and recover from worker restart without corrupting replay or counters.

### Paper-oriented validation

- Environment interaction is counted in Atari transitions after action repeat and reported unambiguously alongside raw frames.
- The `atari_100k` profile limits training data to 100,000 environment transitions.
- Results include multiple seeds and human-normalized score only when the required random/human reference scores are documented.
- Any deviation from a paper or official implementation is recorded in the experiment config and W&B metadata.

## 4. Architectural rules

1. **Correctness before speed.** Start with readable PyTorch/Python implementations and optimize only measured bottlenecks.
2. **Separate algorithm from distribution.** Core model, search, replay schema, losses, and targets must run without Ray.
3. **Separate acting from learning.** Define explicit versioned payloads for trajectories, batches, priorities, and weights.
4. **One canonical configuration.** Hydra composes algorithm, environment, model, runtime, and experiment profiles.
5. **Explicit counters.** Keep separate `env_steps`, `raw_frames`, `learner_steps`, `reanalyzed_positions`, and `episodes`.
6. **Version all weights.** Every self-play trajectory and reanalysis result records the network version that generated it.
7. **No hidden paper deviations.** Profile overrides must be visible in saved resolved configs.
8. **Atari first.** Interfaces may permit future continuous actions, but no continuous-control implementation is required for V2.

## 5. Proposed repository layout

```text
atariagent/
  __init__.py
  cli/
    train.py
    evaluate.py
    benchmark.py
  config/
    config.yaml
    algorithm/{efficientzero,efficientzero_v2}.yaml
    env/atari.yaml
    model/atari.yaml
    runtime/{local,ray}.yaml
    experiment/{demo,atari_100k}.yaml
  envs/
    atari.py
    preprocessing.py
  models/
    common.py
    efficientzero.py
    support.py
  search/
    tree.py
    mcts.py
    gumbel.py
  replay/
    schema.py
    buffer.py
    sum_tree.py
    targets.py
  training/
    losses.py
    learner.py
    schedules.py
  workers/
    self_play.py
    reanalyse.py
    evaluator.py
  distributed/
    interfaces.py
    local.py
    ray_runtime.py
    ray_train.py
  logging/
    metrics.py
    video.py
  checkpoint.py
  utils/
    seeds.py
    profiling.py
tests/
  unit/
  integration/
  reference/
scripts/
  train_local.sh
  train_runpod.sh
```

The exact package name can be adjusted before implementation, but algorithm code should not be placed in `main.py`.

---

# Part I — EfficientZero

## Step 0 — Freeze references and experiment contracts

- [ ] Save links and commit hashes for the EfficientZero paper, supplement, and official implementation.
- [ ] Extract a reference table for Atari preprocessing, model dimensions, support transform, loss weights, optimizer, schedules, replay, reanalysis, MCTS, and evaluation.
- [ ] Mark each item as **paper**, **official-code**, or **project deviation**.
- [ ] Define the trajectory and training-batch schemas before workers are written.
- [ ] Define completion gates for `smoke`, `demo`, and `atari_100k` runs.
- [ ] Record package versions, CUDA version, GPU model, Git commit, resolved Hydra config, and seed in every run.

**Deliverable:** `docs/reference/efficientzero.md` plus schema/config tests.

## Step 1 — Project and quality foundation

- [ ] Convert the project into an importable `atariagent` package.
- [ ] Add development dependencies: `pytest`, `pytest-cov`, `ruff`, and static type checking.
- [ ] Add a CUDA-aware startup diagnostic for Python, Torch, CUDA, ALE, OpenCV, Ray, and GPU memory.
- [ ] Add Hydra entry points for train, evaluate, and benchmark.
- [ ] Add deterministic seeding for Python, NumPy, Torch, CUDA, ALE, and Ray workers.
- [ ] Add CI-safe CPU tests; mark CUDA and long-running tests separately.
- [ ] Add structured logging and W&B offline mode for tests.

**Gate:** formatting, lint, unit tests, config composition, and CPU imports pass from a clean `uv sync`.

## Step 2 — Atari environment pipeline

- [ ] Wrap Gymnasium `ALE/<Game>-v5` environments behind a project-owned interface.
- [ ] Implement and test action repeat/frame skip, max-pooling of recent frames, resizing to `96×96`, RGB handling, frame stacking, reward clipping, no-op starts, and terminal/truncation semantics.
- [ ] Make episodic-life behavior explicit and separate training episodes from true game-over episodes.
- [ ] Expose minimal/full action-set selection through config.
- [ ] Store frames efficiently as `uint8`; convert and normalize only at model boundaries.
- [ ] Add evaluation mode with no exploration noise and configurable reward clipping behavior.
- [ ] Benchmark environment FPS with one and several CPU workers.

**Tests:** observation shape/range, frame ordering, deterministic reset, action mapping, life loss, truncation, and transition/frame counters.

**Gate:** random-policy rollouts complete on at least two Atari games and produce reproducible videos and statistics.

## Step 3 — Mathematical and data primitives

- [ ] Implement the signed scalar transform and inverse transform.
- [ ] Implement scalar-to-categorical-support projection and support-to-scalar decoding.
- [ ] Implement masked categorical losses for value and value-prefix targets.
- [ ] Define trajectory records containing observations, actions, rewards, policies, root values, legal actions, done flags, network version, and timestamps/counters.
- [ ] Implement n-step/TD targets with episode-boundary masking.
- [ ] Implement EfficientZero’s stale-data-aware dynamic TD horizon/off-policy correction.
- [ ] Implement prioritized replay probabilities, importance weights, and priority updates.
- [ ] Implement a memory budget calculator before allocating replay storage.

**Tests:** scalar round trips, support endpoints, exact hand-computed returns, terminal handling, priority distribution, and serialization compatibility.

## Step 4 — EfficientZero network

Implement independent modules with explicit tensor contracts:

- [ ] **Representation network:** stacked RGB observations to a normalized latent state.
- [ ] **Dynamics network:** latent state and discrete action to next latent state.
- [ ] **Value-prefix network:** LSTM-based prefix prediction and hidden-state reset at the configured horizon.
- [ ] **Prediction network:** policy logits and categorical value logits.
- [ ] **Consistency branch:** projector/predictor with a stop-gradient target representation.
- [ ] `initial_inference()` and `recurrent_inference()` APIs shared by learner and search.
- [ ] Target-network creation and parameter synchronization.
- [ ] Initialization, latent normalization, AMP-safe operations, and gradient scaling.

**Tests:** exact shapes for root/unrolled inference, legal-action masks, hidden reset, gradient presence/absence, target-network isolation, AMP forward/backward, and save/load equivalence.

**Gate:** one synthetic batch completes forward, loss, backward, optimizer step, and target update within the 3090 memory budget.

## Step 5 — Correct baseline MCTS

- [ ] Implement node/edge statistics: prior, visit count, value sum, reward/value-prefix, children, and latent references.
- [ ] Implement PUCT selection, expansion through recurrent inference, discounted backup, and min-max value normalization.
- [ ] Implement root exploration noise and temperature-based action selection.
- [ ] Support legal action masks and terminal leaves.
- [ ] Batch leaf inference across roots while preserving a simple unbatched reference path.
- [ ] Track search diagnostics: depth, root entropy, mean/maximum Q, visit distribution, and inference batch utilization.

**Tests:** toy trees with known values, backup signs/discounts, root noise reproducibility, visit counts, temperature limits, and batched/unbatched equivalence.

**Gate:** MCTS improves action selection over raw priors in deterministic toy environments and can drive an Atari environment without memory growth.

## Step 6 — Replay and target preparation

- [ ] Build a bounded prioritized replay buffer over complete game segments.
- [ ] Sample valid start positions with enough context for stacked observations and unroll targets.
- [ ] Avoid crossing true episode boundaries during unroll.
- [ ] Generate value, policy, value-prefix, consistency, masks, and importance-weight targets.
- [ ] Reconstruct stacked observations without storing redundant full stacks where practical.
- [ ] Support replay save/restore or a documented warm-restart mode.
- [ ] Report replay utilization, age distribution, sample latency, and priority statistics.

**Gate:** a generated trajectory can be inserted, sampled at every valid boundary, trained, reprioritized, serialized, and restored.

## Step 7 — Learner and EfficientZero losses

- [ ] Implement unrolled recurrent training over the configured number of steps.
- [ ] Add policy cross-entropy, categorical value loss, categorical value-prefix loss, and SimSiam-style consistency loss.
- [ ] Apply masks, prioritized-replay importance weights, hidden-state gradient scaling, loss coefficients, and unroll normalization exactly as documented.
- [ ] Add optimizer, warm-up/decay schedule, gradient clipping, AMP, and target-network synchronization.
- [ ] Return per-sample priorities based on value prediction error.
- [ ] Log total/component losses, gradient norms, parameter norms, learning rate, throughput, and CUDA memory.
- [ ] Add NaN/Inf checks with a debug batch dump.

**Tests:** zero-mask behavior, loss-weight arithmetic, stopped consistency target, deterministic update, priority calculation, and AMP/full-precision agreement within tolerance.

## Step 8 — Reanalysis

- [ ] Recompute bootstrap values with the current target network.
- [ ] Re-run MCTS for the configured fraction of policy targets.
- [ ] Preserve original targets for positions not selected for policy reanalysis.
- [ ] Apply stale-data-aware TD horizon selection from the sample’s age.
- [ ] Make root-value bootstrap optional and measure its extra search cost.
- [ ] Record source and network version for every regenerated target.
- [ ] Batch reanalysis inference and prevent it from starving learner GPU work.

**Gate:** tests prove that fresh and stale samples choose the expected horizons and that reanalysed targets change only intended fields.

## Step 9 — Local asynchronous pipeline

Implement a non-Ray local runtime first:

```text
CPU Atari env workers -> self-play/MCTS -> replay
                                      replay -> target prep -> learner (RTX 3090)
                                      replay -> reanalysis -> replay updates
                                      weights -> self-play/reanalysis/evaluator
```

- [ ] Begin with a single-process synchronous debug mode.
- [ ] Add bounded multiprocessing queues and backpressure.
- [ ] Batch model inference for self-play and reanalysis.
- [ ] Synchronize immutable versioned weight snapshots at configured intervals.
- [ ] Ensure graceful shutdown drains or safely discards in-flight work.
- [ ] Add periodic evaluation, video capture, atomic checkpoints, and resume.
- [ ] Expose worker counts and CPU affinity through Hydra.

**Gate:** a fixed-seed integration run executes collection, replay, reanalysis, learning, evaluation, checkpointing, and resume without deadlock.

## Step 10 — Local RTX 3090 profiles and optimization

Create three configurations:

1. **`smoke`** — minutes, tiny replay/model/search; validates mechanics only.
2. **`demo`** — hours, reduced but algorithm-complete; expected to improve over random.
3. **`atari_100k`** — reference-oriented settings; may take substantially longer.

Optimization order:

- [ ] Profile environment, search, replay, transfer, learner, and reanalysis separately.
- [ ] Improve inference batching and eliminate host/device synchronization.
- [ ] Use `uint8` replay storage, pinned staging buffers, AMP, channels-last where measured beneficial, and `torch.compile` only after numerical parity tests.
- [ ] Tune actor/reanalysis rates to maintain a healthy replay ratio.
- [ ] Add optional Cython/C++ tree storage/search only if MCTS remains the measured bottleneck.
- [ ] Record throughput before and after every optimization.

**Do not** optimize by removing reanalysis, consistency learning, or value-prefix learning from the canonical demo.

## Step 11 — EfficientZero validation

Validation ladder:

- [ ] Random-policy Atari baseline.
- [ ] Network-only policy baseline without search.
- [ ] Search with an untrained model smoke test.
- [ ] Short overfit test on a frozen replay slice.
- [ ] End-to-end smoke run.
- [ ] Local demo with at least three evaluation seeds.
- [ ] One full 100k-transition run.
- [ ] Multiple-seed benchmark after the full pipeline is stable.

Ablations to diagnose failures:

- [ ] no consistency loss;
- [ ] no reanalysis;
- [ ] fixed versus dynamic TD horizon;
- [ ] raw policy versus MCTS policy;
- [ ] value-prefix reset behavior.

**Milestone:** tag a reproducible `efficientzero-v1` release before starting V2.

---

# Part II — EfficientZero V2 for Atari

## Step 12 — Freeze the V2 delta

- [ ] Save the V2 paper, appendix, official repository commit, and released Atari config.
- [ ] Build a delta table against this project’s validated EfficientZero implementation.
- [ ] Confirm Atari-specific behavior independently from continuous-control settings.
- [ ] Record the official repository warning that released Atari behavior may not reproduce every paper result.
- [ ] Keep an `algorithm=efficientzero` regression suite active while adding V2.

Expected V2 Atari deltas include:

- sampled Gumbel search;
- root sequential halving;
- completed/mixed Q-value computation;
- search-improved policy targets rather than normalized visit counts alone;
- V2 value-target mixing and Atari loss/config changes;
- lower-search-budget operation where supported by the reference config.

## Step 13 — Generalize search interfaces without breaking V1

- [ ] Define a planner interface returning selected action, target policy, root value, and diagnostics.
- [ ] Keep the existing PUCT planner unchanged behind that interface.
- [ ] Add candidate-action abstractions; for Atari, candidates are discrete legal actions.
- [ ] Reuse latent inference and tree storage where semantics match.
- [ ] Add golden regression tests proving unchanged V1 search outputs for fixed seeds.

## Step 14 — Implement sampled Gumbel search

- [ ] Implement seeded Gumbel noise generation.
- [ ] Implement root action scoring from policy logits and transformed completed Q values.
- [ ] Implement sequential-halving simulation allocation for arbitrary legal action counts and small budgets.
- [ ] Implement non-root V2 selection rules.
- [ ] Implement completed Q values for visited and unvisited candidate actions.
- [ ] Derive the V2 search-improved policy target exactly from the paper/reference code.
- [ ] Handle games whose action counts are smaller than the nominal candidate/top-action settings.
- [ ] Batch recurrent inference across trees.

**Tests:** hand-calculated ranking, deterministic Gumbel samples, simulation-budget conservation, elimination rounds, completed-Q edge cases, legal-action masking, and official-code parity fixtures where licensing permits.

## Step 15 — Implement V2 targets and learner changes

- [ ] Add V2 mixed value targets and TD-lambda behavior according to the Atari reference configuration.
- [ ] Add V2 policy objective using the planner’s improved policy target.
- [ ] Add only the entropy term used by the discrete Atari configuration; do not import continuous SAC-style temperature logic by assumption.
- [ ] Apply V2 loss coefficients, optimizer, scheduler, support discretization, and gradient clipping through algorithm-specific Hydra config.
- [ ] Keep common representation/dynamics/value-prefix/consistency code shared unless a documented V2 difference requires a fork.
- [ ] Add side-by-side tests for V1 and V2 target generation from the same synthetic trajectory.

## Step 16 — Integrate and validate V2 locally

- [ ] Run all V1 regression tests.
- [ ] Run V2 search/target/learner unit tests.
- [ ] Complete synchronous and asynchronous smoke runs.
- [ ] Compare V1 and V2 search quality at equal simulation budgets on toy tasks and Atari replay states.
- [ ] Run the same local demo game and evaluation protocol used for V1.
- [ ] Profile whether V2’s lower simulation count produces better environment-step or wall-clock efficiency.
- [ ] Run a 100k-transition V2 experiment only after local stability.

**Milestone:** tag a reproducible `efficientzero-v2-atari` release.

---

# Part III — Ray and Runpod scaling

## Step 17 — Add Ray without coupling it to the algorithm

Use two Ray layers for different responsibilities:

- **Ray Core actors:** environment/self-play workers, replay shards, reanalysis workers, evaluators, and weight distribution.
- **Ray Train `TorchTrainer`:** one learner process per GPU, DDP gradient synchronization, rank-aware checkpoint/reporting.

Do not use RLlib.

- [ ] Define local and Ray implementations of the same worker/service interfaces.
- [ ] Make all cross-process payloads bounded, versioned, and measurable.
- [ ] Place large immutable weight snapshots in the Ray object store.
- [ ] Shard replay when a single replay actor becomes a measured bottleneck.
- [ ] Reserve explicit CPU/GPU resources for every actor.
- [ ] Avoid fractional GPU sharing until profiling proves safe and memory limits are enforced.

## Step 18 — Multi-GPU learner with Ray Train

- [ ] Wrap only the learner loop in `TorchTrainer`.
- [ ] Use one Ray Train worker per GPU and PyTorch DDP.
- [ ] Ensure each rank receives equal-size valid batches and performs the same number of updates.
- [ ] Aggregate new priorities and metrics without duplicate updates.
- [ ] Restrict checkpoint writing and global-step ownership to rank 0.
- [ ] Broadcast/restore learner, optimizer, scaler, scheduler, target network, RNG, and counters consistently.
- [ ] Publish versioned inference weights independently of DDP checkpoints.
- [ ] Test one worker locally before two workers on a multi-GPU Runpod instance.

**Important:** online self-play/reanalysis does not become distributed merely by wrapping the learner in Ray Train; those services remain Ray Core actors and must be scaled separately.

## Step 19 — Runpod single-node multi-GPU

- [ ] Build a pinned container or reproducible bootstrap script with matching CUDA/Torch versions.
- [ ] Mount persistent storage for checkpoints, resolved configs, logs, and optional replay snapshots.
- [ ] Store W&B credentials as Runpod secrets, never in the repository.
- [ ] Start with one node and one learner GPU.
- [ ] Scale to one learner process per GPU.
- [ ] Allocate remaining GPU capacity to batched self-play/reanalysis inference only after learner scaling is correct.
- [ ] Validate NCCL, object-store memory, `/dev/shm`, CPU count, disk throughput, and shutdown behavior.
- [ ] Compare samples/sec, updates/sec, GPU utilization, and cost per million transitions against the local 3090.

## Step 20 — Multi-node Runpod, only if justified

- [ ] Use a stable Ray head/worker startup procedure and explicit network/security settings.
- [ ] Use S3-compatible or shared storage visible to every node for Ray Train checkpoints.
- [ ] Verify node loss and restart on a short fault-injection run.
- [ ] Ensure replay ownership and checkpoint semantics survive actor reconstruction.
- [ ] Measure network transfer from weight broadcasts, replay batches, and priority updates.
- [ ] Scale actors and learners independently; stop when cost-normalized throughput no longer improves.

---

# 6. Configuration strategy

Hydra groups should separate concerns:

```yaml
algorithm: efficientzero          # or efficientzero_v2
env: atari
model: atari
runtime: local                    # or ray
experiment: demo                  # or atari_100k
```

Every run saves:

- the fully resolved Hydra config;
- Git commit and dirty status;
- dependency lock hash;
- system/CUDA/GPU details;
- all seeds;
- algorithm/reference version;
- checkpoint/replay versions.

Config validation must reject combinations that alter semantics accidentally, such as an unroll longer than available targets, invalid support bounds, too few search simulations, or frame-stack mismatches.

# 7. W&B metric contract

At minimum, log:

- **Environment:** clipped/unclipped return, episode length, lives, environment transitions, raw frames.
- **Learner:** total and component losses, learning rate, gradient norm, AMP scale, updates/sec.
- **Search:** simulations/action, depth, root value, root entropy, visit entropy, max Q, inference batch size.
- **Replay:** size, fill ratio, sample age, priority distribution, replay ratio, sampling latency.
- **Reanalysis:** positions/sec, fraction reanalysed, target age, weight-version lag.
- **System:** actor FPS, queue depth, CPU/RAM, GPU utilization/memory, checkpoint duration.
- **Evaluation:** fixed-seed returns, videos, best/last checkpoint, random baseline comparison.

# 8. Commit and release sequence

Use small, reviewable commits in this order:

1. package/config/test foundation;
2. Atari environment and tests;
3. support math and replay schemas;
4. EfficientZero model;
5. reference MCTS;
6. replay and target generation;
7. learner/losses;
8. reanalysis;
9. local asynchronous runtime;
10. local demo and optimizations;
11. EfficientZero validation/release;
12. V2 planner abstraction and Gumbel search;
13. V2 targets/learner;
14. V2 validation/release;
15. Ray Core runtime;
16. Ray Train DDP learner;
17. Runpod deployment and scaling.

Do not combine the first EfficientZero implementation, V2 changes, and Ray scale-out in one development branch or validation step.

# 9. Decision gates to ask before costly work

These decisions are intentionally deferred until evidence is available:

1. **Before the first multi-hour demo:** choose the primary Atari game and measurable target. Default smoke candidate: `Pong`; the released V2 config can also be checked with `Asterix`.
2. **Before replay implementation:** choose RAM/disk replay persistence based on measured trajectory size and available local RAM.
3. **Before optimized MCTS:** profile the correct Python/Torch version, then choose Torch vectorization, Cython/C++, or another backend.
4. **Before the first paid Runpod run:** choose GPU type/count, budget ceiling, persistent storage, and expected run duration.
5. **Before multi-node deployment:** confirm that single-node scaling is insufficient and define recovery requirements.
6. **Before claiming reproduction:** select games, seed count, evaluation protocol, and acceptable variance from reported scores.

# 10. Reference sources

- EfficientZero paper: <https://arxiv.org/abs/2111.00210>
- Official EfficientZero implementation: <https://github.com/YeWR/EfficientZero>
- EfficientZero V2 paper: <https://proceedings.mlr.press/v235/wang24at.html>
- Official EfficientZero V2 implementation: <https://github.com/Shengjiewang-Jason/EfficientZeroV2>
- Gymnasium Atari/ALE: <https://gymnasium.farama.org/environments/atari/>
- Ray Train PyTorch guide: <https://docs.ray.io/en/latest/train/getting-started-pytorch.html>
- Ray GPU scaling guide: <https://docs.ray.io/en/latest/train/user-guides/using-gpus.html>

The papers and pinned official source revisions—not this summary—remain authoritative for equations and reproduction hyperparameters.
