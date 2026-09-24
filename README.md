# AtariAgent: EfficientZero for Atari 100k

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](pyproject.toml)

**Reproduce EfficientZero-style Atari 100k experiments on a single consumer GPU.**

AtariAgent is an independent PyTorch/C++ implementation of [EfficientZero](https://arxiv.org/abs/2111.00210) for the Atari 100k benchmark. It combines the Atari model and learning objectives introduced by EfficientZero with selected [EfficientZero V2](https://arxiv.org/abs/2403.00564) ideas, including Gumbel tree search and mixed value targets.

The project is intended both for reproducing sample-efficient Atari experiments and as a compact base for new model-based reinforcement-learning work. It uses one process and one CUDA GPU instead of a distributed Ray cluster or a multi-GPU worker topology.

This is primarily a **systems and accessibility contribution**, not a new reinforcement-learning algorithm or a bit-for-bit reimplementation of either official repository.

## Reproduction snapshot

The following scores come from completed local runs in `runs/` and `evaluations/`. Every AtariAgent entry is the **final 120,000-update checkpoint** after 100,000 environment transitions, trained with seed `2`, Gumbel search, 16 simulations, and BF16. Scores are raw returns from 16 complete evaluation episodes using seeds 2–17, greedy actions, sticky-action `ALE/*-v5` environments, and no terminal-on-life-loss evaluation.

| Game | Reanalysis cache clearing | AtariAgent final score, mean ± episode std | Median | EfficientZero paper |
| --- | ---: | ---: | ---: | ---: |
| Alien | Disabled | **816.9 ± 241.3** | 670 | 808.5 ± 204.4 |
| Breakout | Every 200 updates | **343.3 ± 86.2** | 411 | 414.1 ± 17.4 |
| Ms. Pac-Man | Every 200 updates | **1,649.4 ± 247.0** | 1,820 | 1,281.2 ± 130.4 |

These results show that the implementation learns effectively within the 100k interaction budget, but they are **not yet a full benchmark reproduction**. Each AtariAgent number is one training run evaluated over 16 episodes. The paper values are means over three independently trained agents, each evaluated with 32 seeds, as reported in the [EfficientZero supplementary material](https://papers.nips.cc/paper_files/paper/2021/file/d5eca8dc3820cad9fe56a3bafda65ca1-Supplemental.pdf). Differences in evaluation sample size and implementation mean the columns should be treated as a reference rather than a strict statistical comparison.

Training writes the raw per-episode rewards, checkpoint paths, means, medians, and standard deviations to `evaluations/<game>/agent_evaluations.json` so results can be inspected rather than inferred from a plot.

## Measured wall-clock performance

On `ALE/Alien-v5`, the optimized Gumbel configuration completed **100,000 environment transitions and 120,000 learner updates**—100,000 during data collection plus 20,000 offline updates—in **2 hours 55 minutes**. This time includes the final 16-episode evaluation.

| Hardware | Search | Reanalysis cache | Precision | Time |
| --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 3060 Ti (8 GB), Intel Core i9-13900KF | Gumbel, 16 simulations | No periodic clearing | BF16 | 2:55:39 |

This is one historical systems result, not a throughput guarantee. Wall-clock time varies by game, search budget, cache freshness, CPU, and software version. The measured run disabled periodic cache clearing; current training instead ramps policy-cache clearing from 100 to 1,000 learner updates over the first half of collection. For comparison, a measured 50-simulation PUCT run on the same machine took approximately 4 hours 6 minutes.

## Why AtariAgent?

- **Single-process, single-GPU training.** No Ray, DDP, or multi-GPU worker topology is required.
- **Native search and reanalysis.** Batched PUCT/Gumbel tree traversal and target-network inference run through an asynchronous in-process C++/LibTorch pipeline, reducing Python serialization, tensor copies, and CPU/GPU synchronization.
- **Replay-state reanalysis cache.** Search values and policies are cached with bounded age, while deterministic target-network bootstrap predictions are cached until the next model publication. Repeated replay samples evaluate only cache misses.
- **Explicit freshness/throughput trade-off.** Hydra settings control periodic cache clearing, allowing experiments to choose between fresher targets and higher throughput.
- **V1/V2 search comparison.** Conventional PUCT and EfficientZero V2-style Gumbel top-*m* sequential halving are available in the same implementation.
- **Modern experiment tooling.** Typed Hydra configuration, `uv`, BF16, `torch.compile`, bounded asynchronous prefetching, checkpoints, parallel evaluation, and tests are included.

## Quick start

### Requirements

Training currently requires Linux and an NVIDIA CUDA GPU. The optimized configuration requires BF16 support. Native extensions are compiled during installation, so a CUDA toolkit with `nvcc`, CMake 3.20 or newer, and a C++20 compiler are also required.

The measured setup used:

- NVIDIA GeForce RTX 3060 Ti with 8 GB VRAM
- Intel Core i9-13900KF and 64 GB system RAM
- Pop!_OS 24.04 LTS
- Python 3.13.11 and `uv` 0.9.21
- PyTorch 2.13.0+cu130 and CUDA toolkit 13.3
- CMake 3.28.3 and GCC 13.3

Other compatible versions may work but have not yet been included in the tested matrix.

### Install

Install [`uv`](https://docs.astral.sh/uv/), then clone and synchronize the locked environment:

```bash
git clone https://github.com/Gzsiceberg/AtariAgent.git
cd AtariAgent
uv sync
```

Verify the installation, native extensions, and Atari environment:

```bash
uv run python -c "import ale_py, gymnasium as gym; gym.register_envs(ale_py); env = gym.make('ALE/Alien-v5'); env.close()"
uv run pytest -q
```

### End-to-end smoke run

Before starting a full experiment, this small run checks environment collection, replay, native reanalysis, learning, and checkpointing. Its score is not meaningful.

```bash
uv run python scripts/train_agent.py \
    self_play.total_transitions=400 \
    replay.max_transitions=400 \
    replay.warmup_transitions=256 \
    training.steps=10 \
    training.final_steps=0 \
    training.updates_per_iteration=10 \
    training.batch_size=32 \
    training.compile_model=false \
    checkpoint.path=/tmp/atariagent-smoke/agent_latest.pt \
    checkpoint.pre_final_snapshot_path=/tmp/atariagent-smoke/agent_pre_final.pt \
    evaluation.enabled=false
```

### Train an agent

The default preset uses **PUCT**, bounds cached reanalysis targets to 200 learner updates, and trains on Alien:

```bash
uv run python scripts/train_agent.py
```

cuDNN convolution autotuning is disabled by default (`training.cudnn_benchmark=false`) to avoid repeated algorithm searches and GPU synchronization as reanalysis cache-miss batch sizes change. This is a process-wide setting for training and native inference; BF16, TF32, and learner compilation remain unchanged. Set `training.cudnn_benchmark=true` to compare the previous behavior. Autotuning is controlled independently of `training.deterministic`.

Optional [ROSMO-style behavior regularization](https://arxiv.org/abs/2210.05980) can be enabled with `loss.behavior_regularization_weight=0.1` (default: `0.0`, disabled). This adds `-log π(a|s)` for replay actions whose detached model-based advantage `r̂ + γ V̂(next) - V̂(current)` is strictly positive. Predicted immediate rewards are recovered from value-prefix differences, respecting LSTM resets; γ uses the existing frame-skip-adjusted training discount. The loss covers valid replay transitions (not the final unroll state), uses replay importance weights and the existing unroll scaling, and applies during both online and final offline updates. This uses the current learner's recurrent model predictions; it does not enable ROSMO policy improvement or a separate offline-data pipeline.

Choose another Atari environment through a Hydra override:

```bash
uv run python scripts/train_agent.py environment.id=ALE/Breakout-v5
```

Use the EfficientZero V2-style Gumbel preset:

```bash
uv run python scripts/train_agent.py \
    search=gumbel \
    environment.id=ALE/Breakout-v5
```

The two search presets are:

| Preset | Search | Periodic cache clearing | Target TTL | Select with |
| --- | --- | --- | --- | --- |
| PUCT (default) | Conventional PUCT MCTS | Disabled (`0`) | 200 updates | `search=puct` |
| Gumbel | Top-*m* sequential halving | Every 400 learner updates | 200 updates | `search=gumbel` |

All settings can be overridden from the command line. Inspect the resolved default configuration with:

```bash
uv run python scripts/train_agent.py --cfg job
```

Checkpoints are written under `checkpoints/<game>/`. Evaluation data and plots are written under `evaluations/<game>/`.

### Rerun only the final learner phase

Immediately before `training.final_steps`, training writes
`checkpoints/<game>/agent_pre_final.pt`. Unlike model-only checkpoints, this
snapshot includes the replay buffer, optimizer, target network, update counter,
and RNG states. With the default 100k RGB replay it is roughly 3 GB.

Rerun only the final learner-only phase with:

```bash
uv run python scripts/train_agent.py \
    environment.id=ALE/Alien-v5 \
    checkpoint.resume_pre_final_path=checkpoints/Alien-v5/agent_pre_final.pt
```

The resumed run skips self-play and starts at update 100,000. To evaluate the
loaded model before applying any final-phase updates, add
`evaluation.evaluate_on_resume=true`. That baseline is recorded at update
100,000 in the configured evaluation history before training continues.
Override `checkpoint.path` and the `evaluation.*_path` settings if you want to
preserve outputs from an earlier final-phase run. Set
`checkpoint.pre_final_snapshot_path=null` to disable snapshot creation. Only
load snapshots you trust; replay snapshots use Python pickle through
`torch.load`.

### Dynamics action encoding

The default `model.action_embedding=true` projects the normalized action plane
into 16 channels with LayerNorm. To use EfficientZero V1's raw scalar action
plane instead (no action projection or LayerNorm):

```bash
uv run python scripts/train_agent.py \
    environment.id=ALE/Qbert-v5 search=gumbel model.action_embedding=false
```

Only action encoding changes; the rest of the network is unchanged. The option
applies to training, self-play, and native reanalysis. Checkpoint evaluation
reads the saved setting; checkpoints without it retain the previous `true`
default. The two modes have incompatible dynamics weights, so start a new run
when switching modes. To resume a pre-final snapshot, select the same mode used
to create it.

### Evaluate a checkpoint

Training performs periodic and final evaluation automatically. A checkpoint can also be evaluated independently:

```bash
uv run python scripts/eval_agent.py checkpoints/Alien-v5/agent_latest.pt
```

Override the number of episodes, device, or evaluation search:

```bash
uv run python scripts/eval_agent.py \
    checkpoints/Alien-v5/agent_latest.pt \
    --episodes 16 \
    --device cuda \
    --search-algorithm gumbel \
    --num-simulations 16
```

### Watch a trained agent

On a machine with a graphical display:

```bash
uv run python scripts/watch_agent.py checkpoints/Alien-v5/agent_latest.pt
```

### Optional Weights & Biases logging

Weights & Biases is disabled by default, so training runs without an account or network connection. Install and enable it with:

```bash
uv sync --extra wandb
uv run wandb login
uv run python scripts/train_agent.py \
    wandb.enabled=true \
    wandb.project=AtariAgent
```

The W&B entity, project, and tags can also be set through overrides such as `wandb.entity=<entity>` and `wandb.tags=[atari,baseline]`. `train/value_loss` and `train/reward_loss` report MAE, `mean(abs(prediction - target))`, for decoded scalar values and cumulative reward prefixes over valid targets, without replay weighting; optimization still uses categorical cross-entropy. Search-target diagnostics include entropy, maximum action probability, and effective action count `exp(H)`. Reanalysis diagnostics include exact cache hit rate.

Four learning-signal metrics are logged at `training.log_every` under `train/`:
- `importance_weight_mean`: mean replay importance weight; small values suppress the data-loss scale.
- `importance_weight_ess_fraction`: `(sum(w)^2 / sum(w^2)) / batch_size`; near 1 means balanced weights, small values mean concentrated weights.
- `representation_feature_variance`: across-sample population variance of root latent features, averaged over coordinates.
- `dynamics_feature_variance`: the same variance at each unroll depth, averaged by valid sample count; padding is excluded.

Feature variances approaching zero can indicate collapse. They use the existing augmented training forwards (not extra inference passes), so compare trends rather than applying a universal cutoff. All four diagnostics are computed without gradients and do not change the training objective.

## Using AtariAgent as a research base

The main extension points are:

```text
configs/                       Hydra experiment configurations
src/atariagent/agent.py        Agent and action-selection interface
src/atariagent/models/         Representation, dynamics, and prediction models
src/atariagent/search/         Python search configuration and interface
src/atariagent/training/       Learner, reanalysis, checkpoints, and logging
src/atariagent/selfplay.py     Atari preprocessing and self-play collection
cpp/search/                    Native batched tree traversal
cpp/reanalysis/                Native target reanalysis and caching
cpp/models/                    LibTorch inference models
scripts/                       Training, evaluation, watching, and benchmarks
tests/                         Unit and integration tests
```

Hydra configuration lives in `configs/train_agent.yaml`, with search presets in `configs/search/`. New experiments normally require configuration overrides rather than edits to the training script.

## How this differs from the official implementations

AtariAgent retains the central EfficientZero learning ideas while redesigning the execution pipeline for constrained hardware.

| Aspect | EfficientZero V1 | EfficientZero V2 | AtariAgent |
| --- | --- | --- | --- |
| Primary scope | Atari 100k | Discrete and continuous control across Atari and DeepMind Control | Atari 100k |
| Atari search | PUCT MCTS, normally 50 simulations | Gumbel top-*m* sequential halving, normally 16 simulations | Native PUCT by default; native Gumbel is also available |
| Value targets | Bootstrapped *n*-step targets; policy reanalysis and root-value targets are separately configured | Mixed TD/search value targets with full policy reanalysis | V2-style mixed value targets and delayed-target-network reanalysis |
| Prioritized replay | Priority exponent 0.6; importance exponent annealed from 0.4 to 1.0; standard launcher inserts at the current maximum | Priority and importance exponents 1.0 with a 0.1 weight floor; maximum-priority insertion | Configurable V1/V2 sampling and weighting; V2 insertion by default, or V1 buffer-maximum-only insertion with `replay.use_max_priority=true` |
| Runtime architecture | Distributed Ray workers with a C++/Cython tree | Distributed Ray workers with Cython/C++ search | Python learner with an asynchronous in-process C++ reanalysis worker |
| Reanalysis reuse | Recomputes sampled targets in reanalysis workers | Recomputes sampled targets in reanalysis workers | Caches targets by replay state and searches only cache misses |
| Hardware objective | Official README recommends four RTX 3090 GPUs for high-throughput training | Official example launches with two GPUs and supports broader workloads | Consumer single-GPU experiments |
| License | GPL-3.0 | GPL-3.0 | MIT |

New trajectories receive a shared priority `max(current_buffer_max, max(trajectory_errors))`, matching V2 replay insertion. The buffer maximum defaults to **1 only when empty**, is measured before FIFO eviction, and is not a historical maximum. Trajectory errors are individual prediction/bootstrap absolute errors plus epsilon, excluding lookahead-only starts; insertion does not add epsilon again. Existing transition priorities are unchanged at insertion, and subsequent learner updates set individual priorities. Every sampled root has a supervised value target and updates its priority, including short tails with zero bootstrap. Snapshot loading preserves saved priorities rather than reinitializing them. Stored `initial_priorities` retain the raw individual errors for compatibility.

Training uses separate prediction and dynamics supervision. Policy/value losses and search reanalysis are restricted to states inside the original block. Reward and consistency losses use every available recorded transition in the unroll, including lookahead, with the recorded actions/rewards/observations; only unavailable padded transitions are masked. The first lookahead state has **no zero-value supervision**. This deliberately differs from V2's shared recurrent mask. TD targets retain V2's Atari rule: bootstrap endpoints lie strictly inside the original block, while lookahead rewards can contribute to returns. Replay stores one `reachable_mask` for the root and recorded successor states. `action_mask` is its zero-copy `[:, 1:]` view; `value_mask` intersects it with original-block policy availability. The learner uses this stored mask directly, without constructing a root-plus-action concatenation. Mixed targets use V2's strict `index > collected_transitions - threshold` recent-sample test. Final collection still flushes pending blocks—unlike V2—retaining available lookahead as context without duplicating replay starts.

Target-network publication clears both policy and value caches on its fixed 1,000-update schedule. Independently, `reanalysis.cache_target_ttl` expires policy/search entries after a bounded number of learner updates (200 by default). Raw bootstrap values need no TTL because they are deterministic for fixed target-network weights.

## Scope and limitations

- Training requires an NVIDIA CUDA GPU; CPU training is not supported.
- The published snapshot contains single-training-seed evidence rather than a complete multi-seed, 26-game Atari 100k benchmark.
- AtariAgent intentionally mixes V1 and V2 design choices and is not expected to reproduce either official implementation bit for bit.
- Aggressive reanalysis caching improves throughput but changes target freshness. This trade-off should be reported with experimental results.
- Wall-clock measurements are hardware- and software-specific.

## Papers and official implementations

- Weirui Ye et al., [“Mastering Atari Games with Limited Data”](https://arxiv.org/abs/2111.00210), NeurIPS 2021 — [official EfficientZero repository](https://github.com/YeWR/EfficientZero).
- Shengjie Wang et al., [“EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data”](https://arxiv.org/abs/2403.00564), ICML 2024 — [official EfficientZero V2 repository](https://github.com/Shengjiewang-Jason/EfficientZeroV2).

This project is an independent implementation and is not affiliated with the paper authors or official repositories.

## Contributing

Bug reports, reproduction results, documentation improvements, and focused pull requests are welcome. Please include the resolved Hydra configuration, commit hash, hardware, software versions, training seed, and raw evaluation statistics when reporting an experiment.

## License

AtariAgent is released under the [MIT License](LICENSE).
