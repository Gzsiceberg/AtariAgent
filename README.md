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

This is one measured systems result, not a throughput guarantee. Wall-clock time varies by game, search budget, cache-clearing interval, CPU, and software version. For example, a measured 50-simulation PUCT run on the same machine took approximately 4 hours 6 minutes.

Reproduce the measured Alien configuration with:

```bash
uv run python scripts/train_agent.py \
    search=gumbel \
    reanalysis.cache_clear_interval=0
```

The standard Gumbel preset clears the cache every 400 updates to favor fresher targets, so it may take longer than the result above.

## Why AtariAgent?

- **Single-process, single-GPU training.** No Ray, DDP, or multi-GPU worker topology is required.
- **Native search and reanalysis.** Batched PUCT/Gumbel tree traversal and target-network inference run through an asynchronous in-process C++/LibTorch pipeline, reducing Python serialization, tensor copies, and CPU/GPU synchronization.
- **Replay-state reanalysis cache.** Corrected TD values, search values, and policies are cached by replay state. Repeated replay samples search only cache misses.
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
    checkpoint.keep_representative=1 \
    checkpoint.path=/tmp/atariagent-smoke/agent_latest.pt \
    evaluation.enabled=false
```

### Train an agent

The default preset uses **PUCT**, keeps cached reanalysis targets for the complete run, and trains on Alien:

```bash
uv run python scripts/train_agent.py
```

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

| Preset | Search | Cache clearing | Select with |
| --- | --- | --- | --- |
| PUCT (default) | Conventional PUCT MCTS | Disabled (`0`) | `search=puct` |
| Gumbel | Top-*m* sequential halving | Every 400 learner updates | `search=gumbel` |

All settings can be overridden from the command line. Inspect the resolved default configuration with:

```bash
uv run python scripts/train_agent.py --cfg job
```

Checkpoints are written under `checkpoints/<game>/`. Evaluation data and plots are written under `evaluations/<game>/`.

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

The W&B entity, project, and tags can also be set through overrides such as `wandb.entity=<entity>` and `wandb.tags=[atari,baseline]`.

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
| Runtime architecture | Distributed Ray workers with a C++/Cython tree | Distributed Ray workers with Cython/C++ search | Python learner with an asynchronous in-process C++ reanalysis worker |
| Reanalysis reuse | Recomputes sampled targets in reanalysis workers | Recomputes sampled targets in reanalysis workers | Caches targets by replay state and searches only cache misses |
| Hardware objective | Official README recommends four RTX 3090 GPUs for high-throughput training | Official example launches with two GPUs and supports broader workloads | Consumer single-GPU experiments |
| License | GPL-3.0 | GPL-3.0 | MIT |

A cache interval of `0` maximizes reuse but allows cached targets to outlive target-network updates. Use a positive `reanalysis.cache_clear_interval` when target freshness is more important than maximum throughput.

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
