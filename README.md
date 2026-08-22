# AtariAgent

**EfficientZero for Atari 100k on a single consumer GPU.**

AtariAgent is an independent, performance-oriented implementation of EfficientZero for the [Atari 100k](https://arxiv.org/abs/2111.00210) setting. It combines the Atari model and learning objectives introduced by EfficientZero with selected EfficientZero V2 ideas, including Gumbel tree search and mixed value targets.

This is primarily a **systems and accessibility contribution**, not a new reinforcement-learning algorithm. The goal is to make complete EfficientZero experiments practical without a distributed Ray cluster or multiple high-end GPUs.

## Measured wall-clock performance

On `ALE/Alien-v5`, the optimized Gumbel configuration completed **100,000 environment transitions and 120,000 learner updates**—100,000 during data collection plus 20,000 offline updates—in **2 hours 55 minutes**. The final 16-episode evaluation is included.

| Hardware | Search | Reanalysis cache | Precision | Time |
| --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 3060 Ti (8 GB), Intel Core i9-13900KF | Gumbel, 16 simulations | No periodic clearing | BF16 | 2:55:39 |

This is one measured systems result, not an Atari score benchmark. Wall-clock time varies by game, search budget, cache-clearing interval, CPU, and software version. For example, a measured 50-simulation PUCT run on the same machine took approximately 4 hours 6 minutes.

An equivalent configuration can be selected with:

```bash
uv run python scripts/train_agent.py \
    search=gumbel \
    training.reanalysis_cache_clear_interval=0
```

The standard Gumbel preset clears the cache every 400 updates to favor fresher targets; therefore it may take longer than the result above.

## Key contributions

- **Single-process, single-GPU training.** The training pipeline does not require Ray, DDP, or a multi-GPU worker topology.
- **Native search and reanalysis.** Batched PUCT/Gumbel tree traversal and target-network inference run through an asynchronous in-process C++/LibTorch pipeline, reducing Python serialization, tensor copies, and CPU/GPU synchronization.
- **Replay-state reanalysis cache.** Corrected TD values, search values, and policies are cached by replay state. Repeated replay samples search only cache misses.
- **Explicit freshness/throughput trade-off.** Hydra settings control periodic cache clearing, allowing experiments to choose between fresher targets and higher throughput.
- **V1/V2 search comparison.** Both conventional PUCT and EfficientZero V2-style Gumbel top-*m* sequential halving are available in the same implementation.
- **Modern experiment tooling.** Typed Hydra configuration, `uv`, BF16, `torch.compile`, bounded asynchronous prefetching, checkpoints, parallel evaluation, and tests are included.

## Getting started

### Requirements

The currently tested setup is Linux with:

- An NVIDIA CUDA GPU; the optimized configuration requires BF16 support
- A CUDA-enabled PyTorch installation and matching CUDA toolkit/`nvcc`
- Python 3.13 or newer
- CMake 3.20 or newer
- A C++20 compiler
- [`uv`](https://docs.astral.sh/uv/)

The native extensions are compiled during installation.

### Install

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

All settings can be overridden from the command line. To inspect the resolved default configuration:

```bash
uv run python scripts/train_agent.py --cfg job
```

Checkpoints are written under `checkpoints/<game>/`, and evaluation data and plots are written under `evaluations/<game>/`.

### Optional Weights & Biases logging

Weights & Biases is **disabled by default**, so training runs without a W&B account or network connection. To install and enable it:

```bash
uv sync --extra wandb
uv run wandb login
uv run python scripts/train_agent.py \
    wandb.enabled=true \
    wandb.project=AtariAgent
```

The W&B entity, project, and tags can also be set with Hydra overrides such as `wandb.entity=<entity>` and `wandb.tags=[atari,baseline]`.

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

## How this differs from the official implementations

AtariAgent is not a bit-for-bit reproduction of either official repository. It retains the central EfficientZero learning ideas while redesigning the execution pipeline for constrained hardware.

| Aspect | EfficientZero V1 | EfficientZero V2 | AtariAgent |
| --- | --- | --- | --- |
| Primary scope | Atari 100k | Discrete and continuous control across Atari and DeepMind Control | Atari 100k |
| Atari search | PUCT MCTS, normally 50 simulations | Gumbel top-*m* sequential halving, normally 16 simulations | Native PUCT by default; native Gumbel is also available |
| Value targets | Bootstrapped *n*-step targets; policy reanalysis and root-value targets are separately configured | Mixed TD/search value targets with full policy reanalysis | V2-style mixed value targets and delayed-target-network reanalysis |
| Runtime architecture | Distributed Ray workers with a C++/Cython tree | Distributed Ray workers with Cython/C++ search | Python learner with an asynchronous in-process C++ reanalysis worker |
| Reanalysis reuse | Recomputes sampled targets in reanalysis workers | Recomputes sampled targets in reanalysis workers | Caches targets by replay state and searches only cache misses |
| Hardware objective | Official README recommends four RTX 3090 GPUs for high-throughput training | Official example launches with two GPUs and supports broader workloads | Consumer single-GPU experiments |
| License | GPL-3.0 | GPL-3.0 | MIT |

A cache interval of `0` maximizes reuse but allows cached targets to outlive target-network updates. Use a positive `training.reanalysis_cache_clear_interval` when target freshness is more important than maximum throughput.

## Papers

- Weirui Ye et al., [“Mastering Atari Games with Limited Data”](https://arxiv.org/abs/2111.00210), NeurIPS 2021.
- Shengjie Wang et al., [“EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data”](https://arxiv.org/abs/2403.00564), ICML 2024.

This project is an independent implementation and is not affiliated with the authors of the official EfficientZero repositories.

## License

AtariAgent is released under the [MIT License](LICENSE).
