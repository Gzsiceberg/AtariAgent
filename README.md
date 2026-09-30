# AtariAgent: EfficientZero for Atari 100k

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](pyproject.toml)

An independent PyTorch/C++ implementation of EfficientZero-style learning on a **single CUDA GPU**, without Ray or a distributed training cluster. AtariAgent combines EfficientZero learning objectives with EfficientZero V2-style Gumbel search and mixed value targets.

## Results

These single-seed results use the default [`configs/train_agent.yaml`](configs/train_agent.yaml): seed **2**, Gumbel search with **16 simulations**, BF16, a mixed-value threshold of **20,000**, and `deterministic=false`. Training uses 100,000 environment transitions and 120,000 learner updates: 100,000 during collection plus 20,000 final offline updates.

Scores are the **highest evaluation mean across 14 evaluations**, each using 16 episodes—not final-checkpoint scores. Runtime is logged run elapsed time (hours:minutes:seconds), excluding queue wait; it varies with hardware and game. Missing results are **N/A**.

| Game | AtariAgent best eval mean | Runtime | Human | EfficientZero V1 | EfficientZero V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| alien | N/A | N/A | 7,127.7 | 808.5 | 1,557.7 |
| amidar | N/A | N/A | 1,719.5 | 148.6 | 184.9 |
| assault | 1,724.625 | 2:58:49 | 742 | 1,263.1 | 1,757.5 |
| asterix | 33,606.25 | 3:30:59 | 8,503.3 | 25,557.8 | 61,810 |
| bank-heist | N/A | N/A | 753.1 | 351 | 1,316.7 |
| battle-zone | N/A | N/A | 37,187.5 | 13,871.2 | 14,433.3 |
| boxing | 39 | 3:14:44 | 12.1 | 52.7 | 75 |
| breakout | 380 | 6:02:52 | 30.5 | 414.1 | 400.1 |
| chopper-command | N/A | N/A | 7,387.8 | 1,117.3 | 1,196.6 |
| crazy-climber | 130,275 | 3:29:40 | 35,829.4 | 83,940.2 | 112,363.3 |
| demon-attack | 17,960.3125 | 3:37:14 | 1,971 | 13,003.9 | 22,773.5 |
| freeway | N/A | N/A | 29.6 | 21.8 | 0 |
| frostbite | N/A | N/A | 4,334.7 | 296.3 | 1,136.3 |
| gopher | 3,628.75 | 3:43:52 | 2,412.5 | 3,260.3 | 3,868.7 |
| hero | N/A | N/A | 30,826.4 | 9,315.9 | 9,705 |
| jamesbond | 393.75 | 2:48:38 | 302.8 | 517 | 468.3 |
| kangaroo | N/A | N/A | 3,035 | 724.1 | 1,886.7 |
| krull | 9,260.625 | 3:00:22 | 2,665.5 | 5,663.3 | 9,080 |
| kung-fu-master | 35,712.5 | 3:01:35 | 22,736.3 | 30,944.8 | 28,883.3 |
| ms-pacman | N/A | N/A | 6,951.6 | 1,281.2 | 2,251 |
| pong | 14.5 | 4:12:16 | 14.6 | 20.1 | 20.8 |
| private-eye | N/A | N/A | 69,571.3 | 96.7 | 99.8 |
| qbert | 16,204.6875 | 2:59:01 | 13,455 | 13,781.9 | 16,058.3 |
| road-runner | 36,450 | 3:32:29 | 7,845 | 17,751.3 | 27,516.7 |
| seaquest | N/A | N/A | 42,054.7 | 1,100.2 | 1,974 |
| up-n-down | 22,555 | 3:06:40 | 11,693.2 | 17,264.2 | 15,224.3 |

### Human-normalized aggregates

All methods use the **same 14 measured games**, excluding N/A entries. These are subset results, not full 26-game benchmark aggregates.

| Method | Normalized mean | Normalized median |
| --- | ---: | ---: |
| AtariAgent | 416.69% | 306.64% |
| EfficientZero V1 | 343.16% | 213.37% |
| EfficientZero V2 | 469.36% | 323.28% |

AtariAgent exceeds human scores on **13/14** games and EfficientZero V1 on **10/14**. Paper results are reference values, not local reruns; their reporting protocols differ from these single-seed, best-evaluation results.

### Cautions

- **Highly unstable games:** Qbert, Pong, Kung-fu-master, Gopher, Up-n-down, and Jamesbond can show large differences across runs and evaluations. Peak scores may not be sustained. Multiple seeds and predefined evaluation protocols are needed for reliable comparisons.
- **Prioritized replay has a large impact on results.** Priority initialization, sampling exponent, importance weights, and priority updates materially affect learning. Keep these settings fixed and report them when comparing experiments.
- Best-over-evaluations selection can inflate scores. This is a partial experimental snapshot, not a complete multi-seed reproduction of either paper.

## Quick start

Requires Linux, an NVIDIA CUDA GPU, Python 3.13+, and a CUDA toolkit with `nvcc`, CMake 3.20+, and a C++20 compiler. BF16 requires compatible hardware.

```bash
git clone https://github.com/Gzsiceberg/AtariAgent.git
cd AtariAgent
uv sync
uv run pytest -q

# Train with the default configuration
uv run python scripts/train_agent.py

# Select another game
uv run python scripts/train_agent.py environment.id=ALE/Breakout-v5

# Inspect all settings
uv run python scripts/train_agent.py --cfg job
```

For runs that save checkpoints:

```bash
uv run python scripts/eval_agent.py checkpoints/Alien-v5/agent_latest.pt
uv run python scripts/watch_agent.py checkpoints/Alien-v5/agent_latest.pt
```

Watching requires a graphical display. Optional W&B logging uses your own account: `uv sync --extra wandb`, `uv run wandb login`, then train with `wandb.enabled=true`.

### Schedule experiments with jobd

I use [jobd](https://github.com/Gzsiceberg/jobd), a job queue and worker scheduler, to run training jobs on Vast.ai instances. An AI coding agent, jobd, and the instructions in [`vastai/vastai_setup.md`](vastai/vastai_setup.md) make it convenient to provision instances, configure workers, and submit experiments:

```bash
SEED=2 ./scripts/train_game_experiments.sh --mixed-value-threshold-20000
jobd -l
```

The script schedules all 26 games by default. Use `START_GAME` / `END_GAME` for an inclusive range or `DRY_RUN=1` to preview. Workers need a configured checkout and logging credentials. The script disables checkpoints and local evaluation files but retains training logs and W&B metrics. See [`vastai/vastai_setup.md`](vastai/vastai_setup.md) for worker provisioning.

## Implementation

- Native batched PUCT/Gumbel search and asynchronous C++/LibTorch reanalysis.
- Cached replay-state search targets with configurable freshness limits.
- Mixed value targets, prioritized replay, BF16, and learner compilation.
- Hydra configuration, `uv` dependency management, and pytest tests.

The implementation mixes V1/V2 design choices and is not a bit-for-bit reproduction. Reanalysis caching trades target freshness for throughput. Training requires CUDA; CPU training is unsupported.

## References and license

- [EfficientZero V1 paper](https://arxiv.org/abs/2111.00210) · [Official implementation](https://github.com/YeWR/EfficientZero)
- [EfficientZero V2 paper](https://arxiv.org/abs/2403.00564) · [Official implementation](https://github.com/Shengjiewang-Jason/EfficientZeroV2)

AtariAgent is independent and unaffiliated with the paper authors. Released under the [MIT License](LICENSE).
