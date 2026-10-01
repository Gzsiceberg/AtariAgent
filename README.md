# AtariAgent: EfficientZero for Atari 100k

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](pyproject.toml)

An independent PyTorch/C++ implementation of EfficientZero-style learning on a **single CUDA GPU**, without Ray or a distributed training cluster. AtariAgent combines EfficientZero learning objectives with EfficientZero V2-style Gumbel search and mixed value targets.

## Results

These single-seed results use the default [`configs/train_agent.yaml`](configs/train_agent.yaml): seed **2**, Gumbel search with **16 simulations**, BF16, a mixed-value threshold of **20,000**, and `deterministic=false`. Training uses 100,000 environment transitions and 120,000 learner updates: 100,000 during collection plus 20,000 final offline updates.

Scores are the **highest evaluation mean across 14 evaluations**, each using 16 episodes—not final-checkpoint scores.

| Game | AtariAgent | Runtime | Human | EfficientZero V1 | EfficientZero V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| alien | 561.9 | 2:45:42 | 7,127.7 | 808.5 | 1,557.7 |
| amidar | 182.8 | 2:54:00 | 1,719.5 | 148.6 | 184.9 |
| assault | 1,724.6 | 2:58:49 | 742.0 | 1,263.1 | 1,757.5 |
| asterix | 33,606.3 | 3:30:59 | 8,503.3 | 25,557.8 | 61,810.0 |
| bank-heist | 345.0 | 2:30:03 | 753.1 | 351.0 | 1,316.7 |
| battle-zone | 9,187.5 | 2:47:14 | 37,187.5 | 13,871.2 | 14,433.3 |
| boxing | 39.0 | 3:14:44 | 12.1 | 52.7 | 75.0 |
| breakout | 380.0 | 6:02:52 | 30.5 | 414.1 | 400.1 |
| chopper-command | 2,593.8 | 2:29:04 | 7,387.8 | 1,117.3 | 1,196.6 |
| crazy-climber | 130,275.0 | 3:29:40 | 35,829.4 | 83,940.2 | 112,363.3 |
| demon-attack | 17,960.3 | 3:37:14 | 1,971.0 | 13,003.9 | 22,773.5 |
| freeway | 0.0 | 2:23:17 | 29.6 | 21.8 | 0.0 |
| frostbite | 4,159.4 | 2:41:25 | 4,334.7 | 296.3 | 1,136.3 |
| gopher | 3,628.8 | 3:43:52 | 2,412.5 | 3,260.3 | 3,868.7 |
| hero | 8,726.6 | 3:07:11 | 30,826.4 | 9,315.9 | 9,705.0 |
| jamesbond | 393.8 | 2:48:38 | 302.8 | 517.0 | 468.3 |
| kangaroo | 1,512.5 | 2:20:40 | 3,035.0 | 724.1 | 1,886.7 |
| krull | 9,260.6 | 3:00:22 | 2,665.5 | 5,663.3 | 9,080.0 |
| kung-fu-master | 35,712.5 | 3:01:35 | 22,736.3 | 30,944.8 | 28,883.3 |
| ms-pacman | 1,370.6 | 2:44:35 | 6,951.6 | 1,281.2 | 2,251.0 |
| pong | 14.5 | 4:12:16 | 14.6 | 20.1 | 20.8 |
| private-eye | 100.0 | 2:38:13 | 69,571.3 | 96.7 | 99.8 |
| qbert | 16,204.7 | 2:59:01 | 13,455.0 | 13,781.9 | 16,058.3 |
| road-runner | 36,450.0 | 3:32:29 | 7,845.0 | 17,751.3 | 27,516.7 |
| seaquest | 1,855.0 | 3:00:07 | 42,054.7 | 1,100.2 | 1,974.0 |
| up-n-down | 22,555.0 | 3:06:40 | 11,693.2 | 17,264.2 | 15,224.3 |

### Human-normalized aggregates

Per-game human normalization is `100 × (score − random) / (human − random)`; aggregates are the arithmetic mean and median of these normalized scores.

| Method | Normalized mean | Normalized median |
| --- | ---: | ---: |
| AtariAgent | 235.8% | 110.2% |
| EfficientZero V1 | 194.3% | 109.0% |
| EfficientZero V2 | 268.0% | 123.5% |

AtariAgent exceeds human scores on **13/26** games, EfficientZero V1 on **17/26**, and EfficientZero V2 on **9/26**. Total logged runtime across runs is **81.7 GPU-hours**, averaging **3.1 hours** per game (median **3.0 hours**).

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
