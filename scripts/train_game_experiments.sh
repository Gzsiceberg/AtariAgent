#!/usr/bin/env bash
# Submit the 26 Atari 100k games to jobd's shared controller queue.
# Uses train_agent.yaml defaults, enables W&B, and disables pre-final snapshots.
# Requires workers provisioned by vastai/setup.yml.
# No local files or credentials are sent to workers; results stay on workers.
# Supply W&B authentication through jobd queue secrets.
# This script neither restarts workers nor manages power or queue concurrency.
#
# Usage: ./scripts/train_game_experiments.sh [--gumbel]
#   --gumbel: use search=gumbel instead of the default PUCT search
# Optional environment:
#   SEED: training seed (default 2)
#   RUN_ID: run label (default timestamp); use distinct IDs on shared storage
#   RUN_ROOT: worker output root (default runs/game_experiments/$RUN_ID;
#             partial ranges append /games_START-END)
#   WORKER_REPO_ROOT: checkout on every worker (default /workspace/AtariAgent)
#   START_GAME / END_GAME: inclusive game numbers (default 1 / 26)
#   WANDB_PROJECT: default AtariAgent; WANDB_ENTITY: optional
#   DRY_RUN=1: print commands without jobd or filesystem changes
# Existing game output directories are rejected on the executing worker.
# No completion-based skipping or automatic retries. Monitor with jobd -l.

set -Eeuo pipefail

experiment=default
search_overrides=()
for option in "$@"; do
    case "$option" in
        --gumbel) experiment=gumbel; search_overrides=("search=gumbel") ;;
        -h|--help) echo "Usage: ${0##*/} [--gumbel]"; exit 0 ;;
        *) printf 'Error: unknown option: %s\n' "$option" >&2; exit 1 ;;
    esac
done

SEED="${SEED:-2}"
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || (( SEED > 4294967295 )); then
    printf 'SEED must be an integer between 0 and 4294967295.\n' >&2
    exit 1
fi
START_GAME="${START_GAME:-1}"
END_GAME="${END_GAME:-26}"
if [[ ! "$START_GAME" =~ ^[1-9][0-9]?$ || ! "$END_GAME" =~ ^[1-9][0-9]?$ ]] ||
    (( START_GAME > END_GAME || END_GAME > 26 )); then
    printf 'Game range must satisfy 1 <= START_GAME <= END_GAME <= 26 (integers, no leading zeros).\n' >&2
    exit 1
fi
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != 1 ]] && ! command -v jobd >/dev/null 2>&1; then
    echo "Error: 'jobd' not found." >&2
    exit 1
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_RUN_ROOT="runs/game_experiments/$RUN_ID"
if (( START_GAME != 1 || END_GAME != 26 )); then
    DEFAULT_RUN_ROOT+="/games_${START_GAME}-${END_GAME}"
fi
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
WORKER_REPO_ROOT="${WORKER_REPO_ROOT:-/workspace/AtariAgent}"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

job_count=0
printf 'Selected game numbers %s–%s\n' "$START_GAME" "$END_GAME"
# Stable game IDs: do not renumber when changing scheduling order.
while IFS='|' read -r number game environment_id; do
    (( number >= START_GAME && number <= END_GAME )) || continue
    output_dir="$RUN_ROOT/$game/$experiment"
    args=(
        "seed=$SEED"
        "environment.id=$environment_id"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        "checkpoint.pre_final_snapshot_path=null"
        "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
        "wandb.enabled=true"
        "wandb.project=$WANDB_PROJECT"
        "wandb.name=${environment_id##*/}_${experiment}_seed${SEED}_${RUN_ID}"
        "wandb.tags=[jobd,atari-100k,$experiment,all-games,$game]"
        "${search_overrides[@]}"
    )
    [[ -z "$WANDB_ENTITY" ]] || args+=("wandb.entity=$WANDB_ENTITY")
    # Positional arguments keep paths and Hydra overrides safe from shell expansion.
    command=(jobd bash -ec '
        set -o pipefail
        cd -- "$1"
        output=$2
        shift 2
        mkdir -p -- "$(dirname -- "$output")"
        mkdir -- "$output"
        uv run --no-sync python scripts/train_agent.py "$@" 2>&1 | tee "$output/training.log"
    ' "jobd-$game" "$WORKER_REPO_ROOT" "$output_dir" "${args[@]}")
    printf '\n=== Scheduling %s (%s) ===\nCommand:' "$game" "$environment_id"
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then
        # Never retry: a lost response might already have queued the job.
        "${command[@]}"
    fi
    job_count=$((job_count + 1))
done <<'GAMES'
1|asterix|ALE/Asterix-v5
2|bank-heist|ALE/BankHeist-v5
3|battle-zone|ALE/BattleZone-v5
4|alien|ALE/Alien-v5
5|amidar|ALE/Amidar-v5
6|assault|ALE/Assault-v5
7|boxing|ALE/Boxing-v5
8|breakout|ALE/Breakout-v5
9|chopper-command|ALE/ChopperCommand-v5
10|crazy-climber|ALE/CrazyClimber-v5
11|demon-attack|ALE/DemonAttack-v5
12|freeway|ALE/Freeway-v5
13|frostbite|ALE/Frostbite-v5
14|gopher|ALE/Gopher-v5
15|hero|ALE/Hero-v5
16|jamesbond|ALE/Jamesbond-v5
17|kangaroo|ALE/Kangaroo-v5
18|krull|ALE/Krull-v5
19|kung-fu-master|ALE/KungFuMaster-v5
20|ms-pacman|ALE/MsPacman-v5
21|pong|ALE/Pong-v5
22|private-eye|ALE/PrivateEye-v5
23|qbert|ALE/Qbert-v5
24|road-runner|ALE/RoadRunner-v5
25|seaquest|ALE/Seaquest-v5
26|up-n-down|ALE/UpNDown-v5
GAMES

if [[ "$DRY_RUN" == 1 ]]; then
    printf '\nDry run complete: %s jobs; nothing submitted or written.\n' "$job_count"
else
    printf '\nSubmitted %s training job(s) to jobd; worker output root: %s\n' "$job_count" "$RUN_ROOT"
    jobd -l
fi
