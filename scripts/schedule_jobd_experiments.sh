#!/usr/bin/env bash
# Submit Gumbel target-400, mixed-value-10000, cache-TTL-50 experiments.
# Both use the game selected by ENVIRONMENT_ID:
# 0 / gumbel_t400_mv10000_ttl_50_v1max: V1 maximum-priority insertion
# 1 / gumbel_t400_mv10000_ttl_50_clip: default insertion, importance-weight floor 0.1
# No local files or credentials are sent to workers.
#
# Usage: ./scripts/schedule_jobd_experiments.sh [all | 0 | 1 | experiment_name]
# Example: DRY_RUN=1 ENVIRONMENT_ID=Qbert-v5 ./scripts/schedule_jobd_experiments.sh all
#
# Submission environment:
#   DRY_RUN=1: print commands without jobd, credentials, or filesystem changes
# Worker paths (not paths on the submitting machine):
#   WORKER_REPO_ROOT: checkout on every worker (default /workspace/AtariAgent)
#   RUN_ROOT: absolute or relative to checkout (default runs/jobd_sweep/$RUN_ID)
#   SNAPSHOT_PATH: optional pre-final snapshot, accessible on every worker
# Training options:
#   ENVIRONMENT_ID: default ALE/UpNDown-v5 (ALE/ added if omitted)
#   RUN_ID: default timestamp; SEED: default 2; TRAINING_STEPS: default 100000
#   WANDB_PROJECT: default AtariAgent; WANDB_ENTITY: optional
#
# Prepare each worker with uv sync --extra wandb and W&B authentication, then
# start its controller worker with jobd --restart from the desired environment.
# Existing workers do not inherit new credentials from the submitting shell.
# No workers are restarted or hosts shut down by this script. Results stay on
# the executing worker. Shared output storage requires distinct RUN_ID values.

set -Eeuo pipefail

usage() {
    echo "Usage: ${0##*/} [all | 0/gumbel_t400_mv10000_ttl_50_v1max | 1/gumbel_t400_mv10000_ttl_50_clip]"
}

declare -a experiments=()
if [[ $# -eq 0 ]]; then set -- all; fi
for selection in "$@"; do
    case "$selection" in
        -h|--help) usage; exit 0 ;;
        all)
            if [[ $# -ne 1 ]]; then
                echo "Error: 'all' must be used alone." >&2
                exit 1
            fi
            experiments=(gumbel_t400_mv10000_ttl_50_v1max gumbel_t400_mv10000_ttl_50_clip)
            ;;
        0|gumbel_t400_mv10000_ttl_50_v1max) experiments+=(gumbel_t400_mv10000_ttl_50_v1max) ;;
        1|gumbel_t400_mv10000_ttl_50_clip) experiments+=(gumbel_t400_mv10000_ttl_50_clip) ;;
        *) printf 'Error: unknown experiment: %s\n' "$selection" >&2; usage >&2; exit 1 ;;
    esac
done

DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != 1 ]]; then
    if ! command -v jobd >/dev/null 2>&1; then
        echo "Error: 'jobd' not found." >&2
        exit 1
    fi
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/jobd_sweep/$RUN_ID}"
WORKER_REPO_ROOT="${WORKER_REPO_ROOT:-/workspace/AtariAgent}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-ALE/UpNDown-v5}"
ENVIRONMENT_ID="ALE/${ENVIRONMENT_ID#ALE/}"
SEED="${SEED:-2}"
TRAINING_STEPS="${TRAINING_STEPS:-100000}"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-}"

# Send a self-contained command, not a path to a script on the submitter.
# Positional arguments avoid shell interpolation of paths and Hydra overrides.
worker_command='set -Eeuo pipefail
repo=$1
output=$2
shift 2
cd -- "$repo"
# Refuse to overwrite a previous run on this worker.
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
mkdir -p -- "$output/checkpoints" "$output/evaluations"
uv run python scripts/train_agent.py "$@" 2>&1 | tee "$output/training.log"'

declare -A scheduled=()
job_count=0
for experiment in "${experiments[@]}"; do
    [[ -z "${scheduled[$experiment]:-}" ]] || continue
    experiment_environment="$ENVIRONMENT_ID"
    overrides=(
        "search=gumbel"
        "checkpoint.collection_interval=10000"
        "training.mixed_value_threshold=10000"
        "reanalysis.target_update_interval=400"
        "reanalysis.cache_targets=true"
        "reanalysis.cache_target_ttl=50"
    )
    case "$experiment" in
        gumbel_t400_mv10000_ttl_50_v1max)
            overrides+=("replay.use_max_priority=true")
            ;;
        gumbel_t400_mv10000_ttl_50_clip)
            overrides+=("replay.use_max_priority=false" "replay.priority_weight_clip=0.1")
            ;;
    esac
    output_dir="$RUN_ROOT/$experiment"
    args=(
        "seed=$SEED"
        "environment.id=$experiment_environment"
        "training.steps=$TRAINING_STEPS"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        "checkpoint.pre_final_snapshot_path=null"
        "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        "evaluation.enabled=true"
        "wandb.enabled=true"
        "wandb.project=$WANDB_PROJECT"
        "wandb.name=${experiment_environment##*/}_${experiment}_seed${SEED}_${RUN_ID}"
        "wandb.tags=[jobd,${experiment}]"
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
    )
    [[ -z "$WANDB_ENTITY" ]] || args+=("wandb.entity=$WANDB_ENTITY")
    [[ -z "$SNAPSHOT_PATH" ]] || args+=("checkpoint.resume_pre_final_path=$SNAPSHOT_PATH")
    args+=("${overrides[@]}")
    command=(jobd bash -c "$worker_command" "jobd-$experiment" "$WORKER_REPO_ROOT" "$output_dir" "${args[@]}")
    printf '\n=== Scheduling %s ===\nCommand:' "$experiment"
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then
        # Never retry automatically: a lost response might already have queued it.
        "${command[@]}"
    fi
    scheduled[$experiment]=1
    job_count=$((job_count + 1))
done

if [[ "$DRY_RUN" == 1 ]]; then
    printf '\nDry run complete: %s jobs; nothing submitted or written.\n' "$job_count"
else
    printf '\nSubmitted %s training job(s) to jobd; worker output root: %s\n' "$job_count" "$RUN_ROOT"
    jobd -l
fi
