#!/usr/bin/env bash
# Schedule independent Bank Heist experiments with task spooler (tsp).
#
# 1 / value_loss_coeff:    loss.value_weight=0.5
# 2 / consistency_weight:  loss.consistency_weight=5
# 3 / priority_alpha:      replay.priority_alpha=0.6
# Each experiment uses 10000 final learner-only updates.
#
# All experiments:
# - enable W&B logging
# - keep final snapshot path enabled
# - train on ALE/BankHeist-v5 using default settings otherwise
# - run full collection before the final phase unless SNAPSHOT_PATH is provided
#
# Usage:
#   ./scripts/schedule_tsp_experiments.sh           # all three
#   ./scripts/schedule_tsp_experiments.sh all       # all three
#   ./scripts/schedule_tsp_experiments.sh 1         # value loss only
#   ./scripts/schedule_tsp_experiments.sh consistency_weight
#   ./scripts/schedule_tsp_experiments.sh 1 3       # selected experiments
#
# Optional env vars:
#   RUN_ID:          stable label for wandb/run directories (default timestamp)
#   RUN_ROOT:        experiment root directory
#   SEED:            random seed (default 2)
#   SNAPSHOT_PATH:   optional checkpoint to resume from with `checkpoint.resume_pre_final_path`
#   TRAINING_STEPS:  number of collection updates (default 100000)
#   WANDB_PROJECT:   wandb project (default AtariAgent)
#   WANDB_ENTITY:    wandb entity (default not set)
#   WANDB_API_KEY:   W&B API key (prompted with hidden input if unset/empty)
#   RUNPOD_POD_ID:  if nonempty, queue pod shutdown after training
#   DRY_RUN:         set to 1 to only print planned commands
#
# Notes:
# - `tsp` must be installed and running.
# - If SNAPSHOT_PATH is not set, each job runs full collection + final phase.
# - `checkpoint.pre_final_snapshot_path` is still explicitly set per job.

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    printf 'Usage: %s [all | 1/value_loss_coeff | 2/consistency_weight | 3/priority_alpha ...]\n' "${0##*/}"
}

# Validate the complete selection before prompting or creating any jobs.
declare -a experiments=()
if [[ $# -eq 0 ]]; then
    set -- all
fi
for selection in "$@"; do
    case "$selection" in
        -h|--help) usage; exit 0 ;;
        all)
            if [[ $# -ne 1 ]]; then
                echo "Error: 'all' must be used alone." >&2
                usage >&2
                exit 1
            fi
            experiments=(value_loss_coeff consistency_weight priority_alpha)
            ;;
        1|value_loss_coeff) experiments+=(value_loss_coeff) ;;
        2|consistency_weight) experiments+=(consistency_weight) ;;
        3|priority_alpha) experiments+=(priority_alpha) ;;
        *)
            printf 'Error: unknown experiment: %s\n' "$selection" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v tsp >/dev/null 2>&1; then
    echo "Error: task spooler 'tsp' not found." >&2
    exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/tsp_sweep/$RUN_ID}"
ENVIRONMENT_ID="ALE/BankHeist-v5"
SEED="${SEED:-2}"
TRAINING_STEPS="${TRAINING_STEPS:-100000}"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

# Pass credentials only through the environment, not generated job scripts.
while [[ -z "${WANDB_API_KEY:-}" ]]; do
    printf 'W&B API key: ' >&2
    if ! IFS= read -r -s WANDB_API_KEY; then
        printf '\nError: unable to read WANDB_API_KEY; export it before running non-interactively.\n' >&2
        exit 1
    fi
    printf '\n' >&2
    if [[ -z "$WANDB_API_KEY" ]]; then
        printf 'API key must not be empty.\n' >&2
    fi
done
export WANDB_API_KEY

mkdir -p "$(dirname -- "$RUN_ROOT")"
if [[ "$DRY_RUN" != "1" ]]; then
    if ! mkdir -- "$RUN_ROOT"; then
        echo "Run root already exists: $RUN_ROOT" >&2
        exit 1
    fi
fi

# Build the experiment's common args.
build_common_args() {
    local output_dir=$1
    local experiment_name=$2

    local -a args=(
        "seed=$SEED"
        "environment.id=$ENVIRONMENT_ID"
        "training.steps=$TRAINING_STEPS"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        "checkpoint.pre_final_snapshot_path=$output_dir/checkpoints/agent_pre_final.pt"
        "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        "evaluation.enabled=true"
        "wandb.enabled=true"
        "wandb.project=$WANDB_PROJECT"
        "wandb.name=${ENVIRONMENT_ID##*/}_${experiment_name}_seed${SEED}_${RUN_ID}"
        "wandb.tags=[tsp,${experiment_name}]"
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
    )

    if [[ -n "$WANDB_ENTITY" ]]; then
        args+=("wandb.entity=$WANDB_ENTITY")
    fi
    if [[ -n "$SNAPSHOT_PATH" ]]; then
        args+=("checkpoint.resume_pre_final_path=$SNAPSHOT_PATH")
    fi

    printf '%s\n' "${args[@]}"
}

schedule() {
    local name=$1
    local -a overrides=("${@:2}")

    local output_dir="$RUN_ROOT/$name"
    local log_path="$output_dir/training.log"
    local job_script="$output_dir/job.sh"

    mkdir -p "$output_dir/checkpoints" "$output_dir/evaluations"

    local -a args=(
        "uv" "run" "python" "scripts/train_agent.py"
    )

    while IFS= read -r arg; do
        args+=("$arg")
    done < <(build_common_args "$output_dir" "$name")

    for arg in "${overrides[@]}"; do
        args+=("$arg")
    done

    {
        echo '#!/usr/bin/env bash'
        echo 'set -Eeuo pipefail'
        printf 'cd %q\n' "$REPO_ROOT"
        printf '(%q' "${args[0]}"
        for ((i = 1; i < ${#args[@]}; i++)); do
            printf ' %q' "${args[$i]}"
        done
        printf ') | tee %q\n' "$log_path"
    } > "$job_script"
    chmod +x "$job_script"

    printf "\n=== Scheduling %s ===\n" "$name"
    printf 'Output dir: %s\n' "$output_dir"
    printf 'tsp label: %s\n' "$name"

    # Print the exact command for auditing.
    printf 'Command:'
    printf ' %q' "${args[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    tsp -L "$name" bash "$job_script"
}

declare -A scheduled=()
job_count=0
for experiment in "${experiments[@]}"; do
    # Repeated selectors must not schedule duplicate jobs.
    if [[ -n "${scheduled[$experiment]:-}" ]]; then
        continue
    fi
    case "$experiment" in
        value_loss_coeff) override="loss.value_weight=0.5" ;;
        consistency_weight) override="loss.consistency_weight=5" ;;
        priority_alpha) override="replay.priority_alpha=0.6" ;;
    esac
    schedule "$experiment" "training.final_steps=10000" "$override"
    scheduled[$experiment]=1
    job_count=$((job_count + 1))
done

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    printf '\nCommand: tsp runpodctl stop pod %q\n' "$RUNPOD_POD_ID"
    if [[ "$DRY_RUN" != "1" ]]; then
        tsp runpodctl stop pod "$RUNPOD_POD_ID"
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. Commands were not scheduled.\n'
else
    printf '\nScheduled %s training job(s) via task-spooler under %s\n' "$job_count" "$RUN_ROOT"
    tsp -l
fi
