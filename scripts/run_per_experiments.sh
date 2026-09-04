#!/usr/bin/env bash
# Run the three prioritized-replay experiments described in TODO.md.
#
# The baseline uses the EfficientZero V1 schedule (alpha=0.6 and beta linearly
# annealed from 0.4 to 1.0). The two ablations hold beta at 1.0 and vary alpha.
# Runs execute sequentially and write to separate directories.
#
# Usage:
#   ./scripts/run_per_experiments.sh
#
# Optional overrides:
#   ENVIRONMENT_ID=ALE/Alien-v5 ./scripts/run_per_experiments.sh
#   RUN_ID=my-run SKIP_COMPLETED=1 ./scripts/run_per_experiments.sh
#   WANDB_ENABLED=true WANDB_PROJECT=AtariAgent ./scripts/run_per_experiments.sh
#   DRY_RUN=1 ./scripts/run_per_experiments.sh

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/per_experiments/$RUN_ID}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-ALE/Asterix-v5}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_FINAL_UPDATE="${EXPECTED_FINAL_UPDATE:-120000}"

mkdir -p "$RUN_ROOT"

is_completed() {
    local evaluation_path="$1"
    [[ -f "$evaluation_path" ]] && \
        uv run python scripts/summarize_game_experiments.py is-complete \
            "$evaluation_path" \
            --expected-final-update "$EXPECTED_FINAL_UPDATE"
}

run_experiment() {
    local name="$1"
    local alpha="$2"
    local beta_initial="$3"
    local beta_final="$4"
    local output_dir="$RUN_ROOT/$name"
    local evaluation_path="$output_dir/evaluations/agent_evaluations.json"

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_completed "$evaluation_path"; then
        printf '\n=== Skipping completed experiment: %s ===\n' "$name"
        return
    fi

    mkdir -p "$output_dir/checkpoints" "$output_dir/evaluations"

    local -a command=(
        uv run python scripts/train_agent.py
        "environment.id=$ENVIRONMENT_ID"
        replay.per_mode=v1
        "replay.priority_alpha=$alpha"
        "replay.priority_beta_initial=$beta_initial"
        "replay.priority_beta_final=$beta_final"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        "checkpoint.pre_final_snapshot_path=$output_dir/checkpoints/agent_pre_final.pt"
        "evaluation.data_path=$evaluation_path"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        training.progress_mode=always
        training.progress_interval_seconds=10
        "wandb.enabled=$WANDB_ENABLED"
        "wandb.project=$WANDB_PROJECT"
        "wandb.name=${ENVIRONMENT_ID##*/}_${name}"
        "wandb.tags=[per-ablation,$name]"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
        command+=("wandb.entity=$WANDB_ENTITY")
    fi

    printf '\n=== Starting experiment: %s ===\n' "$name"
    printf 'PER mode=v1 alpha=%s beta=%s -> %s\n' \
        "$alpha" "$beta_initial" "$beta_final"
    printf 'Output directory: %s\n' "$output_dir"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}" 2>&1 | tee "$output_dir/training.log"
}

printf 'Run ID: %s\n' "$RUN_ID"
printf 'Output root: %s\n' "$RUN_ROOT"
printf 'Environment: %s\n' "$ENVIRONMENT_ID"

run_experiment baseline 0.6 0.4 1.0
run_experiment alpha-1-beta-1-v1 1.0 1.0 1.0

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete; no training was started.\n'
else
    printf '\nAll PER experiments completed. Results: %s\n' "$RUN_ROOT"
fi
