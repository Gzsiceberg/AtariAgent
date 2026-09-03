#!/usr/bin/env bash
# Run a baseline and target-update-interval final-phase ablation from the
# same 100k pre-final snapshot.
#
# The ablation changes only target publication frequency:
#   cache_targets=true, target_update_interval=200
# Baseline keeps the normal final-phase settings:
#   cache_targets=true, target_update_interval=1000
#
# Each invocation creates a new W&B run. The resumed 100k checkpoint is
# evaluated before training, followed by evaluations every 5k updates at
# 105k, 110k, 115k, and 120k, so every run contains the same baseline.
#
# Usage:
#   ./scripts/run_final_phase_ablations.sh
#
# Optional overrides:
#   RUN_ID=my-ablation ./scripts/run_final_phase_ablations.sh
#   SNAPSHOT_PATH=/path/to/agent_pre_final.pt ./scripts/run_final_phase_ablations.sh
#   WANDB_PROJECT=AtariAgent WANDB_ENTITY=my-entity ./scripts/run_final_phase_ablations.sh
#   RUN_ID=existing-run SKIP_COMPLETED=1 ./scripts/run_final_phase_ablations.sh
#   DRY_RUN=1 ./scripts/run_final_phase_ablations.sh

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/final_phase_ablations/$RUN_ID}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-checkpoints/Asterix-v5/agent_pre_final.pt}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-ALE/Asterix-v5}"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_FINAL_UPDATE=120000

if [[ "$DRY_RUN" != "1" && ! -f "$SNAPSHOT_PATH" ]]; then
    printf 'Pre-final snapshot not found: %s\n' "$SNAPSHOT_PATH" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"

is_completed() {
    local evaluation_path="$1"
    uv run python scripts/summarize_game_experiments.py is-complete \
        "$evaluation_path" \
        --expected-final-update "$EXPECTED_FINAL_UPDATE"
}

run_ablation() {
    local name="$1"
    local cache_targets="$2"
    local target_update_interval="$3"
    local output_dir="$RUN_ROOT/$name"
    local evaluation_path="$output_dir/evaluations/agent_evaluations.json"
    local wandb_entity_override="wandb.entity=null"

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_completed "$evaluation_path"; then
        printf '\n=== Skipping completed ablation: %s ===\n' "$name"
        return
    fi

    if [[ -n "$WANDB_ENTITY" ]]; then
        wandb_entity_override="wandb.entity=$WANDB_ENTITY"
    fi

    mkdir -p "$output_dir/checkpoints" "$output_dir/evaluations"

    local -a command=(
        uv run python scripts/train_agent.py
        "environment.id=$ENVIRONMENT_ID"
        "checkpoint.resume_pre_final_path=$SNAPSHOT_PATH"
        "checkpoint.pre_final_snapshot_path=null"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        "evaluation.enabled=true"
        "evaluation.evaluate_on_resume=true"
        "evaluation.data_path=$evaluation_path"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        "reanalysis.cache_targets=$cache_targets"
        "reanalysis.target_update_interval=$target_update_interval"
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
        "wandb.enabled=true"
        "wandb.project=$WANDB_PROJECT"
        "$wandb_entity_override"
        "wandb.tags=[final-phase-ablation,$name]"
    )

    printf '\n=== %s ===\n' "$name"
    printf 'cache_targets=%s target_update_interval=%s\n' \
        "$cache_targets" "$target_update_interval"
    printf 'output=%s\n' "$output_dir"
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}" 2>&1 | tee "$output_dir/training.log"
}

# Independent one-factor ablation: only target update frequency differs.
run_ablation "target-interval-200" true 200

# Run the unchanged control last to measure normal resumed behavior.
run_ablation "baseline" true 1000

printf '\nAll final-phase ablations completed. Results: %s\n' "$RUN_ROOT"
