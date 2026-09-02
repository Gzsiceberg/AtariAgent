#!/usr/bin/env bash
# Run a baseline and two independent final-phase ablations from the same
# 100k pre-final snapshot.
#
# Baseline keeps the normal final-phase settings:
#   cache_targets=true, target_update_interval=1000
# Test 1 changes only reanalysis caching:
#   cache_targets=false, target_update_interval=1000
# Test 2 changes only target publication frequency:
#   cache_targets=true, target_update_interval=200
#
# Each invocation creates a new W&B run. The resumed 100k checkpoint is
# evaluated before training, followed by the scheduled 110k and 120k
# evaluations, so each run contains its own common baseline.
#
# Usage:
#   ./scripts/run_final_phase_ablations.sh
#
# Optional overrides:
#   RUN_ID=my-ablation ./scripts/run_final_phase_ablations.sh
#   SNAPSHOT_PATH=/path/to/agent_pre_final.pt ./scripts/run_final_phase_ablations.sh
#   WANDB_PROJECT=AtariAgent WANDB_ENTITY=my-entity ./scripts/run_final_phase_ablations.sh
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
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" && ! -f "$SNAPSHOT_PATH" ]]; then
    printf 'Pre-final snapshot not found: %s\n' "$SNAPSHOT_PATH" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"

run_ablation() {
    local name="$1"
    local cache_targets="$2"
    local target_update_interval="$3"
    local output_dir="$RUN_ROOT/$name"
    local wandb_entity_override="wandb.entity=null"

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
        "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        "reanalysis.cache_targets=$cache_targets"
        "reanalysis.target_update_interval=$target_update_interval"
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

# Run the unchanged control first to measure normal resumed behavior.
run_ablation "baseline" true 1000

# Independent one-factor ablation: only caching differs from the baseline.
run_ablation "no-cache" false 1000

# Independent one-factor ablation: only target update frequency differs.
run_ablation "target-interval-200" true 200

printf '\nAll final-phase ablations completed. Results: %s\n' "$RUN_ROOT"
