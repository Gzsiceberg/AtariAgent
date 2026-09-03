#!/usr/bin/env bash
# Run three independent final-phase ablations from the same 100k pre-final
# snapshot. All arms use the current V1-style PER and target interval 1000.
#
# Relative to the normal final-phase settings, the arms independently:
#   1. increase mixed_value_threshold from 5000 to 20000;
#   2. preserve EfficientZero V2 replay freshness during learner-only updates;
#   3. disable reanalysis target caching.
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
GAME_NAME="${ENVIRONMENT_ID##*/}"
GAME_NAME="${GAME_NAME%-v5}"
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
    local mixed_value_threshold="$3"
    local preserve_mixed_value_freshness="$4"
    local target_update_interval=1000
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
        "training.mixed_value_threshold=$mixed_value_threshold"
        "training.preserve_mixed_value_freshness=$preserve_mixed_value_freshness"
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
        "wandb.enabled=true"
        "wandb.project=$WANDB_PROJECT"
        "$wandb_entity_override"
        "wandb.name=${GAME_NAME}_${name}"
        "wandb.tags=[final-phase-ablation,$name]"
    )

    printf '\n=== %s ===\n' "$name"
    printf 'cache_targets=%s target_update_interval=%s ' \
        "$cache_targets" "$target_update_interval"
    printf 'mixed_value_threshold=%s preserve_mixed_value_freshness=%s\n' \
        "$mixed_value_threshold" "$preserve_mixed_value_freshness"
    printf 'output=%s\n' "$output_dir"
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}" 2>&1 | tee "$output_dir/training.log"
}

# Keep replay samples on TD values longer, while retaining current final-phase
# age advancement.
run_ablation "mixed-threshold-20000" true 20000 false

# Keep the original 5000-transition threshold, but do not increase transition
# ages merely because learner-only updates elapsed.
run_ablation "preserve-v2-freshness" true 5000 true

# Re-run no-cache with the current V1-style PER; this is the clean cache
# comparison that the old-PER no-cache run could not provide.
run_ablation "no-cache-current-per" false 5000 false

printf '\nAll final-phase ablations completed. Results: %s\n' "$RUN_ROOT"
