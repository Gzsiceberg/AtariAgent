#!/usr/bin/env bash
# Run V2 PER during the cached-target final phase from the 100k pre-final
# snapshot. Existing runs already cover the V1 cached-target baseline.
#
# V2 reproduces old AtariAgent PER: alpha=beta=1 and a normalized importance-
# sampling weight floor of 0.1. This tests whether V2 PER reproduces the severe
# collapse in the current code.
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
    [[ -f "$evaluation_path" ]] && \
        uv run python scripts/summarize_game_experiments.py is-complete \
            "$evaluation_path" \
            --expected-final-update "$EXPECTED_FINAL_UPDATE"
}

run_ablation() {
    local name="$1"
    local per_mode="$2"
    local cache_targets="$3"
    local target_update_interval=1000
    local mixed_value_threshold=5000
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
        "replay.per_mode=$per_mode"
        "reanalysis.cache_targets=$cache_targets"
        "reanalysis.target_update_interval=$target_update_interval"
        "reanalysis.mcts_bootstrap_final_phase=false"
        "training.value_target=mixed"
        "training.mixed_value_threshold=$mixed_value_threshold"
        "training.preserve_mixed_value_freshness=false"
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
        "wandb.enabled=true"
        "wandb.project=$WANDB_PROJECT"
        "$wandb_entity_override"
        "wandb.name=${GAME_NAME}_${name}"
        "wandb.tags=[final-phase-ablation,$name]"
    )

    printf '\n=== %s ===\n' "$name"
    printf 'per_mode=%s cache_targets=%s target_update_interval=%s ' \
        "$per_mode" "$cache_targets" "$target_update_interval"
    printf 'mixed_value_threshold=%s mcts_bootstrap_final_phase=false\n' \
        "$mixed_value_threshold"
    printf 'output=%s\n' "$output_dir"
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}" 2>&1 | tee "$output_dir/training.log"
}

run_ablation "v2-per-cached" v2 true

printf '\nV2 cached-target experiment completed. Results: %s\n' "$RUN_ROOT"
