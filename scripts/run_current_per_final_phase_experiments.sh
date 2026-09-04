#!/usr/bin/env bash
# Run the two selected current-PER final-phase experiments from the same 100k
# Asterix pre-final snapshot:
#   1. baseline mixed-value threshold (5k)
#   2. larger mixed-value threshold (20k)
#
# Both arms use cached reanalysis targets, target interval 1000, and the
# current V1-style PER settings (alpha 0.6, beta 0.4 -> 1.0).
#
# Usage:
#   ./scripts/run_current_per_final_phase_experiments.sh
#
# Optional overrides:
#   RUN_ID=my-run ./scripts/run_current_per_final_phase_experiments.sh
#   SNAPSHOT_PATH=/path/to/agent_pre_final.pt ./scripts/run_current_per_final_phase_experiments.sh
#   WANDB_PROJECT=AtariAgent WANDB_ENTITY=my-entity ./scripts/run_current_per_final_phase_experiments.sh
#   RUN_ID=existing-run SKIP_COMPLETED=1 ./scripts/run_current_per_final_phase_experiments.sh
#   DRY_RUN=1 ./scripts/run_current_per_final_phase_experiments.sh

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/current_per_final_phase/$RUN_ID}"
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
    [[ -f "$evaluation_path" ]] && uv run python scripts/summarize_game_experiments.py is-complete \
        "$evaluation_path" \
        --expected-final-update "$EXPECTED_FINAL_UPDATE"
}

run_experiment() {
    local name="$1"
    local mixed_value_threshold="$2"
    local output_dir="$RUN_ROOT/$name"
    local evaluation_path="$output_dir/evaluations/agent_evaluations.json"
    local wandb_entity_override="wandb.entity=null"

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_completed "$evaluation_path"; then
        printf '\n=== Skipping completed experiment: %s ===\n' "$name"
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
        "replay.per_mode=v1"
        "replay.priority_alpha=0.6"
        "replay.priority_beta_initial=0.4"
        "replay.priority_beta_final=1.0"
        "replay.priority_epsilon=0.000001"
        "reanalysis.cache_targets=true"
        "reanalysis.target_update_interval=1000"
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
        "wandb.tags=[current-per-final-phase,$name]"
    )

    printf '\n=== %s ===\n' "$name"
    printf 'PER alpha=0.6 beta=0.4->1.0 mixed_value_threshold=%s\n' \
        "$mixed_value_threshold"
    printf 'cache_targets=true target_update_interval=1000 output=%s\n' \
        "$output_dir"
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    "${command[@]}" 2>&1 | tee "$output_dir/training.log"
}

run_experiment "current-per-baseline" 5000
run_experiment "current-per-threshold-20000" 20000

printf '\nAll current-PER final-phase experiments completed. Results: %s\n' "$RUN_ROOT"
