#!/usr/bin/env bash
# One training process / W&B run: V2 PER through 100k collection updates,
# then V1 PER + mixed threshold 20k for 10k learner-only updates.
# Replay and optimizer stay in memory; no pre-final snapshot is written.
# V1 beta follows the global .4 -> 1 schedule over 110k updates, not a restart.
#
# Usage: ./scripts/run_per_recipe.sh
# Overrides: RUN_ID, RUN_ROOT, SEED, ENVIRONMENT_ID, WANDB_ENABLED,
#            WANDB_PROJECT, WANDB_ENTITY, DRY_RUN=1.
# Existing output roots are rejected to protect prior experiments.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/per_recipe/$RUN_ID}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-ALE/Asterix-v5}"
SEED="${SEED:-2}"
DRY_RUN="${DRY_RUN:-0}"

command=(
    uv run python scripts/train_agent.py
    search=puct self_play.num_simulations=50
    "seed=$SEED" "environment.id=$ENVIRONMENT_ID"
    self_play.total_transitions=100000 training.steps=100000
    training.final_steps=10000
    replay.per_mode=v2 replay.final_per_mode=v1
    replay.priority_alpha=0.6 replay.priority_beta_initial=0.4
    replay.priority_beta_final=1.0
    reanalysis.cache_targets=true reanalysis.target_update_interval=1000
    reanalysis.cache_target_ttl=200
    training.value_target=mixed training.mixed_value_start_step=30000
    training.mixed_value_threshold=5000
    training.final_mixed_value_threshold=20000
    training.preserve_mixed_value_freshness=false
    training.optimizer=sgd training.learning_rate=0.2
    loss.consistency_weight=2.0
    checkpoint.pre_final_snapshot_path=null checkpoint.resume_pre_final_path=null
    "checkpoint.path=$RUN_ROOT/checkpoints/agent_latest.pt"
    checkpoint.collection_interval=5000 checkpoint.final_interval=2500
    evaluation.enabled=true evaluation.evaluate_on_resume=false
    evaluation.episodes=32 evaluation.num_envs=32
    "evaluation.data_path=$RUN_ROOT/evaluations/agent_evaluations.json"
    "evaluation.plot_path=$RUN_ROOT/evaluations/agent_evaluation.png"
    training.progress_mode=always training.progress_interval_seconds=10
    "wandb.enabled=${WANDB_ENABLED:-true}"
    "wandb.project=${WANDB_PROJECT:-AtariAgent}"
    "wandb.entity=${WANDB_ENTITY:-null}"
    "wandb.name=${ENVIRONMENT_ID##*/}_${RUN_ID}_v2-to-v1"
    "wandb.tags=[per-recipe,v2-to-v1]"
)
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run complete; no training or output directories created.\n'
    exit 0
fi
mkdir -p -- "$(dirname -- "$RUN_ROOT")"
if ! mkdir -- "$RUN_ROOT"; then
    printf 'Choose a new RUN_ID/RUN_ROOT; refusing to overwrite: %s\n' "$RUN_ROOT" >&2
    exit 1
fi
mkdir -p -- "$RUN_ROOT/checkpoints" "$RUN_ROOT/evaluations"
printf '%q ' "${command[@]}" > "$RUN_ROOT/command.sh"
printf '\n' >> "$RUN_ROOT/command.sh"
"${command[@]}" 2>&1 | tee "$RUN_ROOT/training.log"
printf '\nRecipe complete. Compare checkpoints at 100k, 102.5k, 105k, 107.5k and 110k.\nResults: %s\n' "$RUN_ROOT"
