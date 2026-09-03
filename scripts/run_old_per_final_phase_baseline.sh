#!/usr/bin/env bash
# Re-run the cached old-PER final-phase baseline using the historical code,
# then restore the branch/ref from which this script was started.
#
# The default commit uses EfficientZero V2-style PER: priorities are sampled
# directly, full importance correction is applied, and normalized importance
# weights are floored at 0.1.
#
# Usage:
#   ./scripts/run_old_per_final_phase_baseline.sh
#
# Optional overrides:
#   COMMIT_HASH=<commit> ./scripts/run_old_per_final_phase_baseline.sh
#   RUN_ID=my-run ./scripts/run_old_per_final_phase_baseline.sh
#   SNAPSHOT_PATH=/path/to/agent_pre_final.pt ./scripts/run_old_per_final_phase_baseline.sh
#   WANDB_PROJECT=AtariAgent WANDB_ENTITY=my-entity ./scripts/run_old_per_final_phase_baseline.sh
#   DRY_RUN=1 ./scripts/run_old_per_final_phase_baseline.sh

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMMIT_HASH="${COMMIT_HASH:-13249a2cc531e1e917a0938dfd0805e936af0d7b}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/old_per_final_phase/$RUN_ID}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-checkpoints/Asterix-v5/agent_pre_final.pt}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-ALE/Asterix-v5}"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DRY_RUN="${DRY_RUN:-0}"

ORIGINAL_BRANCH="$(git branch --show-current)"
if [[ -n "$ORIGINAL_BRANCH" ]]; then
    RESTORE_REF="$ORIGINAL_BRANCH"
else
    RESTORE_REF="$(git rev-parse HEAD)"
fi
CHECKED_OUT_HISTORICAL_COMMIT=0

restore_checkout() {
    local exit_status=$?
    trap - EXIT INT TERM

    if [[ "$CHECKED_OUT_HISTORICAL_COMMIT" == "1" ]]; then
        printf '\n=== Restoring checkout: %s ===\n' "$RESTORE_REF"
        if ! git checkout "$RESTORE_REF"; then
            printf 'Failed to restore checkout to %s. Restore it manually.\n' \
                "$RESTORE_REF" >&2
            exit 1
        fi
    fi

    exit "$exit_status"
}
trap restore_checkout EXIT INT TERM

if [[ -n "$(git status --porcelain=v1 --untracked-files=no)" ]]; then
    printf 'Tracked files have uncommitted changes; refusing to switch commits.\n' >&2
    printf 'Commit or stash those changes before running this script.\n' >&2
    exit 1
fi

if ! git rev-parse --verify "${COMMIT_HASH}^{commit}" >/dev/null 2>&1; then
    printf 'Commit not found: %s\n' "$COMMIT_HASH" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" && ! -f "$SNAPSHOT_PATH" ]]; then
    printf 'Pre-final snapshot not found: %s\n' "$SNAPSHOT_PATH" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/evaluations"

printf 'Starting checkout: %s\n' "$RESTORE_REF"
printf 'Historical old-PER commit: %s\n' "$COMMIT_HASH"
printf 'Output: %s\n' "$RUN_ROOT"

# Use checkout explicitly so the historical source, config, and lockfile all
# match the original old-PER implementation.
git checkout "$COMMIT_HASH"
CHECKED_OUT_HISTORICAL_COMMIT=1

# Verify that checkout resolved to the requested commit before launching.
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "${COMMIT_HASH}^{commit}")" ]]; then
    printf 'Historical commit checkout verification failed.\n' >&2
    exit 1
fi

wandb_entity_override="wandb.entity=null"
if [[ -n "$WANDB_ENTITY" ]]; then
    wandb_entity_override="wandb.entity=$WANDB_ENTITY"
fi

command=(
    uv run python scripts/train_agent.py
    "environment.id=$ENVIRONMENT_ID"
    "checkpoint.resume_pre_final_path=$SNAPSHOT_PATH"
    "checkpoint.pre_final_snapshot_path=null"
    "checkpoint.path=$RUN_ROOT/checkpoints/agent_latest.pt"
    "checkpoint.keep_representative=24"
    "evaluation.enabled=true"
    "evaluation.evaluate_on_resume=true"
    "evaluation.data_path=$RUN_ROOT/evaluations/agent_evaluations.json"
    "evaluation.plot_path=$RUN_ROOT/evaluations/agent_evaluation.png"
    "reanalysis.cache_targets=true"
    "reanalysis.target_update_interval=1000"
    "training.value_target=mixed"
    "training.mixed_value_threshold=5000"
    "wandb.enabled=true"
    "wandb.project=$WANDB_PROJECT"
    "$wandb_entity_override"
    "wandb.tags=[old-per-final-phase,baseline]"
)

printf '\n=== Old-PER cached baseline ===\n'
printf 'cache_targets=true target_update_interval=1000 mixed_value_threshold=5000\n'
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$DRY_RUN" != "1" ]]; then
    "${command[@]}" 2>&1 | tee "$RUN_ROOT/training.log"
fi

printf '\nOld-PER baseline completed. Results: %s\n' "$RUN_ROOT"
