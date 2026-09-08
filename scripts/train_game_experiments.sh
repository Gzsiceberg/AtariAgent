#!/usr/bin/env bash
# Schedule the 26 Atari 100k games with task spooler (tsp).
# Uses configs/train_agent.yaml defaults, enables W&B, and disables the large
# pre-final snapshot. Each game runs once; failures do not block other games.
# This script exits after enqueueing. No supervision or power management.
# Run only one batch at a time; wait for its summary before rerunning.
#
# Usage:
#   uv sync --extra wandb --frozen
#   uv run wandb login
#   tsp -S 1  # one training job at a time
#   ./scripts/train_game_experiments.sh
#
# Optional env vars:
#   SEED:           training seed (default 2)
#   RUN_ID:         run label (default seed_$SEED)
#   RUN_ROOT:       output root (default runs/game_experiments/$RUN_ID)
#   WANDB_PROJECT:  W&B project (default AtariAgent)
#   WANDB_ENTITY:   optional W&B entity
#   DRY_RUN:        set to 1 to generate scripts without scheduling
#
# Requires tsp -S 1 so the summary runs after all games.
# Monitor with tsp -l or tsp -t JOB_ID.
# Job IDs (including the final summary job) are recorded in $RUN_ROOT/tsp_jobs.tsv.
# The summary job updates $RUN_ROOT/results.csv, comparing available evaluations
# with $RUN_ROOT/paper_scores.csv (random, human, EfficientZero V1/V2 scores).
# Rerun the same command after the batch finishes to skip completed games
# and schedule the rest.
# Completion means the final evaluation reached update 120000. Incomplete games
# restart from scratch. Each RUN_ROOT belongs to one seed; use a different
# RUN_ID/RUN_ROOT for another seed or a fresh repeat.
# Examples:
#   SEED=3 ./scripts/train_game_experiments.sh
#   RUN_ID=my-old-run SEED=2 ./scripts/train_game_experiments.sh

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v tsp >/dev/null 2>&1; then
    printf "Error: task spooler 'tsp' not found.\n" >&2
    exit 1
fi

SEED="${SEED:-2}"
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || (( SEED > 4294967295 )); then
    printf 'SEED must be an integer between 0 and 4294967295.\n' >&2
    exit 1
fi
RUN_ID="${RUN_ID:-seed_$SEED}"
RUN_ROOT="${RUN_ROOT:-runs/game_experiments/$RUN_ID}"
EXPERIMENT_NAME="default"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DRY_RUN="${DRY_RUN:-0}"

EXPECTED_FINAL_UPDATE=120000
SCHEDULED=0
SKIPPED=0

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd -- "$RUN_ROOT" && pwd)"
if [[ -f "$RUN_ROOT/seed.txt" ]]; then
    read -r saved_seed <"$RUN_ROOT/seed.txt"
elif [[ -f "$RUN_ROOT/games.tsv" ]]; then
    # Runs created before SEED was configurable used train_agent.yaml's seed 2.
    saved_seed=2
else
    saved_seed="$SEED"
fi
if [[ "$saved_seed" != "$SEED" ]]; then
    printf 'Run root belongs to seed %s, not %s; choose another RUN_ID/RUN_ROOT.\n' \
        "$saved_seed" "$SEED" >&2
    exit 1
fi
printf '%s\n' "$SEED" >"$RUN_ROOT/seed.txt"

# Preserve reference data for optional comparisons using
# scripts/summarize_game_experiments.py. These are not stopping criteria, and
# single-seed results are not like-for-like reproductions of multi-seed papers.
if [[ ! -f "$RUN_ROOT/paper_scores.csv" ]]; then
    cp "$REPO_ROOT/scripts/atari_100k_paper_scores.csv" "$RUN_ROOT/paper_scores.csv"
fi
GAME_MANIFEST="$RUN_ROOT/games.tsv"
cat >"$GAME_MANIFEST" <<'GAMES'
asterix|ALE/Asterix-v5
bank-heist|ALE/BankHeist-v5
battle-zone|ALE/BattleZone-v5
alien|ALE/Alien-v5
amidar|ALE/Amidar-v5
assault|ALE/Assault-v5
boxing|ALE/Boxing-v5
breakout|ALE/Breakout-v5
chopper-command|ALE/ChopperCommand-v5
crazy-climber|ALE/CrazyClimber-v5
demon-attack|ALE/DemonAttack-v5
freeway|ALE/Freeway-v5
frostbite|ALE/Frostbite-v5
gopher|ALE/Gopher-v5
hero|ALE/Hero-v5
jamesbond|ALE/Jamesbond-v5
kangaroo|ALE/Kangaroo-v5
krull|ALE/Krull-v5
kung-fu-master|ALE/KungFuMaster-v5
ms-pacman|ALE/MsPacman-v5
pong|ALE/Pong-v5
private-eye|ALE/PrivateEye-v5
qbert|ALE/Qbert-v5
road-runner|ALE/RoadRunner-v5
seaquest|ALE/Seaquest-v5
up-n-down|ALE/UpNDown-v5
GAMES

schedule() {
    local game="$1" environment_id="$2"
    local output_dir="$RUN_ROOT/$game/$EXPERIMENT_NAME"
    local job_script="$output_dir/job.sh"
    local log_path="$output_dir/training.log"
    local label="$RUN_ID/seed_$SEED/$game" job_id
    local evaluation_path="$output_dir/evaluations/agent_evaluations.json"

    if [[ -f "$evaluation_path" ]] && uv run python \
        scripts/summarize_game_experiments.py is-complete "$evaluation_path" \
        --expected-final-update "$EXPECTED_FINAL_UPDATE"; then
        printf 'Skipping completed game: %s (seed %s)\n' "$game" "$SEED"
        ((SKIPPED += 1))
        return
    fi

    mkdir -p "$output_dir/checkpoints" "$output_dir/evaluations"
    local -a args=(
        uv run python scripts/train_agent.py
        "seed=$SEED"
        "environment.id=$environment_id"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        checkpoint.pre_final_snapshot_path=null
        "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        training.progress_mode=always
        training.progress_interval_seconds=10
        wandb.enabled=true
        "wandb.project=$WANDB_PROJECT"
        "wandb.name='\${environment_slug:\${environment.id}}_${EXPERIMENT_NAME}_seed\${seed}_${RUN_ID}'"
        "wandb.tags=[atari-100k,default,all-games,$game]"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
        args+=("wandb.entity=$WANDB_ENTITY")
    fi

    {
        printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
        printf 'cd %q\n' "$REPO_ROOT"
        printf '%q ' "${args[@]}"
        printf '2>&1 | tee %q\n' "$log_path"
    } >"$job_script"
    chmod +x "$job_script"

    printf '\n=== Scheduling %s (%s) ===\n' "$game" "$environment_id"
    printf 'Output dir: %s\n' "$output_dir"
    printf 'tsp label: %s\n' "$label"
    printf 'Command: '
    printf '%q ' "${args[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    job_id="$(tsp -L "$label" bash "$job_script")"
    ((SCHEDULED += 1))
    printf '%s\t%s\n' "$job_id" "$game" >>"$RUN_ROOT/tsp_jobs.tsv"
    printf 'Queued as tsp job %s\n' "$job_id"
}

schedule_summary() {
    local job_id
    local -a args=(
        uv run python scripts/summarize_game_experiments.py summarize "$RUN_ROOT"
        --experiment-name "$EXPERIMENT_NAME"
        --expected-final-update "$EXPECTED_FINAL_UPDATE"
        --paper-scores "$RUN_ROOT/paper_scores.csv"
    )
    printf '\nSummary command: '
    printf '%q ' "${args[@]}"
    printf '\n'
    [[ "$DRY_RUN" != "1" ]] || return 0

    job_id="$(tsp -L "$RUN_ID/seed_$SEED/summary" "${args[@]}")"
    printf '%s\tsummary\n' "$job_id" >>"$RUN_ROOT/tsp_jobs.tsv"
    printf 'Queued summary as tsp job %s: %s/results.csv\n' "$job_id" "$RUN_ROOT"
}

touch "$RUN_ROOT/tsp_jobs.tsv"
while IFS='|' read -r game environment_id; do
    [[ -n "$game" ]] || continue
    schedule "$game" "$environment_id"
done <"$GAME_MANIFEST"
schedule_summary

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete. Commands were not scheduled.\n'
else
    printf '\nScheduled %s game jobs plus summary; skipped %s games under %s\n' "$SCHEDULED" "$SKIPPED" "$RUN_ROOT"
    tsp -l
fi
