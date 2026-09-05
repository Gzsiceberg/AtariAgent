#!/usr/bin/env bash
# Run 200-step target updates with TTL disabled, the default baseline, and
# Gumbel search sequentially.
# All other training parameters come from the Hydra defaults.
#
# Usage: ./scripts/run_per_recipe.sh
# Overrides: RUN_ID, RUN_ROOT, DRY_RUN=1.
# Existing output roots are rejected to protect prior experiments.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/per_recipe/$RUN_ID}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p -- "$(dirname -- "$RUN_ROOT")"
    if ! mkdir -- "$RUN_ROOT"; then
        printf 'Choose a new RUN_ID/RUN_ROOT; refusing to overwrite: %s\n' "$RUN_ROOT" >&2
        exit 1
    fi
fi

run_experiment() {
    local name="$1"
    shift
    local output_dir="$RUN_ROOT/$name"
    local -a command=(
        uv run python scripts/train_agent.py
        "$@"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        checkpoint.pre_final_snapshot_path=null
        "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
    )
    printf '\n%s command:' "$name"
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    mkdir -p -- "$output_dir/checkpoints" "$output_dir/evaluations"
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    "${command[@]}" 2>&1 | tee "$output_dir/training.log"
}

run_experiment target_update_200_no_ttl \
    reanalysis.target_update_interval=200 reanalysis.cache_target_ttl=0
run_experiment baseline
run_experiment gumbel search=gumbel

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete; no training or output directories created.\n'
else
    printf '\nExperiments complete. Results: %s\n' "$RUN_ROOT"
fi
