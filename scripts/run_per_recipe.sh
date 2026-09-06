#!/usr/bin/env bash
# Run two independent ablations: disable cache target TTL; set PER alpha to 1.0.
# Previous experiments are commented out.
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
        "training.progress_mode=always"
        "training.progress_interval_seconds=10"
        wandb.enabled=true
        "wandb.name='\${environment_slug:\${environment.id}}_${name}_seed\${seed}_${RUN_ID}'"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        "checkpoint.pre_final_snapshot_path=$output_dir/checkpoints/agent_pre_final.pt"
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

# run_experiment target_update_200_no_ttl \
#     reanalysis.target_update_interval=200 reanalysis.cache_target_ttl=0
# run_experiment baseline
# run_experiment gumbel search=gumbel

# run_experiment puct_v2_t200_no_ttl_mv5000 \
#     search=puct replay.per_mode=v2 \
#     reanalysis.target_update_interval=200 reanalysis.cache_target_ttl=0 \
#     training.mixed_value_threshold=5000

run_experiment no_cache_target_ttl \
    reanalysis.cache_target_ttl=0

run_experiment priority_alpha_1 \
    replay.priority_alpha=1.0

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete; no training or output directories created.\n'
else
    printf '\nExperiments complete. Results: %s\n' "$RUN_ROOT"
fi
