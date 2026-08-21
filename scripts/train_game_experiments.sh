#!/usr/bin/env bash
# Train MsPacman and Breakout sequentially, then suspend the host.

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Authenticate now rather than risking a password prompt after hours of training.
printf 'Authorizing automatic suspend (sudo may prompt for your password)...\n'
sudo -v

SUDO_KEEPALIVE_PID=""
cleanup() {
    if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Refresh the sudo timestamp while training so the final suspend is non-interactive.
(
    while sleep 60; do
        sudo -n -v || exit 1
    done
) &
SUDO_KEEPALIVE_PID=$!

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="runs/game_experiments/$RUN_ID"
mkdir -p "$RUN_ROOT"

run_training() {
    local game="$1"
    local environment_id="$2"
    local name="$3"
    shift 3
    local output_dir="$RUN_ROOT/$game/$name"
    mkdir -p "$output_dir"

    {
        printf '\n=== Starting %s: %s ===\n' "$game" "$name"
        printf 'Output directory: %s\n' "$output_dir"

        uv run python scripts/train_agent.py \
            "environment.id=$environment_id" \
            "checkpoint.path=$output_dir/checkpoints/agent_latest.pt" \
            "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json" \
            "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png" \
            "wandb.tags=[automation,$game,$name]" \
            "$@"

        printf '=== Finished %s: %s ===\n' "$game" "$name"
    } 2>&1 | tee "$output_dir/training.log"
}

run_training mspacman ALE/MsPacman-v5 reanalysis-cache-clear-200 \
    training.reanalysis_cache_clear_interval=200

run_training breakout ALE/Breakout-v5 reanalysis-cache-clear-200 \
    training.reanalysis_cache_clear_interval=200

printf '\nAll training runs completed successfully. Suspending the host.\n'
sync

if ! kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then
    printf 'Cannot suspend: sudo authorization keepalive stopped unexpectedly.\n' >&2
    exit 1
fi

sudo -n systemctl suspend
