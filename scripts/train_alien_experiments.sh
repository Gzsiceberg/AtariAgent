#!/usr/bin/env bash
# Run one Alien training experiment, then suspend the host.

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
RUN_ROOT="runs/alien_experiments/$RUN_ID"
mkdir -p "$RUN_ROOT"

run_training() {
    local name="$1"
    shift
    local output_dir="$RUN_ROOT/$name"
    mkdir -p "$output_dir"

    {
        printf '\n=== Starting %s ===\n' "$name"
        printf 'Output directory: %s\n' "$output_dir"

        uv run python scripts/train_agent.py \
            environment.id=ALE/Alien-v5 \
            "checkpoint.path=$output_dir/checkpoints/agent_latest.pt" \
            "evaluation.data_path=$output_dir/evaluations/agent_evaluations.json" \
            "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png" \
            "wandb.tags=[automation,alien,$name]" \
            "$@"

        printf '=== Finished %s ===\n' "$name"
    } 2>&1 | tee "$output_dir/training.log"
}

run_training reanalysis-cache-clear-200 \
    training.reanalysis_cache_clear_interval=200

printf '\nTraining run completed successfully. Suspending the host.\n'
sync

if ! kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then
    printf 'Cannot suspend: sudo authorization keepalive stopped unexpectedly.\n' >&2
    exit 1
fi

sudo -n systemctl suspend
