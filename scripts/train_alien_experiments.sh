#!/usr/bin/env bash
# Run three Alien training experiments sequentially, then suspend the host.

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

# The configured target update interval is 800 unless explicitly overridden.
run_training target-update-400 \
    training.target_update_interval=400

run_training optimizer-adam \
    training.optimizer=adam \
    training.learning_rate=0.001

run_training search-puct-simulations-50 \
    self_play.search_algorithm=puct \
    self_play.num_simulations=50

printf '\nAll three training runs completed successfully. Suspending the host.\n'
sync

if ! kill -0 "$SUDO_KEEPALIVE_PID" 2>/dev/null; then
    printf 'Cannot suspend: sudo authorization keepalive stopped unexpectedly.\n' >&2
    exit 1
fi

sudo -n systemctl suspend
