#!/usr/bin/env bash
# Train the 26 Atari 100k games sequentially using configs/train_agent.yaml.
# Keep model-only checkpoints, but disable the large pre-final snapshot.
#
# The batch blocks automatic sleep while it is running and, by default,
# suspends the host whether the batch succeeds, fails, or is interrupted.
# A privileged suspend helper is created at launch, so sudo authorization does
# not need to remain cached for the entire multi-day run.
#
# Usage:
#   uv run wandb login
#   ./scripts/train_game_experiments.sh
#
# Useful overrides:
#   RUN_ID=my-run ./scripts/train_game_experiments.sh
#   RUN_ID=my-run SKIP_COMPLETED=1 ./scripts/train_game_experiments.sh  # resume
#   SUSPEND_WHEN_DONE=0 ./scripts/train_game_experiments.sh
#   MAX_GAME_ATTEMPTS=1 ./scripts/train_game_experiments.sh
#   DRY_RUN=1 SUSPEND_WHEN_DONE=0 ./scripts/train_game_experiments.sh

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="runs/game_experiments/$RUN_ID"
EXPERIMENT_NAME="default"
WANDB_PROJECT="${WANDB_PROJECT:-AtariAgent}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SUSPEND_WHEN_DONE="${SUSPEND_WHEN_DONE:-1}"
MAX_GAME_ATTEMPTS="${MAX_GAME_ATTEMPTS:-2}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"
MIN_INITIAL_DISK_GB="${MIN_INITIAL_DISK_GB:-30}"
MIN_PER_GAME_DISK_GB="${MIN_PER_GAME_DISK_GB:-2}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_FINAL_UPDATE=120000
PAPER_SCORES_SOURCE="$REPO_ROOT/scripts/atari_100k_paper_scores.csv"

mkdir -p "$RUN_ROOT"

# Keep a snapshot of the paper reference data used for this run. These are
# comparison targets, not stopping criteria. AtariAgent's default evaluation
# uses one training seed and 16 episodes, so results are not a like-for-like
# reproduction of the papers' multi-seed evaluations.
PAPER_SCORES="$RUN_ROOT/paper_scores.csv"
cp "$PAPER_SCORES_SOURCE" "$PAPER_SCORES"

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

BATCH_COMPLETED=0
FINALIZING=0
SLEEP_INHIBITOR_PID=""
SLEEP_INHIBITOR_STOP=""
SUSPEND_HELPER_PID=""
SUSPEND_TRIGGER=""
CONTROL_DIR=""

write_status() {
    local state="$1"
    local detail="${2:-}"
    local temporary="$RUN_ROOT/batch_status.txt.tmp"
    {
        printf 'state=%s\n' "$state"
        printf 'detail=%s\n' "$detail"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'pid=%s\n' "$$"
    } >"$temporary"
    mv "$temporary" "$RUN_ROOT/batch_status.txt"
}

available_disk_gb() {
    df -Pk "$RUN_ROOT" | awk 'NR == 2 { print int($4 / 1024 / 1024) }'
}

require_disk_space() {
    local required_gb="$1"
    local available_gb
    available_gb="$(available_disk_gb)"
    if (( available_gb < required_gb )); then
        printf 'Insufficient disk space: %s GiB available, %s GiB required.\n' \
            "$available_gb" "$required_gb" >&2
        return 1
    fi
}

start_sleep_inhibitor() {
    command -v systemd-inhibit >/dev/null || {
        printf 'systemd-inhibit is required for this unattended run.\n' >&2
        return 1
    }

    CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/atariagent-batch.XXXXXX")"
    chmod 700 "$CONTROL_DIR"
    SLEEP_INHIBITOR_STOP="$CONTROL_DIR/stop-inhibitor"
    SUSPEND_TRIGGER="$CONTROL_DIR/suspend"

    # The child exits when explicitly stopped or if this script is killed.
    # systemd releases the inhibitor as soon as the child exits.
    sudo -n nohup systemd-inhibit \
        --what=sleep:idle \
        --who=AtariAgent \
        --why="AtariAgent 26-game training batch $RUN_ID" \
        --mode=block \
        bash -c '
            stop_file="$1"
            parent_pid="$2"
            while [[ ! -e "$stop_file" ]] && kill -0 "$parent_pid" 2>/dev/null; do
                sleep 2
            done
        ' bash "$SLEEP_INHIBITOR_STOP" "$$" \
        </dev/null >"$RUN_ROOT/sleep-inhibitor.log" 2>&1 &
    SLEEP_INHIBITOR_PID=$!
    sleep 1
    kill -0 "$SLEEP_INHIBITOR_PID" 2>/dev/null || {
        printf 'Failed to acquire the systemd sleep inhibitor.\n' >&2
        return 1
    }
}

start_suspend_helper() {
    [[ "$SUSPEND_WHEN_DONE" == "1" ]] || return 0

    # This root-owned helper survives for the whole batch and does not depend
    # on sudo's timestamp. It suspends after the trigger appears or after an
    # abrupt parent death. It waits for the sleep inhibitor to be released.
    sudo -n nohup bash -c '
        parent_pid="$1"
        trigger="$2"
        inhibitor_pid="$3"

        while kill -0 "$parent_pid" 2>/dev/null && [[ ! -e "$trigger" ]]; do
            sleep 2
        done
        while kill -0 "$inhibitor_pid" 2>/dev/null; do
            sleep 1
        done
        sync
        systemctl suspend
    ' bash "$$" "$SUSPEND_TRIGGER" "$SLEEP_INHIBITOR_PID" \
        </dev/null >"$RUN_ROOT/suspend-helper.log" 2>&1 &
    SUSPEND_HELPER_PID=$!
    sleep 1
    kill -0 "$SUSPEND_HELPER_PID" 2>/dev/null || {
        printf 'Failed to start the privileged suspend helper.\n' >&2
        return 1
    }
}

finalize() {
    local exit_code="$1"
    if [[ "$FINALIZING" == "1" ]]; then
        return
    fi
    FINALIZING=1
    trap - EXIT INT TERM
    set +e

    if [[ "$DRY_RUN" != "1" ]]; then
        if [[ "$BATCH_COMPLETED" == "1" ]]; then
            write_status complete "all 26 games completed"
        else
            write_status failed "batch exited with status $exit_code"
        fi

        # Release the blocker before asking the privileged helper to suspend.
        if [[ -n "$SLEEP_INHIBITOR_STOP" ]]; then
            touch "$SLEEP_INHIBITOR_STOP"
        fi
        if [[ -n "$SLEEP_INHIBITOR_PID" ]]; then
            wait "$SLEEP_INHIBITOR_PID" 2>/dev/null
        fi

        if [[ "$SUSPEND_WHEN_DONE" == "1" ]]; then
            printf '\nBatch exited with status %s; suspending the host.\n' "$exit_code"
            if [[ -n "$SUSPEND_HELPER_PID" ]]; then
                touch "$SUSPEND_TRIGGER"
                wait "$SUSPEND_HELPER_PID"
                if [[ "$?" != "0" ]]; then
                    printf 'Automatic suspend helper failed.\n' >&2
                fi
            else
                # Setup failed before the long-lived helper started. The sudo
                # authorization is still fresh, so make one direct attempt.
                sudo -n systemctl suspend || \
                    printf 'Fallback automatic suspend failed.\n' >&2
            fi
        fi
    fi

    if [[ -n "$CONTROL_DIR" ]]; then
        rm -rf "$CONTROL_DIR"
    fi
    exit "$exit_code"
}

trap 'finalize "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

validate_options() {
    for value_name in MAX_GAME_ATTEMPTS RETRY_DELAY_SECONDS MIN_INITIAL_DISK_GB MIN_PER_GAME_DISK_GB; do
        local value="${!value_name}"
        if [[ ! "$value" =~ ^[0-9]+$ ]]; then
            printf '%s must be a non-negative integer, got %q.\n' "$value_name" "$value" >&2
            return 1
        fi
    done
    if (( MAX_GAME_ATTEMPTS < 1 )); then
        printf 'MAX_GAME_ATTEMPTS must be at least 1.\n' >&2
        return 1
    fi
}

validate_options

if [[ "$DRY_RUN" != "1" ]]; then
    require_disk_space "$MIN_INITIAL_DISK_GB"
    command -v nvidia-smi >/dev/null || {
        printf 'nvidia-smi is required for GPU health checks.\n' >&2
        exit 1
    }
    command -v sudo >/dev/null || {
        printf 'sudo is required to block automatic sleep.\n' >&2
        exit 1
    }
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    printf 'Authorizing sleep protection and automatic suspend (sudo may prompt once now)...\n'
    sudo -v
    start_sleep_inhibitor
    start_suspend_helper

    # Ensure the optional logger is installed without changing uv.lock.
    uv sync --extra wandb --frozen
    uv run wandb --version >/dev/null
fi

write_summary() {
    uv run python scripts/summarize_game_experiments.py summarize "$RUN_ROOT" \
        --experiment-name "$EXPERIMENT_NAME" \
        --expected-final-update "$EXPECTED_FINAL_UPDATE" \
        --paper-scores "$PAPER_SCORES"
}

is_completed() {
    local evaluation_path="$1"
    uv run python scripts/summarize_game_experiments.py is-complete \
        "$evaluation_path" \
        --expected-final-update "$EXPECTED_FINAL_UPDATE"
}

check_host_health() {
    require_disk_space "$MIN_PER_GAME_DISK_GB"
    nvidia-smi --query-gpu=name,temperature.gpu,memory.total \
        --format=csv,noheader >/dev/null
}

run_training() {
    local game="$1"
    local environment_id="$2"
    local output_dir="$RUN_ROOT/$game/$EXPERIMENT_NAME"
    local evaluation_path="$output_dir/evaluations/agent_evaluations.json"
    mkdir -p "$output_dir"

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_completed "$evaluation_path"; then
        printf 'Skipping completed game: %s (%s)\n' "$game" "$environment_id"
        write_status running "skipped completed game $game"
        return
    fi

    local -a command=(
        uv run python scripts/train_agent.py
        "environment.id=$environment_id"
        "checkpoint.path=$output_dir/checkpoints/agent_latest.pt"
        checkpoint.pre_final_snapshot_path=null
        "evaluation.data_path=$evaluation_path"
        "evaluation.plot_path=$output_dir/evaluations/agent_evaluation.png"
        training.progress_mode=always
        training.progress_interval_seconds=10
        wandb.enabled=true
        "wandb.project=$WANDB_PROJECT"
        "wandb.name='\${environment_slug:\${environment.id}}_${EXPERIMENT_NAME}_seed\${seed}_${RUN_ID}'"
        "wandb.tags=[atari-100k,default,all-games,$game]"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
        command+=("wandb.entity=$WANDB_ENTITY")
    fi

    printf '\n=== Starting %s (%s) ===\n' "$game" "$environment_id"
    printf 'Output directory: %s\n' "$output_dir"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'Command: '
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi

    local attempt=1
    local status=1
    while (( attempt <= MAX_GAME_ATTEMPTS )); do
        check_host_health
        write_status running "$game attempt $attempt/$MAX_GAME_ATTEMPTS"
        local attempt_log="$output_dir/training-attempt-$(printf '%02d' "$attempt").log"

        set +e
        (
            printf 'Attempt: %s/%s\n' "$attempt" "$MAX_GAME_ATTEMPTS"
            printf 'W&B project: %s\n' "$WANDB_PROJECT"
            "${command[@]}"
            command_status=$?
            if (( command_status == 0 )); then
                printf '=== Finished %s (%s) ===\n' "$game" "$environment_id"
            else
                printf '=== Failed %s (%s), status %s ===\n' \
                    "$game" "$environment_id" "$command_status" >&2
            fi
            exit "$command_status"
        ) 2>&1 | tee "$attempt_log"
        status=${PIPESTATUS[0]}
        set -e
        ln -sfn "$(basename "$attempt_log")" "$output_dir/training.log"

        if (( status == 0 )); then
            write_summary
            return 0
        fi
        if (( attempt >= MAX_GAME_ATTEMPTS )); then
            printf 'Game %s failed after %s attempt(s).\n' \
                "$game" "$MAX_GAME_ATTEMPTS" >&2
            return "$status"
        fi

        printf 'Retrying %s in %s seconds.\n' "$game" "$RETRY_DELAY_SECONDS" >&2
        write_status retrying "$game after status $status"
        sleep "$RETRY_DELAY_SECONDS"
        ((attempt += 1))
    done
}

printf 'Run ID: %s\n' "$RUN_ID"
printf 'Output root: %s\n' "$RUN_ROOT"
printf 'Training config: configs/train_agent.yaml (defaults)\n'
printf 'W&B project: %s\n' "$WANDB_PROJECT"
printf 'Automatic suspend on exit: %s\n' "$SUSPEND_WHEN_DONE"
write_status starting "validating 26-game batch"

while IFS='|' read -r game environment_id; do
    [[ -n "$game" ]] || continue
    run_training "$game" "$environment_id"
done <"$GAME_MANIFEST"

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run complete; no training was started.\n'
    exit 0
fi

write_summary
BATCH_COMPLETED=1
printf '\nAll 26 Atari training runs completed successfully.\n'
printf 'Comparison summary: %s/results.csv\n' "$RUN_ROOT"
sync
