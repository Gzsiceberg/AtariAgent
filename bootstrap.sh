#!/usr/bin/env bash
# Run on a fresh instance with uv installed and JOBD_WORKER_TOKEN exported:
#   nohup ./bootstrap.sh > bootstrap.log 2>&1 < /dev/null &
set -Eeuo pipefail
trap 'status=$?; printf "Bootstrap failed at line %s (exit %s).\n" "$LINENO" "$status" >&2; exit "$status"' ERR

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"

echo '[1/4] Syncing dependencies'
uv sync --frozen --extra wandb

echo '[2/4] Installing jobd'
curl -fsSL https://github.com/Gzsiceberg/jobd/releases/latest/download/install.sh | sh

echo '[3/4] Checking token and restarting worker'
if [[ -z "${JOBD_WORKER_TOKEN:-}" ]]; then
    echo 'Error: set JOBD_WORKER_TOKEN before running bootstrap.' >&2
    false
fi
jobd worker restart

echo '[4/4] Verifying worker token'
jobd auth verify-worker-token

echo 'Bootstrap complete'
