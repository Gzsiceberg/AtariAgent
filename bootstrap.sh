#!/usr/bin/env bash
# Fresh workers only. Prerequisites: uv, procps, OpenCV system libraries,
# and an exported JOBD_WORKER_TOKEN. Ansible provisions these prerequisites.
set -Eeuo pipefail
trap 'status=$?; printf "Bootstrap failed at line %s (exit %s).\n" "$LINENO" "$status" >&2; exit "$status"' ERR
umask 077

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"
: "${JOBD_WORKER_TOKEN:?Set JOBD_WORKER_TOKEN before running bootstrap}"
command -v pgrep >/dev/null
if pgrep -x jobd-worker >/dev/null; then
    echo 'Error: a jobd worker is already running; refusing to interrupt it.' >&2
    exit 1
fi

echo '[1/5] Syncing dependencies'
uv sync --frozen --extra wandb

echo '[2/5] Checking OpenCV and CUDA readiness'
uv run --no-sync python - <<'PY'
import cv2
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.ones(1, device="cuda").sum().item() != 1:
    raise RuntimeError("CUDA computation failed")
print(f"OpenCV {cv2.__version__}; CUDA ready: {torch.cuda.get_device_name(0)}")
PY

echo '[3/5] Installing jobd'
curl -fsSL https://github.com/Gzsiceberg/jobd/releases/latest/download/install.sh | sh

echo '[4/5] Verifying worker token'
jobd auth verify-worker-token

echo '[5/5] Starting worker'
# start never interrupts an already-running worker, unlike restart.
jobd worker start

echo 'Bootstrap complete'
