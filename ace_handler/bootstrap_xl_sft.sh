#!/usr/bin/env bash
# Bootstrap ACE XL-SFT correctly into checkpoints/, upgrade ACE-Step code, run handler.
set -euo pipefail

export ACESTEP_CONFIG_PATH="acestep-v15-xl-sft"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export ACE_FORCE_CONFIG="acestep-v15-xl-sft"

REPO="${ACESTEP_REPO:-/app/acestep-repo}"
CKPT_DIR="${REPO}/checkpoints"
TARGET="${CKPT_DIR}/acestep-v15-xl-sft"
MARKER="${CKPT_DIR}/.ace_xl_sft_ready_v2"

mkdir -p "$CKPT_DIR" "$HF_HOME"
echo "[bootstrap] repo=$REPO target=$TARGET"

# Keep ACE-Step code recent enough for SFT (DCW default fix)
if [[ -d "$REPO/.git" ]]; then
  echo "[bootstrap] Updating ACE-Step repo to latest main..."
  cd "$REPO"
  git remote set-url origin https://github.com/ace-step/ACE-Step-1.5.git || true
  git fetch --depth 1 origin main || true
  git checkout -f FETCH_HEAD || git checkout -f main || true
  pip3 install --no-cache-dir --break-system-packages --no-deps -e "$REPO" || true
fi

if [[ ! -f "$MARKER" || ! -d "$TARGET" ]]; then
  echo "[bootstrap] Downloading ACE-Step/acestep-v15-xl-sft into checkpoints/ ..."
  rm -rf "$TARGET"
  export TARGET
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import os
target = os.environ["TARGET"]
os.makedirs(os.path.dirname(target), exist_ok=True)
path = snapshot_download(
    "ACE-Step/acestep-v15-xl-sft",
    local_dir=target,
)
print("[bootstrap] downloaded to", path)
PY
  if [[ ! -d "$TARGET" ]]; then
    echo "[bootstrap] TARGET missing after download" >&2
    exit 3
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
else
  echo "[bootstrap] XL-SFT checkpoint already present"
fi

HANDLER_URL="${ACE_HANDLER_URL:-https://raw.githubusercontent.com/leidige/yue2-runpod-worker/main/ace_handler/handler.py}"
echo "[bootstrap] Fetching handler: $HANDLER_URL"
curl -fsSL "$HANDLER_URL" -o /app/handler.py
grep -q 'leidige-src_audio_fix' /app/handler.py

echo "[bootstrap] Starting XL-SFT handler (config=$ACESTEP_CONFIG_PATH)"
exec env \
  ACESTEP_CONFIG_PATH=acestep-v15-xl-sft \
  ACE_FORCE_CONFIG=acestep-v15-xl-sft \
  python3 -u /app/handler.py
