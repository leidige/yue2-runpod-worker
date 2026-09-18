#!/usr/bin/env bash
# Bootstrap ACE XL-SFT on first cold start, then run patched serverless handler.
set -euo pipefail

export ACESTEP_CONFIG_PATH="acestep-v15-xl-sft"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export ACE_FORCE_CONFIG="acestep-v15-xl-sft"

MARKER="${HF_HOME}/.ace_xl_sft_ready"
mkdir -p "$HF_HOME"

echo "[bootstrap] FORCED config=$ACESTEP_CONFIG_PATH"

if [[ ! -f "$MARKER" ]]; then
  echo "[bootstrap] Downloading ACE-Step XL-SFT DiT (first boot)..."
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import os, sys
cache = os.environ.get("HF_HOME", "/root/.cache/huggingface")
try:
    path = snapshot_download(
        "ACE-Step/Ace-Step1.5",
        allow_patterns=["acestep-v15-xl-sft/*"],
        cache_dir=cache,
    )
    print("[bootstrap] XL-SFT download complete:", path)
except Exception as e:
    print("[bootstrap] DOWNLOAD FAILED:", e, file=sys.stderr)
    raise
PY
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"
else
  echo "[bootstrap] XL-SFT marker present"
fi

if [[ "$ACESTEP_CONFIG_PATH" == *turbo* ]]; then
  echo "[bootstrap] REFUSING turbo config" >&2
  exit 2
fi

HANDLER_URL="${ACE_HANDLER_URL:-https://raw.githubusercontent.com/leidige/yue2-runpod-worker/ce7ec28/ace_handler/handler.py}"
echo "[bootstrap] Fetching handler: $HANDLER_URL"
curl -fsSL "$HANDLER_URL" -o /app/handler.py
grep -q 'leidige-src_audio_fix' /app/handler.py

echo "[bootstrap] Starting XL-SFT handler..."
exec env ACESTEP_CONFIG_PATH=acestep-v15-xl-sft python3 -u /app/handler.py
