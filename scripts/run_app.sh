#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_app.sh [checkpoint-folder]
# Example: ./scripts/run_app.sh experiments/finetune_20261116_153000/checkpoints
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -d .venv ]]; then source .venv/bin/activate; else echo "Missing .venv. Create it and install requirements first." >&2; exit 1; fi
export IMAGE_MODE="${IMAGE_MODE:-real}"
export IMAGE_TILE="${IMAGE_TILE:-256}"
if [[ $# -ge 1 && -n "$1" ]]; then export IMAGE_CHECKPOINT_DIR="$1"; fi
if [[ -n "${IMAGE_CHECKPOINT_DIR:-}" ]]; then
  for model in dncnn nafnet restormer; do
    [[ -f "$IMAGE_CHECKPOINT_DIR/$model.pth" ]] || { echo "Missing $IMAGE_CHECKPOINT_DIR/$model.pth" >&2; exit 1; }
  done
fi
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
