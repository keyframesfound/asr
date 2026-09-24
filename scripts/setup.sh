#!/usr/bin/env bash
# venv → pip install → download Whisper, SenseVoice, and Parakeet into models/.
# Safe to re-run: weights that are already present are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
python -m pip install -r requirements.txt

# ./run may have exported offline hub flags in this shell. The download
# script also clears them; unset here so pip/hub see the network too.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

python "$ROOT/scripts/download_models.py"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "iFlytek is cloud-only. For that engine, copy .env.example to .env and fill the keys."
fi
