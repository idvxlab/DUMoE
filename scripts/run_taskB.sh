#!/usr/bin/env bash
# Task B: frozen window-aligned interaction.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.example.env"
cd "${ROOT}"
python downstream/taskB/run_interaction.py \
  --samples-csv "${SAMPLES_CSV}" \
  --tweet-emb-shards-dir "${TWEET_EMB_SHARDS_DIR}" \
  --seeds 42,52,62 \
  --tweet-reduce-dim 256 \
  --max-pos-per-window-record 10 \
  --max-neg-per-window-record 10 \
  --device "${DEVICE:-cuda:0}" \
  "$@"
