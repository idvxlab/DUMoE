#!/usr/bin/env bash
# Frozen-repr probes: Task A (classification) then Task B (interaction).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.example.env"
cd "${ROOT}"

echo "[Task A] window interest classification"
python downstream/taskA/run_classification.py \
  --samples-csv "${SAMPLES_CSV}" \
  --seeds 42,52,62 \
  --probe-label-smoothing 0.1 \
  --device "${DEVICE:-cuda:0}"

echo "[Task B] window-aligned interaction"
python downstream/taskB/run_interaction.py \
  --samples-csv "${SAMPLES_CSV}" \
  --tweet-emb-shards-dir "${TWEET_EMB_SHARDS_DIR}" \
  --seeds 42,52,62 \
  --tweet-reduce-dim 256 \
  --max-pos-per-window-record 10 \
  --max-neg-per-window-record 10 \
  --device "${DEVICE:-cuda:0}"
