#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.example.env"

cd "${ROOT}/baselines"
python project/run_unified_pipeline.py \
  --models all \
  --samples-csv "${SAMPLES_CSV}" \
  --labels-cache "${LABELS_CACHE}" \
  --user-profile-pt "${USER_PROFILE_PT}" \
  --tweet-emb-shards-dir "${TWEET_EMB_SHARDS_DIR}" \
  --seq-len 300 \
  --window-size 30 \
  --epochs 25 \
  --batch-size 128 \
  --output-dir "${OUTPUT_ROOT}/baselines" \
  --preprocess-cache-dir "${PREPROCESS_CACHE_DIR}" \
  --device "${DEVICE:-cuda:0}"
