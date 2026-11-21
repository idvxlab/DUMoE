#!/usr/bin/env bash
# Paper main model (Ours / Full): Stage1 backbone → Stage2 adapters → Stage3 gate.
# Both heads are trained together each step: L = cls_weight * L_cls + rec_weight * L_rec.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.example.env"

cd "${ROOT}/dumoe"
python pipeline_multitask_vs_baselines.py \
  --training-strategy staged \
  --stage1-epochs 15 \
  --stage2-epochs 5 \
  --stage3-epochs 5 \
  --lr-stage1 1e-4 \
  --lr-stage2 5e-4 \
  --lr-stage3 1e-3 \
  --batch-size 128 \
  --rec-weight 1.5 \
  --cls-weight 1.0 \
  --ablation none \
  --seed 42 \
  --user-profile-pt "${USER_PROFILE_PT}" \
  --tweet-emb-shards-dir "${TWEET_EMB_SHARDS_DIR}" \
  --samples-dir "${SAMPLES_DIR}" \
  --user-tweet-ids-map "${USER_TWEET_IDS_MAP}" \
  --labels-cache "${LABELS_CACHE}" \
  --use-cached-labels \
  --output-root "${OUTPUT_ROOT}/dumoe_staged" \
  --exp-name ours_staged \
  --device "${DEVICE:-cuda:0}"
