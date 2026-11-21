#!/usr/bin/env bash
# Ablation: w/o stage-wise training (backbone + adapters + gate jointly).
# Paper main model is staged — see run_dumoe_staged.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.example.env"

cd "${ROOT}/dumoe"
python pipeline_multitask_vs_baselines.py \
  --training-strategy joint \
  --joint-epochs 25 \
  --lr-joint 3e-4 \
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
  --output-root "${OUTPUT_ROOT}/dumoe_joint" \
  --exp-name abl_joint_all_modules \
  --device "${DEVICE:-cuda:0}"
