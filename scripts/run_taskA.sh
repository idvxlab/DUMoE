#!/usr/bin/env bash
# Task A: frozen window interest classification.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.example.env"
cd "${ROOT}"
python downstream/taskA/run_classification.py \
  --samples-csv "${SAMPLES_CSV}" \
  --seeds 42,52,62 \
  --probe-label-smoothing 0.1 \
  --device "${DEVICE:-cuda:0}" \
  "$@"
