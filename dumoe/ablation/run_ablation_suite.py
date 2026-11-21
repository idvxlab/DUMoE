#!/usr/bin/env python3
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = BASE_DIR / "time_slice_train.py"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

SEEDS = [42, 2023, 3407]
VARIANTS = [
    ("Full", "none"),
    ("A1_NoAdapter", "no_adapter"),
    ("A2_NoGating", "no_gating"),
    ("A3_OnlyUser", "only_user"),
    ("A4_OnlyShort", "only_short"),
    ("A5_OnlyLong", "only_long"),
]

METRIC_KEYS = [
    "top1_acc",
    "top3_acc",
    "top5_acc",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "recall_macro",
]

ARTIFACTS = [
    "time_slice_metrics.json",
    "time_slice_predictions.json",
    "time_slice_model.pth",
    "time_slice_label_map.json",
]


def _extract_test_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    for item in payload:
        if isinstance(item, dict) and item.get("stage") == "test":
            m = item.get("test_metrics", {})
            return m if isinstance(m, dict) else {}
    return {}


def _build_cmd(ablation: str, seed: int):
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--ablation", ablation,
        "--seed", str(seed),
        "--batch-size", "32",
        "--stage1-epochs", "10",
        "--stage2-epochs", "4",
        "--stage3-epochs", "15",
        "--lr-stage1", "1e-4",
        "--lr-stage2", "5e-4",
        "--lr-stage3", "1e-3",
        "--balanced-sampler",
        "--early-stop-patience", "3",
        "--early-stop-min-delta", "1e-4",
        "--use-cached-intersection",
        "--use-cached-needed-msg-ids",
        "--use-cached-labels",
        "--label-parse-progress-every", "0",
    ]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for variant_name, ablation in VARIANTS:
        for seed in SEEDS:
            run_dir = RESULTS_DIR / variant_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            cmd = _build_cmd(ablation, seed)
            print(f"[Run] variant={variant_name} ablation={ablation} seed={seed}")
            print("[Run] " + " ".join(cmd))

            started = time.time()
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.time() - started

            for name in ARTIFACTS:
                src = BASE_DIR / name
                if src.exists():
                    shutil.copy2(src, run_dir / name)

            metrics = _extract_test_metrics(run_dir / "time_slice_metrics.json")
            row = {
                "variant": variant_name,
                "ablation": ablation,
                "seed": seed,
                "status": "ok" if proc.returncode == 0 else "failed",
                "return_code": proc.returncode,
                "elapsed_seconds": f"{elapsed:.1f}",
            }
            for k in METRIC_KEYS:
                v = metrics.get(k)
                row[k] = "" if v is None else v
            rows.append(row)

            print(f"[Done] status={row['status']} rc={proc.returncode} elapsed={elapsed:.1f}s")

    records_path = OUTPUT_DIR / "run_records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "ablation",
                "seed",
                "status",
                "return_code",
                "elapsed_seconds",
                *METRIC_KEYS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Summary] Wrote run records to {records_path}")


if __name__ == "__main__":
    main()
