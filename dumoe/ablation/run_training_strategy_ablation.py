#!/usr/bin/env python3
"""
：staged（） vs joint（）， full model（--ablation none）。

 Full ：
  -  epoch  pipeline_multitask_vs_baselines.py / README ：Stage1=15, Stage2=5, Stage3=5。
  - Joint  epoch  JOINT_EPOCH_MODE （）， staged 。

：time_slice_train.py  Stage2  ** adapter  stage2_epochs **，
 staged 「adapter 」 num_classes * stage2_epochs， stage2_epochs。
"""
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

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 5
STAGE3_EPOCHS = 5
LR_STAGE1 = 1e-4
LR_STAGE2 = 5e-4
LR_STAGE3 = 1e-3

LR_JOINT = 3e-4

LABEL_PARSE_PROGRESS_EVERY = 500

NUM_CLASSES = 15

JOINT_EPOCH_MODE = "macro"


def _joint_epochs() -> int:
    if JOINT_EPOCH_MODE == "strict":
        return STAGE1_EPOCHS + NUM_CLASSES * STAGE2_EPOCHS + STAGE3_EPOCHS
    if JOINT_EPOCH_MODE == "macro":
        return STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS
    raise ValueError(f"Unknown JOINT_EPOCH_MODE={JOINT_EPOCH_MODE!r}, use 'macro' or 'strict'")


SEEDS = [42, 2023, 3407]
VARIANTS = [
    ("S0_StagedRef", "staged"),
    ("S1_JointAllModules", "joint"),
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


def _build_cmd(strategy: str, seed: int, joint_epochs: int):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--ablation",
        "none",
        "--training-strategy",
        strategy,
        "--seed",
        str(seed),
        "--batch-size",
        "32",
        "--stage1-epochs",
        str(STAGE1_EPOCHS),
        "--stage2-epochs",
        str(STAGE2_EPOCHS),
        "--stage3-epochs",
        str(STAGE3_EPOCHS),
        "--lr-stage1",
        str(LR_STAGE1),
        "--lr-stage2",
        str(LR_STAGE2),
        "--lr-stage3",
        str(LR_STAGE3),
        "--balanced-sampler",
        "--early-stop-patience",
        "3",
        "--early-stop-min-delta",
        "1e-4",
        "--use-cached-intersection",
        "--use-cached-needed-msg-ids",
        "--use-cached-labels",
        "--label-parse-progress-every",
        str(LABEL_PARSE_PROGRESS_EVERY),
    ]
    if strategy == "joint":
        cmd.extend(["--joint-epochs", str(joint_epochs), "--lr-joint", str(LR_JOINT)])
    return cmd


def main():
    joint_epochs = _joint_epochs()
    print(
        f"[Config] staged stages: {STAGE1_EPOCHS}+{STAGE2_EPOCHS}+{STAGE3_EPOCHS} "
        f"(stage2  adapter  {STAGE2_EPOCHS} epoch， {NUM_CLASSES}×{STAGE2_EPOCHS})"
    )
    print(f"[Config] JOINT_EPOCH_MODE={JOINT_EPOCH_MODE} -> joint_epochs={joint_epochs} lr_joint={LR_JOINT}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for variant_name, strategy in VARIANTS:
        for seed in SEEDS:
            run_dir = RESULTS_DIR / variant_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            cmd = _build_cmd(strategy, seed, joint_epochs)
            print(f"[Run] variant={variant_name} strategy={strategy} seed={seed}")
            print("[Run] " + " ".join(cmd))
            print(
                "[Run]  stdout/stderr （）。\n"
                f"      : {log_path}\n"
                f"      : tail -f {log_path}",
                flush=True,
            )

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
                "training_strategy": strategy,
                "ablation": "none",
                "seed": seed,
                "joint_epoch_mode": JOINT_EPOCH_MODE,
                "joint_epochs": joint_epochs,
                "status": "ok" if proc.returncode == 0 else "failed",
                "return_code": proc.returncode,
                "elapsed_seconds": f"{elapsed:.1f}",
            }
            for k in METRIC_KEYS:
                v = metrics.get(k)
                row[k] = "" if v is None else v
            rows.append(row)

            print(f"[Done] status={row['status']} rc={proc.returncode} elapsed={elapsed:.1f}s")

    records_path = OUTPUT_DIR / "training_strategy_run_records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "training_strategy",
                "ablation",
                "seed",
                "joint_epoch_mode",
                "joint_epochs",
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
