#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = BASE_DIR / "pipeline_multitask_vs_baselines.py"
OUTPUT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_ROOT / "results"
REGISTRY_CSV = OUTPUT_ROOT / "representation_registry.csv"
REGISTRY_JSON = OUTPUT_ROOT / "representation_registry.json"

VARIANTS = [
    {"name": "Full", "ablation": "none", "training_strategy": "staged", "export_mode": "thresholded"},
    {"name": "NoAdapter", "ablation": "no_adapter", "training_strategy": "staged", "export_mode": "base"},
    {"name": "NoGating", "ablation": "no_gating", "training_strategy": "staged", "export_mode": "adapter"},
    {"name": "OnlyUser", "ablation": "only_user", "training_strategy": "staged", "export_mode": "thresholded"},
    {"name": "OnlyShort", "ablation": "only_short", "training_strategy": "staged", "export_mode": "thresholded"},
    {"name": "OnlyLong", "ablation": "only_long", "training_strategy": "staged", "export_mode": "thresholded"},
    {
        "name": "RecOnly",
        "ablation": "none",
        "training_strategy": "staged",
        "export_mode": "thresholded",
        "overrides": {"cls_weight": 0.0},
    },
    {"name": "JointAllModules", "ablation": "none", "training_strategy": "joint", "export_mode": "thresholded"},
]


def _parse_seeds(seeds_str: str) -> List[int]:
    out: List[int] = []
    for chunk in str(seeds_str).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out or [42]


def _parse_variant_list(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(chunk)
    return out


def _build_sweep_rows(args, seed: int, skip_variants: set, only_variants: set) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for variant in VARIANTS:
        if only_variants and variant["name"] not in only_variants:
            continue
        if variant["name"] in skip_variants:
            continue
        row = {
            "name": variant["name"],
            "seed": seed,
            "ablation": variant["ablation"],
            "training_strategy": variant["training_strategy"],
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage3_epochs": args.stage3_epochs,
            "joint_epochs": args.joint_epochs,
            "batch_size": args.batch_size,
            "export_batch_size": args.export_batch_size,
            "lr_stage1": args.lr_stage1,
            "lr_stage2": args.lr_stage2,
            "lr_stage3": args.lr_stage3,
            "lr_joint": args.lr_joint,
            "rec_weight": args.rec_weight,
            "cls_weight": args.cls_weight,
            "rec_neg_reduction": args.rec_neg_reduction,
            "rec_num_neg": args.rec_num_neg,
            "rec_on_stage1": args.rec_on_stage1,
            "rec_on_stage2": args.rec_on_stage2,
            "rec_on_stage3": args.rec_on_stage3,
            "short_recent": args.short_recent,
            "long_recent": args.long_recent,
            "window_size": args.window_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "topk_adapters": args.topk_adapters,
            "num_workers": args.num_workers,
            "amp": int(args.amp),
            "device": args.device,
            "export_mode": variant["export_mode"],
            "export_gate_mode": args.export_gate_mode,
            "export_kl_threshold": args.export_kl_threshold,
            "export_kl_soft_scale": args.export_kl_soft_scale,
            "export_gate_min_alpha": args.export_gate_min_alpha,
            "export_gate_max_alpha": args.export_gate_max_alpha,
            "export_amp": int(args.export_amp),
        }
        row.update(variant.get("overrides", {}))
        rows.append(row)
    return rows


def _build_pipeline_cmd(args, sweep_path: Path, seed_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--sweep-json",
        str(sweep_path),
        "--output-root",
        str(seed_root),
        "--label-user-dirs",
        str(args.label_user_dirs),
        "--samples-dir",
        str(args.samples_dir),
        "--user-profile-pt",
        str(args.user_profile_pt),
        "--tweet-emb-shards-dir",
        str(args.tweet_emb_shards_dir),
        "--train-subsample-ratio",
        str(args.train_subsample_ratio),
        "--device",
        str(args.device),
        "--num-workers",
        str(args.num_workers),
        "--amp",
        "1" if args.amp else "0",
        "--export-amp",
        "1" if args.export_amp else "0",
        "--export-mode",
        str(args.export_mode_fallback),
    ]
    if args.sample_cache_dir:
        cmd.extend(["--sample-cache-dir", str(args.sample_cache_dir)])
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.persistent_workers:
        cmd.append("--persistent-workers")
    if args.use_cached_labels:
        cmd.append("--use-cached-labels")
    if args.labels_cache:
        cmd.extend(["--labels-cache", str(args.labels_cache)])
    if args.label_parse_progress_every is not None:
        cmd.extend(["--label-parse-progress-every", str(args.label_parse_progress_every)])
    if args.use_cached_intersection:
        cmd.append("--use-cached-intersection")
    if args.intersection_cache:
        cmd.extend(["--intersection-cache", str(args.intersection_cache)])
    if args.use_cached_needed_msg_ids:
        cmd.append("--use-cached-needed-msg-ids")
    if args.needed_msg_ids_cache:
        cmd.extend(["--needed-msg-ids-cache", str(args.needed_msg_ids_cache)])
    return cmd


def _collect_registry_from_summary(summary_path: Path, status: str) -> List[Dict[str, object]]:
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows: List[Dict[str, object]] = []
    for exp in payload.get("experiments", []):
        cfg = exp.get("config", {})
        ckpt = exp.get("best_ckpt", "")
        emb = exp.get("dynamic_embedding", "")
        rows.append(
            {
                "variant": cfg.get("name", ""),
                "ablation": cfg.get("ablation", ""),
                "training_strategy": cfg.get("training_strategy", ""),
                "seed": cfg.get("seed", ""),
                "status": status if ckpt and emb else "missing",
                "checkpoint_path": ckpt,
                "embedding_path": emb,
                "summary_path": str(summary_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multitask representation variants and export embeddings")
    parser.add_argument("--seeds", type=str, default="42,2023,3407")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--export-batch-size", type=int, default=256)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--stage3-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=25)
    parser.add_argument("--lr-stage1", type=float, default=1e-4)
    parser.add_argument("--lr-stage2", type=float, default=5e-4)
    parser.add_argument("--lr-stage3", type=float, default=1e-3)
    parser.add_argument("--lr-joint", type=float, default=3e-4)

    parser.add_argument("--rec-weight", type=float, default=1.5)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--rec-neg-reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--rec-num-neg", type=int, default=10)
    parser.add_argument("--rec-on-stage1", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rec-on-stage2", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rec-on-stage3", type=int, default=1, choices=[0, 1])

    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--short-recent", type=int, default=30)
    parser.add_argument("--long-recent", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--topk-adapters", type=int, default=3)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--export-amp", type=int, default=1, choices=[0, 1])

    parser.add_argument("--export-gate-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--export-kl-threshold", type=float, default=0.012)
    parser.add_argument("--export-kl-soft-scale", type=float, default=0.01)
    parser.add_argument("--export-gate-min-alpha", type=float, default=0.05)
    parser.add_argument("--export-gate-max-alpha", type=float, default=0.80)

    parser.add_argument("--label-user-dirs", type=str, default="")
    parser.add_argument("--labels-cache", type=str, default="")
    parser.add_argument("--use-cached-labels", action="store_true")
    parser.add_argument("--label-parse-progress-every", type=int, default=0)

    parser.add_argument("--samples-dir", type=str, default="dataset/local/message_user_samples")
    parser.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    parser.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    parser.add_argument("--sample-cache-dir", type=str, default="")

    parser.add_argument("--use-cached-intersection", action="store_true")
    parser.add_argument("--intersection-cache", type=str, default="")
    parser.add_argument("--use-cached-needed-msg-ids", action="store_true")
    parser.add_argument("--needed-msg-ids-cache", type=str, default="")

    parser.add_argument("--train-subsample-ratio", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-variants",
        type=str,
        default="",
        help="Comma-separated variant names to exclude from the sweep, e.g. Full,NoAdapter",
    )
    parser.add_argument(
        "--only-variants",
        type=str,
        default="",
        help="Comma-separated variant names to run exclusively, e.g. RecOnly",
    )

    parser.add_argument(
        "--export-mode-fallback",
        type=str,
        default="thresholded",
        choices=["base", "adapter", "thresholded"],
        help="Pipeline default export_mode (overridden per variant in sweep).",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    skip_variants = set(_parse_variant_list(args.skip_variants))
    only_variants = set(_parse_variant_list(args.only_variants))

    registry_rows: List[Dict[str, object]] = []

    for seed in seeds:
        seed_root = RESULTS_DIR / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        sweep_path = seed_root / "sweep.json"

        sweep_rows = _build_sweep_rows(args, seed, skip_variants, only_variants)
        if args.skip_existing:
            filtered_rows = []
            for row in sweep_rows:
                run_dir = seed_root / str(row["name"])
                ckpt = run_dir / "best_model.pth"
                emb = run_dir / "dynamic_user_embeddings.pt"
                if ckpt.exists() and emb.exists():
                    continue
                filtered_rows.append(row)
            sweep_rows = filtered_rows

        if sweep_rows:
            sweep_path.write_text(json.dumps(sweep_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            log_path = seed_root / "pipeline.log"
            cmd = _build_pipeline_cmd(args, sweep_path, seed_root)
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
            status = "ok" if proc.returncode == 0 else "failed"
            print(f"[Done] seed={seed} status={status} elapsed={elapsed:.1f}s")
        else:
            status = "skipped"
            print(f"[Skip] seed={seed} all variants already exist")

        summary_path = seed_root / "summary.json"
        registry_rows.extend(_collect_registry_from_summary(summary_path, status))

    if registry_rows:
        fieldnames = [
            "variant",
            "ablation",
            "training_strategy",
            "seed",
            "status",
            "checkpoint_path",
            "embedding_path",
            "summary_path",
        ]
        with REGISTRY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(registry_rows)

        REGISTRY_JSON.write_text(json.dumps(registry_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] registry saved: {REGISTRY_CSV}")


if __name__ == "__main__":
    main()
