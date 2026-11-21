#!/usr/bin/env python3
"""
（ pipeline_multitask_vs_baselines.py）。

（，）：

  1. wo_adapter           — w/o Adapter：ablation=no_adapter, staged
  2. wo_gating            — w/o Gating：ablation=no_gating, staged
  3. static_only          — Static Only：ablation=only_user（），staged
  4. short_term_only      — Short-term Only：ablation=only_short，staged
  5. long_term_only       — Long-term Only：ablation=only_long，staged
  6. wo_stage_wise        — w/o Stage-wise Training：ablation=none, training_strategy=joint

 Full baseline：ablation=none, staged（）。

：sweep  pipeline_multitask_vs_baselines.ExpConfig ，
 Full（） checkpoint  config ；
ablation / training_strategy / export_mode。 export_mode （ adapter
 base ），。

export_mode （ pipeline ）：
  - no_adapter → base（ adapter ）
  - no_gating → adapter（）
  -  → thresholded

：joint  joint_epochs / lr_joint； staged ，
， --joint-epochs（「」）。

： pipeline ， seed 
pipeline_ablation_controlled.log（）。

： pipeline  --use-cached-labels（ time_slice_labels_cache.json ）；
 json， --no-use-cached-labels。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline_multitask_vs_baselines import ExpConfig

PIPELINE_SCRIPT = BASE_DIR / "pipeline_multitask_vs_baselines.py"
OUTPUT_ROOT_DEFAULT = Path(__file__).resolve().parent / "results_controlled"
REGISTRY_CSV_DEFAULT = Path(__file__).resolve().parent / "ablation_registry.csv"
REGISTRY_JSON_DEFAULT = Path(__file__).resolve().parent / "ablation_registry.json"


ABLATION_SUITE: List[Dict[str, object]] = [
    {
        "id": "wo_adapter",
        "title_zh": "w/o Adapter",
        "name": "abl_wo_adapter",
        "ablation": "no_adapter",
        "training_strategy": "staged",
        "export_mode": "base",
    },
    {
        "id": "wo_gating",
        "title_zh": "w/o Gating",
        "name": "abl_wo_gating",
        "ablation": "no_gating",
        "training_strategy": "staged",
        "export_mode": "adapter",
    },
    {
        "id": "static_only",
        "title_zh": "Static Only",
        "name": "abl_static_only",
        "ablation": "only_user",
        "training_strategy": "staged",
        "export_mode": "thresholded",
    },
    {
        "id": "short_term_only",
        "title_zh": "Short-term Only",
        "name": "abl_short_term_only",
        "ablation": "only_short",
        "training_strategy": "staged",
        "export_mode": "thresholded",
    },
    {
        "id": "long_term_only",
        "title_zh": "Long-term Only",
        "name": "abl_long_term_only",
        "ablation": "only_long",
        "training_strategy": "staged",
        "export_mode": "thresholded",
    },
    {
        "id": "wo_stage_wise",
        "title_zh": "w/o Stage-wise Training",
        "name": "abl_wo_stage_wise",
        "ablation": "none",
        "training_strategy": "joint",
        "export_mode": "thresholded",
    },
]

FULL_BASELINE: Dict[str, object] = {
    "id": "full_baseline",
    "title_zh": "Full（ +  + Adapter + Gating）",
    "name": "abl_full_baseline",
    "ablation": "none",
    "training_strategy": "staged",
    "export_mode": "thresholded",
}

_PIPELINE_ABLATIONS = frozenset({"none", "no_adapter", "no_gating", "only_user", "only_short", "only_long"})
_PIPELINE_STRATEGIES = frozenset({"staged", "joint"})
_EXPORT_MODES = frozenset({"base", "adapter", "thresholded"})


def _validate_ablation_suite() -> None:
    """ pipeline_multitask_vs_baselines.train_one_experiment / export_embeddings 。"""
    for row in (*ABLATION_SUITE, FULL_BASELINE):
        ab = str(row["ablation"]).lower().strip()
        ts = str(row["training_strategy"]).lower().strip()
        em = str(row["export_mode"]).lower().strip()
        if ab not in _PIPELINE_ABLATIONS:
            raise ValueError(f"variant {row.get('id')!r}: ablation={ab!r}  pipeline ")
        if ts not in _PIPELINE_STRATEGIES:
            raise ValueError(f"variant {row.get('id')!r}: training_strategy={ts!r}  staged|joint")
        if em not in _EXPORT_MODES:
            raise ValueError(f"variant {row.get('id')!r}: export_mode={em!r}  base|adapter|thresholded")
        if ab == "no_adapter" and em != "base":
            raise ValueError(
                f"variant {row.get('id')!r}: ablation=no_adapter  export_mode=base（ {em!r}）"
            )
        if ab == "no_gating" and em == "base":
            raise ValueError(
                f"variant {row.get('id')!r}: ablation=no_gating  export_mode=base（ adapter  thresholded）"
            )


_validate_ablation_suite()


def _ids_map() -> Dict[str, Dict[str, object]]:
    m: Dict[str, Dict[str, object]] = {str(v["id"]): v for v in ABLATION_SUITE}
    m[str(FULL_BASELINE["id"])] = FULL_BASELINE
    return m


def _parse_csv_ids(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


def _parse_seeds(seeds_str: str) -> List[int]:
    out: List[int] = []
    for chunk in str(seeds_str).split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out or [42]


def _select_variants(
    include_full: bool,
    only_ids: Sequence[str],
    skip_ids: Sequence[str],
) -> List[Dict[str, object]]:
    skip = set(skip_ids)
    if only_ids:
        id_map = _ids_map()
        ordered: List[Dict[str, object]] = []
        for vid in only_ids:
            if vid not in id_map:
                raise ValueError(f" variant id: {vid!r}，: {sorted(id_map.keys())}")
            if vid in skip:
                continue
            ordered.append(id_map[vid])
        return ordered

    rows = list(ABLATION_SUITE)
    if include_full:
        rows = [FULL_BASELINE] + rows
    return [r for r in rows if str(r["id"]) not in skip]


def _variant_to_exp_config(variant: Dict[str, object], args: argparse.Namespace, seed: int) -> ExpConfig:
    """ Full  args  ExpConfig； variant。"""
    return ExpConfig(
        name=str(variant["name"]),
        seed=int(seed),
        ablation=str(variant["ablation"]),
        training_strategy=str(variant["training_strategy"]),
        export_mode=str(variant["export_mode"]),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage3_epochs=int(args.stage3_epochs),
        joint_epochs=int(args.joint_epochs),
        batch_size=int(args.batch_size),
        export_batch_size=int(args.export_batch_size),
        lr_stage1=float(args.lr_stage1),
        lr_stage2=float(args.lr_stage2),
        lr_stage3=float(args.lr_stage3),
        lr_joint=float(args.lr_joint),
        weight_decay=float(args.weight_decay),
        rec_weight=float(args.rec_weight),
        cls_weight=float(args.cls_weight),
        rec_neg_reduction=str(args.rec_neg_reduction),
        rec_num_neg=int(args.rec_num_neg),
        rec_on_stage1=bool(int(args.rec_on_stage1)),
        rec_on_stage2=bool(int(args.rec_on_stage2)),
        rec_on_stage3=bool(int(args.rec_on_stage3)),
        short_recent=int(args.short_recent),
        long_recent=int(args.long_recent),
        window_size=int(args.window_size),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        topk_adapters=int(args.topk_adapters),
        num_workers=int(args.num_workers),
        amp=bool(int(args.amp)),
        device=str(args.device),
        export_gate_mode=str(args.export_gate_mode),
        export_kl_threshold=float(args.export_kl_threshold),
        export_kl_soft_scale=float(args.export_kl_soft_scale),
        export_gate_min_alpha=float(args.export_gate_min_alpha),
        export_gate_max_alpha=float(args.export_gate_max_alpha),
        export_amp=bool(int(args.export_amp)),
    )


def _build_sweep_rows(
    variants: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    seed: int,
) -> List[Dict[str, object]]:
    expected_keys = {f.name for f in fields(ExpConfig)}
    sweep: List[Dict[str, object]] = []
    for variant in variants:
        cfg = _variant_to_exp_config(variant, args, seed)
        row = asdict(cfg)
        if set(row.keys()) != expected_keys:
            raise RuntimeError(
                f"sweep  ExpConfig : missing={expected_keys - set(row.keys())} "
                f"extra={set(row.keys()) - expected_keys}"
            )
        sweep.append(row)
    return sweep


def _build_pipeline_cmd(args: argparse.Namespace, sweep_path: Path, seed_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
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
    if args.label_parse_progress_every:
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


def _run_pipeline_streaming_to_terminal_and_log(cmd: List[str], log_path: Path, cwd: str) -> int:
    """ stdout/stderr ，。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# cmd: {' '.join(cmd)}\n"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(header)
        logf.flush()
        sys.stdout.write(f"[Ablation]  pipeline （ {log_path}）\n{header}")
        sys.stdout.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        finally:
            proc.stdout.close()
        return int(proc.wait())


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
                "variant_id": cfg.get("name", ""),
                "ablation": cfg.get("ablation", ""),
                "training_strategy": cfg.get("training_strategy", ""),
                "export_mode": cfg.get("export_mode", ""),
                "seed": cfg.get("seed", ""),
                "status": status if ckpt and emb else "missing",
                "checkpoint_path": ckpt,
                "embedding_path": emb,
                "summary_path": str(summary_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="： +  Full baseline（ pipeline）")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--registry-csv", type=str, default=str(REGISTRY_CSV_DEFAULT))
    parser.add_argument("--registry-json", type=str, default=str(REGISTRY_JSON_DEFAULT))

    parser.add_argument("--include-full-baseline", action="store_true", help=" Full baseline（staged + none）")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=" variant id，， wo_adapter,wo_gating（ ABLATION_SUITE）",
    )
    parser.add_argument("--skip-variants", type=str, default="", help=" variant id")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--export-batch-size", type=int, default=256)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--stage3-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
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

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--export-amp", type=int, default=1, choices=[0, 1])

    parser.add_argument("--export-gate-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--export-kl-threshold", type=float, default=0.012)
    parser.add_argument("--export-kl-soft-scale", type=float, default=0.01)
    parser.add_argument("--export-gate-min-alpha", type=float, default=0.05)
    parser.add_argument("--export-gate-max-alpha", type=float, default=0.80)

    parser.add_argument(
        "--label-user-dirs",
        type=str,
        default="",
    )
    parser.add_argument("--labels-cache", type=str, default="")
    parser.add_argument(
        "--use-cached-labels",
        dest="use_cached_labels",
        action="store_true",
        help=" pipeline： time_slice_labels_cache.json （， multitask Full ）",
    )
    parser.add_argument(
        "--no-use-cached-labels",
        dest="use_cached_labels",
        action="store_false",
        help=" pipeline：， label-user-dirs ",
    )
    parser.set_defaults(use_cached_labels=True)
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
    parser.add_argument("--skip-existing", action="store_true", help=" best_model.pth  dynamic_user_embeddings.pt  variant")
    parser.add_argument("--dry-run", action="store_true", help=" sweep.json ，")

    parser.add_argument("--print-suite", action="store_true", help="option", )
    args = parser.parse_args()

    if args.print_suite:
        print(json.dumps([{"id": x["id"], "title_zh": x["title_zh"], **{k: x[k] for k in ("name", "ablation", "training_strategy", "export_mode")}} for x in ABLATION_SUITE], ensure_ascii=False, indent=2))
        print("[Full baseline]", json.dumps({k: FULL_BASELINE[k] for k in ("id", "title_zh", "name", "ablation", "training_strategy", "export_mode")}, ensure_ascii=False))
        return

    if not PIPELINE_SCRIPT.is_file():
        raise FileNotFoundError(f" pipeline: {PIPELINE_SCRIPT}")

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    only_ids = _parse_csv_ids(args.only)
    skip_ids = _parse_csv_ids(args.skip_variants)
    variants = _select_variants(include_full=args.include_full_baseline, only_ids=only_ids, skip_ids=skip_ids)

    seeds = _parse_seeds(args.seeds)
    registry_rows: List[Dict[str, object]] = []

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        sweep_path = seed_root / "sweep_ablation_controlled.json"

        sweep_rows = _build_sweep_rows(variants, args, seed)
        if args.skip_existing:
            filtered: List[Dict[str, object]] = []
            for row in sweep_rows:
                run_dir = seed_root / str(row["name"])
                if (run_dir / "best_model.pth").is_file() and (run_dir / "dynamic_user_embeddings.pt").is_file():
                    continue
                filtered.append(row)
            sweep_rows = filtered

        if not sweep_rows:
            print(f"[Skip] seed={seed}  variant（ skip-existing ）")
            summary_path = seed_root / "summary.json"
            registry_rows.extend(_collect_registry_from_summary(summary_path, "skipped"))
            continue

        sweep_path.write_text(json.dumps(sweep_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = _build_pipeline_cmd(args, sweep_path, seed_root)

        print(f"[Sweep] seed={seed} path={sweep_path} n={len(sweep_rows)}")
        if args.dry_run:
            print(json.dumps(sweep_rows, ensure_ascii=False, indent=2))
            print("[Dry-run] " + " ".join(cmd))
            continue

        log_path = seed_root / "pipeline_ablation_controlled.log"
        started = time.time()
        rc = _run_pipeline_streaming_to_terminal_and_log(cmd, log_path, cwd=str(BASE_DIR))
        elapsed = time.time() - started
        status = "ok" if rc == 0 else "failed"
        print(f"[Done] seed={seed} status={status} elapsed={elapsed:.1f}s log={log_path}")

        summary_path = seed_root / "summary.json"
        registry_rows.extend(_collect_registry_from_summary(summary_path, status))

    if args.dry_run:
        return

    reg_csv = Path(args.registry_csv).expanduser()
    reg_json = Path(args.registry_json).expanduser()
    if registry_rows:
        reg_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "variant_id",
            "ablation",
            "training_strategy",
            "export_mode",
            "seed",
            "status",
            "checkpoint_path",
            "embedding_path",
            "summary_path",
        ]
        with reg_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(registry_rows)
        reg_json.write_text(json.dumps(registry_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] registry: {reg_csv}")


if __name__ == "__main__":
    main()
