#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation checkpoints → Task A classification + Task B interaction.

Uses downstream/taskA/export_from_checkpoint.py (loose load) then
downstream/taskB/run_interaction.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DYN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DYN_ROOT.parent
EXPORT_PY = REPO_ROOT / "downstream" / "taskA" / "export_from_checkpoint.py"
INTERACTION_PY = REPO_ROOT / "downstream" / "taskB" / "run_interaction.py"

DEFAULT_VARIANTS = ("S0_StagedRef", "S1_JointAllModules")
DEFAULT_PRETRAIN_SEEDS = (42, 2023, 3407)


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=" checkpoint → TaskA  + ")
    p.add_argument(
        "--results-root",
        type=str,
        default=str(DYN_ROOT / "ablation_dynamic_model" / "results"),
        help="results root from run_training_strategy_ablation.py",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DYN_ROOT / "ablation_dynamic_model" / "downstream_outputs"),
        help="option",
    )
    p.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS), help="option")
    p.add_argument("--pretrain-seeds", type=str, default=",".join(map(str, DEFAULT_PRETRAIN_SEEDS)), help="option")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--classification-only", action="store_true", help="option")
    p.add_argument("--interaction-only", action="store_true", help="run interaction only")
    p.add_argument("--skip-export-classification", action="store_true", help="skip export if embeddings already exist")
    p.add_argument(
        "--interaction-sample-ratio",
        type=float,
        default=1.0,
        help="subsample ratio for interaction eval",
    )
    p.add_argument("--interaction-seeds", type=str, default="42,52,62", help="option")
    p.add_argument("--interaction-epochs", type=int, default=10)
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    pretrain_seeds = [int(x.strip()) for x in args.pretrain_seeds.split(",") if x.strip()]

    emb_index: dict[str, dict[str, str]] = {}
    classification_summaries: dict[str, object] = {}

    if not args.skip_export_classification and not args.interaction_only:
        if not EXPORT_PY.is_file():
            raise FileNotFoundError(EXPORT_PY)
        for v in variants:
            for s in pretrain_seeds:
                ckpt = results_root / v / f"seed_{s}" / "time_slice_model.pth"
                if not ckpt.is_file():
                    print(f"[Skip]  checkpoint: {ckpt}", flush=True)
                    continue
                tag = f"{v}_pt{s}"
                sub = out_root / tag
                sub.mkdir(parents=True, exist_ok=True)
                out_emb = (sub / "user_window_repr_thresholded.pt").resolve()
                out_json = (sub / "taska_classification_probe.json").resolve()
                cmd = [
                    sys.executable,
                    str(EXPORT_PY),
                    "--checkpoint",
                    str(ckpt),
                    "--loose-checkpoint-load",
                    "--output-emb",
                    str(out_emb),
                    "--output-json",
                    str(out_json),
                    "--split",
                    "all",
                    "--device",
                    str(args.device),
                ]
                _run(cmd)
                emb_index[tag] = {"file": str(out_emb)}
                try:
                    classification_summaries[tag] = json.loads(out_json.read_text(encoding="utf-8"))
                except Exception as e:
                    classification_summaries[tag] = {"status": "error", "error": str(e)}

        sum_path = out_root / "taska_classification_all_runs.json"
        sum_path.write_text(
            json.dumps(classification_summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] TaskA  -> {sum_path}", flush=True)

    if not args.classification_only:
        if not INTERACTION_PY.is_file():
            raise FileNotFoundError(INTERACTION_PY)
        if args.skip_export_classification or args.interaction_only:
            emb_index = {}
            for sub in sorted(out_root.iterdir()):
                if not sub.is_dir():
                    continue
                pt = sub / "user_window_repr_thresholded.pt"
                if pt.is_file():
                    emb_index[sub.name] = {"file": str(pt.resolve())}
        if not emb_index:
            raise RuntimeError(" embedding； --output-dir")

        index_path = out_root / "embedding_index_strategy_ablation.json"
        index_path.write_text(json.dumps(emb_index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK]  {index_path}  {len(emb_index)} ", flush=True)

        out_inter = out_root / "downstream_interaction_binary_strategy_ablation.json"
        cmd = [
            sys.executable,
            str(INTERACTION_PY),
            "--embedding-index-json",
            str(index_path),
            "--dynamic-embedding",
            "NONE",
            "--output-json",
            str(out_inter),
            "--seeds",
            str(args.interaction_seeds),
            "--epochs",
            str(int(args.interaction_epochs)),
            "--device",
            str(args.device),
            "--interaction-sample-ratio",
            str(float(args.interaction_sample_ratio)),
        ]
        _run(cmd)
        print(f"[OK]  -> {out_inter}", flush=True)

    print("[Done] run_strategy_ablation_downstream ", flush=True)


if __name__ == "__main__":
    main()
