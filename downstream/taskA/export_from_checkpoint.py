#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task A: export frozen window vectors from a DUMoE checkpoint, then run the classification probe."""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMOE_DIR = REPO_ROOT / "dumoe"
for _p in (str(REPO_ROOT), str(DUMOE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline_multitask_vs_baselines import (  # noqa: E402
    ExpConfig,
    TARGET_CATEGORIES,
    _entry_has_context,
    export_embeddings,
)
from time_slice_data_loader import (  # noqa: E402
    TimeSlicePreferenceDataset,
    build_label_map,
    build_split_entries,
    build_user_tweet_ids_map,
    collate_fn,
    collect_needed_msg_ids_from_entries,
    load_time_slice_labels_from_user_dirs,
    load_user_features_from_pt,
)
from time_slice_msg_feature_loader import (  # noqa: E402
    collect_user_ids_from_compact_shards,
    load_msg_features_from_compact_shards,
)


def exp_config_from_checkpoint(cd: Dict) -> ExpConfig:
    cfg = ExpConfig(name=str(cd.get("name", "mtask_rec")))
    for f in dataclasses.fields(ExpConfig):
        if f.name in cd and f.name != "name":
            setattr(cfg, f.name, cd[f.name])
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Export multitask user_repr then run TaskA classification metrics")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_model.pth",
    )
    p.add_argument(
        "--output-emb",
        type=str,
        default="",
        help=" .pt； --output-json  multitask_repr_for_taskA.pt",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Where to write Task A probe metrics JSON (ignored with --export-only)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help=" split （ all； pipeline  train/val/test ）",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help=" --output-emb，",
    )
    p.add_argument(
        "--metrics-py",
        type=str,
        default=str(Path(__file__).resolve().parent / "run_classification.py"),
    )
    p.add_argument("--samples-csv", type=str, default="dataset/ids/windows.csv")
    p.add_argument(
        "--label-user-dirs",
        type=str,
        default="",
    )
    p.add_argument("--labels-cache", type=str, default="dataset/local/labels_cache.json")
    p.add_argument("--samples-dir", type=str, default="dataset/local/message_user_samples")
    p.add_argument(
        "--user-profile-pt",
        type=str,
        default="dataset/local/user_profile.pt",
    )
    p.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    p.add_argument("--train-subsample-ratio", type=float, default=1.0)
    p.add_argument("--data-seed", type=int, default=42, help=" pipeline  train/val/test  seed（ 42）")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--classification-seeds", type=str, default="42,52,62")
    p.add_argument("--classification-epochs", type=int, default=20)
    p.add_argument("--classification-batch-size", type=int, default=512)
    p.add_argument("--classification-lr", type=float, default=1e-3)
    p.add_argument("--force-random-split", action="store_true", help="Pass through to Task A probe")
    p.add_argument("--skip-export", action="store_true", help=" output-emb ")
    args = p.parse_args()
    if not args.export_only and not str(args.output_json).strip():
        raise SystemExit(" --output-json（ --export-only）")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if str(args.output_emb).strip():
        out_emb = Path(args.output_emb)
    else:
        out_emb = out_json.parent / "multitask_repr_for_taskA.pt"
    out_emb.parent.mkdir(parents=True, exist_ok=True)

    if not (args.skip_export and out_emb.is_file()):
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cd = ckpt.get("config") or {}
        if not isinstance(cd, dict):
            raise RuntimeError("checkpoint  config ")
        cfg = exp_config_from_checkpoint(cd)
        num_classes = int(ckpt.get("num_classes") or len(TARGET_CATEGORIES))

        random.seed(int(args.data_seed))
        torch.manual_seed(int(args.data_seed))

        label_user_dirs = [x.strip() for x in args.label_user_dirs.split(",") if x.strip()]
        labels_by_user, csv_path_by_user = load_time_slice_labels_from_user_dirs(
            user_dirs=label_user_dirs,
            cache_path=args.labels_cache,
            use_cache=True,
            progress_every=500,
        )
        user_feature_map, user_feat_dim = load_user_features_from_pt(args.user_profile_pt)
        users_in_shards = collect_user_ids_from_compact_shards(args.tweet_emb_shards_dir, verbose=True)
        users = set(labels_by_user.keys()) & set(csv_path_by_user.keys()) & set(user_feature_map.keys()) & users_in_shards
        labels_by_user = {u: labels_by_user[u] for u in users}
        csv_path_by_user = {u: csv_path_by_user[u] for u in users}
        user_feature_map = {u: user_feature_map[u] for u in users}
        user_tweet_ids_map = build_user_tweet_ids_map(csv_path_by_user)
        label_to_id = build_label_map(TARGET_CATEGORIES)
        train_entries, val_entries, test_entries = build_split_entries(labels_by_user, label_to_id)

        cfg_window_size = int(cfg.window_size)
        cfg_long_recent = int(cfg.long_recent)
        train_entries = [e for e in train_entries if _entry_has_context(e, user_tweet_ids_map, cfg_window_size)]
        val_entries = [e for e in val_entries if _entry_has_context(e, user_tweet_ids_map, cfg_window_size)]
        test_entries = [e for e in test_entries if _entry_has_context(e, user_tweet_ids_map, cfg_window_size)]

        ratio = float(args.train_subsample_ratio)
        if ratio <= 0 or ratio > 1.0:
            raise ValueError("--train-subsample-ratio  (0, 1] ")
        if ratio < 1.0:
            rnd = random.Random(int(args.data_seed))

            def _subsample(entries: List[Tuple[str, int, str, int]], name: str):
                if len(entries) <= 1:
                    return entries
                keep_n = max(1, int(len(entries) * ratio))
                idxs = list(range(len(entries)))
                rnd.shuffle(idxs)
                keep = set(idxs[:keep_n])
                return [e for i, e in enumerate(entries) if i in keep]

            train_entries = _subsample(train_entries, "train")
            val_entries = _subsample(val_entries, "val")
            test_entries = _subsample(test_entries, "test")

        sp = str(args.split).lower().strip()
        if sp == "all":
            export_entries = train_entries + val_entries + test_entries
            split_tags = (
                ["train"] * len(train_entries)
                + ["val"] * len(val_entries)
                + ["test"] * len(test_entries)
            )
        elif sp == "train":
            export_entries = train_entries
            split_tags = ["train"] * len(export_entries)
        elif sp == "val":
            export_entries = val_entries
            split_tags = ["val"] * len(export_entries)
        else:
            export_entries = test_entries
            split_tags = ["test"] * len(export_entries)

        needed = collect_needed_msg_ids_from_entries(
            export_entries,
            user_tweet_ids_map,
            window_size=cfg_window_size,
            long_recent=cfg_long_recent,
        )
        msg_features_dict, msg_feat_dim = load_msg_features_from_compact_shards(
            shards_dir=args.tweet_emb_shards_dir,
            needed_msg_ids=needed,
            verbose=True,
        )

        export_ds = TimeSlicePreferenceDataset(
            user_feature_map,
            msg_features_dict,
            user_tweet_ids_map,
            export_entries,
            msg_feat_dim=msg_feat_dim,
            user_feat_dim=user_feat_dim,
            short_recent=int(cfg.short_recent),
            long_recent=int(cfg.long_recent),
            window_size=int(cfg.window_size),
        )
        dl_common = dict(
            batch_size=int(cfg.export_batch_size),
            collate_fn=collate_fn,
            num_workers=max(0, int(cfg.num_workers)),
            pin_memory=torch.cuda.is_available(),
        )
        export_loader = DataLoader(export_ds, shuffle=False, **dl_common)

        print(f"[Export] split={sp} entries={len(export_ds)} -> {out_emb}", flush=True)
        export_embeddings(
            model_ckpt_path=ckpt_path,
            export_loader=export_loader,
            msg_feat_dim=msg_feat_dim,
            user_feat_dim=user_feat_dim,
            num_classes=num_classes,
            device_str=str(args.device),
            out_pt_path=out_emb,
            split_tags=split_tags,
            ablation=str(cfg.ablation),
            mode=str(cfg.export_mode),
            gate_mode=str(cfg.export_gate_mode),
            kl_threshold=float(cfg.export_kl_threshold),
            kl_soft_scale=float(cfg.export_kl_soft_scale),
            gate_min_alpha=float(cfg.export_gate_min_alpha),
            gate_max_alpha=float(cfg.export_gate_max_alpha),
            amp=bool(int(cfg.export_amp)),
        )
        pl = torch.load(str(out_emb), map_location="cpu", weights_only=False)
        emb_tensor = pl.get("embeddings") if isinstance(pl, dict) else None
        n_samples = int(emb_tensor.size(0)) if isinstance(emb_tensor, torch.Tensor) else 0
        print(f"[Export] done checkpoint={ckpt_path} num_samples={n_samples}", flush=True)
    else:
        print(f"[Export] skip (exists): {out_emb}", flush=True)

    if args.export_only:
        print(f"[OK] export-only -> {out_emb}", flush=True)
        return

    metrics_py = Path(args.metrics_py)
    if not metrics_py.is_file():
        raise FileNotFoundError(metrics_py)

    cmd = [
        sys.executable,
        str(metrics_py),
        "--embedding-files",
        str(out_emb),
        "--samples-csv",
        str(args.samples_csv),
        "--seeds",
        str(args.classification_seeds),
        "--epochs",
        str(int(args.classification_epochs)),
        "--batch-size",
        str(int(args.classification_batch_size)),
        "--lr",
        str(float(args.classification_lr)),
        "--device",
        str(args.device),
        "--output-json",
        str(out_json),
    ]
    if args.force_random_split:
        cmd.append("--force-random-split")

    print("[Run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"[OK] TaskA classification metrics -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
