#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

CUR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CUR_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from project.data.processor import (
    UnifiedWindowDataset,
    build_id2idx_from_train,
    build_item_embedding_table_fixed,
    build_records_by_split,
    load_msg_features_from_compact_shards,
    load_or_build_user_tweet_ids_map,
    load_user_features_from_pt,
)
from project.models.baselines import build_model
from project.trainer.unified_trainer import Trainer


ALL_MODELS = ["SASRec", "BERT4Rec", "MIND", "PeterRec", "PTUM", "UniSRec", "HORAE", "DisMIR", "HM4SR"]


def _state_dict_without_frozen_item_table(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """/， checkpoint  GB 。"""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("item_embedding.")
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        # Speed up matmul on Ampere+ GPUs with negligible quality impact for training.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def main() -> None:
    p = argparse.ArgumentParser(description="Unified 8-baseline pretraining pipeline")
    p.add_argument("--models", type=str, default="all", help="comma names or all")
    p.add_argument("--samples-csv", type=str, default="dataset/ids/windows.csv")
    p.add_argument("--labels-cache", type=str, default="dataset/local/labels_cache.json")
    p.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    p.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--num-classes", type=int, default=15)
    # Keep baseline training budget aligned with the user's dynamic model joint setup (joint_epochs=25).
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4, help="dataloader worker processes")
    p.add_argument("--prefetch-factor", type=int, default=4, help="dataloader prefetch factor when num_workers > 0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision training")
    p.add_argument("--log-every-batches", type=int, default=200, help="print train progress every N batches")
    p.add_argument("--verbose-data", action="store_true", help="print detailed val/test data loading logs")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/baselines")
    p.add_argument(
        "--preprocess-cache-dir",
        type=str,
        default="cache",
        help="cache directory for preprocessed user sequence map",
    )
    p.add_argument("--export-split", type=str, default="test", choices=["train", "val", "test"], help="which split to export embeddings")
    p.add_argument("--do-downstream", action="store_true", help="optional: run downstream tasks after pretraining")
    p.add_argument(
        "--use-profile",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="profile usage switch: auto(default baselines=False, DisMIR=True), true, false",
    )
    p.add_argument(
        "--dismir-objective",
        type=str,
        default="next_item",
        choices=["next_item", "classification"],
        help="training objective for DisMIR; default next_item for fairness",
    )
    clean_argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    args = p.parse_args(clean_argv)

    set_seed(int(args.seed))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[Stage] Building shared preprocessing context...", flush=True)
    t_stage = time.time()
    records_by_split = build_records_by_split(
        samples_csv=args.samples_csv,
        max_train=int(args.max_train),
        max_val=int(args.max_val),
        max_test=int(args.max_test),
    )
    all_users = {r.user_id for s in ["train", "val", "test"] for r in records_by_split[s]}
    user_tweet_ids_map = load_or_build_user_tweet_ids_map(
        labels_cache=args.labels_cache,
        users=all_users,
        cache_dir=str(args.preprocess_cache_dir),
        verbose=True,
    )
    needed_msg_ids = set()
    for s in ["train", "val", "test"]:
        for r in records_by_split[s]:
            ordered = user_tweet_ids_map.get(r.user_id, [])
            st = int(r.window_idx) * int(args.window_size)
            needed_msg_ids.update(str(x) for x in ordered[st : st + int(args.seq_len)])
    print(f"[Stage] Loading message features once for all splits: needed_ids={len(needed_msg_ids)}", flush=True)
    shared_msg_feats, shared_msg_dim = load_msg_features_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir, needed_msg_ids=needed_msg_ids, verbose=True
    )
    shared_user_feat_map, shared_profile_dim = load_user_features_from_pt(args.user_profile_pt)

    id2idx, sorted_ids = build_id2idx_from_train(
        records_by_split["train"],
        user_tweet_ids_map,
        shared_msg_feats,
        int(args.seq_len),
        int(args.window_size),
    )
    shared_emb = build_item_embedding_table_fixed(sorted_ids, shared_msg_feats, int(shared_msg_dim))
    print(
        f"[Stage] vocab_items={len(sorted_ids)} embed_rows={shared_emb.num_embeddings} "
        f"(shared context {time.time() - t_stage:.1f}s)",
        flush=True,
    )

    print("[Stage] Building train dataset...", flush=True)
    t_stage = time.time()
    train_ds = UnifiedWindowDataset(
        records_by_split["train"],
        user_tweet_ids_map,
        shared_user_feat_map,
        int(shared_profile_dim),
        int(args.seq_len),
        int(args.window_size),
        id2idx,
        verbose=True,
    )
    print(f"[Stage] Train dataset ready: samples={len(train_ds)} time={time.time() - t_stage:.1f}s", flush=True)

    print("[Stage] Building val dataset...", flush=True)
    t_stage = time.time()
    val_ds = UnifiedWindowDataset(
        records_by_split["val"],
        user_tweet_ids_map,
        shared_user_feat_map,
        int(shared_profile_dim),
        int(args.seq_len),
        int(args.window_size),
        id2idx,
        verbose=bool(args.verbose_data),
    )
    print(f"[Stage] Val dataset ready: samples={len(val_ds)} time={time.time() - t_stage:.1f}s", flush=True)

    print("[Stage] Building test dataset...", flush=True)
    t_stage = time.time()
    test_ds = UnifiedWindowDataset(
        records_by_split["test"],
        user_tweet_ids_map,
        shared_user_feat_map,
        int(shared_profile_dim),
        int(args.seq_len),
        int(args.window_size),
        id2idx,
        verbose=bool(args.verbose_data),
    )
    print(f"[Stage] Test dataset ready: samples={len(test_ds)} time={time.time() - t_stage:.1f}s", flush=True)

    nw = max(0, int(args.num_workers))
    common_loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": nw,
        "pin_memory": torch.cuda.is_available(),
    }
    if nw > 0:
        common_loader_kwargs["persistent_workers"] = True
        common_loader_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common_loader_kwargs)
    print(
        f"[Stage] Dataloaders ready: train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}"
        ,
        flush=True,
    )

    chosen: List[str] = ALL_MODELS if args.models.strip().lower() == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    summary: Dict[str, Dict] = {"mode": "pretrain_only", "models": {}}

    use_profile_arg = str(args.use_profile).lower()
    use_profile_override = None if use_profile_arg == "auto" else (use_profile_arg == "true")
    shared_item_embedding = shared_emb

    for name in chosen:
        print(f"[Stage] Start model={name}", flush=True)
        model = build_model(
            name=name,
            msg_dim=int(shared_msg_dim),
            profile_dim=int(shared_profile_dim),
            item_emb_table=shared_item_embedding,
            num_classes=int(args.num_classes),
            embed_dim=256,
            dismir_objective=str(args.dismir_objective),
            use_profile=use_profile_override,
        )
        trainer = Trainer(
            model=model,
            device=args.device,
            lr=float(args.lr),
            wd=float(args.weight_decay),
            use_amp=(not bool(args.no_amp)),
        )

        best_val = 1e9
        best_state = None
        for ep in range(1, int(args.epochs) + 1):
            tr = trainer.train_epoch(
                train_loader, model_name=name, epoch=ep, log_every=int(args.log_every_batches)
            )
            print(f"[{name}] start val epoch={ep}", flush=True)
            va = trainer.evaluate(
                val_loader,
                model_name=name,
                epoch=ep,
                split_name="val",
                log_every=max(1, int(args.log_every_batches)),
            )
            print(f"[{name}] epoch={ep} train_loss={tr:.6f} val_loss={va:.6f}", flush=True)
            if va < best_val:
                best_val = va
                best_state = _state_dict_without_frozen_item_table(model)

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        model_out = out_root / name
        model_out.mkdir(parents=True, exist_ok=True)
        ckpt = model_out / "encoder_best.pth"
        torch.save(
            {"model": name, "state_dict": _state_dict_without_frozen_item_table(model), "config": vars(args)},
            ckpt,
        )

        export_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[str(args.export_split)]
        print(f"[{name}] start export split={args.export_split}", flush=True)
        emb_map = trainer.extract_embeddings(
            export_loader,
            model_name=name,
            split_name=str(args.export_split),
            log_every=max(1, int(args.log_every_batches)),
        )
        emb_path = model_out / f"taskA_user_embeddings__{name}__256d.pt"
        torch.save({"embeddings": emb_map, "meta": {"model": name, "dim": 256}}, emb_path)

        pretrain_metrics = {
            "best_val_loss": float(best_val),
            "num_embeddings": len(emb_map),
            "embedding_file": str(emb_path),
            "checkpoint": str(ckpt),
            "export_split": str(args.export_split),
            "dismir_objective": str(args.dismir_objective) if str(name).lower() == "dismir" else "n/a",
        }
        (model_out / "pretrain_metrics.json").write_text(
            json.dumps(pretrain_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if bool(args.do_downstream):
            from project.trainer.unified_trainer import run_downstream_tasks

            downstream_json = model_out / "downstream_metrics.json"
            run_downstream_tasks(
                emb_map,
                export_loader,
                shared_emb.weight.detach(),
                str(downstream_json),
                device=args.device,
            )
            pretrain_metrics["downstream_file"] = str(downstream_json)

        summary["models"][name] = pretrain_metrics
        print(f"[{name}] done: embeddings={len(emb_map)}", flush=True)

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] summary saved: {out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

