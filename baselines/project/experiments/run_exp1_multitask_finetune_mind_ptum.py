#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1： baseline（MIND / PTUM）。

（ pipeline_multitask_vs_baselines.ExpConfig ）：
  L = cls_weight * L_cls + rec_weight * L_rec
   cls_weight=1.0, rec_weight=1.5
  L_cls = FocalLoss(gamma=2, alpha=class_weight)
  L_rec = -E[ logσ(u·v+) + mean_k logσ(-u·v-_k) ]   (neg-sampling BPR/logistic)

：
  1)  outputs_full/{MIND,PTUM}/encoder_best.pth（PTUM  item_embedding）
  2)  TaskA 
  3) Export user embeddings, then Task A (classification) / Task B (interaction)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CUR_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CUR_DIR.parent
PARENT_DIR = PROJECT_DIR.parent
REPO_ROOT = PARENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.processor import (
    UnifiedWindowDataset,
    build_records_by_split,
    load_or_build_user_tweet_ids_map,
    load_user_features_from_pt,
)
from project.data.local_feature_loaders import load_msg_features_matrix_from_compact_shards
from project.models.baselines import build_model
from project.trainer.unified_trainer import Trainer, move_to_device


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def compute_neg_sampling_rec_loss(
    user_repr: torch.Tensor,
    pos_item_repr: torch.Tensor,
    neg_item_repr: torch.Tensor,
    neg_reduction: str = "mean",
) -> torch.Tensor:
    pos_logits = (user_repr * pos_item_repr).sum(dim=-1)
    pos_term = F.logsigmoid(pos_logits)
    neg_logits = torch.einsum("bd,bkd->bk", user_repr, neg_item_repr)
    neg_logs = F.logsigmoid(-neg_logits)
    nr = str(neg_reduction).lower().strip()
    if nr == "sum":
        neg_term = neg_logs.sum(dim=-1)
    elif nr == "mean":
        neg_term = neg_logs.mean(dim=-1)
    else:
        raise ValueError(f"neg_reduction  sum|mean，: {neg_reduction!r}")
    return -(pos_term + neg_term).mean()


class MultitaskFinetuneModel(nn.Module):
    """ encode() backbone  + item_proj， L_cls + λ L_rec。"""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        embed_dim: int,
        msg_dim: int,
        num_neg: int = 5,
        neg_reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Linear(int(embed_dim), int(num_classes))
        self.item_proj = nn.Sequential(
            nn.Linear(int(msg_dim), int(embed_dim)),
            nn.ReLU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )
        self.num_neg = int(num_neg)
        self.neg_reduction = str(neg_reduction)

    def encode(self, seq_ids, seq_mask, profile):
        return self.backbone.encode(seq_ids, seq_mask, profile)

    def compute_losses(
        self,
        batch: dict,
        criterion_cls: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_ids, seq_mask, profile, target_item = self.backbone.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        cls_loss = criterion_cls(self.cls_head(z), batch["label"])

        pos_raw = self.backbone.item_embedding(target_item.clamp(0, self.backbone.num_items - 1))
        # sample negatives
        bsz = int(target_item.size(0))
        neg_ids = torch.randint(
            low=1,
            high=max(2, self.backbone.num_items),
            size=(bsz, self.num_neg),
            device=target_item.device,
        )
        # avoid accidental equality with positive (cheap resample once)
        neg_ids = torch.where(neg_ids == target_item.unsqueeze(1), (neg_ids % max(1, self.backbone.num_items - 1)) + 1, neg_ids)
        neg_raw = self.backbone.item_embedding(neg_ids)

        pos_r = F.normalize(self.item_proj(pos_raw), dim=-1)
        neg_r = F.normalize(self.item_proj(neg_raw), dim=-1)
        z_n = F.normalize(z, dim=-1)
        rec_loss = compute_neg_sampling_rec_loss(z_n, pos_r, neg_r, neg_reduction=self.neg_reduction)
        return cls_loss, rec_loss, z


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    cnt = Counter(int(x) for x in labels)
    weights = torch.ones(int(num_classes), dtype=torch.float32)
    total = float(sum(cnt.values()))
    for c in range(int(num_classes)):
        n = float(cnt.get(c, 0))
        weights[c] = total / (float(num_classes) * max(1.0, n))
    return weights


@torch.no_grad()
def eval_classification_head(
    model: MultitaskFinetuneModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    from downstream.taskA.run_classification import compute_metrics

    model.eval()
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for batch in loader:
        b = move_to_device(batch, device)
        z = model.encode(b["seq_ids"], b["seq_mask"], b["profile"])
        logits_all.append(model.cls_head(z).detach().cpu())
        y_all.append(b["label"].detach().cpu())
    if not logits_all:
        return {"hit@1": 0.0, "hit@3": 0.0, "ndcg@1": 0.0, "ndcg@3": 0.0, "recall": 0.0, "kl": 0.0}
    return compute_metrics(torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0))


@torch.no_grad()
def export_embeddings(model: MultitaskFinetuneModel, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    emb: Dict[str, torch.Tensor] = {}
    for batch in loader:
        keys = batch["key"]
        b = move_to_device(batch, device)
        z = model.encode(b["seq_ids"], b["seq_mask"], b["profile"]).detach().cpu()
        for i, k in enumerate(keys):
            emb[str(k)] = z[i].float().view(-1)
    return emb


def load_backbone_checkpoint(
    name: str,
    ckpt_path: Path,
    msg_dim: int,
    profile_dim: int,
    item_emb_table: nn.Embedding,
    num_classes: int,
    embed_dim: int,
    device: torch.device,
) -> nn.Module:
    model = build_model(
        name=name,
        msg_dim=int(msg_dim),
        profile_dim=int(profile_dim),
        item_emb_table=item_emb_table,
        num_classes=int(num_classes),
        embed_dim=int(embed_dim),
        use_profile=False,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    filtered = {k: v for k, v in sd.items() if not str(k).startswith("item_embedding.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"[{name}] loaded {ckpt_path.name}: kept={len(filtered)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return model.to(device)


def finetune_one_model(
    name: str,
    args: argparse.Namespace,
    shared: dict,
) -> Dict:
    device = shared["device"]
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_backbone_checkpoint(
        name=name,
        ckpt_path=Path(args.ckpt_root) / name / "encoder_best.pth",
        msg_dim=shared["msg_dim"],
        profile_dim=shared["profile_dim"],
        item_emb_table=shared["item_emb"],
        num_classes=int(args.num_classes),
        embed_dim=int(args.embed_dim),
        device=device,
    )
    model = MultitaskFinetuneModel(
        backbone=backbone,
        num_classes=int(args.num_classes),
        embed_dim=int(args.embed_dim),
        msg_dim=shared["msg_dim"],
        num_neg=int(args.num_neg),
        neg_reduction=str(args.neg_reduction),
    ).to(device)

    train_labels = [int(r.label_id) for r in shared["records"]["train"]]
    class_w = compute_class_weights(train_labels, int(args.num_classes)).to(device)
    criterion = FocalLoss(alpha=class_w, gamma=float(args.focal_gamma))

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = (not bool(args.no_amp)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = 1e9
    best_state = None
    history = []
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        tot_cls = tot_rec = tot = 0.0
        n = 0
        t0 = time.time()
        for step, batch in enumerate(shared["train_loader"], start=1):
            b = move_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                cls_loss, rec_loss, _ = model.compute_losses(b, criterion)
                loss = float(args.cls_weight) * cls_loss + float(args.rec_weight) * rec_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = int(b["seq_ids"].size(0))
            tot_cls += float(cls_loss.item()) * bs
            tot_rec += float(rec_loss.item()) * bs
            tot += float(loss.item()) * bs
            n += bs
            if step % max(1, int(args.log_every_batches)) == 0:
                print(
                    f"[{name}][ep{ep}] step={step}/{len(shared['train_loader'])} "
                    f"loss={tot/max(1,n):.4f} cls={tot_cls/max(1,n):.4f} rec={tot_rec/max(1,n):.4f}",
                    flush=True,
                )

        # val loss
        model.eval()
        v_tot = v_n = 0.0
        with torch.no_grad():
            for batch in shared["val_loader"]:
                b = move_to_device(batch, device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    cls_loss, rec_loss, _ = model.compute_losses(b, criterion)
                    loss = float(args.cls_weight) * cls_loss + float(args.rec_weight) * rec_loss
                bs = int(b["seq_ids"].size(0))
                v_tot += float(loss.item()) * bs
                v_n += bs
        tr = tot / max(1, n)
        va = v_tot / max(1.0, v_n)
        test_head = eval_classification_head(model, shared["test_loader"], device)
        row = {
            "epoch": ep,
            "train_loss": tr,
            "val_loss": va,
            "test_head": test_head,
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)
        print(
            f"[{name}] epoch={ep} train={tr:.4f} val={va:.4f} "
            f"test_hit@1={test_head['hit@1']:.4f} ndcg@3={test_head['ndcg@3']:.4f}",
            flush=True,
        )
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("backbone.item_embedding.")}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    ckpt_out = out_dir / "multitask_finetune_best.pth"
    torch.save(
        {
            "model": name,
            "state_dict": {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.item_embedding.")},
            "config": vars(args),
            "best_val_loss": float(best_val),
            "cls_weight": float(args.cls_weight),
            "rec_weight": float(args.rec_weight),
        },
        ckpt_out,
    )

    # export embeddings on all splits for downstream
    emb_all: Dict[str, torch.Tensor] = {}
    for split_name, loader in [
        ("train", shared["train_loader"]),
        ("val", shared["val_loader"]),
        ("test", shared["test_loader"]),
    ]:
        part = export_embeddings(model, loader, device)
        emb_all.update(part)
        print(f"[{name}] exported {split_name}={len(part)}", flush=True)

    emb_path = out_dir / f"taskA_user_embeddings__{name}_MTFT__256d.pt"
    torch.save({"embeddings": emb_all, "meta": {"model": f"{name}_MTFT", "dim": int(args.embed_dim)}}, emb_path)

    test_head_final = eval_classification_head(model, shared["test_loader"], device)
    result = {
        "status": "ok",
        "checkpoint": str(ckpt_out),
        "embedding_file": str(emb_path),
        "best_val_loss": float(best_val),
        "test_multitask_head": test_head_final,
        "history": history,
        "loss_cfg": {
            "cls_weight": float(args.cls_weight),
            "rec_weight": float(args.rec_weight),
            "focal_gamma": float(args.focal_gamma),
            "num_neg": int(args.num_neg),
            "neg_reduction": str(args.neg_reduction),
        },
    }
    (out_dir / "exp1_finetune_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_downstream_evals(args: argparse.Namespace, emb_files: List[str]) -> Dict:
    """Run Task A classification and Task B interaction probes."""
    out: Dict[str, str] = {}
    cls_out = Path(args.output_dir) / "downstream_user_classification_MTFT.json"
    inter_out = Path(args.output_dir) / "downstream_interaction_binary_MTFT.json"
    index_json = Path(args.output_dir) / "embedding_index_MTFT.json"
    index = {}
    for fp in emb_files:
        name = Path(fp).stem.replace("taskA_user_embeddings__", "")
        index[name] = {"file": fp}
    index_json.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # Task A
    cmd_cls = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "downstream" / "taskA" / "run_classification.py"),
        "--samples-csv",
        args.samples_csv,
        "--embedding-files",
        ",".join(emb_files),
        "--seeds",
        args.seeds,
        "--probe-label-smoothing",
        "0.1",
        "--restrict-split",
        "test",
        "--device",
        args.device,
        "--output-json",
        str(cls_out),
    ]
    # Task B
    cmd_inter = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "downstream" / "taskB" / "run_interaction.py"),
        "--samples-dir",
        args.samples_dir,
        "--embedding-index-json",
        str(index_json),
        "--dynamic-embedding",
        "NONE",
        "--tweet-emb-shards-dir",
        args.tweet_emb_shards_dir,
        "--seeds",
        args.seeds,
        "--epochs",
        str(int(args.downstream_inter_epochs)),
        "--batch-size",
        str(int(args.downstream_batch_size)),
        "--device",
        args.device,
        "--pretrain-split",
        "test",
        "--taska-samples-csv",
        args.samples_csv,
        "--max-pos-per-window-record",
        "10",
        "--max-neg-per-window-record",
        "10",
        "--output-json",
        str(inter_out),
    ]

    import subprocess

    print("[Exp1] running Task A classification...", flush=True)
    subprocess.check_call(cmd_cls)
    out["classification_json"] = str(cls_out)
    print("[Exp1] running Task B interaction...", flush=True)
    subprocess.check_call(cmd_inter)
    out["interaction_json"] = str(inter_out)
    return out


def build_shared_context(args: argparse.Namespace) -> dict:
    set_seed(int(args.seed))
    device = torch.device(
        args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Exp1] device={device}", flush=True)

    records = build_records_by_split(
        samples_csv=args.samples_csv,
        max_train=int(args.max_train),
        max_val=int(args.max_val),
        max_test=int(args.max_test),
    )
    all_users = {r.user_id for s in ["train", "val", "test"] for r in records[s]}
    user_tweet_ids_map = load_or_build_user_tweet_ids_map(
        labels_cache=args.labels_cache,
        users=all_users,
        cache_dir=str(args.preprocess_cache_dir),
        verbose=True,
    )
    needed = set()
    for s in ["train", "val", "test"]:
        for r in records[s]:
            ordered = user_tweet_ids_map.get(r.user_id, [])
            st = int(r.window_idx) * int(args.window_size)
            needed.update(str(x) for x in ordered[st : st + int(args.seq_len)])
    print(f"[Exp1] loading msg feats needed={len(needed)}", flush=True)
    msg_mat, id2row, msg_dim = load_msg_features_matrix_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir, needed_msg_ids=needed, verbose=True
    )
    user_feat_map, profile_dim = load_user_features_from_pt(args.user_profile_pt)
    ids_set = set()
    for r in records["train"]:
        ordered = user_tweet_ids_map.get(r.user_id, [])
        st = int(r.window_idx) * int(args.window_size)
        for mid in ordered[st : st + int(args.seq_len)]:
            s = str(mid)
            if s in id2row:
                ids_set.add(s)
    sorted_ids = sorted(ids_set)
    id2idx = {s: i + 1 for i, s in enumerate(sorted_ids)}
    emb_mat = torch.zeros((len(sorted_ids) + 1, int(msg_dim)), dtype=torch.float32)
    for i, s in enumerate(sorted_ids):
        emb_mat[i + 1] = msg_mat[id2row[s]]
    item_emb = nn.Embedding.from_pretrained(emb_mat, freeze=True, padding_idx=0)
    del msg_mat, id2row, emb_mat
    import gc

    gc.collect()

    def _mk_loader(split: str, shuffle: bool) -> DataLoader:
        ds = UnifiedWindowDataset(
            records[split],
            user_tweet_ids_map,
            user_feat_map,
            int(profile_dim),
            int(args.seq_len),
            int(args.window_size),
            id2idx,
            verbose=True,
        )
        return DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
        )

    return {
        "device": device,
        "records": records,
        "msg_dim": int(msg_dim),
        "profile_dim": int(profile_dim),
        "item_emb": item_emb,
        "train_loader": _mk_loader("train", True),
        "val_loader": _mk_loader("val", False),
        "test_loader": _mk_loader("test", False),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Exp1: multitask finetune MIND & PTUM (L_cls + 1.5 L_rec)")
    p.add_argument("--models", type=str, default="MIND,PTUM")
    p.add_argument("--ckpt-root", type=str, default="outputs/baselines")
    p.add_argument("--samples-csv", type=str, default="dataset/ids/windows.csv")
    p.add_argument(
        "--labels-cache",
        type=str,
        default="dataset/local/labels_cache.json",
        help="， preprocess-cache-dir  user_tweet_ids_map ",
    )
    p.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    p.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    p.add_argument("--preprocess-cache-dir", type=str, default="cache")
    p.add_argument("--samples-dir", type=str, default="dataset/local/message_user_samples")
    p.add_argument("--output-dir", type=str, default="outputs/exp1_multitask_finetune")
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--num-classes", type=int, default=15)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--cls-weight", type=float, default=1.0)
    p.add_argument("--rec-weight", type=float, default=1.5, help=" ExpConfig.rec_weight ")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--num-neg", type=int, default=5)
    p.add_argument("--neg-reduction", type=str, default="mean", choices=["mean", "sum"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="42,52,62", help=" seeds")
    p.add_argument("--log-every-batches", type=int, default=200)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--skip-downstream", action="store_true")
    p.add_argument("--downstream-inter-epochs", type=int, default=5)
    p.add_argument("--downstream-batch-size", type=int, default=1024)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    shared = build_shared_context(args)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    models = ["PTUM" if m.upper() == "PUTM" else m for m in models]
    summary = {"models": {}, "loss": {"cls_weight": args.cls_weight, "rec_weight": args.rec_weight}}
    emb_files: List[str] = []

    prev_summary_path = Path(args.output_dir) / "summary.json"
    if prev_summary_path.exists():
        try:
            prev = json.loads(prev_summary_path.read_text(encoding="utf-8"))
            for k, v in (prev.get("models") or {}).items():
                if k in models or not isinstance(v, dict):
                    continue
                fp = v.get("embedding_file")
                if fp and Path(fp).exists():
                    summary["models"][k] = v
                    emb_files.append(str(fp))
                    print(f"[Exp1] keep previous model result: {k}", flush=True)
        except Exception as e:
            print(f"[Exp1] warn: failed to merge previous summary: {e}", flush=True)
    for d in sorted(Path(args.output_dir).glob("*")):
        if not d.is_dir() or d.name in summary["models"] or d.name in models:
            continue
        mets = d / "exp1_finetune_metrics.json"
        if not mets.exists():
            continue
        try:
            v = json.loads(mets.read_text(encoding="utf-8"))
            fp = v.get("embedding_file")
            if fp and Path(fp).exists():
                summary["models"][d.name] = v
                emb_files.append(str(fp))
                print(f"[Exp1] keep finished dir result: {d.name}", flush=True)
        except Exception as e:
            print(f"[Exp1] warn: skip {d.name}: {e}", flush=True)

    for name in models:
        summary["models"][name] = finetune_one_model(name, args, shared)
        emb_files.append(summary["models"][name]["embedding_file"])

    seen = set()
    emb_files = [x for x in emb_files if not (x in seen or seen.add(x))]

    if not args.skip_downstream:
        summary["downstream"] = run_downstream_evals(args, emb_files)

    out = Path(args.output_dir) / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Exp1 summary -> {out}", flush=True)


if __name__ == "__main__":
    main()
