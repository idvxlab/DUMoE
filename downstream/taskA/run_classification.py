#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_samples(csv_path: str) -> Dict[str, Tuple[int, str]]:
    out: Dict[str, Tuple[int, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out[str(row["user_id_t"])] = (int(row["label_id"]), str(row.get("split_tag", "")))
    return out


def load_emb_dict(pt_path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise RuntimeError(f"{pt_path}  embeddings ")

    emb_obj = payload["embeddings"]
    out: Dict[str, torch.Tensor] = {}

    if isinstance(emb_obj, dict):
        for k, v in emb_obj.items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            out[str(k)] = t.detach().cpu()
        return out

    if isinstance(emb_obj, torch.Tensor):
        user_ids = payload.get("user_ids")
        window_indices = payload.get("window_indices")
        if (
            isinstance(user_ids, list)
            and isinstance(window_indices, list)
            and len(user_ids) == emb_obj.size(0)
            and len(window_indices) == emb_obj.size(0)
        ):
            for i, (uid, widx) in enumerate(zip(user_ids, window_indices)):
                out[f"{uid}__w{int(widx):03d}"] = emb_obj[i].detach().cpu()
            return out

    raise RuntimeError(f"{pt_path}  embeddings ")


def collect_embedding_files(summary_json: str, embedding_files: str) -> List[str]:
    files: List[str] = []
    if summary_json.strip():
        payload = json.loads(Path(summary_json).read_text(encoding="utf-8"))
        models = payload.get("models", {})
        if isinstance(models, dict) and not models and isinstance(payload, dict):
            models = {
                k: v
                for k, v in payload.items()
                if isinstance(v, dict) and (v.get("embedding_file") or v.get("file"))
            }
        if isinstance(models, dict):
            for _, meta in models.items():
                if isinstance(meta, dict):
                    p = str(meta.get("embedding_file") or meta.get("file", "")).strip()
                    if p:
                        files.append(p)
    if embedding_files.strip():
        files.extend([x.strip() for x in embedding_files.split(",") if x.strip()])

    out: List[str] = []
    seen = set()
    for p in files:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, int(n_cls)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dcg_at_rank(rank_1based: int) -> float:
    return 1.0 / np.log2(float(rank_1based) + 1.0)


def kl_divergence_vs_label_smoothing(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """ ``time_slice_train.compute_metrics``  ``kl_div`` 。

    ``PreferenceModelMultiTask``（``model_multitask.py``）：
    ``target = one_hot * 0.9 + 0.1/C``， ``F.kl_div(log(probs), target, reduction="batchmean")``。
    """
    target_dist = F.one_hot(labels, int(num_classes)).float()
    target_dist = target_dist * 0.9 + 0.1 / float(num_classes)
    return float(F.kl_div(torch.log(probs + 1e-12), target_dist, reduction="batchmean").item())


def compute_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if y.numel() == 0:
        return {
            "hit@1": 0.0,
            "hit@3": 0.0,
            "ndcg@1": 0.0,
            "ndcg@3": 0.0,
            "recall": 0.0,
            "kl": 0.0,
        }

    probs = torch.softmax(logits, dim=1)
    n_cls = int(logits.size(1))

    # Hit@K / NDCG@K
    top1 = torch.topk(logits, k=1, dim=1).indices
    top3 = torch.topk(logits, k=min(3, n_cls), dim=1).indices
    y_col = y.view(-1, 1)

    hit1 = float((top1 == y_col).any(dim=1).float().mean().item())
    hit3_mask = (top3 == y_col).any(dim=1)
    hit3 = float(hit3_mask.float().mean().item())

    ndcg1 = hit1
    ndcg3_vals: List[float] = []
    for i in range(y.size(0)):
        yi = int(y[i].item())
        row = top3[i]
        match = (row == yi).nonzero(as_tuple=False)
        if match.numel() == 0:
            ndcg3_vals.append(0.0)
        else:
            rank_1based = int(match[0, 0].item()) + 1
            ndcg3_vals.append(_dcg_at_rank(rank_1based))
    ndcg3 = float(np.mean(ndcg3_vals))

    # Recall: macro recall over classes present in y
    pred = logits.argmax(dim=1)
    recalls = []
    for c in y.unique().tolist():
        c = int(c)
        gt_mask = y.eq(c)
        denom = int(gt_mask.sum().item())
        if denom == 0:
            continue
        tp = int((pred.eq(c) & gt_mask).sum().item())
        recalls.append(float(tp) / float(denom))
    recall_macro = float(np.mean(recalls)) if recalls else 0.0

    kl = kl_divergence_vs_label_smoothing(probs, y, n_cls)

    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "ndcg@1": ndcg1,
        "ndcg@3": ndcg3,
        "recall": recall_macro,
        "kl": kl,
    }


def train_and_eval_one_seed(
    x: torch.Tensor,
    y: torch.Tensor,
    split_tags: List[str],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    force_random_split: bool = False,
    probe_label_smoothing: float = 0.0,
) -> Dict[str, float]:
    set_seed(seed)

    idx_train = [i for i, s in enumerate(split_tags) if s == "train"]
    idx_val = [i for i, s in enumerate(split_tags) if s == "val"]
    idx_test = [i for i, s in enumerate(split_tags) if s == "test"]
    if bool(force_random_split) or min(len(idx_train), len(idx_val), len(idx_test)) == 0:
        # Some exported embedding files (e.g., export_split=test) only overlap one split.
        # Fallback to a deterministic random split per seed for a runnable comparison.
        n = int(x.size(0))
        if n < 10:
            raise RuntimeError("，。")
        perm = torch.randperm(n)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        idx_train = perm[:n_train].tolist()
        idx_val = perm[n_train : n_train + n_val].tolist()
        idx_test = perm[n_train + n_val : n_train + n_val + n_test].tolist()

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    n_cls = int(y.max().item() + 1)
    model = MLP(int(x.size(1)), n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    ls = float(probe_label_smoothing)
    if ls < 0.0 or ls >= 1.0:
        raise ValueError("--probe-label-smoothing  [0, 1) ")
    crit = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0.0 else nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(batch_size), shuffle=True)

    best_val = -1.0
    best_state = None
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.to(device)).cpu()
            val_metrics = compute_metrics(val_logits, y_val)
            val_score = val_metrics["ndcg@3"]  #  rank-aware  best epoch
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test.to(device)).cpu()
    metrics = compute_metrics(test_logits, y_test)
    metrics["num_train"] = float(len(idx_train))
    metrics["num_val"] = float(len(idx_val))
    metrics["num_test"] = float(len(idx_test))
    return metrics


def aggregate_seed_metrics(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = ["hit@1", "hit@3", "ndcg@1", "ndcg@3", "recall", "kl"]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="3-seed downstream user classification compare with ranking/distribution metrics")
    p.add_argument("--samples-csv", type=str, default="dataset/ids/windows.csv")
    p.add_argument("--summary-json", type=str, default="", help="optional: outputs_full/summary.json")
    p.add_argument("--embedding-files", type=str, default="", help="optional: comma-separated embedding .pt files")
    p.add_argument("--dynamic-embedding", type=str, default="", help="dynamic model embedding .pt")
    p.add_argument(
        "--dynamic-embedding-first",
        action="store_true",
        help=" --dynamic-embedding  embedding （ JSON  Dynamic ）",
    )
    p.add_argument("--seeds", type=str, default="42,52,62", help="comma-separated random seeds")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--restrict-split",
        type=str,
        default="",
        choices=["", "train", "val", "test"],
        help=" split_tag ； test  --force-random-split， train/val  80/10/10 ",
    )
    p.add_argument(
        "--force-random-split",
        action="store_true",
        help="force random 80/10/10 split (old protocol), ignore split_tag partition",
    )
    p.add_argument(
        "--probe-label-smoothing",
        type=float,
        default=0.0,
        help=(
            " MLP  CrossEntropy label_smoothing； 0.1  KL  "
            "target_dist=one_hot*0.9+0.1/C （ time_slice_train / ），"
            " Dynamic  KL 。 0 。"
        ),
    )
    p.add_argument("--output-json", type=str, required=True)
    clean_argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    args = p.parse_args(clean_argv)

    files = collect_embedding_files(args.summary_json, args.embedding_files)
    if args.dynamic_embedding.strip():
        de = args.dynamic_embedding.strip()
        if bool(args.dynamic_embedding_first):
            files.insert(0, de)
        else:
            files.append(de)
    if not files:
        raise RuntimeError(" embedding ， --summary-json  --embedding-files/--dynamic-embedding")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise RuntimeError("seeds ")

    sample_map = load_samples(args.samples_csv)
    results: Dict[str, Dict] = {}

    for fp in files:
        name = Path(fp).stem.replace("taskA_user_embeddings__", "")
        emb = load_emb_dict(fp)
        keys = [k for k in emb.keys() if k in sample_map]
        if str(args.restrict_split):
            wanted = str(args.restrict_split)
            keys = [k for k in keys if sample_map[k][1] == wanted]
        if len(keys) == 0:
            results[name] = {"status": "no_overlap"}
            print(f"[{name}] no overlap")
            continue

        x = torch.stack([emb[k].float().view(-1) for k in keys], dim=0)
        y = torch.tensor([sample_map[k][0] for k in keys], dtype=torch.long)
        split_tags = [sample_map[k][1] for k in keys]

        seed_rows: List[Dict[str, float]] = []
        for sd in seeds:
            row = train_and_eval_one_seed(
                x=x,
                y=y,
                split_tags=split_tags,
                seed=sd,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                device=args.device,
                force_random_split=bool(args.force_random_split),
                probe_label_smoothing=float(args.probe_label_smoothing),
            )
            seed_rows.append(row)
            print(
                f"[{name}] seed={sd} "
                f"hit@1={row['hit@1']:.4f} hit@3={row['hit@3']:.4f} "
                f"ndcg@1={row['ndcg@1']:.4f} ndcg@3={row['ndcg@3']:.4f} "
                f"recall={row['recall']:.4f} kl={row['kl']:.6f}"
            )

        meta: Dict[str, object] = {
            "status": "ok",
            "num_overlap": len(keys),
            "seeds": seeds,
            "per_seed_test": seed_rows,
            "aggregate_test": aggregate_seed_metrics(seed_rows),
        }
        if str(args.restrict_split):
            meta["restrict_split"] = str(args.restrict_split)
            meta["downstream_note"] = (
                " restrict_split  force_random_split，"
                "train_and_eval_one_seed  80/10/10  train/val/test。"
            )
        if float(args.probe_label_smoothing) > 0.0:
            meta["probe_label_smoothing"] = float(args.probe_label_smoothing)
            meta["kl_eval_note"] = (
                "KL  kl_divergence_vs_label_smoothing（one_hot*0.9+0.1/C）；"
                " CE，。"
            )
        results[name] = meta

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()

