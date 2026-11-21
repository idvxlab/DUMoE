#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3： tweet  UNITE （）。

：
   user_id_t， long_recent(=300)  tweet  UNITE  mean。
   1536（ 1532 ）。

：
  1) ：Linear(1536→256)+MLP 
  2) ：user Linear(1536→256)  tweet concat， BinaryMLP
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CUR_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CUR_DIR.parent
PARENT_DIR = PROJECT_DIR.parent
REPO_ROOT = PARENT_DIR.parent
for _p in (str(REPO_ROOT), str(PARENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from project.data.processor import (  # noqa: E402
    build_records_by_split,
    load_or_build_user_tweet_ids_map,
)
from project.data.local_feature_loaders import (  # noqa: E402
    load_msg_features_matrix_from_compact_shards,
)
from downstream.taskA.run_classification import (  # noqa: E402
    aggregate_seed_metrics,
    compute_metrics,
    load_samples,
)
from downstream.taskB.run_interaction import (  # noqa: E402
    aggregate,
    auc_binary,
    binary_metrics_from_logits,
    build_tweet_bank_and_ix,
    downstream_train_val_test_indices,
    load_message_samples,
    load_taska_split_tags,
    load_tweet_features,
    ranking_metrics_at_k,
)


class DimReduceProbe(nn.Module):
    def __init__(self, in_dim: int, n_cls: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(int(in_dim), int(hidden)), nn.ReLU())
        self.cls = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(n_cls)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.reduce(x))


class InteractionWithUserReduce(nn.Module):
    def __init__(self, user_dim: int, tweet_dim: int, user_out_dim: int = 256, tweet_reduce_dim: int = 0):
        super().__init__()
        self.user_proj = nn.Sequential(nn.Linear(int(user_dim), int(user_out_dim)), nn.ReLU())
        rd = int(tweet_reduce_dim)
        if rd > 0:
            self.tweet_proj = nn.Linear(int(tweet_dim), rd)
            head_in = int(user_out_dim) + rd
        else:
            self.tweet_proj = None
            head_in = int(user_out_dim) + int(tweet_dim)
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, user_x: torch.Tensor, tweet_x: torch.Tensor) -> torch.Tensor:
        u = self.user_proj(user_x)
        t = self.tweet_proj(tweet_x) if self.tweet_proj is not None else tweet_x
        return self.head(torch.cat([u, t], dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_unite_mean_embeddings(
    records_by_split: Dict[str, list],
    user_tweet_ids_map: Dict[str, List[str]],
    msg_mat: torch.Tensor,
    id2row: Dict[str, int],
    msg_dim: int,
    window_size: int,
    long_recent: int,
    log_every: int = 5000,
) -> Dict[str, torch.Tensor]:
    emb: Dict[str, torch.Tensor] = {}
    msg_mat = msg_mat.float()
    for split, recs in records_by_split.items():
        n_recs = len(recs)
        for i, r in enumerate(recs, start=1):
            ordered = user_tweet_ids_map.get(r.user_id, [])
            st = int(r.window_idx) * int(window_size)
            mids = ordered[st : st + int(long_recent)]
            rows: List[int] = []
            for mid in mids:
                j = id2row.get(str(mid))
                if j is not None:
                    rows.append(int(j))
            if not rows:
                emb[r.key] = torch.zeros(int(msg_dim), dtype=torch.float32)
            else:
                emb[r.key] = msg_mat[rows].mean(dim=0)
            if log_every > 0 and (i % int(log_every) == 0 or i == n_recs):
                print(
                    f"[Exp3] building unite-mean split={split} {i}/{n_recs} keys_so_far={len(emb)}",
                    flush=True,
                )
        print(f"[Exp3] built unite-mean split={split} keys_so_far={len(emb)}", flush=True)
    return emb


def _split_indices_like_task1(
    split_tags: List[str],
    n: int,
    seed: int,
    force_random: bool,
) -> Tuple[List[int], List[int], List[int]]:
    idx_train = [i for i, s in enumerate(split_tags) if s == "train"]
    idx_val = [i for i, s in enumerate(split_tags) if s == "val"]
    idx_test = [i for i, s in enumerate(split_tags) if s == "test"]
    if force_random or min(len(idx_train), len(idx_val), len(idx_test)) == 0:
        set_seed(seed)
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
    return idx_train, idx_val, idx_test


def train_eval_classification(
    emb: Dict[str, torch.Tensor],
    sample_map: Dict[str, Tuple[int, str]],
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    restrict_split: str,
    probe_label_smoothing: float,
) -> Dict:
    keys = [k for k in emb.keys() if k in sample_map]
    if restrict_split:
        keys = [k for k in keys if sample_map[k][1] == restrict_split]
    if not keys:
        return {"status": "no_overlap"}

    x = torch.stack([emb[k].float().view(-1) for k in keys], dim=0)
    y = torch.tensor([sample_map[k][0] for k in keys], dtype=torch.long)
    split_tags = [sample_map[k][1] for k in keys]
    in_dim = int(x.size(1))
    n_cls = int(y.max().item()) + 1

    seed_rows: List[Dict[str, float]] = []
    for sd in seeds:
        idx_train, idx_val, idx_test = _split_indices_like_task1(
            split_tags=split_tags,
            n=int(x.size(0)),
            seed=int(sd),
            force_random=bool(restrict_split),  #  80/10/10
        )
        x_tr, y_tr = x[idx_train], y[idx_train]
        x_va, y_va = x[idx_val], y[idx_val]
        x_te, y_te = x[idx_test], y[idx_test]

        set_seed(int(sd))
        model = DimReduceProbe(in_dim, n_cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-2)
        ls = float(probe_label_smoothing)
        crit = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=int(batch_size), shuffle=True)

        best_val = -1.0
        best_state = None
        for _ in range(int(epochs)):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_m = compute_metrics(model(x_va.to(device)).cpu(), y_va)
            if float(val_m["ndcg@3"]) > best_val:
                best_val = float(val_m["ndcg@3"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        model.eval()
        with torch.no_grad():
            row = compute_metrics(model(x_te.to(device)).cpu(), y_te)
        row["num_train"] = float(len(idx_train))
        row["num_val"] = float(len(idx_val))
        row["num_test"] = float(len(idx_test))
        seed_rows.append(row)
        print(
            f"[Exp3][cls] seed={sd} hit@1={row['hit@1']:.4f} hit@3={row['hit@3']:.4f} "
            f"ndcg@3={row['ndcg@3']:.4f} recall={row['recall']:.4f}",
            flush=True,
        )

    return {
        "status": "ok",
        "user_feat_dim": in_dim,
        "reduce_to": 256,
        "num_overlap": len(keys),
        "seeds": list(seeds),
        "per_seed_test": seed_rows,
        "aggregate_test": aggregate_seed_metrics(seed_rows),
        "restrict_split": restrict_split,
        "probe_label_smoothing": float(probe_label_smoothing),
    }


def train_eval_interaction(
    emb: Dict[str, torch.Tensor],
    samples_dir: str,
    tweet_emb_shards_dir: str,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_pos: int,
    max_neg: int,
    tweet_reduce_dim: int,
    taska_samples_csv: str = "",
    pretrain_split: str = "",
) -> Dict:
    allowed = set(emb.keys())
    split_filter_meta: Dict = {}
    csv_path = str(taska_samples_csv or "").strip()
    want_split = str(pretrain_split or "").strip()
    if (csv_path and not want_split) or (want_split and not csv_path):
        raise RuntimeError("taska_samples_csv  pretrain_split ")
    if csv_path and want_split:
        tag_map = load_taska_split_tags(csv_path)
        before = len(allowed)
        allowed = {k for k in allowed if tag_map.get(str(k)) == want_split}
        split_filter_meta = {
            "taska_samples_csv": csv_path,
            "pretrain_split": want_split,
            "keys_before_split_filter": int(before),
            "keys_after_split_filter": int(len(allowed)),
        }
        print(
            f"[Exp3][PretrainSplit] split={want_split} keys {before} -> {len(allowed)} "
            f"(same as Exp1 Task B; user-disjoint train/val/test)",
            flush=True,
        )
        if not allowed:
            return {"status": "no_keys_after_pretrain_split", "pretrain_split_filter": split_filter_meta}

    triples, needed_tweet_ids, cap_meta = load_message_samples(
        samples_dir=samples_dir,
        allowed_user_keys=allowed,
        max_files=0,
        max_pos_per_window_record=int(max_pos),
        max_neg_per_window_record=int(max_neg),
        per_window_cap_seed=42,
    )
    if not triples:
        return {"status": "no_overlap", "cap_meta": cap_meta, "pretrain_split_filter": split_filter_meta}

    print(f"[Exp3] interaction triples={len(triples)} tweets={len(needed_tweet_ids)}", flush=True)
    tweet_feat_map = load_tweet_features(needed_tweet_ids, tweet_emb_shards_dir)

    # drop triples missing tweet feats
    valid = [(uk, tid, lb) for uk, tid, lb in triples if str(tid) in tweet_feat_map]
    user_keys = [a for a, _, _ in valid]
    tweet_ids = [b for _, b, _ in valid]
    labels = torch.tensor([c for _, _, c in valid], dtype=torch.long)
    user_x = torch.stack([emb[k].float().view(-1) for k in user_keys], dim=0)
    tweet_bank, tweet_ix = build_tweet_bank_and_ix(tweet_feat_map, tweet_ids)

    seed_rows: List[Dict[str, float]] = []
    for sd in seeds:
        set_seed(int(sd))
        idx_tr, idx_va, idx_te, split_meta = downstream_train_val_test_indices(
            user_keys=user_keys, seed=int(sd), split_mode="user"
        )
        model = InteractionWithUserReduce(
            user_dim=int(user_x.size(1)),
            tweet_dim=int(tweet_bank.size(1)),
            user_out_dim=256,
            tweet_reduce_dim=int(tweet_reduce_dim),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-2)

        def _train_epoch(indices: List[int]) -> float:
            model.train()
            order = list(indices)
            random.shuffle(order)
            total = 0.0
            n = 0
            for i0 in range(0, len(order), int(batch_size)):
                bi = torch.tensor(order[i0 : i0 + int(batch_size)], dtype=torch.long)
                ux = user_x[bi].to(device)
                tx = tweet_bank[tweet_ix[bi]].to(device)
                yb = labels[bi].float().to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(ux, tx), yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * int(bi.numel())
                n += int(bi.numel())
            return total / max(1, n)

        @torch.no_grad()
        def _eval(indices: List[int]) -> Dict[str, float]:
            model.eval()
            logits_all = []
            y_all = []
            for i0 in range(0, len(indices), int(batch_size)):
                bi = torch.tensor(indices[i0 : i0 + int(batch_size)], dtype=torch.long)
                ux = user_x[bi].to(device)
                tx = tweet_bank[tweet_ix[bi]].to(device)
                logits_all.append(model(ux, tx).detach().cpu())
                y_all.append(labels[bi])
            logits = torch.cat(logits_all, dim=0)
            y = torch.cat(y_all, dim=0)
            m = binary_metrics_from_logits(logits, y)
            probs = torch.sigmoid(logits)
            m["auc"] = auc_binary(y, probs)
            rk = ranking_metrics_at_k(
                user_keys=[user_keys[i] for i in indices],
                y_true=y,
                scores=probs,
                k=10,
            )
            m.update(rk)
            return m

        best_val = -1.0
        best_state = None
        for ep in range(1, int(epochs) + 1):
            tr = _train_epoch(idx_tr)
            val_m = _eval(idx_va)
            print(
                f"[Exp3][inter] seed={sd} ep={ep} train_loss={tr:.4f} "
                f"val_f1={val_m['f1']:.4f} val_auc={val_m['auc']:.4f}",
                flush=True,
            )
            if float(val_m["f1"]) > best_val:
                best_val = float(val_m["f1"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        test_m = _eval(idx_te)
        test_m["num_train"] = float(len(idx_tr))
        test_m["num_val"] = float(len(idx_va))
        test_m["num_test"] = float(len(idx_te))
        test_m["downstream_split_meta"] = split_meta
        seed_rows.append(test_m)
        print(
            f"[Exp3][inter] seed={sd} TEST f1={test_m['f1']:.4f} auc={test_m['auc']:.4f} "
            f"ndcg@10={test_m['ndcg@10']:.4f}",
            flush=True,
        )

    out = {
        "status": "ok",
        "user_feat_dim": int(user_x.size(1)),
        "user_reduce_to": 256,
        "tweet_dim": int(tweet_bank.size(1)),
        "tweet_reduce_dim": int(tweet_reduce_dim),
        "num_samples_used": len(valid),
        "num_unique_tweets": int(tweet_bank.size(0)),
        "num_common_user_keys": len(set(user_keys)),
        "cap_meta": cap_meta,
        "seeds": list(seeds),
        "per_seed_test": seed_rows,
        "aggregate_test": aggregate(seed_rows),
    }
    if split_filter_meta:
        out["pretrain_split_filter"] = split_filter_meta
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Exp3: Unite-mean long-history features for downstream")
    p.add_argument("--samples-csv", type=str, default="dataset/ids/windows.csv")
    p.add_argument(
        "--labels-cache",
        type=str,
        default="dataset/local/labels_cache.json",
    )
    p.add_argument(
        "--tweet-emb-shards-dir",
        type=str,
        default="dataset/local/embeddings_pt",
    )
    p.add_argument(
        "--preprocess-cache-dir",
        type=str,
        default="cache",
    )
    p.add_argument(
        "--samples-dir",
        type=str,
        default="dataset/local/message_user_samples",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp3_unite_mean",
    )
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--long-recent", type=int, default=300)
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--cls-epochs", type=int, default=20)
    p.add_argument("--inter-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--inter-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--restrict-split", type=str, default="test", choices=["", "train", "val", "test"])
    p.add_argument("--probe-label-smoothing", type=float, default=0.1)
    p.add_argument("--max-pos-per-window-record", type=int, default=10)
    p.add_argument("--max-neg-per-window-record", type=int, default=10)
    p.add_argument("--tweet-reduce-dim", type=int, default=0)
    p.add_argument(
        "--taska-samples-csv",
        type=str,
        default="dataset/ids/windows.csv",
        help="Same as Exp1 Task B: restrict keys via --pretrain-split",
    )
    p.add_argument(
        "--pretrain-split",
        type=str,
        default="test",
        help=" TaskA  split  user__wXXX；（）",
    )
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--skip-classification", action="store_true")
    p.add_argument("--skip-interaction", action="store_true")
    p.add_argument("--reuse-emb-pt", type=str, default="", help=" unite-mean .pt ")
    p.add_argument("--force-rebuild-emb", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    emb_path = Path(args.reuse_emb_pt.strip()) if args.reuse_emb_pt.strip() else (out_dir / "unite_mean_embeddings__1536d.pt")

    if emb_path.exists() and not bool(args.force_rebuild_emb):
        obj = torch.load(str(emb_path), map_location="cpu", weights_only=False)
        emb = obj["embeddings"] if isinstance(obj, dict) and "embeddings" in obj else obj
        print(f"[Exp3] loaded embeddings: {emb_path} keys={len(emb)}", flush=True)
    else:
        records = build_records_by_split(
            samples_csv=args.samples_csv,
            max_train=int(args.max_train),
            max_val=int(args.max_val),
            max_test=int(args.max_test),
        )
        all_users = {r.user_id for recs in records.values() for r in recs}
        user_tweet_ids_map = load_or_build_user_tweet_ids_map(
            labels_cache=args.labels_cache,
            users=all_users,
            cache_dir=str(args.preprocess_cache_dir),
            verbose=True,
        )
        needed: Set[str] = set()
        for s in records:
            for r in records[s]:
                ordered = user_tweet_ids_map.get(r.user_id, [])
                st = int(r.window_idx) * int(args.window_size)
                needed.update(str(x) for x in ordered[st : st + int(args.long_recent)])
        print(f"[Exp3] loading msg feats needed={len(needed)}", flush=True)
        msg_mat, id2row, msg_dim = load_msg_features_matrix_from_compact_shards(
            shards_dir=args.tweet_emb_shards_dir, needed_msg_ids=needed, verbose=True
        )
        if needed and len(id2row) < len(needed):
            miss = len(needed) - len(id2row)
            print(f"⚠️  {miss} ", flush=True)
        emb = build_unite_mean_embeddings(
            records_by_split=records,
            user_tweet_ids_map=user_tweet_ids_map,
            msg_mat=msg_mat,
            id2row=id2row,
            msg_dim=int(msg_dim),
            window_size=int(args.window_size),
            long_recent=int(args.long_recent),
        )
        del msg_mat, id2row
        import gc

        gc.collect()
        torch.save(
            {
                "embeddings": emb,
                "meta": {
                    "dim": int(msg_dim),
                    "long_recent": int(args.long_recent),
                    "window_size": int(args.window_size),
                    "note": "mean of Unite tweet features over long history",
                },
            },
            emb_path,
        )
        print(f"[Exp3] saved embeddings: {emb_path} keys={len(emb)} dim={msg_dim}", flush=True)

    summary: Dict = {
        "embedding_file": str(emb_path),
        "loss_note": "N/A (frozen Unite mean features + trainable reduce heads)",
        "unite_dim_note": "actual Unite tweet dim is 1536 (not 1532)",
    }

    if not args.skip_classification:
        sample_map = load_samples(args.samples_csv)
        cls_res = train_eval_classification(
            emb=emb,
            sample_map=sample_map,
            seeds=seeds,
            epochs=int(args.cls_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=device,
            restrict_split=str(args.restrict_split),
            probe_label_smoothing=float(args.probe_label_smoothing),
        )
        cls_path = out_dir / "classification_metrics.json"
        cls_path.write_text(json.dumps({"UniteMean": cls_res}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["classification"] = str(cls_path)

    if not args.skip_interaction:
        inter_res = train_eval_interaction(
            emb=emb,
            samples_dir=args.samples_dir,
            tweet_emb_shards_dir=args.tweet_emb_shards_dir,
            seeds=seeds,
            epochs=int(args.inter_epochs),
            batch_size=int(args.inter_batch_size),
            lr=float(args.lr),
            device=device,
            max_pos=int(args.max_pos_per_window_record),
            max_neg=int(args.max_neg_per_window_record),
            tweet_reduce_dim=int(args.tweet_reduce_dim),
            taska_samples_csv=str(args.taska_samples_csv),
            pretrain_split=str(args.pretrain_split),
        )
        inter_path = out_dir / "interaction_metrics.json"
        if inter_path.exists():
            bak = out_dir / "interaction_metrics_all_splits_backup.json"
            if not bak.exists():
                bak.write_text(inter_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[Exp3] backed up previous interaction metrics -> {bak}", flush=True)
        inter_path.write_text(
            json.dumps({"UniteMean": inter_res}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["interaction"] = str(inter_path)
        summary["pretrain_split"] = str(args.pretrain_split)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Exp3 summary -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
