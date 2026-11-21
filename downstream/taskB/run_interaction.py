#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def load_emb_dict(pt_path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise RuntimeError(f"{pt_path}  embeddings ")
    emb_obj = payload["embeddings"]
    out: Dict[str, torch.Tensor] = {}

    if isinstance(emb_obj, dict):
        for k, v in emb_obj.items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            out[str(k)] = t.detach().cpu().float().view(-1)
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
                out[f"{uid}__w{int(widx):03d}"] = emb_obj[i].detach().cpu().float().view(-1)
            return out

    raise RuntimeError(f"{pt_path}  embeddings ")


def load_taska_split_tags(csv_path: str) -> Dict[str, str]:
    """Map `user_id_t` → `split_tag` from the shared windows CSV (same file as Task A)."""
    out: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid_t = str(row.get("user_id_t", "")).strip()
            if not uid_t:
                continue
            out[uid_t] = str(row.get("split_tag", "")).strip()
    return out


def load_embedding_files(index_json: str, dynamic_embedding: str) -> Dict[str, str]:
    idx = json.loads(Path(index_json).read_text(encoding="utf-8"))
    files: Dict[str, str] = {}
    if isinstance(idx, dict):
        for model, meta in idx.items():
            if isinstance(meta, dict):
                fp = str(meta.get("file", "")).strip()
                if fp:
                    files[str(model)] = fp
    de = str(dynamic_embedding).strip()
    if de and de.upper() not in {"NONE", "OFF", "-"}:
        files["DynamicModel"] = de
    return files


def _cand_keys_source_only(user_id: str, target_window_idx: int) -> List[str]:
    # Windows are reverse-ordered in this dataset:
    # smaller idx => more recent time. Therefore t-1 maps to target_window_idx + 1.
    source_window_idx = int(target_window_idx) + 1
    return [f"{user_id}__w{int(source_window_idx):03d}"]


def parse_window_idx_from_user_key(uk: str) -> int:
    """ ``user_id__w007`` 。"""
    s = str(uk)
    if "__w" not in s:
        raise ValueError(f" user_key  __w###，: {uk!r}")
    _, wpart = s.rsplit("__w", 1)
    return int(str(wpart).strip())


def filter_valid_triples_by_window_rowcount(
    valid: List[Tuple[str, str, int]],
    min_rows: int,
) -> Tuple[List[Tuple[str, str, int]], Dict[str, Any]]:
    """「 < min_rows」； window 。"""
    if int(min_rows) <= 0:
        return valid, {}
    cnt: Dict[int, int] = defaultdict(int)
    for uk, _, _ in valid:
        cnt[parse_window_idx_from_user_key(uk)] += 1
    keep_w = {w for w, c in cnt.items() if c >= int(min_rows)}
    before = len(valid)
    out = [(uk, tid, lb) for (uk, tid, lb) in valid if parse_window_idx_from_user_key(uk) in keep_w]
    meta = {
        "min_rows_per_window": int(min_rows),
        "n_triples_before": int(before),
        "n_triples_after": int(len(out)),
        "windows_dropped": sorted([w for w in cnt if w not in keep_w]),
    }
    return out, meta


def downstream_train_val_test_indices(
    user_keys: Sequence[str],
    seed: int,
    split_mode: str,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    """ train/val/test 。

    - ``user``：， raw user_id  8/1/1 （ seed）。
    - ``window``：；** w = **（ _cand_keys_source_only ）。
      distinct  ≥3： w （） 8/1/1 ****（train=，test=）。
       2 （ pretrain-split）： → train； seed  val/test。
       raw user  split； seed。
    """
    mode = str(split_mode).strip().lower()
    n = len(user_keys)
    if n == 0:
        raise RuntimeError("user_keys ")

    if mode == "user":
        raw_users = [str(u).split("__w", 1)[0] for u in user_keys]
        uniq_users = sorted(set(raw_users))
        if len(uniq_users) < 3:
            raise RuntimeError(" user ，")
        set_seed(int(seed))
        perm_u = torch.randperm(len(uniq_users))
        users_shuf = [uniq_users[int(i)] for i in perm_u.tolist()]
        n_u_train = max(1, int(len(users_shuf) * 0.8))
        n_u_val = max(1, int(len(users_shuf) * 0.1))
        n_u_test = len(users_shuf) - n_u_train - n_u_val
        if n_u_test <= 0:
            n_u_test = 1
            if n_u_train > n_u_val:
                n_u_train -= 1
            else:
                n_u_val -= 1
        u_train = set(users_shuf[:n_u_train])
        u_val = set(users_shuf[n_u_train : n_u_train + n_u_val])
        u_test = set(users_shuf[n_u_train + n_u_val : n_u_train + n_u_val + n_u_test])
        idx_train_l = [i for i, u in enumerate(raw_users) if str(u) in u_train]
        idx_val_l = [i for i, u in enumerate(raw_users) if str(u) in u_val]
        idx_test_l = [i for i, u in enumerate(raw_users) if str(u) in u_test]
        meta: Dict[str, Any] = {
            "scheme": "user_disjoint",
            "seed": int(seed),
            "n_users": len(uniq_users),
            "frac_train_users": float(n_u_train) / float(len(uniq_users)),
        }
    elif mode == "window":
        w_per_i = [parse_window_idx_from_user_key(uk) for uk in user_keys]
        uniq_w_sorted_old_to_new = sorted(set(w_per_i), reverse=True)
        n_w = len(uniq_w_sorted_old_to_new)
        if n_w < 2:
            raise RuntimeError(
                f" 2 ， distinct={n_w}。"
                " pretrain-split=test ， --pretrain-split /  split，"
                " --window-min-rows-per-window  distinct。"
            )
        if n_w == 2:
            w_old, w_new = int(uniq_w_sorted_old_to_new[0]), int(uniq_w_sorted_old_to_new[1])
            idx_train_l = [i for i in range(n) if w_per_i[i] == w_old]
            new_idx = [i for i in range(n) if w_per_i[i] == w_new]
            if len(new_idx) < 2:
                raise RuntimeError(
                    f"「」 w={w_new}  {len(new_idx)} ， val/test。"
                    " --downstream-split user。"
                )
            set_seed(int(seed))
            shuf = new_idx.copy()
            random.shuffle(shuf)
            n_val_new = max(1, len(shuf) // 2)
            idx_val_l = shuf[:n_val_new]
            idx_test_l = shuf[n_val_new:]
            meta = {
                "scheme": "window_time_two_windows",
                "independent_of_seed": False,
                "subsplit_seed": int(seed),
                "time_semantics": " w = （）； w = ",
                "split_rule": "train=；val/test=（）",
                "n_distinct_windows": 2,
                "windows_ordered_old_to_new": [w_old, w_new],
                "train_window_index_min_max": [w_old, w_old],
                "val_test_window_index_min_max": [w_new, w_new],
                "val_window_index_min_max": [w_new, w_new],
                "test_window_index_min_max": [w_new, w_new],
                "n_train_windows": 1,
                "n_val_windows": 1,
                "n_test_windows": 1,
            }
        else:
            n_train_w = max(1, int(n_w * 0.8))
            n_val_w = max(1, int(n_w * 0.1))
            n_test_w = n_w - n_train_w - n_val_w
            if n_test_w <= 0:
                n_test_w = 1
                if n_train_w > n_val_w:
                    n_train_w -= 1
                else:
                    n_val_w -= 1
            train_ws = set(uniq_w_sorted_old_to_new[:n_train_w])
            val_ws = set(uniq_w_sorted_old_to_new[n_train_w : n_train_w + n_val_w])
            test_ws = set(uniq_w_sorted_old_to_new[n_train_w + n_val_w :])
            idx_train_l = [i for i in range(n) if w_per_i[i] in train_ws]
            idx_val_l = [i for i in range(n) if w_per_i[i] in val_ws]
            idx_test_l = [i for i in range(n) if w_per_i[i] in test_ws]
            meta = {
                "scheme": "window_time_disjoint",
                "independent_of_seed": True,
                "time_semantics": " w = （）； w = ",
                "split_rule": "distinct  w ， 8/1/1 ：train=，test=",
                "n_distinct_windows": int(n_w),
                "windows_ordered_old_to_new": [int(x) for x in uniq_w_sorted_old_to_new],
                "n_train_windows": int(n_train_w),
                "n_val_windows": int(n_val_w),
                "n_test_windows": int(n_test_w),
                "train_window_index_min_max": [min(train_ws), max(train_ws)],
                "val_window_index_min_max": [min(val_ws), max(val_ws)],
                "test_window_index_min_max": [min(test_ws), max(test_ws)],
            }
    else:
        raise ValueError(f" downstream split : {split_mode!r}（ user / window）")

    if min(len(idx_train_l), len(idx_val_l), len(idx_test_l)) == 0:
        raise RuntimeError(f" split  mode={mode!r} train/val/test={len(idx_train_l)}/{len(idx_val_l)}/{len(idx_test_l)}")
    meta["n_samples_train"] = len(idx_train_l)
    meta["n_samples_val"] = len(idx_val_l)
    meta["n_samples_test"] = len(idx_test_l)
    return idx_train_l, idx_val_l, idx_test_l, meta


def load_message_samples(
    samples_dir: str,
    allowed_user_keys: set[str],
    max_files: int = 0,
    max_pos_per_window_record: int = 0,
    max_neg_per_window_record: int = 0,
    per_window_cap_seed: int = 42,
) -> Tuple[List[Tuple[str, str, int]], set[str], Dict[str, Any]]:
    """ rec ： pos/neg 。

     ``samples_dir``  ``*/window_*.json`` ， ``message_user_samples.jsonl``。
    """
    root = Path(samples_dir)
    objs: List[Dict[str, Any]] = []
    jsonl_path = None
    if root.is_file() and root.suffix.lower() == ".jsonl":
        jsonl_path = root
    elif root.is_dir() and (root / "message_user_samples.jsonl").is_file():
        jsonl_path = root / "message_user_samples.jsonl"
    if jsonl_path is not None:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    objs.append(obj)
        if int(max_files) > 0:
            objs = objs[: int(max_files)]
    else:
        files = sorted(root.glob("*/window_*.json"))
        if int(max_files) > 0:
            files = files[: int(max_files)]
        for fp in files:
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                objs.append(obj)

    cap_p = int(max_pos_per_window_record)
    cap_n = int(max_neg_per_window_record)
    if cap_p < 0 or cap_n < 0:
        raise ValueError("max_pos_per_window_record / max_neg_per_window_record ")
    rng = random.Random(int(per_window_cap_seed)) if (cap_p > 0 or cap_n > 0) else None
    n_pos_before = 0
    n_pos_after = 0
    n_neg_before = 0
    n_neg_after = 0
    n_records_emitted = 0

    out: List[Tuple[str, str, int]] = []
    needed_tweet_ids: set[str] = set()
    for obj in objs:
        uid = str(obj.get("user_id", "")).strip()
        if not uid:
            continue
        target_window_idx = int(obj.get("target_window_idx", obj.get("group_t", -1)))
        if target_window_idx < 0:
            # malformed record fallback: if target is unavailable, fall back to source_window_idx
            source_window_idx = int(obj.get("source_window_idx", -1))
            if source_window_idx < 0:
                continue
            cand_keys = [f"{uid}__w{int(source_window_idx):03d}"]
        else:
            cand_keys = _cand_keys_source_only(uid, target_window_idx)

        picked_key = None
        for k in cand_keys:
            if k in allowed_user_keys:
                picked_key = k
                break
        if picked_key is None:
            continue
        n_records_emitted += 1

        pos_ids = obj.get("pos_item_ids", [])
        neg_ids = obj.get("neg_item_ids", [])
        if not isinstance(pos_ids, list) or not isinstance(neg_ids, list):
            continue

        pos_clean = [str(tid) for tid in pos_ids]
        neg_clean = [str(tid) for tid in neg_ids]
        if rng is not None:
            n_pos_before += len(pos_clean)
            n_neg_before += len(neg_clean)
            if cap_p > 0 and len(pos_clean) > cap_p:
                pos_clean = rng.sample(pos_clean, cap_p)
            if cap_n > 0 and len(neg_clean) > cap_n:
                neg_clean = rng.sample(neg_clean, cap_n)
            n_pos_after += len(pos_clean)
            n_neg_after += len(neg_clean)

        for t in pos_clean:
            out.append((picked_key, t, 1))
            needed_tweet_ids.add(t)
        for t in neg_clean:
            out.append((picked_key, t, 0))
            needed_tweet_ids.add(t)

    cap_meta: Dict[str, Any] = {}
    if rng is not None:
        cap_meta = {
            "max_pos_per_window_record": cap_p,
            "max_neg_per_window_record": cap_n,
            "seed": int(per_window_cap_seed),
            "n_pos_triples_before_cap": int(n_pos_before),
            "n_pos_triples_after_cap": int(n_pos_after),
            "n_neg_triples_before_cap": int(n_neg_before),
            "n_neg_triples_after_cap": int(n_neg_after),
            "n_window_records_emitted": int(n_records_emitted),
        }
    return out, needed_tweet_ids, cap_meta


def load_tweet_features(tweet_ids: set[str], tweet_emb_shards_dir: str) -> Dict[str, torch.Tensor]:
    dyn_dir = Path(".").resolve()
    if str(dyn_dir) not in sys.path:
        sys.path.insert(0, str(dyn_dir))
    from time_slice_msg_feature_loader import load_msg_features_from_compact_shards  # type: ignore

    msg_feats, _msg_dim = load_msg_features_from_compact_shards(
        shards_dir=tweet_emb_shards_dir,
        needed_msg_ids=tweet_ids,
        verbose=True,
    )
    out: Dict[str, torch.Tensor] = {}
    for k, v in msg_feats.items():
        out[str(k)] = v.detach().cpu().float().view(-1)
    return out


class BinaryMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BinaryInteractionModel(nn.Module):
    """ user  + tweet UNITE； tweet  MLP 。"""

    def __init__(self, user_dim: int, tweet_dim: int, tweet_reduce_dim: int):
        super().__init__()
        self.user_dim = int(user_dim)
        self.tweet_dim = int(tweet_dim)
        rd = int(tweet_reduce_dim)
        self.use_tweet_reduce = rd > 0
        if self.use_tweet_reduce:
            self.tweet_proj = nn.Linear(self.tweet_dim, rd)
            mlp_in = self.user_dim + rd
        else:
            self.tweet_proj = None
            mlp_in = self.user_dim + self.tweet_dim
        self.head = BinaryMLP(mlp_in)

    def forward(self, user_x: torch.Tensor, tweet_x: torch.Tensor) -> torch.Tensor:
        if self.use_tweet_reduce:
            tw = self.tweet_proj(tweet_x)
            x = torch.cat([user_x, tw], dim=1)
        else:
            x = torch.cat([user_x, tweet_x], dim=1)
        return self.head(x)


class BinaryFineTuneModel(nn.Module):
    def __init__(self, user_init: torch.Tensor, tweet_dim: int, tweet_reduce_dim: int = 0):
        super().__init__()
        n_users, user_dim = int(user_init.size(0)), int(user_init.size(1))
        self.user_backbone = nn.Embedding(n_users, user_dim)
        with torch.no_grad():
            self.user_backbone.weight.copy_(user_init)
        rd = int(tweet_reduce_dim)
        self.use_tweet_reduce = rd > 0
        if self.use_tweet_reduce:
            self.tweet_proj = nn.Linear(int(tweet_dim), rd)
            head_in = user_dim + rd
        else:
            self.tweet_proj = None
            head_in = user_dim + int(tweet_dim)
        self.head = BinaryMLP(head_in)

    def forward(self, user_idx: torch.Tensor, tweet_x: torch.Tensor) -> torch.Tensor:
        user_x = self.user_backbone(user_idx)
        if self.tweet_proj is not None:
            tweet_x = self.tweet_proj(tweet_x)
        x = torch.cat([user_x, tweet_x], dim=1)
        return self.head(x)


def binary_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).long()
    y = y.long()

    acc = float((pred == y).float().mean().item())
    tp = float(((pred == 1) & (y == 1)).sum().item())
    tn = float(((pred == 0) & (y == 0)).sum().item())
    fp = float(((pred == 1) & (y == 0)).sum().item())
    fn = float(((pred == 0) & (y == 1)).sum().item())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    bce = float(F.binary_cross_entropy(probs, y.float()).item())
    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "bce": bce,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def auc_binary(y_true: torch.Tensor, scores: torch.Tensor) -> float:
    y = y_true.detach().cpu().float()
    s = scores.detach().cpu().float()
    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank-based AUC (Mann-Whitney U). Ties handled by average ranks.
    order = torch.argsort(s, stable=True)
    ranks = torch.zeros_like(s, dtype=torch.float64)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float64)
    pos_ranks_sum = float(ranks[y == 1].sum().item())
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def build_tweet_bank_and_ix(
    tweet_feat_map: Dict[str, torch.Tensor],
    ordered_tweet_ids: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ tweet  tweet_bank [U,D]， tweet_ix [N] 。

     ``torch.stack([... for _ in ordered_tweet_ids])``  [N,D] 
    （Python  +  stack ）。 batch  bank 。
    """
    if not ordered_tweet_ids:
        raise RuntimeError("ordered_tweet_ids ")
    try:
        import pandas as pd  # type: ignore

        codes, uniques = pd.factorize(pd.Series(list(ordered_tweet_ids), dtype=object), sort=False)
        rows: List[torch.Tensor] = []
        for u in uniques:
            tid = str(u)
            if tid not in tweet_feat_map:
                raise KeyError(tid)
            rows.append(tweet_feat_map[tid].float().view(-1))
        bank = torch.stack(rows, dim=0).contiguous()
        ix = torch.from_numpy(np.asarray(codes, dtype=np.int64))
        return bank, ix
    except ImportError:
        uniq: List[str] = []
        seen: set[str] = set()
        for t in ordered_tweet_ids:
            ts = str(t)
            if ts not in seen:
                seen.add(ts)
                uniq.append(ts)
        bank = torch.stack([tweet_feat_map[t].float().view(-1) for t in uniq], dim=0).contiguous()
        id2row = {t: i for i, t in enumerate(uniq)}
        ix = torch.tensor([id2row[str(t)] for t in ordered_tweet_ids], dtype=torch.long)
        return bank, ix


def ranking_metrics_at_k(user_keys: Sequence[str], y_true: torch.Tensor, scores: torch.Tensor, k: int = 10) -> Dict[str, float]:
    # Group by user_id_t, rank tweet candidates by score descending.
    grouped: Dict[str, List[Tuple[float, int]]] = {}
    y = y_true.detach().cpu().long().tolist()
    s = scores.detach().cpu().float().tolist()
    for uk, yi, si in zip(user_keys, y, s):
        grouped.setdefault(str(uk), []).append((float(si), int(yi)))

    hit_vals: List[float] = []
    ndcg_vals: List[float] = []
    for _, items in grouped.items():
        items.sort(key=lambda x: x[0], reverse=True)
        topk = items[: int(k)]
        rel_topk = [int(lb) for _, lb in topk]
        hit = 1.0 if any(lb == 1 for lb in rel_topk) else 0.0
        hit_vals.append(hit)

        dcg = 0.0
        for i, rel in enumerate(rel_topk, start=1):
            if rel > 0:
                dcg += 1.0 / np.log2(float(i) + 1.0)
        total_pos = sum(int(lb) for _, lb in items)
        ideal_len = min(int(k), total_pos)
        idcg = 0.0
        for i in range(1, ideal_len + 1):
            idcg += 1.0 / np.log2(float(i) + 1.0)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_vals.append(float(ndcg))

    if not hit_vals:
        return {"hit@10": 0.0, "ndcg@10": 0.0}
    return {"hit@10": float(np.mean(hit_vals)), "ndcg@10": float(np.mean(ndcg_vals))}


def train_one_seed(
    user_x: torch.Tensor,
    tweet_bank: torch.Tensor,
    tweet_ix: torch.Tensor,
    y: torch.Tensor,
    user_keys: Sequence[str],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    tweet_reduce_dim: int = 0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    epoch_history_mode: str = "off",
    log_prefix: str = "",
    verbose_training: bool = False,
    downstream_split: str = "user",
    max_train_batches: int = 0,
) -> Dict[str, Any]:
    set_seed(seed)
    n = int(y.size(0))
    if len(user_keys) != n or int(user_x.size(0)) != n or int(tweet_ix.size(0)) != n:
        raise RuntimeError("user_x / tweet_ix / y / user_keys ")

    idx_train_l, idx_val_l, idx_test_l, split_meta = downstream_train_val_test_indices(
        user_keys, int(seed), str(downstream_split)
    )

    idx_train = torch.tensor(idx_train_l, dtype=torch.long)
    idx_val = torch.tensor(idx_val_l, dtype=torch.long)
    idx_test = torch.tensor(idx_test_l, dtype=torch.long)

    tw_train_ix = tweet_ix[idx_train]
    u_train_x, y_train = user_x[idx_train], y[idx_train]
    u_val_x, y_val = user_x[idx_val], y[idx_val]
    u_test_x, y_test = user_x[idx_test], y[idx_test]
    tw_val = tweet_bank[tweet_ix[idx_val]]
    tw_test = tweet_bank[tweet_ix[idx_test]]
    user_test = [str(user_keys[int(i)]) for i in idx_test.tolist()]

    tweet_dim = int(tweet_bank.size(1))
    model = BinaryInteractionModel(
        user_dim=int(user_x.size(1)),
        tweet_dim=tweet_dim,
        tweet_reduce_dim=int(tweet_reduce_dim),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    crit = nn.BCEWithLogitsLoss()
    pin = torch.cuda.is_available()
    loader = DataLoader(
        TensorDataset(u_train_x, tw_train_ix, y_train),
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=pin,
    )

    pfx = str(log_prefix).strip() or f"[seed={seed}]"
    if verbose_training:
        print(
            f"{pfx} start split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} "
            f"max_epochs={epochs} batches/epoch≈{max(1, (len(idx_train) + int(batch_size) - 1) // int(batch_size))}",
            flush=True,
        )

    best_val = -1.0
    best_state = None
    best_epoch = 0
    stall = 0
    es_pat = int(early_stopping_patience)
    es_delta = float(early_stopping_min_delta)
    stopped_early = False
    last_ep = 0
    eh_mode = str(epoch_history_mode).strip().lower()
    log_val_eh = eh_mode in ("val", "val_test")
    log_test_eh = eh_mode == "val_test"
    epoch_history: List[Dict[str, Any]] = []
    max_tb = int(max_train_batches)
    train_batches_total = 0
    for ep in range(int(epochs)):
        last_ep = ep + 1
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_nb = 0
        hit_batch_cap = False
        for ub, tix_b, yb in loader:
            ub = ub.to(device, non_blocking=pin)
            tb = tweet_bank[tix_b].to(device, non_blocking=pin)
            yb = yb.to(device).float()
            opt.zero_grad()
            loss = crit(model(ub, tb), yb)
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_loss_nb += 1
            train_batches_total += 1
            if max_tb > 0 and train_batches_total >= max_tb:
                hit_batch_cap = True
                break

        model.eval()
        with torch.no_grad():
            val_logits = model(u_val_x.to(device), tw_val.to(device)).cpu()
            val_m = binary_metrics_from_logits(val_logits, y_val)
        if log_val_eh:
            entry_eh: Dict[str, Any] = {"epoch": ep + 1, "val": {k: float(v) for k, v in val_m.items()}}
            if log_test_eh:
                with torch.no_grad():
                    te_logits = model(u_test_x.to(device), tw_test.to(device)).cpu()
                te_m = binary_metrics_from_logits(te_logits, y_test)
                te_scores = torch.sigmoid(te_logits)
                te_m["auc"] = auc_binary(y_test, te_scores)
                entry_eh["test"] = {k: float(v) for k, v in te_m.items()}
            epoch_history.append(entry_eh)
        vf1 = float(val_m["f1"])
        avg_tr_loss = epoch_loss_sum / max(1, epoch_loss_nb)
        if vf1 > best_val + es_delta:
            best_val = vf1
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        stall_show = f"{stall}/{es_pat}" if es_pat > 0 else f"{stall}/off"
        if verbose_training:
            print(
                f"{pfx} epoch {ep + 1}/{int(epochs)} train_loss={avg_tr_loss:.4f} "
                f"val_f1={vf1:.4f} val_acc={float(val_m['acc']):.4f} best_val_f1={best_val:.4f} stall={stall_show}",
                flush=True,
            )
        if es_pat > 0 and stall >= es_pat:
            stopped_early = True
            break
        if hit_batch_cap:
            break

    if verbose_training:
        print(
            f"{pfx} train_loop_done ep_run={last_ep} best_epoch={best_epoch} best_val_f1={best_val:.4f} "
            f"early_stopped={int(stopped_early)} train_batches={train_batches_total} -> load best ckpt & eval test",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        test_logits = model(u_test_x.to(device), tw_test.to(device)).cpu()
    m = binary_metrics_from_logits(test_logits, y_test)
    test_scores = torch.sigmoid(test_logits)
    m["auc"] = auc_binary(y_test, test_scores)
    m.update(ranking_metrics_at_k(user_test, y_test, test_scores, k=10))
    m["num_train"] = float(len(idx_train))
    m["num_val"] = float(len(idx_val))
    m["num_test"] = float(len(idx_test))
    m["epochs_run"] = float(last_ep)
    m["best_epoch"] = float(best_epoch)
    m["best_val_f1"] = float(best_val)
    m["early_stopped"] = 1.0 if stopped_early else 0.0
    m["train_batches_total"] = float(train_batches_total)
    if max_tb > 0:
        m["max_train_batches_limit"] = float(max_tb)
    if epoch_history:
        m["epoch_history"] = epoch_history
    m["downstream_split_meta"] = split_meta
    return m


def train_one_seed_finetune(
    user_x: torch.Tensor,
    tweet_bank: torch.Tensor,
    tweet_ix: torch.Tensor,
    y: torch.Tensor,
    user_keys: Sequence[str],
    seed: int,
    epochs: int,
    batch_size: int,
    head_lr: float,
    backbone_lr: float,
    freeze_backbone_epochs: int,
    device: str,
    tweet_reduce_dim: int = 0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    epoch_history_mode: str = "off",
    log_prefix: str = "",
    verbose_training: bool = False,
    downstream_split: str = "user",
    max_train_batches: int = 0,
) -> Dict[str, Any]:
    set_seed(seed)
    n = int(y.size(0))
    if len(user_keys) != n or int(tweet_ix.size(0)) != n:
        raise RuntimeError("user_keys / tweet_ix ")

    raw_users = [str(u).split("__w", 1)[0] for u in user_keys]
    uniq_users = sorted(set(raw_users))
    if len(uniq_users) < 3:
        raise RuntimeError(" user_id ，")
    uid2idx = {u: i for i, u in enumerate(uniq_users)}
    sample_uid_idx = torch.tensor([uid2idx[u] for u in raw_users], dtype=torch.long)

    idx_train_l, idx_val_l, idx_test_l, split_meta = downstream_train_val_test_indices(
        user_keys, int(seed), str(downstream_split)
    )

    # build per-raw-user init embedding by mean over windows
    sums = torch.zeros((len(uniq_users), int(user_x.size(1))), dtype=torch.float32)
    cnts = torch.zeros((len(uniq_users),), dtype=torch.float32)
    sm = str(downstream_split).strip().lower()
    init_rows: List[int] = list(idx_train_l) if sm == "window" else list(range(n))
    for i in init_rows:
        ui = int(sample_uid_idx[i].item())
        sums[ui] += user_x[i].float()
        cnts[ui] += 1.0
    user_init = sums / cnts.clamp_min(1.0).unsqueeze(1)

    idx_train = torch.tensor(idx_train_l, dtype=torch.long)
    idx_val = torch.tensor(idx_val_l, dtype=torch.long)
    idx_test = torch.tensor(idx_test_l, dtype=torch.long)

    train_uid_idx = sample_uid_idx[idx_train]
    val_uid_idx = sample_uid_idx[idx_val]
    test_uid_idx = sample_uid_idx[idx_test]
    tw_train_ix = tweet_ix[idx_train]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]
    tweet_val = tweet_bank[tweet_ix[idx_val]]
    tweet_test = tweet_bank[tweet_ix[idx_test]]
    user_test = [str(user_keys[int(i)]) for i in idx_test.tolist()]

    tweet_dim = int(tweet_bank.size(1))
    model = BinaryFineTuneModel(
        user_init=user_init,
        tweet_dim=tweet_dim,
        tweet_reduce_dim=int(tweet_reduce_dim),
    ).to(device)
    crit = nn.BCEWithLogitsLoss()
    pin = torch.cuda.is_available()
    loader = DataLoader(
        TensorDataset(train_uid_idx, tw_train_ix, y_train),
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=pin,
    )

    pfx = str(log_prefix).strip() or f"[seed={seed}]"
    if verbose_training:
        print(
            f"{pfx} finetune start split train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} "
            f"max_epochs={epochs} freeze_backbone_epochs={freeze_backbone_epochs} "
            f"batches/epoch≈{max(1, (len(idx_train) + int(batch_size) - 1) // int(batch_size))}",
            flush=True,
        )

    best_val = -1.0
    best_state = None
    best_epoch = 0
    stall = 0
    es_pat = int(early_stopping_patience)
    es_delta = float(early_stopping_min_delta)
    stopped_early = False
    last_ep = 0
    eh_mode = str(epoch_history_mode).strip().lower()
    log_val_eh = eh_mode in ("val", "val_test")
    log_test_eh = eh_mode == "val_test"
    epoch_history: List[Dict[str, Any]] = []
    max_tb = int(max_train_batches)
    train_batches_total = 0
    for ep in range(int(epochs)):
        last_ep = ep + 1
        train_backbone = ep >= int(freeze_backbone_epochs)
        model.user_backbone.weight.requires_grad = bool(train_backbone)
        head_params = list(model.head.parameters())
        if model.tweet_proj is not None:
            head_params += list(model.tweet_proj.parameters())
        params = [{"params": head_params, "lr": float(head_lr)}]
        if train_backbone:
            params.append({"params": model.user_backbone.parameters(), "lr": float(backbone_lr)})
        opt = torch.optim.Adam(params)

        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_nb = 0
        hit_batch_cap = False
        for ub, tix_b, yb in loader:
            ub = ub.to(device, non_blocking=pin)
            tb = tweet_bank[tix_b].to(device, non_blocking=pin)
            yb = yb.to(device).float()
            opt.zero_grad()
            loss = crit(model(ub, tb), yb)
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_loss_nb += 1
            train_batches_total += 1
            if max_tb > 0 and train_batches_total >= max_tb:
                hit_batch_cap = True
                break

        model.eval()
        with torch.no_grad():
            val_logits = model(val_uid_idx.to(device), tweet_val.to(device)).cpu()
            val_m = binary_metrics_from_logits(val_logits, y_val)
        if log_val_eh:
            entry_eh = {"epoch": ep + 1, "val": {k: float(v) for k, v in val_m.items()}}
            if log_test_eh:
                with torch.no_grad():
                    te_logits = model(test_uid_idx.to(device), tweet_test.to(device)).cpu()
                te_m = binary_metrics_from_logits(te_logits, y_test)
                te_scores = torch.sigmoid(te_logits)
                te_m["auc"] = auc_binary(y_test, te_scores)
                entry_eh["test"] = {k: float(v) for k, v in te_m.items()}
            epoch_history.append(entry_eh)
        vf1 = float(val_m["f1"])
        avg_tr_loss = epoch_loss_sum / max(1, epoch_loss_nb)
        bb = "bb_on" if train_backbone else "bb_off"
        if vf1 > best_val + es_delta:
            best_val = vf1
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
        stall_show = f"{stall}/{es_pat}" if es_pat > 0 else f"{stall}/off"
        if verbose_training:
            print(
                f"{pfx} epoch {ep + 1}/{int(epochs)} {bb} train_loss={avg_tr_loss:.4f} "
                f"val_f1={vf1:.4f} val_acc={float(val_m['acc']):.4f} best_val_f1={best_val:.4f} stall={stall_show}",
                flush=True,
            )
        if es_pat > 0 and stall >= es_pat:
            stopped_early = True
            break
        if hit_batch_cap:
            break

    if verbose_training:
        print(
            f"{pfx} finetune_loop_done ep_run={last_ep} best_epoch={best_epoch} best_val_f1={best_val:.4f} "
            f"early_stopped={int(stopped_early)} train_batches={train_batches_total} -> load best ckpt & eval test",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_uid_idx.to(device), tweet_test.to(device)).cpu()
    m = binary_metrics_from_logits(test_logits, y_test)
    test_scores = torch.sigmoid(test_logits)
    m["auc"] = auc_binary(y_test, test_scores)
    m.update(ranking_metrics_at_k(user_test, y_test, test_scores, k=10))
    m["num_train"] = float(len(idx_train))
    m["num_val"] = float(len(idx_val))
    m["num_test"] = float(len(idx_test))
    m["epochs_run"] = float(last_ep)
    m["best_epoch"] = float(best_epoch)
    m["best_val_f1"] = float(best_val)
    m["early_stopped"] = 1.0 if stopped_early else 0.0
    m["train_batches_total"] = float(train_batches_total)
    if max_tb > 0:
        m["max_train_batches_limit"] = float(max_tb)
    if epoch_history:
        m["epoch_history"] = epoch_history
    m["downstream_split_meta"] = split_meta
    m["_finetune_export"] = {
        "uniq_users": list(uniq_users),
        "user_backbone_weight": model.user_backbone.weight.detach().cpu().clone(),
    }
    return m


def window_keys_to_finetuned_emb_dict(
    uniq_users: Sequence[str],
    user_backbone_weight: torch.Tensor,
    window_user_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """ ``BinaryFineTuneModel.user_backbone``（ raw user ） ``user__w###`` 。"""
    uid2i = {str(u): i for i, u in enumerate(uniq_users)}
    W = user_backbone_weight
    out: Dict[str, torch.Tensor] = {}
    for uk in window_user_keys:
        raw = str(uk).split("__w", 1)[0]
        out[str(uk)] = W[int(uid2i[raw])].detach().float().view(-1).clone()
    return out


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = ["acc", "precision", "recall", "f1", "bce", "auc", "hit@10", "ndcg@10"]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Binary user-interaction downstream compare on shared samples")
    p.add_argument("--samples-dir", type=str, default="dataset/local/message_user_samples")
    p.add_argument("--embedding-index-json", type=str, default="outputs/test_only_v2/export_test_only_index.json")
    p.add_argument(
        "--dynamic-embedding",
        type=str,
        default="outputs/taskA_user_window_embeddings.pt",
        help=" DynamicModel ； NONE  --embedding-index-json ",
    )
    p.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help=">0： batch（ epoch ）， best  test；"
        "0 （ epoch  train loader）。。",
    )
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--finetune-model-names", type=str, default="", help="comma-separated model names for backbone finetune (e.g., DynamicModel)")
    p.add_argument("--head-lr", type=float, default=1e-3, help="lr for classifier head when finetuning")
    p.add_argument("--backbone-lr", type=float, default=5e-5, help="lr for user backbone when finetuning")
    p.add_argument("--freeze-backbone-epochs", type=int, default=2, help="epochs to train head-only before unfreezing backbone")
    p.add_argument(
        "--save-finetuned-dynamic-embedding",
        type=str,
        default="",
        help=" run  finetune： seed  user_backbone  --dynamic-embedding  .pt（ {seed} ， .seed{seed}.pt）",
    )
    p.add_argument("--max-window-files", type=int, default=0, help="debug: limit number of window json files")
    p.add_argument(
        "--max-pos-per-window-record",
        type=int,
        default=0,
        help=" window_*.json （0=）； random.sample ， --per-window-cap-seed。",
    )
    p.add_argument(
        "--max-neg-per-window-record",
        type=int,
        default=0,
        help="，。",
    )
    p.add_argument(
        "--per-window-cap-seed",
        type=int,
        default=42,
        help=" pos/neg （glob ，）。",
    )
    p.add_argument(
        "--taska-samples-csv",
        type=str,
        default="",
        help=" --pretrain-split ： taskA CSV  split_tag  user_id_t，",
    )
    p.add_argument(
        "--pretrain-split",
        type=str,
        default="",
        choices=["", "train", "val", "test"],
        help="train/val/test： CSV  split_tag （）",
    )
    p.add_argument(
        "--downstream-split",
        type=str,
        default="user",
        choices=["user", "window"],
        help=" train/val/test ：user= raw user_id  8/1/1 （）；"
        "window= w###： w ，distinct  w （） 8/1/1 ，"
        "train 、test （ split）。",
    )
    p.add_argument(
        "--window-min-rows-per-window",
        type=int,
        default=0,
        help=" --downstream-split window ：（0=）。"
        " distinct ； 2 「 train /  val|test」。",
    )
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument(
        "--interaction-sample-ratio",
        type=float,
        default=1.0,
        help=" (user_key,tweet,label)， 0.1； 1.0 ",
    )
    p.add_argument(
        "--interaction-subsample-seed",
        type=int,
        default=42,
        help="--interaction-sample-ratio<1 ，",
    )
    p.add_argument(
        "--tweet-reduce-dim",
        type=int,
        default=0,
        help=">0： tweet UNITE  Linear(tweet_dim->)， MLP ；"
        "0：， concat(user, full_tweet)（）。。",
    )
    p.add_argument(
        "--models-first",
        type=str,
        default="",
        help="，； embedding 。"
        " DynamicModel。（ DynamicModel ）。",
    )
    p.add_argument(
        "--models-only",
        type=str,
        default="",
        help="，（ baseline ）；"
        "common_keys  embedding 。 DynamicModel。",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=" F1  epoch （ min_delta）；0 （ --epochs）。",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help=" F1 （ 0）。",
    )
    p.add_argument(
        "--epoch-history",
        type=str,
        default="off",
        choices=["off", "val", "val_test"],
        help=" epoch  per_seed  epoch_history：off（）；val：；"
        "val_test：+（ epoch  metrics  hit@10/ndcg，）。",
    )
    p.add_argument(
        "--verbose-training",
        action="store_true",
        help=" user_x 、 epoch  train_loss/val （）。",
    )
    clean_argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    args = p.parse_args(clean_argv)

    emb_files = load_embedding_files(args.embedding_index_json, args.dynamic_embedding)
    if not emb_files:
        raise RuntimeError(" embedding ")

    mo_raw = str(args.models_only).strip()
    if mo_raw:
        wanted_order = [x.strip() for x in mo_raw.split(",") if x.strip()]
        emb_pick: Dict[str, str] = {}
        for n in wanted_order:
            if n not in emb_files:
                raise RuntimeError(f"--models-only （）: {n}")
            emb_pick[n] = emb_files[n]
        emb_files = emb_pick
        print(f"[ModelsOnly] : {list(emb_files.keys())}", flush=True)

    emb_maps: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, fp in emb_files.items():
        emb_maps[name] = load_emb_dict(fp)
        print(f"[LoadEmb] {name}: {len(emb_maps[name])}")

    # fairness: use user_id_t keys available for all models.
    key_sets = [set(m.keys()) for m in emb_maps.values()]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    if not common_keys:
        raise RuntimeError(" user_id_t keys")
    print(f"[CommonUsers] {len(common_keys)}")

    split_filter_meta: Dict[str, object] = {}
    csv_path = str(args.taska_samples_csv).strip()
    want_split = str(args.pretrain_split).strip()
    if (csv_path and not want_split) or (want_split and not csv_path):
        raise RuntimeError("--taska-samples-csv  --pretrain-split ")
    if csv_path and want_split:
        tag_map = load_taska_split_tags(csv_path)
        before = len(common_keys)
        common_keys = {k for k in common_keys if tag_map.get(str(k)) == want_split}
        split_filter_meta = {
            "taska_samples_csv": csv_path,
            "pretrain_split": want_split,
            "keys_before_split_filter": int(before),
            "keys_after_split_filter": int(len(common_keys)),
        }
        print(
            f"[PretrainSplit] split={want_split} keys {before} -> {len(common_keys)} "
            f"( train/val/test )"
        )
        if not common_keys:
            raise RuntimeError(" pretrain_split  common_keys ， CSV  embedding ")

    for name, emap in emb_maps.items():
        n_usable = len(set(emap.keys()) & common_keys)
        print(f"[AlignedEmb] {name}: usable_on_common={n_usable}")

    if int(args.max_pos_per_window_record) < 0 or int(args.max_neg_per_window_record) < 0:
        raise ValueError("--max-pos-per-window-record / --max-neg-per-window-record ")
    triples, needed_tweet_ids, window_record_cap_meta = load_message_samples(
        samples_dir=args.samples_dir,
        allowed_user_keys=common_keys,
        max_files=int(args.max_window_files),
        max_pos_per_window_record=int(args.max_pos_per_window_record),
        max_neg_per_window_record=int(args.max_neg_per_window_record),
        per_window_cap_seed=int(args.per_window_cap_seed),
    )
    if not triples:
        raise RuntimeError("")
    print(f"[Samples] interactions={len(triples)} users={len(set(k for k,_,_ in triples))} tweets={len(needed_tweet_ids)}")
    if window_record_cap_meta:
        print(
            f"[Samples] per-window cap pos<={window_record_cap_meta['max_pos_per_window_record']} "
            f"neg<={window_record_cap_meta['max_neg_per_window_record']} seed={window_record_cap_meta['seed']} "
            f"pos {window_record_cap_meta['n_pos_triples_before_cap']}->{window_record_cap_meta['n_pos_triples_after_cap']} "
            f"neg {window_record_cap_meta['n_neg_triples_before_cap']}->{window_record_cap_meta['n_neg_triples_after_cap']}",
            flush=True,
        )

    ratio = float(args.interaction_sample_ratio)
    if ratio <= 0 or ratio > 1.0:
        raise ValueError("--interaction-sample-ratio  (0, 1] ")
    subsample_meta: Dict[str, object] = {}
    n_triples_before = len(triples)
    if ratio < 1.0:
        rng = random.Random(int(args.interaction_subsample_seed))
        idxs = list(range(n_triples_before))
        rng.shuffle(idxs)
        keep_n = max(1, int(n_triples_before * ratio))
        sel = idxs[:keep_n]
        triples = [triples[i] for i in sel]
        needed_tweet_ids = {str(t) for (_, t, _) in triples}
        subsample_meta = {
            "ratio": ratio,
            "seed": int(args.interaction_subsample_seed),
            "n_triples_before": int(n_triples_before),
            "n_triples_after": int(len(triples)),
        }
        print(
            f"[Samples] interaction subsample ratio={ratio} -> {len(triples)}/{n_triples_before} triples "
            f"(seed={args.interaction_subsample_seed}), tweets_needed={len(needed_tweet_ids)}",
            flush=True,
        )

    tweet_feat_map = load_tweet_features(needed_tweet_ids, args.tweet_emb_shards_dir)
    valid = [(uk, tid, lb) for (uk, tid, lb) in triples if tid in tweet_feat_map]
    if not valid:
        raise RuntimeError(" tweet_id ")
    print(f"[Samples] valid_after_tweet_feat={len(valid)}")

    window_filter_meta: Dict[str, Any] = {}
    if str(args.downstream_split).strip().lower() == "window" and int(args.window_min_rows_per_window) > 0:
        valid, window_filter_meta = filter_valid_triples_by_window_rowcount(
            valid, int(args.window_min_rows_per_window)
        )
        if not valid:
            raise RuntimeError(
                " --window-min-rows-per-window ，。"
            )
        print(
            f"[WindowFilter] min_rows={window_filter_meta['min_rows_per_window']} "
            f"triples {window_filter_meta['n_triples_before']}->{window_filter_meta['n_triples_after']} "
            f"dropped_windows={window_filter_meta['windows_dropped']}",
            flush=True,
        )

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    results: Dict[str, Dict] = {}

    # fixed sample order for all models
    ordered_user_keys = [uk for (uk, _, _) in valid]
    ordered_tweet_ids = [tid for (_, tid, _) in valid]
    y = torch.tensor([int(lb) for (_, _, lb) in valid], dtype=torch.long)
    print("[Stage] building tweet_bank + tweet_ix (dedup + index)...", flush=True)
    tweet_bank, tweet_ix = build_tweet_bank_and_ix(tweet_feat_map, ordered_tweet_ids)
    nuniq = int(tweet_bank.size(0))
    print(
        f"[Stage] tweet_bank shape={tuple(tweet_bank.shape)} tweet_ix={tuple(tweet_ix.shape)} "
        f"unique_tweets={nuniq}",
        flush=True,
    )
    del tweet_feat_map
    finetune_name_set = set(x.strip() for x in str(args.finetune_model_names).split(",") if x.strip())

    trd = int(args.tweet_reduce_dim)
    if trd < 0:
        raise ValueError("--tweet-reduce-dim ")

    mf_raw = str(args.models_first).strip()
    if mf_raw:
        priority = [x.strip() for x in mf_raw.split(",") if x.strip()]
        seen_order: set[str] = set()
        ordered_model_names: List[str] = []
        for n in priority:
            if n not in emb_maps:
                print(f"[Warn] --models-first ，: {n}", flush=True)
                continue
            if n not in seen_order:
                ordered_model_names.append(n)
                seen_order.add(n)
        for n in emb_maps.keys():
            if n not in seen_order:
                ordered_model_names.append(n)
                seen_order.add(n)
    else:
        ordered_model_names = list(emb_maps.keys())
    print(f"[ModelOrder] {' -> '.join(ordered_model_names)}", flush=True)

    dsplit = str(args.downstream_split).strip().lower()
    if dsplit == "window":
        _itr, _iva, _ite, wm = downstream_train_val_test_indices(ordered_user_keys, 0, "window")
        twr = wm.get("test_window_index_min_max") or wm.get("val_test_window_index_min_max")
        print(
            f"[DownstreamSplit] mode=window  train/val/test={wm['n_samples_train']}/"
            f"{wm['n_samples_val']}/{wm['n_samples_test']} distinct_windows={wm['n_distinct_windows']} "
            f"test_w_range={twr}",
            flush=True,
        )

    vt = bool(args.verbose_training)
    for model_name in ordered_model_names:
        emap = emb_maps[model_name]
        if vt:
            print(f"[{model_name}] stacking user_x ({len(ordered_user_keys)} rows)...", flush=True)
        user_x = torch.stack([emap[uk].float().view(-1) for uk in ordered_user_keys], dim=0)
        if vt:
            print(f"[{model_name}] user_x ready shape={tuple(user_x.shape)}", flush=True)

        rows: List[Dict[str, Any]] = []
        for sd in seeds:
            lp = f"[{model_name}] seed={sd}"
            if model_name in finetune_name_set:
                row = train_one_seed_finetune(
                    user_x=user_x,
                    tweet_bank=tweet_bank,
                    tweet_ix=tweet_ix,
                    y=y,
                    user_keys=ordered_user_keys,
                    seed=sd,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    head_lr=float(args.head_lr),
                    backbone_lr=float(args.backbone_lr),
                    freeze_backbone_epochs=int(args.freeze_backbone_epochs),
                    device=args.device,
                    tweet_reduce_dim=trd,
                    early_stopping_patience=int(args.early_stopping_patience),
                    early_stopping_min_delta=float(args.early_stopping_min_delta),
                    epoch_history_mode=str(args.epoch_history),
                    log_prefix=lp,
                    verbose_training=vt,
                    downstream_split=str(args.downstream_split),
                    max_train_batches=int(args.max_train_batches),
                )
            else:
                row = train_one_seed(
                    user_x=user_x,
                    tweet_bank=tweet_bank,
                    tweet_ix=tweet_ix,
                    y=y,
                    user_keys=ordered_user_keys,
                    seed=sd,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    device=args.device,
                    tweet_reduce_dim=trd,
                    early_stopping_patience=int(args.early_stopping_patience),
                    early_stopping_min_delta=float(args.early_stopping_min_delta),
                    epoch_history_mode=str(args.epoch_history),
                    log_prefix=lp,
                    verbose_training=vt,
                    downstream_split=str(args.downstream_split),
                    max_train_batches=int(args.max_train_batches),
                )
            save_emb = str(args.save_finetuned_dynamic_embedding).strip()
            if model_name in finetune_name_set and isinstance(row, dict):
                exp = row.pop("_finetune_export", None)
                if save_emb and isinstance(exp, dict) and "uniq_users" in exp and "user_backbone_weight" in exp:
                    base = Path(save_emb)
                    if "{seed}" in save_emb:
                        out_pt = Path(save_emb.format(seed=int(sd)))
                    else:
                        out_pt = base.parent / f"{base.stem}.seed{int(sd)}{base.suffix}"
                    out_pt.parent.mkdir(parents=True, exist_ok=True)
                    emb_dict = window_keys_to_finetuned_emb_dict(
                        exp["uniq_users"],
                        exp["user_backbone_weight"],
                        ordered_user_keys,
                    )
                    meta = {
                        "source_model": str(model_name),
                        "finetune_seed": int(sd),
                        "n_window_keys": len(emb_dict),
                        "user_dim": int(exp["user_backbone_weight"].size(1)),
                    }
                    torch.save({"embeddings": emb_dict, "meta": meta}, str(out_pt))
                    print(f"[SaveFinetunedEmb] {out_pt} keys={len(emb_dict)} dim={meta['user_dim']}", flush=True)
            rows.append(row)
            es_note = ""
            if int(args.early_stopping_patience) > 0:
                es_note = (
                    f" ep_run={int(row['epochs_run'])} best_ep={int(row['best_epoch'])} "
                    f"best_val_f1={row['best_val_f1']:.4f} early_stop={int(row['early_stopped'])}"
                )
            print(
                f"[{model_name}] seed={sd} acc={row['acc']:.4f} "
                f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
                f"f1={row['f1']:.4f} bce={row['bce']:.6f} "
                f"auc={row['auc']:.4f} hit@10={row['hit@10']:.4f} ndcg@10={row['ndcg@10']:.4f}"
                f"{es_note}",
                flush=True,
            )

        row_out: Dict[str, object] = {
            "status": "ok",
            "downstream_split_mode": str(args.downstream_split),
            "num_common_user_keys": len(common_keys),
            "num_samples_used": len(valid),
            "num_unique_tweets": len(set(ordered_tweet_ids)),
            "seeds": seeds,
            "per_seed_test": rows,
            "aggregate_test": aggregate(rows),
            "tweet_downstream": {
                "tweet_reduce_dim": trd,
                "tweet_dim": int(tweet_bank.size(1)),
                "user_dim": int(user_x.size(1)),
            },
        }
        if split_filter_meta:
            row_out["pretrain_split_filter"] = split_filter_meta
        if subsample_meta:
            row_out["interaction_subsample"] = subsample_meta
        if window_filter_meta:
            row_out["window_min_rows_filter"] = window_filter_meta
        if window_record_cap_meta:
            row_out["window_record_pos_neg_cap"] = window_record_cap_meta
        if int(args.max_train_batches) > 0:
            row_out["max_train_batches"] = int(args.max_train_batches)
        if int(args.early_stopping_patience) > 0:
            row_out["early_stopping"] = {
                "monitor": "val_f1",
                "patience": int(args.early_stopping_patience),
                "min_delta": float(args.early_stopping_min_delta),
                "max_epochs": int(args.epochs),
            }
        if str(args.epoch_history) != "off":
            row_out["epoch_history_config"] = {"mode": str(args.epoch_history)}
        results[model_name] = row_out

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out}", flush=True)


if __name__ == "__main__":
    main()

