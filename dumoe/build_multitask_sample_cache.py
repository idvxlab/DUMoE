#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from time_slice_data_loader import (
    build_split_entries,
    build_user_tweet_ids_map,
    collect_needed_msg_ids_from_entries,
    load_time_slice_labels_from_user_dirs,
    load_user_features_from_pt,
)
from time_slice_msg_feature_loader import load_msg_features_from_compact_shards
from pipeline_multitask_vs_baselines import TARGET_CATEGORIES, RecSampleBank, _entry_has_context


def _default_label_user_dirs() -> str:
    return ""


def _build_feature_table_and_index(
    msg_features_dict: Dict[str, torch.Tensor],
    msg_feat_dim: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    table = torch.zeros((len(msg_features_dict) + 1, msg_feat_dim), dtype=torch.float16)
    id_to_idx: Dict[str, int] = {}
    row = 1
    for mid, feat in msg_features_dict.items():
        t = feat if isinstance(feat, torch.Tensor) else torch.tensor(feat)
        table[row] = t.to(dtype=torch.float16)
        id_to_idx[str(mid)] = row
        row += 1
    return table, id_to_idx


def _fill_fixed_ids_and_mask(
    ordered_tweet_ids: List[str],
    start: int,
    length: int,
    id_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.zeros((length,), dtype=torch.int32)
    mask = torch.ones((length,), dtype=torch.bool)
    valid_count = 0
    for i, mid in enumerate(ordered_tweet_ids[start:start + length]):
        idx = id_to_idx.get(str(mid), 0)
        if idx > 0:
            ids[i] = idx
            mask[i] = False
            valid_count += 1
    if valid_count == 0 and length > 0:
        mask[0] = False
    return ids, mask


def _fill_fixed_short_ids(
    ordered_tweet_ids: List[str],
    start: int,
    length: int,
    id_to_idx: Dict[str, int],
) -> torch.Tensor:
    ids = torch.zeros((length,), dtype=torch.int32)
    for i, mid in enumerate(ordered_tweet_ids[start:start + length]):
        ids[i] = id_to_idx.get(str(mid), 0)
    return ids


def _save_payload(path: Path, payload, *, legacy_serialization: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path), _use_new_zipfile_serialization=not legacy_serialization)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multitask sample cache for train/val/test.")
    parser.add_argument("--label-user-dirs", type=str, default=_default_label_user_dirs())
    parser.add_argument("--labels-cache", type=str, default="dataset/local/labels_cache.json")
    parser.add_argument("--intersection-cache", type=str, default="dataset/local/user_intersection.json")
    parser.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    parser.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    parser.add_argument("--samples-dir", type=str, default="dataset/local/message_user_samples")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--short-recent", type=int, default=30)
    parser.add_argument("--long-recent", type=int, default=300)
    parser.add_argument("--use-cached-labels", action="store_true")
    parser.add_argument("--use-cached-needed-msg-ids", action="store_true")
    parser.add_argument("--needed-msg-ids-cache", type=str, default="")
    parser.add_argument("--only-msg-table", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_user_dirs = [x.strip() for x in args.label_user_dirs.split(",") if x.strip()]
    print("[Cache] loading labels...", flush=True)
    labels_by_user, csv_path_by_user = load_time_slice_labels_from_user_dirs(
        user_dirs=label_user_dirs,
        cache_path=args.labels_cache,
        use_cache=bool(args.use_cached_labels),
        progress_every=0,
    )

    print("[Cache] loading user features...", flush=True)
    user_feature_map, user_feat_dim = load_user_features_from_pt(args.user_profile_pt)

    intersection_users: Optional[Set[str]] = None
    if args.intersection_cache and os.path.exists(args.intersection_cache):
        try:
            payload = json.loads(Path(args.intersection_cache).read_text(encoding="utf-8"))
            intersection_users = set(str(x) for x in payload.get("intersection_user_ids", []))
        except Exception:
            intersection_users = None

    if intersection_users is not None:
        labels_by_user = {u: labels_by_user[u] for u in labels_by_user if str(u) in intersection_users}
        csv_path_by_user = {u: csv_path_by_user[u] for u in csv_path_by_user if str(u) in intersection_users}

    labels_by_user = {u: labels_by_user[u] for u in labels_by_user if u in user_feature_map and u in csv_path_by_user}
    user_feature_map = {u: user_feature_map[u] for u in labels_by_user}
    csv_path_by_user = {u: csv_path_by_user[u] for u in labels_by_user}

    print("[Cache] building ordered tweet ids...", flush=True)
    user_tweet_ids_map = build_user_tweet_ids_map(csv_path_by_user)
    labels_by_user = {u: labels_by_user[u] for u in labels_by_user if u in user_tweet_ids_map and len(user_tweet_ids_map[u]) > 0}
    user_feature_map = {u: user_feature_map[u] for u in labels_by_user}
    user_tweet_ids_map = {u: user_tweet_ids_map[u] for u in labels_by_user}

    label_to_id = {label: idx for idx, label in enumerate(TARGET_CATEGORIES)}
    train_entries, val_entries, test_entries = build_split_entries(labels_by_user, label_to_id)
    train_entries = [e for e in train_entries if _entry_has_context(e, user_tweet_ids_map, args.window_size)]
    val_entries = [e for e in val_entries if _entry_has_context(e, user_tweet_ids_map, args.window_size)]
    test_entries = [e for e in test_entries if _entry_has_context(e, user_tweet_ids_map, args.window_size)]

    rec_bank = RecSampleBank(args.samples_dir)

    needed_cache = args.needed_msg_ids_cache.strip() or str(output_dir / "needed_msg_ids.pt")
    needed_msg_ids: Optional[Set[str]] = None
    if args.use_cached_needed_msg_ids and os.path.exists(needed_cache):
        cached_ids = torch.load(needed_cache, map_location="cpu", weights_only=False)
        if isinstance(cached_ids, (list, tuple, set)):
            needed_msg_ids = set(str(x) for x in cached_ids)
            print(f"[Cache] loaded needed ids: {len(needed_msg_ids)}", flush=True)

    if needed_msg_ids is None:
        needed_msg_ids = collect_needed_msg_ids_from_entries(
            train_entries + val_entries + test_entries,
            user_tweet_ids_map,
            window_size=args.window_size,
            long_recent=args.long_recent,
        )
    needed_msg_ids.update(rec_bank.all_item_ids)
    torch.save(sorted(needed_msg_ids), needed_cache)

    print("[Cache] loading message features...", flush=True)
    msg_features_dict, msg_feat_dim = load_msg_features_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir,
        needed_msg_ids=needed_msg_ids,
        verbose=True,
    )
    msg_table, msg_id_to_idx = _build_feature_table_and_index(msg_features_dict, msg_feat_dim)
    # Large tensors were previously corrupted by the default zip serializer on the target host.
    _save_payload(output_dir / "msg_feature_table_fp16.pt", msg_table, legacy_serialization=True)
    print(f"[Cache] saved msg feature table: {tuple(msg_table.shape)}", flush=True)
    del msg_features_dict

    if args.only_msg_table:
        print("[OK] rebuilt msg_feature_table_fp16.pt only", flush=True)
        return

    rec_bank.build_tensor_cache(msg_id_to_row=msg_id_to_idx)
    _save_payload(
        output_dir / "rec_bank_cache.pt",
        {
            "key_to_row": rec_bank.key_to_row,
            "pos_idx_table_cpu": rec_bank.pos_idx_table_cpu,
            "neg_idx_table_cpu": rec_bank.neg_idx_table_cpu,
            "pos_len_cpu": rec_bank.pos_len_cpu,
            "neg_len_cpu": rec_bank.neg_len_cpu,
        },
    )

    long_no_month_len = max(1, int(args.long_recent) - int(args.short_recent))

    def build_split_payload(entries: List[Tuple[str, int, str, int]], split_tag: str) -> Dict[str, object]:
        items = {
            "x_explicit": [],
            "short_msg_indices": [],
            "long_msg_indices": [],
            "long_no_month_msg_indices": [],
            "src_key_padding_mask": [],
            "src_key_padding_mask_no_month": [],
            "labels": [],
            "user_ids": [],
            "window_indices": [],
            "months": [],
            "split_tags": [],
        }
        for uid, window_idx, month, label in entries:
            ordered = user_tweet_ids_map.get(uid, [])
            start = int(window_idx) * int(args.window_size)
            user_feat = user_feature_map[str(uid)]
            if not isinstance(user_feat, torch.Tensor):
                user_feat = torch.tensor(user_feat, dtype=torch.float32)
            user_feat = user_feat.float().view(-1)
            if user_feat.numel() < user_feat_dim:
                user_feat = torch.cat([user_feat, torch.zeros((user_feat_dim - user_feat.numel(),), dtype=user_feat.dtype)], dim=0)
            elif user_feat.numel() > user_feat_dim:
                user_feat = user_feat[:user_feat_dim]

            short_idx = _fill_fixed_short_ids(ordered, start, int(args.short_recent), msg_id_to_idx)
            long_idx, long_mask = _fill_fixed_ids_and_mask(ordered, start, int(args.long_recent), msg_id_to_idx)
            long_no_idx, long_no_mask = _fill_fixed_ids_and_mask(
                ordered,
                start + int(args.short_recent),
                long_no_month_len,
                msg_id_to_idx,
            )

            items["x_explicit"].append(user_feat.to(dtype=torch.float16))
            items["short_msg_indices"].append(short_idx)
            items["long_msg_indices"].append(long_idx)
            items["long_no_month_msg_indices"].append(long_no_idx)
            items["src_key_padding_mask"].append(long_mask)
            items["src_key_padding_mask_no_month"].append(long_no_mask)
            items["labels"].append(int(label))
            items["user_ids"].append(str(uid))
            items["window_indices"].append(int(window_idx))
            items["months"].append(str(month))
            items["split_tags"].append(split_tag)

        return {
            "x_explicit": torch.stack(items["x_explicit"], dim=0).to(dtype=torch.float16),
            "short_msg_indices": torch.stack(items["short_msg_indices"], dim=0),
            "long_msg_indices": torch.stack(items["long_msg_indices"], dim=0),
            "long_no_month_msg_indices": torch.stack(items["long_no_month_msg_indices"], dim=0),
            "src_key_padding_mask": torch.stack(items["src_key_padding_mask"], dim=0),
            "src_key_padding_mask_no_month": torch.stack(items["src_key_padding_mask_no_month"], dim=0),
            "labels": torch.tensor(items["labels"], dtype=torch.long),
            "user_ids": items["user_ids"],
            "window_indices": items["window_indices"],
            "months": items["months"],
            "split_tags": items["split_tags"],
            "meta": {
                "num_samples": len(items["labels"]),
                "user_feat_dim": user_feat_dim,
                "msg_feat_dim": msg_feat_dim,
                "short_recent": int(args.short_recent),
                "long_recent": int(args.long_recent),
                "window_size": int(args.window_size),
            },
        }

    split_map = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }
    counts = {}
    for split_name, entries in split_map.items():
        print(f"[Cache] building split={split_name} samples={len(entries)}", flush=True)
        payload = build_split_payload(entries, split_name)
        _save_payload(output_dir / f"{split_name}.pt", payload)
        counts[split_name] = len(entries)

    meta = {
        "train_path": str(output_dir / "train.pt"),
        "val_path": str(output_dir / "val.pt"),
        "test_path": str(output_dir / "test.pt"),
        "msg_feature_table_path": str(output_dir / "msg_feature_table_fp16.pt"),
        "rec_bank_cache_path": str(output_dir / "rec_bank_cache.pt"),
        "counts": counts,
        "user_feat_dim": user_feat_dim,
        "msg_feat_dim": msg_feat_dim,
        "short_recent": int(args.short_recent),
        "long_recent": int(args.long_recent),
        "window_size": int(args.window_size),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] sample cache built: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
