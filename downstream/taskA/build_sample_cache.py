#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMOE_DIR = REPO_ROOT / "dumoe"
for _p in (str(REPO_ROOT), str(DUMOE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from time_slice_data_loader import (
    build_user_tweet_ids_map,
    collect_needed_msg_ids_from_entries,
    load_time_slice_labels_from_user_dirs,
    load_user_features_from_pt,
)
from time_slice_msg_feature_loader import load_msg_features_from_compact_shards


TARGET_CATEGORIES = [
    "Entertainment", "Sports", "Technology", "Business", "Education", "Gaming", "Politics", "Art",
    "Culture", "Science", "Food", "Daily Life", "Fashion & Beauty", "Health & Fitness", "News",
]


def _default_label_user_dirs() -> str:
    return ""


def normalize_label(x: str) -> str:
    return "".join(str(x or "").split())


def build_label_map(categories: List[str]) -> Dict[str, int]:
    return {normalize_label(label): idx for idx, label in enumerate(categories)}


def split_tag_from_window(window_idx: int, train_months: int = 8, val_months: int = 2, test_months: int = 2) -> str:
    if window_idx < train_months:
        return "train"
    if window_idx < train_months + val_months:
        return "val"
    if window_idx < train_months + val_months + test_months:
        return "test"
    return "other"


def load_intersection_users(path: str) -> Optional[Set[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = payload.get("intersection_user_ids", [])
    return set(str(x) for x in ids)


def build_entries_all(
    labels_by_user: Dict[str, List[Dict]],
    label_to_id: Dict[str, int],
) -> List[Tuple[str, int, str, int]]:
    entries: List[Tuple[str, int, str, int]] = []
    for uid, months in labels_by_user.items():
        for idx, m in enumerate(months):
            if not isinstance(m, dict):
                continue
            label_str = normalize_label(m.get("label", ""))
            if label_str not in label_to_id:
                continue
            month = str(m.get("month", f"w{idx:03d}"))
            entries.append((str(uid), int(idx), month, int(label_to_id[label_str])))
    return entries


def filter_entries_with_context(
    entries: List[Tuple[str, int, str, int]],
    user_tweet_ids_map: Dict[str, List[str]],
    window_size: int,
) -> List[Tuple[str, int, str, int]]:
    out: List[Tuple[str, int, str, int]] = []
    for entry in entries:
        uid, window_idx, _, _ = entry
        start = int(window_idx) * int(window_size)
        if start < len(user_tweet_ids_map.get(uid, [])):
            out.append(entry)
    return out


def _build_feature_table_and_index(
    msg_features_dict: Dict[str, torch.Tensor],
    msg_feat_dim: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    # row 0 is always zero vector for missing ids / padded ids
    table = torch.zeros((len(msg_features_dict) + 1, msg_feat_dim), dtype=torch.float16)
    id_to_idx: Dict[str, int] = {}

    i = 1
    for mid, feat in msg_features_dict.items():
        f = feat if isinstance(feat, torch.Tensor) else torch.tensor(feat)
        table[i] = f.to(dtype=torch.float16)
        id_to_idx[str(mid)] = i
        i += 1

    return table, id_to_idx


def _fill_fixed_ids_and_mask(
    ordered_tweet_ids: List[str],
    start: int,
    length: int,
    id_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.zeros((length,), dtype=torch.int32)
    mask = torch.ones((length,), dtype=torch.bool)

    selected = ordered_tweet_ids[start:start + length]
    valid_count = 0
    for i, mid in enumerate(selected):
        idx = id_to_idx.get(str(mid), 0)
        if idx > 0:
            ids[i] = idx
            mask[i] = False
            valid_count += 1

    # If none found, keep first one as zero-row (vector zeros) but mark it valid.
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
    selected = ordered_tweet_ids[start:start + length]
    for i, mid in enumerate(selected):
        ids[i] = id_to_idx.get(str(mid), 0)
    return ids


def _save_shard(
    shard_path: Path,
    items: Dict[str, List],
    user_feat_dim: int,
    short_recent: int,
    long_recent: int,
) -> int:
    n = len(items["labels"])
    if n == 0:
        return 0

    payload = {
        "x_explicit": torch.stack(items["x_explicit"], dim=0).to(dtype=torch.float16),
        "short_msg_indices": torch.stack(items["short_msg_indices"], dim=0),
        "long_msg_indices": torch.stack(items["long_msg_indices"], dim=0),
        "long_no_month_msg_indices": torch.stack(items["long_no_month_msg_indices"], dim=0),
        "src_key_padding_mask": torch.stack(items["src_key_padding_mask"], dim=0),
        "src_key_padding_mask_no_month": torch.stack(items["src_key_padding_mask_no_month"], dim=0),
        "labels": torch.tensor(items["labels"], dtype=torch.long),
        "user_ids": list(items["user_ids"]),
        "window_indices": list(items["window_indices"]),
        "months": list(items["months"]),
        "split_tags": list(items["split_tags"]),
        "meta": {
            "num_samples": n,
            "user_feat_dim": user_feat_dim,
            "short_recent": short_recent,
            "long_recent": long_recent,
        },
    }

    torch.save(payload, str(shard_path))
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TaskA sample cache (all valid windows)")
    parser.add_argument("--label-user-dirs", type=str, default=_default_label_user_dirs())
    parser.add_argument("--labels-cache", type=str, default="dataset/local/labels_cache.json")
    parser.add_argument("--intersection-cache", type=str, default="dataset/local/user_intersection.json")
    parser.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    parser.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")

    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--short-recent", type=int, default=30)
    parser.add_argument("--long-recent", type=int, default=300)

    parser.add_argument("--use-cached-labels", action="store_true")
    parser.add_argument("--label-parse-progress-every", type=int, default=500)
    parser.add_argument("--needed-msg-ids-cache", type=str, default="")
    parser.add_argument("--use-cached-needed-msg-ids", action="store_true")

    parser.add_argument("--output-dir", type=str, default="./TASKA")
    parser.add_argument("--cache-subdir", type=str, default="cache/taskA_samples_all_windows")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / args.cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    table_path = cache_dir / "msg_feature_table_fp16.pt"
    meta_path = cache_dir / "meta.json"

    needed_ids_cache = args.needed_msg_ids_cache.strip()
    if not needed_ids_cache:
        needed_ids_cache = str(output_dir / "taskA_needed_msg_ids_all_windows.pt")

    label_user_dirs = [x.strip() for x in args.label_user_dirs.split(",") if x.strip()]

    print("[TaskA-Cache] loading labels...")
    labels_by_user, csv_path_by_user = load_time_slice_labels_from_user_dirs(
        user_dirs=label_user_dirs,
        cache_path=args.labels_cache,
        use_cache=bool(args.use_cached_labels),
        progress_every=max(0, int(args.label_parse_progress_every)),
    )

    inter_users = load_intersection_users(args.intersection_cache)
    if inter_users is not None:
        labels_by_user = {u: v for u, v in labels_by_user.items() if str(u) in inter_users}
        csv_path_by_user = {u: v for u, v in csv_path_by_user.items() if str(u) in inter_users}
        print(f"[TaskA-Cache] intersection filter applied: users={len(labels_by_user)}")

    print("[TaskA-Cache] loading user profile features...")
    user_feature_map, user_feat_dim = load_user_features_from_pt(args.user_profile_pt)

    labels_by_user = {u: labels_by_user[u] for u in labels_by_user.keys() if u in user_feature_map}
    csv_path_by_user = {u: csv_path_by_user[u] for u in labels_by_user.keys() if u in csv_path_by_user}
    user_feature_map = {u: user_feature_map[u] for u in labels_by_user.keys()}

    print("[TaskA-Cache] building ordered tweet id map...")
    user_tweet_ids_map = build_user_tweet_ids_map(csv_path_by_user)

    labels_by_user = {u: labels_by_user[u] for u in labels_by_user.keys() if u in user_tweet_ids_map and len(user_tweet_ids_map[u]) > 0}
    user_feature_map = {u: user_feature_map[u] for u in labels_by_user.keys()}
    user_tweet_ids_map = {u: user_tweet_ids_map[u] for u in labels_by_user.keys()}

    label_to_id = build_label_map(TARGET_CATEGORIES)

    entries_all = build_entries_all(labels_by_user, label_to_id)
    entries_all = filter_entries_with_context(entries_all, user_tweet_ids_map, window_size=args.window_size)
    print(f"[TaskA-Cache] total valid windows={len(entries_all)}")

    needed_msg_ids: Optional[Set[str]] = None
    if args.use_cached_needed_msg_ids and os.path.exists(needed_ids_cache):
        cached_ids = torch.load(needed_ids_cache, map_location="cpu")
        if isinstance(cached_ids, (list, tuple, set)):
            needed_msg_ids = set(str(x) for x in cached_ids)
            print(f"[TaskA-Cache] loaded cached needed msg ids: {len(needed_msg_ids)}")

    if needed_msg_ids is None:
        needed_msg_ids = collect_needed_msg_ids_from_entries(
            entries=entries_all,
            user_tweet_ids_map=user_tweet_ids_map,
            window_size=args.window_size,
            long_recent=args.long_recent,
        )
        torch.save(sorted(needed_msg_ids), needed_ids_cache)
        print(f"[TaskA-Cache] collected needed msg ids: {len(needed_msg_ids)}")
        print(f"[TaskA-Cache] saved needed msg ids cache: {needed_ids_cache}")

    if args.resume and meta_path.exists() and table_path.exists():
        print("[TaskA-Cache] resume mode: reusing existing msg feature table")
        table = torch.load(str(table_path), map_location="cpu")
        if not isinstance(table, torch.Tensor) or table.dim() != 2:
            raise RuntimeError("existing msg feature table format invalid")
        msg_feat_dim = int(table.size(1))

        # Rebuild msg_id->idx mapping from shards to keep consistency after resume.
        # This still needs msg features metadata but avoids re-writing table file.
        print("[TaskA-Cache] rebuilding msg id index map...")
        msg_features_dict, _ = load_msg_features_from_compact_shards(
            shards_dir=args.tweet_emb_shards_dir,
            needed_msg_ids=needed_msg_ids,
            verbose=True,
        )
        _, msg_id_to_idx = _build_feature_table_and_index(msg_features_dict, msg_feat_dim)
        del msg_features_dict
    else:
        print("[TaskA-Cache] loading tweet feature shards...")
        msg_features_dict, msg_feat_dim = load_msg_features_from_compact_shards(
            shards_dir=args.tweet_emb_shards_dir,
            needed_msg_ids=needed_msg_ids,
            verbose=True,
        )
        print(f"[TaskA-Cache] loaded msg features: {len(msg_features_dict)} dim={msg_feat_dim}")

        print("[TaskA-Cache] building compact message feature table (fp16)...")
        table, msg_id_to_idx = _build_feature_table_and_index(msg_features_dict, msg_feat_dim)
        torch.save(table, str(table_path))
        print(f"[TaskA-Cache] saved msg feature table: {table_path} shape={tuple(table.shape)}")
        del msg_features_dict

    # Resume from existing shards if requested.
    existing = sorted(cache_dir.glob("shard_*.pt")) if args.resume else []
    done_shards = len(existing)
    start_idx = done_shards * int(args.shard_size)
    if start_idx > 0:
        print(f"[TaskA-Cache] resume from shard={done_shards} sample_index={start_idx}")

    shard_items = {
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

    total_written = start_idx
    shard_id = done_shards

    for i in range(start_idx, len(entries_all)):
        uid, window_idx, month, label = entries_all[i]
        ordered = user_tweet_ids_map.get(uid, [])
        start = int(window_idx) * int(args.window_size)

        x_explicit = user_feature_map[uid].to(dtype=torch.float16)

        short_idx = _fill_fixed_short_ids(
            ordered_tweet_ids=ordered,
            start=start,
            length=int(args.short_recent),
            id_to_idx=msg_id_to_idx,
        )

        long_idx, long_mask = _fill_fixed_ids_and_mask(
            ordered_tweet_ids=ordered,
            start=start,
            length=int(args.long_recent),
            id_to_idx=msg_id_to_idx,
        )

        # Keep TaskA cache aligned with the original training dataset logic:
        # long_ids_no_month = ordered[start + short_recent : start + long_recent]
        # So its effective length is (long_recent - short_recent), not long_recent.
        long_no_month_len = max(1, int(args.long_recent) - int(args.short_recent))
        long_no_month_idx, long_no_month_mask = _fill_fixed_ids_and_mask(
            ordered_tweet_ids=ordered,
            start=start + int(args.short_recent),
            length=long_no_month_len,
            id_to_idx=msg_id_to_idx,
        )

        shard_items["x_explicit"].append(x_explicit)
        shard_items["short_msg_indices"].append(short_idx)
        shard_items["long_msg_indices"].append(long_idx)
        shard_items["long_no_month_msg_indices"].append(long_no_month_idx)
        shard_items["src_key_padding_mask"].append(long_mask)
        shard_items["src_key_padding_mask_no_month"].append(long_no_month_mask)
        shard_items["labels"].append(int(label))
        shard_items["user_ids"].append(str(uid))
        shard_items["window_indices"].append(int(window_idx))
        shard_items["months"].append(str(month))
        shard_items["split_tags"].append(split_tag_from_window(int(window_idx)))

        if len(shard_items["labels"]) >= int(args.shard_size):
            shard_path = cache_dir / f"shard_{shard_id:05d}.pt"
            n = _save_shard(
                shard_path=shard_path,
                items=shard_items,
                user_feat_dim=int(user_feat_dim),
                short_recent=int(args.short_recent),
                long_recent=int(args.long_recent),
            )
            total_written += n
            print(f"[TaskA-Cache] wrote {shard_path.name} samples={n} total_written={total_written}")
            shard_id += 1
            for k in shard_items:
                shard_items[k] = []

    if len(shard_items["labels"]) > 0:
        shard_path = cache_dir / f"shard_{shard_id:05d}.pt"
        n = _save_shard(
            shard_path=shard_path,
            items=shard_items,
            user_feat_dim=int(user_feat_dim),
            short_recent=int(args.short_recent),
            long_recent=int(args.long_recent),
        )
        total_written += n
        print(f"[TaskA-Cache] wrote {shard_path.name} samples={n} total_written={total_written}")

    final_shards = sorted(cache_dir.glob("shard_*.pt"))
    meta = {
        "cache_version": "taskA_sample_cache_v1_indices",
        "note": "Index-based cache. Encode with downstream/taskA/export_embeddings.py",
        "num_samples": int(total_written),
        "num_shards": len(final_shards),
        "shard_size": int(args.shard_size),
        "user_feat_dim": int(user_feat_dim),
        "msg_feat_dim": int(table.size(1)),
        "short_recent": int(args.short_recent),
        "long_recent": int(args.long_recent),
        "window_size": int(args.window_size),
        "msg_feature_table": str(table_path),
        "needed_msg_ids_cache": str(needed_ids_cache),
        "labels_cache": str(args.labels_cache),
        "intersection_cache": str(args.intersection_cache),
        "label_to_id": {k: int(v) for k, v in label_to_id.items()},
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[TaskA-Cache] done. shards={len(final_shards)} samples={total_written}")
    print(f"[TaskA-Cache] meta: {meta_path}")


if __name__ == "__main__":
    main()
