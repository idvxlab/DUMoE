#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared window-table I/O for Task A and Task B (not Task-A-only).

Parse labels, read per-user tweet id sequences, and build window entries.
No training.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DUMOE_DIR = REPO_ROOT / "dumoe"
for _p in (str(REPO_ROOT), str(DUMOE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from time_slice_data_loader import (  # type: ignore
    build_user_tweet_ids_map,
    load_time_slice_labels_from_user_dirs,
    load_user_features_from_pt,
)


TARGET_CATEGORIES: List[str] = [
    "Entertainment",
    "Sports",
    "Technology",
    "Business",
    "Education",
    "Gaming",
    "Politics",
    "Art",
    "Culture",
    "Science",
    "Food",
    "Daily Life",
    "Fashion & Beauty",
    "Health & Fitness",
    "News",
]


def normalize_label(x: str) -> str:
    return "".join(str(x or "").split())


def build_label_map(categories: Sequence[str]) -> Dict[str, int]:
    return {normalize_label(label): idx for idx, label in enumerate(categories)}


def split_tag_from_window(
    window_idx: int, train_months: int = 8, val_months: int = 2, test_months: int = 2
) -> str:
    if window_idx < train_months:
        return "train"
    if window_idx < train_months + val_months:
        return "val"
    if window_idx < train_months + val_months + test_months:
        return "test"
    return "other"


@dataclass(frozen=True)
class WindowEntry:
    user_id: str
    window_idx: int
    month: str
    label_id: int
    split_tag: str

    @property
    def user_id_t(self) -> str:
        return f"{self.user_id}__w{self.window_idx:03d}"


def default_label_user_dirs() -> List[str]:
    return []


def load_intersection_users(path: str) -> Optional[Set[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = payload.get("intersection_user_ids", [])
    return set(str(x) for x in ids)


def build_entries_all_windows(
    label_user_dirs: Sequence[str],
    labels_cache: str,
    intersection_cache: str,
    user_profile_pt: str,
    window_size: int = 30,
    use_cached_labels: bool = False,
    label_parse_progress_every: int = 500,
) -> Tuple[List[WindowEntry], Dict[str, List[str]], Dict[str, int]]:
    """
    Returns:
    - entries: “”
    - user_tweet_ids_map: user ->  tweet_id 
    - label_to_id: （ label_id）
    """
    labels_by_user, csv_path_by_user = load_time_slice_labels_from_user_dirs(
        user_dirs=list(label_user_dirs),
        cache_path=labels_cache,
        use_cache=bool(use_cached_labels),
        progress_every=max(0, int(label_parse_progress_every)),
    )

    inter_users = load_intersection_users(intersection_cache)
    if inter_users is not None:
        labels_by_user = {u: v for u, v in labels_by_user.items() if str(u) in inter_users}
        csv_path_by_user = {u: v for u, v in csv_path_by_user.items() if str(u) in inter_users}

    user_feature_map, _ = load_user_features_from_pt(user_profile_pt)
    labels_by_user = {u: labels_by_user[u] for u in labels_by_user.keys() if u in user_feature_map}
    csv_path_by_user = {u: csv_path_by_user[u] for u in labels_by_user.keys() if u in csv_path_by_user}

    user_tweet_ids_map = build_user_tweet_ids_map(csv_path_by_user)
    labels_by_user = {
        u: labels_by_user[u] for u in labels_by_user.keys() if u in user_tweet_ids_map and len(user_tweet_ids_map[u]) > 0
    }
    user_tweet_ids_map = {u: user_tweet_ids_map[u] for u in labels_by_user.keys()}

    label_to_id = build_label_map(TARGET_CATEGORIES)

    entries: List[WindowEntry] = []
    for uid, months in labels_by_user.items():
        for idx, m in enumerate(months):
            if not isinstance(m, dict):
                continue
            label_str = normalize_label(m.get("label", ""))
            if label_str not in label_to_id:
                continue
            month = str(m.get("month", f"w{idx:03d}"))
            start = int(idx) * int(window_size)
            if start >= len(user_tweet_ids_map.get(uid, [])):
                continue
            entries.append(
                WindowEntry(
                    user_id=str(uid),
                    window_idx=int(idx),
                    month=month,
                    label_id=int(label_to_id[label_str]),
                    split_tag=split_tag_from_window(int(idx)),
                )
            )

    return entries, user_tweet_ids_map, label_to_id


def write_samples_csv(entries: Sequence[WindowEntry], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id_t", "label_id", "user_id", "window_idx", "month", "split_tag"])
        for e in entries:
            w.writerow([e.user_id_t, e.label_id, e.user_id, e.window_idx, e.month, e.split_tag])


def write_samples_jsonl(entries: Sequence[WindowEntry], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(
                json.dumps(
                    {
                        "user_id_t": e.user_id_t,
                        "label_id": int(e.label_id),
                        "user_id": e.user_id,
                        "window_idx": int(e.window_idx),
                        "month": e.month,
                        "split_tag": e.split_tag,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_samples_csv(path: str) -> List[WindowEntry]:
    out: List[WindowEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                WindowEntry(
                    user_id=str(row["user_id"]),
                    window_idx=int(row["window_idx"]),
                    month=str(row.get("month", "")),
                    label_id=int(row["label_id"]),
                    split_tag=str(row.get("split_tag", "")),
                )
            )
    return out

