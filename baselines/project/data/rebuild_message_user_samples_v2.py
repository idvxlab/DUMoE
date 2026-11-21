#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

TIME_FMT = "%a %b %d %H:%M:%S +0000 %Y"


@dataclass
class TweetRow:
    ts: float
    tweet_id: str
    user_id: str


@dataclass
class WindowBlock:
    target_window_idx: int
    source_window_idx: int
    pos_item_ids: List[str]
    start_ts: float
    end_ts: float


def _parse_csv_tweets(csv_path: str, user_id: str) -> List[Tuple[datetime, str]]:
    rows: List[Tuple[datetime, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            for row in reader:
                if not row:
                    continue
                norm = {(k or "").strip("\ufeff"): v for k, v in row.items()}
                tid = str(norm.get("tweet_id", "") or "").strip()
                publish_time = str(norm.get("publish_time", "") or "").strip()
                if not tid or not publish_time:
                    continue
                try:
                    dt = datetime.strptime(publish_time, TIME_FMT)
                except Exception:
                    continue
                rows.append((dt, tid))
    except Exception:
        return []
    # reverse chronological: smaller window index -> more recent
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def _build_user_windows(rows: List[Tuple[datetime, str]], window_size: int) -> List[WindowBlock]:
    out: List[WindowBlock] = []
    n = len(rows)
    num_windows = (n + window_size - 1) // window_size
    for t in range(num_windows):
        start = t * window_size
        end = min((t + 1) * window_size, n)
        block = rows[start:end]
        if len(block) == 0:
            continue
        # strict format: use user_id_{t-1}. In this reverse indexing, t-1 => t+1.
        source_idx = t + 1
        pos_ids = [tid for _, tid in block]
        ts_vals = [dt.timestamp() for dt, _ in block]
        out.append(
            WindowBlock(
                target_window_idx=t,
                source_window_idx=source_idx,
                pos_item_ids=pos_ids,
                start_ts=min(ts_vals),
                end_ts=max(ts_vals),
            )
        )
    return out


def _sample_negatives_from_time_range(
    all_rows_sorted_asc: List[TweetRow],
    all_ts_asc: List[float],
    user_id: str,
    pos_ids: List[str],
    start_ts: float,
    end_ts: float,
    rng: random.Random,
) -> List[str]:
    l = bisect_left(all_ts_asc, start_ts)
    r = bisect_right(all_ts_asc, end_ts)
    k = len(pos_ids)
    if k <= 0:
        return []

    pos_set = set(pos_ids)
    cand: List[str] = []
    seen: set[str] = set()
    for i in range(l, r):
        row = all_rows_sorted_asc[i]
        # strict rule: same time range, non-user, not in positive ids
        if row.user_id == user_id:
            continue
        if row.tweet_id in pos_set:
            continue
        if row.tweet_id in seen:
            continue
        seen.add(row.tweet_id)
        cand.append(row.tweet_id)

    if len(cand) < k:
        return []
    return rng.sample(cand, k)


def rebuild(args: argparse.Namespace) -> None:
    rng = random.Random(int(args.seed))
    labels_payload = json.loads(Path(args.labels_cache).read_text(encoding="utf-8"))
    csv_path_by_user = labels_payload.get("csv_path_by_user", {})
    if not isinstance(csv_path_by_user, dict):
        raise RuntimeError("labels_cache  csv_path_by_user")

    allowed_user_keys: set[str] = set()
    allowed_raw_users: set[str] = set()
    allowed_pt = str(args.allowed_user_keys_pt).strip()
    if allowed_pt:
        payload = torch.load(allowed_pt, map_location="cpu", weights_only=False)
        emb = payload.get("embeddings", {}) if isinstance(payload, dict) else {}
        if not isinstance(emb, dict):
            raise RuntimeError("allowed_user_keys_pt ： embeddings(dict)")
        allowed_user_keys = set(str(k) for k in emb.keys())
        if not allowed_user_keys:
            raise RuntimeError("allowed_user_keys_pt  user_id_t keys")
        allowed_raw_users = set(k.split("__w", 1)[0] for k in allowed_user_keys)

    users = sorted(str(u) for u in csv_path_by_user.keys())
    if allowed_raw_users:
        users = [u for u in users if u in allowed_raw_users]
    if int(args.max_users) > 0:
        users = users[: int(args.max_users)]

    user_rows_map: Dict[str, List[Tuple[datetime, str]]] = {}
    all_rows: List[TweetRow] = []
    for uid in users:
        csv_path = str(csv_path_by_user.get(uid, "")).strip()
        if not csv_path:
            continue
        rows = _parse_csv_tweets(csv_path, uid)
        if not rows:
            continue
        user_rows_map[uid] = rows
        for dt, tid in rows:
            all_rows.append(TweetRow(ts=dt.timestamp(), tweet_id=str(tid), user_id=uid))

    all_rows.sort(key=lambda x: x.ts)
    all_ts_asc = [r.ts for r in all_rows]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stat = Counter()
    progress_every = max(1, int(args.progress_every_users))
    for i_user, (uid, rows) in enumerate(user_rows_map.items(), start=1):
        if i_user % progress_every == 0:
            print(
                f"[Progress] users={i_user}/{len(user_rows_map)} "
                f"written_windows={stat.get('written_windows', 0)} "
                f"skip_neg_not_enough={stat.get('skip_neg_not_enough', 0)} "
                f"skip_no_t_minus_1_window={stat.get('skip_no_t_minus_1_window', 0)} "
                f"skip_not_in_allowed_user_keys={stat.get('skip_not_in_allowed_user_keys', 0)}",
                flush=True,
            )
        windows = _build_user_windows(rows, int(args.window_size))
        if not windows:
            stat["user_no_window"] += 1
            continue

        user_dir = out_root / uid
        user_dir.mkdir(parents=True, exist_ok=True)

        for wb in windows:
            # strict use of user_id_{t-1}; skip when source window doesn't exist
            if wb.source_window_idx >= len(windows):
                stat["skip_no_t_minus_1_window"] += 1
                continue
            source_key = f"{uid}__w{int(wb.source_window_idx):03d}"
            if allowed_user_keys and source_key not in allowed_user_keys:
                stat["skip_not_in_allowed_user_keys"] += 1
                continue

            neg_ids = _sample_negatives_from_time_range(
                all_rows_sorted_asc=all_rows,
                all_ts_asc=all_ts_asc,
                user_id=uid,
                pos_ids=wb.pos_item_ids,
                start_ts=wb.start_ts,
                end_ts=wb.end_ts,
                rng=rng,
            )
            if len(neg_ids) != len(wb.pos_item_ids):
                stat["skip_neg_not_enough"] += 1
                continue

            payload = {
                "user_id": uid,
                "source_window_idx": int(wb.source_window_idx),
                "target_window_idx": int(wb.target_window_idx),
                "group_t": int(wb.target_window_idx),
                "pos_item_ids": [int(x) if str(x).isdigit() else str(x) for x in wb.pos_item_ids],
                "neg_item_ids": [int(x) if str(x).isdigit() else str(x) for x in neg_ids],
                "time_range": {"start_ts": wb.start_ts, "end_ts": wb.end_ts},
            }
            out_fp = user_dir / f"window_{wb.target_window_idx:03d}.json"
            out_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            stat["written_windows"] += 1
            stat["written_pos_samples"] += len(wb.pos_item_ids)
            stat["written_neg_samples"] += len(neg_ids)

    report = {
        "output_dir": str(out_root),
        "num_users_input": len(users),
        "num_users_with_rows": len(user_rows_map),
        "num_all_rows_global_pool": len(all_rows),
        "stats": dict(stat),
        "config": {
            "window_size": int(args.window_size),
            "seed": int(args.seed),
            "strict_t_minus_1_by_reverse_indexing": True,
            "allowed_user_keys_pt": allowed_pt,
        },
    }
    report_fp = out_root / "_rebuild_report.json"
    report_fp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] rebuilt samples at: {out_root}")
    print(f"[OK] report: {report_fp}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild message_user_samples with strict (user_id_t-1, tweet_id, label)")
    p.add_argument(
        "--labels-cache",
        type=str,
        default="dataset/local/labels_cache.json",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="dataset/local/message_user_samples_v2",
    )
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-users", type=int, default=0, help="debug only: limit users")
    p.add_argument("--progress-every-users", type=int, default=200, help="print progress every N users")
    p.add_argument(
        "--allowed-user-keys-pt",
        type=str,
        default="",
        help="Optional: only keep windows whose source user_id_t exists in this embeddings .pt",
    )
    args = p.parse_args()
    rebuild(args)


if __name__ == "__main__":
    main()

