#!/usr/bin/env python3
"""Sort each user's posts newest-first and split into groups of 30.

This is the window construction used before multimodal labeling and before the
train/val/test split (window indices 0-7 / 8-9 / 10-11).
"""
from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def parse_date(date_str: str) -> datetime:
    text = (date_str or "").strip()
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%a %b %d %H:%M:%S +0000 %Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.min


def read_user_tweets(csv_path: Path, seen_ids: Set[str]) -> List[Dict]:
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    header_index = None
    for i, line in enumerate(lines):
        if ("Tweet Date" in line and "Tweet URL" in line) or (
            "publish_time" in line and ("tweet_id" in line or "Tweet ID" in line)
        ):
            header_index = i
            break
    if header_index is None:
        print(f"[skip] no header in {csv_path}")
        return []

    reader = csv.DictReader(io.StringIO("".join(lines[header_index:])))
    tweets: List[Dict] = []
    for row in reader:
        tweet_url = (row.get("Tweet URL") or row.get("tweet_url") or "").strip()
        content = (row.get("Tweet Content") or row.get("full_text") or "").strip()
        date = (row.get("Tweet Date") or row.get("publish_time") or "").strip()
        if not tweet_url and not content and not date:
            continue
        unique_id = tweet_url or f"{content[:100]}|{date}"
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)
        tweets.append(row)
    return tweets


def sort_newest_first(tweets: List[Dict]) -> List[Dict]:
    def key(row: Dict):
        return parse_date(str(row.get("Tweet Date") or row.get("publish_time") or ""))

    return sorted(tweets, key=key, reverse=True)


def group_tweets(tweets: List[Dict], group_size: int = 30) -> List[List[Dict]]:
    return [tweets[i : i + group_size] for i in range(0, len(tweets), group_size)]


def save_group(group: List[Dict], output_path: Path) -> None:
    if not group:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(group[0].keys()))
        writer.writeheader()
        writer.writerows(group)


def username_from_filename(name: str) -> str:
    if name.endswith("-tweets.csv"):
        return name[: -len("-tweets.csv")]
    if name.endswith("_full_tweets.csv"):
        return name[: -len("_full_tweets.csv")]
    if name.endswith(".csv"):
        return name[:-4]
    return name


def main() -> None:
    p = argparse.ArgumentParser(description="Group user posts into windows of 30")
    p.add_argument("--input-root", required=True, help="Root with user_tweet_* folders or CSVs")
    p.add_argument("--output-root", required=True)
    p.add_argument("--group-size", type=int, default=30)
    args = p.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    seen: Set[str] = set()

    csv_files = sorted(input_root.rglob("*.csv"))
    by_user: Dict[str, List[Path]] = {}
    for path in csv_files:
        if path.name in {"group_tweets_by_date.py"}:
            continue
        user = username_from_filename(path.name)
        by_user.setdefault(user, []).append(path)

    print(f"users={len(by_user)} csv_files={len(csv_files)}")
    for user, paths in sorted(by_user.items()):
        rows: List[Dict] = []
        for path in paths:
            rows.extend(read_user_tweets(path, seen))
        ordered = sort_newest_first(rows)
        groups = group_tweets(ordered, args.group_size)
        user_dir = output_root / user
        for i, group in enumerate(groups):
            save_group(group, user_dir / f"group_{i:03d}_{len(group)}tweets.csv")
        print(f"{user}: tweets={len(ordered)} groups={len(groups)}")


if __name__ == "__main__":
    main()
