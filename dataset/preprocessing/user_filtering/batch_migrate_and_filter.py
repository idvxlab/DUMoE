#!/usr/bin/env python3
"""Migrate per-topic user CSVs and keep posts inside the collection date range.

Default range matches the filtering stage used before labeling:
2025-09-01 inclusive through 2026-03-01 exclusive of later months.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

CATEGORIES = [
    "Business",
    "Culture",
    "Technology",
    "Fashion_&_Beauty",
    "Entertainment",
    "Sports",
    "Health_&_Fitness",
    "Food",
    "News",
    "Education",
    "Gaming",
    "Politics",
    "Science",
    "Art",
    "Daily_Life",
]


def parse_tweet_date(date_str: str):
    text = (date_str or "").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%a %b %d %H:%M:%S +0000 %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def migrate_category(old_path: Path, new_path: Path) -> tuple[int, int]:
    new_path.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for csv_file in old_path.glob("*-tweets.csv"):
        dst = new_path / csv_file.name
        if dst.exists():
            ok += 1
            continue
        try:
            shutil.copy2(csv_file, dst)
            ok += 1
        except Exception as exc:
            print(f"[error] copy {csv_file} -> {dst}: {exc}")
            fail += 1
    return ok, fail


def filter_date_range(csv_path: Path, start: datetime, end: datetime) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else None
    if not rows or not fieldnames:
        return 0
    date_key = None
    for cand in ("Tweet Date", "publish_time", "created_at"):
        if cand in fieldnames:
            date_key = cand
            break
    if date_key is None:
        return 0
    kept = []
    for row in rows:
        dt = parse_tweet_date(str(row.get(date_key, "")))
        if dt is None:
            continue
        if start <= dt < end:
            kept.append(row)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    return len(kept)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old-root", required=True)
    p.add_argument("--new-root", required=True)
    p.add_argument("--start", default="2025-09-01")
    p.add_argument("--end", default="2026-03-01")
    args = p.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    old_root = Path(args.old_root)
    new_root = Path(args.new_root)

    for cat in CATEGORIES:
        old = old_root / f"user_tweet_{cat}"
        new = new_root / f"user_tweet_{cat}"
        if not old.is_dir():
            print(f"[skip] missing {old}")
            continue
        ok, fail = migrate_category(old, new)
        print(f"[migrate] {cat}: ok={ok} fail={fail}")
        total = 0
        for csv_file in new.glob("*-tweets.csv"):
            total += filter_date_range(csv_file, start, end)
        print(f"[filter] {cat}: kept_rows={total}")


if __name__ == "__main__":
    main()
