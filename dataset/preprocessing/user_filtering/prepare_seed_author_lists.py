#!/usr/bin/env python3
"""Split seed authors into the 15 topic lists used for timeline collection."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

TOPICS = (
    "art",
    "business",
    "culture",
    "education",
    "entertainment",
    "fashion",
    "food",
    "gaming",
    "health",
    "daily_life",
    "news",
    "politics",
    "science",
    "sports",
    "technology",
)

# Aliases seen in older seed CSVs -> canonical topic id
TOPIC_ALIASES = {
    "art_final": "art",
    "game": "gaming",
    "life": "daily_life",
    "fashion_&_beauty": "fashion",
    "health_&_fitness": "health",
}


def normalize_topic(raw: str) -> str:
    key = (raw or "").strip().lower().replace(" ", "_")
    return TOPIC_ALIASES.get(key, key)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-csv", required=True, help="CSV with username,topics columns")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    rows_by_topic = {t: [] for t in TOPICS}
    seen = {t: set() for t in TOPICS}
    invalid = Counter()

    with open(args.source_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            username = (row.get("username") or "").strip()
            if not username:
                continue
            for part in (row.get("topics") or "").split("|"):
                topic = normalize_topic(part)
                if topic not in rows_by_topic:
                    if topic:
                        invalid[topic] += 1
                    continue
                key = username.casefold()
                if key in seen[topic]:
                    continue
                seen[topic].add(key)
                rows_by_topic[topic].append(
                    {
                        "username": username,
                        "source_user_id": (row.get("user_id") or "").strip(),
                        "display_name": (row.get("display_name") or "").strip(),
                        "followers_count": (row.get("followers_count") or "").strip(),
                        "source_topics": (row.get("topics") or "").strip(),
                    }
                )

    out_root = Path(args.output_root)
    for topic, rows in rows_by_topic.items():
        topic_dir = out_root / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        out = topic_dir / "users.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "username",
                    "source_user_id",
                    "display_name",
                    "followers_count",
                    "source_topics",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"{topic}\t{len(rows)}\t{out}")

    if invalid:
        print(f"invalid_topics={dict(invalid)}")


if __name__ == "__main__":
    main()
