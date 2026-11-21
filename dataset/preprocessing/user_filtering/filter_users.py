#!/usr/bin/env python3
"""Filter candidate users by follower count and optionally subsample.

Production rule used for DUMoE: keep rows with Followers Count <= 5000.
Optional random subsample (default 2000) with a fixed seed for reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_file(
    source: Path,
    output: Path,
    max_followers: int = 5000,
    sample_size: int = 2000,
    seed: int = 42,
) -> None:
    df = pd.read_csv(source, dtype={"Tweet ID": str, "User ID": str}, low_memory=False)
    if "Followers Count" not in df.columns:
        raise KeyError(f"missing Followers Count in {source}")
    df["Followers Count"] = pd.to_numeric(df["Followers Count"], errors="coerce")
    kept = df[df["Followers Count"] <= int(max_followers)].copy()
    if kept.empty:
        print(f"[skip] no rows after follower filter: {source}")
        return
    if sample_size > 0 and len(kept) > sample_size:
        kept = kept.sample(n=int(sample_size), random_state=int(seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[ok] {source.name}: {len(df)} -> {len(kept)} -> {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Filter users by follower count")
    p.add_argument("--root", type=str, required=True, help="Root directory to walk")
    p.add_argument("--input-name", type=str, default="merged_data.csv")
    p.add_argument("--output-name", type=str, default="filtered_users.csv")
    p.add_argument("--max-followers", type=int, default=5000)
    p.add_argument("--sample-size", type=int, default=2000, help="0 = keep all after filter")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.root)
    for path in root.rglob(args.input_name):
        out = path.with_name(args.output_name)
        try:
            filter_file(path, out, args.max_followers, args.sample_size, args.seed)
        except Exception as exc:
            print(f"[error] {path}: {exc}")


if __name__ == "__main__":
    main()
