#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from windows_io import WindowEntry, write_samples_csv, write_samples_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export window samples (csv/jsonl) from a Task A cache")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="dataset/local/taskA_samples",
        help="Directory written by downstream/taskA/build_sample_cache.py",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/taskA")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f" cache meta: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    shard_paths = sorted(cache_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"cache  shard_*.pt: {cache_dir}")

    entries: List[WindowEntry] = []
    for sp in shard_paths:
        shard: Dict = torch.load(str(sp), map_location="cpu")
        user_ids = shard.get("user_ids", [])
        window_indices = shard.get("window_indices", [])
        months = shard.get("months", [])
        split_tags = shard.get("split_tags", [])
        labels = shard.get("labels")

        if labels is None:
            raise RuntimeError(f"{sp.name}  labels ")
        if isinstance(labels, torch.Tensor):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)

        n = len(labels_list)
        for i in range(n):
            entries.append(
                WindowEntry(
                    user_id=str(user_ids[i]),
                    window_idx=int(window_indices[i]),
                    month=str(months[i]) if i < len(months) else "",
                    label_id=int(labels_list[i]),
                    split_tag=str(split_tags[i]) if i < len(split_tags) else "",
                )
            )

    label_to_id = meta.get("label_to_id", {})

    csv_path = str(output_dir / "taskA_window_samples.csv")
    jsonl_path = str(output_dir / "taskA_window_samples.jsonl")
    meta_path = str(output_dir / "taskA_window_samples_meta.json")

    write_samples_csv(entries, csv_path)
    write_samples_jsonl(entries, jsonl_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(entries),
                "cache_dir": str(cache_dir),
                "window_size": int(meta.get("window_size", 0) or 0),
                "label_to_id": {k: int(v) for k, v in label_to_id.items()},
                "csv": csv_path,
                "jsonl": jsonl_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] samples={len(entries)}")
    print(f"[OK] csv: {csv_path}")
    print(f"[OK] jsonl: {jsonl_path}")
    print(f"[OK] meta: {meta_path}")


if __name__ == "__main__":
    main()

