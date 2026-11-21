#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMOE_DIR = REPO_ROOT / "dumoe"
for _p in (str(REPO_ROOT), str(DUMOE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from time_slice_msg_feature_loader import load_msg_features_from_compact_shards


def _safe_torch_save(obj, path: Path) -> None:
    # Use legacy serialization to avoid broken zip archives on this environment.
    torch.save(obj, str(path), _use_new_zipfile_serialization=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunked msg feature table for TaskA cache")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="dataset/local/taskA_samples",
    )
    parser.add_argument(
        "--needed-msg-ids-cache",
        type=str,
        default="./TASKA/taskA_needed_msg_ids_all_windows.pt",
    )
    parser.add_argument(
        "--tweet-emb-shards-dir",
        type=str,
        default="dataset/local/embeddings_pt",
    )
    parser.add_argument("--rows-per-chunk", type=int, default=100000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    needed_raw = torch.load(args.needed_msg_ids_cache, map_location="cpu", weights_only=False)
    if not isinstance(needed_raw, (list, tuple, set)):
        raise RuntimeError("needed msg ids cache format invalid")
    needed_msg_ids: Set[str] = set(str(x) for x in needed_raw)

    print(f"[TaskA-TableChunks] loading msg features by needed ids: {len(needed_msg_ids)}")
    msg_features_dict, msg_feat_dim = load_msg_features_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir,
        needed_msg_ids=needed_msg_ids,
        verbose=True,
    )
    print(f"[TaskA-TableChunks] loaded msg features: {len(msg_features_dict)} dim={msg_feat_dim}")

    table = torch.zeros((len(msg_features_dict) + 1, msg_feat_dim), dtype=torch.float16)

    i = 1
    for _, feat in msg_features_dict.items():
        f = feat if isinstance(feat, torch.Tensor) else torch.tensor(feat)
        table[i] = f.to(dtype=torch.float16)
        i += 1

    chunks_dir = cache_dir / "msg_feature_table_fp16_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    rows_per_chunk = int(args.rows_per_chunk)
    part_id = 0
    total_rows = int(table.size(0))

    print(
        f"[TaskA-TableChunks] start chunking: total_rows={total_rows} "
        f"rows_per_chunk={rows_per_chunk}"
    )

    for start in range(0, total_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, total_rows)
        part = table[start:end].contiguous().clone()
        part_path = chunks_dir / f"part_{part_id:05d}.pt"

        _safe_torch_save(part, part_path)
        print(f"[TaskA-TableChunks] wrote {part_path.name} rows={end-start}")

        part_id += 1

    meta["msg_feature_table_chunks_dir"] = str(chunks_dir)
    meta["msg_feature_table_chunk_rows"] = rows_per_chunk
    meta["msg_feature_table_chunk_parts"] = part_id
    meta["msg_feat_dim"] = int(msg_feat_dim)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[TaskA-TableChunks] updated meta: {meta_path}")
    print(f"[TaskA-TableChunks] done parts={part_id} total_rows={total_rows}")


if __name__ == "__main__":
    main()
