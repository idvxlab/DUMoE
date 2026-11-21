#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local loaders for compact tweet shards and user-profile tensors.

Shard format: embeddings [N, D], tweet_ids [N], ...
User profile: features [U, Dp], user_ids [U] (no screen names required).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from tqdm import tqdm


def load_user_features_from_pt(path: str) -> Tuple[Dict[str, torch.Tensor], int]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "features" not in obj:
        raise RuntimeError(f"user profile pt : {path}")
    feats = obj["features"].float()
    dim = int(feats.size(1))
    names = [str(x) for x in obj.get("screen_names", [])]
    if len(names) != int(feats.size(0)):
        raise RuntimeError("screen_names  features ")
    out: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        out[name] = feats[i].contiguous()
    return out, dim


def load_msg_features_from_compact_shards(
    shards_dir: str,
    needed_msg_ids: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], int]:
    root = Path(shards_dir)
    shard_files = sorted(root.glob("w*_tweet_emb_shard_*.pt"))
    if not shard_files:
        # fallback: any *shard*.pt
        shard_files = sorted(root.glob("*shard*.pt"))
    if not shard_files:
        raise RuntimeError(f" tweet emb shards: {shards_dir}")

    need: Optional[Set[str]] = None
    if needed_msg_ids is not None:
        need = {str(x) for x in needed_msg_ids}

    out: Dict[str, torch.Tensor] = {}
    msg_dim: Optional[int] = None
    iterator: Iterable[Path] = shard_files
    if verbose:
        iterator = tqdm(shard_files, desc="Loading time-slice message feature shards")

    for fp in iterator:
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "embeddings" not in obj or "tweet_ids" not in obj:
            if verbose:
                print(f"[Shard] skip unexpected format: {fp}", flush=True)
            continue
        emb = obj["embeddings"]
        tids = obj["tweet_ids"]
        if msg_dim is None:
            msg_dim = int(emb.size(1))
        n = int(emb.size(0))
        tid_cpu = tids.detach().cpu().tolist() if torch.is_tensor(tids) else list(tids)
        for i in range(n):
            tid = str(int(tid_cpu[i]))
            if need is not None and tid not in need:
                continue
            if tid in out:
                continue
            out[tid] = emb[i].float().contiguous()
        if verbose:
            print(f"[Shard] Loaded {fp.name}: kept={len(out)}", flush=True)
        if need is not None and len(out) >= len(need):
            break

    if msg_dim is None:
        raise RuntimeError(" shards  msg_dim")
    if verbose:
        print(f"[Shard] done: feats={len(out)} dim={msg_dim}", flush=True)
    return out, int(msg_dim)


def load_msg_features_matrix_from_compact_shards(
    shards_dir: str,
    needed_msg_ids: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int], int]:
    """： [N,D] + id->row， Tensor  RSS。"""
    root = Path(shards_dir)
    shard_files = sorted(root.glob("w*_tweet_emb_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(root.glob("*shard*.pt"))
    if not shard_files:
        raise RuntimeError(f" tweet emb shards: {shards_dir}")

    need: Optional[Set[str]] = {str(x) for x in needed_msg_ids} if needed_msg_ids is not None else None
    id2row: Dict[str, int] = {}
    chunks: list[torch.Tensor] = []
    msg_dim: Optional[int] = None
    iterator: Iterable[Path] = shard_files
    if verbose:
        iterator = tqdm(shard_files, desc="Loading tweet emb shards (matrix)")

    for fp in iterator:
        if verbose:
            print(f"[Shard] Loading {fp.name} ({fp.stat().st_size / 1e6:.1f} MB)", flush=True)
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "embeddings" not in obj or "tweet_ids" not in obj:
            if verbose:
                print(f"[Shard] skip unexpected format: {fp}", flush=True)
            continue
        emb = obj["embeddings"].float()
        tids = obj["tweet_ids"]
        if msg_dim is None:
            msg_dim = int(emb.size(1))
        tid_cpu = tids.detach().cpu().tolist() if torch.is_tensor(tids) else list(tids)
        sel: list[int] = []
        sel_ids: list[str] = []
        for i, raw in enumerate(tid_cpu):
            tid = str(int(raw))
            if need is not None and tid not in need:
                continue
            if tid in id2row:
                continue
            sel.append(i)
            sel_ids.append(tid)
        if sel:
            chunk = emb[sel].contiguous().clone()
            base = sum(int(c.size(0)) for c in chunks)
            for j, tid in enumerate(sel_ids):
                id2row[tid] = base + j
            chunks.append(chunk)
        del obj, emb
        if verbose:
            print(f"[Shard] Done {fp.name}: kept={len(id2row)}", flush=True)
        if need is not None and len(id2row) >= len(need):
            break

    if msg_dim is None:
        raise RuntimeError(" shards  msg_dim")
    mat = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, int(msg_dim), dtype=torch.float32)
    if verbose:
        print(
            f"[Shard] matrix done: rows={mat.size(0)} dim={msg_dim} "
            f"bytes≈{mat.numel() * mat.element_size() / 1e9:.2f}GB",
            flush=True,
        )
    return mat, id2row, int(msg_dim)
