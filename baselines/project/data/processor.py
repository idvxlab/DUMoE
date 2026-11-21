from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DUMOE_DIR = REPO_ROOT / "dumoe"
for _p in (str(REPO_ROOT), str(DUMOE_DIR), str(REPO_ROOT / "baselines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from downstream.windows_io import WindowEntry, read_samples_csv  # type: ignore

DYNAMIC_DIR = DUMOE_DIR
if DYNAMIC_DIR.exists() and str(DYNAMIC_DIR) not in sys.path:
    sys.path.insert(0, str(DYNAMIC_DIR))

try:
    from time_slice_data_loader import (  # type: ignore
        build_user_tweet_ids_map,
        load_user_features_from_pt,
    )
    from time_slice_msg_feature_loader import load_msg_features_from_compact_shards  # type: ignore
except ImportError:
    from project.data.local_feature_loaders import (  # type: ignore
        load_msg_features_from_compact_shards,
        load_user_features_from_pt,
    )

    def build_user_tweet_ids_map(csv_path_by_user, verbose: bool = True):  # type: ignore
        raise RuntimeError(
            " time_slice_data_loader.build_user_tweet_ids_map；"
            " .， cache/user_tweet_ids_map__*.pt"
        )


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


@dataclass
class SampleRecord:
    key: str
    user_id: str
    window_idx: int
    label_id: int
    split_tag: str


def _entry_to_record(e: WindowEntry) -> SampleRecord:
    return SampleRecord(
        key=e.user_id_t,
        user_id=e.user_id,
        window_idx=int(e.window_idx),
        label_id=int(e.label_id),
        split_tag=str(e.split_tag),
    )


def build_records_by_split(
    samples_csv: str,
    max_train: int = 0,
    max_val: int = 0,
    max_test: int = 0,
) -> Dict[str, List[SampleRecord]]:
    entries = read_samples_csv(samples_csv)
    buckets: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    for e in entries:
        tag = str(e.split_tag).strip().lower()
        if tag not in buckets:
            continue
        buckets[tag].append(_entry_to_record(e))

    def _cap(xs: List[SampleRecord], cap: int) -> List[SampleRecord]:
        if cap and cap > 0:
            return xs[:cap]
        return xs

    return {
        "train": _cap(buckets["train"], max_train),
        "val": _cap(buckets["val"], max_val),
        "test": _cap(buckets["test"], max_test),
    }


def _csv_map_from_labels_cache(labels_cache: str) -> Dict[str, str]:
    payload = json.loads(Path(labels_cache).read_text(encoding="utf-8"))
    mapping = payload.get("csv_path_by_user", {})
    if not isinstance(mapping, dict):
        raise RuntimeError("labels_cache  csv_path_by_user")
    out: Dict[str, str] = {}
    for uid, path in mapping.items():
        p = str(path or "").strip()
        if p:
            out[str(uid)] = p
    return out


def load_or_build_user_tweet_ids_map(
    labels_cache: str,
    users: Set[str],
    cache_dir: str,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key_src = "|".join(sorted(users))
    h = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_root / f"user_tweet_ids_map__{h}.pt"

    if cache_path.exists():
        if verbose:
            print(f"[Data] load cached user_tweet_ids_map: {cache_path}", flush=True)
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            return obj

    # Public anonymized release: a single map next to the labels cache.
    sibling = Path(labels_cache).resolve().parent / "user_tweet_ids_map.pt"
    if sibling.exists():
        if verbose:
            print(f"[Data] load anonymized user_tweet_ids_map: {sibling}", flush=True)
        obj = torch.load(str(sibling), map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            torch.save(obj, cache_path)
            return obj

    csv_map = _csv_map_from_labels_cache(labels_cache)
    filtered = {u: csv_map[u] for u in users if u in csv_map}
    if verbose:
        print(
            f"[Data] build_user_tweet_ids_map users={len(filtered)}/{len(users)} "
            f"(from labels_cache)",
            flush=True,
        )
    mp = build_user_tweet_ids_map(filtered)
    torch.save(mp, cache_path)
    return mp


def build_id2idx_from_train(
    train_records: Sequence[SampleRecord],
    user_tweet_ids_map: Dict[str, List[str]],
    msg_feats: Dict[str, torch.Tensor],
    seq_len: int,
    window_size: int,
) -> Tuple[Dict[str, int], List[str]]:
    ids_set: Set[str] = set()
    for r in train_records:
        ordered = user_tweet_ids_map.get(r.user_id, [])
        st = int(r.window_idx) * int(window_size)
        for mid in ordered[st : st + int(seq_len)]:
            s = str(mid)
            if s in msg_feats:
                ids_set.add(s)
    sorted_ids = sorted(ids_set)
    id2idx = {s: i + 1 for i, s in enumerate(sorted_ids)}
    return id2idx, sorted_ids


def build_item_embedding_table_fixed(
    sorted_msg_ids: Sequence[str],
    msg_feats: Dict[str, torch.Tensor],
    msg_dim: int,
) -> nn.Embedding:
    n = len(sorted_msg_ids)
    mat = torch.zeros((n + 1, int(msg_dim)), dtype=torch.float32)
    for i, s in enumerate(sorted_msg_ids):
        v = msg_feats[str(s)].float().view(-1)
        mat[i + 1] = v[: int(msg_dim)]
    return nn.Embedding.from_pretrained(mat, freeze=True, padding_idx=0)


class UnifiedWindowDataset(Dataset):
    def __init__(
        self,
        records: List[SampleRecord],
        user_tweet_ids_map: Dict[str, List[str]],
        user_feat_map: Dict[str, torch.Tensor],
        profile_dim: int,
        seq_len: int,
        window_size: int,
        id2idx: Dict[str, int],
        verbose: bool = False,
    ) -> None:
        self.records = list(records)
        self.user_tweet_ids_map = user_tweet_ids_map
        self.user_feat_map = user_feat_map
        self.profile_dim = int(profile_dim)
        self.seq_len = int(seq_len)
        self.window_size = int(window_size)
        self.id2idx = id2idx
        self.verbose = bool(verbose)

        self._indices: List[int] = []
        for i, r in enumerate(self.records):
            if self._valid_len_for_record(r) >= 2:
                self._indices.append(i)
        if self.verbose:
            print(
                f"[Dataset] split keeps {len(self._indices)}/{len(self.records)} "
                f"(>=2 valid items with embeddings)",
                flush=True,
            )

    def _valid_len_for_record(self, r: SampleRecord) -> int:
        ordered = self.user_tweet_ids_map.get(r.user_id, [])
        st = int(r.window_idx) * int(self.window_size)
        n = 0
        for mid in ordered[st : st + self.seq_len]:
            if str(mid) in self.id2idx:
                n += 1
        return n

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[self._indices[idx]]
        ordered = self.user_tweet_ids_map.get(r.user_id, [])
        st = int(r.window_idx) * int(self.window_size)
        mids = ordered[st : st + self.seq_len]

        seq_ids = torch.zeros(self.seq_len, dtype=torch.long)
        seq_mask = torch.ones(self.seq_len, dtype=torch.bool)
        j = 0
        for mid in mids:
            if j >= self.seq_len:
                break
            s = str(mid)
            ix = self.id2idx.get(s)
            if ix is None:
                continue
            seq_ids[j] = ix
            seq_mask[j] = False
            j += 1

        prof = self.user_feat_map.get(r.user_id)
        if prof is None:
            profile = torch.zeros(self.profile_dim, dtype=torch.float32)
        else:
            profile = prof.float().view(-1)[: self.profile_dim]
            if profile.numel() < self.profile_dim:
                pad = torch.zeros(self.profile_dim - profile.numel(), dtype=torch.float32)
                profile = torch.cat([profile, pad], dim=0)

        return {
            "key": r.key,
            "seq_ids": seq_ids,
            "seq_mask": seq_mask,
            "profile": profile,
            "label": torch.tensor(int(r.label_id), dtype=torch.long),
        }


__all__ = [
    "SampleRecord",
    "UnifiedWindowDataset",
    "build_records_by_split",
    "build_id2idx_from_train",
    "build_item_embedding_table_fixed",
    "load_or_build_user_tweet_ids_map",
    "load_msg_features_from_compact_shards",
    "load_user_features_from_pt",
    "masked_mean",
]
