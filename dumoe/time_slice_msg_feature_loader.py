import os
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from tqdm import tqdm


def _normalize_id(v) -> str:
    # Shards can store ids as tensor scalars; convert them to plain string IDs.
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    return str(v)


def list_shard_paths(shards_dir: str) -> List[str]:
    if not os.path.isdir(shards_dir):
        raise FileNotFoundError(f" shards : {shards_dir}")
    shards = [
        os.path.join(shards_dir, fn)
        for fn in os.listdir(shards_dir)
        if fn.endswith('.pt') and 'tweet_emb_shard' in fn
    ]
    if not shards:
        raise FileNotFoundError(f" .pt shards: {shards_dir}")
    return sorted(shards)


def collect_needed_msg_ids(msg_map: Dict[str, List[Dict]], user_ids: Iterable[str]) -> Set[str]:
    needed: Set[str] = set()
    for uid in user_ids:
        months = msg_map.get(uid, [])
        for m in months:
            for mid in m.get('messages', []) or []:
                needed.add(str(mid))
    return needed


def load_msg_features_from_compact_shards(
    shards_dir: str,
    needed_msg_ids: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], int]:
    shard_paths = list_shard_paths(shards_dir)
    msg_features: Dict[str, torch.Tensor] = {}
    msg_feat_dim = None

    remaining = None
    if needed_msg_ids is not None:
        remaining = set(str(mid) for mid in needed_msg_ids)

    iterator = shard_paths
    if verbose:
        need_msg = "all tweets in shards" if remaining is None else f"up to {len(remaining)} tweet ids"
        print(f"[Data] Loading embedding tensors from shards ({need_msg})...", flush=True)
        iterator = tqdm(shard_paths, desc="Loading time-slice message feature shards")

    for shard_idx, shard_path in enumerate(iterator, start=1):
        shard_start = time.time()
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        if verbose:
            print(f"[Shard] Loading {shard_idx}/{len(shard_paths)} {os.path.basename(shard_path)} ({shard_size_mb:.1f} MB)")

        shard = torch.load(shard_path, map_location='cpu')

        if isinstance(shard, dict) and 'tweet_ids' in shard and ('features' in shard or 'embeddings' in shard):
            msg_ids = shard['tweet_ids']
            features = shard.get('features', shard.get('embeddings'))
        elif isinstance(shard, dict):
            msg_ids = None
            features = None
        else:
            raise ValueError(f"Shard : {shard_path}")

        if msg_feat_dim is None:
            if features is not None:
                msg_feat_dim = features.shape[1]
            else:
                first_val = next(iter(shard.values()))
                msg_feat_dim = first_val.shape[0]

        if features is not None:
            if remaining is None:
                for idx, mid in enumerate(msg_ids):
                    msg_features[_normalize_id(mid)] = features[idx].clone()
            else:
                for idx, mid in enumerate(msg_ids):
                    mid_str = _normalize_id(mid)
                    if mid_str in remaining:
                        msg_features[mid_str] = features[idx].clone()
                        remaining.remove(mid_str)
                if not remaining:
                    break
        else:
            if remaining is None:
                for mid, feat in shard.items():
                    msg_features[str(mid)] = feat.clone()
            else:
                for mid, feat in shard.items():
                    mid_str = str(mid)
                    if mid_str in remaining:
                        msg_features[mid_str] = feat.clone()
                        remaining.remove(mid_str)
                if not remaining:
                    break

        if verbose:
            elapsed = time.time() - shard_start
            remaining_count = len(remaining) if remaining is not None else None
            if remaining_count is not None:
                print(f"[Shard] Done {shard_idx}/{len(shard_paths)} in {elapsed:.1f}s, remaining={remaining_count}")
            else:
                print(f"[Shard] Done {shard_idx}/{len(shard_paths)} in {elapsed:.1f}s, loaded={len(msg_features)}")

        del shard
        del features
        del msg_ids

    if msg_feat_dim is None:
        raise RuntimeError(" shards 。")

    if remaining:
        missing = list(remaining)[:5]
        print(f"⚠️  {len(remaining)} ，: {missing}")

    return msg_features, msg_feat_dim


def collect_user_ids_from_compact_shards(shards_dir: str, verbose: bool = True) -> Set[str]:
    shard_paths = list_shard_paths(shards_dir)
    users: Set[str] = set()

    if verbose:
        print(
            f"[Data] Scanning {len(shard_paths)} shard files for user_ids only (metadata; tweet vectors not loaded)...",
            flush=True,
        )

    iterator = shard_paths
    if verbose:
        iterator = tqdm(shard_paths, desc="Collecting users from tweet embedding shards")

    for shard_path in iterator:
        shard = torch.load(shard_path, map_location='cpu')
        if isinstance(shard, dict) and 'user_ids' in shard:
            uids = shard.get('user_ids')
            user_vocab = shard.get('user_vocab')

            # In current embeddings_pt shards, user_ids are int indices into user_vocab.
            if isinstance(user_vocab, list) and hasattr(uids, '__iter__'):
                if isinstance(uids, torch.Tensor):
                    uniq = torch.unique(uids).tolist()
                    for idx in uniq:
                        idx = int(idx)
                        if 0 <= idx < len(user_vocab):
                            name = str(user_vocab[idx]).strip()
                            if name:
                                users.add(name)
                else:
                    for uid in uids:
                        key = _normalize_id(uid)
                        try:
                            idx = int(float(key))
                        except Exception:
                            idx = -1
                        if 0 <= idx < len(user_vocab):
                            name = str(user_vocab[idx]).strip()
                            if name:
                                users.add(name)
            elif hasattr(uids, '__iter__'):
                for uid in uids:
                    users.add(_normalize_id(uid))
        del shard

    return users