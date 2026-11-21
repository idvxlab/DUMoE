import csv
import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset


TIME_FMT = "%a %b %d %H:%M:%S +0000 %Y"


def load_time_slice_labels(path: str) -> Dict[str, List[Dict]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("time_slice_lables_500.json  dict")
    return {uid: block.get('months', []) for uid, block in data.items()}


def load_time_slice_labels_from_user_dirs(
    user_dirs: List[str],
    cache_path: str = "",
    use_cache: bool = False,
    progress_every: int = 500,
) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            labels = payload.get("labels_by_user", {})
            csv_map = payload.get("csv_path_by_user", {})
            if isinstance(labels, dict) and isinstance(csv_map, dict):
                print(f"[Data] Loaded label cache from {cache_path}: users={len(labels)}")
                return labels, csv_map
        except Exception as exc:
            print(f"[WARN] Failed loading label cache, fallback to rebuild: {exc}")

    labels_by_user: Dict[str, List[Dict]] = {}
    csv_path_by_user: Dict[str, str] = {}
    started = time.time()
    total_files = 0
    parsed_users = 0

    def _ingest_user_block(uid: str, block: Dict) -> int:
        if not isinstance(block, dict):
            return 0

        windows = block.get("windows", [])
        csv_path = str(block.get("csv_path", "") or "").strip()

        parsed: List[Tuple[int, str]] = []
        for w in windows:
            if not isinstance(w, dict):
                continue
            label = str(w.get("label", "") or "").strip()
            if not label:
                continue
            try:
                group_t = int(w.get("group_t"))
            except Exception:
                continue
            parsed.append((group_t, label))

        if not parsed:
            return 0

        parsed.sort(key=lambda x: x[0])
        months = [
            {
                "month": f"w{group_t:03d}",
                "label": label,
                "group_t": group_t,
            }
            for group_t, label in parsed
        ]

        # Prefer the one with more windows if duplicate appears across sources.
        if uid not in labels_by_user or len(months) > len(labels_by_user[uid]):
            labels_by_user[uid] = months
            if csv_path:
                csv_path_by_user[uid] = csv_path
        return 1

    for source in user_dirs:
        p = Path(source)
        if p.exists() and p.is_dir():
            total_files += sum(1 for _ in p.glob("*.json"))
        elif p.exists() and p.is_file() and p.suffix.lower() == ".json":
            total_files += 1

    seen_files = 0

    for source in user_dirs:
        p = Path(source)
        if not p.exists():
            continue

        if p.is_dir():
            files = sorted(p.glob("*.json"))
            source_mode = "dir"
        elif p.is_file() and p.suffix.lower() == ".json":
            files = [p]
            source_mode = "merged"
        else:
            continue

        for fp in files:
            seen_files += 1
            if progress_every > 0 and seen_files % progress_every == 0:
                elapsed = max(1e-6, time.time() - started)
                rate = seen_files / elapsed
                print(
                    f"[Data] Label parse progress: {seen_files}/{total_files or '?'} "
                    f"({rate:.1f} files/s)"
                )
            try:
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            if source_mode == "dir":
                parsed_users += _ingest_user_block(fp.stem, data)
            else:
                # merged shard file: { user_id: { windows, csv_path, ... }, ... }
                shard_users = 0
                for uid, block in data.items():
                    parsed_users += _ingest_user_block(str(uid), block)
                    shard_users += 1
                print(f"[Data] Parsed merged shard {fp.name}: users={shard_users}")

    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "labels_by_user": labels_by_user,
                        "csv_path_by_user": csv_path_by_user,
                    },
                    f,
                    ensure_ascii=False,
                )
            print(f"[Data] Saved label cache to {cache_path}")
        except Exception as exc:
            print(f"[WARN] Failed saving label cache: {exc}")

    elapsed = time.time() - started
    print(
        f"[Data] Label parse done: scanned={seen_files} parsed_users={parsed_users} "
        f"users={len(labels_by_user)} elapsed={elapsed:.1f}s"
    )

    return labels_by_user, csv_path_by_user


def load_user_features_from_pt(path: str) -> Tuple[Dict[str, torch.Tensor], int]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f" PT : {path}")

    features = payload.get("features")
    user_ids = payload.get("user_ids")
    screen_names = payload.get("screen_names")
    if not isinstance(features, torch.Tensor) or not isinstance(user_ids, list):
        raise ValueError(f" PT  features/user_ids: {path}")
    if features.dim() != 2:
        raise ValueError(f": {tuple(features.shape)}")
    if features.shape[0] != len(user_ids):
        raise ValueError(
            f" user_ids : feats={features.shape[0]} user_ids={len(user_ids)}"
        )

    feat_map: Dict[str, torch.Tensor] = {}
    for i, uid in enumerate(user_ids):
        feat = features[i].float().clone()
        feat_map[str(uid)] = feat
        if isinstance(screen_names, list) and i < len(screen_names):
            raw = screen_names[i]
            if raw is None:
                continue
            sn = str(raw).strip()
            if sn:
                feat_map[sn] = feat
    return feat_map, int(features.shape[1])


def _load_ordered_tweet_ids_from_csv(csv_path: str) -> List[str]:
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

    rows.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in rows]


def build_user_tweet_ids_map(csv_path_by_user: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for uid, csv_path in csv_path_by_user.items():
        tids = _load_ordered_tweet_ids_from_csv(csv_path)
        if tids:
            out[uid] = tids
    return out


def load_user_tweet_ids_map_pt(path: str) -> Dict[str, List[str]]:
    """Load an already-built (optionally anonymized) user -> ordered tweet-id map."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"user_tweet_ids_map : {path}")
    out: Dict[str, List[str]] = {}
    for uid, tids in payload.items():
        seq = [str(x) for x in list(tids) if str(x)]
        if seq:
            out[str(uid)] = seq
    return out


def collect_needed_msg_ids_from_entries(
    entries: List[Tuple[str, int, str, int]],
    user_tweet_ids_map: Dict[str, List[str]],
    window_size: int = 30,
    long_recent: int = 300,
) -> Set[str]:
    needed: Set[str] = set()
    for uid, window_idx, _, _ in entries:
        ordered = user_tweet_ids_map.get(uid, [])
        start = window_idx * window_size
        for tid in ordered[start : start + long_recent]:
            needed.add(str(tid))
    return needed


def load_msg_map(path: str) -> Dict[str, List[Dict]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("filtered_matched_msgs_last_year_monthly.json  dict")
    return {uid: block.get('months', []) for uid, block in data.items()}


def build_label_map(categories: List[str]) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(categories)}


def build_split_entries(
    labels_by_user: Dict[str, List[Dict]],
    label_to_id: Dict[str, int],
    train_months: int = 8,
    val_months: int = 2,
    test_months: int = 2,
) -> Tuple[List[Tuple[str, int, str, int]], List[Tuple[str, int, str, int]], List[Tuple[str, int, str, int]]]:
    train_entries, val_entries, test_entries = [], [], []
    for uid, months in labels_by_user.items():
        for idx, m in enumerate(months):
            label_str = m.get('label')
            if label_str not in label_to_id:
                continue
            entry = (uid, idx, m.get('month'), label_to_id[label_str])
            if idx < train_months:
                train_entries.append(entry)
            elif idx < train_months + val_months:
                val_entries.append(entry)
            elif idx < train_months + val_months + test_months:
                test_entries.append(entry)
    return train_entries, val_entries, test_entries


class TimeSlicePreferenceDataset(Dataset):
    def __init__(
        self,
        user_feature_map: Dict[str, torch.Tensor],
        msg_features_dict: Dict[str, torch.Tensor],
        user_tweet_ids_map: Dict[str, List[str]],
        entries: List[Tuple[str, int, str, int]],
        msg_feat_dim: int = 1536,
        user_feat_dim: int = 1542,
        short_recent: int = 30,
        long_recent: int = 300,
        window_size: int = 30,
    ):
        self.user_feature_map = user_feature_map
        self.msg_features_dict = msg_features_dict
        self.user_tweet_ids_map = user_tweet_ids_map
        self.entries = entries
        self.msg_feat_dim = msg_feat_dim
        self.user_feat_dim = user_feat_dim
        self.short_recent = short_recent
        self.long_recent = long_recent
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        user_id, window_idx, month_str, label = self.entries[idx]
        x_explicit = self._load_user_feature(user_id)

        ordered_tweet_ids = self.user_tweet_ids_map.get(user_id, [])
        start = window_idx * self.window_size

        short_ids = ordered_tweet_ids[start : start + self.short_recent]
        long_ids = ordered_tweet_ids[start : start + self.long_recent]
        long_ids_no_month = ordered_tweet_ids[start + self.short_recent : start + self.long_recent]

        short_feats = self._get_msg_feats(short_ids)
        x_implicit = self._pad_or_truncate_tensor(short_feats, self.short_recent, self.msg_feat_dim)

        short_feats_no_month: List[torch.Tensor] = []
        x_implicit_no_month = self._pad_or_truncate_tensor(short_feats_no_month, self.short_recent, self.msg_feat_dim)

        long_feats = self._get_msg_feats(long_ids)
        if len(long_feats) == 0:
            x_cross_time = torch.zeros((1, self.msg_feat_dim), dtype=torch.float32)
        else:
            x_cross_time = torch.stack(long_feats, dim=0)

        long_feats_no_month = self._get_msg_feats(long_ids_no_month)
        if len(long_feats_no_month) == 0:
            x_cross_time_no_month = torch.zeros((1, self.msg_feat_dim), dtype=torch.float32)
        else:
            x_cross_time_no_month = torch.stack(long_feats_no_month, dim=0)

        return {
            'user_id': user_id,
            'window_idx': int(window_idx),
            'month': month_str,
            'x_explicit': x_explicit,
            'x_implicit': x_implicit,
            'x_cross_time': x_cross_time,
            'x_implicit_no_month': x_implicit_no_month,
            'x_cross_time_no_month': x_cross_time_no_month,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _get_msg_feats(self, msg_ids: Iterable[str]) -> List[torch.Tensor]:
        feats = []
        for mid in msg_ids:
            f = self.msg_features_dict.get(str(mid))
            if f is None:
                try:
                    f = self.msg_features_dict.get(int(mid))
                except Exception:
                    f = None
            if f is not None:
                if not isinstance(f, torch.Tensor):
                    f = torch.tensor(f, dtype=torch.float32)
                else:
                    f = f.float()
                feats.append(f)
        return feats

    def _load_user_feature(self, user_id: str) -> torch.Tensor:
        feats = self.user_feature_map.get(str(user_id))
        if feats is None:
            raise ValueError(f" {user_id}  user_profile_1542.pt ")
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, dtype=torch.float32)
        feats = feats.float().view(-1)
        if feats.numel() < self.user_feat_dim:
            pad = torch.zeros((self.user_feat_dim - feats.numel(),), dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=0)
        elif feats.numel() > self.user_feat_dim:
            feats = feats[: self.user_feat_dim]
        return feats

    @staticmethod
    def _pad_or_truncate_tensor(seq: List[torch.Tensor], pad_to: int, dim: int) -> torch.Tensor:
        if len(seq) > pad_to:
            seq = seq[:pad_to]
        if len(seq) == 0:
            return torch.zeros((pad_to, dim), dtype=torch.float32)
        stacked = torch.stack(seq, dim=0)
        if stacked.shape[0] < pad_to:
            padding = torch.zeros((pad_to - stacked.shape[0], dim), dtype=stacked.dtype)
            stacked = torch.cat([stacked, padding], dim=0)
        return stacked


def _pad_long_sequences(long_seqs: List[torch.Tensor]):
    max_len = max([t.shape[0] for t in long_seqs]) if long_seqs else 0
    dim = long_seqs[0].shape[1] if long_seqs else 0
    padded = []
    pad_mask = []
    for t in long_seqs:
        l = t.shape[0]
        if l < max_len:
            pad = torch.zeros((max_len - l, dim), dtype=t.dtype)
            t_pad = torch.cat([t, pad], dim=0)
            mask = torch.zeros((max_len,), dtype=torch.bool)
            mask[l:] = True
        else:
            t_pad = t
            mask = torch.zeros((max_len,), dtype=torch.bool)
        padded.append(t_pad)
        pad_mask.append(mask)
    x_cross_time = torch.stack(padded, dim=1) if padded else torch.zeros((0, 0, 0))
    src_key_padding_mask = torch.stack(pad_mask, dim=0) if pad_mask else torch.zeros((0, 0), dtype=torch.bool)
    return x_cross_time, src_key_padding_mask


def collate_fn(batch: List[Dict]):
    x_explicit = torch.stack([b['x_explicit'] for b in batch], dim=0)
    x_implicit = torch.stack([b['x_implicit'] for b in batch], dim=0)

    long_seqs = [b['x_cross_time'] for b in batch]
    x_cross_time, src_key_padding_mask = _pad_long_sequences(long_seqs)

    long_no_month_seqs = [b['x_cross_time_no_month'] for b in batch]
    x_cross_time_no_month, src_key_padding_mask_no_month = _pad_long_sequences(long_no_month_seqs)

    x_implicit_no_month = torch.stack([b['x_implicit_no_month'] for b in batch], dim=0)
    labels = torch.stack([b['label'] for b in batch], dim=0)
    user_ids = [b['user_id'] for b in batch]
    window_indices = [int(b['window_idx']) for b in batch]
    months = [b['month'] for b in batch]

    return {
        'user_ids': user_ids,
        'window_indices': window_indices,
        'months': months,
        'x_explicit': x_explicit,
        'x_implicit': x_implicit,
        'x_cross_time': x_cross_time,
        'src_key_padding_mask': src_key_padding_mask,
        'x_implicit_no_month': x_implicit_no_month,
        'x_cross_time_no_month': x_cross_time_no_month,
        'src_key_padding_mask_no_month': src_key_padding_mask_no_month,
        'labels': labels,
    }