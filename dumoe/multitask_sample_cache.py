#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class RecTensorBank:
    def __init__(
        self,
        key_to_row: Dict,
        pos_idx_table_cpu: torch.Tensor,
        neg_idx_table_cpu: torch.Tensor,
        pos_len_cpu: torch.Tensor,
        neg_len_cpu: torch.Tensor,
    ) -> None:
        self.key_to_row = key_to_row
        self.pos_idx_table_cpu = pos_idx_table_cpu
        self.neg_idx_table_cpu = neg_idx_table_cpu
        self.pos_len_cpu = pos_len_cpu
        self.neg_len_cpu = neg_len_cpu
        self.all_item_ids: set[str] = set()
        self.by_source_key: Dict = {}

    @classmethod
    def load(cls, path: Path | str) -> "RecTensorBank":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(
            key_to_row=payload.get("key_to_row", {}),
            pos_idx_table_cpu=payload.get("pos_idx_table_cpu", torch.zeros((0, 1), dtype=torch.long)),
            neg_idx_table_cpu=payload.get("neg_idx_table_cpu", torch.zeros((0, 1), dtype=torch.long)),
            pos_len_cpu=payload.get("pos_len_cpu", torch.zeros((0,), dtype=torch.long)),
            neg_len_cpu=payload.get("neg_len_cpu", torch.zeros((0,), dtype=torch.long)),
        )


class CachedTimeSliceDataset(Dataset):
    def __init__(self, payload_path: Path | str, msg_feature_table: torch.Tensor):
        payload = torch.load(str(payload_path), map_location="cpu", weights_only=False)
        self.payload_path = str(payload_path)
        self.msg_feature_table = msg_feature_table.float()

        self.x_explicit = payload["x_explicit"]
        self.short_msg_indices = payload["short_msg_indices"]
        self.long_msg_indices = payload["long_msg_indices"]
        self.long_no_month_msg_indices = payload["long_no_month_msg_indices"]
        self.src_key_padding_mask = payload["src_key_padding_mask"]
        self.src_key_padding_mask_no_month = payload["src_key_padding_mask_no_month"]
        self.labels = payload["labels"]
        self.user_ids = list(payload["user_ids"])
        self.window_indices = list(payload["window_indices"])
        self.months = list(payload["months"])
        self.split_tags = list(payload.get("split_tags", []))
        self.meta = dict(payload.get("meta", {}))

        self.msg_feat_dim = int(self.meta.get("msg_feat_dim", self.msg_feature_table.size(1)))
        self.short_recent = int(self.meta.get("short_recent", self.short_msg_indices.size(1)))
        self.zero_short = torch.zeros((self.short_recent, self.msg_feat_dim), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def _gather_fixed(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        return self.msg_feature_table[idx_tensor.long()].float()

    def _gather_variable(self, idx_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        feats = self.msg_feature_table[idx_tensor.long()].float()
        valid = ~mask_tensor.bool()
        feats = feats[valid]
        if feats.numel() == 0:
            return torch.zeros((1, self.msg_feat_dim), dtype=torch.float32)
        return feats

    def __getitem__(self, idx: int) -> Dict:
        x_explicit = self.x_explicit[idx].float()
        short_idx = self.short_msg_indices[idx]
        long_idx = self.long_msg_indices[idx]
        long_no_idx = self.long_no_month_msg_indices[idx]
        long_mask = self.src_key_padding_mask[idx]
        long_no_mask = self.src_key_padding_mask_no_month[idx]

        return {
            "user_id": self.user_ids[idx],
            "window_idx": int(self.window_indices[idx]),
            "month": self.months[idx],
            "x_explicit": x_explicit,
            "x_implicit": self._gather_fixed(short_idx),
            "x_cross_time": self._gather_variable(long_idx, long_mask),
            "x_implicit_no_month": self.zero_short,
            "x_cross_time_no_month": self._gather_variable(long_no_idx, long_no_mask),
            "label": self.labels[idx].long(),
        }
