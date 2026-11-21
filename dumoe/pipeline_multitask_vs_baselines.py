#!/usr/bin/env python3
import argparse
import gc
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_multitask import PreferenceModelMultiTask
from time_slice_data_loader import (
    TimeSlicePreferenceDataset,
    build_label_map,
    build_split_entries,
    build_user_tweet_ids_map,
    collate_fn,
    collect_needed_msg_ids_from_entries,
    load_time_slice_labels_from_user_dirs,
    load_user_features_from_pt,
    load_user_tweet_ids_map_pt,
)
from time_slice_msg_feature_loader import (
    collect_user_ids_from_compact_shards,
    load_msg_features_from_compact_shards,
)
from multitask_sample_cache import CachedTimeSliceDataset, RecTensorBank


TARGET_CATEGORIES = [
    "Entertainment", "Sports", "Technology", "Business", "Education", "Gaming", "Politics", "Art",
    "Culture", "Science", "Food", "Daily Life", "Fashion & Beauty", "Health & Fitness", "News",
]


@dataclass
class ExpConfig:
    name: str
    seed: int = 42
    ablation: str = "none"
    training_strategy: str = "staged"
    start_stage: int = 1
    resume_ckpt: str = ""
    stage1_epochs: int = 15
    stage2_epochs: int = 5
    stage3_epochs: int = 5
    joint_epochs: int = 25
    batch_size: int = 128
    export_batch_size: int = 256
    lr_stage1: float = 1e-4
    lr_stage2: float = 5e-4
    lr_stage3: float = 1e-3
    lr_joint: float = 3e-4
    weight_decay: float = 1e-2
    rec_weight: float = 1.5
    cls_weight: float = 1.0
    rec_neg_reduction: str = "mean"
    rec_num_neg: int = 10
    rec_on_stage1: bool = True
    rec_on_stage2: bool = True
    rec_on_stage3: bool = True
    short_recent: int = 30
    long_recent: int = 300
    window_size: int = 30
    hidden_dim: int = 256
    dropout: float = 0.4
    topk_adapters: int = 3
    num_workers: int = 4
    amp: bool = True
    device: str = "cuda:0"
    export_mode: str = "thresholded"
    export_gate_mode: str = "soft"
    export_kl_threshold: float = 0.012
    export_kl_soft_scale: float = 0.01
    export_gate_min_alpha: float = 0.05
    export_gate_max_alpha: float = 0.80
    export_amp: bool = True


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros((num_classes,), dtype=torch.float32)
    for y in labels:
        counts[int(y)] += 1.0
    total = counts.sum().clamp_min(1.0)
    w = total / counts.clamp_min(1.0)
    w = w / w.mean().clamp_min(1e-12)
    return w.to(device)


def build_msg_feature_table(msg_features_dict: Dict[str, torch.Tensor], msg_feat_dim: int):
    """
     msg_features_dict ， GPU gather。
    row 0 （/ id）。
    """
    id_to_row: Dict[str, int] = {}
    table = torch.zeros((len(msg_features_dict) + 1, int(msg_feat_dim)), dtype=torch.float32)
    row = 1
    for mid, feat in msg_features_dict.items():
        id_to_row[str(mid)] = row
        t = feat if isinstance(feat, torch.Tensor) else torch.tensor(feat)
        table[row] = t.float().view(-1)[: int(msg_feat_dim)]
        row += 1
    return table, id_to_row


def _iter_rec_sample_objs(samples_dir: str):
    """Yield rec-window dicts from a jsonl file or */window_*.json tree."""
    root = Path(samples_dir)
    jsonl_path = None
    if root.is_file() and root.suffix.lower() == ".jsonl":
        jsonl_path = root
    elif root.is_dir():
        cand = root / "message_user_samples.jsonl"
        if cand.is_file():
            jsonl_path = cand
    if jsonl_path is not None:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return
    if root.is_dir():
        for fp in sorted(root.glob("*/window_*.json")):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


class RecSampleBank:
    def __init__(self, samples_dir: str):
        self.by_source_key: Dict[Tuple[str, int], Tuple[List[str], List[str]]] = {}
        self.all_item_ids: set[str] = set()
        self.key_to_row: Dict[Tuple[str, int], int] = {}
        self.pos_idx_table_cpu: Optional[torch.Tensor] = None
        self.neg_idx_table_cpu: Optional[torch.Tensor] = None
        self.pos_len_cpu: Optional[torch.Tensor] = None
        self.neg_len_cpu: Optional[torch.Tensor] = None
        for obj in _iter_rec_sample_objs(samples_dir):
            uid = str(obj.get("user_id", "")).strip()
            target_widx = int(obj.get("target_window_idx", obj.get("group_t", -1)))
            if not uid or target_widx < 0:
                continue
            pos = [str(x) for x in obj.get("pos_item_ids", []) if str(x)]
            neg = [str(x) for x in obj.get("neg_item_ids", []) if str(x)]
            if not pos or not neg:
                continue
            source_widx = int(target_widx) + 1
            self.by_source_key[(uid, source_widx)] = (pos, neg)
            self.all_item_ids.update(pos)
            self.all_item_ids.update(neg)

    def build_tensor_cache(self, msg_id_to_row: Dict[str, int]) -> None:
        """
         (user_id, source_window_idx) ->  item （CPU Tensor）。
         key->row  + GPU gather， item  Python 。
        """
        rows: List[Tuple[Tuple[str, int], List[int], List[int]]] = []
        max_pos = 0
        max_neg = 0
        for key, (pos_ids, neg_ids) in self.by_source_key.items():
            pos_idx = [int(msg_id_to_row.get(str(mid), 0)) for mid in pos_ids]
            neg_idx = [int(msg_id_to_row.get(str(mid), 0)) for mid in neg_ids]
            pos_idx = [x for x in pos_idx if x > 0]
            neg_idx = [x for x in neg_idx if x > 0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            max_pos = max(max_pos, len(pos_idx))
            max_neg = max(max_neg, len(neg_idx))
            rows.append((key, pos_idx, neg_idx))

        if not rows:
            self.key_to_row = {}
            self.pos_idx_table_cpu = torch.zeros((0, 1), dtype=torch.long)
            self.neg_idx_table_cpu = torch.zeros((0, 1), dtype=torch.long)
            self.pos_len_cpu = torch.zeros((0,), dtype=torch.long)
            self.neg_len_cpu = torch.zeros((0,), dtype=torch.long)
            return

        n = len(rows)
        pos_tbl = torch.zeros((n, max_pos), dtype=torch.long)
        neg_tbl = torch.zeros((n, max_neg), dtype=torch.long)
        pos_len = torch.zeros((n,), dtype=torch.long)
        neg_len = torch.zeros((n,), dtype=torch.long)
        key_to_row: Dict[Tuple[str, int], int] = {}

        for i, (key, p_idx, n_idx) in enumerate(rows):
            key_to_row[key] = i
            pos_len[i] = len(p_idx)
            neg_len[i] = len(n_idx)
            pos_tbl[i, : len(p_idx)] = torch.tensor(p_idx, dtype=torch.long)
            neg_tbl[i, : len(n_idx)] = torch.tensor(n_idx, dtype=torch.long)

        self.key_to_row = key_to_row
        self.pos_idx_table_cpu = pos_tbl
        self.neg_idx_table_cpu = neg_tbl
        self.pos_len_cpu = pos_len
        self.neg_len_cpu = neg_len


def _entry_has_context(entry: Tuple[str, int, str, int], user_tweet_ids_map: Dict[str, List[str]], window_size: int) -> bool:
    uid, window_idx, _, _ = entry
    return int(window_idx) * int(window_size) < len(user_tweet_ids_map.get(uid, []))


def _load_intersection_users(
    intersection_cache: Path,
    users_labels: set,
    users_userfeat: set,
    users_csv: set,
) -> Tuple[set, bool]:
    if not intersection_cache.exists():
        return set(), False
    try:
        payload = json.loads(intersection_cache.read_text(encoding="utf-8"))
        cached_users = set(str(x) for x in payload.get("intersection_user_ids", []))
        intersection = cached_users & users_labels & users_userfeat & users_csv
        return intersection, True
    except Exception:
        return set(), False


def _safe_full_mask_with_anchor(src_key_padding_mask: torch.Tensor) -> torch.Tensor:
    safe_mask = torch.ones_like(src_key_padding_mask, dtype=torch.bool)
    if safe_mask.dim() == 2 and safe_mask.size(1) > 0:
        safe_mask[:, 0] = False
    return safe_mask


def apply_input_ablation(
    ablation: str,
    x_explicit: torch.Tensor,
    x_implicit: torch.Tensor,
    x_cross_time: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
):
    if ablation == "only_user":
        x_implicit = torch.zeros_like(x_implicit)
        x_cross_time = torch.zeros_like(x_cross_time)
        src_key_padding_mask = _safe_full_mask_with_anchor(src_key_padding_mask)
    elif ablation == "only_short":
        x_explicit = torch.zeros_like(x_explicit)
        x_cross_time = torch.zeros_like(x_cross_time)
        src_key_padding_mask = _safe_full_mask_with_anchor(src_key_padding_mask)
    elif ablation == "only_long":
        x_explicit = torch.zeros_like(x_explicit)
        x_implicit = torch.zeros_like(x_implicit)

    if src_key_padding_mask.dim() == 2 and src_key_padding_mask.size(1) > 0:
        all_masked = src_key_padding_mask.all(dim=1)
        if all_masked.any():
            src_key_padding_mask = src_key_padding_mask.clone()
            src_key_padding_mask[all_masked, 0] = False

    return x_explicit, x_implicit, x_cross_time, src_key_padding_mask


def build_rec_batch_tensors(
    batch_user_ids: Sequence[str],
    batch_window_indices: Sequence[int],
    rec_bank: RecSampleBank,
    msg_feature_table: torch.Tensor,
    rec_num_neg: int,
    device: torch.device,
    rng: random.Random,  # ， Python 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = len(batch_user_ids)
    feat_dim = int(msg_feature_table.size(1))
    pos = torch.zeros((bsz, feat_dim), dtype=torch.float32, device=device)
    neg = torch.zeros((bsz, int(rec_num_neg), feat_dim), dtype=torch.float32, device=device)
    has = torch.zeros((bsz,), dtype=torch.bool, device=device)

    if (
        rec_bank.pos_idx_table_cpu is None
        or rec_bank.neg_idx_table_cpu is None
        or rec_bank.pos_len_cpu is None
        or rec_bank.neg_len_cpu is None
    ):
        return pos, neg, has

    row_ids = torch.full((bsz,), -1, dtype=torch.long)
    for i, (uid, widx) in enumerate(zip(batch_user_ids, batch_window_indices)):
        rid = rec_bank.key_to_row.get((str(uid), int(widx)), -1)
        row_ids[i] = int(rid)

    valid = row_ids >= 0
    if not valid.any():
        return pos, neg, has

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
    row_valid = row_ids[valid].to(dtype=torch.long)

    valid_idx = valid_idx.to(device, non_blocking=True)
    row_valid = row_valid.to(device, non_blocking=True)

    pos_tbl = rec_bank.pos_idx_table_cpu.to(device, non_blocking=True)
    neg_tbl = rec_bank.neg_idx_table_cpu.to(device, non_blocking=True)
    pos_len = rec_bank.pos_len_cpu.to(device, non_blocking=True)[row_valid]
    neg_len = rec_bank.neg_len_cpu.to(device, non_blocking=True)[row_valid]

    has_valid = (pos_len > 0) & (neg_len > 0)
    if not has_valid.any():
        return pos, neg, has

    valid_idx = valid_idx[has_valid]
    row_valid = row_valid[has_valid]
    pos_len = pos_len[has_valid]
    neg_len = neg_len[has_valid]

    pos_rand = torch.floor(torch.rand((row_valid.size(0),), device=device) * pos_len.float()).long()
    pos_item_row = pos_tbl[row_valid].gather(1, pos_rand.unsqueeze(1)).squeeze(1)  # [Bv]

    k = int(rec_num_neg)
    neg_rand = torch.floor(torch.rand((row_valid.size(0), k), device=device) * neg_len.unsqueeze(1).float()).long()
    neg_item_row = neg_tbl[row_valid].gather(1, neg_rand)  # [Bv, K]

    pos_feat = msg_feature_table[pos_item_row]             # [Bv, D]
    neg_feat = msg_feature_table[neg_item_row]             # [Bv, K, D]

    pos[valid_idx] = pos_feat
    neg[valid_idx] = neg_feat
    has[valid_idx] = True
    return pos, neg, has


@torch.no_grad()
def evaluate_cls(model, loader, criterion, device, ablation: str = "none") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for batch in loader:
        x_explicit = batch["x_explicit"].to(device)
        x_implicit = batch["x_implicit"].to(device)
        x_cross_time = batch["x_cross_time"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        y = batch["labels"].to(device)
        x_explicit, x_implicit, x_cross_time, mask = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit,
            x_cross_time,
            mask,
        )
        logits = model(x_explicit, x_implicit, x_cross_time, mask)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += int(y.size(0))
    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
    }


@torch.no_grad()
def evaluate_cls_and_rec(
    model,
    loader,
    criterion,
    device,
    rec_bank: RecSampleBank,
    msg_feature_table: torch.Tensor,
    rec_num_neg: int,
    seed: int = 42,
    rec_neg_reduction: str = "mean",
    ablation: str = "none",
) -> Dict[str, float]:
    model.eval()
    rng = random.Random(int(seed))

    total_cls_loss = 0.0
    total_cls_correct = 0
    total_cls_n = 0

    total_rec_loss = 0.0
    total_rec_n = 0
    total_pair_acc = 0.0
    total_pair_n = 0

    for batch in loader:
        x_explicit = batch["x_explicit"].to(device)
        x_implicit = batch["x_implicit"].to(device)
        x_cross_time = batch["x_cross_time"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        labels = batch["labels"].to(device)
        user_ids = list(batch["user_ids"])
        window_indices = list(batch["window_indices"])

        x_explicit, x_implicit, x_cross_time, mask = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit,
            x_cross_time,
            mask,
        )

        logits, user_repr = model(
            x_explicit, x_implicit, x_cross_time, mask,
            return_user_repr=True,
        )
        cls_loss = criterion(logits, labels)
        total_cls_loss += float(cls_loss.item()) * int(labels.size(0))
        total_cls_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_cls_n += int(labels.size(0))

        pos_item_emb, neg_item_emb, has_rec = build_rec_batch_tensors(
            batch_user_ids=user_ids,
            batch_window_indices=window_indices,
            rec_bank=rec_bank,
            msg_feature_table=msg_feature_table,
            rec_num_neg=int(rec_num_neg),
            device=device,
            rng=rng,
        )
        if has_rec.any():
            u = user_repr[has_rec]
            pos_proj, neg_proj = model.project_items(pos_item_emb[has_rec], neg_item_emb[has_rec])
            rec_loss = model.compute_neg_sampling_rec_loss(
                user_repr=u,
                pos_item_repr=pos_proj,
                neg_item_repr=neg_proj,
                neg_reduction=rec_neg_reduction,
            )
            n_rec = int(has_rec.sum().item())
            total_rec_loss += float(rec_loss.item()) * n_rec
            total_rec_n += n_rec

            pos_score = (u * pos_proj).sum(dim=-1)  # [B_rec]
            neg_score = torch.einsum("bd,bkd->bk", u, neg_proj).mean(dim=-1)  # [B_rec]
            pair_acc = (pos_score > neg_score).float().mean()
            total_pair_acc += float(pair_acc.item()) * n_rec
            total_pair_n += n_rec

    return {
        "cls_loss": total_cls_loss / max(1, total_cls_n),
        "cls_acc": total_cls_correct / max(1, total_cls_n),
        "rec_loss": total_rec_loss / max(1, total_rec_n),
        "rec_pair_acc": total_pair_acc / max(1, total_pair_n),
        "rec_covered_samples": float(total_rec_n),
        "cls_samples": float(total_cls_n),
    }


def _rec_module_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """item_proj + next_item_head（）；Stage2  adapter ， requires_grad 。"""
    params: List[torch.nn.Parameter] = []
    if hasattr(model, "item_proj"):
        params.extend(list(model.item_proj.parameters()))
    if hasattr(model, "next_item_head"):
        params.extend(list(model.next_item_head.parameters()))
    return params


def train_one_experiment(
    cfg: ExpConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rec_bank: RecSampleBank,
    msg_feature_table: torch.Tensor,
    msg_feat_dim: int,
    user_feat_dim: int,
    num_classes: int,
    out_dir: Path,
) -> Path:
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    model = PreferenceModelMultiTask(
        user_feat_dim=user_feat_dim,
        msg_feat_dim=msg_feat_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=num_classes,
        lstm_layers=1,
        transformer_layers=2,
        nhead=8,
        dropout=cfg.dropout,
        adapter_bottleneck=64,
        use_adapter=False,
        topk_adapters=cfg.topk_adapters,
    ).to(device)

    train_labels: List[int] = []
    ds = getattr(train_loader, "dataset", None)
    if ds is not None and hasattr(ds, "entries"):
        train_labels = [int(e[3]) for e in ds.entries]
    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    def run_one_epoch(
        optimizer: torch.optim.Optimizer,
        adapter_index: Optional[int],
        filter_labels: Optional[List[int]],
        enable_rec: bool,
        ablation: str,
    ) -> Dict[str, float]:
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_rec = 0.0
        total_n = 0
        for batch in train_loader:
            x_explicit = batch["x_explicit"].to(device)
            x_implicit = batch["x_implicit"].to(device)
            x_cross_time = batch["x_cross_time"].to(device)
            mask = batch["src_key_padding_mask"].to(device)
            labels = batch["labels"].to(device)
            user_ids = list(batch["user_ids"])
            window_indices = list(batch["window_indices"])

            if filter_labels is not None:
                m = torch.zeros_like(labels, dtype=torch.bool)
                for lb in filter_labels:
                    m = m | (labels == int(lb))
                if not m.any():
                    continue
                x_explicit = x_explicit[m]
                x_implicit = x_implicit[m]
                x_cross_time = x_cross_time[:, m, :]
                mask = mask[m]
                labels = labels[m]
                m_list = m.detach().cpu().tolist()
                user_ids = [u for u, keep in zip(user_ids, m_list) if keep]
                window_indices = [w for w, keep in zip(window_indices, m_list) if keep]

            x_explicit, x_implicit, x_cross_time, mask = apply_input_ablation(
                ablation,
                x_explicit,
                x_implicit,
                x_cross_time,
                mask,
            )

            with torch.amp.autocast("cuda", enabled=(cfg.amp and device.type == "cuda")):
                logits, user_repr = model(
                    x_explicit, x_implicit, x_cross_time, mask,
                    adapter_index=adapter_index,
                    return_user_repr=True,
                )
                cls_loss = criterion(logits, labels)

                rec_loss = torch.tensor(0.0, device=device)
                if enable_rec:
                    pos_item_emb, neg_item_emb, has_rec = build_rec_batch_tensors(
                        batch_user_ids=user_ids,
                        batch_window_indices=window_indices,
                        rec_bank=rec_bank,
                        msg_feature_table=msg_feature_table,
                        rec_num_neg=int(cfg.rec_num_neg),
                        device=device,
                        rng=rng,
                    )
                    if has_rec.any():
                        pos_proj, neg_proj = model.project_items(pos_item_emb[has_rec], neg_item_emb[has_rec])
                        rec_loss = model.compute_neg_sampling_rec_loss(
                            user_repr=user_repr[has_rec],
                            pos_item_repr=pos_proj,
                            neg_item_repr=neg_proj,
                            neg_reduction=str(cfg.rec_neg_reduction),
                        )
                loss = cfg.cls_weight * cls_loss + cfg.rec_weight * rec_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = int(labels.size(0))
            total_n += n
            total_loss += float(loss.item()) * n
            total_cls += float(cls_loss.item()) * n
            total_rec += float(rec_loss.item()) * n
        return {
            "loss": total_loss / max(1, total_n),
            "cls": total_cls / max(1, total_n),
            "rec": total_rec / max(1, total_n),
            "n": float(total_n),
        }

    ablation = str(cfg.ablation).lower().strip()
    training_strategy = str(cfg.training_strategy).lower().strip()
    start_stage = int(getattr(cfg, "start_stage", 1))
    resume_ckpt = str(getattr(cfg, "resume_ckpt", "")).strip()
    if ablation not in {"none", "no_adapter", "no_gating", "only_user", "only_short", "only_long"}:
        raise ValueError(f"Unsupported ablation={ablation}")
    if training_strategy not in {"staged", "joint"}:
        raise ValueError(f"Unsupported training_strategy={training_strategy}")
    if start_stage not in {1, 2, 3}:
        raise ValueError(f"Unsupported start_stage={start_stage}")
    if training_strategy == "joint" and start_stage != 1:
        raise ValueError("start_stage only supports staged training")

    model.use_uniform_adapter_fusion = (ablation == "no_gating")

    best_val_acc = -1.0
    best_path = out_dir / "best_model.pth"

    if start_stage > 1:
        if not resume_ckpt:
            raise ValueError("resume_ckpt is required when start_stage > 1")
        ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        print(
            f"[{cfg.name}] resume from ckpt={resume_ckpt} start_stage={start_stage}",
            flush=True,
        )

    if training_strategy == "joint":
        for p in model.parameters():
            p.requires_grad = True

        if ablation == "no_adapter":
            model.use_adapter = False
            model.use_uniform_adapter_fusion = False
            for adapter in model.adapter_pool:
                for p in adapter.parameters():
                    p.requires_grad = False
            for p in model.gating_network.parameters():
                p.requires_grad = False
        elif ablation == "no_gating":
            model.use_adapter = True
            model.use_uniform_adapter_fusion = True
            for p in model.gating_network.parameters():
                p.requires_grad = False
        else:
            model.use_adapter = True
            model.use_uniform_adapter_fusion = False

        opt_joint = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr_joint,
            weight_decay=cfg.weight_decay,
        )
        enable_rec_joint = bool(cfg.rec_on_stage1 or cfg.rec_on_stage2 or cfg.rec_on_stage3)
        for ep in range(1, int(cfg.joint_epochs) + 1):
            tr = run_one_epoch(
                opt_joint,
                adapter_index=None,
                filter_labels=None,
                enable_rec=enable_rec_joint,
                ablation=ablation,
            )
            val = evaluate_cls(model, val_loader, criterion, device, ablation=ablation)
            print(
                f"[{cfg.name}][Joint] ep={ep} train_loss={tr['loss']:.4f} cls={tr['cls']:.4f} rec={tr['rec']:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}",
                flush=True,
            )
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(cfg),
                        "best_val_acc": best_val_acc,
                        "msg_feat_dim": msg_feat_dim,
                        "user_feat_dim": user_feat_dim,
                        "num_classes": num_classes,
                        "best_stage": "joint",
                        "best_epoch": ep,
                    },
                    str(best_path),
                )
        return best_path

    run_stage2 = True
    run_stage3 = True
    if ablation == "no_adapter":
        run_stage2 = False
        run_stage3 = False
    elif ablation == "no_gating":
        run_stage3 = False

    # Stage1: backbone
    if start_stage <= 1:
        model.set_training_stage(1)
        opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
        for ep in range(1, int(cfg.stage1_epochs) + 1):
            tr = run_one_epoch(
                opt1,
                adapter_index=None,
                filter_labels=None,
                enable_rec=bool(cfg.rec_on_stage1),
                ablation=str(cfg.ablation),
            )
            val = evaluate_cls(model, val_loader, criterion, device, ablation=ablation)
            print(f"[{cfg.name}][S1] ep={ep} train_loss={tr['loss']:.4f} cls={tr['cls']:.4f} rec={tr['rec']:.4f} val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}", flush=True)
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "best_val_acc": best_val_acc, "msg_feat_dim": msg_feat_dim, "user_feat_dim": user_feat_dim, "num_classes": num_classes, "best_stage": "stage1", "best_epoch": ep}, str(best_path))

    # Stage2: adapter pool, train each adapter with its class samples
    if run_stage2 and start_stage <= 2:
        model.set_training_stage(2)
        rec_side_params = _rec_module_params(model)
        for cls_idx in range(int(num_classes)):
            for p in model.adapter_pool[cls_idx].parameters():
                p.requires_grad = True
            opt2 = torch.optim.AdamW(
                list(model.adapter_pool[cls_idx].parameters()) + rec_side_params,
                lr=cfg.lr_stage2,
                weight_decay=cfg.weight_decay,
            )
            for ep in range(1, int(cfg.stage2_epochs) + 1):
                tr = run_one_epoch(
                    opt2,
                    adapter_index=cls_idx,
                    filter_labels=[cls_idx],
                    enable_rec=bool(cfg.rec_on_stage2),
                    ablation=str(cfg.ablation),
                )
                print(
                    f"[{cfg.name}][S2][adapter={cls_idx}] ep={ep} train_loss={tr['loss']:.4f} cls={tr['cls']:.4f} "
                    f"rec={tr['rec']:.4f} samples={int(tr['n'])}",
                    flush=True,
                )
            for p in model.adapter_pool[cls_idx].parameters():
                p.requires_grad = False
        val2 = evaluate_cls(model, val_loader, criterion, device, ablation=ablation)
        print(f"[{cfg.name}][S2] val_loss={val2['loss']:.4f} val_acc={val2['acc']:.4f}", flush=True)
        if val2["acc"] > best_val_acc:
            best_val_acc = val2["acc"]
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "best_val_acc": best_val_acc, "msg_feat_dim": msg_feat_dim, "user_feat_dim": user_feat_dim, "num_classes": num_classes, "best_stage": "stage2", "best_epoch": int(cfg.stage2_epochs)}, str(best_path))

    # Stage3: gating network
    if run_stage3 and start_stage <= 3:
        model.set_training_stage(3)
        opt3 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage3, weight_decay=cfg.weight_decay)
        for ep in range(1, int(cfg.stage3_epochs) + 1):
            tr = run_one_epoch(
                opt3,
                adapter_index=None,
                filter_labels=None,
                enable_rec=bool(cfg.rec_on_stage3),
                ablation=str(cfg.ablation),
            )
            val = evaluate_cls(model, val_loader, criterion, device, ablation=ablation)
            print(
                f"[{cfg.name}][S3] ep={ep} train_loss={tr['loss']:.4f} cls={tr['cls']:.4f} rec={tr['rec']:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}",
                flush=True,
            )
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "best_val_acc": best_val_acc, "msg_feat_dim": msg_feat_dim, "user_feat_dim": user_feat_dim, "num_classes": num_classes, "best_stage": "stage3", "best_epoch": ep}, str(best_path))
    return best_path


@torch.no_grad()
def export_embeddings(
    model_ckpt_path: Path,
    export_loader: DataLoader,
    msg_feat_dim: int,
    user_feat_dim: int,
    num_classes: int,
    device_str: str,
    out_pt_path: Path,
    split_tags: Sequence[str],
    ablation: str,
    mode: str,
    gate_mode: str,
    kl_threshold: float,
    kl_soft_scale: float,
    gate_min_alpha: float,
    gate_max_alpha: float,
    amp: bool,
) -> None:
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")
    ckpt = torch.load(str(model_ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = PreferenceModelMultiTask(
        user_feat_dim=user_feat_dim,
        msg_feat_dim=msg_feat_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_classes=num_classes,
        lstm_layers=1,
        transformer_layers=2,
        nhead=8,
        dropout=float(cfg.get("dropout", 0.4)),
        adapter_bottleneck=64,
        use_adapter=False,
        topk_adapters=int(cfg.get("topk_adapters", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    ablation = str(ablation).lower().strip()
    if ablation == "no_adapter":
        model.use_adapter = False
        model.use_uniform_adapter_fusion = False
    elif ablation == "no_gating":
        model.use_adapter = True
        model.use_uniform_adapter_fusion = True
    else:
        model.use_adapter = True
        model.use_uniform_adapter_fusion = False

    model.eval()

    use_amp = bool(amp and device.type == "cuda")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    embeddings_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    user_ids_list: List[str] = []
    window_indices_list: List[int] = []
    months_list: List[str] = []
    split_tags_list: List[str] = []

    offset = 0
    for batch in export_loader:
        x_explicit = batch["x_explicit"].to(device)
        x_implicit = batch["x_implicit"].to(device)
        x_cross_time = batch["x_cross_time"].to(device)
        mask = batch["src_key_padding_mask"].to(device)

        x_implicit_no_month = batch["x_implicit_no_month"].to(device)
        x_cross_time_no_month = batch["x_cross_time_no_month"].to(device)
        mask_no_month = batch["src_key_padding_mask_no_month"].to(device)

        x_explicit, x_implicit, x_cross_time, mask = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit,
            x_cross_time,
            mask,
        )
        _, x_implicit_no_month, x_cross_time_no_month, mask_no_month = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit_no_month,
            x_cross_time_no_month,
            mask_no_month,
        )

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                repr_batch = model.encode_user_repr(
                    x_explicit=x_explicit,
                    x_implicit=x_implicit,
                    x_cross_time=x_cross_time,
                    src_key_padding_mask=mask,
                    x_implicit_no_month=x_implicit_no_month,
                    x_cross_time_no_month=x_cross_time_no_month,
                    src_key_padding_mask_no_month=mask_no_month,
                    mode=str(mode),
                    kl_threshold=float(kl_threshold),
                    gate_mode=str(gate_mode),
                    kl_soft_scale=float(kl_soft_scale),
                    gate_min_alpha=float(gate_min_alpha),
                    gate_max_alpha=float(gate_max_alpha),
                    return_gate_info=False,
                )
        else:
            repr_batch = model.encode_user_repr(
                x_explicit=x_explicit,
                x_implicit=x_implicit,
                x_cross_time=x_cross_time,
                src_key_padding_mask=mask,
                x_implicit_no_month=x_implicit_no_month,
                x_cross_time_no_month=x_cross_time_no_month,
                src_key_padding_mask_no_month=mask_no_month,
                mode=str(mode),
                kl_threshold=float(kl_threshold),
                gate_mode=str(gate_mode),
                kl_soft_scale=float(kl_soft_scale),
                gate_min_alpha=float(gate_min_alpha),
                gate_max_alpha=float(gate_max_alpha),
                return_gate_info=False,
            )

        embeddings_list.append(repr_batch.detach().cpu().float())
        labels_list.append(batch["labels"].detach().cpu())
        user_ids_list.extend(list(batch["user_ids"]))
        window_indices_list.extend([int(x) for x in batch["window_indices"]])
        months_list.extend(list(batch["months"]))

        bsz = int(repr_batch.size(0))
        if split_tags:
            split_tags_list.extend(split_tags[offset: offset + bsz])
        offset += bsz

    embeddings = torch.cat(embeddings_list, dim=0) if embeddings_list else torch.zeros((0, 0))
    labels = torch.cat(labels_list, dim=0) if labels_list else torch.zeros((0,), dtype=torch.long)

    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": embeddings,
        "labels": labels,
        "user_ids": user_ids_list,
        "window_indices": window_indices_list,
        "months": months_list,
        "split_tags": split_tags_list,
        "meta": {
            "checkpoint": str(model_ckpt_path),
            "ablation": ablation,
            "mode": str(mode),
            "gate_mode": str(gate_mode),
            "kl_threshold": float(kl_threshold),
            "num_samples": int(embeddings.size(0)),
            "dim": int(embeddings.size(1)) if embeddings.numel() else 0,
            "source": "PreferenceModelMultiTask",
        },
    }
    torch.save(payload, str(out_pt_path))


@torch.no_grad()
def load_model_from_ckpt(
    model_ckpt_path: Path,
    msg_feat_dim: int,
    user_feat_dim: int,
    num_classes: int,
    device_str: str,
    ablation: str = "none",
):
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")
    ckpt = torch.load(str(model_ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = PreferenceModelMultiTask(
        user_feat_dim=user_feat_dim,
        msg_feat_dim=msg_feat_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_classes=num_classes,
        lstm_layers=1,
        transformer_layers=2,
        nhead=8,
        dropout=float(cfg.get("dropout", 0.4)),
        adapter_bottleneck=64,
        use_adapter=False,
        topk_adapters=int(cfg.get("topk_adapters", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    ablation = str(ablation).lower().strip()
    if ablation == "no_adapter":
        model.use_adapter = False
        model.use_uniform_adapter_fusion = False
    elif ablation == "no_gating":
        model.use_adapter = True
        model.use_uniform_adapter_fusion = True
    else:
        model.use_adapter = True
        model.use_uniform_adapter_fusion = False
    model.eval()
    return model, device


def build_experiment_configs(args) -> List[ExpConfig]:
    if args.sweep_json and Path(args.sweep_json).exists():
        rows = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))
        out: List[ExpConfig] = []
        for i, row in enumerate(rows):
            base = ExpConfig(name=f"exp_{i:02d}")
            for k, v in dict(row).items():
                if hasattr(base, k):
                    setattr(base, k, v)
            if not base.name:
                base.name = f"exp_{i:02d}"
            out.append(base)
        return out
    return [
        ExpConfig(
            name=args.exp_name,
            seed=args.seed,
            ablation=args.ablation,
            training_strategy=args.training_strategy,
            start_stage=args.start_stage,
            resume_ckpt=args.resume_ckpt,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            joint_epochs=args.joint_epochs,
            batch_size=args.batch_size,
            export_batch_size=args.export_batch_size,
            lr_stage1=args.lr_stage1,
            lr_stage2=args.lr_stage2,
            lr_stage3=args.lr_stage3,
            lr_joint=args.lr_joint,
            rec_weight=args.rec_weight,
            cls_weight=float(args.cls_weight),
            rec_neg_reduction=str(args.rec_neg_reduction),
            rec_num_neg=args.rec_num_neg,
            rec_on_stage1=bool(args.rec_on_stage1),
            rec_on_stage2=bool(args.rec_on_stage2),
            rec_on_stage3=bool(args.rec_on_stage3),
            export_mode=args.export_mode,
            export_gate_mode=args.export_gate_mode,
            export_kl_threshold=args.export_kl_threshold,
            export_kl_soft_scale=args.export_kl_soft_scale,
            export_gate_min_alpha=args.export_gate_min_alpha,
            export_gate_max_alpha=args.export_gate_max_alpha,
            export_amp=bool(args.export_amp),
            amp=bool(args.amp),
            device=args.device,
        )
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Train multitask representation and report own val/test metrics.")
    p.add_argument("--label-user-dirs", type=str, default="")
    p.add_argument("--samples-dir", type=str, default="dataset/local/message_user_samples")
    p.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    p.add_argument(
        "--user-tweet-ids-map",
        type=str,
        default="",
        help="Optional .pt map {user_id: [tweet_id,...]}. Use this for the anonymized public dataset (no raw tweet CSVs).",
    )
    p.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    p.add_argument("--sample-cache-dir", type=str, default="")
    p.add_argument("--output-root", type=str, default="outputs/dumoe")
    p.add_argument("--exp-name", type=str, default="mtask_rec")
    p.add_argument("--sweep-json", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation", type=str, default="none", choices=["none", "no_adapter", "no_gating", "only_user", "only_short", "only_long"])
    p.add_argument("--training-strategy", type=str, default="staged", choices=["staged", "joint"])
    p.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument("--stage1-epochs", type=int, default=15)
    p.add_argument("--stage2-epochs", type=int, default=5)
    p.add_argument("--stage3-epochs", type=int, default=5)
    p.add_argument("--joint-epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--export-batch-size", type=int, default=256)
    p.add_argument("--lr-stage1", type=float, default=1e-4)
    p.add_argument("--lr-stage2", type=float, default=5e-4)
    p.add_argument("--lr-stage3", type=float, default=1e-3)
    p.add_argument("--lr-joint", type=float, default=3e-4)
    p.add_argument("--rec-weight", type=float, default=1.5)
    p.add_argument("--cls-weight", type=float, default=1.0, help=" loss （ rec_weight ）")
    p.add_argument(
        "--rec-neg-reduction",
        type=str,
        default="mean",
        choices=["sum", "mean"],
        help=" loss  K  log ：mean=（ cls ）；sum=（ K ）",
    )
    p.add_argument("--rec-num-neg", type=int, default=10)
    p.add_argument("--rec-on-stage1", type=int, default=1, choices=[0, 1])
    p.add_argument("--rec-on-stage2", type=int, default=1, choices=[0, 1])
    p.add_argument("--rec-on-stage3", type=int, default=1, choices=[0, 1])
    p.add_argument(
        "--train-subsample-ratio",
        type=float,
        default=1.0,
        help="train/val/test ； 1.0 ，",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--export-amp", type=int, default=1, choices=[0, 1])

    p.add_argument("--export-mode", type=str, default="thresholded", choices=["base", "adapter", "thresholded"])
    p.add_argument("--export-gate-mode", type=str, default="soft", choices=["soft", "hard"])
    p.add_argument("--export-kl-threshold", type=float, default=0.012)
    p.add_argument("--export-kl-soft-scale", type=float, default=0.01)
    p.add_argument("--export-gate-min-alpha", type=float, default=0.05)
    p.add_argument("--export-gate-max-alpha", type=float, default=0.80)

    p.add_argument("--use-cached-labels", action="store_true")
    p.add_argument("--labels-cache", type=str, default="")
    p.add_argument("--label-parse-progress-every", type=int, default=0)
    p.add_argument("--use-cached-intersection", action="store_true")
    p.add_argument("--intersection-cache", type=str, default="")
    p.add_argument("--use-cached-needed-msg-ids", action="store_true")
    p.add_argument("--needed-msg-ids-cache", type=str, default="")
    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = "."
    intersection_cache = Path(args.intersection_cache or os.path.join(data_root, "time_slice_user_intersection_stats.json"))
    needed_msg_ids_cache = Path(args.needed_msg_ids_cache or os.path.join(data_root, "time_slice_needed_msg_ids.pt"))
    labels_cache_path = args.labels_cache or os.path.join(data_root, "time_slice_labels_cache.json")
    cfgs = build_experiment_configs(args)
    cfg_window_size = max((int(c.window_size) for c in cfgs), default=30)
    cfg_long_recent = max((int(c.long_recent) for c in cfgs), default=300)

    label_to_id = build_label_map(TARGET_CATEGORIES)
    sample_cache_dir = Path(args.sample_cache_dir).expanduser() if str(args.sample_cache_dir).strip() else None
    using_sample_cache = bool(sample_cache_dir and sample_cache_dir.exists())

    train_entries = val_entries = test_entries = None
    if using_sample_cache:
        meta = json.loads((sample_cache_dir / "meta.json").read_text(encoding="utf-8"))
        user_feat_dim = int(meta["user_feat_dim"])
        msg_feat_dim = int(meta["msg_feat_dim"])
        msg_feature_table_cpu = torch.load(str(sample_cache_dir / "msg_feature_table_fp16.pt"), map_location="cpu", weights_only=False).float()
        rec_bank = RecTensorBank.load(sample_cache_dir / "rec_bank_cache.pt")
        print(
            f"[Cache] loaded sample cache: dir={sample_cache_dir} "
            f"train={meta['counts'].get('train', 0)} val={meta['counts'].get('val', 0)} test={meta['counts'].get('test', 0)}",
            flush=True,
        )
        print(
            f"[RecCache] keys={len(rec_bank.key_to_row)} "
            f"pos_tbl={tuple(rec_bank.pos_idx_table_cpu.shape) if rec_bank.pos_idx_table_cpu is not None else None} "
            f"neg_tbl={tuple(rec_bank.neg_idx_table_cpu.shape) if rec_bank.neg_idx_table_cpu is not None else None}",
            flush=True,
        )
    else:
        label_user_dirs = [x.strip() for x in args.label_user_dirs.split(",") if x.strip()]
        labels_by_user, csv_path_by_user = load_time_slice_labels_from_user_dirs(
            user_dirs=label_user_dirs,
            cache_path=labels_cache_path,
            use_cache=bool(args.use_cached_labels),
            progress_every=max(0, int(args.label_parse_progress_every)),
        )
        user_feature_map, user_feat_dim = load_user_features_from_pt(args.user_profile_pt)

        users_labels = set(labels_by_user.keys())
        users_userfeat = set(user_feature_map.keys())
        users_csv = set(csv_path_by_user.keys())
        prebuilt_map = None
        map_path = str(args.user_tweet_ids_map or "").strip()
        if map_path:
            prebuilt_map = load_user_tweet_ids_map_pt(map_path)
            users_csv = set(prebuilt_map.keys())

        users_intersection = None
        if args.use_cached_intersection:
            cached_users, ok = _load_intersection_users(
                intersection_cache,
                users_labels,
                users_userfeat,
                users_csv,
            )
            if ok:
                users_intersection = cached_users

        if users_intersection is None:
            if prebuilt_map is not None:
                users_intersection = users_labels & users_userfeat & users_csv
            else:
                users_in_shards = collect_user_ids_from_compact_shards(args.tweet_emb_shards_dir, verbose=True)
                users_intersection = users_labels & users_userfeat & users_in_shards & users_csv
            try:
                intersection_cache.write_text(
                    json.dumps(
                        {
                            "labels_users": len(users_labels),
                            "user_feat_users": len(users_userfeat),
                            "csv_users": len(users_csv),
                            "intersection_users": len(users_intersection),
                            "intersection_user_ids": sorted(users_intersection),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        labels_by_user = {u: labels_by_user[u] for u in users_intersection}
        csv_path_by_user = {u: csv_path_by_user[u] for u in users_intersection if u in csv_path_by_user}
        user_feature_map = {u: user_feature_map[u] for u in users_intersection}

        if prebuilt_map is not None:
            user_tweet_ids_map = {u: prebuilt_map[u] for u in users_intersection if u in prebuilt_map}
        else:
            user_tweet_ids_map = build_user_tweet_ids_map(csv_path_by_user)
        users_with_tweets = {u for u, tids in user_tweet_ids_map.items() if len(tids) > 0}
        labels_by_user = {u: labels_by_user[u] for u in users_with_tweets}
        user_feature_map = {u: user_feature_map[u] for u in users_with_tweets}
        user_tweet_ids_map = {u: user_tweet_ids_map[u] for u in users_with_tweets}
        train_entries, val_entries, test_entries = build_split_entries(labels_by_user, label_to_id)

        train_entries = [e for e in train_entries if _entry_has_context(e, user_tweet_ids_map, cfg_window_size)]
        val_entries = [e for e in val_entries if _entry_has_context(e, user_tweet_ids_map, cfg_window_size)]
        test_entries = [e for e in test_entries if _entry_has_context(e, user_tweet_ids_map, cfg_window_size)]

        ratio = float(args.train_subsample_ratio)
        if ratio <= 0 or ratio > 1.0:
            raise ValueError("--train-subsample-ratio must be within (0, 1]")
        if ratio < 1.0:
            rnd = random.Random(int(args.seed))

            def _subsample(entries: List[Tuple[str, int, str, int]], name: str):
                if len(entries) <= 1:
                    print(f"[Data] {name} subsample skip: size={len(entries)}", flush=True)
                    return entries
                keep_n = max(1, int(len(entries) * ratio))
                idxs = list(range(len(entries)))
                rnd.shuffle(idxs)
                keep = set(idxs[:keep_n])
                out = [e for i, e in enumerate(entries) if i in keep]
                print(f"[Data] {name} subsample enabled: ratio={ratio:.3f} keep={len(out)}/{len(entries)}", flush=True)
                return out

            train_entries = _subsample(train_entries, "train")
            val_entries = _subsample(val_entries, "val")
            test_entries = _subsample(test_entries, "test")

        rec_bank = RecSampleBank(args.samples_dir)
        needed_msg_ids = None
        if args.use_cached_needed_msg_ids and needed_msg_ids_cache.exists():
            try:
                cached_ids = torch.load(str(needed_msg_ids_cache), map_location="cpu", weights_only=False)
                if isinstance(cached_ids, (list, tuple, set)):
                    needed_msg_ids = set(str(x) for x in cached_ids)
            except Exception:
                needed_msg_ids = None

        if needed_msg_ids is None:
            needed_msg_ids = collect_needed_msg_ids_from_entries(
                train_entries + val_entries + test_entries,
                user_tweet_ids_map,
                window_size=cfg_window_size,
                long_recent=cfg_long_recent,
            )

        needed_msg_ids.update(rec_bank.all_item_ids)
        try:
            torch.save(sorted(needed_msg_ids), str(needed_msg_ids_cache))
        except Exception:
            pass
        msg_features_dict, msg_feat_dim = load_msg_features_from_compact_shards(
            shards_dir=args.tweet_emb_shards_dir,
            needed_msg_ids=needed_msg_ids,
            verbose=True,
        )
        msg_feature_table_cpu, msg_id_to_row = build_msg_feature_table(msg_features_dict, msg_feat_dim)
        rec_bank.build_tensor_cache(msg_id_to_row)
        print(
            f"[RecCache] keys={len(rec_bank.key_to_row)} "
            f"pos_tbl={tuple(rec_bank.pos_idx_table_cpu.shape) if rec_bank.pos_idx_table_cpu is not None else None} "
            f"neg_tbl={tuple(rec_bank.neg_idx_table_cpu.shape) if rec_bank.neg_idx_table_cpu is not None else None}",
            flush=True,
        )

    rec_table_device_cache: Dict[str, torch.Tensor] = {}

    def _get_rec_table(device_str: str) -> torch.Tensor:
        if torch.cuda.is_available() and str(device_str).startswith("cuda"):
            key = str(device_str)
            if key not in rec_table_device_cache:
                rec_table_device_cache[key] = msg_feature_table_cpu.to(torch.device(device_str), non_blocking=True)
            return rec_table_device_cache[key]
        return msg_feature_table_cpu

    summary = {"experiments": []}
    for cfg in cfgs:
        run_dir = output_root / cfg.name
        run_dir.mkdir(parents=True, exist_ok=True)

        if using_sample_cache:
            train_ds = CachedTimeSliceDataset(sample_cache_dir / "train.pt", msg_feature_table_cpu)
            val_ds = CachedTimeSliceDataset(sample_cache_dir / "val.pt", msg_feature_table_cpu)
            test_ds = CachedTimeSliceDataset(sample_cache_dir / "test.pt", msg_feature_table_cpu)
            split_tags = list(getattr(test_ds, "split_tags", [])) or ["test"] * len(test_ds)
        else:
            split_tags = ["test"] * len(test_entries)
            train_ds = TimeSlicePreferenceDataset(user_feature_map, msg_features_dict, user_tweet_ids_map, train_entries, msg_feat_dim=msg_feat_dim, user_feat_dim=user_feat_dim, short_recent=cfg.short_recent, long_recent=cfg.long_recent, window_size=cfg.window_size)
            val_ds = TimeSlicePreferenceDataset(user_feature_map, msg_features_dict, user_tweet_ids_map, val_entries, msg_feat_dim=msg_feat_dim, user_feat_dim=user_feat_dim, short_recent=cfg.short_recent, long_recent=cfg.long_recent, window_size=cfg.window_size)
            test_ds = TimeSlicePreferenceDataset(user_feature_map, msg_features_dict, user_tweet_ids_map, test_entries, msg_feat_dim=msg_feat_dim, user_feat_dim=user_feat_dim, short_recent=cfg.short_recent, long_recent=cfg.long_recent, window_size=cfg.window_size)

        dl_common = dict(
            collate_fn=collate_fn,
            num_workers=max(0, cfg.num_workers),
            pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        )
        if args.persistent_workers and dl_common["num_workers"] > 0:
            dl_common["persistent_workers"] = True

        train_loader = DataLoader(train_ds, shuffle=True, batch_size=cfg.batch_size, **dl_common)
        val_loader = DataLoader(val_ds, shuffle=False, batch_size=cfg.batch_size, **dl_common)
        test_loader = DataLoader(test_ds, shuffle=False, batch_size=cfg.batch_size, **dl_common)
        export_loader = DataLoader(test_ds, shuffle=False, batch_size=cfg.export_batch_size, **dl_common)

        print(f"[Pipeline] start exp={cfg.name}", flush=True)
        rec_table_device = _get_rec_table(cfg.device)
        best_ckpt = train_one_experiment(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            rec_bank=rec_bank,
            msg_feature_table=rec_table_device,
            msg_feat_dim=msg_feat_dim,
            user_feat_dim=user_feat_dim,
            num_classes=len(label_to_id),
            out_dir=run_dir,
        )

        eval_model, eval_device = load_model_from_ckpt(
            model_ckpt_path=best_ckpt,
            msg_feat_dim=msg_feat_dim,
            user_feat_dim=user_feat_dim,
            num_classes=len(label_to_id),
            device_str=cfg.device,
            ablation=str(cfg.ablation),
        )
        val_labels = val_ds.labels.tolist() if hasattr(val_ds, "labels") else [int(e[3]) for e in val_entries]
        val_class_weights = compute_class_weights(val_labels, len(label_to_id), eval_device)
        val_criterion = FocalLoss(alpha=val_class_weights, gamma=2.0)
        val_repr_metrics = evaluate_cls_and_rec(
            model=eval_model,
            loader=val_loader,
            criterion=val_criterion,
            device=eval_device,
            rec_bank=rec_bank,
            msg_feature_table=rec_table_device if rec_table_device.device == eval_device else msg_feature_table_cpu.to(eval_device, non_blocking=True),
            rec_num_neg=int(cfg.rec_num_neg),
            seed=int(cfg.seed),
            rec_neg_reduction=str(cfg.rec_neg_reduction),
            ablation=str(cfg.ablation),
        )
        test_labels = test_ds.labels.tolist() if hasattr(test_ds, "labels") else [int(e[3]) for e in test_entries]
        test_class_weights = compute_class_weights(test_labels, len(label_to_id), eval_device)
        test_criterion = FocalLoss(alpha=test_class_weights, gamma=2.0)
        test_repr_metrics = evaluate_cls_and_rec(
            model=eval_model,
            loader=test_loader,
            criterion=test_criterion,
            device=eval_device,
            rec_bank=rec_bank,
            msg_feature_table=rec_table_device if rec_table_device.device == eval_device else msg_feature_table_cpu.to(eval_device, non_blocking=True),
            rec_num_neg=int(cfg.rec_num_neg),
            seed=int(cfg.seed),
            rec_neg_reduction=str(cfg.rec_neg_reduction),
            ablation=str(cfg.ablation),
        )
        print(
            f"[{cfg.name}][VAL ] cls_acc={val_repr_metrics['cls_acc']:.4f} "
            f"cls_loss={val_repr_metrics['cls_loss']:.4f} "
            f"rec_pair_acc={val_repr_metrics['rec_pair_acc']:.4f} "
            f"rec_loss={val_repr_metrics['rec_loss']:.4f} "
            f"rec_covered={int(val_repr_metrics['rec_covered_samples'])}",
            flush=True,
        )
        print(
            f"[{cfg.name}][TEST] cls_acc={test_repr_metrics['cls_acc']:.4f} "
            f"cls_loss={test_repr_metrics['cls_loss']:.4f} "
            f"rec_pair_acc={test_repr_metrics['rec_pair_acc']:.4f} "
            f"rec_loss={test_repr_metrics['rec_loss']:.4f} "
            f"rec_covered={int(test_repr_metrics['rec_covered_samples'])}",
            flush=True,
        )

        dyn_emb_path = run_dir / "dynamic_user_embeddings.pt"
        export_embeddings(
            model_ckpt_path=best_ckpt,
            export_loader=export_loader,
            msg_feat_dim=msg_feat_dim,
            user_feat_dim=user_feat_dim,
            num_classes=len(label_to_id),
            device_str=cfg.device,
            out_pt_path=dyn_emb_path,
            split_tags=split_tags,
            ablation=str(cfg.ablation),
            mode=str(cfg.export_mode),
            gate_mode=str(cfg.export_gate_mode),
            kl_threshold=float(cfg.export_kl_threshold),
            kl_soft_scale=float(cfg.export_kl_soft_scale),
            gate_min_alpha=float(cfg.export_gate_min_alpha),
            gate_max_alpha=float(cfg.export_gate_max_alpha),
            amp=bool(cfg.export_amp),
        )
        summary["experiments"].append(
            {
                "config": asdict(cfg),
                "best_ckpt": str(best_ckpt),
                "repr_val_metrics": val_repr_metrics,
                "repr_test_metrics": test_repr_metrics,
                "dynamic_embedding": str(dyn_emb_path),
            }
        )

        del eval_model
        if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    out_summary = output_root / "summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] summary saved: {out_summary}", flush=True)


if __name__ == "__main__":
    main()
