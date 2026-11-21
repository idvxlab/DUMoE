import os
import json
import math
import argparse
import random
import time
import contextlib
import io
from collections import defaultdict
from typing import Dict, List, Tuple

print("[Boot] Script start: entering imports...", flush=True)
print("[Boot] Importing torch and core deps...", flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

print("[Boot] Imports complete, entering main() soon...", flush=True)

from model import PreferenceModel
from time_slice_data_loader import (
    load_time_slice_labels_from_user_dirs,
    load_user_features_from_pt,
    build_user_tweet_ids_map,
    load_user_tweet_ids_map_pt,
    build_label_map,
    build_split_entries,
    collect_needed_msg_ids_from_entries,
    TimeSlicePreferenceDataset,
    collate_fn,
)
from time_slice_msg_feature_loader import (
    collect_user_ids_from_compact_shards,
    load_msg_features_from_compact_shards,
)


TARGET_CATEGORIES = [
    "Entertainment", "Sports", "Technology", "Business", "Education", "Gaming", "Politics", "Art",
    "Culture", "Science", "Food", "Daily Life", "Fashion & Beauty", "Health & Fitness", "News"
]

_ROC_AUC_IMPORT_TRIED = False
_ROC_AUC_SCORE = None


def _default_label_user_dirs() -> str:
    return ""


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object] | None) -> None:
    if not state:
        return
    try:
        if "python_random" in state:
            random.setstate(state["python_random"])
    except Exception:
        pass
    try:
        if "torch_cpu" in state:
            torch.random.set_rng_state(state["torch_cpu"])
    except Exception:
        pass
    try:
        if "numpy" in state:
            import numpy as np
            np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and "torch_cuda_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception:
        pass


def atomic_torch_save(payload: Dict[str, object], path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _get_roc_auc_score_fn():
    global _ROC_AUC_IMPORT_TRIED, _ROC_AUC_SCORE
    if _ROC_AUC_IMPORT_TRIED:
        return _ROC_AUC_SCORE

    _ROC_AUC_IMPORT_TRIED = True
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from sklearn.metrics import roc_auc_score as _roc_auc_score
        _ROC_AUC_SCORE = _roc_auc_score
    except Exception as e:
        _ROC_AUC_SCORE = None
        print(f"[Warn] roc_auc_ovr disabled because sklearn/scipy import failed: {e}")
    return _ROC_AUC_SCORE


def _month_sort_key(month: str):
    m = str(month)
    digits = "".join(ch for ch in m if ch.isdigit())
    if digits:
        try:
            return (0, int(digits), m)
        except Exception:
            pass
    return (1, m)


def compute_switch_metrics(records: List[Tuple[str, str, int, int]]) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for uid, month, y_true, y_pred in records:
        grouped[str(uid)].append((str(month), int(y_true), int(y_pred)))

    tp = fp = fn = tn = 0
    user_mae: List[float] = []
    users_with_transitions = 0

    for _, seq in grouped.items():
        seq.sort(key=lambda x: _month_sort_key(x[0]))
        if len(seq) < 2:
            continue

        users_with_transitions += 1
        true_ids = [x[1] for x in seq]
        pred_ids = [x[2] for x in seq]

        true_switches = 0
        pred_switches = 0
        for i in range(1, len(seq)):
            t_change = int(true_ids[i] != true_ids[i - 1])
            p_change = int(pred_ids[i] != pred_ids[i - 1])
            true_switches += t_change
            pred_switches += p_change

            if t_change == 1 and p_change == 1:
                tp += 1
            elif t_change == 0 and p_change == 1:
                fp += 1
            elif t_change == 1 and p_change == 0:
                fn += 1
            else:
                tn += 1

        user_mae.append(abs(float(pred_switches - true_switches)))

    total = tp + fp + fn + tn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "switch_precision": float(precision),
        "switch_recall": float(recall),
        "switch_f1": float(f1),
        "switch_acc": float((tp + tn) / max(1, total)),
        "true_switch_rate": float((tp + fn) / max(1, total)),
        "pred_switch_rate": float((tp + fp) / max(1, total)),
        "switch_count_mae": float(sum(user_mae) / max(1, len(user_mae))),
        "switch_eval_users": float(users_with_transitions),
    }


def build_switch_flag_map(
    entries: List[Tuple[str, int, str, int]],
    min_window_idx: int = 1,
) -> Dict[Tuple[str, int], float]:
    grouped: Dict[str, List[Tuple[int, str, int]]] = defaultdict(list)
    for uid, window_idx, month, label in entries:
        grouped[str(uid)].append((int(window_idx), str(month), int(label)))

    switch_map: Dict[Tuple[str, int], float] = {}
    for uid, seq in grouped.items():
        seq.sort(key=lambda x: x[0])
        prev_label = None
        for window_idx, month, label in seq:
            is_switch = 0.0
            if prev_label is not None and window_idx >= min_window_idx and label != prev_label:
                is_switch = 1.0
            switch_map[(uid, int(window_idx))] = is_switch
            prev_label = label

    return switch_map


def compute_transition_window_metrics(
    records: List[Tuple[str, int, int, int, List[int]]]
) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[int, int, int, List[int]]]] = defaultdict(list)
    for uid, window_idx, y_true, y_pred, top3_ids in records:
        grouped[str(uid)].append((int(window_idx), int(y_true), int(y_pred), [int(x) for x in top3_ids]))

    switch_top1_hits = 0
    switch_top3_hits = 0
    switch_count = 0
    nonswitch_top1_hits = 0
    nonswitch_count = 0

    for _, seq in grouped.items():
        seq.sort(key=lambda x: x[0])
        if len(seq) < 2:
            continue

        prev_true = None
        for window_idx, y_true, y_pred, top3_ids in seq:
            if prev_true is None:
                prev_true = y_true
                continue

            is_switch = int(y_true != prev_true)
            if is_switch == 1:
                switch_count += 1
                if y_pred == y_true:
                    switch_top1_hits += 1
                if y_true in top3_ids:
                    switch_top3_hits += 1
            else:
                nonswitch_count += 1
                if y_pred == y_true:
                    nonswitch_top1_hits += 1

            prev_true = y_true

    return {
        "switch_window_top1": switch_top1_hits / max(1, switch_count),
        "switch_window_top3": switch_top3_hits / max(1, switch_count),
        "switch_window_count": float(switch_count),
        "non_switch_window_top1": nonswitch_top1_hits / max(1, nonswitch_count),
        "non_switch_window_count": float(nonswitch_count),
    }


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 3, 5)) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    metrics["accuracy"] = (preds == labels).float().mean().item()

    num_classes = logits.shape[1]
    recalls = []
    for cls in range(num_classes):
        cls_mask = labels == cls
        if cls_mask.sum() == 0:
            continue
        recalls.append((preds[cls_mask] == cls).float().mean().item())
    metrics["recall_macro"] = sum(recalls) / max(1, len(recalls))

    max_k = max(topk)
    _, topk_idx = torch.topk(probs, k=max_k, dim=1)
    for k in topk:
        correct = topk_idx[:, :k].eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"top{k}_acc"] = correct

    for k in topk:
        hits = topk_idx[:, :k].eq(labels.unsqueeze(1))
        gains = hits.float()
        discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=logits.device).float())
        dcg = (gains * discounts).sum(dim=1)
        ndcg = dcg.mean().item()
        metrics[f"ndcg@{k}"] = ndcg

    target_dist = F.one_hot(labels, num_classes).float()
    target_dist = target_dist * 0.9 + 0.1 / num_classes
    kl = F.kl_div(torch.log(probs + 1e-12), target_dist, reduction="batchmean").item()
    metrics["kl_div"] = kl

    auc_fn = _get_roc_auc_score_fn()
    if auc_fn is not None:
        try:
            y_true = labels.cpu().numpy()
            y_score = probs.detach().cpu().numpy()
            metrics["roc_auc_ovr"] = auc_fn(y_true, y_score, multi_class="ovr")
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")
    else:
        metrics["roc_auc_ovr"] = float("nan")

    return metrics


def _forward_with_adapter_flag(
    model: nn.Module,
    use_adapter: bool,
    x_explicit: torch.Tensor,
    x_implicit: torch.Tensor,
    x_cross_time: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    original_flag = model.use_adapter
    model.use_adapter = use_adapter
    try:
        logits = model(x_explicit, x_implicit, x_cross_time, src_key_padding_mask)
    finally:
        model.use_adapter = original_flag
    return logits


@torch.no_grad()
def kl_gated_logits(
    model: nn.Module,
    x_explicit: torch.Tensor,
    x_implicit: torch.Tensor,
    x_cross_time: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    x_implicit_no_month: torch.Tensor,
    x_cross_time_no_month: torch.Tensor,
    src_key_padding_mask_no_month: torch.Tensor,
    kl_threshold: float,
    gate_mode: str = "hard",
    kl_soft_scale: float = 0.02,
    gate_min_alpha: float = 0.0,
    gate_max_alpha: float = 1.0,
    logit_temperature: float = 1.0,
) -> torch.Tensor:
    logits_base = _forward_with_adapter_flag(
        model, False, x_explicit, x_implicit, x_cross_time, src_key_padding_mask
    )
    logits_no_month = _forward_with_adapter_flag(
        model, False, x_explicit, x_implicit_no_month, x_cross_time_no_month, src_key_padding_mask_no_month
    )

    probs_base = torch.softmax(logits_base, dim=1)
    probs_no_month = torch.softmax(logits_no_month, dim=1)
    kl = (probs_base * (probs_base.add(1e-12).log() - probs_no_month.add(1e-12).log())).sum(dim=1)

    logits_adapter = _forward_with_adapter_flag(
        model, True, x_explicit, x_implicit, x_cross_time, src_key_padding_mask
    )

    if gate_mode == "soft":
        num_classes = probs_base.size(1)
        entropy = -(probs_base * probs_base.add(1e-12).log()).sum(dim=1)
        entropy_norm = entropy / max(1e-6, math.log(num_classes))

        z = (kl - kl_threshold) / max(1e-6, kl_soft_scale)
        alpha_kl = torch.sigmoid(z)
        alpha = alpha_kl * (0.5 + 0.5 * entropy_norm)
        alpha = alpha.clamp(min=gate_min_alpha, max=gate_max_alpha).unsqueeze(1)
        logits_final = (1.0 - alpha) * logits_base + alpha * logits_adapter
    else:
        activate_mask = kl > kl_threshold
        if activate_mask.any():
            logits_final = logits_base.clone()
            logits_final[activate_mask] = logits_adapter[activate_mask]
        else:
            logits_final = logits_base

    if logit_temperature > 0 and abs(logit_temperature - 1.0) > 1e-8:
        logits_final = logits_final / logit_temperature

    return logits_final


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
            label_smoothing=0.1,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes, device=device)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 0)
        if count > 0:
            weights[cls_idx] = (total / (num_classes * count)) ** 0.5
    return weights


def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / max(1, cnt) for cls, cnt in counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"[NaNCheck] {name} contains NaN/Inf")


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


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    filter_labels=None,
    log_every: int = 50,
    adapter_index: int = None,
    ablation: str = "none",
):
    model.train()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_steps = len(loader) if hasattr(loader, "__len__") else None

    for step, batch in enumerate(loader, start=1):
        x_explicit = batch["x_explicit"].to(device)
        x_implicit = batch["x_implicit"].to(device)
        x_cross_time = batch["x_cross_time"].to(device)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)
        labels = batch["labels"].to(device)

        if filter_labels is not None:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for lbl in filter_labels:
                mask = mask | (labels == lbl)
            if mask.sum() == 0:
                continue
            x_explicit = x_explicit[mask]
            x_implicit = x_implicit[mask]
            x_cross_time = x_cross_time[:, mask, :]
            src_key_padding_mask = src_key_padding_mask[mask]
            labels = labels[mask]

        x_explicit, x_implicit, x_cross_time, src_key_padding_mask = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit,
            x_cross_time,
            src_key_padding_mask,
        )

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                x_explicit,
                x_implicit,
                x_cross_time,
                src_key_padding_mask,
                adapter_index=adapter_index,
            )
            _assert_finite_tensor("train_logits", logits)
            loss = criterion(logits, labels)
            _assert_finite_tensor("train_loss", loss)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if log_every and step % log_every == 0:
            avg_loss = total_loss / max(1, total_samples)
            avg_acc = total_correct / max(1, total_samples)
            if total_steps:
                print(f"  [Batch {step}/{total_steps}] loss={avg_loss:.4f} acc={avg_acc:.4f}")
            else:
                print(f"  [Batch {step}] loss={avg_loss:.4f} acc={avg_acc:.4f}")

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_kl_adapter: bool = False,
    kl_threshold: float = 0.1,
    gate_mode: str = "hard",
    kl_soft_scale: float = 0.02,
    gate_min_alpha: float = 0.0,
    gate_max_alpha: float = 1.0,
    logit_temperature: float = 1.0,
    ablation: str = "none",
):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_logits = []
    all_labels = []
    switch_records: List[Tuple[str, str, int, int]] = []
    transition_records: List[Tuple[str, int, int, int, List[int]]] = []

    for batch in loader:
        x_explicit = batch["x_explicit"].to(device)
        x_implicit = batch["x_implicit"].to(device)
        x_cross_time = batch["x_cross_time"].to(device)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)
        labels = batch["labels"].to(device)

        x_implicit_no_month = batch["x_implicit_no_month"].to(device)
        x_cross_time_no_month = batch["x_cross_time_no_month"].to(device)
        src_key_padding_mask_no_month = batch["src_key_padding_mask_no_month"].to(device)

        x_explicit, x_implicit, x_cross_time, src_key_padding_mask = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit,
            x_cross_time,
            src_key_padding_mask,
        )
        _, x_implicit_no_month, x_cross_time_no_month, src_key_padding_mask_no_month = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit_no_month,
            x_cross_time_no_month,
            src_key_padding_mask_no_month,
        )

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_kl_adapter:
                logits = kl_gated_logits(
                    model,
                    x_explicit,
                    x_implicit,
                    x_cross_time,
                    src_key_padding_mask,
                    x_implicit_no_month,
                    x_cross_time_no_month,
                    src_key_padding_mask_no_month,
                    kl_threshold,
                    gate_mode,
                    kl_soft_scale,
                    gate_min_alpha,
                    gate_max_alpha,
                    logit_temperature,
                )
            else:
                logits = model(x_explicit, x_implicit, x_cross_time, src_key_padding_mask)
                if logit_temperature > 0 and abs(logit_temperature - 1.0) > 1e-8:
                    logits = logits / logit_temperature

            _assert_finite_tensor("eval_logits", logits)
            loss = criterion(logits, labels)
            _assert_finite_tensor("eval_loss", loss)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        batch_user_ids = batch.get("user_ids", [])
        batch_months = batch.get("months", [])
        batch_window_indices = batch.get("window_indices", [])

        y_true = labels.detach().cpu().tolist()
        y_pred = preds.detach().cpu().tolist()
        top3_idx = torch.topk(logits.detach(), k=min(3, logits.size(1)), dim=1).indices.cpu().tolist()

        if len(batch_user_ids) == labels.size(0) and len(batch_months) == labels.size(0):
            for uid, month, yt, yp in zip(batch_user_ids, batch_months, y_true, y_pred):
                switch_records.append((str(uid), str(month), int(yt), int(yp)))

        if len(batch_user_ids) == labels.size(0) and len(batch_window_indices) == labels.size(0):
            for uid, widx, yt, yp, t3 in zip(batch_user_ids, batch_window_indices, y_true, y_pred, top3_idx):
                transition_records.append((str(uid), int(widx), int(yt), int(yp), [int(x) for x in t3]))

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    metrics = {}
    if all_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(logits_cat, labels_cat)
        metrics.update(compute_switch_metrics(switch_records))
        if transition_records:
            metrics.update(compute_transition_window_metrics(transition_records))

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples), metrics


@torch.no_grad()
def predict_probabilities(
    model,
    loader,
    device,
    label_to_id: Dict[str, int],
    use_kl_adapter: bool = False,
    kl_threshold: float = 0.1,
    gate_mode: str = "hard",
    kl_soft_scale: float = 0.02,
    gate_min_alpha: float = 0.0,
    gate_max_alpha: float = 1.0,
    logit_temperature: float = 1.0,
    ablation: str = "none",
) -> Dict[str, Dict[str, List[float]]]:
    id_to_label = {v: k for k, v in label_to_id.items()}
    model.eval()
    results: Dict[str, Dict[str, List[float]]] = {}

    for batch in loader:
        x_explicit = batch["x_explicit"].to(device)
        x_implicit = batch["x_implicit"].to(device)
        x_cross_time = batch["x_cross_time"].to(device)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)

        x_implicit_no_month = batch["x_implicit_no_month"].to(device)
        x_cross_time_no_month = batch["x_cross_time_no_month"].to(device)
        src_key_padding_mask_no_month = batch["src_key_padding_mask_no_month"].to(device)

        x_explicit, x_implicit, x_cross_time, src_key_padding_mask = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit,
            x_cross_time,
            src_key_padding_mask,
        )
        _, x_implicit_no_month, x_cross_time_no_month, src_key_padding_mask_no_month = apply_input_ablation(
            ablation,
            x_explicit,
            x_implicit_no_month,
            x_cross_time_no_month,
            src_key_padding_mask_no_month,
        )

        if use_kl_adapter:
            logits = kl_gated_logits(
                model,
                x_explicit,
                x_implicit,
                x_cross_time,
                src_key_padding_mask,
                x_implicit_no_month,
                x_cross_time_no_month,
                src_key_padding_mask_no_month,
                kl_threshold,
                gate_mode,
                kl_soft_scale,
                gate_min_alpha,
                gate_max_alpha,
                logit_temperature,
            )
        else:
            logits = model(x_explicit, x_implicit, x_cross_time, src_key_padding_mask)
            if logit_temperature > 0 and abs(logit_temperature - 1.0) > 1e-8:
                logits = logits / logit_temperature

        _assert_finite_tensor("predict_logits", logits)
        probs = torch.softmax(logits, dim=1).cpu().tolist()

        for uid, month, prob in zip(batch["user_ids"], batch["months"], probs):
            results.setdefault(uid, {})[month] = {
                id_to_label[i]: prob[i] for i in range(len(prob))
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Time-slice preference training/inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained model checkpoint")
    parser.add_argument("--infer-only", action="store_true", help="Skip training and only run test inference")
    parser.add_argument("--stage3-only", action="store_true", help="Skip stage1/2 and continue only stage3 training")
    parser.add_argument("--use-kl-adapter", action="store_true", help="Enable KL-gated adapter activation in inference")
    parser.add_argument("--kl-threshold", type=float, default=0.1, help="KL threshold to activate adapter")
    parser.add_argument("--kl-thresholds", type=str, default="", help="Comma-separated KL thresholds to sweep on validation set")
    parser.add_argument("--kl-metric", type=str, default="top1_acc", help="Metric key used to select best KL threshold on validation")

    parser.add_argument("--kl-score-recall-weight", type=float, default=0.3, help="Weight for recall_macro in multi-objective KL score")
    parser.add_argument("--kl-score-top3-weight", type=float, default=0.1, help="Weight for top3_acc in multi-objective KL score")
    parser.add_argument("--kl-score-switch-weight", type=float, default=0.5, help="Weight for switch_f1 in multi-objective KL score")
    parser.add_argument("--kl-score-switch-window-top1-weight", type=float, default=1.0, help="Weight for switch_window_top1 in KL selection")
    parser.add_argument("--kl-score-switch-window-top3-weight", type=float, default=0.3, help="Weight for switch_window_top3 in KL selection")

    parser.add_argument("--kl-max-auc-drop", type=float, default=0.01, help="Max allowed AUC drop from no-gate baseline when selecting KL threshold")
    parser.add_argument("--kl-max-top1-drop", type=float, default=0.002, help="Max allowed top1 drop from no-gate baseline when selecting KL threshold")
    parser.add_argument("--kl-max-nonswitch-top1-drop", type=float, default=0.003, help="Max allowed non_switch_window_top1 drop from no-gate baseline when selecting KL threshold")

    parser.add_argument("--kl-selection-mode", type=str, default="multi", choices=["single", "multi"], help="single: legacy metric, multi: constrained multi-objective selection")
    parser.add_argument("--kl-sweep-start", type=float, default=0.01, help="Auto KL sweep start")
    parser.add_argument("--kl-sweep-end", type=float, default=0.06, help="Auto KL sweep end")
    parser.add_argument("--kl-sweep-step", type=float, default=0.005, help="Auto KL sweep step")

    parser.add_argument("--gate-mode", type=str, default="soft", choices=["hard", "soft"], help="Gate fusion mode")
    parser.add_argument("--kl-soft-scale", type=float, default=0.02, help="Soft gate sigmoid scale on KL")
    parser.add_argument("--gate-min-alpha", type=float, default=0.0, help="Soft gate minimum alpha")
    parser.add_argument("--gate-max-alpha", type=float, default=1.0, help="Soft gate maximum alpha")
    parser.add_argument("--gate-logit-temperature", type=float, default=1.0, help="Temperature for calibrated logits in evaluation/inference")

    parser.add_argument("--stage3-gain-margin", type=float, default=0.02, help="Adapter-vs-base loss margin for gain supervision target")
    parser.add_argument("--stage3-gain-weight", type=float, default=0.3, help="Weight for gain supervision BCE loss in stage3")
    parser.add_argument("--stage3-switch-weight", type=float, default=0.0, help="Additional weight for switch windows in stage3 loss: w = 1 + stage3_switch_weight * is_switch")
    parser.add_argument("--stage3-switch-min-window-idx", type=int, default=1, help="Only windows with idx >= this are eligible for switch supervision weight")

    parser.add_argument("--balanced-sampler", action="store_true", help="Use weighted sampling to improve recall on minority classes")
    parser.add_argument("--gate-kl-weight", type=float, default=0.2, help="Stage3 gate-weight KL loss weight (set 0 to disable)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last saved training state")
    parser.add_argument("--allow-ablation-resume", action="store_true", help="Allow --resume in ablation mode (default is disabled to prevent contamination)")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Stop a stage if metric doesn't improve for N epochs")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum improvement to reset early stop counter")

    parser.add_argument("--label-user-dirs", type=str, default=_default_label_user_dirs(), help="Comma-separated label sources (dirs or merged json files)")
    parser.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt", help="Path to user_profile_1542.pt")
    parser.add_argument(
        "--user-tweet-ids-map",
        type=str,
        default="",
        help="Optional .pt map {user_id: [tweet_id,...]} for anonymized data (skips raw tweet CSVs).",
    )
    parser.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt", help="Directory containing w*_tweet_emb_shard_*.pt")

    parser.add_argument("--window-size", type=int, default=30, help="Label window size in tweets")
    parser.add_argument("--short-recent", type=int, default=30, help="Short-term sequence length (recent tweets)")
    parser.add_argument("--long-recent", type=int, default=300, help="Long-term sequence length (recent tweets)")
    parser.add_argument("--report-only", action="store_true", help="Only compute intersection/user stats and exit")

    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--stage1-epochs", type=int, default=10, help="Epochs for Stage1")
    parser.add_argument("--stage2-epochs", type=int, default=4, help="Epochs for each adapter in Stage2")
    parser.add_argument("--stage3-epochs", type=int, default=5, help="Epochs for Stage3")
    parser.add_argument("--lr-stage1", type=float, default=1e-4, help="Learning rate for Stage1")
    parser.add_argument("--lr-stage2", type=float, default=5e-4, help="Learning rate for Stage2")
    parser.add_argument("--lr-stage3", type=float, default=1e-3, help="Learning rate for Stage3")
    parser.add_argument("--training-strategy", type=str, default="staged", choices=["staged", "joint"], help="staged: Stage1->Stage2->Stage3, joint: train backbone+adapter+gate together")
    parser.add_argument("--joint-epochs", type=int, default=25, help="Total epochs for joint training strategy")
    parser.add_argument("--lr-joint", type=float, default=3e-4, help="Learning rate for joint training strategy")

    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")

    parser.add_argument("--use-cached-intersection", action="store_true", help="Reuse intersection user ids from cache if available")
    parser.add_argument("--intersection-cache", type=str, default="", help="Path to cached intersection json (default: time_slice_user_intersection_stats.json)")
    parser.add_argument("--use-cached-needed-msg-ids", action="store_true", help="Reuse cached needed message ids if available")
    parser.add_argument("--needed-msg-ids-cache", type=str, default="", help="Path to cached needed message ids .pt (default under base dir)")
    parser.add_argument("--use-cached-labels", action="store_true", help="Load parsed labels/csv_path map from local json cache when available")
    parser.add_argument("--labels-cache", type=str, default="", help="Path to cached labels json (default under base dir)")
    parser.add_argument("--label-parse-progress-every", type=int, default=500, help="Print label parse progress every N files (0 to disable)")

    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_adapter", "no_gating", "only_user", "only_short", "only_long"], help="Ablation mode")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to store model/checkpoints/metrics/predictions")

    args = parser.parse_args()

    data_root = "."
    base = (args.output_dir or "").strip() or data_root
    os.makedirs(base, exist_ok=True)

    metrics_path = os.path.join(base, "time_slice_metrics.json")
    resume_state_path = os.path.join(base, "time_slice_train_state.pth")
    intersection_report_path = os.path.join(data_root, "time_slice_user_intersection_stats.json")
    needed_msg_ids_cache_path = os.path.join(data_root, "time_slice_needed_msg_ids.pt")
    labels_cache_path = os.path.join(data_root, "time_slice_labels_cache.json")
    metrics_log = []
    label_user_dirs = [x.strip() for x in args.label_user_dirs.split(",") if x.strip()]

    if args.intersection_cache:
        intersection_report_path = args.intersection_cache
    if args.needed_msg_ids_cache:
        needed_msg_ids_cache_path = args.needed_msg_ids_cache
    if args.labels_cache:
        labels_cache_path = args.labels_cache

    set_global_seed(args.seed)
    print(f"[Config] seed={args.seed} batch_size={args.batch_size} stage1={args.stage1_epochs} stage2={args.stage2_epochs} stage3={args.stage3_epochs}")
    print(f"[Config] short_recent={args.short_recent} long_recent={args.long_recent} window_size={args.window_size}")
    print(f"[Config] lr_stage1={args.lr_stage1} lr_stage2={args.lr_stage2} lr_stage3={args.lr_stage3}")
    print(f"[Config] training_strategy={args.training_strategy} joint_epochs={args.joint_epochs} lr_joint={args.lr_joint}")
    print(f"[Config] ablation={args.ablation}")
    print(f"[Config] output_dir={base}")

    print("[Data] Loading labels from configured sources...")
    labels_by_user, csv_path_by_user = load_time_slice_labels_from_user_dirs(
        label_user_dirs,
        cache_path=labels_cache_path,
        use_cache=args.use_cached_labels,
        progress_every=max(0, int(args.label_parse_progress_every)),
    )
    print(f"[Data] Loaded labels for {len(labels_by_user)} users from label sources")

    print("[Data] Loading user profile features...")
    user_feature_map, user_feat_dim = load_user_features_from_pt(args.user_profile_pt)
    print(f"[Data] Loaded user profile features: users={len(user_feature_map)} dim={user_feat_dim}")

    users_labels = set(labels_by_user.keys())
    users_userfeat = set(user_feature_map.keys())
    users_csv = set(csv_path_by_user.keys())
    prebuilt_map = None
    map_path = str(getattr(args, "user_tweet_ids_map", "") or "").strip()
    if map_path:
        prebuilt_map = load_user_tweet_ids_map_pt(map_path)
        users_csv = set(prebuilt_map.keys())
        print(f"[Data] Loaded user_tweet_ids_map: users={len(prebuilt_map)} from {map_path}")

    users_intersection = None
    users_in_tweet_shards = set()
    if args.use_cached_intersection and os.path.exists(intersection_report_path):
        try:
            with open(intersection_report_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_users = set(str(x) for x in cached.get("intersection_user_ids", []))
            users_intersection = cached_users & users_labels & users_userfeat & users_csv
            print(f"[Data] Loaded cached intersection: {len(cached_users)} -> usable={len(users_intersection)}")
        except Exception as exc:
            print(f"[WARN] Failed loading cached intersection, fallback to rebuild: {exc}")

    if users_intersection is None:
        if prebuilt_map is not None:
            users_intersection = users_labels & users_userfeat & users_csv
            print(
                "[Data] User intersection (anonymized map) | "
                f"labels={len(users_labels)} user_features={len(users_userfeat)} "
                f"tweet_map={len(users_csv)} intersection={len(users_intersection)}"
            )
        else:
            print("[Data] Collecting users from tweet embedding shards...")
            users_in_tweet_shards = collect_user_ids_from_compact_shards(args.tweet_emb_shards_dir, verbose=True)
            print(f"[Data] Users in tweet embedding shards: {len(users_in_tweet_shards)}")
            users_intersection = users_labels & users_userfeat & users_in_tweet_shards & users_csv

        print(
            "[Data] User intersection | "
            f"labels={len(users_labels)} user_features={len(users_userfeat)} "
            f"tweet_shards={len(users_in_tweet_shards)} csv_meta={len(users_csv)} "
            f"intersection={len(users_intersection)}"
        )

        with open(intersection_report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels_users": len(users_labels),
                    "user_feature_users": len(users_userfeat),
                    "tweet_shard_users": len(users_in_tweet_shards),
                    "csv_meta_users": len(users_csv),
                    "intersection_users": len(users_intersection),
                    "intersection_user_ids": sorted(users_intersection),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Data] Saved intersection report to {intersection_report_path}")

    if args.report_only:
        print("[Data] --report-only enabled, exit.")
        return

    labels_by_user = {u: labels_by_user[u] for u in users_intersection}
    csv_path_by_user = {u: csv_path_by_user[u] for u in users_intersection if u in csv_path_by_user}
    user_feature_map = {u: user_feature_map[u] for u in users_intersection}

    if prebuilt_map is not None:
        print("[Data] Using prebuilt user_tweet_ids_map (no raw tweet CSVs).")
        user_tweet_ids_map = {u: prebuilt_map[u] for u in users_intersection if u in prebuilt_map}
    else:
        print("[Data] Building per-user ordered tweet id map from csv_path...")
        user_tweet_ids_map = build_user_tweet_ids_map(csv_path_by_user)
    users_with_tweets = {u for u, tids in user_tweet_ids_map.items() if len(tids) > 0}
    users_intersection = users_intersection & users_with_tweets
    print(f"[Data] Users with non-empty ordered tweet ids: {len(users_intersection)}")

    labels_by_user = {u: labels_by_user[u] for u in users_intersection}
    user_feature_map = {u: user_feature_map[u] for u in users_intersection}
    user_tweet_ids_map = {u: user_tweet_ids_map[u] for u in users_intersection}

    label_to_id = build_label_map(TARGET_CATEGORIES)
    train_entries, val_entries, test_entries = build_split_entries(labels_by_user, label_to_id)

    def _entry_has_context(entry):
        uid, window_idx, _, _ = entry
        start = window_idx * args.window_size
        return start < len(user_tweet_ids_map.get(uid, []))

    train_entries = [e for e in train_entries if _entry_has_context(e)]
    val_entries = [e for e in val_entries if _entry_has_context(e)]
    test_entries = [e for e in test_entries if _entry_has_context(e)]
    print(f"[Data] Split entries: train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}")

    train_switch_flag_map = build_switch_flag_map(
        train_entries,
        min_window_idx=max(0, int(args.stage3_switch_min_window_idx)),
    )
    train_switch_ratio = 0.0
    if train_switch_flag_map:
        train_switch_ratio = sum(train_switch_flag_map.values()) / len(train_switch_flag_map)
    print(
        f"[Stage3] switch_weight={args.stage3_switch_weight:.4f} "
        f"train_switch_ratio={train_switch_ratio:.4f} "
        f"switch_flags={len(train_switch_flag_map)}"
    )

    all_entries = train_entries + val_entries + test_entries
    needed_msg_ids = None
    if args.use_cached_needed_msg_ids and os.path.exists(needed_msg_ids_cache_path):
        try:
            cached_ids = torch.load(needed_msg_ids_cache_path, map_location="cpu")
            if isinstance(cached_ids, (list, tuple, set)):
                needed_msg_ids = set(str(x) for x in cached_ids)
                print(f"[Data] Loaded cached needed_msg_ids: {len(needed_msg_ids)}")
        except Exception as exc:
            print(f"[WARN] Failed loading needed_msg_ids cache, fallback to rebuild: {exc}")

    if needed_msg_ids is None:
        print(f"[Data] Collecting needed message IDs from {len(all_entries)} entries...")
        needed_msg_ids = collect_needed_msg_ids_from_entries(
            all_entries,
            user_tweet_ids_map,
            window_size=args.window_size,
            long_recent=args.long_recent,
        )
        print(f"[Data] Collected {len(needed_msg_ids)} unique message IDs")
        try:
            torch.save(sorted(needed_msg_ids), needed_msg_ids_cache_path)
            print(f"[Data] Saved needed_msg_ids cache to {needed_msg_ids_cache_path}")
        except Exception as exc:
            print(f"[WARN] Failed saving needed_msg_ids cache: {exc}")

    print("[Data] Loading message feature shards...")
    msg_features_dict, msg_feat_dim = load_msg_features_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir,
        needed_msg_ids=needed_msg_ids,
        verbose=True,
    )
    print(f"[Data] Loaded {len(msg_features_dict)} message features with dim={msg_feat_dim}")

    print("[Data] Building datasets...")
    train_ds = TimeSlicePreferenceDataset(
        user_feature_map,
        msg_features_dict,
        user_tweet_ids_map,
        train_entries,
        msg_feat_dim=msg_feat_dim,
        user_feat_dim=user_feat_dim,
        short_recent=args.short_recent,
        long_recent=args.long_recent,
        window_size=args.window_size,
    )
    val_ds = TimeSlicePreferenceDataset(
        user_feature_map,
        msg_features_dict,
        user_tweet_ids_map,
        val_entries,
        msg_feat_dim=msg_feat_dim,
        user_feat_dim=user_feat_dim,
        short_recent=args.short_recent,
        long_recent=args.long_recent,
        window_size=args.window_size,
    )
    test_ds = TimeSlicePreferenceDataset(
        user_feature_map,
        msg_features_dict,
        user_tweet_ids_map,
        test_entries,
        msg_feat_dim=msg_feat_dim,
        user_feat_dim=user_feat_dim,
        short_recent=args.short_recent,
        long_recent=args.long_recent,
        window_size=args.window_size,
    )

    train_labels = [entry[3] for entry in train_entries]
    dl_common = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": max(0, args.num_workers),
        "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
    }
    if dl_common["num_workers"] > 0:
        dl_common["persistent_workers"] = True

    if args.balanced_sampler:
        sampler = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_ds, sampler=sampler, **dl_common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_common)

    val_loader = DataLoader(val_ds, shuffle=False, **dl_common)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_common)
    print(f"[Data] Dataloaders ready: train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}")

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreferenceModel(
        user_feat_dim=user_feat_dim,
        msg_feat_dim=msg_feat_dim,
        num_classes=len(label_to_id),
        hidden_dim=256,
        lstm_layers=1,
        transformer_layers=2,
        nhead=8,
        dropout=0.4,
        adapter_bottleneck=(checkpoint.get("config", {}).get("adapter_bottleneck", 64) if checkpoint else 64),
        use_adapter=False,
        topk_adapters=(checkpoint.get("config", {}).get("topk_adapters", 3) if checkpoint else 3),
        use_uniform_adapter_fusion=(args.ablation == "no_gating"),
    ).to(device)

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if args.ablation != "none" and args.resume and not args.allow_ablation_resume:
        print(
            "[Ablation] --resume is disabled by default in ablation mode to avoid"
            " mixed-stage contamination. Use --allow-ablation-resume to override."
        )
        args.resume = False

    resume_state = None
    if args.resume and os.path.exists(resume_state_path):
        resume_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        restore_rng_state(resume_state.get("rng_state"))
        print(f"[Resume] Loaded training state from {resume_state_path}")

    class_weights = compute_class_weights(train_labels, len(label_to_id), device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def save_checkpoint(stage_name: str):
        model_path = os.path.join(base, "time_slice_model.pth")
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "user_feat_dim": user_feat_dim,
                    "msg_feat_dim": msg_feat_dim,
                    "num_classes": len(label_to_id),
                    "adapter_bottleneck": 64,
                    "topk_adapters": 3,
                },
                "stage": stage_name,
            },
            model_path,
        )
        print(f"Saved model to {model_path} (stage={stage_name})")

    def save_train_state(stage: str, epoch: int, optimizer_obj, scaler_obj, adapter_idx: int = None, adapter_epoch: int = None):
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_obj.state_dict() if optimizer_obj is not None else None,
                "scaler_state_dict": scaler_obj.state_dict() if scaler_obj is not None else None,
                "stage": stage,
                "epoch": epoch,
                "adapter_idx": adapter_idx,
                "adapter_epoch": adapter_epoch,
                "rng_state": capture_rng_state(),
            },
            resume_state_path,
        )
        print(f"[Resume] Saved training state to {resume_state_path} (stage={stage}, epoch={epoch})")

    resume_stage = str(resume_state.get("stage")) if resume_state else None
    metrics_log.append({"stage": "meta", "ablation": args.ablation, "seed": args.seed})

    if not args.infer_only:
        run_stage1 = resume_stage in (None, "stage1")
        run_stage2 = resume_stage in (None, "stage1", "stage2")
        run_stage3 = resume_stage in (None, "stage1", "stage2", "stage3")

        if args.training_strategy == "joint":
            # Joint strategy uses a single end-to-end training stage and skips stage2/stage3.
            run_stage2 = False
            run_stage3 = False

        if args.stage3_only:
            run_stage1 = False
            run_stage2 = False
            run_stage3 = True

        if args.ablation == "no_adapter":
            run_stage2 = False
            run_stage3 = False
        elif args.ablation == "no_gating":
            run_stage3 = False

        if run_stage1:
            stage1_tag = "joint" if args.training_strategy == "joint" else "stage1"
            if args.training_strategy == "joint":
                print("[Joint] Starting end-to-end training (backbone+adapter+gate)")
                stage1_epochs = args.joint_epochs

                for p in model.parameters():
                    p.requires_grad = True

                if args.ablation == "no_adapter":
                    model.use_adapter = False
                    model.use_uniform_adapter_fusion = False
                    for adapter in model.adapter_pool:
                        for p in adapter.parameters():
                            p.requires_grad = False
                    for p in model.gating_network.parameters():
                        p.requires_grad = False
                elif args.ablation == "no_gating":
                    model.use_adapter = True
                    model.use_uniform_adapter_fusion = True
                    for p in model.gating_network.parameters():
                        p.requires_grad = False
                else:
                    model.use_adapter = True
                    model.use_uniform_adapter_fusion = False

                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_joint, weight_decay=1e-2)
                scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
            else:
                print("[Stage1] Starting backbone training")
                stage1_epochs = args.stage1_epochs

            start_epoch = 1
            best_val = None
            no_improve = 0

            if resume_state and resume_stage == stage1_tag:
                start_epoch = resume_state.get("epoch", 0) + 1
                if resume_state.get("optimizer_state_dict"):
                    optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                if resume_state.get("scaler_state_dict"):
                    scaler.load_state_dict(resume_state["scaler_state_dict"])
                print(f"[Resume] {stage1_tag} from epoch {start_epoch}")

            for epoch in range(start_epoch, stage1_epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    scaler,
                    ablation=args.ablation,
                )
                val_loss, val_acc, val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    ablation=args.ablation,
                )
                metric_str = " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items() if v == v])
                print(f"[Stage1] Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} {metric_str}")

                metrics_log.append(
                    {
                        "stage": stage1_tag,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_metrics": val_metrics,
                    }
                )

                save_train_state(stage1_tag, epoch, optimizer, scaler)

                current_val = val_metrics.get("top1_acc", val_acc)
                if best_val is None or current_val > best_val + args.early_stop_min_delta:
                    best_val = current_val
                    no_improve = 0
                    best_model_path = os.path.join(base, f"time_slice_model_best_{stage1_tag}.pth")
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "best_metric": current_val,
                            "stage": stage1_tag,
                            "epoch": epoch,
                        },
                        best_model_path,
                    )
                    print(f"[{stage1_tag}] Saved new best checkpoint to {best_model_path} ({current_val:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= args.early_stop_patience:
                        print(f"[{stage1_tag}] Early stop at epoch {epoch} (no improvement {no_improve} epochs)")
                        break

            save_checkpoint(stage1_tag)
        else:
            print(f"[Resume] Skip Stage1 because resume stage is {resume_stage}")

        if run_stage2:
            model.set_training_stage(2)
            print("[Stage2] Training adapter pool")
            stage2_epochs = args.stage2_epochs

            for param in model.parameters():
                param.requires_grad = False

            start_adapter = 0
            start_adapter_epoch = 1
            if resume_state and resume_stage == "stage2":
                start_adapter = resume_state.get("adapter_idx", 0)
                start_adapter_epoch = resume_state.get("adapter_epoch", 0) + 1
                print(f"[Resume] Stage2 from adapter {start_adapter} epoch {start_adapter_epoch}")

            for cls_idx in range(start_adapter, len(label_to_id)):
                print(f"[Stage2] Adapter {cls_idx + 1}/{len(label_to_id)}")
                for param in model.adapter_pool[cls_idx].parameters():
                    param.requires_grad = True

                optimizer = torch.optim.AdamW(model.adapter_pool[cls_idx].parameters(), lr=args.lr_stage2, weight_decay=1e-2)

                if resume_state and resume_stage == "stage2" and cls_idx == start_adapter:
                    epoch_start = start_adapter_epoch
                    if resume_state.get("optimizer_state_dict"):
                        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                    if resume_state.get("scaler_state_dict"):
                        scaler.load_state_dict(resume_state["scaler_state_dict"])
                else:
                    epoch_start = 1

                best_loss = None
                no_improve = 0
                for epoch in range(epoch_start, stage2_epochs + 1):
                    train_loss, train_acc = train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        scaler,
                        filter_labels=[cls_idx],
                        adapter_index=cls_idx,
                        ablation=args.ablation,
                    )

                    if train_loss > 0:
                        print(f"[Adapter {cls_idx}] Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.4f}")
                        metrics_log.append(
                            {
                                "stage": "stage2",
                                "adapter": cls_idx,
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "train_acc": train_acc,
                            }
                        )

                    save_train_state("stage2", epoch, optimizer, scaler, adapter_idx=cls_idx, adapter_epoch=epoch)

                    if best_loss is None or train_loss < best_loss - args.early_stop_min_delta:
                        best_loss = train_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= args.early_stop_patience:
                            print(f"[Stage2] Adapter {cls_idx} early stop at epoch {epoch}")
                            break

                for param in model.adapter_pool[cls_idx].parameters():
                    param.requires_grad = False

            save_checkpoint("stage2")
        else:
            print(f"[Resume] Skip Stage2 because resume stage is {resume_stage}")

        if run_stage3:
            model.set_training_stage(3)
            print("[Stage3] Training gating network")
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage3, weight_decay=1e-2)

            def kl_divergence_loss_per_sample(logits, targets, num_classes):
                target_dist = F.one_hot(targets, num_classes).float()
                target_dist = target_dist * 0.9 + 0.1 / num_classes
                pred_dist = F.log_softmax(logits, dim=-1)
                return F.kl_div(pred_dist, target_dist, reduction="none").sum(dim=1)

            def gate_weight_kl_loss_per_sample(gate_weights, targets, num_classes):
                target_dist = F.one_hot(targets, num_classes).float()
                target_dist = target_dist * 0.9 + 0.1 / num_classes
                return F.kl_div(torch.log(gate_weights + 1e-12), target_dist, reduction="none").sum(dim=1)

            def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                w = weights / (weights.mean() + 1e-12)
                return (values * w).mean()

            def weighted_bce_loss(logits_pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                loss = F.binary_cross_entropy_with_logits(logits_pred, target, reduction="none")
                return weighted_mean(loss, weights)

            stage3_epochs = args.stage3_epochs
            start_epoch = 1
            best_loss = None
            no_improve = 0

            if resume_state and resume_stage == "stage3":
                start_epoch = resume_state.get("epoch", 0) + 1
                if resume_state.get("optimizer_state_dict"):
                    optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                if resume_state.get("scaler_state_dict"):
                    scaler.load_state_dict(resume_state["scaler_state_dict"])
                print(f"[Resume] Stage3 from epoch {start_epoch}")

            for epoch in range(start_epoch, stage3_epochs + 1):
                model.train()
                use_amp = device.type == "cuda"
                total_loss = 0.0
                total_samples = 0

                for batch in train_loader:
                    x_explicit = batch["x_explicit"].to(device)
                    x_implicit = batch["x_implicit"].to(device)
                    x_cross_time = batch["x_cross_time"].to(device)
                    src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                    labels = batch["labels"].to(device)
                    batch_user_ids = batch.get("user_ids", [])
                    batch_window_indices = batch.get("window_indices", [])

                    x_explicit, x_implicit, x_cross_time, src_key_padding_mask = apply_input_ablation(
                        args.ablation,
                        x_explicit,
                        x_implicit,
                        x_cross_time,
                        src_key_padding_mask,
                    )

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits, gate_weights = model(
                            x_explicit,
                            x_implicit,
                            x_cross_time,
                            src_key_padding_mask,
                            return_gate_weights=True,
                        )
                        _assert_finite_tensor("stage3_logits", logits)
                        _assert_finite_tensor("stage3_gate_weights", gate_weights)

                        logits_base = _forward_with_adapter_flag(
                            model,
                            False,
                            x_explicit,
                            x_implicit,
                            x_cross_time,
                            src_key_padding_mask,
                        )
                        _assert_finite_tensor("stage3_base_logits", logits_base)

                        base_ce = F.cross_entropy(logits_base, labels, reduction="none")
                        gate_ce = F.cross_entropy(logits, labels, reduction="none")
                        gain_target = (base_ce - gate_ce > args.stage3_gain_margin).float()

                        gate_conf = gate_weights.max(dim=1).values.clamp(min=1e-6, max=1 - 1e-6)
                        gate_conf_logits = torch.log(gate_conf) - torch.log1p(-gate_conf)

                        switch_w = torch.ones_like(gain_target)
                        if (
                            args.stage3_switch_weight > 0
                            and len(batch_user_ids) == labels.size(0)
                            and len(batch_window_indices) == labels.size(0)
                        ):
                            switch_flags = [
                                train_switch_flag_map.get((str(uid), int(widx)), 0.0)
                                for uid, widx in zip(batch_user_ids, batch_window_indices)
                            ]
                            switch_flags_t = torch.tensor(switch_flags, dtype=gain_target.dtype, device=gain_target.device)
                            switch_w = 1.0 + args.stage3_switch_weight * switch_flags_t

                        sample_w = class_weights[labels]
                        gain_loss = weighted_bce_loss(gate_conf_logits, gain_target, sample_w * switch_w)

                        kl_main = kl_divergence_loss_per_sample(logits, labels, len(label_to_id))
                        loss = weighted_mean(kl_main, switch_w)

                        if args.gate_kl_weight > 0:
                            kl_gate = gate_weight_kl_loss_per_sample(gate_weights, labels, len(label_to_id))
                            loss = loss + args.gate_kl_weight * weighted_mean(kl_gate, switch_w)

                        if args.stage3_gain_weight > 0:
                            loss = loss + args.stage3_gain_weight * gain_loss

                        _assert_finite_tensor("stage3_loss", loss)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)

                stage3_loss = total_loss / max(1, total_samples)
                print(f"[Stage3] Epoch {epoch}: loss={stage3_loss:.4f}")
                metrics_log.append({"stage": "stage3", "epoch": epoch, "train_loss": stage3_loss})
                save_train_state("stage3", epoch, optimizer, scaler)

                if best_loss is None or stage3_loss < best_loss - args.early_stop_min_delta:
                    best_loss = stage3_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= args.early_stop_patience:
                        print(f"[Stage3] Early stop at epoch {epoch}")
                        break

            save_checkpoint("stage3")
        else:
            print(f"[Resume] Skip Stage3 because resume stage is {resume_stage}")

    if args.ablation == "no_adapter":
        model.use_adapter = False
        model.use_uniform_adapter_fusion = False
    elif args.ablation == "no_gating":
        model.use_adapter = True
        model.use_uniform_adapter_fusion = True

    skip_kl_for_ablation = args.ablation in ("no_adapter", "no_gating")
    if skip_kl_for_ablation:
        print(f"[Ablation] Skip KL-gated inference for ablation={args.ablation}")

    sweep_thresholds = []
    if args.kl_thresholds and not skip_kl_for_ablation:
        sweep_thresholds = [float(x) for x in args.kl_thresholds.split(",") if x.strip()]
    elif not skip_kl_for_ablation:
        t = args.kl_sweep_start
        while t <= args.kl_sweep_end + 1e-12:
            sweep_thresholds.append(round(t, 6))
            t += args.kl_sweep_step

    best_kl_threshold = None
    if sweep_thresholds:
        print(f"[KL Sweep] thresholds={sweep_thresholds}")

        _, _, val_base_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_kl_adapter=False,
            kl_threshold=args.kl_threshold,
            gate_mode=args.gate_mode,
            kl_soft_scale=args.kl_soft_scale,
            gate_min_alpha=args.gate_min_alpha,
            gate_max_alpha=args.gate_max_alpha,
            logit_temperature=args.gate_logit_temperature,
            ablation=args.ablation,
        )

        base_auc = val_base_metrics.get("roc_auc_ovr")
        base_top1 = val_base_metrics.get("top1_acc", val_base_metrics.get("accuracy", 0.0))
        base_nonswitch_top1 = val_base_metrics.get("non_switch_window_top1", 0.0)

        best_score = None
        best_metrics = None

        for threshold in sweep_thresholds:
            _, _, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_kl_adapter=True,
                kl_threshold=threshold,
                gate_mode=args.gate_mode,
                kl_soft_scale=args.kl_soft_scale,
                gate_min_alpha=args.gate_min_alpha,
                gate_max_alpha=args.gate_max_alpha,
                logit_temperature=args.gate_logit_temperature,
                ablation=args.ablation,
            )

            if args.kl_selection_mode == "multi":
                top1 = val_metrics.get("top1_acc", val_metrics.get("accuracy", 0.0))
                rec = val_metrics.get("recall_macro", 0.0)
                top3 = val_metrics.get("top3_acc", 0.0)
                switch_f1 = val_metrics.get("switch_f1", 0.0)
                switch_top1 = val_metrics.get("switch_window_top1", 0.0)
                switch_top3 = val_metrics.get("switch_window_top3", 0.0)

                score = float(
                    top1
                    + args.kl_score_recall_weight * rec
                    + args.kl_score_top3_weight * top3
                    + args.kl_score_switch_weight * switch_f1
                    + args.kl_score_switch_window_top1_weight * switch_top1
                    + args.kl_score_switch_window_top3_weight * switch_top3
                )
            else:
                score = val_metrics.get(args.kl_metric)
                if score is None or score != score:
                    score = val_metrics.get("accuracy")

            pass_auc_constraint = True
            auc = val_metrics.get("roc_auc_ovr")
            if base_auc is not None and base_auc == base_auc and auc is not None and auc == auc:
                pass_auc_constraint = (base_auc - auc) <= args.kl_max_auc_drop

            top1_now = val_metrics.get("top1_acc", val_metrics.get("accuracy", 0.0))
            nonswitch_top1_now = val_metrics.get("non_switch_window_top1", 0.0)
            pass_top1_constraint = (base_top1 - top1_now) <= args.kl_max_top1_drop
            pass_nonswitch_constraint = (base_nonswitch_top1 - nonswitch_top1_now) <= args.kl_max_nonswitch_top1_drop

            if (
                pass_auc_constraint
                and pass_top1_constraint
                and pass_nonswitch_constraint
                and (best_score is None or (score is not None and score > best_score))
            ):
                best_score = score
                best_kl_threshold = threshold
                best_metrics = val_metrics

            metrics_log.append(
                {
                    "stage": "kl_threshold_search",
                    "threshold": threshold,
                    "metric_key": ("multi_score" if args.kl_selection_mode == "multi" else args.kl_metric),
                    "metric_value": score,
                    "pass_auc_constraint": pass_auc_constraint,
                    "pass_top1_constraint": pass_top1_constraint,
                    "pass_nonswitch_constraint": pass_nonswitch_constraint,
                    "val_metrics": val_metrics,
                }
            )

        if best_kl_threshold is not None:
            metric_name = ("multi_score" if args.kl_selection_mode == "multi" else args.kl_metric)
            print(f"[KL Sweep] best_threshold={best_kl_threshold} best_{metric_name}={best_score}")
            metrics_log.append(
                {
                    "stage": "kl_threshold_best",
                    "threshold": best_kl_threshold,
                    "metric_key": metric_name,
                    "metric_value": best_score,
                    "val_metrics": best_metrics,
                }
            )

    use_kl_adapter_infer = (not skip_kl_for_ablation) and (args.use_kl_adapter or best_kl_threshold is not None)
    kl_threshold = best_kl_threshold if best_kl_threshold is not None else args.kl_threshold

    test_loss, test_acc, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_kl_adapter=use_kl_adapter_infer,
        kl_threshold=kl_threshold,
        gate_mode=args.gate_mode,
        kl_soft_scale=args.kl_soft_scale,
        gate_min_alpha=args.gate_min_alpha,
        gate_max_alpha=args.gate_max_alpha,
        logit_temperature=args.gate_logit_temperature,
        ablation=args.ablation,
    )
    metric_str = " ".join([f"{k}={v:.4f}" for k, v in test_metrics.items() if v == v])
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} {metric_str}")

    metrics_log.append(
        {
            "stage": "test",
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_metrics": test_metrics,
        }
    )

    if not args.infer_only:
        label_map_path = os.path.join(base, "time_slice_label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_to_id, f, ensure_ascii=False, indent=2)
        print(f"Saved label map to {label_map_path}")

    preds = predict_probabilities(
        model,
        test_loader,
        device,
        label_to_id,
        use_kl_adapter=use_kl_adapter_infer,
        kl_threshold=kl_threshold,
        gate_mode=args.gate_mode,
        kl_soft_scale=args.kl_soft_scale,
        gate_min_alpha=args.gate_min_alpha,
        gate_max_alpha=args.gate_max_alpha,
        logit_temperature=args.gate_logit_temperature,
        ablation=args.ablation,
    )
    output_path = os.path.join(base, "time_slice_predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {output_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
