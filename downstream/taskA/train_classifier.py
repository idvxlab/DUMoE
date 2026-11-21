#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from heads import ClassifierHead


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")
        if labels.dim() != 1:
            labels = labels.view(-1)
        if embeddings.size(0) != labels.size(0):
            raise ValueError("embeddings and labels size mismatch")
        self.embeddings = embeddings.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


def split_tag_from_window(window_idx: int, train_months: int = 8, val_months: int = 2, test_months: int = 2) -> str:
    if window_idx < train_months:
        return "train"
    if window_idx < train_months + val_months:
        return "val"
    if window_idx < train_months + val_months + test_months:
        return "test"
    return "other"


def _subset_payload_by_indices(payload: Dict, idxs: List[int]) -> Dict:
    embs = payload["embeddings"].index_select(0, torch.tensor(idxs, dtype=torch.long)) if idxs else torch.zeros((0, payload["embeddings"].size(1)), dtype=payload["embeddings"].dtype)
    labels = payload["labels"].index_select(0, torch.tensor(idxs, dtype=torch.long)) if idxs else torch.zeros((0,), dtype=torch.long)
    return {
        "embeddings": embs,
        "labels": labels,
        "user_ids": [payload.get("user_ids", [])[i] for i in idxs] if payload.get("user_ids") else [],
        "window_indices": [int(payload.get("window_indices", [])[i]) for i in idxs] if payload.get("window_indices") else [],
        "months": [payload.get("months", [])[i] for i in idxs] if payload.get("months") else [],
        "meta": payload.get("meta", {}),
    }


def _load_payload(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "embeddings" not in payload or "labels" not in payload:
        raise ValueError(f"invalid payload format: {path}")
    return payload


def load_split_payloads(
    all_embeddings_pt: Optional[str],
    train_embeddings_pt: Optional[str],
    val_embeddings_pt: Optional[str],
    test_embeddings_pt: Optional[str],
) -> Tuple[Dict, Dict, Dict]:
    if train_embeddings_pt and val_embeddings_pt and test_embeddings_pt:
        return _load_payload(train_embeddings_pt), _load_payload(val_embeddings_pt), _load_payload(test_embeddings_pt)

    if not all_embeddings_pt:
        raise ValueError("Either split pt files or all_embeddings_pt must be provided")

    all_payload = _load_payload(all_embeddings_pt)
    n = int(all_payload["labels"].size(0))

    split_tags = all_payload.get("split_tags")
    if isinstance(split_tags, list) and len(split_tags) == n:
        train_idxs = [i for i, t in enumerate(split_tags) if t == "train"]
        val_idxs = [i for i, t in enumerate(split_tags) if t == "val"]
        test_idxs = [i for i, t in enumerate(split_tags) if t == "test"]
    else:
        window_indices = [int(x) for x in all_payload.get("window_indices", [])]
        if len(window_indices) != n:
            raise ValueError("all payload has no valid split tags and no window_indices")
        train_idxs = [i for i, w in enumerate(window_indices) if split_tag_from_window(w) == "train"]
        val_idxs = [i for i, w in enumerate(window_indices) if split_tag_from_window(w) == "val"]
        test_idxs = [i for i, w in enumerate(window_indices) if split_tag_from_window(w) == "test"]

    return (
        _subset_payload_by_indices(all_payload, train_idxs),
        _subset_payload_by_indices(all_payload, val_idxs),
        _subset_payload_by_indices(all_payload, test_idxs),
    )


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float((preds == labels).float().mean().item())
    metrics["top1_acc"] = metrics["accuracy"]

    top3_k = min(3, probs.size(1))
    top3 = torch.topk(probs, k=top3_k, dim=1).indices
    metrics["top3_acc"] = float(top3.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item())

    recalls: List[float] = []
    for cls in range(probs.size(1)):
        mask = labels == cls
        if mask.sum().item() == 0:
            continue
        recalls.append(float((preds[mask] == cls).float().mean().item()))
    metrics["recall_macro"] = float(sum(recalls) / max(1, len(recalls)))

    return metrics


def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    if total == 0:
        return 0.0, {"accuracy": 0.0, "top1_acc": 0.0, "top3_acc": 0.0, "recall_macro": 0.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels)
    return total_loss / max(1, total), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task A classifier on frozen embeddings")
    parser.add_argument("--all-embeddings-pt", type=str, default="outputs/taskA_user_window_embeddings.pt")
    parser.add_argument("--train-embeddings-pt", type=str, default="")
    parser.add_argument("--val-embeddings-pt", type=str, default="")
    parser.add_argument("--test-embeddings-pt", type=str, default="")

    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--output-dir", type=str, default="./TASKA")
    parser.add_argument("--run-name", type=str, default="taskA_classifier_thresholded")

    args = parser.parse_args()

    set_seed(args.seed)

    train_payload, val_payload, test_payload = load_split_payloads(
        all_embeddings_pt=args.all_embeddings_pt,
        train_embeddings_pt=args.train_embeddings_pt or None,
        val_embeddings_pt=args.val_embeddings_pt or None,
        test_embeddings_pt=args.test_embeddings_pt or None,
    )

    d_in = int(train_payload["embeddings"].size(1)) if train_payload["embeddings"].dim() == 2 else 0
    if d_in <= 0:
        raise RuntimeError("invalid embedding dim")

    train_ds = EmbeddingDataset(train_payload["embeddings"], train_payload["labels"])
    val_ds = EmbeddingDataset(val_payload["embeddings"], val_payload["labels"])
    test_ds = EmbeddingDataset(test_payload["embeddings"], test_payload["labels"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=max(0, args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers))

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    model = ClassifierHead(
        d_in=d_in,
        num_classes=int(args.num_classes),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.run_name}_best.pth"
    metrics_path = out_dir / f"{args.run_name}_metrics.json"

    history: List[Dict] = []
    best_val = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)

        train_loss = total_loss / max(1, total)
        val_loss, val_metrics = run_eval(model, val_loader, criterion, device)

        top1 = float(val_metrics.get("top1_acc", val_metrics.get("accuracy", 0.0)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
        )

        print(
            f"[TaskA][Epoch {epoch}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} top1={top1:.6f} top3={val_metrics.get('top3_acc', 0.0):.6f}"
        )

        if best_val is None or top1 > best_val + args.early_stop_min_delta:
            best_val = top1
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "d_in": d_in,
                        "num_classes": int(args.num_classes),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                    },
                    "best_val_top1": best_val,
                    "best_epoch": best_epoch,
                },
                str(ckpt_path),
            )
        else:
            no_improve += 1
            if no_improve >= int(args.early_stop_patience):
                print(f"[TaskA] early stop at epoch {epoch}")
                break

    best_payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    model.to(device)

    test_loss, test_metrics = run_eval(model, test_loader, criterion, device)

    summary = {
        "run_name": args.run_name,
        "d_in": d_in,
        "num_classes": int(args.num_classes),
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val if best_val is not None else 0.0),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "history": history,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[TaskA] saved best classifier: {ckpt_path}")
    print(f"[TaskA] saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
