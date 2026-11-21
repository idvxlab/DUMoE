#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMOE_DIR = REPO_ROOT / "dumoe"
for _p in (str(REPO_ROOT), str(DUMOE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import PreferenceModel


def save_payload(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _slice_batch(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return t[start:end]


def _load_msg_table(meta: Dict, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor], int, int]:
    chunks_dir = str(meta.get("msg_feature_table_chunks_dir", "")).strip()
    chunk_rows = int(meta.get("msg_feature_table_chunk_rows", 0))
    feat_dim = int(meta.get("msg_feat_dim", 0))

    if chunks_dir:
        cdir = Path(chunks_dir)
        chunk_paths = sorted(cdir.glob("part_*.pt"))
        if not chunk_paths:
            raise RuntimeError(f"chunk dir configured but empty: {cdir}")
        if chunk_rows <= 0:
            raise RuntimeError("invalid msg_feature_table_chunk_rows in meta")

        print(f"[TaskA-CacheExport] loading msg feature table chunks from {cdir}")
        chunks: List[torch.Tensor] = []
        inferred_feat_dim = 0

        for p in chunk_paths:
            t = torch.load(str(p), map_location="cpu", weights_only=False)
            if not isinstance(t, torch.Tensor) or t.dim() != 2:
                raise RuntimeError(f"invalid chunk format: {p}")
            if inferred_feat_dim == 0:
                inferred_feat_dim = int(t.size(1))
            elif int(t.size(1)) != inferred_feat_dim:
                raise RuntimeError(f"chunk feat dim mismatch: {p} got {int(t.size(1))}, expect {inferred_feat_dim}")
            chunks.append(t.to(device, non_blocking=True))

        if feat_dim <= 0:
            feat_dim = inferred_feat_dim
        elif feat_dim != inferred_feat_dim:
            raise RuntimeError(f"meta msg_feat_dim={feat_dim} but chunk feat dim={inferred_feat_dim}")

        print(
            f"[TaskA-CacheExport] loaded {len(chunks)} chunks on {device}, "
            f"chunk_rows={chunk_rows}, feat_dim={feat_dim}"
        )
        return torch.empty(0, device=device), chunks, chunk_rows, feat_dim

    table_path_str = str(meta.get("msg_feature_table", "")).strip()
    if not table_path_str:
        raise RuntimeError("meta has neither msg_feature_table_chunks_dir nor msg_feature_table")

    table_path = Path(table_path_str)
    print(f"[TaskA-CacheExport] loading msg feature table: {table_path}")
    msg_table = torch.load(str(table_path), map_location="cpu", weights_only=False)
    if not isinstance(msg_table, torch.Tensor) or msg_table.dim() != 2:
        raise RuntimeError("msg feature table format invalid")

    msg_table = msg_table.to(device, non_blocking=True)
    feat_dim = int(msg_table.size(1))
    print(f"[TaskA-CacheExport] loaded single table on {device}, shape={tuple(msg_table.shape)}")
    return msg_table, [], 0, feat_dim


def _gather_from_chunks(
    chunks: List[torch.Tensor],
    chunk_rows: int,
    idx: torch.Tensor,
    feat_dim: int,
) -> torch.Tensor:
    out = torch.zeros((idx.size(0), idx.size(1), feat_dim), device=idx.device, dtype=torch.float32)
    if not chunks:
        return out

    chunk_ids = torch.div(idx, chunk_rows, rounding_mode="floor")
    local_idx = idx - chunk_ids * chunk_rows

    for cid, table_chunk in enumerate(chunks):
        mask = chunk_ids == cid
        if not mask.any():
            continue
        pos = mask.nonzero(as_tuple=False)
        rows = pos[:, 0]
        cols = pos[:, 1]
        vals = local_idx[rows, cols]
        out[rows, cols] = table_chunk[vals].float()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TaskA embeddings from prebuilt sample cache")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
    )
    parser.add_argument("--output-dir", type=str, default="./TASKA")
    parser.add_argument("--output-prefix", type=str, default="taskA_user_window_embeddings")
    parser.add_argument("--cache-dir", type=str, default="dataset/local/taskA_samples")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--mode", type=str, default="thresholded", choices=["base", "adapter", "thresholded"])
    parser.add_argument("--kl-threshold", type=float, default=0.012)
    parser.add_argument("--gate-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--kl-soft-scale", type=float, default=0.01)
    parser.add_argument("--gate-min-alpha", type=float, default=0.05)
    parser.add_argument("--gate-max-alpha", type=float, default=0.80)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every-shard", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"cache meta not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    shard_paths = sorted(cache_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"no shard files found under {cache_dir}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})

    model = PreferenceModel(
        user_feat_dim=int(cfg.get("user_feat_dim", int(meta.get("user_feat_dim", 1542)))),
        msg_feat_dim=int(cfg.get("msg_feat_dim", int(meta.get("msg_feat_dim", 1536)))),
        num_classes=int(cfg.get("num_classes", len(meta.get("label_to_id", {})) or 15)),
        hidden_dim=256,
        lstm_layers=1,
        transformer_layers=2,
        nhead=8,
        dropout=0.4,
        adapter_bottleneck=int(cfg.get("adapter_bottleneck", 64)),
        use_adapter=(str(checkpoint.get("stage", "")).lower() != "stage1"),
        topk_adapters=int(cfg.get("topk_adapters", 3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = model.to(device)
    model.eval()

    msg_table, msg_chunks, chunk_rows, feat_dim = _load_msg_table(meta, device)

    embeddings_chunks: List[torch.Tensor] = []
    labels_all: List[int] = []
    user_ids_all: List[str] = []
    window_indices_all: List[int] = []
    months_all: List[str] = []
    split_tags_all: List[str] = []

    total_samples = 0
    total_batches = 0

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(
        f"[TaskA-CacheExport] start export: shards={len(shard_paths)} "
        f"batch_size={args.batch_size} device={device} amp={use_amp}"
    )

    for shard_i, shard_path in enumerate(shard_paths, start=1):
        payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)

        x_explicit = payload["x_explicit"]
        short_idx = payload["short_msg_indices"]
        long_idx = payload["long_msg_indices"]
        long_no_month_idx = payload["long_no_month_msg_indices"]
        mask = payload["src_key_padding_mask"]
        mask_no_month = payload["src_key_padding_mask_no_month"]
        labels = payload["labels"]

        user_ids = payload["user_ids"]
        window_indices = payload["window_indices"]
        months = payload["months"]
        split_tags = payload["split_tags"]

        n = int(labels.size(0))

        for start in range(0, n, int(args.batch_size)):
            end = min(start + int(args.batch_size), n)

            b_x_explicit = _slice_batch(x_explicit, start, end).to(device, non_blocking=True).float()
            b_short_idx = _slice_batch(short_idx, start, end).to(device, non_blocking=True).long()
            b_long_idx = _slice_batch(long_idx, start, end).to(device, non_blocking=True).long()
            b_long_no_month_idx = _slice_batch(long_no_month_idx, start, end).to(device, non_blocking=True).long()
            b_mask = _slice_batch(mask, start, end).to(device, non_blocking=True)
            b_mask_no_month = _slice_batch(mask_no_month, start, end).to(device, non_blocking=True)

            if msg_chunks:
                b_x_implicit = _gather_from_chunks(msg_chunks, chunk_rows, b_short_idx, feat_dim)
                b_x_cross_time = _gather_from_chunks(msg_chunks, chunk_rows, b_long_idx, feat_dim).transpose(0, 1).contiguous()
                b_x_cross_time_no_month = _gather_from_chunks(msg_chunks, chunk_rows, b_long_no_month_idx, feat_dim).transpose(0, 1).contiguous()
            else:
                b_x_implicit = msg_table[b_short_idx].float()
                b_x_cross_time = msg_table[b_long_idx].float().transpose(0, 1).contiguous()
                b_x_cross_time_no_month = msg_table[b_long_no_month_idx].float().transpose(0, 1).contiguous()

            b_x_implicit_no_month = torch.zeros_like(b_x_implicit)

            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        repr_batch = model.encode_user_repr(
                            x_explicit=b_x_explicit,
                            x_implicit=b_x_implicit,
                            x_cross_time=b_x_cross_time,
                            src_key_padding_mask=b_mask,
                            x_implicit_no_month=b_x_implicit_no_month,
                            x_cross_time_no_month=b_x_cross_time_no_month,
                            src_key_padding_mask_no_month=b_mask_no_month,
                            mode=args.mode,
                            kl_threshold=args.kl_threshold,
                            gate_mode=args.gate_mode,
                            kl_soft_scale=args.kl_soft_scale,
                            gate_min_alpha=args.gate_min_alpha,
                            gate_max_alpha=args.gate_max_alpha,
                            return_gate_info=False,
                        )
                else:
                    repr_batch = model.encode_user_repr(
                        x_explicit=b_x_explicit,
                        x_implicit=b_x_implicit,
                        x_cross_time=b_x_cross_time,
                        src_key_padding_mask=b_mask,
                        x_implicit_no_month=b_x_implicit_no_month,
                        x_cross_time_no_month=b_x_cross_time_no_month,
                        src_key_padding_mask_no_month=b_mask_no_month,
                        mode=args.mode,
                        kl_threshold=args.kl_threshold,
                        gate_mode=args.gate_mode,
                        kl_soft_scale=args.kl_soft_scale,
                        gate_min_alpha=args.gate_min_alpha,
                        gate_max_alpha=args.gate_max_alpha,
                        return_gate_info=False,
                    )

            embeddings_chunks.append(repr_batch.detach().cpu())
            labels_all.extend(int(x) for x in labels[start:end].tolist())
            user_ids_all.extend(str(x) for x in user_ids[start:end])
            window_indices_all.extend(int(x) for x in window_indices[start:end])
            months_all.extend(str(x) for x in months[start:end])
            split_tags_all.extend(str(x) for x in split_tags[start:end])

            total_samples += (end - start)
            total_batches += 1
            if total_batches % 50 == 0:
                print(f"[TaskA-CacheExport] processed batches={total_batches} samples={total_samples}")

        print(f"[TaskA-CacheExport] finished shard {shard_i}/{len(shard_paths)}: {shard_path.name} samples={n}")

        if args.save_every_shard:
            tmp_path = out_dir / f"{args.output_prefix}_{args.mode}_partial_after_shard_{shard_i:05d}.pt"
            tmp_embeddings = torch.cat(embeddings_chunks, dim=0) if embeddings_chunks else torch.zeros((0, 256), dtype=torch.float32)
            tmp_labels = torch.tensor(labels_all, dtype=torch.long)
            save_payload(
                tmp_path,
                {
                    "embeddings": tmp_embeddings,
                    "labels": tmp_labels,
                    "user_ids": list(user_ids_all),
                    "window_indices": list(window_indices_all),
                    "months": list(months_all),
                    "split_tags": list(split_tags_all),
                },
            )
            print(f"[TaskA-CacheExport] saved partial: {tmp_path.name}")

    embeddings = torch.cat(embeddings_chunks, dim=0) if embeddings_chunks else torch.zeros((0, 256), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_all, dtype=torch.long)

    if embeddings.size(0) != labels_tensor.size(0):
        raise RuntimeError("embeddings/labels size mismatch")

    dim_out = int(embeddings.size(1)) if embeddings.dim() == 2 and embeddings.size(0) > 0 else 0
    print(f"[TaskA-CacheExport] export done: samples={embeddings.size(0)} dim={dim_out}")

    all_payload = {
        "embeddings": embeddings,
        "labels": labels_tensor,
        "user_ids": user_ids_all,
        "window_indices": window_indices_all,
        "months": months_all,
        "split_tags": split_tags_all,
        "meta": {
            "mode": args.mode,
            "kl_threshold": args.kl_threshold,
            "gate_mode": args.gate_mode,
            "kl_soft_scale": args.kl_soft_scale,
            "gate_min_alpha": args.gate_min_alpha,
            "gate_max_alpha": args.gate_max_alpha,
            "repr_dim": dim_out,
            "num_classes": int(cfg.get("num_classes", 15)),
            "checkpoint": args.checkpoint,
            "cache_dir": str(cache_dir),
            "device": str(device),
        },
    }

    all_path = out_dir / f"{args.output_prefix}_{args.mode}_all_windows.pt"
    save_payload(all_path, all_payload)

    split_paths = {}
    split_counts = {}
    for split_name in ["train", "val", "test"]:
        idxs = [i for i, t in enumerate(split_tags_all) if t == split_name]
        split_counts[split_name] = len(idxs)

        if len(idxs) == 0:
            split_embeddings = torch.zeros((0, dim_out), dtype=embeddings.dtype)
            split_labels = torch.zeros((0,), dtype=torch.long)
            split_user_ids: List[str] = []
            split_window_indices: List[int] = []
            split_months: List[str] = []
        else:
            index_tensor = torch.tensor(idxs, dtype=torch.long)
            split_embeddings = embeddings.index_select(0, index_tensor)
            split_labels = labels_tensor.index_select(0, index_tensor)
            split_user_ids = [user_ids_all[i] for i in idxs]
            split_window_indices = [window_indices_all[i] for i in idxs]
            split_months = [months_all[i] for i in idxs]

        split_payload = {
            "embeddings": split_embeddings,
            "labels": split_labels,
            "user_ids": split_user_ids,
            "window_indices": split_window_indices,
            "months": split_months,
            "meta": all_payload["meta"],
        }
        split_path = out_dir / f"{args.output_prefix}_{args.mode}_{split_name}.pt"
        save_payload(split_path, split_payload)
        split_paths[split_name] = str(split_path)

    stats = {
        "all_windows": int(embeddings.size(0)),
        "repr_dim": dim_out,
        "split_counts": split_counts,
        "all_path": str(all_path),
        "split_paths": split_paths,
        "users": len(set(user_ids_all)),
    }
    stats_path = out_dir / f"{args.output_prefix}_{args.mode}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[TaskA-CacheExport] saved all windows file: {all_path}")
    print(f"[TaskA-CacheExport] saved split files: {split_paths}")
    print(f"[TaskA-CacheExport] saved stats: {stats_path}")


if __name__ == "__main__":
    main()
