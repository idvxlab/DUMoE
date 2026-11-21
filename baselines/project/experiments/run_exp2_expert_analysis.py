#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2：（Adapter） / 。

 PreferenceModel  gating_network + adapter_pool ：
  1) （gate  / argmax ）
  2) （argmax  vs ）
  3)  H(w) = -Σ_i w_i log w_i
  4) （/、/、= / ≠）

Pass ``--checkpoint`` to a trained PreferenceModel ``.pth`` (thresholded gate, ``return_gate_info=True``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

CUR_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CUR_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent
DUMOE_DIR = REPO_ROOT / "dumoe"
PARENT_DIR = PROJECT_DIR.parent
DYNAMIC_DIR = DUMOE_DIR

for p in [str(REPO_ROOT), str(PARENT_DIR), str(DUMOE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from downstream.windows_io import read_samples_csv  # type: ignore
from model import PreferenceModel  # type: ignore
from time_slice_data_loader import load_user_features_from_pt  # type: ignore
from time_slice_msg_feature_loader import load_msg_features_from_compact_shards  # type: ignore

TIME_FMT = "%a %b %d %H:%M:%S +0000 %Y"


def _load_csv_path_by_user(labels_cache: str) -> Dict[str, str]:
    payload = json.loads(Path(labels_cache).read_text(encoding="utf-8"))
    mapping = payload.get("csv_path_by_user", {})
    return {str(k): str(v) for k, v in mapping.items() if str(v).strip()}


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
                pub = str(norm.get("publish_time", "") or "").strip()
                if not tid or not pub:
                    continue
                try:
                    dt = datetime.strptime(pub, TIME_FMT)
                except Exception:
                    continue
                rows.append((dt, tid))
    except Exception:
        return []
    rows.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in rows]


def _build_user_tweet_ids_map(csv_path_by_user: Dict[str, str], users: Set[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for uid in users:
        p = csv_path_by_user.get(uid)
        if not p:
            continue
        tids = _load_ordered_tweet_ids_from_csv(p)
        if tids:
            out[uid] = tids
    return out


def _pad_vec(v: torch.Tensor, dim: int) -> torch.Tensor:
    x = v.float().view(-1)
    if x.numel() < dim:
        x = torch.cat([x, torch.zeros((dim - x.numel(),), dtype=x.dtype)], dim=0)
    return x[:dim]


def _gather_seq(
    msg_ids: Sequence[str],
    msg_feats: Dict[str, torch.Tensor],
    pad_to: int,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = torch.zeros((pad_to, dim), dtype=torch.float32)
    mask = torch.ones((pad_to,), dtype=torch.bool)
    j = 0
    for mid in msg_ids[:pad_to]:
        t = msg_feats.get(str(mid))
        if t is None:
            continue
        feats[j] = t.float().view(-1)[:dim]
        mask[j] = False
        j += 1
    if j == 0 and pad_to > 0:
        mask[0] = False
    return feats, mask


def _load_label_names(path: str) -> Dict[int, str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(v): str(k) for k, v in obj.items()}


def _entropy(w: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(w, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    return float(-(p * np.log(p + eps)).sum())


def _save_confusion_png(
    cm: np.ndarray,
    class_names: List[str],
    out_png: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Warn] matplotlib unavailable, skip png: {e}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Argmax expert (adapter)",
        title="Expert routing confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Exp2: expert/adapter interpretability analysis")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PreferenceModel checkpoint (.pth)",
    )
    p.add_argument(
        "--samples-csv",
        type=str,
        default="dataset/ids/windows.csv",
    )
    p.add_argument(
        "--labels-cache",
        type=str,
        default="",
        help="Optional labels_cache.json (csv_path_by_user). Empty = skip CSV timeline load.",
    )
    p.add_argument(
        "--label-map",
        type=str,
        default="dataset/labels/interest_categories.json",
    )
    p.add_argument(
        "--user-profile-pt",
        type=str,
        default="dataset/local/user_profile.pt",
    )
    p.add_argument(
        "--tweet-emb-shards-dir",
        type=str,
        default="dataset/local/embeddings_pt",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp2_expert_analysis",
    )
    p.add_argument("--restrict-split", type=str, default="test", choices=["", "train", "val", "test"])
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--short-recent", type=int, default=30)
    p.add_argument("--long-recent", type=int, default=300)
    p.add_argument("--mode", type=str, default="thresholded", choices=["base", "adapter", "thresholded"])
    p.add_argument("--kl-threshold", type=float, default=0.012)
    p.add_argument("--gate-mode", type=str, default="soft", choices=["soft", "hard"])
    p.add_argument("--kl-soft-scale", type=float, default=0.01)
    p.add_argument("--gate-min-alpha", type=float, default=0.05)
    p.add_argument("--gate-max-alpha", type=float, default=0.80)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="debug: only first N samples")
    p.add_argument("--num-qualitative", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2name = _load_label_names(args.label_map)
    num_classes = max(id2name.keys()) + 1 if id2name else 15
    class_names = [id2name.get(i, str(i)) for i in range(num_classes)]

    entries = read_samples_csv(args.samples_csv)
    if str(args.restrict_split):
        want = str(args.restrict_split)
        entries = [e for e in entries if str(e.split_tag).strip().lower() == want]
    if int(args.limit) > 0:
        entries = entries[: int(args.limit)]
    users = {e.user_id for e in entries}
    print(f"[Exp2] entries={len(entries)} users={len(users)} split={args.restrict_split!r}", flush=True)

    # Prefer cached tweet-id map if available
    cache_pt = PROJECT_DIR / "cache" / "user_tweet_ids_map__53e0ae86b7538dd8.pt"
    if cache_pt.exists():
        user_tweet_ids_map = torch.load(str(cache_pt), map_location="cpu", weights_only=False)
        print(f"[Exp2] load tweet-id cache: {cache_pt}", flush=True)
    else:
        csv_map = _load_csv_path_by_user(args.labels_cache)
        user_tweet_ids_map = _build_user_tweet_ids_map(csv_map, users)
        print(f"[Exp2] built tweet-id map users={len(user_tweet_ids_map)}", flush=True)

    user_feat_map, user_feat_dim = load_user_features_from_pt(args.user_profile_pt)
    needed: Set[str] = set()
    for e in entries:
        ordered = user_tweet_ids_map.get(e.user_id, [])
        start = int(e.window_idx) * int(args.window_size)
        for tid in ordered[start : start + int(args.long_recent)]:
            needed.add(str(tid))
        for tid in ordered[start + int(args.short_recent) : start + int(args.short_recent) + int(args.long_recent)]:
            needed.add(str(tid))
    print(f"[Exp2] needed_msg_ids={len(needed)}", flush=True)
    msg_feats, msg_feat_dim = load_msg_features_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir, needed_msg_ids=needed, verbose=True
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = PreferenceModel(
        user_feat_dim=int(cfg.get("user_feat_dim", user_feat_dim)),
        msg_feat_dim=int(cfg.get("msg_feat_dim", msg_feat_dim)),
        num_classes=int(cfg.get("num_classes", num_classes)),
        hidden_dim=256,
        lstm_layers=1,
        transformer_layers=2,
        nhead=8,
        dropout=0.4,
        adapter_bottleneck=int(cfg.get("adapter_bottleneck", 64)),
        use_adapter=True,
        topk_adapters=int(cfg.get("topk_adapters", 3)),
    )
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    for p_ in model.parameters():
        p_.requires_grad = False
    device = torch.device(
        args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    model = model.to(device).eval()
    print(f"[Exp2] model on {device}, n_adapters={len(model.adapter_pool)}", flush=True)

    rows: List[Dict] = []
    gate_sum = np.zeros((num_classes,), dtype=np.float64)
    expert_counts = Counter()
    label_counts = Counter()
    entropies: List[float] = []
    activated_flags: List[int] = []
    kls: List[float] = []
    alphas: List[float] = []
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for i, e in enumerate(entries, start=1):
            ordered = user_tweet_ids_map.get(e.user_id, [])
            start = int(e.window_idx) * int(args.window_size)
            short_ids = ordered[start : start + int(args.short_recent)]
            long_ids = ordered[start : start + int(args.long_recent)]
            long_nm_ids = ordered[
                start + int(args.short_recent) : start + int(args.short_recent) + int(args.long_recent)
            ]

            ufeat = user_feat_map.get(e.user_id, torch.zeros((int(user_feat_dim),), dtype=torch.float32))
            x_explicit = _pad_vec(ufeat, int(user_feat_dim)).unsqueeze(0).to(device)
            x_imp, _ = _gather_seq(short_ids, msg_feats, int(args.short_recent), int(msg_feat_dim))
            x_long, m_long = _gather_seq(long_ids, msg_feats, int(args.long_recent), int(msg_feat_dim))
            x_long_nm, m_long_nm = _gather_seq(long_nm_ids, msg_feats, int(args.long_recent), int(msg_feat_dim))
            x_imp_nm = torch.zeros_like(x_imp)

            _rep, info = model.encode_user_repr(
                x_explicit=x_explicit,
                x_implicit=x_imp.unsqueeze(0).to(device),
                x_cross_time=x_long.unsqueeze(1).to(device),
                src_key_padding_mask=m_long.unsqueeze(0).to(device),
                x_implicit_no_month=x_imp_nm.unsqueeze(0).to(device),
                x_cross_time_no_month=x_long_nm.unsqueeze(1).to(device),
                src_key_padding_mask_no_month=m_long_nm.unsqueeze(0).to(device),
                mode=args.mode,
                kl_threshold=float(args.kl_threshold),
                gate_mode=args.gate_mode,
                kl_soft_scale=float(args.kl_soft_scale),
                gate_min_alpha=float(args.gate_min_alpha),
                gate_max_alpha=float(args.gate_max_alpha),
                return_gate_info=True,
            )

            y = int(e.label_id)
            gw = info.get("gate_weights")
            if gw is None:
                # base mode: no experts
                w = np.zeros((num_classes,), dtype=np.float64)
                expert = -1
            else:
                w = gw.detach().float().cpu().numpy().reshape(-1)
                if w.size < num_classes:
                    ww = np.zeros((num_classes,), dtype=np.float64)
                    ww[: w.size] = w
                    w = ww
                else:
                    w = w[:num_classes]
                expert = int(np.argmax(w))

            ent = _entropy(w) if expert >= 0 else 0.0
            kl = float(info["kl"].reshape(-1)[0].item()) if "kl" in info else float("nan")
            act = int(bool(info["activate_mask"].reshape(-1)[0].item())) if "activate_mask" in info else 0
            alpha = float(info["alpha"].reshape(-1)[0].item()) if "alpha" in info else float("nan")

            gate_sum += w
            if expert >= 0:
                expert_counts[expert] += 1
                if 0 <= y < num_classes:
                    cm[y, expert] += 1
            label_counts[y] += 1
            entropies.append(ent)
            activated_flags.append(act)
            kls.append(kl)
            alphas.append(alpha)

            rows.append(
                {
                    "user_id_t": e.user_id_t,
                    "user_id": e.user_id,
                    "window_idx": int(e.window_idx),
                    "label_id": y,
                    "label_name": id2name.get(y, str(y)),
                    "expert_id": expert,
                    "expert_name": id2name.get(expert, str(expert)) if expert >= 0 else "none",
                    "entropy": ent,
                    "kl": kl,
                    "activated": act,
                    "alpha": alpha,
                    "gate_weights": w.tolist(),
                    "top3_experts": [
                        {
                            "expert_id": int(j),
                            "expert_name": id2name.get(int(j), str(j)),
                            "weight": float(w[j]),
                        }
                        for j in np.argsort(-w)[:3]
                    ]
                    if expert >= 0
                    else [],
                }
            )
            if i % 500 == 0 or i == len(entries):
                print(f"[Exp2] processed {i}/{len(entries)}", flush=True)

    n = max(1, len(rows))
    mean_gate = (gate_sum / float(n)).tolist()
    usage_freq = {str(k): int(v) for k, v in sorted(expert_counts.items())}
    usage_ratio = {str(k): float(v) / float(n) for k, v in sorted(expert_counts.items())}

    # qualitative examples
    nq = int(args.num_qualitative)
    by_ent = sorted(rows, key=lambda r: r["entropy"], reverse=True)
    high_ent = by_ent[:nq]
    low_ent = list(reversed(by_ent[-nq:])) if len(by_ent) >= nq else list(reversed(by_ent))
    activated = [r for r in rows if int(r["activated"]) == 1]
    not_act = [r for r in rows if int(r["activated"]) == 0]
    match = [r for r in rows if int(r["expert_id"]) == int(r["label_id"]) and int(r["expert_id"]) >= 0]
    mismatch = [r for r in rows if int(r["expert_id"]) != int(r["label_id"]) and int(r["expert_id"]) >= 0]

    def _pick(xs: List[Dict], k: int) -> List[Dict]:
        return xs[:k]

    qualitative = {
        "high_entropy": _pick(high_ent, nq),
        "low_entropy": _pick(low_ent, nq),
        "activated": _pick(sorted(activated, key=lambda r: -float(r.get("kl", 0.0) or 0.0)), nq),
        "not_activated": _pick(sorted(not_act, key=lambda r: float(r.get("kl", 0.0) or 0.0)), nq),
        "expert_equals_label": _pick(sorted(match, key=lambda r: -max(r["gate_weights"] or [0.0])), nq),
        "expert_not_label": _pick(sorted(mismatch, key=lambda r: -max(r["gate_weights"] or [0.0])), nq),
    }

    # agreement
    agree = float(np.trace(cm) / max(1, cm.sum()))

    summary = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "gate_mode": args.gate_mode,
        "kl_threshold": float(args.kl_threshold),
        "num_samples": len(rows),
        "num_classes": num_classes,
        "class_names": class_names,
        "expert_usage": {
            "mean_gate_weights": mean_gate,
            "argmax_counts": usage_freq,
            "argmax_ratio": usage_ratio,
            "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        },
        "routing_entropy": {
            "mean": float(np.mean(entropies)) if entropies else 0.0,
            "std": float(np.std(entropies)) if entropies else 0.0,
            "max_possible": float(math.log(max(2, num_classes))),
            "normalized_mean": float(np.mean(entropies) / max(1e-6, math.log(max(2, num_classes))))
            if entropies
            else 0.0,
        },
        "threshold_gating": {
            "activated_ratio": float(np.mean(activated_flags)) if activated_flags else 0.0,
            "kl_mean": float(np.nanmean(kls)) if kls else float("nan"),
            "kl_std": float(np.nanstd(kls)) if kls else float("nan"),
            "alpha_mean": float(np.nanmean(alphas)) if alphas else float("nan"),
            "alpha_std": float(np.nanstd(alphas)) if alphas else float("nan"),
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "agreement_diag_ratio": agree,
            "note": "rows=true label, cols=argmax expert",
        },
        "qualitative_examples": {
            k: [
                {
                    "user_id_t": r["user_id_t"],
                    "label": r["label_name"],
                    "expert": r["expert_name"],
                    "entropy": r["entropy"],
                    "kl": r["kl"],
                    "activated": r["activated"],
                    "alpha": r["alpha"],
                    "top3": r["top3_experts"],
                }
                for r in vs
            ]
            for k, vs in qualitative.items()
        },
    }

    (out_dir / "expert_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # per-sample csv (compact)
    with open(out_dir / "per_sample_routing.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "user_id_t",
                "user_id",
                "window_idx",
                "label_id",
                "label_name",
                "expert_id",
                "expert_name",
                "entropy",
                "kl",
                "activated",
                "alpha",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    np.save(out_dir / "confusion_matrix.npy", cm)
    _save_confusion_png(cm, class_names, out_dir / "confusion_matrix.png")

    # usage bar data
    usage_rows = [
        {
            "expert_id": i,
            "expert_name": class_names[i],
            "mean_weight": float(mean_gate[i]),
            "argmax_count": int(expert_counts.get(i, 0)),
            "argmax_ratio": float(expert_counts.get(i, 0)) / float(n),
        }
        for i in range(num_classes)
    ]
    (out_dir / "expert_usage.json").write_text(
        json.dumps(usage_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[OK] Exp2 done: n={len(rows)} entropy_mean={summary['routing_entropy']['mean']:.4f} "
        f"activated_ratio={summary['threshold_gating']['activated_ratio']:.4f} "
        f"expert=label_agree={agree:.4f}",
        flush=True,
    )
    print(f"[OK] outputs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
