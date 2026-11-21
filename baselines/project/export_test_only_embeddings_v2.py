#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

CUR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CUR_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from project.data.processor import build_records_by_split, load_or_build_user_tweet_ids_map
from project.data.processor import load_msg_features_from_compact_shards, load_user_features_from_pt
from project.models.baselines import build_model
from project.trainer.unified_trainer import Trainer


@dataclass
class _Rec:
    key: str
    user_id: str
    window_idx: int


class TestOnlySeqEmbDataset(Dataset):
    def __init__(
        self,
        records: List[_Rec],
        user_tweet_ids_map: Dict[str, List[str]],
        msg_feats: Dict[str, torch.Tensor],
        msg_dim: int,
        user_feat_map: Dict[str, torch.Tensor],
        profile_dim: int,
        seq_len: int = 300,
        window_size: int = 30,
    ) -> None:
        self.records = records
        self.user_tweet_ids_map = user_tweet_ids_map
        self.msg_feats = msg_feats
        self.msg_dim = int(msg_dim)
        self.user_feat_map = user_feat_map
        self.profile_dim = int(profile_dim)
        self.seq_len = int(seq_len)
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        r = self.records[idx]
        ordered = self.user_tweet_ids_map.get(r.user_id, [])
        st = int(r.window_idx) * int(self.window_size)
        mids = ordered[st : st + int(self.seq_len)]
        kept = [mid for mid in mids if str(mid) in self.msg_feats]

        seq_emb = torch.zeros((self.seq_len, self.msg_dim), dtype=torch.float32)
        seq_mask = torch.ones((self.seq_len,), dtype=torch.bool)
        wi = 0
        for mid in kept:
            if wi >= self.seq_len:
                break
            v = self.msg_feats.get(str(mid))
            if v is None:
                continue
            seq_emb[wi] = v.float().view(-1)[: self.msg_dim]
            seq_mask[wi] = False
            wi += 1
        if (~seq_mask).sum() == 0 and self.seq_len > 0:
            seq_mask[0] = False

        profile = self.user_feat_map.get(r.user_id, torch.zeros((self.profile_dim,), dtype=torch.float32)).float()
        if profile.numel() < self.profile_dim:
            pad = torch.zeros((self.profile_dim - profile.numel(),), dtype=torch.float32)
            profile = torch.cat([profile.view(-1), pad], dim=0)
        else:
            profile = profile.view(-1)[: self.profile_dim]

        # Put seq_emb into "seq_ids" slot: model.lookup_seq_emb handles 3D float inputs.
        return {"key": r.key, "seq_ids": seq_emb, "seq_mask": seq_mask, "profile": profile, "label": torch.tensor(0)}


def _load_test_records(samples_csv: str) -> List[_Rec]:
    recs = build_records_by_split(samples_csv)["test"]
    out: List[_Rec] = []
    for r in recs:
        out.append(_Rec(key=str(r.key), user_id=str(r.user_id), window_idx=int(r.window_idx)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export test-only user embeddings from encoder checkpoints (no leakage).")
    p.add_argument("--samples-csv", type=str, default="dataset/ids/windows.csv")
    p.add_argument("--labels-cache", type=str, default="dataset/local/labels_cache.json")
    p.add_argument("--user-profile-pt", type=str, default="dataset/local/user_profile.pt")
    p.add_argument("--tweet-emb-shards-dir", type=str, default="dataset/local/embeddings_pt")
    p.add_argument("--preprocess-cache-dir", type=str, default="cache")
    p.add_argument("--outputs-full", type=str, default="outputs/baselines")
    p.add_argument("--output-root", type=str, default="outputs/test_only_v2")
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    clean_argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    args = p.parse_args(clean_argv)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    test_records = _load_test_records(args.samples_csv)
    users = {r.user_id for r in test_records}
    user_tweet_ids_map = load_or_build_user_tweet_ids_map(
        labels_cache=args.labels_cache, users=users, cache_dir=str(args.preprocess_cache_dir), verbose=True
    )

    needed_msg_ids: Set[str] = set()
    for r in test_records:
        ordered = user_tweet_ids_map.get(r.user_id, [])
        st = int(r.window_idx) * int(args.window_size)
        needed_msg_ids.update(str(x) for x in ordered[st : st + int(args.seq_len)])
    print(f"[Stage] load msg feats for test-only needed_ids={len(needed_msg_ids)}", flush=True)
    msg_feats, msg_dim = load_msg_features_from_compact_shards(
        shards_dir=args.tweet_emb_shards_dir, needed_msg_ids=needed_msg_ids, verbose=True
    )
    user_feat_map, profile_dim = load_user_features_from_pt(args.user_profile_pt)

    ds = TestOnlySeqEmbDataset(
        records=test_records,
        user_tweet_ids_map=user_tweet_ids_map,
        msg_feats=msg_feats,
        msg_dim=msg_dim,
        user_feat_map=user_feat_map,
        profile_dim=profile_dim,
        seq_len=int(args.seq_len),
        window_size=int(args.window_size),
    )

    nw = max(0, int(args.num_workers))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
    )
    print(f"[Stage] test-only dataloader ready: samples={len(ds)} batches={len(loader)}", flush=True)

    models = ["SASRec", "BERT4Rec", "MIND", "PeterRec", "PTUM", "UniSRec", "HORAE", "DisMIR", "HM4SR"]
    out_index: Dict[str, Dict] = {}

    for name in models:
        ckpt = Path(args.outputs_full) / name / "encoder_best.pth"
        if not ckpt.exists():
            print(f"[SKIP] {name}: missing ckpt {ckpt}", flush=True)
            continue
        dst_dir = out_root / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"taskA_user_embeddings__{name}__256d__test_only.pt"
        if dst.exists() and (not bool(args.overwrite)):
            print(f"[SKIP] {name}: exists {dst}", flush=True)
            out_index[name] = {"file": str(dst), "num_embeddings": -1}
            continue

        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload.get("model_state_dict", payload.get("model", None)))
        if not isinstance(state_dict, dict):
            state_dict = payload.get("state_dict", {})
        # For export, we feed raw seq embeddings, so we do NOT need to load huge frozen item_embedding weights.
        # Remove them to avoid shape mismatch and unnecessary memory use.
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("item_embedding.") or k.startswith("to_item_space."))}
        # Build a model with a dummy embedding table; weights will be overwritten by state_dict.
        dummy_item = torch.zeros((2, int(msg_dim)), dtype=torch.float32)
        model = build_model(
            name=name,
            msg_dim=int(msg_dim),
            profile_dim=int(profile_dim),
            item_emb_table=dummy_item,
            num_classes=15,
            embed_dim=256,
            dismir_objective="next_item",
            use_profile=None,
        )
        model.load_state_dict(state_dict, strict=False)
        trainer = Trainer(model=model, device=str(args.device), use_amp=True)
        emb_map = trainer.extract_embeddings(loader, model_name=name, split_name="test_only", log_every=200)
        torch.save({"embeddings": emb_map, "meta": {"model": name, "dim": 256, "split": "test_only"}}, dst)
        print(f"[OK] {name}: exported={len(emb_map)} -> {dst}", flush=True)
        out_index[name] = {"file": str(dst), "num_embeddings": int(len(emb_map))}

    (out_root / "export_test_only_index.json").write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] index saved: {out_root / 'export_test_only_index.json'}", flush=True)


if __name__ == "__main__":
    main()

