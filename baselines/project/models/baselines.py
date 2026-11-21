from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from project.models.base import UserRepModel


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask: True  padding（）。"""
    valid = (~mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


class BaseSequentialModel(UserRepModel):
    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        use_profile: bool,
    ) -> None:
        super().__init__()
        self.msg_dim = int(msg_dim)
        self.profile_dim = int(profile_dim)
        if isinstance(item_emb_table, nn.Embedding):
            self.item_embedding = item_emb_table
            self.item_embedding.weight.requires_grad_(False)
            self.num_items = int(self.item_embedding.weight.size(0))
            self.item_dim = int(self.item_embedding.weight.size(1))
        else:
            tens = item_emb_table
            self.num_items = int(tens.size(0))
            self.item_dim = int(tens.size(1))
            self.item_embedding = nn.Embedding.from_pretrained(tens.float(), freeze=True)
        self.embed_dim = int(embed_dim)
        self.use_profile = bool(use_profile)
        self.to_item_space = nn.Linear(self.embed_dim, self.item_dim)
        self.profile_proj = nn.Sequential(nn.Linear(self.profile_dim, self.embed_dim), nn.ReLU())

    def lookup_seq_emb(self, seq_input: torch.Tensor) -> torch.Tensor:
        if seq_input.dim() == 3:
            return seq_input
        return self.item_embedding(seq_input)

    def split_next_item_batch(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_mask = batch["seq_mask"].clone()
        seq_ids = batch["seq_ids"]
        profile = batch["profile"]
        bsz, slen = seq_mask.shape
        valid_len = (~seq_mask).sum(dim=1).clamp(min=1)
        target_pos = (valid_len - 1).clamp(max=slen - 1)
        target_item = seq_ids.gather(1, target_pos.unsqueeze(1)).squeeze(1).clamp(
            0, self.num_items - 1
        )
        row = torch.arange(bsz, device=seq_mask.device)
        seq_mask[row, target_pos] = True
        return seq_ids, seq_mask, profile, target_item

    def next_item_logits(self, z: torch.Tensor) -> torch.Tensor:
        z_item = self.to_item_space(z)
        return torch.matmul(z_item, self.item_embedding.weight.t())

    def next_item_loss(self, z: torch.Tensor, target_item: torch.Tensor) -> torch.Tensor:
        logits = self.next_item_logits(z)
        return F.cross_entropy(logits, target_item.clamp(0, self.num_items - 1))

    def _profile_fuse(self, z: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        if not self.use_profile:
            return z
        return z + self.profile_proj(profile)


class SASRecModel(BaseSequentialModel):
    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        use_profile: bool,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.in_proj = nn.Linear(msg_dim, embed_dim)
        ff = max(256, embed_dim * 8)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        seq_emb = self.lookup_seq_emb(seq_ids)
        h = self.encoder(self.in_proj(seq_emb), src_key_padding_mask=seq_mask)
        z = masked_mean(h, seq_mask)
        return F.normalize(self._profile_fuse(z, profile), dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids, seq_mask, profile, target_item = self.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        return self.next_item_loss(z, target_item)


class HM4SRModel(BaseSequentialModel):
    """HM4SR ， SASRec  next-item 。

    （RecBole + Games ） ``Baselines/HM4SR``。
    ``run_unified_pipeline``  baseline **** ``UnifiedWindowDataset``、
    ``--epochs / --batch-size / --lr / ...``  UNITE item ；
     id / text / image ， softmax  Transformer
    ， next-item CE  HM4SR  batch  ID （）。
    """

    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        use_profile: bool,
        idcl_temperature: float = 0.2,
        idcl_weight: float = 0.1,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.idcl_temperature = float(idcl_temperature)
        self.idcl_weight = float(idcl_weight)
        self.proj_id = nn.Linear(self.msg_dim, self.embed_dim)
        self.proj_txt = nn.Linear(self.msg_dim, self.embed_dim)
        self.proj_img = nn.Linear(self.msg_dim, self.embed_dim)
        self.branch_gate = nn.Linear(3 * self.embed_dim, 3)
        ff = max(256, self.embed_dim * 4)
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        seq_emb = self.lookup_seq_emb(seq_ids)
        id_h = self.proj_id(seq_emb)
        txt_h = self.proj_txt(seq_emb)
        img_h = self.proj_img(seq_emb)
        cat = torch.cat([id_h, txt_h, img_h], dim=-1)
        g = torch.softmax(self.branch_gate(cat), dim=-1)
        fused = g[..., 0:1] * id_h + g[..., 1:2] * txt_h + g[..., 2:3] * img_h
        h = self.encoder(fused, src_key_padding_mask=seq_mask)
        z = masked_mean(h, seq_mask)
        return F.normalize(self._profile_fuse(z, profile), dim=-1)

    def _idcl_loss(self, z: torch.Tensor, pos_id: torch.Tensor) -> torch.Tensor:
        t = self.idcl_temperature
        seq_output = F.normalize(z, dim=1)
        pid = pos_id.clamp(0, self.num_items - 1)
        pos_items_emb = F.normalize(self.proj_id(self.item_embedding(pid)), dim=1)
        pos_logits = torch.exp((seq_output * pos_items_emb).sum(dim=1) / t)
        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / t
        same_pos_id = pid.unsqueeze(1) == pid.unsqueeze(0)
        same_pos_id = torch.logical_xor(
            same_pos_id, torch.eye(pid.size(0), dtype=torch.bool, device=pid.device)
        )
        neg_logits = torch.where(same_pos_id, torch.zeros_like(neg_logits), torch.exp(neg_logits))
        denom = neg_logits.sum(dim=1).clamp(min=1e-8)
        return (-torch.log(pos_logits / denom)).mean()

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids, seq_mask, profile, target_item = self.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        ce = self.next_item_loss(z, target_item)
        return ce + self.idcl_weight * self._idcl_loss(z, target_item)


class BERT4RecModel(BaseSequentialModel):
    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        use_profile: bool,
        mask_ratio: float = 0.15,
        num_negatives: int = 100,
        max_masks_per_seq: int = 5,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.in_proj = nn.Linear(msg_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        ff = max(256, embed_dim * 8)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.mask_ratio = float(mask_ratio)
        self.mask_head = nn.Linear(embed_dim, self.item_dim)
        self.num_negatives = int(num_negatives)
        self.max_masks_per_seq = int(max_masks_per_seq)

    def _sample_mask_positions(self, seq_mask: torch.Tensor) -> torch.Tensor:
        bsz, slen = seq_mask.shape
        valid = ~seq_mask
        rand = torch.rand((bsz, slen), device=seq_mask.device)
        mask_pos = (rand < self.mask_ratio) & valid
        if self.max_masks_per_seq > 0:
            keep_rank = torch.rand((bsz, slen), device=seq_mask.device)
            keep_rank = keep_rank.masked_fill(~mask_pos, -1.0)
            topk = min(self.max_masks_per_seq, slen)
            topk_idx = keep_rank.topk(k=topk, dim=1).indices
            limited = torch.zeros_like(mask_pos)
            limited.scatter_(1, topk_idx, True)
            mask_pos = mask_pos & limited
        return mask_pos

    def _sample_negatives(self, pos_items: torch.Tensor) -> torch.Tensor:
        device = pos_items.device
        neg = torch.randint(
            low=0,
            high=self.num_items,
            size=(pos_items.size(0), self.num_negatives),
            device=device,
        )
        pos_expand = pos_items.unsqueeze(1)
        for _ in range(8):
            collision = neg.eq(pos_expand)
            if not collision.any():
                break
            neg = torch.where(collision, torch.randint_like(neg, high=self.num_items), neg)
        return neg

    def _encode_tokens(
        self,
        seq_ids: torch.Tensor,
        seq_mask: torch.Tensor,
        forced_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_emb = self.lookup_seq_emb(seq_ids)
        x = self.in_proj(seq_emb)
        if forced_mask is None:
            mask_pos = self._sample_mask_positions(seq_mask)
        else:
            mask_pos = forced_mask & (~seq_mask)
        x = torch.where(mask_pos.unsqueeze(-1), self.mask_token.view(1, 1, -1), x)
        h = self.encoder(x, src_key_padding_mask=seq_mask)
        return h, mask_pos

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        no_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
        h, _ = self._encode_tokens(seq_ids, seq_mask, forced_mask=no_mask)
        z = masked_mean(h, seq_mask)
        return F.normalize(self._profile_fuse(z, profile), dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids = batch["seq_ids"]
        seq_mask = batch["seq_mask"]
        h, mask_pos = self._encode_tokens(seq_ids, seq_mask, forced_mask=None)
        tries = 0
        while not mask_pos.any() and tries < 10:
            h, mask_pos = self._encode_tokens(seq_ids, seq_mask, forced_mask=None)
            tries += 1
        if not mask_pos.any():
            forced = torch.zeros_like(seq_mask, dtype=torch.bool)
            first_valid = (~seq_mask).float().argmax(dim=1)
            forced[torch.arange(forced.size(0), device=forced.device), first_valid] = True
            h, mask_pos = self._encode_tokens(seq_ids, seq_mask, forced_mask=forced)

        masked_h = h[mask_pos]
        query = self.mask_head(masked_h)
        pos_items = seq_ids[mask_pos].clamp(0, self.num_items - 1)
        neg_items = self._sample_negatives(pos_items)
        pos_emb = self.item_embedding(pos_items).unsqueeze(1)
        neg_emb = self.item_embedding(neg_items)
        cand_emb = torch.cat([pos_emb, neg_emb], dim=1)
        logits = torch.einsum("md,mkd->mk", query, cand_emb)
        target = torch.zeros((logits.size(0),), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)


class MINDModel(BaseSequentialModel):
    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        use_profile: bool,
        num_interests: int = 4,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.in_proj = nn.Linear(msg_dim, embed_dim)
        self.num_interests = int(num_interests)
        self.query = nn.Parameter(torch.randn(self.num_interests, self.embed_dim) * 0.02)

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        seq_emb = self.lookup_seq_emb(seq_ids)
        h = self.in_proj(seq_emb)
        attn = torch.einsum("blh,kh->blk", h, self.query)
        attn = attn.masked_fill(seq_mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(attn, dim=1)
        interests = torch.einsum("blh,blk->bkh", h, attn)
        z = interests.mean(dim=1)
        return F.normalize(self._profile_fuse(z, profile), dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids, seq_mask, profile, target_item = self.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        return self.next_item_loss(z, target_item)


class ProfileFusionModel(BaseSequentialModel):
    """PeterRec / PTUM / HORAE 。"""

    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        arch: str,
        use_profile: bool,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.arch = str(arch).lower()
        self.in_proj = nn.Linear(msg_dim, embed_dim)
        if self.arch == "ptum":
            self.short_len = 30
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
                nn.Sigmoid(),
            )
            self.fuse = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.ReLU())
        elif self.arch == "horae":
            self.msg_update = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
            self.fuse = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU())
        else:
            self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.fuse = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.ReLU())

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        seq_emb = self.lookup_seq_emb(seq_ids)
        h = self.in_proj(seq_emb)
        if self.use_profile:
            p = self.profile_proj(profile)
        else:
            p = torch.zeros(h.size(0), self.embed_dim, device=h.device, dtype=h.dtype)

        if self.arch == "ptum":
            long_h = masked_mean(h, seq_mask)
            sl = min(self.short_len, h.size(1))
            short_h = masked_mean(h[:, :sl], seq_mask[:, :sl])
            g = self.gate(torch.cat([short_h, long_h, p], dim=-1))
            u = g * short_h + (1.0 - g) * long_h
            z = self.fuse(torch.cat([u, p, u * p], dim=-1))
            return F.normalize(z, dim=-1)

        if self.arch == "horae":
            msg_pool = masked_mean(self.msg_update(h), seq_mask)
            z = self.fuse(torch.cat([msg_pool, p], dim=-1))
            return F.normalize(z, dim=-1)

        out, _ = self.gru(h)
        hs = masked_mean(out, seq_mask)
        z = self.fuse(torch.cat([hs, p, hs * p], dim=-1))
        return F.normalize(z, dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids, seq_mask, profile, target_item = self.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        return self.next_item_loss(z, target_item)


class UniSRecModel(BaseSequentialModel):
    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        embed_dim: int,
        use_profile: bool,
        temperature: float = 0.07,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.in_proj = nn.Linear(msg_dim, embed_dim)
        ff = max(256, embed_dim * 8)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=ff,
            dropout=0.2,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.temperature = float(temperature)

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        seq_emb = self.lookup_seq_emb(seq_ids)
        h = self.encoder(self.in_proj(seq_emb), src_key_padding_mask=seq_mask)
        z = masked_mean(h, seq_mask)
        return F.normalize(self._profile_fuse(z, profile), dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids, seq_mask, profile, target_item = self.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        return self.next_item_loss(z, target_item)


class DisMIRModel(BaseSequentialModel):
    def __init__(
        self,
        msg_dim: int,
        profile_dim: int,
        item_emb_table: Union[torch.Tensor, nn.Embedding],
        num_classes: int,
        embed_dim: int,
        objective: str,
        use_profile: bool,
    ) -> None:
        super().__init__(msg_dim, profile_dim, item_emb_table, embed_dim, use_profile)
        self.in_proj = nn.Linear(msg_dim, embed_dim)
        ff = max(256, embed_dim * 8)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.cls = nn.Linear(embed_dim, int(num_classes))
        self.objective = str(objective).lower()

    def encode(
        self, seq_ids: torch.Tensor, seq_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        seq_emb = self.lookup_seq_emb(seq_ids)
        h = self.encoder(self.in_proj(seq_emb), src_key_padding_mask=seq_mask)
        z = masked_mean(h, seq_mask)
        return F.normalize(self._profile_fuse(z, profile), dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.encode(batch["seq_ids"], batch["seq_mask"], batch["profile"])

    def compute_loss(self, batch: dict) -> torch.Tensor:
        seq_ids, seq_mask, profile, target_item = self.split_next_item_batch(batch)
        z = self.encode(seq_ids, seq_mask, profile)
        if self.objective in {"classification", "label"}:
            return F.cross_entropy(self.cls(z), batch["label"])
        return self.next_item_loss(z, target_item)


def build_model(
    name: str,
    msg_dim: int,
    profile_dim: int,
    item_emb_table: Union[torch.Tensor, nn.Embedding],
    num_classes: int = 15,
    embed_dim: int = 256,
    dismir_objective: str = "next_item",
    use_profile: bool | None = None,
) -> UserRepModel:
    key = str(name).lower()
    resolved = bool(use_profile) if use_profile is not None else key == "dismir"

    if key == "sasrec":
        return SASRecModel(msg_dim, profile_dim, item_emb_table, embed_dim, resolved)
    if key == "bert4rec":
        return BERT4RecModel(msg_dim, profile_dim, item_emb_table, embed_dim, resolved)
    if key == "mind":
        return MINDModel(msg_dim, profile_dim, item_emb_table, embed_dim, resolved)
    if key == "peterrec":
        return ProfileFusionModel(
            msg_dim, profile_dim, item_emb_table, embed_dim, "peterrec", resolved
        )
    if key == "ptum":
        return ProfileFusionModel(msg_dim, profile_dim, item_emb_table, embed_dim, "ptum", resolved)
    if key == "unisrec":
        return UniSRecModel(msg_dim, profile_dim, item_emb_table, embed_dim, resolved)
    if key == "horae":
        return ProfileFusionModel(msg_dim, profile_dim, item_emb_table, embed_dim, "horae", resolved)
    if key == "hm4sr":
        return HM4SRModel(msg_dim, profile_dim, item_emb_table, embed_dim, resolved)
    if key == "dismir":
        return DisMIRModel(
            msg_dim,
            profile_dim,
            item_emb_table,
            num_classes,
            embed_dim,
            dismir_objective,
            resolved,
        )
    raise ValueError(f"unsupported model: {name}")
