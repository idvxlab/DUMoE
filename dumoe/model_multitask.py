import torch
import torch.nn.functional as F
from typing import Dict, Optional

from model import PreferenceModel


class PreferenceModelMultiTask(PreferenceModel):
    """
    ：
    - ：（CrossEntropy）
    - ：next-item （cosine / mse / neg_sampling）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.output_fc.in_features
        msg_feat_dim = self.explicit_mlp[-1].out_features
        if not hasattr(self, "next_item_head"):
            self.next_item_head = torch.nn.Linear(hidden_dim, msg_feat_dim)
        self.item_proj = torch.nn.Linear(msg_feat_dim, hidden_dim)

    def _get_user_repr(
        self,
        x_explicit: torch.Tensor,
        x_implicit: torch.Tensor,
        x_cross_time: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        adapter_index: Optional[int] = None,
    ):
        fused_feat = self._extract_fused_feat(
            x_explicit=x_explicit,
            x_implicit=x_implicit,
            x_cross_time=x_cross_time,
            src_key_padding_mask=src_key_padding_mask,
        )

        if adapter_index is not None:
            if adapter_index < 0 or adapter_index >= len(self.adapter_pool):
                raise ValueError(f"adapter_index={adapter_index} ， [0, {len(self.adapter_pool) - 1}]")
            user_repr = self.adapter_pool[adapter_index](fused_feat)
            gate_weights = torch.zeros(fused_feat.size(0), self.num_classes, device=fused_feat.device, dtype=fused_feat.dtype)
            gate_weights[:, adapter_index] = 1.0
            return user_repr, gate_weights

        if self.use_adapter:
            user_repr, gate_weights = self._compose_adapter_repr(
                fused_feat=fused_feat,
                temperature=temperature,
                adapter_index=None,
            )
            return user_repr, gate_weights

        dummy_weights = torch.zeros(fused_feat.size(0), self.num_classes, device=fused_feat.device, dtype=fused_feat.dtype)
        return fused_feat, dummy_weights

    def forward(
        self,
        x_explicit: torch.Tensor,
        x_implicit: torch.Tensor,
        x_cross_time: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_gate_weights: bool = False,
        adapter_index: Optional[int] = None,
        return_next_item_pred: bool = False,
        return_user_repr: bool = False,
    ):
        user_repr, gate_weights = self._get_user_repr(
            x_explicit=x_explicit,
            x_implicit=x_implicit,
            x_cross_time=x_cross_time,
            src_key_padding_mask=src_key_padding_mask,
            temperature=temperature,
            adapter_index=adapter_index,
        )

        logits = self.output_fc(user_repr)
        next_item_pred = self.next_item_head(user_repr)

        if return_gate_weights and return_next_item_pred and return_user_repr:
            return logits, gate_weights, next_item_pred, user_repr
        if return_gate_weights and return_next_item_pred:
            return logits, gate_weights, next_item_pred
        if return_gate_weights and return_user_repr:
            return logits, gate_weights, user_repr
        if return_next_item_pred and return_user_repr:
            return logits, next_item_pred, user_repr
        if return_gate_weights:
            return logits, gate_weights
        if return_next_item_pred:
            return logits, next_item_pred
        if return_user_repr:
            return logits, user_repr
        return logits

    def compute_neg_sampling_rec_loss(
        self,
        user_repr: torch.Tensor,
        pos_item_repr: torch.Tensor,
        neg_item_repr: torch.Tensor,
        neg_reduction: str = "mean",
    ) -> torch.Tensor:
        """
        （batch  mean）：
        - neg_reduction="sum":
          L = -mean_b [ log σ(u^T v_pos) + Σ_k log σ(-u^T v_neg_k) ]
        - neg_reduction="mean"（）:
          L = -mean_b [ log σ(u^T v_pos) + (1/K) Σ_k log σ(-u^T v_neg_k) ]
           K  logits ， loss  rec_weight。

        :
        - user_repr: [B, D]
        - pos_item_repr: [B, D]
        - neg_item_repr: [B, K, D]
        """
        if user_repr.dim() != 2 or pos_item_repr.dim() != 2 or neg_item_repr.dim() != 3:
            raise ValueError("： user_repr=[B,D], pos_item_repr=[B,D], neg_item_repr=[B,K,D]")
        if user_repr.size(0) != pos_item_repr.size(0) or user_repr.size(0) != neg_item_repr.size(0):
            raise ValueError("batch ")
        if user_repr.size(1) != pos_item_repr.size(1) or user_repr.size(1) != neg_item_repr.size(2):
            raise ValueError("embedding ")

        pos_logits = (user_repr * pos_item_repr).sum(dim=-1)
        pos_term = F.logsigmoid(pos_logits)

        neg_logits = torch.einsum("bd,bkd->bk", user_repr, neg_item_repr)
        neg_logs = F.logsigmoid(-neg_logits)
        nr = str(neg_reduction).lower().strip()
        if nr == "sum":
            neg_term = neg_logs.sum(dim=-1)
        elif nr == "mean":
            neg_term = neg_logs.mean(dim=-1)
        else:
            raise ValueError(f"neg_reduction  sum|mean，: {neg_reduction!r}")

        return -(pos_term + neg_term).mean()

    def project_items(
        self,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
    ):
        """
         item embedding（ 1536 ） user_repr （hidden_dim， 256）。
        - pos_item_emb: [B, D_item]
        - neg_item_emb: [B, K, D_item]
        :
        - pos_item_proj: [B, D_user]
        - neg_item_proj: [B, K, D_user]
        """
        if pos_item_emb.dim() != 2 or neg_item_emb.dim() != 3:
            raise ValueError("project_items ： pos=[B,D], neg=[B,K,D]")
        bsz, k_neg, d_item = neg_item_emb.shape
        if pos_item_emb.size(0) != bsz or pos_item_emb.size(1) != d_item:
            raise ValueError("project_items ")

        pos_item_proj = self.item_proj(pos_item_emb)
        neg_flat = neg_item_emb.view(bsz * k_neg, d_item)
        neg_proj = self.item_proj(neg_flat).view(bsz, k_neg, -1)
        return pos_item_proj, neg_proj

    def compute_multi_task_loss(
        self,
        logits: torch.Tensor,
        cls_labels: torch.Tensor,
        user_repr: Optional[torch.Tensor] = None,
        next_item_pred: Optional[torch.Tensor] = None,
        next_item_target: Optional[torch.Tensor] = None,
        neg_item_targets: Optional[torch.Tensor] = None,
        cls_loss_weight: float = 1.0,
        next_item_loss_weight: float = 1.5,
        next_item_loss_type: str = "cosine",
        rec_neg_reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        cls_loss = F.cross_entropy(logits, cls_labels)
        total_loss = cls_loss_weight * cls_loss

        losses: Dict[str, torch.Tensor] = {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "next_item_loss": torch.zeros_like(cls_loss),
        }

        if next_item_target is None:
            return losses

        mode = str(next_item_loss_type).lower().strip()
        if mode == "cosine":
            if next_item_pred is None:
                raise ValueError("next_item_loss_type='cosine'  next_item_pred")
            pred_norm = F.normalize(next_item_pred, p=2, dim=-1)
            target_norm = F.normalize(next_item_target, p=2, dim=-1)
            next_item_loss = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()
        elif mode == "mse":
            if next_item_pred is None:
                raise ValueError("next_item_loss_type='mse'  next_item_pred")
            next_item_loss = F.mse_loss(next_item_pred, next_item_target)
        elif mode == "neg_sampling":
            if user_repr is None:
                raise ValueError("next_item_loss_type='neg_sampling'  user_repr")
            if neg_item_targets is None:
                raise ValueError("next_item_loss_type='neg_sampling'  neg_item_targets")
            next_item_loss = self.compute_neg_sampling_rec_loss(
                user_repr=user_repr,
                pos_item_repr=next_item_target,
                neg_item_repr=neg_item_targets,
                neg_reduction=rec_neg_reduction,
            )
        else:
            raise ValueError(f" next_item_loss_type={mode}， cosine/mse/neg_sampling")

        total_loss = total_loss + next_item_loss_weight * next_item_loss
        losses["total_loss"] = total_loss
        losses["next_item_loss"] = next_item_loss
        return losses

    def set_training_stage(self, stage: int):
        """
        ； item_proj / next_item_head ，
         loss （ pipeline  cls_weight/rec_weight ）。
        """
        super().set_training_stage(stage)

        for p in self.item_proj.parameters():
            p.requires_grad = True
        if hasattr(self, "next_item_head"):
            for p in self.next_item_head.parameters():
                p.requires_grad = True

        self._print_multitask_trainable_status(stage)

    def _print_multitask_trainable_status(self, stage: int):
        """ stage 。"""
        def _count_params(module: torch.nn.Module) -> int:
            return sum(int(p.numel()) for p in module.parameters())

        def _count_trainable(module: torch.nn.Module) -> int:
            return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)

        item_total = _count_params(self.item_proj)
        item_train = _count_trainable(self.item_proj)

        msg = (
            f"[MultiTask][Stage {stage}] "
            f"item_proj trainable={item_train}/{item_total}"
        )

        if hasattr(self, "next_item_head"):
            head_total = _count_params(self.next_item_head)
            head_train = _count_trainable(self.next_item_head)
            msg += f", next_item_head trainable={head_train}/{head_total}"

        print(msg)
