# Import PyTorch nn modules
# torch: tensors and autograd
import torch
# nn: layers and losses
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class Adapter(nn.Module):
    """
    Lightweight adapter for category-specific transforms
    Similar to Per-Pcs LoRA-style adapters
    
    Design:
    - down_proj: high-dim -> bottleneck
    - activation: ReLU
    - up_proj: bottleneck -> original dim
    - residual: out = x + Adapter(x)
    """
    def __init__(self, input_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # Init near identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor):
        """
        Forward: x -> down -> ReLU -> dropout -> up
        Return residual x + Adapter(x)
        """
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return x + h  # 


class GatingNetwork(nn.Module):
    """
    Gate that weights adapters from fused features
    Similar to Per-Pcs gating
    
    Design:
    - Input: fused feature vector
    - Output: K adapter weights (softmax)
    - LayerNorm for stable training
    """
    def __init__(self, input_dim: int, num_adapters: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_adapters = num_adapters
        
        # Two-layer MLP gate
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_adapters)
        )
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0, topk: Optional[int] = None):
        """
        Forward: compute adapter weights
        
        Args:
        - x: [batch, input_dim]
        - temperature: softmax temperature
        - topk: keep top-k adapters
        
        Returns:
        - weights: [batch, num_adapters]
        - selected_indices if topk is set
        """
        logits = self.gate_mlp(x)  # [batch, num_adapters]
        
        if topk is not None and topk < self.num_adapters:
            topk_values, topk_indices = torch.topk(logits, k=topk, dim=-1)  # [batch, topk]
            
            weights_topk = F.softmax(topk_values / temperature, dim=-1)  # [batch, topk]
            
            weights = torch.zeros_like(logits, dtype=weights_topk.dtype)  # [batch, num_adapters]
            weights.scatter_(1, topk_indices, weights_topk)  #  topk 
            
            return weights, topk_indices
        else:
            weights = F.softmax(logits / temperature, dim=-1)  # [batch, num_adapters]
            return weights, None


class PreferenceModel(nn.Module):
    """
    PreferenceModel —  Per-Pcs 
    
    ：
     (MLP + LSTM + Transformer) →  MLP → 
    Gating Network → Adapter Pool () → 
    
     ( convert_1770458085.md):
    1. ： Adapter  Gate，
    2. Adapter Pool ： Adapter
    3. ： Adapter，
    
    Args:
    - user_feat_dim: （ 1536）
    - msg_feat_dim: （ 1536）
    - hidden_dim: （ 256）
    - num_classes: （ 15）
    - lstm_layers: LSTM 
    - transformer_layers: Transformer Encoder 
    - nhead: Transformer 
    - adapter_bottleneck: Adapter （ 64）
    - use_adapter:  Adapter （ 1  False）
    - topk_adapters: Top-K  K （None ）
    """

    def __init__(self,
                 user_feat_dim: int = 1536,
                 msg_feat_dim: int = 1536,
                 hidden_dim: int = 256,
                 num_classes: int = 15,
                 lstm_layers: int = 1,
                 transformer_layers: int = 2,
                 nhead: int = 8,
                 dropout: float = 0.5,
                 adapter_bottleneck: int = 64,
                 use_adapter: bool = False,
                 topk_adapters: Optional[int] = None,
                 use_uniform_adapter_fusion: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_adapter = use_adapter
        self.topk_adapters = topk_adapters
        self.use_uniform_adapter_fusion = use_uniform_adapter_fusion

        # -----------------------------
        # -----------------------------
        self.explicit_mlp = nn.Sequential(
            nn.Linear(user_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, msg_feat_dim)  # 
        )

        # -----------------------------
        # -----------------------------
        self.lstm = nn.LSTM(input_size=msg_feat_dim, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True)
        self.lstm_out_fc = nn.Sequential(
            nn.Linear(hidden_dim, msg_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -----------------------------
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(d_model=msg_feat_dim, nhead=nhead, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # -----------------------------
        # -----------------------------
        fused_feat_dim = msg_feat_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # -----------------------------
        # -----------------------------
        self.adapter_pool = nn.ModuleList([
            Adapter(input_dim=hidden_dim, bottleneck_dim=adapter_bottleneck, dropout=dropout)
            for _ in range(num_classes)
        ])
        
        # -----------------------------
        # -----------------------------
        self.gating_network = GatingNetwork(
            input_dim=hidden_dim,
            num_adapters=num_classes,
            hidden_dim=128,
            dropout=dropout
        )
        
        # -----------------------------
        # -----------------------------
        self.output_fc = nn.Linear(hidden_dim, num_classes)

    def _extract_fused_feat(
        self,
        x_explicit: torch.Tensor,
        x_implicit: torch.Tensor,
        x_cross_time: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ backbone ， [B, hidden_dim]。"""
        explicit_feat = self.explicit_mlp(x_explicit)

        lstm_out, _ = self.lstm(x_implicit)
        lstm_feat = self.lstm_out_fc(lstm_out[:, -1, :])

        transformer_out = self.transformer(x_cross_time, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            mask_t = src_key_padding_mask.transpose(0, 1).unsqueeze(-1)
            valid_mask = ~mask_t
            transformer_out = transformer_out * valid_mask.float()
            sum_out = transformer_out.sum(dim=0)
            count = valid_mask.sum(dim=0).float()
            count = torch.clamp(count, min=1.0)
            out_cross_time = sum_out / count
        else:
            out_cross_time = transformer_out.mean(dim=0)

        fused_input = torch.cat([explicit_feat, lstm_feat, out_cross_time], dim=-1)
        fused_feat = self.fusion_mlp(fused_input)
        return fused_feat

    def _compose_adapter_repr(
        self,
        fused_feat: torch.Tensor,
        temperature: float = 1.0,
        adapter_index: Optional[int] = None,
    ):
        """ fused_feat  adapter ， (adapted_feat, gate_weights)。"""
        if adapter_index is not None:
            if adapter_index < 0 or adapter_index >= len(self.adapter_pool):
                raise ValueError(f"adapter_index={adapter_index} ， [0, {len(self.adapter_pool) - 1}]")
            adapted_feat = self.adapter_pool[adapter_index](fused_feat)
            gate_weights = torch.zeros(fused_feat.size(0), self.num_classes, device=fused_feat.device, dtype=fused_feat.dtype)
            gate_weights[:, adapter_index] = 1.0
            return adapted_feat, gate_weights

        if self.use_uniform_adapter_fusion:
            adapted_feat = torch.zeros_like(fused_feat)
            for adapter in self.adapter_pool:
                adapted_feat = adapted_feat + adapter(fused_feat)
            adapted_feat = adapted_feat / float(len(self.adapter_pool))
            gate_weights = torch.full(
                (fused_feat.size(0), self.num_classes),
                1.0 / float(self.num_classes),
                device=fused_feat.device,
                dtype=fused_feat.dtype,
            )
            return adapted_feat, gate_weights

        gate_weights, _ = self.gating_network(
            fused_feat,
            temperature=temperature,
            topk=self.topk_adapters,
        )

        adapted_feat = torch.zeros_like(fused_feat)
        for i, adapter in enumerate(self.adapter_pool):
            adapter_out = adapter(fused_feat)
            adapted_feat = adapted_feat + gate_weights[:, i:i+1] * adapter_out

        return adapted_feat, gate_weights

    @torch.no_grad()
    def encode_user_repr(
        self,
        x_explicit: torch.Tensor,
        x_implicit: torch.Tensor,
        x_cross_time: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        x_implicit_no_month: Optional[torch.Tensor] = None,
        x_cross_time_no_month: Optional[torch.Tensor] = None,
        src_key_padding_mask_no_month: Optional[torch.Tensor] = None,
        mode: str = "thresholded",
        kl_threshold: float = 0.012,
        gate_mode: str = "soft",
        kl_soft_scale: float = 0.01,
        gate_min_alpha: float = 0.05,
        gate_max_alpha: float = 0.80,
        adapter_index: Optional[int] = None,
        return_gate_info: bool = False,
    ):
        """ frozen ，。"""
        mode = str(mode).lower().strip()
        if mode not in {"base", "adapter", "thresholded"}:
            raise ValueError(f" mode={mode}， base/adapter/thresholded")

        base_repr = self._extract_fused_feat(
            x_explicit=x_explicit,
            x_implicit=x_implicit,
            x_cross_time=x_cross_time,
            src_key_padding_mask=src_key_padding_mask,
        )

        if mode == "base":
            if return_gate_info:
                return base_repr, {"mode": "base", "repr_dim": int(base_repr.size(-1))}
            return base_repr

        adapter_repr, gate_weights = self._compose_adapter_repr(
            fused_feat=base_repr,
            temperature=1.0,
            adapter_index=adapter_index,
        )

        if mode == "adapter":
            if return_gate_info:
                return adapter_repr, {
                    "mode": "adapter",
                    "repr_dim": int(adapter_repr.size(-1)),
                    "gate_weights": gate_weights,
                }
            return adapter_repr

        if x_implicit_no_month is None or x_cross_time_no_month is None or src_key_padding_mask_no_month is None:
            raise ValueError("mode='thresholded'  no-month ")

        no_month_repr = self._extract_fused_feat(
            x_explicit=x_explicit,
            x_implicit=x_implicit_no_month,
            x_cross_time=x_cross_time_no_month,
            src_key_padding_mask=src_key_padding_mask_no_month,
        )

        base_logits = self.output_fc(base_repr)
        no_month_logits = self.output_fc(no_month_repr)

        probs_base = torch.softmax(base_logits, dim=1)
        probs_no_month = torch.softmax(no_month_logits, dim=1)
        kl = (probs_base * (probs_base.add(1e-12).log() - probs_no_month.add(1e-12).log())).sum(dim=1)

        gate_mode = str(gate_mode).lower().strip()
        if gate_mode == "soft":
            num_classes = max(2, int(probs_base.size(1)))
            entropy = -(probs_base * probs_base.add(1e-12).log()).sum(dim=1)
            entropy_norm = entropy / max(1e-6, math.log(num_classes))

            z = (kl - kl_threshold) / max(1e-6, kl_soft_scale)
            alpha_kl = torch.sigmoid(z)
            alpha = alpha_kl * (0.5 + 0.5 * entropy_norm)
            alpha = alpha.clamp(min=gate_min_alpha, max=gate_max_alpha).unsqueeze(1)
            final_repr = (1.0 - alpha) * base_repr + alpha * adapter_repr
            activate_mask = kl > kl_threshold
        elif gate_mode == "hard":
            activate_mask = kl > kl_threshold
            final_repr = base_repr.clone()
            if activate_mask.any():
                final_repr[activate_mask] = adapter_repr[activate_mask]
            alpha = activate_mask.float().unsqueeze(1)
        else:
            raise ValueError(f" gate_mode={gate_mode}， hard/soft")

        if return_gate_info:
            gate_info: Dict[str, object] = {
                "mode": "thresholded",
                "repr_dim": int(final_repr.size(-1)),
                "gate_mode": gate_mode,
                "kl": kl,
                "activate_mask": activate_mask,
                "activated_count": int(activate_mask.sum().item()),
                "activated_ratio": float(activate_mask.float().mean().item()),
                "alpha": alpha,
                "gate_weights": gate_weights,
            }
            return final_repr, gate_info

        return final_repr

    def forward(self, x_explicit: torch.Tensor, x_implicit: torch.Tensor, x_cross_time: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None, 
                temperature: float = 1.0,
                return_gate_weights: bool = False,
                adapter_index: Optional[int] = None):
        """
        。，，
         use_adapter  Adapter 。

        ：
        - x_explicit: ， [batch, user_feat_dim]，（）。
        - x_implicit: ， [batch, seq_len, msg_feat_dim]，（）。
        - x_cross_time: ， [seq_len, batch, msg_feat_dim]，Transformer （）。
        - src_key_padding_mask:  [batch, seq_len]， Transformer  padding。
        - temperature: Gating Network  softmax 
    - return_gate_weights: （）
    - adapter_index:  Adapter（Stage2  Adapter ），

        ：
        - logits: ， [batch, num_classes]，（）。
        - gate_weights: ()  return_gate_weights=True， Adapter 
        """

        # ============================================================
        # ============================================================
        
        # x_explicit: [batch, user_feat_dim]
        explicit_feat = self.explicit_mlp(x_explicit)  # [batch, msg_feat_dim]

        # x_implicit: [batch, seq_len, msg_feat_dim]
        lstm_out, _ = self.lstm(x_implicit)           # lstm_out: [batch, seq_len, hidden_dim]
        lstm_feat = self.lstm_out_fc(lstm_out[:, -1, :])  #  -> [batch, msg_feat_dim]

        # x_cross_time: [seq_len, batch, msg_feat_dim]
        transformer_out = self.transformer(x_cross_time, src_key_padding_mask=src_key_padding_mask)
        # transformer_out: [seq_len, batch, msg_feat_dim]

        if src_key_padding_mask is not None:
            mask_t = src_key_padding_mask.transpose(0, 1).unsqueeze(-1) # [seq_len, batch, 1]
            
            valid_mask = ~mask_t 
            
            transformer_out = transformer_out * valid_mask.float()
            
            sum_out = transformer_out.sum(dim=0) # [batch, msg_feat_dim]
            
            count = valid_mask.sum(dim=0).float() # [batch, 1]
            
            count = torch.clamp(count, min=1.0)
            
            out_cross_time = sum_out / count # [batch, msg_feat_dim]
        else:
            out_cross_time = transformer_out.mean(dim=0)  # [batch, msg_feat_dim]

        fused_input = torch.cat([explicit_feat, lstm_feat, out_cross_time], dim=-1)  # [batch, msg_feat_dim*3]
        
        fused_feat = self.fusion_mlp(fused_input)  # [batch, hidden_dim]

        # ============================================================
        # ============================================================
        
        if adapter_index is not None:
            if adapter_index < 0 or adapter_index >= len(self.adapter_pool):
                raise ValueError(f"adapter_index={adapter_index} ， [0, {len(self.adapter_pool) - 1}]")
            adapter_out = self.adapter_pool[adapter_index](fused_feat)
            logits = self.output_fc(adapter_out)
            if return_gate_weights:
                gate_weights = torch.zeros(fused_feat.size(0), self.num_classes, device=fused_feat.device)
                gate_weights[:, adapter_index] = 1.0
                return logits, gate_weights
            return logits

        if self.use_adapter:
            if self.use_uniform_adapter_fusion:
                adapted_feat = torch.zeros_like(fused_feat)  # [batch, hidden_dim]
                for adapter in self.adapter_pool:
                    adapted_feat = adapted_feat + adapter(fused_feat)
                adapted_feat = adapted_feat / float(len(self.adapter_pool))
                logits = self.output_fc(adapted_feat)

                if return_gate_weights:
                    gate_weights = torch.full(
                        (fused_feat.size(0), self.num_classes),
                        1.0 / float(self.num_classes),
                        device=fused_feat.device,
                        dtype=fused_feat.dtype,
                    )
                    return logits, gate_weights
                return logits

            gate_weights, selected_indices = self.gating_network(
                fused_feat, 
                temperature=temperature, 
                topk=self.topk_adapters
            )  # gate_weights: [batch, num_classes]
            
            batch_size = fused_feat.size(0)
            adapted_feat = torch.zeros_like(fused_feat)  # [batch, hidden_dim]
            
            for i, adapter in enumerate(self.adapter_pool):
                adapter_out = adapter(fused_feat)  # [batch, hidden_dim]
                
                adapted_feat = adapted_feat + gate_weights[:, i:i+1] * adapter_out
            
            logits = self.output_fc(adapted_feat)  # [batch, num_classes]
            
            if return_gate_weights:
                return logits, gate_weights
            else:
                return logits
        else:
            logits = self.output_fc(fused_feat)  # [batch, num_classes]
            
            if return_gate_weights:
                dummy_weights = torch.zeros(fused_feat.size(0), self.num_classes, device=fused_feat.device)
                return logits, dummy_weights
            else:
                return logits
    
    # ============================================================
    # ============================================================
    
    def set_training_stage(self, stage: int):
        """
        ，
        
         1 ():
        - ：MLP + LSTM + Transformer + fusion_mlp + output_fc
        - ：Adapter Pool + Gating Network
        - use_adapter = False
        
         2 (Adapter Pool ):
        - ： (MLP + LSTM + Transformer + fusion_mlp)
        - ：Adapter Pool ( Adapter )
        - ：Gating Network
        - use_adapter = True
        
         3 (Gating Network ):
        - ： + Adapter Pool
        - ：Gating Network
        - use_adapter = True
        """
        if stage == 1:
            print(" 1: ")
            self.use_adapter = False
            
            for param in self.explicit_mlp.parameters():
                param.requires_grad = True
            for param in self.lstm.parameters():
                param.requires_grad = True
            for param in self.lstm_out_fc.parameters():
                param.requires_grad = True
            for param in self.transformer.parameters():
                param.requires_grad = True
            for param in self.fusion_mlp.parameters():
                param.requires_grad = True
            for param in self.output_fc.parameters():
                param.requires_grad = True
            
            for adapter in self.adapter_pool:
                for param in adapter.parameters():
                    param.requires_grad = False
            for param in self.gating_network.parameters():
                param.requires_grad = False
                
        elif stage == 2:
            print(" 2: Adapter Pool ")
            self.use_adapter = True
            
            for param in self.explicit_mlp.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.lstm_out_fc.parameters():
                param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.fusion_mlp.parameters():
                param.requires_grad = False
            for param in self.output_fc.parameters():
                param.requires_grad = False
            
            for adapter in self.adapter_pool:
                for param in adapter.parameters():
                    param.requires_grad = True
            
            for param in self.gating_network.parameters():
                param.requires_grad = False
                
        elif stage == 3:
            print(" 3: Gating Network ")
            self.use_adapter = True
            
            for param in self.explicit_mlp.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.lstm_out_fc.parameters():
                param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.fusion_mlp.parameters():
                param.requires_grad = False
            for param in self.output_fc.parameters():
                param.requires_grad = False
            for adapter in self.adapter_pool:
                for param in adapter.parameters():
                    param.requires_grad = False
            
            for param in self.gating_network.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f": {stage}， 1, 2  3")
    
    def get_trainable_params_info(self):
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        ratio = 100 * trainable_params / all_params if all_params > 0 else 0
        print(f": {trainable_params:,} / {all_params:,} ({ratio:.2f}%)")
        return trainable_params, all_params


if __name__ == "__main__":
    # -----------------------------
    # -----------------------------
    print("=" * 60)
    print("")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 5
    user_feat_dim = 1536
    msg_feat_dim = 1536
    num_classes = 15

    x_explicit = torch.randn(batch_size, user_feat_dim)
    x_implicit = torch.randn(batch_size, seq_len, msg_feat_dim)
    x_cross_time = torch.randn(seq_len, batch_size, msg_feat_dim)

    model = PreferenceModel(
        user_feat_dim=user_feat_dim, 
        msg_feat_dim=msg_feat_dim, 
        num_classes=num_classes,
        hidden_dim=256,
        adapter_bottleneck=64,
        use_adapter=False,  #  Adapter
        topk_adapters=3     # Top-3 
    )
    
    print("\n" + "=" * 60)
    print(" 1: ")
    print("=" * 60)
    model.set_training_stage(1)
    model.get_trainable_params_info()
    
    logits_stage1 = model(x_explicit, x_implicit, x_cross_time)
    print(f" logits : {logits_stage1.shape}")  # : [4, 15]
    print(f": [{logits_stage1.min().item():.4f}, {logits_stage1.max().item():.4f}]")
    
    print("\n" + "=" * 60)
    print(" 2: Adapter Pool ")
    print("=" * 60)
    model.set_training_stage(2)
    model.get_trainable_params_info()
    
    logits_stage2 = model(x_explicit, x_implicit, x_cross_time)
    print(f" logits : {logits_stage2.shape}")  # : [4, 15]
    print(f": [{logits_stage2.min().item():.4f}, {logits_stage2.max().item():.4f}]")
    
    print("\n" + "=" * 60)
    print(" 3: Gating Network ")
    print("=" * 60)
    model.set_training_stage(3)
    model.get_trainable_params_info()
    
    logits_stage3, gate_weights = model(x_explicit, x_implicit, x_cross_time, 
                                         temperature=1.0, return_gate_weights=True)
    print(f" logits : {logits_stage3.shape}")  # : [4, 15]
    print(f": [{logits_stage3.min().item():.4f}, {logits_stage3.max().item():.4f}]")
    print(f": {gate_weights.shape}")  # : [4, 15]
    print(f": {gate_weights.sum(dim=1)}")  # :  ≈ 1.0
    print(f"\n Adapter :")
    print(gate_weights[0])  # 
    
    print("\n" + "=" * 60)
    print(" Top-K ")
    print("=" * 60)
    top3_values, top3_indices = torch.topk(gate_weights[0], k=3)
    print(f"Top-3 Adapter : {top3_indices.tolist()}")
    print(f"Top-3 Adapter : {top3_values.tolist()}")
    
    print("\n" + "=" * 60)
    print("！")
    print("=" * 60)

