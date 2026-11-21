from __future__ import annotations

import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        lr: float = 1e-3,
        wd: float = 0.01,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        dev = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(wd))
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_epoch(
        self,
        loader: DataLoader,
        model_name: str = "",
        epoch: int = 0,
        log_every: int = 200,
    ) -> float:
        self.model.train()
        total = 0.0
        n = 0
        t0 = time.time()
        num_batches = len(loader)
        for step, batch in enumerate(loader, start=1):
            b = move_to_device(batch, self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                loss = self.model.compute_loss(b)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            bs = int(b["seq_ids"].size(0))
            total += float(loss.item()) * bs
            n += bs
            if log_every > 0 and (step % log_every == 0 or step == num_batches):
                avg_loss = total / max(1, n)
                elapsed = time.time() - t0
                avg_step = elapsed / max(1, step)
                eta = avg_step * max(0, num_batches - step)
                prefix = f"[{model_name}]" if model_name else "[train]"
                print(
                    f"{prefix}[epoch {epoch}] batch {step}/{num_batches} "
                    f"avg_loss={avg_loss:.6f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        model_name: str = "",
        epoch: int = 0,
        split_name: str = "eval",
        log_every: int = 200,
    ) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        t0 = time.time()
        num_batches = len(loader)
        for step, batch in enumerate(loader, start=1):
            b = move_to_device(batch, self.device)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                loss = self.model.compute_loss(b)
            bs = int(b["seq_ids"].size(0))
            total += float(loss.item()) * bs
            n += bs
            if log_every > 0 and (step % log_every == 0 or step == num_batches):
                avg_loss = total / max(1, n)
                elapsed = time.time() - t0
                avg_step = elapsed / max(1, step)
                eta = avg_step * max(0, num_batches - step)
                prefix = f"[{model_name}]" if model_name else "[eval]"
                ep = f"[epoch {epoch}]" if epoch > 0 else ""
                print(
                    f"{prefix}{ep}[{split_name}] batch {step}/{num_batches} "
                    f"avg_loss={avg_loss:.6f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        return total / max(1, n)

    @torch.no_grad()
    def extract_embeddings(
        self,
        loader: DataLoader,
        model_name: str = "",
        split_name: str = "export",
        log_every: int = 200,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        emb: Dict[str, torch.Tensor] = {}
        t0 = time.time()
        num_batches = len(loader)
        for step, batch in enumerate(loader, start=1):
            keys = batch["key"]
            b = move_to_device(batch, self.device)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                z = self.model.encode(b["seq_ids"], b["seq_mask"], b["profile"]).detach().cpu()
            for i, k in enumerate(keys):
                emb[str(k)] = z[i].float().view(-1)[:256]
            if log_every > 0 and (step % log_every == 0 or step == num_batches):
                elapsed = time.time() - t0
                avg_step = elapsed / max(1, step)
                eta = avg_step * max(0, num_batches - step)
                prefix = f"[{model_name}]" if model_name else "[export]"
                print(
                    f"{prefix}[{split_name}] batch {step}/{num_batches} "
                    f"exported={len(emb)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        return emb
