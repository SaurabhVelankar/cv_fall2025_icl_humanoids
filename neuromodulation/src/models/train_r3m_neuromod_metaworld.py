#!/usr/bin/env python3
"""train_r3m_neuromod_metaworld.py

Minimal downstream training + evaluation on Short-MetaWorld-VLA using:
- Pretrained R3M (resnet50) from torchrl
- Optional neuromodulation blocks
- Behavior cloning head that consumes (image embedding + robot state) -> action
- Optional DistributedDataParallel (DDP) + rank0-only W&B logging

Run (single node, 4 GPUs):
  torchrun --standalone --nproc_per_node=4 train_r3m_neuromod_metaworld.py \
    --batch_size 128 --num_workers 4 --epochs 10 --sync_bn 0

Notes:
- If you enable --sync_bn 1, BN running stats are synchronized across ranks.
- Val MSE is globally reduced across ranks (correct). Per-task metrics are disabled under DDP
  (otherwise they'd reflect only each rank's shard unless you do gather).
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import wandb
from functools import partial


# -----------------------
# DDP helpers
# -----------------------

def ddp_is_on() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_setup():
    if not ddp_is_on():
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, rank, world_size, local_rank


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_ddp(m):
    return m.module if isinstance(m, DDP) else m


@torch.no_grad()
def ddp_allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# -----------------------
# Dataset loading
# -----------------------

def load_short_metaworld_dataset(data_root: str, tasks: Optional[List[str]] = None, image_size: int = 224):
    """Preferred: dataset-provided loader. Fallback: datasets.load_dataset."""
    try:
        from short_metaworld_loader import load_short_metaworld  # type: ignore
        return load_short_metaworld(data_root, tasks=tasks, image_size=image_size)
    except Exception as e:
        print(
            f"[WARN] Could not import/use short_metaworld_loader.load_short_metaworld ({e}). "
            "Falling back to datasets.load_dataset."
        )
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("hz1919810/short-metaworld-vla", split="train")

        class HFWrapper(Dataset):
            def __init__(self, hf_ds):
                self.ds = hf_ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                ex = self.ds[idx]
                return {
                    "image": ex["image"],
                    "state": ex["state"],
                    "action": ex["action"],
                    "task_name": ex.get("task_name", ""),
                    "prompt": ex.get("prompt", ""),
                }

        return HFWrapper(ds)


# -----------------------
# Preprocessing (match R3MTransform defaults: ImageNet norm + resize 244)
# -----------------------

_R3M_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_R3M_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_chw_float01(img: Any) -> torch.Tensor:
    """Convert PIL/numpy/torch image to float32 CHW in [0,1]."""
    try:
        from PIL import Image  # type: ignore

        if isinstance(img, Image.Image):
            img = np.array(img)
    except Exception:
        pass

    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
    elif torch.is_tensor(img):
        t = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)

    if t.ndim == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
        chw = t
    elif t.ndim == 3 and t.shape[-1] in (1, 3):
        chw = t.permute(2, 0, 1)
    else:
        raise ValueError(f"Unexpected image shape: {tuple(t.shape)}")

    chw = chw.contiguous()
    if chw.dtype == torch.uint8:
        chw = chw.float().div(255.0)
    else:
        chw = chw.float()
        if chw.max() > 1.5:
            chw = chw.div(255.0)

    return chw.clamp(0.0, 1.0)


def preprocess_r3m_batch(imgs_chw_01: torch.Tensor, out_size: int = 244) -> torch.Tensor:
    """Resize to out_size and apply ImageNet normalization."""
    x = imgs_chw_01
    if x.shape[-2:] != (out_size, out_size):
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)

    mean = _R3M_MEAN.to(x.device, dtype=x.dtype)
    std = _R3M_STD.to(x.device, dtype=x.dtype)
    return (x - mean) / std


def collate_metaworld(batch: List[Dict[str, Any]], r3m_size: int = 244) -> Dict[str, Any]:
    imgs_chw = [_to_chw_float01(ex["image"]) for ex in batch]
    imgs = torch.stack(imgs_chw, dim=0)
    imgs = preprocess_r3m_batch(imgs, out_size=r3m_size)

    states = torch.stack([torch.as_tensor(ex["state"], dtype=torch.float32) for ex in batch], dim=0)
    actions = torch.stack([torch.as_tensor(ex["action"], dtype=torch.float32) for ex in batch], dim=0)
    task_names = [ex.get("task_name", "") for ex in batch]
    prompts = [ex.get("prompt", "") for ex in batch]

    return {"image": imgs, "state": states, "action": actions, "task_name": task_names, "prompt": prompts}


# def collate_244(batch):
#     return collate_metaworld(batch, r3m_size=244)


# -----------------------
# Neuromodulation modules
# -----------------------

class AdaptiveWhitening(nn.Module):
    def __init__(
        self,
        num_channels: int,
        adaptation_rate: float = 0.01,
        strength: float = 0.3,
        max_gain: float = 0.5,
        train_momentum: float = 0.01,
        eval_momentum: float = 0.1,
        adapt: bool = True,
    ):
        super().__init__()
        self.C = num_channels
        self.eta = adaptation_rate
        self.max_gain = max_gain
        self.train_momentum = train_momentum
        self.eval_momentum = eval_momentum
        self.adapt = adapt

        eps = 1e-4
        s = torch.clamp(torch.tensor(strength), eps, 1 - eps)
        self.strength = nn.Parameter(torch.log(s / (1 - s)))

        W = torch.randn(num_channels, num_channels)
        W, _ = torch.linalg.qr(W)
        self.register_buffer("W", W)

        self.register_buffer("g", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the heavy linear algebra in fp32 for stability under AMP.
        x_in = x
        x_fp32 = x_in.float() if x_in.dtype in (torch.float16, torch.bfloat16) else x_in

        B, C, H, W = x_fp32.shape
        with torch.no_grad():
            x_flat = x_fp32.permute(0, 2, 3, 1).reshape(-1, C)
            G = torch.diag(F.softplus(self.g))
            M = torch.linalg.inv(torch.eye(C, device=x_fp32.device, dtype=x_fp32.dtype) + self.W @ G @ self.W.T)
            y_flat = x_flat @ M.T

            if self.adapt:
                z = y_flat @ self.W
                z_var = z.pow(2).mean(dim=0)
                momentum = self.train_momentum if self.training else self.eval_momentum
                self.running_var.lerp_(z_var.to(self.running_var.dtype), momentum)
                self.g.add_(self.running_var - 1, alpha=self.eta)
                self.g.clamp_(-self.max_gain, self.max_gain)

            y = y_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        s = torch.sigmoid(self.strength)
        y = y.to(dtype=x_in.dtype)
        return (1 - s) * x_in + s * y

    def reset(self):
        self.g.zero_()
        self.running_var.fill_(1.0)


class NAModulation(nn.Module):
    def __init__(
        self,
        num_channels: int,
        reduction: int = 4,
        train_momentum: float = 0.01,
        eval_momentum: float = 0.1,
        adapt: bool = True,
    ):
        super().__init__()
        self.train_momentum = train_momentum
        self.eval_momentum = eval_momentum
        self.adapt = adapt

        hidden = max(num_channels // reduction, 16)
        self.saliency_net = nn.Sequential(
            nn.Linear(num_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.saliency_net[2].bias, -4.0)

        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))
        self.baseline = nn.Parameter(torch.log(torch.exp(torch.ones(num_channels)) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # compute stats in fp32 under AMP
        x_fp32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        x_mean = x_fp32.mean(dim=[2, 3])
        x_var = x_fp32.var(dim=[2, 3], unbiased=False)

        if self.adapt:
            with torch.no_grad():
                momentum = self.train_momentum if self.training else self.eval_momentum
                self.running_mean.lerp_(x_mean.mean(0).to(self.running_mean.dtype), momentum)
                self.running_var.lerp_(x_var.mean(0).to(self.running_var.dtype), momentum)

        mean_dev = (x_mean - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        var_dev = (x_var - self.running_var) / (self.running_var + 1e-6)

        mean_dev = torch.clamp(mean_dev, -5.0, 5.0)
        var_dev = torch.clamp(var_dev, -5.0, 5.0)

        saliency = self.saliency_net(torch.cat([mean_dev, var_dev], dim=1))
        gain = F.softplus(self.baseline + saliency * mean_dev.abs())
        gain = torch.clamp(gain, max=3.0)

        gain = gain.to(dtype=x.dtype)
        return x * gain.view(B, C, 1, 1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)


class AChModulation(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 4, threshold: float = 0.3):
        super().__init__()
        hidden = max(num_channels // reduction, 16)
        self.reliability_net = nn.Sequential(
            nn.Linear(num_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.reliability_net[2].bias, 3.0)
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gap = x.mean(dim=[2, 3])
        reliability = self.reliability_net(gap)
        gate = torch.sigmoid((reliability - self.threshold) * 10.0)
        return x * (gate * reliability).view(B, C, 1, 1)

    def reset(self):
        pass


class NeuromodBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        use_whitening: bool = True,
        use_na: bool = True,
        use_ach: bool = True,
        adapt_whitening: bool = True,
        adapt_na: bool = True,
        na_first: bool = True,
        na_ach_parallel: bool = True,
    ):
        super().__init__()
        self.whitening = AdaptiveWhitening(num_channels, adapt=adapt_whitening) if use_whitening else None
        self.na = NAModulation(num_channels, adapt=adapt_na) if use_na else None
        self.ach = AChModulation(num_channels) if use_ach else None
        self.na_first = na_first
        self.na_ach_parallel = na_ach_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.whitening is not None:
            x = self.whitening(x)

        if self.na_ach_parallel:
            x_in = x
            outs = []
            if self.na is not None:
                outs.append(self.na(x_in))
            if self.ach is not None:
                outs.append(self.ach(x_in))
            if outs:
                x = sum(outs) / len(outs)
        else:
            if self.na_first:
                if self.na is not None:
                    x = self.na(x)
                if self.ach is not None:
                    x = self.ach(x)
            else:
                if self.ach is not None:
                    x = self.ach(x)
                if self.na is not None:
                    x = self.na(x)
        return x

    def reset(self):
        if self.whitening is not None:
            self.whitening.reset()
        if self.na is not None:
            self.na.reset()


NEUROMOD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": dict(use_whitening=False, use_na=False, use_ach=False, na_first=True, na_ach_parallel=False),
    "W": dict(use_whitening=True, use_na=False, use_ach=False, na_first=True, na_ach_parallel=False),
    "NA": dict(use_whitening=False, use_na=True, use_ach=False, na_first=True, na_ach_parallel=False),
    "ACh": dict(use_whitening=False, use_na=False, use_ach=True, na_first=True, na_ach_parallel=False),
    "W_NA": dict(use_whitening=True, use_na=True, use_ach=False, na_first=True, na_ach_parallel=False),
    "W_ACh": dict(use_whitening=True, use_na=False, use_ach=True, na_first=True, na_ach_parallel=False),
    "NA_ACh": dict(use_whitening=False, use_na=True, use_ach=True, na_first=True, na_ach_parallel=False),
    "ACh_NA": dict(use_whitening=False, use_na=True, use_ach=True, na_first=False, na_ach_parallel=False),
    "NA_ACh_parallel": dict(use_whitening=False, use_na=True, use_ach=True, na_first=False, na_ach_parallel=True),
    "W_NA_ACh": dict(use_whitening=True, use_na=True, use_ach=True, na_first=True, na_ach_parallel=False),
    "W_ACh_NA": dict(use_whitening=True, use_na=True, use_ach=True, na_first=False, na_ach_parallel=False),
    "W_NA_ACh_parallel": dict(use_whitening=True, use_na=True, use_ach=True, na_first=False, na_ach_parallel=True),
}


class R3MResNet50WithNeuromod(nn.Module):
    """Wrap a torchvision ResNet50 (from R3MTransform) and insert neuromod blocks."""

    def __init__(
        self,
        backbone_resnet50: nn.Module,
        use_neuromod: bool,
        neuromod_cfg: Dict[str, Any],
        whitening_only_at_start: bool = True,
        adapt_whitening: bool = True,
        adapt_na: bool = True,
    ):
        super().__init__()
        self.backbone = backbone_resnet50
        self.use_neuromod = use_neuromod

        if self.use_neuromod:
            cfg = dict(neuromod_cfg)
            use_whitening = bool(cfg.get("use_whitening", True))
            use_na = bool(cfg.get("use_na", True))
            use_ach = bool(cfg.get("use_ach", True))
            na_first = bool(cfg.get("na_first", True))
            na_ach_parallel = bool(cfg.get("na_ach_parallel", True))

            if use_whitening and whitening_only_at_start:
                self.neuromod0 = NeuromodBlock(64, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod1 = NeuromodBlock(256, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod2 = NeuromodBlock(512, False, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod3 = NeuromodBlock(1024, False, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod4 = NeuromodBlock(2048, False, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
            else:
                self.neuromod0 = NeuromodBlock(64, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod1 = NeuromodBlock(256, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod2 = NeuromodBlock(512, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod3 = NeuromodBlock(1024, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)
                self.neuromod4 = NeuromodBlock(2048, use_whitening, use_na, use_ach, adapt_whitening, adapt_na, na_first, na_ach_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if self.use_neuromod:
            x = self.neuromod0(x)
        x = self.backbone.maxpool(x)

        for block in self.backbone.layer1:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod1(x)

        for block in self.backbone.layer2:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod2(x)

        for block in self.backbone.layer3:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod3(x)

        for block in self.backbone.layer4:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def reset_neuromodulation(self):
        if not self.use_neuromod:
            return
        self.neuromod0.reset()
        self.neuromod1.reset()
        self.neuromod2.reset()
        self.neuromod3.reset()
        self.neuromod4.reset()

    def neuromod_parameters(self):
        if not self.use_neuromod:
            return []
        return (
            list(self.neuromod0.parameters())
            + list(self.neuromod1.parameters())
            + list(self.neuromod2.parameters())
            + list(self.neuromod3.parameters())
            + list(self.neuromod4.parameters())
        )


# -----------------------
# Policy (BC): (image emb + state) -> action
# -----------------------

class BCPolicy(nn.Module):
    def __init__(self, encoder: nn.Module, state_dim: int = 7, action_dim: int = 4, hidden: int = 512):
        super().__init__()
        self.encoder = encoder
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2048 + 768, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(image)
        s = self.state_proj(state)
        return self.head(torch.cat([z, s], dim=-1))


# -----------------------
# Training / evaluation
# -----------------------

@torch.no_grad()
def evaluate_bc(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """DDP-safe evaluation: returns globally reduced MSE."""

    model.eval()

    mse_sum = 0.0
    n = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        act = batch["action"].to(device, non_blocking=True)

        pred = model(img, state)
        loss = F.mse_loss(pred, act, reduction="none").mean(dim=1)

        mse_sum += loss.sum().item()
        n += loss.numel()

    # single allreduce AFTER local accumulation
    if dist.is_available() and dist.is_initialized():
        tn = torch.tensor([mse_sum, float(n)], device=device, dtype=torch.float64)
        tn = ddp_allreduce_sum(tn)
        mse_sum = float(tn[0].item())
        n = int(tn[1].item())

    return {"mse": mse_sum / max(n, 1)}


def train_bc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_head: float,
    lr_neuromod: float,
    use_neuromod: bool,
    bn_train: bool,
    amp: bool,
    max_train_samples: int,
    save_path: str,
    wandb_run=None,
    wandb_log_freq: int = 50,
):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    raw = unwrap_ddp(model)

    if distributed and max_train_samples > 0:
        max_train_samples = max_train_samples // world_size

    # Freeze backbone weights; allow BN running stats to update (if bn_train=True)
    for p in raw.encoder.backbone.parameters():
        p.requires_grad = False

    params = [{"params": raw.head.parameters(), "lr": lr_head}, {"params": raw.state_proj.parameters(), "lr": lr_head}]
    if use_neuromod:
        params.append({"params": raw.encoder.neuromod_parameters(), "lr": lr_neuromod})

    opt = torch.optim.AdamW(params, weight_decay=1e-4)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and device_type == "cuda"))

    best_val = float("inf")
    seen = 0
    global_step = 0

    for ep in range(1, epochs + 1):
        # DDP shuffling control
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(ep)

        model.train()

        # BN stats update behavior
        if bn_train:
            raw.encoder.backbone.train()
        else:
            raw.encoder.backbone.eval()

        # neuromod blocks train mode
        if use_neuromod:
            raw.encoder.neuromod0.train()
            raw.encoder.neuromod1.train()
            raw.encoder.neuromod2.train()
            raw.encoder.neuromod3.train()
            raw.encoder.neuromod4.train()

        raw.head.train()
        raw.state_proj.train()

        running = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"train ep {ep}/{epochs}", leave=False)

        for batch in pbar:
            if max_train_samples > 0 and seen >= max_train_samples:
                break

            img = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type, enabled=(amp and device_type == "cuda")):
                pred = model(img, state)
                loss = F.mse_loss(pred, act)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = img.shape[0]
            running += float(loss.item()) * bs
            count += bs
            seen += bs
            global_step += 1

            if wandb_run is not None and (global_step % max(1, wandb_log_freq) == 0):
                wandb.log(
                    {
                        "train/mse_step": float(loss.item()),
                        "train/seen": int(seen),
                        "epoch": int(ep),
                    },
                    step=global_step,
                )

            pbar.set_postfix(loss=running / max(count, 1), seen=seen)

        val = evaluate_bc(model, val_loader, device)

        if wandb_run is not None:
            wandb.log(
                {
                    "train/mse_epoch": float(running / max(count, 1)),
                    "val/mse": float(val["mse"]),
                    "epoch": int(ep),
                },
                step=global_step,
            )

        if is_main_process(rank):
            print(f"[ep {ep}] train_mse={running/max(count,1):.6f}  val_mse={val['mse']:.6f}")

        if val["mse"] < best_val:
            best_val = val["mse"]

            if is_main_process(rank):
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save({"model": raw.state_dict(), "best_val_mse": best_val}, save_path)
                print(f"  saved best -> {save_path}")

                if wandb_run is not None:
                    wandb.log({"val/best_mse": float(best_val)}, step=global_step)
                    wandb.save(save_path)

        # make sure ranks don't race ahead (especially before best-checkpoint load later)
        if distributed:
            dist.barrier()


# -----------------------
# Optional: in-context (few-shot) ridge regression eval per task
# -----------------------

@torch.no_grad()
def incontext_ridge_eval(
    policy: nn.Module,
    dataset: Dataset,
    device: torch.device,
    ctx: int = 32,
    qry: int = 128,
    ridge: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> Dict[str, float]:
    """Not DDP-parallelized (run it on rank0 only)."""

    rng = random.Random(seed)
    raw = unwrap_ddp(policy)

    by_task: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        ex = dataset[i]
        by_task.setdefault(str(ex.get("task_name", "")), []).append(i)

    if len(by_task) <= 1:
        return {"inctx_mse": float("nan"), "num_tasks": 0.0}

    def get_feat_state_act(indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        raw.eval()

        for start in range(0, len(indices), batch_size):
            idxs = indices[start : start + batch_size]
            batch = [dataset[j] for j in idxs]
            cb = collate_metaworld(batch, r3m_size=244)

            img = cb["image"].to(device)
            st = cb["state"].to(device)
            act = cb["action"].to(device)

            z = raw.encoder(img)
            s = raw.state_proj(st)
            xs.append(torch.cat([z, s], dim=-1))
            ys.append(act)

        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    task_mses = []
    for _, idxs in by_task.items():
        if len(idxs) < (ctx + qry):
            continue
        rng.shuffle(idxs)
        ctx_idx = idxs[:ctx]
        qry_idx = idxs[ctx : ctx + qry]

        Xc, Yc = get_feat_state_act(ctx_idx)
        Xq, Yq = get_feat_state_act(qry_idx)

        XtX = Xc.T @ Xc
        d = XtX.shape[0]
        XtX = XtX + ridge * torch.eye(d, device=XtX.device, dtype=XtX.dtype)
        W = torch.linalg.solve(XtX, Xc.T @ Yc)
        pred = Xq @ W
        task_mses.append(F.mse_loss(pred, Yq).item())

    return {
        "inctx_mse": float(np.mean(task_mses)) if task_mses else float("nan"),
        "num_tasks": float(len(task_mses)),
    }


# -----------------------
# Main
# -----------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/scratch/vjh9526/cv_fall_2025/data/short-metaworld-vla")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task_name list; empty=all")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--lr_neuromod", type=float, default=3e-4)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_frac", type=float, default=0.9)

    parser.add_argument("--max_train_samples", type=int, default=0, help="Cap training samples. 0=all.")
    parser.add_argument("--amp", type=int, default=1)

    parser.add_argument("--bn_train", type=int, default=1, help="1: backbone train (BN stats update). 0: backbone eval.")
    parser.add_argument("--sync_bn", type=int, default=0, help="1: SyncBatchNorm under DDP")

    parser.add_argument(
        "--r3m_cache",
        type=str,
        default="/scratch/vjh9526/cv_fall_2025/r3m_cache",
        help="Optional cache dir for R3M weights (download_path).",
    )

    parser.add_argument("--neuromod_config", type=str, default="W_NA_ACh_parallel", choices=list(NEUROMOD_CONFIGS.keys()))
    parser.add_argument("--no_neuromod", action="store_true", help="Disable neuromodulation (baseline R3M encoder).")

    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint to load.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/scratch/vjh9526/cv_fall_2025/latest_checkpoints_r3m/ckpt_r3m_neuromod.pt",
    )

    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--incontext_eval", action="store_true")
    parser.add_argument("--ctx", type=int, default=32)
    parser.add_argument("--qry", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)

    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="short-metaworld-vla")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_watch", type=int, default=0)
    parser.add_argument("--wandb_log_freq", type=int, default=50)

    args = parser.parse_args()

    distributed, rank, world_size, local_rank = ddp_setup()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Different seed per rank for dataloader randomness/etc.
    set_seed(args.seed + rank)

    if is_main_process(rank):
        print("device:", device)
        if distributed:
            print(f"DDP: rank {rank}/{world_size} local_rank={local_rank}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    base_ds = load_short_metaworld_dataset(args.data_root, tasks=tasks, image_size=224)

    # deterministic split across ranks
    n = len(base_ds)
    idxs = list(range(n))
    random.Random(args.seed).shuffle(idxs)
    split = int(args.train_frac * n)
    train_ds = Subset(base_ds, idxs[:split])
    val_ds = Subset(base_ds, idxs[split:])

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    collate_244 = partial(collate_metaworld, r3m_size=244)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_244,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_244,
    )

    # Load pretrained R3M ResNet50 via torchrl
    from torchrl.envs.transforms import R3MTransform

    r3m_t = R3MTransform(
        in_keys=["pixels"],
        model_name="resnet50",
        download=True,
        download_path=args.r3m_cache,
        size=244,
    )
    backbone = r3m_t[-1].convnet

    use_neuromod = (not args.no_neuromod) and (args.neuromod_config != "baseline")

    encoder = R3MResNet50WithNeuromod(
        backbone_resnet50=backbone,
        use_neuromod=use_neuromod,
        neuromod_cfg=NEUROMOD_CONFIGS[args.neuromod_config],
        whitening_only_at_start=True,
        adapt_whitening=True,
        adapt_na=True,
    ).to(device)

    policy = BCPolicy(encoder=encoder, state_dim=7, action_dim=4).to(device)

    # DDP wrapping
    if distributed:
        if args.sync_bn and device.type == "cuda":
            policy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy)
        policy = DDP(policy, device_ids=[local_rank] if device.type == "cuda" else None)

    # W&B only on rank0
    run = None
    if args.wandb and is_main_process(rank):
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=vars(args),
        )
        if args.wandb_watch:
            wandb.watch(unwrap_ddp(policy), log="all", log_freq=200)

    # Load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        unwrap_ddp(policy).load_state_dict(sd, strict=True)
        if is_main_process(rank):
            print(f"Loaded checkpoint: {args.ckpt}")

    # Eval-only
    if args.eval_only:
        val = evaluate_bc(policy, val_loader, device)
        if is_main_process(rank):
            print("VAL:", val)

        if args.incontext_eval and is_main_process(rank):
            inc = incontext_ridge_eval(policy, val_ds, device=device, ctx=args.ctx, qry=args.qry, ridge=args.ridge, seed=args.seed)
            print("IN-CONTEXT (ridge) VAL:", inc)

        if run is not None:
            run.finish()

        ddp_cleanup()
        return

    # Train
    train_bc(
        model=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_neuromod=args.lr_neuromod,
        use_neuromod=use_neuromod,
        bn_train=bool(args.bn_train),
        amp=bool(args.amp),
        max_train_samples=args.max_train_samples,
        save_path=args.save_path,
        wandb_run=run,
        wandb_log_freq=args.wandb_log_freq,
    )

    # Ensure checkpoint exists before loading on all ranks
    if distributed:
        dist.barrier()

    # Load best & eval
    ckpt = torch.load(args.save_path, map_location="cpu")
    unwrap_ddp(policy).load_state_dict(ckpt["model"], strict=True)
    val = evaluate_bc(policy, val_loader, device)

    if is_main_process(rank):
        print("BEST VAL:", val)
        if args.incontext_eval:
            inc = incontext_ridge_eval(policy, val_ds, device=device, ctx=args.ctx, qry=args.qry, ridge=args.ridge, seed=args.seed)
            print("IN-CONTEXT (ridge) VAL:", inc)

    if run is not None:
        run.finish()

    ddp_cleanup()


if __name__ == "__main__":
    main()
