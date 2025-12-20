import argparse
import os
import json
import random
from typing import Optional, Dict, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.utils.data.dataloader as dl
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from pycocotools.coco import COCO
except Exception:
    COCO = None


# -------------------------------------------------------------------
# Patch DataLoader __del__ to silence "can only test a child process"
# -------------------------------------------------------------------
_orig_del = dl._MultiProcessingDataLoaderIter.__del__
def _safe_del(self):
    try:
        _orig_del(self)
    except AssertionError as e:
        if "can only test a child process" in str(e):
            return
        raise
dl._MultiProcessingDataLoaderIter.__del__ = _safe_del


# ============================================================
# Neuromodulation building blocks (token-space versions)
# ============================================================

class AdaptiveWhitening(nn.Module):
    """
    Whitening module expecting (B, C, H, W).
    Computes a random-projection whitening transform with adaptive gains.
    """
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
        self.C = int(num_channels)
        self.eta = float(adaptation_rate)
        self.max_gain = float(max_gain)
        self.train_momentum = float(train_momentum)
        self.eval_momentum = float(eval_momentum)
        self.adapt = bool(adapt)

        eps = 1e-4
        s = torch.clamp(torch.tensor(float(strength)), eps, 1 - eps)
        self.strength = nn.Parameter(torch.log(s / (1 - s)))

        W = torch.randn(self.C, self.C)
        W, _ = torch.linalg.qr(W)
        self.register_buffer("W", W)

        self.register_buffer("g", torch.zeros(self.C))
        self.register_buffer("running_var", torch.ones(self.C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.C, f"AdaptiveWhitening C mismatch: got {C}, expected {self.C}"

        x_dtype = x.dtype
        x_fp32 = x.float()

        with torch.no_grad():
            x_flat = x_fp32.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

            G = torch.diag(F.softplus(self.g))
            I = torch.eye(C, device=x.device, dtype=x_flat.dtype)
            M = torch.linalg.inv(I + self.W @ G @ self.W.T)  # (C, C)
            y_flat = x_flat @ M.T

            if self.adapt:
                z = y_flat @ self.W
                z_var = z.pow(2).mean(dim=0)

                momentum = self.train_momentum if self.training else self.eval_momentum
                self.running_var.lerp_(z_var, momentum)
                self.g.add_(self.running_var - 1, alpha=self.eta)
                self.g.clamp_(-self.max_gain, self.max_gain)

            y = y_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        strength = torch.sigmoid(self.strength).to(x_fp32.dtype)
        out = (1 - strength) * x_fp32 + strength * y
        return out.to(x_dtype)

    def reset(self):
        self.g.zero_()
        self.running_var.fill_(1.0)


class GroupedAdaptiveWhitening(nn.Module):
    """
    Grouped whitening for ViT patch-projection output (C is large).
    Splits channels into groups and applies AdaptiveWhitening per group.
    """
    def __init__(self, num_channels: int, group_size: int = 64, **aw_kwargs):
        super().__init__()
        C = int(num_channels)
        group_size = int(group_size)
        if C % group_size != 0:
            raise ValueError(f"group_size must divide num_channels: {group_size} !| {C}")
        self.C = C
        self.group_size = group_size
        self.num_groups = C // group_size
        self.groups = nn.ModuleList([
            AdaptiveWhitening(group_size, **aw_kwargs) for _ in range(self.num_groups)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.C
        xs = torch.split(x, self.group_size, dim=1)
        ys = [g(xg) for g, xg in zip(self.groups, xs)]
        return torch.cat(ys, dim=1)

    def reset(self):
        for g in self.groups:
            g.reset()


class PatchTokenBatchNorm1d(nn.Module):
    """
    BN over channels for PATCH TOKENS ONLY.
    Input: (B, N, D). Only tokens in [1+R:] are normalized.
    """
    def __init__(self, dim: int, num_register_tokens: int, momentum: float = 0.1):
        super().__init__()
        self.R = int(num_register_tokens)
        self.bn = nn.BatchNorm1d(dim, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        if x.ndim != 3:
            raise ValueError("PatchTokenBatchNorm1d expects (B, N, D)")
        B, N, D = x.shape
        if N <= 1 + self.R:
            return x

        cls = x[:, :1, :]
        regs = x[:, 1:1 + self.R, :] if self.R > 0 else x[:, 1:1, :]
        patches = x[:, 1 + self.R:, :]  # (B, P, D)

        patches_flat = patches.reshape(-1, D)
        patches_bn = self.bn(patches_flat).reshape(B, -1, D)

        if self.R > 0:
            return torch.cat([cls, regs, patches_bn], dim=1)
        return torch.cat([cls, patches_bn], dim=1)


# -----------------------------
# Shared-weight NA/ACh in token space
# -----------------------------

class SharedNAWeights(nn.Module):
    """
    Weight-tied NA network with a site embedding.
    Inputs are channel deviations (mean_dev, var_dev) and a learned site embedding.
    """
    def __init__(self, dim: int, num_sites: int, site_dim: int = 32, reduction: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.site_emb = nn.Embedding(int(num_sites), int(site_dim))

        hidden = max(self.dim // int(reduction), 16)
        in_dim = 2 * self.dim + site_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.dim, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[2].bias, -4.0)

        self.baseline = nn.Parameter(torch.log(torch.exp(torch.ones(self.dim)) - 1))

    def forward(self, mean_dev: torch.Tensor, var_dev: torch.Tensor, site_id: torch.Tensor) -> torch.Tensor:
        emb = self.site_emb(site_id)  # (B, site_dim)
        x = torch.cat([mean_dev, var_dev, emb], dim=1)
        sal = self.net(x)
        return sal


class NAState(nn.Module):
    """
    Per-site state (running mean/var) + access to shared NA weights.
    Computes a channel-wise gain to apply to tokens.
    """
    def __init__(
        self,
        dim: int,
        shared: SharedNAWeights,
        site_index: int,
        train_momentum: float = 0.01,
        eval_momentum: float = 0.1,
        adapt: bool = True,
        max_gain: float = 3.0,
        clamp_dev: float = 5.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.shared = shared
        self.site_index = int(site_index)
        self.train_momentum = float(train_momentum)
        self.eval_momentum = float(eval_momentum)
        self.adapt = bool(adapt)
        self.max_gain = float(max_gain)
        self.clamp_dev = float(clamp_dev)

        self.register_buffer("running_mean", torch.zeros(self.dim))
        self.register_buffer("running_var", torch.ones(self.dim))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, P, D)
        B, P, D = patch_tokens.shape
        assert D == self.dim

        x_mean = patch_tokens.mean(dim=1)                 # (B, D)
        x_var = patch_tokens.var(dim=1, unbiased=False)   # (B, D)

        if self.adapt:
            with torch.no_grad():
                m = self.train_momentum if self.training else self.eval_momentum
                self.running_mean.lerp_(x_mean.mean(0), m)
                self.running_var.lerp_(x_var.mean(0), m)

        mean_dev = (x_mean - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        var_dev = (x_var - self.running_var) / (self.running_var + 1e-6)

        mean_dev = torch.clamp(mean_dev, -self.clamp_dev, self.clamp_dev)
        var_dev  = torch.clamp(var_dev,  -self.clamp_dev, self.clamp_dev)

        site_id = torch.full((B,), self.site_index, device=patch_tokens.device, dtype=torch.long)
        sal = self.shared(mean_dev, var_dev, site_id)  # (B, D)

        gain = F.softplus(self.shared.baseline + sal * mean_dev.abs())
        gain = torch.clamp(gain, max=self.max_gain)
        return gain  # (B, D)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)


class SharedAChWeights(nn.Module):
    """
    Weight-tied ACh network with site embedding.
    """
    def __init__(self, dim: int, num_sites: int, site_dim: int = 32, reduction: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.site_emb = nn.Embedding(int(num_sites), int(site_dim))

        hidden = max(self.dim // int(reduction), 16)
        in_dim = self.dim + site_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.dim, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[2].bias, 3.0)

        self.threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, x_gap: torch.Tensor, site_id: torch.Tensor) -> torch.Tensor:
        emb = self.site_emb(site_id)
        z = torch.cat([x_gap, emb], dim=1)
        rel = self.net(z)
        gate = torch.sigmoid((rel - self.threshold) * 10.0)
        return gate * rel  # (B, D)


class PatchNeuromodSite(nn.Module):
    """
    Applies optional BN (patch tokens only) + NA and/or ACh to PATCH tokens, leaving CLS+registers unchanged.
    BN is used ONLY on MLP/FFN residual outputs (configured at hook install time).
    """
    def __init__(
        self,
        dim: int,
        num_register_tokens: int,
        na_state: Optional[NAState],
        ach_shared: Optional[SharedAChWeights],
        site_index: int,
        use_bn: bool,
        na_ach_parallel: bool = True,
        na_first: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.R = int(num_register_tokens)
        self.na_state = na_state
        self.ach_shared = ach_shared
        self.site_index = int(site_index)
        self.na_ach_parallel = bool(na_ach_parallel)
        self.na_first = bool(na_first)
        self.bn = PatchTokenBatchNorm1d(dim, num_register_tokens) if use_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) - residual branch output
        if self.bn is not None:
            x = self.bn(x)

        B, N, D = x.shape
        if N <= 1 + self.R:
            return x

        cls = x[:, :1, :]
        regs = x[:, 1:1 + self.R, :] if self.R > 0 else x[:, 1:1, :]
        patches = x[:, 1 + self.R:, :]  # (B, P, D)

        outs: List[torch.Tensor] = []

        if self.na_state is not None:
            gain = self.na_state(patches)  # (B, D)
            outs.append(patches * gain.unsqueeze(1))

        if self.ach_shared is not None:
            x_gap = patches.mean(dim=1)
            site_id = torch.full((B,), self.site_index, device=x.device, dtype=torch.long)
            rel = self.ach_shared(x_gap, site_id)  # (B, D)
            outs.append(patches * rel.unsqueeze(1))

        if not outs:
            patches_out = patches
        elif self.na_ach_parallel:
            patches_out = sum(outs) / float(len(outs))
        else:
            patches_out = patches
            if self.na_first:
                if self.na_state is not None:
                    patches_out = patches_out * self.na_state(patches_out).unsqueeze(1)
                if self.ach_shared is not None:
                    x_gap = patches_out.mean(dim=1)
                    site_id = torch.full((B,), self.site_index, device=x.device, dtype=torch.long)
                    patches_out = patches_out * self.ach_shared(x_gap, site_id).unsqueeze(1)
            else:
                if self.ach_shared is not None:
                    x_gap = patches_out.mean(dim=1)
                    site_id = torch.full((B,), self.site_index, device=x.device, dtype=torch.long)
                    patches_out = patches_out * self.ach_shared(x_gap, site_id).unsqueeze(1)
                if self.na_state is not None:
                    patches_out = patches_out * self.na_state(patches_out).unsqueeze(1)

        if self.R > 0:
            return torch.cat([cls, regs, patches_out], dim=1)
        return torch.cat([cls, patches_out], dim=1)


class CLSNeuromodSite(nn.Module):
    """
    Per-block CLS modulation conditioned on pooled patch stats.
    Uses BOTH NA + ACh always (when enabled).
    concat([na_gain, ach_gain]) -> linear -> sigmoid gate.
    """
    def __init__(
        self,
        dim: int,
        num_register_tokens: int,
        na_state: NAState,
        ach_shared: SharedAChWeights,
        site_index: int,
    ):
        super().__init__()
        self.dim = int(dim)
        self.R = int(num_register_tokens)
        self.na_state = na_state
        self.ach_shared = ach_shared
        self.site_index = int(site_index)
        self.combine = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) - block output
        B, N, D = x.shape
        if N <= 1 + self.R:
            return x

        cls = x[:, 0, :]  # (B, D)
        regs = x[:, 1:1 + self.R, :] if self.R > 0 else None
        patches = x[:, 1 + self.R:, :]  # (B, P, D)

        na_gain = self.na_state(patches)  # (B, D)
        site_id = torch.full((B,), self.site_index, device=x.device, dtype=torch.long)
        ach_gain = self.ach_shared(patches.mean(dim=1), site_id)  # (B, D)

        g = torch.sigmoid(self.combine(torch.cat([na_gain, ach_gain], dim=1)))
        cls_out = cls * (1.0 + 0.5 * g)

        if self.R > 0:
            return torch.cat([cls_out.unsqueeze(1), regs, patches], dim=1)
        return torch.cat([cls_out.unsqueeze(1), patches], dim=1)


# ============================================================
# Datasets
# ============================================================

class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx: Dict[str, int] = {}

        classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}


class COCO2017SingleLabelDataset(Dataset):
    """
    COCO images with a single label per image derived from instance annotations.
    Dominant category by total annotated area (sum of 'area').
    """
    def __init__(
        self,
        coco_root: str,
        split: str = "val2017",
        transform: Optional[transforms.Compose] = None,
        max_images: int = 5000,
        seed: int = 42,
        single_category_only: bool = False,
        min_instances: int = 1,
    ):
        if COCO is None:
            raise ImportError("pycocotools is required. Install with: pip install pycocotools")

        self.transform = transform
        images_dir = os.path.join(coco_root, split)
        ann_file = os.path.join(coco_root, "annotations", f"instances_{split}.json")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Missing COCO images dir: {images_dir}")
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"Missing COCO annotation file: {ann_file}")

        coco = COCO(ann_file)
        img_ids = coco.getImgIds()

        items = []
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            if len(anns) < min_instances:
                continue

            area_by_cat: Dict[int, float] = {}
            for a in anns:
                cid = int(a["category_id"])
                area_by_cat[cid] = area_by_cat.get(cid, 0.0) + float(a.get("area", 0.0))

            cats = list(area_by_cat.keys())
            if single_category_only and len(cats) != 1:
                continue

            dom_cid = max(area_by_cat.items(), key=lambda kv: kv[1])[0]
            img_info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(images_dir, img_info["file_name"])
            if os.path.isfile(img_path):
                items.append((img_path, dom_cid))

        rng = np.random.RandomState(seed)
        rng.shuffle(items)
        if max_images is not None and len(items) > max_images:
            items = items[:max_images]

        cat_ids = sorted({cid for _, cid in items})
        self.catid_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.items = [(p, self.catid_to_idx[cid]) for (p, cid) in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


class RemapLabelsDataset(Dataset):
    """
    Wraps a dataset that returns {"image": ..., "label": ...} and remaps labels via label_map.
    """
    def __init__(self, base: Dataset, label_map: Dict[int, int], unknown_label: int = -1):
        self.base = base
        self.label_map = dict(label_map)
        self.unknown_label = int(unknown_label)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        y = int(item["label"])
        item["label"] = self.label_map.get(y, self.unknown_label)
        return item


# ============================================================
# Metrics
# ============================================================

def compute_geometry_metrics(feats: torch.Tensor, labels: torch.Tensor):
    X = feats.numpy()
    y = labels.numpy()
    classes = np.unique(y)

    means = np.stack([X[y == c].mean(axis=0) for c in classes], axis=0)

    intra_dists = []
    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] > 1:
            diffs = Xc[:, None, :] - Xc[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(-1))
            intra_dists.append(dists[np.triu_indices_from(dists, k=1)].mean())
    d_intra = float(np.mean(intra_dists)) if intra_dists else np.nan

    diffs_means = means[:, None, :] - means[None, :, :]
    dists_means = np.sqrt((diffs_means ** 2).sum(-1))
    inter_vals = dists_means[np.triu_indices_from(dists_means, k=1)]
    d_inter = float(inter_vals.mean())

    sep = d_inter / d_intra if d_intra > 0 else np.nan
    return d_intra, d_inter, sep


def compute_covariance_metrics(feats: torch.Tensor, top_k: int = 10):
    X = feats.numpy()
    X = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X, rowvar=False)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    D = cov.shape[0]
    I = np.eye(D, dtype=cov.dtype)
    frob = np.linalg.norm(cov - I, ord="fro")
    return eigvals[:top_k], frob


def ood_metrics_from_scores(id_scores: np.ndarray, ood_scores: np.ndarray):
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    threshs = np.sort(scores)[::-1]
    tprs, fprs = [], []
    for t in threshs:
        y_pred = scores >= t
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    idx = (np.abs(tprs - 0.95)).argmin()
    fpr95 = float(fprs[idx])
    return float(auroc), float(aupr), fpr95


def few_shot_eval(
    feats: torch.Tensor,
    labels: torch.Tensor,
    shots_list=(1, 3, 5, 10, 15),
    num_epochs: int = 100,
    lr: float = 1e-2,
    device: str = "cuda",
    desc_prefix: str = "few-shot",
):
    X = feats.to(device)
    y = labels.to(device)
    classes = torch.unique(y)
    results = {}

    for k in shots_list:
        print(f"[{desc_prefix}] k={k} shots (classes={len(classes)})")

        support_idx, query_idx = [], []
        for c in classes:
            idx_c = torch.nonzero(y == c, as_tuple=False).view(-1)
            if len(idx_c) < 2:
                continue
            idx_perm = idx_c[torch.randperm(len(idx_c))]
            k_eff = min(int(k), len(idx_perm) - 1)
            support_idx.append(idx_perm[:k_eff])
            query_idx.append(idx_perm[k_eff:])

        support_idx = torch.cat(support_idx)
        query_idx = torch.cat(query_idx)

        X_support, y_support = X[support_idx], y[support_idx]
        X_query, y_query = X[query_idx], y[query_idx]

        clf = nn.Linear(X.shape[1], len(classes)).to(device)
        class_to_new = {int(c.item()): i for i, c in enumerate(classes)}
        y_support_mapped = torch.tensor([class_to_new[int(c.item())] for c in y_support], device=device)

        opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        clf.train()
        for _ in range(num_epochs):
            opt.zero_grad()
            logits = clf(X_support)
            loss = loss_fn(logits, y_support_mapped)
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            preds_q = clf(X_query).argmax(dim=1)
            y_query_mapped = torch.tensor([class_to_new[int(c.item())] for c in y_query], device=device)
            acc = (preds_q == y_query_mapped).float().mean().item()

        print(f"[{desc_prefix}] k={k} shots accuracy: {acc:.4f}")
        results[int(k)] = float(acc)
        del clf

    return results


# ============================================================
# ViT wrapper + neuromod hooks
# ============================================================

def _auto_group_size(hidden_size: int) -> int:
    for gs in (64, 96, 48, 32, 128):
        if hidden_size % gs == 0:
            return gs
    return hidden_size


def _find_patch_conv2d(model: nn.Module, patch_size: int) -> Tuple[str, nn.Conv2d]:
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (m.in_channels == 3
                and isinstance(m.kernel_size, tuple) and m.kernel_size[0] == patch_size and m.kernel_size[1] == patch_size
                and isinstance(m.stride, tuple) and m.stride[0] == patch_size and m.stride[1] == patch_size):
                candidates.append((name, m))
    if not candidates:
        raise RuntimeError("Could not find patch projection Conv2d. Please inspect backbone.named_modules().")
    candidates.sort(key=lambda x: len(x[0]))
    return candidates[0]


def _find_transformer_blocks(backbone: nn.Module, num_layers: int) -> nn.ModuleList:
    if hasattr(backbone, "encoder"):
        enc = backbone.encoder
        for attr in ("layer", "layers", "blocks"):
            if hasattr(enc, attr) and isinstance(getattr(enc, attr), nn.ModuleList):
                blocks = getattr(enc, attr)
                if len(blocks) == num_layers:
                    return blocks
    for m in backbone.modules():
        if isinstance(m, nn.ModuleList) and len(m) == num_layers:
            return m
    raise RuntimeError("Could not locate transformer blocks ModuleList; inspect the backbone structure.")


def _get_submodule(block: nn.Module, preferred_names: Tuple[str, ...], contains: Tuple[str, ...]) -> nn.Module:
    for n in preferred_names:
        if hasattr(block, n):
            return getattr(block, n)
    for name, child in block.named_children():
        lname = name.lower()
        if any(s in lname for s in contains):
            return child
    raise RuntimeError(f"Could not find submodule in block. Tried {preferred_names} / contains={contains}.")


class DINOv3ViTWithNeuromod(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        num_classes: int,
        use_whitening: bool,
        use_patch_na: bool,
        use_patch_ach: bool,
        na_ach_parallel: bool = True,
        na_first: bool = True,
        adapt_whitening: bool = True,
        adapt_na: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.config = self.backbone.config

        self.patch_size = int(getattr(self.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.config, "num_register_tokens", 0))
        self.hidden_size = int(getattr(self.config, "hidden_size", 768))
        self.num_layers = int(getattr(self.config, "num_hidden_layers", 12))

        # Head
        self.rep_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Whitening (only at patch conv)
        self.use_whitening = bool(use_whitening)
        self._patch_grid_last: Optional[torch.Tensor] = None
        if self.use_whitening:
            gs = _auto_group_size(self.hidden_size)
            self.whitening = GroupedAdaptiveWhitening(
                self.hidden_size,
                group_size=gs,
                adapt=adapt_whitening,
            )
        else:
            self.whitening = None

        # Patch neuromod (attn + mlp)
        self.use_patch_na = bool(use_patch_na)
        self.use_patch_ach = bool(use_patch_ach)
        self.use_patch_neuromod = self.use_patch_na or self.use_patch_ach

        self.num_patch_sites = 2 * self.num_layers

        if self.use_patch_na:
            self.patch_na_shared = SharedNAWeights(self.hidden_size, num_sites=self.num_patch_sites)
            self.patch_na_states = nn.ModuleList([
                NAState(self.hidden_size, self.patch_na_shared, site_index=i, adapt=adapt_na)
                for i in range(self.num_patch_sites)
            ])
        else:
            self.patch_na_shared = None
            self.patch_na_states = nn.ModuleList([])

        if self.use_patch_ach:
            self.patch_ach_shared = SharedAChWeights(self.hidden_size, num_sites=self.num_patch_sites)
        else:
            self.patch_ach_shared = None

        self.patch_sites = nn.ModuleList()
        for i in range(self.num_patch_sites):
            is_mlp_site = (i % 2 == 1)
            na_state = self.patch_na_states[i] if self.use_patch_na else None
            ach_shared = self.patch_ach_shared if self.use_patch_ach else None

            self.patch_sites.append(
                PatchNeuromodSite(
                    dim=self.hidden_size,
                    num_register_tokens=self.num_register_tokens,
                    na_state=na_state,
                    ach_shared=ach_shared,
                    site_index=i,
                    use_bn=is_mlp_site,           # BN only for FFN/MLP residual output
                    na_ach_parallel=na_ach_parallel,
                    na_first=na_first,
                )
            )

        # CLS neuromod (per block; always NA + ACh when enabled)
        self.use_cls_neuromod = self.use_patch_neuromod
        if self.use_cls_neuromod:
            self.num_cls_sites = self.num_layers
            self.cls_na_shared = SharedNAWeights(self.hidden_size, num_sites=self.num_cls_sites)
            self.cls_na_states = nn.ModuleList([
                NAState(self.hidden_size, self.cls_na_shared, site_index=i, adapt=adapt_na)
                for i in range(self.num_cls_sites)
            ])
            self.cls_ach_shared = SharedAChWeights(self.hidden_size, num_sites=self.num_cls_sites)
            self.cls_sites = nn.ModuleList([
                CLSNeuromodSite(
                    dim=self.hidden_size,
                    num_register_tokens=self.num_register_tokens,
                    na_state=self.cls_na_states[i],
                    ach_shared=self.cls_ach_shared,
                    site_index=i,
                )
                for i in range(self.num_cls_sites)
            ])
        else:
            self.cls_na_shared = None
            self.cls_na_states = nn.ModuleList([])
            self.cls_ach_shared = None
            self.cls_sites = nn.ModuleList([])

        # Hooks
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

        self.to(device)

    def _install_hooks(self):
        patch_conv_name, patch_conv = _find_patch_conv2d(self.backbone, self.patch_size)

        def patch_conv_hook(_m, _inp, out):
            # out: (B, D, H', W')
            if self.whitening is None:
                self._patch_grid_last = out
                return out
            w_out = self.whitening(out)
            self._patch_grid_last = w_out
            return w_out

        self._hooks.append(patch_conv.register_forward_hook(patch_conv_hook))
        print(f"[hooks] patch projection conv: {patch_conv_name}")

        blocks = _find_transformer_blocks(self.backbone, self.num_layers)

        for b_idx, block in enumerate(blocks):
            attn = _get_submodule(block, ("attention", "attn", "self_attn"), ("attn", "attention"))
            mlp  = _get_submodule(block, ("mlp", "ffn", "feed_forward", "intermediate"), ("mlp", "ffn"))

            attn_site = self.patch_sites[2 * b_idx]
            mlp_site  = self.patch_sites[2 * b_idx + 1]

            def _make_branch_hook(site_module: PatchNeuromodSite):
                def _hook(_m, _inp, out):
                    if not self.use_patch_neuromod:
                        return out
                    if isinstance(out, tuple):
                        y0 = site_module(out[0])
                        return (y0,) + out[1:]
                    return site_module(out)
                return _hook

            self._hooks.append(attn.register_forward_hook(_make_branch_hook(attn_site)))
            self._hooks.append(mlp.register_forward_hook(_make_branch_hook(mlp_site)))

            if self.use_cls_neuromod:
                cls_site = self.cls_sites[b_idx]
                def _make_block_hook(site_module: CLSNeuromodSite):
                    def _hook(_m, _inp, out):
                        if isinstance(out, tuple):
                            y0 = site_module(out[0])
                            return (y0,) + out[1:]
                        return site_module(out)
                    return _hook
                self._hooks.append(block.register_forward_hook(_make_block_hook(cls_site)))

        print(f"[hooks] installed: whitening={self.use_whitening}, patch_neuromod={self.use_patch_neuromod}, cls_neuromod={self.use_cls_neuromod}")

    def reset_neuromod(self):
        if self.whitening is not None:
            self.whitening.reset()
        for s in self.patch_na_states:
            s.reset()
        for s in self.cls_na_states:
            s.reset()

    @torch.no_grad()
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values, return_dict=True)
        x = out.last_hidden_state  # (B, 1+R+P, D)

        R = self.num_register_tokens
        cls = x[:, 0, :]
        patches = x[:, 1 + R:, :] if x.shape[1] > 1 + R else x[:, 1:1, :]
        patch_mean = patches.mean(dim=1) if patches.numel() > 0 else torch.zeros_like(cls)

        rep = self.rep_proj(torch.cat([cls, patch_mean], dim=1))
        return rep

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        rep = self.forward_features(pixel_values)
        return self.classifier(rep)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.rep_proj.parameters()) + list(self.classifier.parameters())
        if self.use_whitening:
            params += list(self.whitening.parameters())
        if self.use_patch_neuromod:
            params += list(self.patch_sites.parameters())
            if self.use_patch_na:
                params += list(self.patch_na_shared.parameters())
            if self.use_patch_ach:
                params += list(self.patch_ach_shared.parameters())
        if self.use_cls_neuromod:
            params += list(self.cls_sites.parameters())
            params += list(self.cls_na_shared.parameters())
            params += list(self.cls_ach_shared.parameters())
        return params

    @torch.no_grad()
    def pop_last_patch_grid(self) -> Optional[torch.Tensor]:
        t = self._patch_grid_last
        self._patch_grid_last = None
        return t


# ============================================================
# Train / Eval
# ============================================================

def freeze_backbone(model: DINOv3ViTWithNeuromod):
    for p in model.backbone.parameters():
        p.requires_grad = False


def train_model(
    model: DINOv3ViTWithNeuromod,
    train_loader: DataLoader,
    num_epochs: int,
    lr_head: float,
    lr_neuromod: float,
    device: str,
):
    head_params = list(model.rep_proj.parameters()) + list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}

    all_trainable = model.get_trainable_params()
    neuromod_params = [p for p in all_trainable if id(p) not in head_ids]

    param_groups = [{"params": head_params, "lr": lr_head}]
    if neuromod_params:
        param_groups.append({"params": neuromod_params, "lr": lr_neuromod})

    optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup_epochs = max(1, num_epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler_cos], milestones=[warmup_epochs]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        model.train()
        model.backbone.eval()  # keep backbone fixed; neuromod/BN live outside backbone so remain train

        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - loss={total_loss/total:.4f}, acc={correct/total:.4f}")


@torch.no_grad()
def extract_features_and_logits(model: DINOv3ViTWithNeuromod, loader: DataLoader, device: str, desc: str):
    model.eval()
    feats_list, logits_list, labels_list = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        feats = model.forward_features(images)
        logits = model.classifier(feats)

        feats_list.append(feats.cpu())
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())

    return torch.cat(feats_list, 0), torch.cat(logits_list, 0), torch.cat(labels_list, 0)


@torch.no_grad()
def extract_stage0_features(
    model: DINOv3ViTWithNeuromod,
    loader: DataLoader,
    device: str,
    max_total_samples: int = 200_000,
    desc: str = "stage0",
):
    model.eval()
    feats_list = []
    collected = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        if collected >= max_total_samples:
            break

        images = batch["image"].to(device)
        _ = model(images)  # run forward to populate patch grid capture via hook
        grid = model.pop_last_patch_grid()  # (B, D, H', W') AFTER whitening if enabled
        if grid is None:
            continue

        B, D, H, W = grid.shape
        x_flat = grid.permute(0, 2, 3, 1).reshape(-1, D)

        remaining = max_total_samples - collected
        if x_flat.size(0) > remaining:
            idx = torch.randperm(x_flat.size(0), device=x_flat.device)[:remaining]
            x_flat = x_flat[idx]

        feats_list.append(x_flat.cpu())
        collected += x_flat.size(0)

    if not feats_list:
        return torch.empty(0, 0)
    return torch.cat(feats_list, 0)


def eval_ood_set(
    model: DINOv3ViTWithNeuromod,
    id_logits: torch.Tensor,
    ood_loader: DataLoader,
    device: str,
    prefix: str,
):
    feats, logits, labels = extract_features_and_logits(model, ood_loader, device, desc=f"{prefix} feats/logits")

    id_probs = F.softmax(id_logits, dim=1).numpy()
    ood_probs = F.softmax(logits, dim=1).numpy()
    id_msp = id_probs.max(axis=1)
    ood_msp = ood_probs.max(axis=1)

    id_energy = torch.logsumexp(id_logits, dim=1).numpy()
    ood_energy = torch.logsumexp(logits, dim=1).numpy()

    msp_auroc, msp_aupr, msp_fpr95 = ood_metrics_from_scores(id_msp, ood_msp)
    eng_auroc, eng_aupr, eng_fpr95 = ood_metrics_from_scores(id_energy, ood_energy)

    d_intra, d_inter, sep = compute_geometry_metrics(feats, labels)
    _, frob = compute_covariance_metrics(feats)

    stage0 = extract_stage0_features(model, ood_loader, device, desc=f"stage0 {prefix}")
    _, frob_stage0 = compute_covariance_metrics(stage0) if stage0.numel() else (None, float("nan"))

    fewshot = few_shot_eval(feats, labels, device=device, desc_prefix=f"{prefix} few-shot")

    out = {
        f"{prefix}_MSP_AUROC": msp_auroc,
        f"{prefix}_MSP_AUPR": msp_aupr,
        f"{prefix}_MSP_FPR95": msp_fpr95,
        f"{prefix}_ENG_AUROC": eng_auroc,
        f"{prefix}_ENG_AUPR": eng_aupr,
        f"{prefix}_ENG_FPR95": eng_fpr95,
        f"{prefix}_d_intra": d_intra,
        f"{prefix}_d_inter": d_inter,
        f"{prefix}_sep": sep,
        f"{prefix}_frob_cov": frob,
        f"{prefix}_frob_cov_stage0": frob_stage0,
    }
    for k, v in fewshot.items():
        out[f"{prefix}_fewshot_{k}"] = v
    return out


# ============================================================
# Main
# ============================================================

EXPERIMENT_CONFIGS = {
    "baseline": dict(use_whitening=False, use_patch_na=False, use_patch_ach=False,
                     na_first=True,  na_ach_parallel=False),

    "W":        dict(use_whitening=True,  use_patch_na=False, use_patch_ach=False,
                     na_first=True,  na_ach_parallel=False),

    "NA":       dict(use_whitening=False, use_patch_na=True,  use_patch_ach=False,
                     na_first=True,  na_ach_parallel=False),

    "ACh":      dict(use_whitening=False, use_patch_na=False, use_patch_ach=True,
                     na_first=True,  na_ach_parallel=False),

    "W_NA":     dict(use_whitening=True,  use_patch_na=True,  use_patch_ach=False,
                     na_first=True,  na_ach_parallel=False),

    "W_ACh":    dict(use_whitening=True,  use_patch_na=False, use_patch_ach=True,
                     na_first=True,  na_ach_parallel=False),

    "NA_ACh":   dict(use_whitening=False, use_patch_na=True,  use_patch_ach=True,
                     na_first=True,  na_ach_parallel=False),

    "ACh_NA":   dict(use_whitening=False, use_patch_na=True,  use_patch_ach=True,
                     na_first=False, na_ach_parallel=False),

    "NA_ACh_parallel": dict(use_whitening=False, use_patch_na=True,  use_patch_ach=True,
                            na_first=False, na_ach_parallel=True),

    "W_NA_ACh": dict(use_whitening=True,  use_patch_na=True,  use_patch_ach=True,
                     na_first=True,  na_ach_parallel=False),

    "W_ACh_NA": dict(use_whitening=True,  use_patch_na=True,  use_patch_ach=True,
                     na_first=False, na_ach_parallel=False),

    "W_NA_ACh_parallel": dict(use_whitening=True, use_patch_na=True, use_patch_ach=True,
                              na_first=False, na_ach_parallel=True),
}


def main():
    parser = argparse.ArgumentParser(description="Neuromod DINOv3 ViT-B/16 ImageNet-100 experiments")
    parser.add_argument("--data_root", type=str, default="/scratch/vjh9526/cn_fall_2025/data/imagenet100_mini")
    parser.add_argument("--model_dir", type=str, default="/scratch/vjh9526/cv_fall_2025/models_vit_dinov3")
    parser.add_argument("--hf_cache_dir", type=str, default="/scratch/vjh9526/cv_fall_2025/hf_cache_dinov3",
                        help="Where to cache Hugging Face downloads (avoids $HOME/.cache).")
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--config", type=str, default="baseline", choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_neuromod", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ood_source", type=str, default="both", choices=["holdout", "coco", "both"])
    parser.add_argument("--coco_root", type=str, default="/scratch/vjh9526/cn_fall_2025/data/coco2017")
    parser.add_argument("--coco_max_images", type=int, default=5000)
    parser.add_argument("--coco_single_category_only", action="store_true")
    parser.add_argument("--coco_min_instances", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    # Force HF caches to the requested folder.
    os.environ["HF_HOME"] = args.hf_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache_dir, "transformers")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # ID / OOD split
    full_dataset = ImageNet100Dataset(root_dir=args.data_root, transform=train_transform)
    all_classes = list(full_dataset.class_to_idx.values())

    test_classes = np.random.RandomState(args.seed).choice(all_classes, size=30, replace=False).tolist()
    train_classes = [c for c in all_classes if c not in test_classes]

    train_classes_sorted = sorted(train_classes)
    label_map = {old: new for new, old in enumerate(train_classes_sorted)}
    num_classes = len(train_classes_sorted)

    train_indices = [i for i, y in enumerate(full_dataset.labels) if y in train_classes]
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataset = RemapLabelsDataset(train_subset, label_map=label_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
    )

    eval_dataset = ImageNet100Dataset(root_dir=args.data_root, transform=test_transform)
    eval_labels = eval_dataset.labels

    id_eval_indices = [i for i, y in enumerate(eval_labels) if y in train_classes]
    ood_indices = [i for i, y in enumerate(eval_labels) if y in test_classes]

    id_eval_subset = torch.utils.data.Subset(eval_dataset, id_eval_indices)
    id_eval_dataset = RemapLabelsDataset(id_eval_subset, label_map=label_map)
    holdout_dataset = torch.utils.data.Subset(eval_dataset, ood_indices)

    id_eval_loader = DataLoader(
        id_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
    )

    coco_loader = None
    if args.ood_source in ("coco", "both"):
        coco_dataset = COCO2017SingleLabelDataset(
            coco_root=args.coco_root,
            split="val2017",
            transform=test_transform,
            max_images=args.coco_max_images,
            seed=args.seed,
            single_category_only=args.coco_single_category_only,
            min_instances=args.coco_min_instances,
        )
        coco_loader = DataLoader(
            coco_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
            pin_memory=True,
        )
        print(f"COCO eval set: {len(coco_dataset)} images, {len(coco_dataset.catid_to_idx)} derived classes")

    # Model
    cfg = EXPERIMENT_CONFIGS[args.config]
    model = DINOv3ViTWithNeuromod(
        model_name=args.model_name,
        cache_dir=args.hf_cache_dir,
        num_classes=num_classes,
        use_whitening=cfg["use_whitening"],
        use_patch_na=cfg["use_patch_na"],
        use_patch_ach=cfg["use_patch_ach"],
        na_ach_parallel=cfg["na_ach_parallel"],
        na_first=cfg["na_first"],
        device=device,
    )

    freeze_backbone(model)

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=args.epochs,
        lr_head=args.lr_head,
        lr_neuromod=args.lr_neuromod,
        device=device,
    )

    ckpt_path = os.path.join(args.model_dir, f"dinov3_vitb16_{args.config}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved model to", ckpt_path)

    # ID eval
    print("Extracting ID features/logits...")
    id_feats, id_logits, id_labels = extract_features_and_logits(model, id_eval_loader, device, desc="ID")
    id_acc = (id_logits.argmax(dim=1) == id_labels).float().mean().item()

    row = {"config": args.config, "ID_acc": float(id_acc), "ood_source": args.ood_source}

    if args.ood_source in ("holdout", "both"):
        row.update(eval_ood_set(model, id_logits, holdout_loader, device, prefix="HOLDOUT"))

    if args.ood_source in ("coco", "both"):
        if coco_loader is None:
            raise RuntimeError("coco_loader not created but ood_source requires it.")
        row.update(eval_ood_set(model, id_logits, coco_loader, device, prefix="COCO"))

    results_path = os.path.join(args.model_dir, f"results_{args.config}.json")
    with open(results_path, "w") as f:
        json.dump(row, f)
    print("Saved metrics to", results_path)

    df = pd.DataFrame([row])
    print("\n=== Summary Results ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
