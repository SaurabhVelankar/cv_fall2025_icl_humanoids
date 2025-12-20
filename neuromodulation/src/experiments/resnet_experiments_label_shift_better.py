import argparse
import os
import numpy as np
from PIL import Image
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

import torch.utils.data.dataloader as dl
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import json

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================
# Neuromodulation modules
# =============================

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

        # learnable blend strength
        eps = 1e-4
        s = torch.clamp(torch.tensor(strength), eps, 1 - eps)
        initial_param = torch.log(s / (1 - s))
        self.strength = nn.Parameter(initial_param)

        # fixed random projection
        W = torch.randn(num_channels, num_channels)
        W, _ = torch.linalg.qr(W)
        self.register_buffer("W", W)

        self.register_buffer("g", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        with torch.no_grad():
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            G = torch.diag(F.softplus(self.g))
            M = torch.linalg.inv(torch.eye(C, device=x.device) + self.W @ G @ self.W.T)
            y_flat = x_flat @ M.T

            if self.adapt:
                z = y_flat @ self.W
                z_var = z.pow(2).mean(dim=0)
                momentum = self.train_momentum if self.training else self.eval_momentum
                self.running_var.lerp_(z_var, momentum)
                self.g.add_(self.running_var - 1, alpha=self.eta)
                self.g.clamp_(-self.max_gain, self.max_gain)

            y_whitened = y_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        strength = torch.sigmoid(self.strength)
        return (1 - strength) * x + strength * y_whitened

    def reset(self):
        self.g.zero_()
        self.running_var.fill_(1.0)


class NAModulation(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 4,
                 train_momentum: float = 0.01, eval_momentum: float = 0.1,
                 adapt: bool = True):
        super().__init__()
        self.C = num_channels
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

        x_mean = x.mean(dim=[2, 3])
        x_var = x.var(dim=[2, 3])

        if self.adapt:
            with torch.no_grad():
                momentum = self.train_momentum if self.training else self.eval_momentum
                self.running_mean.lerp_(x_mean.mean(0), momentum)
                self.running_var.lerp_(x_var.mean(0), momentum)

        mean_dev = (x_mean - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        var_dev = (x_var - self.running_var) / (self.running_var + 1e-6)

        # saliency = self.saliency_net(torch.cat([mean_dev, var_dev], dim=1))
        # gain = F.softplus(self.baseline + saliency * mean_dev.abs())
        
        # ---- NEW: clamp deviations to avoid crazy values ----
        mean_dev = torch.clamp(mean_dev, -5.0, 5.0)
        var_dev  = torch.clamp(var_dev,  -5.0, 5.0)
    
        saliency = self.saliency_net(torch.cat([mean_dev, var_dev], dim=1))
        gain = F.softplus(self.baseline + saliency * mean_dev.abs())
    
        # ---- NEW: clamp gain to avoid massive amplification ----
        gain = torch.clamp(gain, max=3.0)   # e.g. at most 3× per NA pass

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
        x_gap = x.mean(dim=[2, 3])
        reliability = self.reliability_net(x_gap)
        gate = torch.sigmoid((reliability - self.threshold) * 10)
        return x * (gate * reliability).view(B, C, 1, 1)

    def reset(self):
        pass


class NeuromodBlock(nn.Module):
    def __init__(self, num_channels: int,
                 use_whitening=True, use_na=True, use_ach=True,
                 adapt_whitening=True, adapt_na=True,
                 na_first=True, na_ach_parallel=True):
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


class ResNet50WithNeuromod(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        pretrained_path: Optional[str] = None,
        use_whitening: bool = True,
        use_na: bool = True,
        use_ach: bool = True,
        adapt_whitening: bool = True,
        adapt_na: bool = True,
        na_first: bool = True,
        na_ach_parallel: bool = True,
        whitening_only_at_start: bool = True,
    ):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            self.backbone = models.resnet50(pretrained=True)

        self.backbone.fc = nn.Linear(2048, num_classes)

        if pretrained_path is not None and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded weights from {pretrained_path}")
        else:
            print("No custom pretrained weights loaded (ImageNet-pretrained backbone only).")

        self.use_neuromod = use_whitening or use_na or use_ach
        self.adapt_whitening = adapt_whitening
        self.adapt_na = adapt_na
        self.na_first = na_first
        self.na_ach_parallel = na_ach_parallel
        self.whitening_only_at_start = whitening_only_at_start

        self.use_whitening = use_whitening
        self.use_na = use_na
        self.use_ach = use_ach

        if self.use_neuromod:
            if self.use_whitening and self.whitening_only_at_start:
                self.neuromod0 = NeuromodBlock(64, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod1 = NeuromodBlock(256, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod2 = NeuromodBlock(512, False, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod3 = NeuromodBlock(1024, False, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod4 = NeuromodBlock(2048, False, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
            else:
                self.neuromod0 = NeuromodBlock(64, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod1 = NeuromodBlock(256, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod2 = NeuromodBlock(512, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod3 = NeuromodBlock(1024, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
                self.neuromod4 = NeuromodBlock(2048, use_whitening, use_na, use_ach,
                                               adapt_whitening, adapt_na,
                                               na_first, na_ach_parallel)
            print(f"Neuromodulation enabled: whitening={use_whitening}, NA={use_na}, ACh={use_ach}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if self.use_neuromod:
            x = self.neuromod0(x)
        x = self.backbone.maxpool(x)
    
        # ---- layer1: apply neuromod1 after every bottleneck ----
        for block in self.backbone.layer1:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod1(x)
    
        # ---- layer2: neuromod2 after every bottleneck ----
        for block in self.backbone.layer2:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod2(x)
    
        # ---- layer3: neuromod3 after every bottleneck ----
        for block in self.backbone.layer3:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod3(x)
    
        # ---- layer4: neuromod4 after every bottleneck ----
        for block in self.backbone.layer4:
            x = block(x)
            if self.use_neuromod:
                x = self.neuromod4(x)
    
        # Head
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


    def reset_neuromodulation(self):
        if self.use_neuromod:
            self.neuromod0.reset()
            self.neuromod1.reset()
            self.neuromod2.reset()
            self.neuromod3.reset()
            self.neuromod4.reset()

    def get_neuromod_params(self):
        if not self.use_neuromod:
            return []
        params = list(self.neuromod0.parameters())
        params.extend(self.neuromod1.parameters())
        params.extend(self.neuromod2.parameters())
        params.extend(self.neuromod3.parameters())
        params.extend(self.neuromod4.parameters())
        return params

# =============================
# Dataset & loaders
# =============================

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

class RemapLabelsDataset(Dataset):
    """
    Wraps a dataset that returns {"image": ..., "label": ...} and remaps labels via label_map.
    Any label not found in label_map becomes unknown_label (default: -1).
    """
    def __init__(self, base: Dataset, label_map: Dict[int, int], unknown_label: int = -1):
        self.base = base
        self.label_map = dict(label_map)
        self.unknown_label = unknown_label

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        y = int(item["label"])
        item["label"] = self.label_map.get(y, self.unknown_label)
        return item


# =============================
# Experiment configs
# =============================

EXPERIMENT_CONFIGS = {
    "baseline": dict(use_whitening=False, use_na=False, use_ach=False,
                     na_first=True, na_ach_parallel=False),

    "W":        dict(use_whitening=True,  use_na=False, use_ach=False,
                     na_first=True, na_ach_parallel=False),
    "NA":       dict(use_whitening=False, use_na=True,  use_ach=False,
                     na_first=True, na_ach_parallel=False),
    "ACh":      dict(use_whitening=False, use_na=False, use_ach=True,
                     na_first=True, na_ach_parallel=False),

    "W_NA":     dict(use_whitening=True,  use_na=True,  use_ach=False,
                     na_first=True,  na_ach_parallel=False),
    "W_ACh":    dict(use_whitening=True,  use_na=False, use_ach=True,
                     na_first=True,  na_ach_parallel=False),
    "NA_ACh":   dict(use_whitening=False, use_na=True,  use_ach=True,
                     na_first=True,  na_ach_parallel=False),
    "ACh_NA":   dict(use_whitening=False, use_na=True,  use_ach=True,
                     na_first=False, na_ach_parallel=False),

    "NA_ACh_parallel": dict(use_whitening=False, use_na=True,  use_ach=True,
                            na_first=False,  na_ach_parallel=True),

    "W_NA_ACh": dict(use_whitening=True,  use_na=True,  use_ach=True,
                     na_first=True,  na_ach_parallel=False),
    "W_ACh_NA": dict(use_whitening=True,  use_na=True,  use_ach=True,
                     na_first=False, na_ach_parallel=False),

    "W_NA_ACh_parallel": dict(use_whitening=True, use_na=True, use_ach=True,
                              na_first=False, na_ach_parallel=True),
}

def create_model(config_name: str, num_classes: int) -> ResNet50WithNeuromod:
    cfg = EXPERIMENT_CONFIGS[config_name]
    model = ResNet50WithNeuromod(
        num_classes=num_classes,
        pretrained_path=None,  # using ImageNet-pretrained backbone
        use_whitening=cfg["use_whitening"],
        use_na=cfg["use_na"],
        use_ach=cfg["use_ach"],
        adapt_whitening=True,
        adapt_na=True,
        na_first=cfg["na_first"],
        na_ach_parallel=cfg["na_ach_parallel"],
        whitening_only_at_start=True,
    )
    return model.to(device)

def freeze_backbone_except_fc(model: ResNet50WithNeuromod):
    for name, p in model.backbone.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = False

# =============================
# Training / evaluation helpers
# =============================

def train_model(model: ResNet50WithNeuromod,
                config_name: str,
                train_loader: DataLoader,
                num_epochs: int = 10,
                lr_head: float = 1e-3,
                lr_neuromod: float = 1e-3):

    model.train()
    # Always use the same optimizer across configs for fair comparisons
    params = [{"params": model.backbone.fc.parameters(), "lr": lr_head}]

    if config_name != "baseline" and model.use_neuromod:
        params.append({"params": model.get_neuromod_params(), "lr": lr_neuromod})

    optimizer = torch.optim.SGD(params, momentum=0.9)


    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup_epochs = max(1, num_epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler_cos], milestones=[warmup_epochs]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        # Train FC (+ neuromod) while keeping the backbone in eval mode to freeze BN running stats
        model.train()
        model.backbone.eval()
        model.backbone.fc.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"[{config_name}] Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += batch_size

            avg_loss = total_loss / total
            acc = correct / total
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                acc=f"{acc:.4f}",
                lr=f"{current_lr:.2e}",
            )

        scheduler.step()
        print(f"[{config_name}] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")

@torch.no_grad()
def extract_features_and_logits(model: ResNet50WithNeuromod,
                                loader: DataLoader,
                                desc: str = "Extracting"):
    model.eval()
    feats_list, logits_list, labels_list = [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        x = model.backbone.conv1(images)
        x = model.backbone.bn1(x)
        x = model.backbone.relu(x)
        if model.use_neuromod:
            x = model.neuromod0(x)
        x = model.backbone.maxpool(x)

        # layer1 with neuromod per bottleneck
        for block in model.backbone.layer1:
            x = block(x)
            if model.use_neuromod:
                x = model.neuromod1(x)

        # layer2
        for block in model.backbone.layer2:
            x = block(x)
            if model.use_neuromod:
                x = model.neuromod2(x)

        # layer3
        for block in model.backbone.layer3:
            x = block(x)
            if model.use_neuromod:
                x = model.neuromod3(x)

        # layer4
        for block in model.backbone.layer4:
            x = block(x)
            if model.use_neuromod:
                x = model.neuromod4(x)

        x = model.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = model.backbone.fc(feats)

        feats_list.append(feats.cpu())
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())

    feats_all = torch.cat(feats_list, dim=0)
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    return feats_all, logits_all, labels_all


def compute_geometry_metrics(feats: torch.Tensor, labels: torch.Tensor):
    X = feats.numpy()
    y = labels.numpy()
    classes = np.unique(y)

    means = []
    for c in classes:
        means.append(X[y == c].mean(axis=0))
    means = np.stack(means, axis=0)

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

    separation = d_inter / d_intra if d_intra > 0 else np.nan
    return d_intra, d_inter, separation

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

def few_shot_eval(feats: torch.Tensor,
                  labels: torch.Tensor,
                  shots_list = [1, 3, 5, 10, 15],
                  num_epochs: int = 100,
                  lr: float = 1e-2,
                  desc_prefix: str = "few-shot"):
    X = feats.to(device)
    y = labels.to(device)
    classes = torch.unique(y)
    results = {}

    for k in shots_list:
        print(f"[{desc_prefix}] k={k} shots (classes={len(classes)})")
        support_idx = []
        query_idx = []
        for c in classes:
            idx_c = torch.nonzero(y == c, as_tuple=False).view(-1)
            if len(idx_c) < 2:
                continue
            idx_perm = idx_c[torch.randperm(len(idx_c))]
            k_eff = min(k, len(idx_perm) - 1)
            support_idx.append(idx_perm[:k_eff])
            query_idx.append(idx_perm[k_eff:])

        support_idx = torch.cat(support_idx)
        query_idx = torch.cat(query_idx)

        X_support = X[support_idx]
        y_support = y[support_idx]
        X_query = X[query_idx]
        y_query = y[query_idx]

        clf = nn.Linear(X.shape[1], len(classes)).to(device)
        class_to_new = {int(c.item()): i for i, c in enumerate(classes)}
        y_support_mapped = torch.tensor(
            [class_to_new[int(c.item())] for c in y_support], device=device
        )

        opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        clf.train()
        for epoch in range(num_epochs):
            opt.zero_grad()
            logits = clf(X_support)
            loss = loss_fn(logits, y_support_mapped)
            loss.backward()
            opt.step()

            if (epoch + 1) % 20 == 0:
                print(f"  [{desc_prefix}] k={k} epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

        clf.eval()
        with torch.no_grad():
            logits_q = clf(X_query)
            preds_q = logits_q.argmax(dim=1)
            y_query_mapped = torch.tensor(
                [class_to_new[int(c.item())] for c in y_query], device=device
            )
            acc = (preds_q == y_query_mapped).float().mean().item()

        print(f"[{desc_prefix}] k={k} shots accuracy: {acc:.4f}")
        results[k] = acc
        del clf

    return results

@torch.no_grad()
def extract_stage0_features(model: ResNet50WithNeuromod,
                            loader: DataLoader,
                            max_total_samples: int = 200_000,
                            desc: str = "Stage0 feats"):
    """
    Collect up to max_total_samples spatial features (N, C) from stage 0.
    This is a *global* cap over the whole loader, not per batch.
    """
    model.eval()
    feats_list = []
    collected = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        if collected >= max_total_samples:
            break

        images = batch["image"].to(device)

        x = model.backbone.conv1(images)
        x = model.backbone.bn1(x)
        x = model.backbone.relu(x)
        if model.use_neuromod:
            x = model.neuromod0(x)

        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        remaining = max_total_samples - collected
        if x_flat.size(0) > remaining:
            idx = torch.randperm(x_flat.size(0), device=x_flat.device)[:remaining]
            x_flat = x_flat[idx]

        feats_list.append(x_flat.cpu())
        collected += x_flat.size(0)

    if not feats_list:
        # In weird degenerate case (empty loader), return empty tensor
        return torch.empty(0, 0)

    feats_all = torch.cat(feats_list, dim=0)
    return feats_all


# =============================
# Main (CLI)
# =============================

def main():
    parser = argparse.ArgumentParser(description="Neuromod ResNet50 ImageNet-100 experiments")
    parser.add_argument("--data_root", type=str,
                        default="/scratch/vjh9526/cn_fall_2025/data/imagenet100_mini")
    parser.add_argument("--model_dir", type=str,
                        default="/scratch/vjh9526/cn_fall_2025/models_better_imagenet100")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_neuromod", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

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

    # Split ID/OOD classes
    full_dataset = ImageNet100Dataset(root_dir=args.data_root, transform=train_transform)
    all_classes = list(full_dataset.class_to_idx.values())

    np.random.seed(args.seed)
    test_classes = np.random.choice(all_classes, size=30, replace=False).tolist()
    train_classes = [c for c in all_classes if c not in test_classes]

    # --- NEW: closed-world classifier over K known classes only ---
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
        persistent_workers=True,
        pin_memory=True,
    )

    # Eval datasets (use test_transform)
    eval_dataset = ImageNet100Dataset(root_dir=args.data_root, transform=test_transform)
    eval_labels = eval_dataset.labels

    id_eval_indices = [i for i, y in enumerate(eval_labels) if y in train_classes]
    ood_indices = [i for i, y in enumerate(eval_labels) if y in test_classes]

    id_eval_subset = torch.utils.data.Subset(eval_dataset, id_eval_indices)
    id_eval_dataset = RemapLabelsDataset(id_eval_subset, label_map=label_map)

    ood_dataset = torch.utils.data.Subset(eval_dataset, ood_indices)  # keep original OOD labels for geometry/few-shot

    id_eval_loader = DataLoader(
        id_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )


    print(f"\nRunning config: {args.config}")
    model = create_model(args.config, num_classes=num_classes)
    freeze_backbone_except_fc(model)

    train_model(
        model,
        args.config,
        train_loader,
        num_epochs=args.epochs,
        lr_head=args.lr_head,
        lr_neuromod=args.lr_neuromod,
    )

    model_path = os.path.join(args.model_dir, f"resnet50_{args.config}_full.pth")
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)

    print("Extracting ID features/logits...")
    id_feats, id_logits, id_labels = extract_features_and_logits(
        model, id_eval_loader, desc=f"[{args.config}] ID features"
    )
    print("Extracting OOD features/logits...")
    ood_feats, ood_logits, ood_labels = extract_features_and_logits(
        model, ood_loader, desc=f"[{args.config}] OOD features"
    )

    # ID accuracy
    id_preds = id_logits.argmax(dim=1)
    id_acc = (id_preds == id_labels).float().mean().item()

    # OOD scores
    id_probs = F.softmax(id_logits, dim=1).numpy()
    ood_probs = F.softmax(ood_logits, dim=1).numpy()
    id_msp = id_probs.max(axis=1)
    ood_msp = ood_probs.max(axis=1)

    # Stable energy score via logsumexp (T=1). Use logsumexp as the score (higher => more ID-like in this convention).
    id_energy_score = torch.logsumexp(id_logits, dim=1).numpy()
    ood_energy_score = torch.logsumexp(ood_logits, dim=1).numpy()

    msp_auroc, msp_aupr, msp_fpr95 = ood_metrics_from_scores(id_msp, ood_msp)
    eng_auroc, eng_aupr, eng_fpr95 = ood_metrics_from_scores(id_energy_score, ood_energy_score)

    d_intra, d_inter, sep = compute_geometry_metrics(ood_feats, ood_labels)
    eigvals, frob = compute_covariance_metrics(ood_feats)

    stage0_ood_feats = extract_stage0_features(
        model, ood_loader, max_total_samples=200_000, desc=f"[{args.config}] stage0 OOD"
    )
    _, frob_stage0 = compute_covariance_metrics(stage0_ood_feats)


    fewshot = few_shot_eval(ood_feats, ood_labels, desc_prefix=f"{args.config} few-shot")

    row = {
        "config": args.config,
        "ID_acc": id_acc,
        "MSP_AUROC": msp_auroc,
        "MSP_AUPR": msp_aupr,
        "MSP_FPR95": msp_fpr95,
        "ENG_AUROC": eng_auroc,
        "ENG_AUPR": eng_aupr,
        "ENG_FPR95": eng_fpr95,
        "OOD_d_intra": d_intra,
        "OOD_d_inter": d_inter,
        "OOD_sep": sep,
        "OOD_frob_cov": frob,
        "OOD_frob_cov_stage0": frob_stage0,
    }
    for k_shot, acc in fewshot.items():
        row[f"fewshot_{k_shot}"] = acc

    # save per-config metrics to disk so you can aggregate later
    config_metrics_path = os.path.join(args.model_dir, f"results_{args.config}.json")
    with open(config_metrics_path, "w") as f:
        json.dump(row, f)
    print(f"Saved metrics for {args.config} to", config_metrics_path)
    
    df = pd.DataFrame([row])
    print("\n=== Summary Results ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
