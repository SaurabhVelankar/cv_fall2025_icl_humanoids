# Bio-Inspired Neuromodulation for Robust Deep Learning

This repository contains the implementation of bio-inspired neuromodulation mechanisms applied to deep neural networks. These mechanisms—Adaptive Whitening, Noradrenaline, and Acetylcholine—are designed to improve Out-of-Distribution (OOD) generalization, uncertainty estimation, and sample efficiency in both Computer Vision and Robotics domains.

## Methodology

The project implements three core modulation blocks that can be inserted into standard backbones:

1.  **Adaptive Whitening (W):** A module that performs whitening (decorrelation) of feature channels to reduce redundancy.
2.  **Noradrenaline (NA):** A saliency-based gain control mechanism. It amplifies features with high deviation relative to a running statistical baseline, simulating the biological response to surprise.
3.  **Acetylcholine (ACh):** A reliability-based gating mechanism. It acts as a signal-to-noise filter, emphasizing consistent features based on signal reliability.

## Repository Structure

### Computer Vision (ImageNet-100 & COCO)
* **vit_lvd_label_shift_neuromod_only_outputs.py**: Main training and evaluation script for the DINOv3 ViT-B/16 architecture. It implements token-space neuromodulation where NA and ACh are applied to patch tokens. It supports evaluation on ImageNet-100 (ID) and COCO/Holdout (OOD).
* **resnet_experiments_label_shift_better_coco_with_bn.py**: Main training and evaluation script for ResNet50. It implements channel-wise neuromodulation blocks inserted after residual bottlenecks. It supports OOD evaluation on COCO and ImageNet holdout classes.
* **resnet_experiments_label_shift_better.py**: A version of the ResNet experiment script without the Batch Normalization modifications found in the "with_bn" version.
* **resnet_experiments_label_shift_better_with_bn.py**: A duplicate or variation of the ResNet script ensuring Batch Normalization compatibility.

### Robotics (MetaWorld & R3M)
* **train_r3m_neuromod_metaworld.py**: Implementation of Behavior Cloning (BC) on the Short-MetaWorld-VLA benchmark. It uses a frozen R3M (ResNet50) backbone wrapped with neuromodulation layers and trains a policy head. Supports Distributed Data Parallel (DDP) and Weights & Biases logging.

### Analysis & Visualization
* **visualizations_coco_new.py**: A comprehensive plotting script that generates dot plots, heatmaps, and scatter plots to compare different neuromodulation configurations using the JSON results from vision experiments.
* **visualizations_new.py**: An alternative or precursor visualization script for generating performance reports.

## Installation

The code relies on PyTorch, Torchvision, and standard scientific libraries. Specific dependencies include Transformers (for DINOv3), pycocotools (for COCO evaluation), and torchrl (for Robotics/R3M).

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm pillow
pip install transformers
pip install pycocotools
pip install torchrl
pip install wandb

```

## Usage: Computer Vision Experiments

The vision scripts train a linear probe (and fine-tune neuromodulation parameters) on ImageNet-100 and evaluate on OOD datasets.

### Running ResNet50

To run experiments using the ResNet50 backbone:

```bash
python resnet_experiments_label_shift_better_coco_with_bn.py \
  --data_root /path/to/imagenet100 \
  --coco_root /path/to/coco2017 \
  --model_dir ./logs/resnet \
  --config W_NA_ACh_parallel \
  --ood_source both \
  --batch_size 64 \
  --epochs 10

```

### Running ViT (DINOv3)

To run experiments using the DINOv3 backbone:

```bash
python vit_lvd_label_shift_neuromod_only_outputs.py \
  --data_root /path/to/imagenet100 \
  --coco_root /path/to/coco2017 \
  --model_dir ./logs/vit \
  --config NA_ACh_parallel \
  --model_name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --ood_source both

```

### Configuration Options

The `--config` argument controls the neuromodulation strategy. Available keys include:

* **baseline**: Standard backbone (no modulation).
* **W**: Adaptive Whitening only.
* **NA**: Noradrenaline only.
* **ACh**: Acetylcholine only.
* **W_NA_ACh_parallel**: Whitening enabled, with NA and ACh applied in parallel.
* **NA_ACh**: Sequential application (NA followed by ACh).
* **W_NA_ACh**: Whitening enabled, with sequential NA and ACh.

## Usage: Robotics (MetaWorld)

To train a behavior cloning policy on top of a frozen R3M encoder equipped with learnable neuromodulation:

```bash
# Single GPU run
python train_r3m_neuromod_metaworld.py \
  --data_root /path/to/short-metaworld-vla \
  --neuromod_config W_NA_ACh_parallel \
  --epochs 50 \
  --batch_size 128 \
  --bn_train 1 \
  --wandb 1

# Multi-GPU (DDP) run
torchrun --standalone --nproc_per_node=4 train_r3m_neuromod_metaworld.py \
  --data_root /path/to/short-metaworld-vla \
  --neuromod_config W \
  --batch_size 128

```

## Visualization

After running experiments, use the visualization script to generate summary reports and charts from the resulting JSON files.

```bash
python visualizations_coco_new.py \
  --results_dir ./logs/resnet \
  --out_dir ./logs/resnet/plots \
  --order_by OOD_composite

```

**Outputs include:**

* **Dot Plots:** ID Accuracy, OOD AUROC (MSP vs Energy), and OOD FPR95.
* **Geometry Plots:** Feature separation metrics (inter/intra class distance).
* **Few-Shot Curves:** OOD linear probe accuracy vs. number of shots.
* **Heatmap:** Z-scored performance across all metrics for all configurations.

## Key Metrics Explained

* **MSP / Energy AUROC:** Measures the ability to detect OOD samples (higher is better).
* **FPR@95:** False Positive Rate at 95% True Positive Rate (lower is better).
* **Geometry (d_inter/d_intra):** Ratio of inter-class distance to intra-class variance; indicates clustering quality.
* **Covariance Frobenius:** Frobenius norm of the difference between the feature covariance matrix and the Identity matrix; measures decorrelation.
* **Stage0 Cov:** Feature correlation statistics at the earliest layer of the network.

```

