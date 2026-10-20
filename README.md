# Self-Supervised Visual Representation Learning with MoCo v2

> A self-supervised computer vision pipeline that learns transferable image representations from unlabeled data using MoCo v2 and ResNet-50.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-9658B2)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Training-76B900?logo=nvidia&logoColor=white)
![MoCo v2](https://img.shields.io/badge/Method-MoCo%20v2-2E8B57)
![ResNet-50](https://img.shields.io/badge/Backbone-ResNet--50-6A5ACD)

## Overview

Labeled data is expensive to collect. This project learns an image encoder without any labels by pulling together two augmented views of the same image and pushing apart views of different images (contrastive learning). MoCo makes this scale with small batches by keeping a large queue of negatives and encoding them with a slowly moving momentum encoder.

After pretraining, the encoder is frozen. Features are extracted from the downstream datasets and a logistic-regression classifier is trained on top, so the numbers below measure representation quality, not fine-tuning.

## Key Results

| Setting | Result |
|---|---:|
| Backbone | **ResNet-50** |
| Pretraining duration | **200 epochs** |
| Pretraining image size | **96 × 96** |
| Memory queue | **65,536 representations** |
| Evaluation datasets | **MiniImageNet, CUB-200, SUN397** |
| Best validation accuracy | **67.97% on MiniImageNet** |

The learned representations transfer across datasets without any fine-tuning of the encoder.

## Problem Formulation

Supervised image classification depends on large labeled datasets, but high-quality labels are expensive. This project instead learns an encoder:

```text
image -> visual embedding
```

For two augmented views of the same image, the objective keeps their embeddings similar, while embeddings of unrelated images act as negatives. The resulting feature space can be reused for downstream tasks with limited labeled data.

## Method

- **Backbone:** ResNet-50, with a 2-layer MLP projection head (FC, ReLU, FC) producing a 128-d embedding (MoCo v2).
- **Query / key encoders:** same architecture; the key encoder is an exponential moving average of the query encoder, `θ_k ← m·θ_k + (1−m)·θ_q`, with `m = 0.999`.
- **Queue:** 65,536 negative keys carried over from previous mini-batches.
- **Loss:** InfoNCE with temperature `T = 0.2`.
- **Augmentations:** two independent views per image using random resized crop, horizontal flip, color jitter, random grayscale and Gaussian blur.
- **Optimization:** SGD with momentum 0.9, batch size 256, base LR 0.03 with 10 warmup epochs then cosine decay, weight decay 1e-4, 200 epochs.
- **Resolution:** all pretraining images are resized to 96 × 96.

All of these live in [`configs/moco_r50_96.yaml`](configs/moco_r50_96.yaml).

## Downstream evaluation

1. Split the downstream dataset into train / validation / test. Test labels are never used.
2. Extract features with the frozen encoder using multi-view extraction: a center crop, a horizontally flipped crop and five spatial crops, with the feature vectors averaged per image (`--mode tta`).
3. Sanitize invalid values, L2-normalize, then fit a `StandardScaler` on the training features.
4. Train a multinomial logistic-regression classifier on the training split and select the regularization strength `C` on the validation split.
5. Retrain on train + validation with the chosen `C` and predict the test split.

Features are extracted either at the native 96 px or with the images upsampled to 128 px before they reach the frozen encoder.

### Evaluation datasets

- **MiniImageNet:** a compact benchmark for general object-level transfer.
- **CUB-200:** fine-grained bird classification, testing whether the features capture subtle visual differences.
- **SUN397:** scene recognition, testing transfer beyond object-centered images.

## Results

Validation accuracy (%), linear probe:

| Dataset | 96 px | 128 px |
|---|---:|---:|
| CUB-200 | 22.93 | **26.93** |
| MiniImageNet | 65.92 | **67.97** |
| SUN397 | 38.90 | **42.26** |

Test accuracy (%):

| Dataset | Private | Public |
|---|---:|---:|
| CUB-200 | 29.07 | 30.96 |
| MiniImageNet | 69.69 | 68.82 |
| SUN397 | 42.11 | 43.12 |

Evaluating at a larger input size than the pretraining resolution improved validation accuracy on all three datasets, most clearly on the fine-grained (CUB-200) and scene (SUN397) tasks. The dataset and encoder are identical across settings, so the gain comes from the encoder seeing larger spatial support at inference.

## Repository structure

```text
configs/   YAML experiment configs (default: moco_r50_96.yaml)
data/      Dataset download/prep, pretraining dataset, MoCo augmentations
models/    ResNet backbones
moco/      MoCo model: momentum encoder, queue, contrastive loss
engine/    Training loop
scripts/   pretrain_moco.py, make_submission_moco_adv_linprob.py
utils/     Checkpointing and distributed helpers
run_moco_full.slurm   SLURM job for full pretraining on one A100
runs.txt              Example evaluation commands
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch and torchvision for your platform (CUDA or MPS). The code also imports NumPy, pandas and PyYAML.

## Data

`data/prepare_dataset.py` downloads the pretraining shards from the Hugging Face Hub and writes them under `data/pretrain/`, split into sharded sub-directories of 1,000 images so they can be read without loading the whole set into memory. This folder is git-ignored. No labels are used during pretraining.

```bash
python -m data.prepare_dataset
```

## Pretraining

```bash
python -m scripts.pretrain_moco --config configs/moco_r50_96.yaml
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--output-dir`, `--resume <checkpoint>` to continue a run, and `--max-samples N` for quick debugging (`-1` uses the full dataset). Distributed training is supported via `--dist-url`, `--dist-backend`, `--world-size` and `--rank`. Checkpoints are written to `checkpoints/`.

On a SLURM cluster:

```bash
sbatch run_moco_full.slurm
```

## Evaluation

Run the linear probe with multi-view extraction and generate a submission CSV:

```bash
python -m scripts.make_submission_moco_adv_linprob \
  --data-dir test1 \
  --checkpoint checkpoints/checkpoint_latest.pth \
  --output submission_linear_TTA_best.csv \
  --device cuda \
  --batch-size 32 \
  --image-size 128 \
  --mode tta \
  --use-standard-scaler \
  --C-values 0.01 0.5 1 3 5 10 20 30 50 70 100 \
  --max-iter 500
```

`--mode` accepts `center`, `multicrop` or `tta`, and `--device` accepts `cuda`, `mps` or `cpu`.

## Limitations

- Pretraining uses a fixed 96 × 96 resolution, while the best evaluation uses upsampled 128 px inputs, so there is a train/test resolution mismatch.
- Gains from larger inference sizes come from interpolation and may saturate beyond the sizes tested.
- Evaluation is linear probing only; end-to-end fine-tuning could do better.
- The benefit of a larger input size varies by dataset, depending on object scale and task complexity.

## References

1. He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*, CVPR 2020.
2. Chen et al., *Improved Baselines with Momentum Contrastive Learning*, arXiv:2003.04297, 2020.
3. Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020.
4. Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021.

---

This project was done as the final project for the Deep Learning course at NYU.
