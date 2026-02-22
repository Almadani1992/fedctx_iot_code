<div align="center">

# FedCTX-IoT

### A Privacy-Preserving Federated CNN-Transformer Framework  
### with Dual-Pathway Explainability for Real-Time IoT Intrusion Detection

<br>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%20passing-brightgreen?style=flat-square)](tests/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=flat-square)]()

<br>

> 📌 **Paper under review.** Citation and DOI will be added upon publication.

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Datasets](#datasets)
7. [Quick Start](#quick-start)
8. [Step-by-Step Usage](#step-by-step-usage)
9. [Configuration](#configuration)
10. [Baselines](#baselines)
11. [Explainability API](#explainability-api)
12. [Privacy](#privacy)
13. [Running Tests](#running-tests)
14. [Reproducing Results](#reproducing-results)
15. [Troubleshooting](#troubleshooting)
16. [Citation](#citation)
17. [License](#license)

---

## Overview

FedCTX-IoT is a federated learning framework for intrusion detection in heterogeneous IoT networks. It combines three components:

- **Multi-Scale 1D-CNN Backbone** — three parallel depthwise separable convolution branches (kernel sizes 3, 5, 7) extract traffic features at packet, flow, and session granularities simultaneously
- **Personalised Transformer Encoder** — linear-complexity (Linformer) multi-head self-attention; attention heads are kept local per client and never transmitted to the server
- **Dual-Pathway Explainability** — Pathway 1: zero-overhead attention-weight attribution at every inference; Pathway 2: on-demand federated KernelSHAP for forensic analysis

The federated protocol aggregates only CNN backbone parameters globally. Transformer attention heads remain on each device, allowing device-specific adaptation without sharing raw data.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                  EDGE TIER  —  each IoT client                   ║
║                                                                   ║
║  Raw Traffic ──► Feature Engineering                             ║
║                         │                                         ║
║          ┌──────────────▼──────────────────┐                     ║
║          │    Multi-Scale 1D-CNN Backbone   │  ← Globally shared ║
║          │  [kernel=3] [kernel=5] [kernel=7]│                     ║
║          │       Concat & Project → H_CNN   │                     ║
║          └──────────────┬──────────────────┘                     ║
║                         │                                         ║
║          ┌──────────────▼──────────────────┐                     ║
║          │  Personalised Transformer Encoder│  ← Never shared    ║
║          │  Linformer attention · L=4 layers│                     ║
║          └──────┬───────────────┬───────────┘                    ║
║                 │               │                                 ║
║         ┌───────▼──┐    ┌───────▼────────┐                      ║
║         │Classifier│    │  XAI Module    │                       ║
║         │ Softmax  │    │P1: Attention   │                       ║
║         └──────────┘    │P2: SHAP        │                       ║
║                         └────────────────┘                       ║
╚══════════╪══════════════════════════════════════════════════════╝
           │  CNN gradients only  (DP noise + Top-k sparse)
           ▼
╔══════════════════════════════════╗
║  SERVER  —  aggregation          ║
║  Weighted FedAvg (backbone only) ║
║  Aggregate importance vectors Φ  ║
║  Broadcast updated backbone      ║
╚══════════════════════════════════╝
```

---

## Project Structure

```
fedctx_iot/
│
├── config/
│   └── default.yaml             ← all hyperparameters
│
├── data/
│   ├── preprocessing.py         ← IoTPreprocessor
│   └── partitioner.py           ← FederatedPartitioner (IID + Dirichlet)
│
├── models/
│   ├── cnn_backbone.py          ← DepthwiseSeparableConv1d, MultiScaleCNNBackbone
│   ├── transformer_encoder.py   ← LinformerAttention, PersonalisedTransformerEncoder
│   └── fedctx_model.py          ← FedCTXModel (full model + parameter group split)
│
├── federated/
│   ├── client.py                ← FederatedClient (local training + DP + compression)
│   ├── aggregation.py           ← fedavg_backbone(), aggregate_importance()
│   └── server.py                ← FederatedServer (round loop + checkpointing)
│
├── privacy/
│   └── dp_mechanism.py          ← DPMechanism (DP-SGD), TopKCompressor
│
├── xai/
│   ├── attention_xai.py         ← AttentionAttributor (Pathway 1)
│   └── shap_xai.py              ← FederatedSHAP (Pathway 2)
│
├── evaluation/
│   └── metrics.py               ← classification metrics, MI attack, XAI fidelity
│
├── figures/
│   ├── fig2_confusion.py
│   ├── fig3_noniid.py
│   ├── fig4_convergence.py
│   ├── fig5_xai.py
│   ├── fig_roc.py
│   ├── fig_tsne.py
│   ├── fig_noniid_dist.py
│   ├── fig_radar.py
│   └── fig_privacy_tradeoff.py
│
├── experiments/
│   └── run_all.sh               ← full reproduction script
│
├── tests/
│   └── test_model.py            ← 20 unit tests (pytest)
│
├── train.py                     ← training entry point
├── evaluate.py                  ← evaluation entry point
├── requirements.txt
└── setup.py
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.7 (optional, CPU mode works for development)
- 16 GB RAM minimum (32 GB recommended for CICIoT2023)
- ~50 GB disk for all three datasets

Full list: [`requirements.txt`](requirements.txt)

---

## Installation

```bash
git clone https://github.com/<your-username>/fedctx-iot.git
cd fedctx-iot
pip install -r requirements.txt
pip install -e .
```

**Conda:**
```bash
conda create -n fedctx python=3.10
conda activate fedctx
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

**Verify:**
```bash
python -c "
from models.fedctx_model import FedCTXModel
import torch
model = FedCTXModel(in_features=47, n_classes=8)
x = torch.randn(2, 32, 47)
print('Output shape:', model(x).shape)   # torch.Size([2, 8])
print('Params:', model.count_parameters())
"
```

---

## Datasets

| Dataset | Samples | Attack Types | Download |
|---------|---------|-------------|----------|
| CICIoT2023 | 46.7M | 33 (7 categories) | [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| Edge-IIoTset | 20.9M | 14 (5 categories) | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset) |
| N-BaIoT | 7.1M | 10 | [UCI ML](https://archive.ics.uci.edu/dataset/442) |

Place downloaded CSV files under `data/raw/<dataset_name>/`. The preprocessor searches recursively — subfolders are fine.

---

## Quick Start

```python
import torch
from models.fedctx_model import FedCTXModel
from xai.attention_xai import AttentionAttributor

# Build model
model = FedCTXModel(in_features=47, n_classes=8)

# Simulate input: 4 samples, 32 flow sequences, 47 features
X = torch.randn(4, 32, 47)

# Inference
logits = model(X)
preds  = logits.argmax(dim=-1)
print("Predictions:", preds.tolist())

# Real-time feature importance (Pathway 1 — zero overhead)
attributor = AttentionAttributor(model)
importance = attributor.compute(X)     # shape: (47,)
print("Top feature index:", importance.argmax())
```

---

## Step-by-Step Usage

### 1. Preprocess

```bash
python -m data.preprocessing \
    --dataset    ciciot2023 \
    --raw_dir    data/raw/ciciot2023 \
    --out_dir    data/processed/ciciot2023 \
    --n_features 47 \
    --seq_len    32
```

### 2. Partition

```bash
# Non-IID (Dirichlet α=0.5)
python -m data.partitioner \
    --processed_dir data/processed/ciciot2023 \
    --out_dir       data/partitioned/ciciot2023/alpha_0.5 \
    --n_clients     8 \
    --alpha         0.5

# IID (alpha=0 triggers IID mode)
python -m data.partitioner \
    --processed_dir data/processed/ciciot2023 \
    --out_dir       data/partitioned/ciciot2023/iid \
    --n_clients     8 \
    --alpha         0
```

### 3. Train

```bash
python train.py \
    --config          config/default.yaml \
    --dataset         ciciot2023 \
    --model           fedctx \
    --partitioned_dir data/partitioned/ciciot2023/alpha_0.5 \
    --processed_dir   data/processed/ciciot2023 \
    --out_dir         results/ciciot2023/fedctx \
    --gpu             0
```

### 4. Evaluate

```bash
python evaluate.py \
    --checkpoint      results/ciciot2023/fedctx/checkpoint_round_0100.pt \
    --dataset         ciciot2023 \
    --processed_dir   data/processed/ciciot2023 \
    --partitioned_dir data/partitioned/ciciot2023/alpha_0.5 \
    --out_dir         results/ciciot2023/fedctx/eval
```

### 5. Generate figures

```bash
cd figures/
python fig2_confusion.py
python fig3_noniid.py
python fig4_convergence.py
python fig5_xai.py
python fig_roc.py
python fig_tsne.py
python fig_noniid_dist.py
python fig_radar.py
python fig_privacy_tradeoff.py
```

---

## Configuration

All hyperparameters are in `config/default.yaml`:

```yaml
federated:
  n_clients: 8
  client_fraction: 0.5     # fraction selected per round
  n_rounds: 100
  local_epochs: 5

model:
  cnn_kernels: [3, 5, 7]
  cnn_embedding_dim: 128
  transformer_layers: 4
  n_heads: 8
  linformer_rank: 64
  n_classes: 8

training:
  learning_rate: 1.0e-3
  batch_size: 256
  focal_loss_gamma: 2.0

privacy:
  enabled: true
  epsilon: 3.0
  clip_norm: 1.0
  noise_multiplier: 1.1

compression:
  enabled: true
  top_k_ratio: 0.1
  error_feedback: true
```

---

## Baselines

```bash
python train.py --model fedavg_cnn               # FedAvg + 1D-CNN
python train.py --model fedprox_cnn              # FedProx + 1D-CNN
python train.py --model fedavg_lstm              # FedAvg + LSTM
python train.py --model fedavg_cnn_transformer   # FedAvg + CNN-Transformer (no personalisation)
python train.py --model fedrep_cnn_transformer   # FedRep + CNN-Transformer
python train.py --model fedctx                   # FedCTX-IoT (ours)
```

---

## Explainability API

### Pathway 1 — Real-time (zero overhead)

```python
from xai.attention_xai import AttentionAttributor

attributor = AttentionAttributor(model)
importance = attributor.compute(X_tensor)              # (F,) normalised [0,1]

# Client-level importance averaged over local data
phi_k = attributor.compute_local_importance(dataloader, device, n_batches=10)
```

### Pathway 2 — On-demand forensic SHAP

```python
from xai.shap_xai import FederatedSHAP

shap_module = FederatedSHAP(
    model=model,
    background=X_train[:100],    # local data only — no cross-client sharing
    device=device,
    add_dp_noise=True
)
shap_vals = shap_module.explain(X_test[0])

# Compare both pathways
result = shap_module.compare_with_attention(X_test[0], attn_importance=phi_k)
print(f"Pearson r = {result['pearson_r']:.4f}")
```

---

## Privacy

Three independent protection layers:

**1. Federated Learning** — raw traffic never leaves the device.

**2. DP-SGD** — gradients are clipped and noised before every transmission.

```python
from privacy.dp_mechanism import DPMechanism

dp  = DPMechanism(clip_norm=1.0, noise_multiplier=1.1, delta=1e-5)
eps = dp.compute_epsilon(n_samples=50000, batch_size=256)
print(f"ε after 100 rounds: {eps:.2f}")
```

**3. Top-k sparsification** — only 10% of gradient coordinates are transmitted.

**What is never transmitted:** raw traffic, Transformer attention heads, SHAP values (unless explicitly requested by analyst, with DP noise applied).

---

## Running Tests

```bash
# All 20 tests
pytest tests/ -v

# By module
pytest tests/test_model.py::TestCNNBackbone -v
pytest tests/test_model.py::TestDPMechanism -v
pytest tests/test_model.py::TestAggregation -v
pytest tests/test_model.py::TestPartitioner -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Reproducing Results

```bash
bash experiments/run_all.sh
```

This script runs preprocessing, partitioning (4 Dirichlet α values + IID), training for all 6 models on all 3 datasets, privacy budget sweep, evaluation, and figure generation.

> Estimated time: ~48 hours on a single NVIDIA A100.  
> For a quick check, run on N-BaIoT only (~2 hours).

---

## Troubleshooting

**`CUDA out of memory`**  
Reduce batch size in `config/default.yaml`: `batch_size: 64`

**`FileNotFoundError: No CSV files found`**  
Check dataset path. The preprocessor searches recursively:  
`find data/raw/ciciot2023 -name "*.csv" | head`

**`ValueError: not enough samples for sequence_length`**  
Use `--seq_len 16` instead of 32.

**`ImportError: shap`**  
Only needed for Pathway 2: `pip install shap`

---

## Citation

> The paper is currently under review. Citation details will be added upon publication.  
> In the meantime, if you use this code please link to this repository.

```bibtex
@misc{AlMadani2025FedCTX,
  author = {Al-madani, Ali Mansour},
  title  = {{FedCTX-IoT}: A Privacy-Preserving Federated {CNN}-Transformer
            Framework with Dual-Pathway Explainability for
            Real-Time {IoT} Intrusion Detection},
  year   = {2025},
  url    = {https://github.com/<your-username>/fedctx-iot}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If this code is useful to you, please consider giving it a ⭐**

</div>
