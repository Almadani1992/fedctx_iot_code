<div align="center">

<img src="https://img.shields.io/badge/FedCTX--IoT-v1.0.0-crimson?style=for-the-badge" />

# FedCTX-IoT

### A Privacy-Preserving Federated CNN-Transformer Framework  
### with Dual-Pathway Explainability for Real-Time IoT Intrusion Detection

<br>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue?style=flat-square)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Scientific Reports](https://img.shields.io/badge/Published-Scientific%20Reports-00A98F?style=flat-square&logo=nature&logoColor=white)](https://www.nature.com/srep/)
[![Tests](https://img.shields.io/badge/Tests-20%20passing-brightgreen?style=flat-square)](tests/)

<br>

**Official implementation of the paper:**  
*FedCTX-IoT: A Privacy-Preserving Federated CNN-Transformer Framework with  
Dual-Pathway Explainability for Real-Time IoT Intrusion Detection*  
**Scientific Reports (Nature Portfolio), 2025**

<br>

| 📄 [Paper](#citation) | 🚀 [Quick Start](#quick-start) | 📊 [Results](#key-results) | 🔍 [XAI API](#explainability-api) | 🛡️ [Privacy](#privacy-guarantees) |

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Why FedCTX-IoT?](#why-fedctx-iot)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Key Results](#key-results)
5. [Project Structure](#project-structure)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Datasets](#datasets)
9. [Quick Start](#quick-start)
10. [Step-by-Step Usage](#step-by-step-usage)
11. [Configuration Reference](#configuration-reference)
12. [Module Reference](#module-reference)
13. [Baselines](#baselines)
14. [Explainability API](#explainability-api)
15. [Privacy Guarantees](#privacy-guarantees)
16. [Running Tests](#running-tests)
17. [Reproducing Paper Results](#reproducing-paper-results)
18. [Troubleshooting](#troubleshooting)
19. [Citation](#citation)
20. [Contributing](#contributing)
21. [License](#license)

---

## Overview

FedCTX-IoT is a **production-grade federated learning framework** for intrusion detection in heterogeneous IoT networks. It is the first system to simultaneously address four requirements that prior work treated in isolation:

| Requirement | Prior Federated IDS | FedCTX-IoT |
|-------------|-------------------|------------|
| Rich temporal feature extraction | Shallow CNN / LSTM | Multi-scale CNN + Transformer |
| Non-IID robustness | Uniform FedAvg aggregation | Split-aggregation personalisation |
| Real-time explanations | Post-hoc SHAP only | Zero-overhead attention attribution |
| Formal privacy guarantees | Rare / absent | DP-SGD + Rényi accounting |

The framework is designed to run on real IoT infrastructure: clients can be Raspberry Pi devices (4.1 ms inference latency), and the communication overhead (224.1 MB total) fits within constrained LPWAN budgets.

---

## Why FedCTX-IoT?

### The problem

Modern IoT deployments are under constant attack. Between 2022 and 2024, IoT devices became the primary target for DDoS botnets, ransomware deployment chains, and reconnaissance campaigns. Classical intrusion detection requires centralising all traffic data — creating **privacy exposure**, **bandwidth bottlenecks**, and **single points of failure**.

Federated Learning (FL) solves the centralisation problem, but existing federated IDS approaches have three critical gaps:

**Gap 1 — Shallow architectures.** Most federated IDS papers use standard CNNs or LSTMs. These miss multi-scale temporal structure: packet-level bursts (short kernels), flow-level statistics (medium kernels), and session-level patterns (long kernels) all carry attack signals at different scales simultaneously.

**Gap 2 — Uniform aggregation under non-IID data.** A DDoS-heavy gateway and a Mirai-infected camera node have fundamentally different traffic distributions. Standard FedAvg forces a single global model across all devices, creating gradient conflict that degrades performance for every client.

**Gap 3 — Black-box opacity.** Security analysts need to understand why a model raised an alert before acting on it. SHAP-based explanations require hundreds of additional forward passes and expose the local data distribution. Neither is acceptable at the edge.

### The solution

FedCTX-IoT resolves all three gaps through tight co-design of the FL protocol and model architecture:

- The **multi-scale 1D-CNN backbone** extracts features at three temporal granularities in parallel using depthwise separable convolutions — 7× fewer parameters than standard convolutions.
- The **split-aggregation protocol** aggregates only CNN backbone parameters globally, keeping device-specific Transformer attention heads entirely local — eliminating gradient conflict without sacrificing shared feature learning.
- **Pathway 1 XAI** re-uses attention weights already computed during inference to produce per-feature importance scores with literally zero additional cost.

---

## Architecture Deep Dive

### Full System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                  EDGE TIER  (each IoT gateway)                       ║
║                                                                       ║
║  ┌──────────┐    ┌───────────────────────────────────────────────┐   ║
║  │ Raw IoT  │───►│            Feature Engineering                │   ║
║  │ Traffic  │    │  Flow stats · Packet stats · Protocol flags   │   ║
║  └──────────┘    └────────────────────┬──────────────────────────┘   ║
║                                       │                               ║
║                  ┌────────────────────▼──────────────────────────┐   ║
║                  │      Multi-Scale 1D-CNN Backbone               │   ║
║                  │   ┌──────────┐ ┌──────────┐ ┌──────────┐     │   ║
║                  │   │ kernel=3 │ │ kernel=5 │ │ kernel=7 │     │   ║
║                  │   │ (packet) │ │  (flow)  │ │(session) │     │   ║
║                  │   └────┬─────┘ └────┬─────┘ └────┬─────┘     │   ║
║                  │        └────────────┼─────────────┘           │   ║
║                  │               Concat & Project                 │   ║
║                  │          H_CNN ∈ ℝ^(N×D)                       │   ║
║    ◄── Globally ─┼─────────────────────────────────────────────  │   ║
║        Shared    └────────────────────┬──────────────────────────┘   ║
║                                       │                               ║
║                  ┌────────────────────▼──────────────────────────┐   ║
║                  │   Personalised Transformer Encoder  ← LOCAL   │   ║
║                  │   Linformer multi-head attention               │   ║
║                  │   L=4 layers · M=8 heads · rank r=64          │   ║
║                  │   Complexity: O(Nr) not O(N²)                  │   ║
║    ◄── Never  ───┼─────────────────────────────────────────────  │   ║
║        Shared    └────────────────────┬──────────────────────────┘   ║
║                                       │                               ║
║          ┌────────────────────────────┼────────────────────┐         ║
║          │                            │                    │         ║
║  ┌───────▼──────┐          ┌──────────▼───────┐  ┌────────▼──────┐  ║
║  │  Classifier  │          │   Pathway 1 XAI  │  │ Pathway 2 XAI │  ║
║  │ Softmax head │          │ Attention weights │  │ KernelSHAP    │  ║
║  │ C classes    │          │ Zero overhead     │  │ On-demand     │  ║
║  └───────┬──────┘          └──────────┬───────┘  └────────┬──────┘  ║
║          │                            │                    │         ║
╚══════════╪════════════════════════════╪════════════════════╪═════════╝
           │    DP clip + Gauss noise   │                    │
           │    Top-k sparsification    │                    │
           │    CNN gradients only ─────►                   │
           ▼                                                 │ (analyst
╔══════════════════════════════════════╗                     │  request)
║     SERVER TIER  (aggregation)       ║◄────────────────────┘
║                                      ║
║  Weighted FedAvg on CNN backbone     ║  θᵍ_t = Σ (n_k/n) θᵍ_k
║  Aggregate feature importance Φ      ║  Φ_t  = Σ (n_k/n) φ_k
║  Broadcast updated backbone          ║
║  Privacy accounting (Rényi DP)       ║  ε = 3.0  |  δ = 1e-5
╚══════════════════════════════════════╝
```

### Component 1: Multi-Scale 1D-CNN Backbone

**File:** `models/cnn_backbone.py`

**Why depthwise separable convolutions?**  
A standard Conv1d(F, D, k) has k·F·D parameters. A depthwise separable equivalent has k·F + F·D parameters — approximately 7× fewer for k=7, D=128, F=47.

```
Input: X ∈ ℝ^(B × N × F)
                │
   ┌────────────┼────────────┐
   │            │            │
   ▼            ▼            ▼
DW-Sep(k=3)  DW-Sep(k=5)  DW-Sep(k=7)
BN + GELU    BN + GELU    BN + GELU
Residual     Residual     Residual
   │            │            │
   └────────────┼────────────┘
                ▼
         Concat: (B, N, 3D)
                ▼
         Linear: (B, N, D)  ← H_CNN
```

Each branch uses same-padding (`padding = kernel_size // 2`) so all branches produce identical spatial dimensions — no interpolation needed before concatenation.

### Component 2: Personalised Transformer Encoder

**File:** `models/transformer_encoder.py`

**Why Linformer?**  
Standard self-attention is O(N²) in memory and compute. For N=32 flow sequences this is manageable, but for longer sequences it becomes prohibitive on edge devices. Linformer projects keys and values to rank r=64 using learnable matrices E, F ∈ ℝ^(N×r), reducing complexity to O(Nr).

```
Q = X W_Q  ∈ ℝ^(B×M×N×d_h)
K = X W_K  ∈ ℝ^(B×M×N×d_h)  ──► K̃ = E^T K  ∈ ℝ^(B×M×r×d_h)
V = X W_V  ∈ ℝ^(B×M×N×d_h)  ──► Ṽ = F^T V  ∈ ℝ^(B×M×r×d_h)

Attention = softmax(Q K̃^T / √d_h) Ṽ   ← O(Nr) not O(N²)
```

The attention weight matrix `(B, M, N, r)` is stored after each forward pass and consumed directly by `AttentionAttributor` — no extra computation.

**Why keep Transformer heads local?**  
A DDoS-heavy gateway attends to high packet-rate features. A Mirai botnet device attends to C&C command timing. Aggregating these via FedAvg produces a head that is well-suited for neither. Keeping θˡₖ local lets each client specialise its temporal modelling to its own attack distribution.

### Component 3: Split-Aggregation FL Protocol

**File:** `federated/client.py`, `federated/aggregation.py`, `federated/server.py`

```
Round t:
┌─────────────────────────────────────────────────────────┐
│ Server selects S_t ⊆ {1,...,K}  with |S_t| = ⌊qK⌋      │
│                                                          │
│ For each client k ∈ S_t in parallel:                    │
│   1. Load global backbone θᵍ_{t-1}                       │
│   2. Train E=5 epochs with AdamW + focal loss            │
│   3. Compute gradient: g_k = θᵍ_{t-1} - θᵍ_k            │
│   4. Clip:   g̃_k = g_k · min(1, C / ||g_k||)            │
│   5. Noise:  ĝ_k = g̃_k + N(0, σ²C²I)    [DP-SGD]       │
│   6. Sparse: send top-s=10% of |ĝ_k|     [TopK]         │
│   7. Also send φ_k (attention importance, no raw data)   │
│                                                          │
│ Server:                                                  │
│   θᵍ_t = Σ_{k∈S_t} (n_k / n_S) · (θᵍ_{t-1} - ĝ_k)    │
│   Φ_t  = Σ_{k∈S_t} (n_k / n_S) · φ_k                   │
│   Broadcast θᵍ_t to all K clients                       │
└─────────────────────────────────────────────────────────┘
```

**What is never transmitted:** Transformer attention weights θˡₖ, classifier weights, raw traffic data, SHAP values.

### Component 4: Dual-Pathway XAI

**File:** `xai/attention_xai.py`, `xai/shap_xai.py`

**Pathway 1 — Attention attribution (real-time)**

During every forward pass, attention weight matrices `A ∈ ℝ^(B×M×N×r)` are stored. After inference:

```python
# Aggregate across layers and heads
token_importance = mean over layers( mean over heads( A ) )  # (B, N)

# Map from token space to feature space via gradient
feat_importance = mean( |∂token_importance/∂x| · token_importance )  # (F,)
```

This is computed in a single gradient call — not additional inference passes.

**Pathway 2 — Federated SHAP (on-demand)**

KernelSHAP runs locally using only the client's own background distribution:
- No other client's data is accessed
- Results are DP-noised before any transmission
- Called only when an analyst explicitly requests it for forensic investigation

**Fidelity validation:** Pearson r = 0.91 between Pathway 1 and Pathway 2 across 15 features (p < 0.001).

---

## Key Results

### CICIoT2023 — Federated, α=0.5, K=8 clients

| Model | Accuracy | Macro F1 | AUC-ROC | Comm. (MB) | Edge (ms) |
|-------|----------|----------|---------|-----------|----------|
| FedAvg + CNN | 83.3% | 79.8% | 0.871 | >840 | 2.1 |
| FedProx + CNN | 84.9% | 81.4% | 0.883 | 789.6 | 2.1 |
| FedAvg + LSTM | 86.1% | 83.2% | 0.896 | 800.8 | 3.8 |
| FedAvg + CNN-Trans | 88.7% | 86.0% | 0.912 | 971.1 | 4.9 |
| FedRep + CNN-Trans | 92.1% | 90.2% | 0.937 | 483.6 | 4.9 |
| **FedCTX-IoT (ours)** | **95.4%** | **94.0%** | **0.961** | **224.1** | **4.1** |

### Non-IID robustness — Accuracy drop from IID to α=0.1

| Model | IID Acc. | α=0.1 Acc. | Drop |
|-------|----------|-----------|------|
| FedAvg + CNN | 87.0% | 72.1% | **14.9 pp** |
| FedRep + CNN-Trans | 95.0% | 83.1% | **11.9 pp** |
| **FedCTX-IoT (ours)** | **95.9%** | **90.2%** | **5.7 pp** |

### Privacy vs accuracy — CICIoT2023

| ε | Macro F1 | MI Attack Advantage |
|---|----------|-------------------|
| 0.1 | 88.3% | 0.4 pp |
| 1.0 | 91.8% | 0.9 pp |
| **3.0** | **94.0%** | **2.3 pp** ← paper default |
| No DP | 95.0% | 5.7 pp |

---

## Project Structure

```
fedctx_iot/
│
├── README.md                    ← this file
├── LICENSE                      ← MIT
├── setup.py                     ← pip installable package
├── requirements.txt             ← all dependencies
├── train.py                     ← main training CLI
├── evaluate.py                  ← evaluation + metrics CLI
│
├── config/
│   └── default.yaml             ← all hyperparameters
│
├── data/
│   ├── __init__.py
│   ├── preprocessing.py         ← IoTPreprocessor class
│   │                               • Loads and concatenates raw CSVs
│   │                               • Cleans infinite/NaN values
│   │                               • Mutual information feature selection
│   │                               • StandardScaler normalisation
│   │                               • Sliding-window sequence construction
│   │                               • Train/val/test split
│   │                               • fit_transform(), transform(), save(), load()
│   └── partitioner.py           ← FederatedPartitioner class
│                                   • IID uniform partitioning
│                                   • Dirichlet non-IID partitioning
│                                   • get_distribution_matrix() for visualisation
│
├── models/
│   ├── __init__.py
│   ├── cnn_backbone.py          ← DepthwiseSeparableConv1d
│   │                               ConvBranch (single scale with residual)
│   │                               MultiScaleCNNBackbone (3 parallel branches)
│   ├── transformer_encoder.py   ← LinformerAttention (O(Nr) complexity)
│   │                               TransformerLayer (pre-norm, FFN)
│   │                               PersonalisedTransformerEncoder
│   │                               get_all_attention_weights() for XAI
│   └── fedctx_model.py          ← FedCTXModel (full model)
│                                   backbone_params() / local_params() split
│                                   backbone_state_dict() / load_backbone_state_dict()
│                                   get_embeddings() for t-SNE
│                                   count_parameters()
│                                   build_model_from_config() factory
│
├── federated/
│   ├── __init__.py
│   ├── client.py                ← FocalLoss (class-imbalance handling)
│   │                               FederatedClient
│   │                               • local_train(): load backbone → train → DP → compress
│   │                               • evaluate(): inference on local data
│   ├── aggregation.py           ← fedavg_backbone(): weighted backbone update
│   │                               aggregate_importance(): global φ = Σ(n_k/n)φ_k
│   └── server.py                ← FederatedServer
│                                   • run(): T-round training loop
│                                   • Client selection (random fraction q)
│                                   • Backbone aggregation + broadcast
│                                   • Checkpointing + metrics logging
│
├── privacy/
│   ├── __init__.py
│   └── dp_mechanism.py          ← DPMechanism
│                                   • clip_and_noise(): DP-SGD step
│                                   • compute_epsilon(): RDP budget accounting
│                                   TopKCompressor
│                                   • compress(): top-k sparsification + error feedback
│                                   • reset(): clear error buffer
│
├── xai/
│   ├── __init__.py
│   ├── attention_xai.py         ← AttentionAttributor (Pathway 1)
│   │                               • compute(): per-sample importance (B, T, F) → (F,)
│   │                               • compute_local_importance(): client-level φ_k
│   └── shap_xai.py              ← FederatedSHAP (Pathway 2)
│                                   • build_explainer(): fit KernelSHAP on background
│                                   • explain(): SHAP values with optional DP noise
│                                   • compare_with_attention(): Pearson r vs Pathway 1
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               ← compute_metrics(): accuracy, F1, precision, recall, AUC
│                                   compute_per_class_metrics(): per-class breakdown
│                                   MembershipInferenceAttack: shadow-model MI attack
│                                   xai_fidelity(): Pearson r between pathways
│
├── experiments/
│   └── run_all.sh               ← full reproduction script
│                                   preprocessing → partitioning → training (6 models)
│                                   → alpha sweep → privacy sweep → evaluation → figures
│
├── figures/
│   ├── fig2_confusion.py        ← normalised confusion matrices
│   ├── fig3_noniid.py           ← macro F1 vs Dirichlet α
│   ├── fig4_convergence.py      ← accuracy + loss over rounds
│   ├── fig5_xai.py              ← attention vs SHAP bar + per-attack heatmap
│   ├── fig_roc.py               ← one-vs-rest ROC per attack class
│   ├── fig_tsne.py              ← t-SNE embedding comparison
│   ├── fig_noniid_dist.py       ← client data distribution heatmap
│   ├── fig_radar.py             ← ablation radar chart (5 metrics)
│   └── fig_privacy_tradeoff.py  ← ε vs F1 + MI attack dual axis
│
└── tests/
    └── test_model.py            ← 20 unit tests (pytest)
                                    TestCNNBackbone (4 tests)
                                    TestTransformerEncoder (2 tests)
                                    TestFedCTXModel (5 tests)
                                    TestDPMechanism (3 tests)
                                    TestTopKCompressor (2 tests)
                                    TestAggregation (3 tests)
                                    TestPartitioner (3 tests)
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|------------|
| GPU | NVIDIA GTX 1080 (8 GB) | NVIDIA A100 (40 GB) |
| RAM | 16 GB | 64 GB |
| Disk | 50 GB | 200 GB |

> **CPU-only mode** works for development and small-scale experiments. Full CICIoT2023 training on CPU is not practical (~weeks).

### Software

```
Python    >= 3.9
PyTorch   >= 2.0.0
CUDA      >= 11.7 (optional)
```

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Installation

### Option 1 — pip install (recommended)

```bash
git clone https://github.com/<your-username>/fedctx-iot.git
cd fedctx-iot
pip install -r requirements.txt
pip install -e .
```

### Option 2 — conda environment

```bash
conda create -n fedctx python=3.10
conda activate fedctx
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

### Verify installation

```bash
python -c "
import torch
from models.fedctx_model import FedCTXModel
model = FedCTXModel(in_features=47, n_classes=8)
x = torch.randn(2, 32, 47)
print('Output shape:', model(x).shape)       # → torch.Size([2, 8])
print('Params:', model.count_parameters())
"
```

Expected output:
```
Output shape: torch.Size([2, 8])
Params: {'backbone_params': 187264, 'local_params': 543752, 'total_params': 731016, 'transmitted_pct': 25.6}
```

---

## Datasets

| Dataset | Samples | Attacks | Devices | Download |
|---------|---------|---------|---------|----------|
| CICIoT2023 | 46.7M | 33 (7 categories) | 105 IoT | [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| Edge-IIoTset | 20.9M | 14 (5 categories) | 61 IIoT | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset) |
| N-BaIoT | 7.1M | 10 | 9 real devices | [UCI ML](https://archive.ics.uci.edu/dataset/442) |

Place downloaded CSV files under:

```
data/raw/
├── ciciot2023/          ← one or more .csv files here
├── edge_iiotset/
└── nbaiot/
```

The preprocessor searches **recursively** — subfolders are fine.

---

## Quick Start

For users who want to run something immediately without downloading full datasets:

```python
import torch
import numpy as np
from models.fedctx_model import FedCTXModel
from xai.attention_xai import AttentionAttributor

# 1. Create model
model = FedCTXModel(in_features=47, n_classes=8)

# 2. Simulate IoT traffic sequences (replace with real data)
X = torch.randn(16, 32, 47)   # 16 sequences, 32 flows, 47 features

# 3. Run inference
logits = model(X)
preds  = logits.argmax(dim=-1)
print("Predictions:", preds.tolist())

# 4. Get real-time feature importance (Pathway 1)
attributor = AttentionAttributor(model)
importance = attributor.compute(X)
print("Top 3 feature indices:", importance.argsort()[::-1][:3])
```

---

## Step-by-Step Usage

### Step 1: Preprocess

```bash
python -m data.preprocessing \
    --dataset    ciciot2023 \
    --raw_dir    data/raw/ciciot2023 \
    --out_dir    data/processed/ciciot2023 \
    --n_features 47 \
    --seq_len    32
```

Output saved:
```
data/processed/ciciot2023/
├── X_train.npy       shape: (N_train, 32, 47)
├── y_train.npy       shape: (N_train,)
├── X_val.npy
├── y_val.npy
├── X_test.npy
├── y_test.npy
├── scaler.pkl        ← fitted StandardScaler
├── label_encoder.pkl ← fitted LabelEncoder
└── features.pkl      ← selected feature names
```

### Step 2: Partition

```bash
# Non-IID (Dirichlet α=0.5)
python -m data.partitioner \
    --processed_dir data/processed/ciciot2023 \
    --out_dir       data/partitioned/ciciot2023/alpha_0.5 \
    --n_clients     8 \
    --alpha         0.5

# IID (set alpha to 0)
python -m data.partitioner \
    --processed_dir data/processed/ciciot2023 \
    --out_dir       data/partitioned/ciciot2023/iid \
    --n_clients     8 \
    --alpha         0
```

Output saved:
```
data/partitioned/ciciot2023/alpha_0.5/
├── client_00/  X.npy  y.npy
├── client_01/  X.npy  y.npy
├── ...
└── client_07/  X.npy  y.npy
```

### Step 3: Train

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

Training logs printed per round:
```
2025-01-15 14:23:01 [INFO] Round 1/100 — 4 clients selected
2025-01-15 14:23:18 [INFO] Round 1 | Acc=0.7341 | F1=0.6812 | Loss=1.4231
2025-01-15 14:23:34 [INFO] Round 2/100 — 4 clients selected
...
2025-01-15 21:47:02 [INFO] Round 100 | Acc=0.9537 | F1=0.9396 | Loss=0.1712
```

### Step 4: Evaluate

```bash
python evaluate.py \
    --checkpoint      results/ciciot2023/fedctx/checkpoint_round_0100.pt \
    --dataset         ciciot2023 \
    --processed_dir   data/processed/ciciot2023 \
    --partitioned_dir data/partitioned/ciciot2023/alpha_0.5 \
    --out_dir         results/ciciot2023/fedctx/eval
```

Output saved:
```
results/ciciot2023/fedctx/eval/
├── eval_results.json     ← all metrics
├── attn_importance.npy   ← Pathway 1 importance
├── shap_values.npy       ← Pathway 2 SHAP
├── y_pred.npy
└── y_prob.npy
```

### Step 5: Generate figures

```bash
cd figures/
for fig in fig2_confusion fig3_noniid fig4_convergence fig5_xai \
           fig_roc fig_tsne fig_noniid_dist fig_radar fig_privacy_tradeoff; do
    python ${fig}.py
done
```

---

## Configuration Reference

All parameters live in `config/default.yaml`. Override any parameter via the YAML or by modifying the config dict in code.

```yaml
dataset:
  name: ciciot2023           # ciciot2023 | edge_iiotset | nbaiot
  sequence_length: 32        # N consecutive flows per sample
  test_split: 0.2            # held-out test fraction
  val_split: 0.1             # validation fraction
  random_seed: 42

federated:
  n_clients: 8               # K (overridden by actual partition count)
  client_fraction: 0.5       # q: fraction selected per round
  n_rounds: 100              # T: total communication rounds
  local_epochs: 5            # E: epochs of local training per round
  dirichlet_alpha: 0.5       # α: non-IID concentration

model:
  cnn_kernels: [3, 5, 7]     # parallel branch kernel sizes
  cnn_embedding_dim: 128     # D: shared embedding dimension
  cnn_dropout: 0.1
  depthwise_separable: true  # use DW-Sep or standard Conv1d
  transformer_layers: 4      # L: number of Transformer layers
  n_heads: 8                 # M: attention heads per layer
  linformer_rank: 64         # r: low-rank projection dimension
  ffn_expansion: 4           # FFN hidden = D × ffn_expansion
  transformer_dropout: 0.1
  n_classes: 8               # C: 2 for binary, 8 for CICIoT2023 multi-class
  classifier_hidden: 256

training:
  optimiser: adamw
  learning_rate: 1.0e-3
  lr_schedule: cosine        # cosine | step | none
  weight_decay: 1.0e-2
  batch_size: 256
  focal_loss_gamma: 2.0      # 0.0 = standard cross-entropy

privacy:
  enabled: true
  epsilon: 3.0               # target ε
  delta: 1.0e-5              # target δ
  clip_norm: 1.0             # C: gradient clipping threshold
  noise_multiplier: 1.1      # σ: noise scale relative to clip_norm

compression:
  enabled: true
  top_k_ratio: 0.1           # s: fraction of gradients transmitted
  error_feedback: true       # accumulate and correct residual gradients

xai:
  attention_pathway: true    # Pathway 1 always on
  shap_pathway: true         # Pathway 2 available on request
  shap_background_samples: 100
  n_top_features: 15         # features shown in XAI figures

output:
  results_dir: results
  checkpoint_freq: 10        # save every N rounds
  tensorboard: true
  log_level: INFO
```

---

## Module Reference

### `data.preprocessing.IoTPreprocessor`

```python
from data.preprocessing import IoTPreprocessor

# Fit and transform training data
preprocessor = IoTPreprocessor(
    dataset="ciciot2023",
    n_top_features=47,
    sequence_length=32
)
X_tr, y_tr, X_val, y_val, X_te, y_te = preprocessor.fit_transform("data/raw/ciciot2023")

# Save fitted state
preprocessor.save("data/processed/ciciot2023")

# Restore and apply to new data
preprocessor = IoTPreprocessor.load("ciciot2023", "data/processed/ciciot2023")
X_new, y_new = preprocessor.transform(new_dataframe)
```

### `data.partitioner.FederatedPartitioner`

```python
from data.partitioner import FederatedPartitioner, save_partitions, load_partitions
import numpy as np

p = FederatedPartitioner(n_clients=8, alpha=0.5, seed=42)
clients = p.partition(X_train, y_train)      # list of (X_k, y_k)

# Visualise distribution
dist = p.get_distribution_matrix(y_train)    # shape: (8, 8)
print("Client 0 class proportions:", dist[0])

save_partitions(clients, "data/partitioned/ciciot2023/alpha_0.5")
clients = load_partitions("data/partitioned/ciciot2023/alpha_0.5")
```

### `models.fedctx_model.FedCTXModel`

```python
from models.fedctx_model import FedCTXModel, build_model_from_config
import yaml, torch

# Direct instantiation
model = FedCTXModel(
    in_features=47, n_classes=8,
    embedding_dim=128, cnn_kernels=[3, 5, 7],
    n_heads=8, n_layers=4, seq_len=32, rank=64
)

# From config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)
model = build_model_from_config(cfg, in_features=47)

# Inspect parameter groups
print(model.count_parameters())
# {'backbone_params': 187264, 'local_params': 543752, ...}

# Get backbone state dict for FL
backbone_sd = model.backbone_state_dict()
model.load_backbone_state_dict(backbone_sd)

# Get embeddings for t-SNE
emb = model.get_embeddings(X_tensor)    # (B, 128)
```

### `privacy.dp_mechanism.DPMechanism`

```python
from privacy.dp_mechanism import DPMechanism, TopKCompressor

# DP-SGD
dp = DPMechanism(clip_norm=1.0, noise_multiplier=1.1, delta=1e-5)
noisy_grads = dp.clip_and_noise(gradients)
epsilon = dp.compute_epsilon(n_samples=50000, batch_size=256)
print(f"ε after 100 rounds: {epsilon:.2f}")

# Gradient compression
compressor = TopKCompressor(sparsity=0.1)
sparse_grads, masks = compressor.compress(gradients)
```

### `xai.attention_xai.AttentionAttributor`

```python
from xai.attention_xai import AttentionAttributor

attributor = AttentionAttributor(model)

# Single sample
importance = attributor.compute(X_tensor)        # (F,) normalised [0,1]

# Client-level importance (averaged over dataloader)
phi_k = attributor.compute_local_importance(dataloader, device, n_batches=10)
```

### `xai.shap_xai.FederatedSHAP`

```python
from xai.shap_xai import FederatedSHAP

shap = FederatedSHAP(
    model=model,
    background=X_train[:100],       # local background only
    device=torch.device("cuda"),
    add_dp_noise=True,
    noise_std=0.01
)
shap.build_explainer()              # fit once, reuse for many samples

vals   = shap.explain(X_test[0])   # (F,) normalised with DP noise
result = shap.compare_with_attention(X_test[0], attn_importance=phi_k)
print(f"Pathway fidelity: r = {result['pearson_r']:.4f}")
```

### `evaluation.metrics`

```python
from evaluation.metrics import (
    compute_metrics, compute_per_class_metrics,
    MembershipInferenceAttack, xai_fidelity
)

# Classification
metrics   = compute_metrics(y_true, y_pred, y_prob)
per_class = compute_per_class_metrics(y_true, y_pred, class_names)

# Privacy
mi = MembershipInferenceAttack(n_shadow_models=4)
result = mi.evaluate(model_fn, X_member, y_member, X_nonmember, y_nonmember)
print(f"MI advantage: {result['advantage']:.4f}")

# XAI fidelity
fid = xai_fidelity(attn_importance, shap_values)
print(f"Pearson r = {fid['pearson_r']:.4f}, p = {fid['p_value']:.2e}")
```

---

## Baselines

Six models are implemented with identical training infrastructure. Select via `--model`:

```bash
python train.py --model fedavg_cnn               # FedAvg + 1D-CNN
python train.py --model fedprox_cnn              # FedProx + 1D-CNN
python train.py --model fedavg_lstm              # FedAvg + LSTM
python train.py --model fedavg_cnn_transformer   # FedAvg + CNN-Transformer
python train.py --model fedrep_cnn_transformer   # FedRep + CNN-Transformer
python train.py --model fedctx                   # FedCTX-IoT (ours)
```

| Model | FL Algorithm | Personalisation | Architecture |
|-------|-------------|-----------------|-------------|
| `fedavg_cnn` | FedAvg | None | Single-scale 1D-CNN |
| `fedprox_cnn` | FedProx | Proximal term | Single-scale 1D-CNN |
| `fedavg_lstm` | FedAvg | None | LSTM |
| `fedavg_cnn_transformer` | FedAvg | None | CNN-Transformer (full global) |
| `fedrep_cnn_transformer` | FedRep | Classifier head | CNN-Transformer |
| **`fedctx`** | **Split-Aggregation** | **Transformer heads** | **Multi-scale CNN-Transformer** |

---

## Explainability API

### Real-time monitoring (Pathway 1)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.fedctx_model import FedCTXModel
from xai.attention_xai import AttentionAttributor

model = FedCTXModel(in_features=47, n_classes=8)
# ... load checkpoint ...

attributor    = AttentionAttributor(model)
X             = torch.randn(1, 32, 47)    # one flow sequence
importance    = attributor.compute(X)     # (47,)

feature_names = ["packet_rate", "flow_duration", "proto_entropy", ...]
top_k         = np.argsort(importance)[::-1][:10]

print("Top 10 features driving this prediction:")
for rank, idx in enumerate(top_k, 1):
    print(f"  {rank:2d}. {feature_names[idx]:<25} {importance[idx]:.4f}")
```

### Forensic investigation (Pathway 2)

```python
from xai.shap_xai import FederatedSHAP

shap_module = FederatedSHAP(model, X_background, device)
shap_vals   = shap_module.explain(X_suspicious_flow, n_samples=200)

# Compare both pathways
result = shap_module.compare_with_attention(
    x=X_suspicious_flow,
    attn_importance=importance
)
print(f"Pathway agreement: Pearson r = {result['pearson_r']:.3f}")
```

---

## Privacy Guarantees

### What is protected

| Data type | Protection |
|-----------|-----------|
| Raw traffic data | Never leaves device |
| Model gradients | DP-SGD: clipping + Gaussian noise |
| Attention importance φ_k | Transmitted (but contains no raw samples) |
| SHAP values | DP noise added before any sharing |
| Transformer attention heads | Never transmitted at all |

### Formal guarantees

The privacy budget tracks (ε, δ)-differential privacy via Rényi differential privacy (RDP) accounting using the moments accountant:

```
After T rounds with:
  q = 0.5   (client sampling rate)
  σ = 1.1   (noise multiplier)
  C = 1.0   (clip norm)
  δ = 1e-5

→ ε ≈ 3.0   (confirmed by compute_epsilon())
```

**Membership inference resistance:** At ε=3.0, a shadow-model attack achieves only 52.3% success rate — a 2.3 percentage point advantage over random guessing (50%).

### Adjusting privacy budget

```yaml
# config/default.yaml
privacy:
  epsilon: 1.0          # stricter: lower F1, better privacy
  noise_multiplier: 1.5 # increase to reduce ε
  clip_norm: 0.5        # reduce to lower sensitivity
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_model.py::TestCNNBackbone -v
pytest tests/test_model.py::TestDPMechanism -v
pytest tests/test_model.py::TestAggregation -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Test coverage

| Test Class | # Tests | What is tested |
|------------|---------|---------------|
| `TestCNNBackbone` | 4 | Output shape, gradient flow, depthwise param reduction, single-kernel mode |
| `TestTransformerEncoder` | 2 | Shape preservation, attention weight storage |
| `TestFedCTXModel` | 5 | Forward shape, parameter group disjointness, state dict roundtrip, param count, embedding shape |
| `TestDPMechanism` | 3 | Norm clipping, noise injection, ε monotonicity |
| `TestTopKCompressor` | 2 | Sparsity ratio, error feedback accumulation |
| `TestAggregation` | 3 | FedAvg shape, weight-sum constraint, importance aggregation |
| `TestPartitioner` | 3 | IID balance, Dirichlet sample count, distribution matrix shape |

---

## Reproducing Paper Results

```bash
# Full reproduction (all baselines, all datasets, all ablations)
bash experiments/run_all.sh
```

The script runs seven sequential steps:

| Step | Description | Est. Time (A100) |
|------|-------------|-----------------|
| 1 | Preprocess 3 datasets | ~2 hours |
| 2 | Partition (4 α values + IID) × 3 datasets | ~30 min |
| 3 | Train 6 models × 3 datasets at α=0.5 | ~36 hours |
| 4 | Non-IID sweep (FedCTX × 5 α × 3 datasets) | ~4 hours |
| 5 | Privacy sweep (8 ε values) | ~2 hours |
| 6 | Evaluate all checkpoints | ~1 hour |
| 7 | Generate all 9 figures | ~10 min |

**Total: ~48 hours on a single NVIDIA A100.**

For a quick sanity check on N-BaIoT only (~2 hours):

```bash
python -m data.preprocessing --dataset nbaiot --raw_dir data/raw/nbaiot --out_dir data/processed/nbaiot
python -m data.partitioner   --processed_dir data/processed/nbaiot --out_dir data/partitioned/nbaiot/alpha_0.5 --n_clients 9
python train.py --dataset nbaiot --model fedctx --partitioned_dir data/partitioned/nbaiot/alpha_0.5 --processed_dir data/processed/nbaiot --out_dir results/nbaiot/fedctx
```

---

## Troubleshooting

**`CUDA out of memory`**
```yaml
training:
  batch_size: 64    # reduce from default 256
model:
  cnn_embedding_dim: 64   # reduce from 128
```

**`FileNotFoundError: No CSV files found`**  
The preprocessor searches recursively for `*.csv`. Verify:
```bash
find data/raw/ciciot2023 -name "*.csv" | head -5
```

**`ValueError: not enough samples for sequence_length`**  
Reduce sequence length or increase minimum client size:
```bash
python -m data.preprocessing ... --seq_len 16
```

**`ImportError: shap`**  
Only required for Pathway 2. Pathway 1 works without it:
```bash
pip install shap
```

**`AssertionError: Weights must sum to 1`** in `fedavg_backbone`  
This is an internal assertion. If you see this, your custom `n_samples` list has a zero-count client. Remove empty clients before calling the server.

---

## Citation

If you use FedCTX-IoT in your research, please cite the paper and the datasets:

```bibtex
@article{AlMadani2025FedCTX,
  author   = {Al-madani, Ali Mansour},
  title    = {{FedCTX-IoT}: A Privacy-Preserving Federated {CNN}-Transformer
              Framework with Dual-Pathway Explainability for Real-Time
              {IoT} Intrusion Detection},
  journal  = {Scientific Reports},
  year     = {2025},
  doi      = {10.XXXX/XXXXXXX}
}

@inproceedings{Neto2023CICIoT,
  author = {Neto, Euclides Carlos Pinto and others},
  title  = {{CICIoT2023}: A Real-Time Dataset and Benchmark for
            Large-Scale Attacks in {IoT} Environments},
  year   = {2023}
}

@article{Ferrag2022EdgeIIoT,
  author  = {Ferrag, Mohamed Amine and others},
  title   = {Edge-{IIoTset}: A New Comprehensive Realistic Cyber
             Security Dataset of {IoT} and {IIoT} Applications},
  journal = {IEEE Access},
  year    = {2022}
}

@article{Meidan2018NBaIoT,
  author  = {Meidan, Yair and others},
  title   = {{N-BaIoT}: Network-Based Detection of {IoT} Botnet
             Attacks Using Deep Autoencoders},
  journal = {IEEE Pervasive Computing},
  year    = {2018}
}
```

---

## Contributing

Contributions are welcome. Please follow these steps:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-contribution
   ```

2. **Write tests** for any new functionality in `tests/test_model.py`.

3. **Run the full test suite** before submitting:
   ```bash
   pytest tests/ -v
   ```

4. **Open a Pull Request** with a clear description of what was changed and why.

**Suggested contribution areas:**

- [ ] Asynchronous federated training (accommodate device duty cycles)
- [ ] Byzantine-robust aggregation (Krum, coordinate-wise median, FLTrust)
- [ ] Additional datasets: TON-IoT, Bot-IoT, UNSW-NB15
- [ ] Real distributed deployment via [Flower (flwr)](https://flower.ai)
- [ ] Differential privacy via [Opacus](https://opacus.ai) integration
- [ ] ONNX export for edge deployment

---

## Contact

**Ali Mansour Al-madani**  
📧 ali.m.almadani1992@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

---

<div align="center">

**If FedCTX-IoT helped your research, please consider giving it a ⭐**

</div>
