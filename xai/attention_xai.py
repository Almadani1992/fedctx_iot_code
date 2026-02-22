"""
xai/attention_xai.py
====================
Pathway 1: Zero-overhead attention-weight feature attribution.

Aggregates attention weights across all heads and layers during
standard inference, then maps token-level importance back to
input features via gradient-weighted attention.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AttentionAttributor:
    """
    Computes feature importance from Transformer attention weights.

    No additional forward passes required — importance is derived
    from weights already computed during inference.

    Parameters
    ----------
    model : FedCTXModel
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def compute(
        self,
        x: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute per-feature importance for a single sample or mini-batch.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F) or (T, F)

        Returns
        -------
        importance : np.ndarray, shape (F,) — normalised to [0, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.requires_grad_(True)
        self.model.eval()

        logits = self.model(x)
        attn_weights = self.model.get_attention_weights()

        if not attn_weights:
            return np.zeros(x.shape[-1])

        # Aggregate across layers and heads: mean over (M, N, r)
        # attn: (B, M, N, r) → mean over M and r → (B, N)
        token_importance = torch.stack([
            a.mean(dim=(1, 3))   # (B, N)
            for a in attn_weights
        ]).mean(dim=0)           # (B, N)

        # Gradient of token importance w.r.t. input features
        # to map from token space back to feature space
        grad = torch.autograd.grad(
            token_importance.sum(), x,
            retain_graph=False, create_graph=False
        )[0]                     # (B, T, F)

        # Feature importance = mean absolute gradient weighted by attention
        feat_importance = (
            grad.abs() * token_importance.unsqueeze(-1)
        ).mean(dim=(0, 1))       # (F,)

        feat_np = feat_importance.detach().cpu().numpy()
        feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-9)
        return feat_np

    def compute_local_importance(
        self,
        dataloader: DataLoader,
        device:     torch.device,
        n_batches:  int = 10,
    ) -> np.ndarray:
        """
        Compute client-level importance φ_k averaged over n_batches.

        Returns
        -------
        phi_k : np.ndarray, shape (F,)
        """
        self.model.eval()
        importances = []

        for i, (X_b, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            X_b = X_b.to(device)
            imp = self.compute(X_b)
            importances.append(imp)

        if not importances:
            return np.zeros(1)

        return np.mean(importances, axis=0)
