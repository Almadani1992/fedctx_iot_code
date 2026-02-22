"""
xai/shap_xai.py
================
Pathway 2: On-demand federated SHAP analysis.

Uses KernelSHAP on the local model with the local feature
distribution as background. SHAP values are never automatically
transmitted — they are generated only on analyst request and
subject to the same DP noise as model gradients.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FederatedSHAP:
    """
    On-demand SHAP explanation for a single client's model.

    Parameters
    ----------
    model      : FedCTXModel
    background : np.ndarray, shape (n_bg, T, F) — local background dataset
    device     : torch.device
    add_dp_noise : bool — add DP noise to SHAP values before any transmission
    noise_std    : float — standard deviation of DP noise
    """

    def __init__(
        self,
        model:        nn.Module,
        background:   np.ndarray,
        device:       torch.device,
        add_dp_noise: bool  = True,
        noise_std:    float = 0.01,
    ) -> None:
        try:
            import shap
            self._shap = shap
        except ImportError:
            raise ImportError(
                "SHAP is required for Pathway 2. "
                "Install with: pip install shap"
            )

        self.model        = model
        self.background   = background
        self.device       = device
        self.add_dp_noise = add_dp_noise
        self.noise_std    = noise_std

        self._explainer: Optional[object] = None

    def build_explainer(self) -> None:
        """Fit KernelSHAP on the background dataset."""
        logger.info(
            "Building KernelSHAP explainer on %d background samples...",
            len(self.background)
        )
        bg_flat = self.background.reshape(len(self.background), -1)

        def predict_fn(x_flat: np.ndarray) -> np.ndarray:
            T, F = self.background.shape[1], self.background.shape[2]
            x = torch.tensor(
                x_flat.reshape(-1, T, F), dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()

        self._explainer = self._shap.KernelExplainer(predict_fn, bg_flat)
        logger.info("KernelSHAP explainer ready.")

    def explain(
        self,
        x:         np.ndarray,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Compute SHAP values for a single sample.

        Parameters
        ----------
        x        : np.ndarray, shape (T, F) or (1, T, F)
        n_samples : number of KernelSHAP coalition samples

        Returns
        -------
        shap_values : np.ndarray, shape (F,) — aggregated over T, classes
        """
        if self._explainer is None:
            self.build_explainer()

        if x.ndim == 3:
            x = x[0]

        T, F = x.shape
        x_flat = x.reshape(1, -1)

        shap_vals = self._explainer.shap_values(x_flat, nsamples=n_samples)

        if isinstance(shap_vals, list):
            shap_arr = np.stack(shap_vals, axis=0).mean(axis=0)
        else:
            shap_arr = shap_vals

        shap_feat = shap_arr.reshape(T, F).mean(axis=0)

        if self.add_dp_noise:
            shap_feat += np.random.normal(0, self.noise_std, shap_feat.shape)

        norm = np.abs(shap_feat).max() + 1e-9
        return shap_feat / norm

    def compare_with_attention(
        self,
        x:           np.ndarray,
        attn_importance: np.ndarray,
        n_samples:   int = 100,
    ) -> dict:
        """
        Compare SHAP and attention importances using Pearson correlation.

        Returns a dict with shap_values, pearson_r, p_value.
        """
        from scipy.stats import pearsonr
        shap_vals = self.explain(x, n_samples)
        r, p = pearsonr(attn_importance, shap_vals)
        return {
            "shap_values":   shap_vals,
            "attn_values":   attn_importance,
            "pearson_r":     float(r),
            "p_value":       float(p),
        }
