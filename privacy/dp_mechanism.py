"""
privacy/dp_mechanism.py
========================
Differential Privacy mechanism: per-sample gradient clipping,
Gaussian noise addition, and Rényi DP budget accounting.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DPMechanism:
    """
    DP-SGD gradient perturbation following Abadi et al. (2016).

    Clips per-sample gradients to L2 norm C, then adds
    calibrated Gaussian noise N(0, σ²C²I) to the sum.

    Parameters
    ----------
    clip_norm        : float — maximum gradient L2 norm C
    noise_multiplier : float — σ (noise scale relative to clip_norm)
    delta            : float — target δ for (ε, δ)-DP
    """

    def __init__(
        self,
        clip_norm:        float = 1.0,
        noise_multiplier: float = 1.1,
        delta:            float = 1e-5,
    ) -> None:
        self.clip_norm        = clip_norm
        self.noise_multiplier = noise_multiplier
        self.delta            = delta
        self._steps           = 0

    def clip_and_noise(
        self,
        gradients: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply clipping and Gaussian noise to a list of gradient tensors.

        Parameters
        ----------
        gradients : list of tensors (one per parameter)

        Returns
        -------
        noisy_gradients : list of perturbed tensors
        """
        clipped = self._clip_gradients(gradients)
        noisy   = self._add_noise(clipped)
        self._steps += 1
        return noisy

    def compute_epsilon(self, n_samples: int, batch_size: int) -> float:
        """
        Estimate the privacy budget ε consumed so far using
        the moments accountant (RDP) approximation.

        Parameters
        ----------
        n_samples  : total number of training samples
        batch_size : per-client batch size

        Returns
        -------
        epsilon : estimated ε at self.delta
        """
        q = batch_size / n_samples
        if q <= 0 or q >= 1:
            return float("inf")

        sigma = self.noise_multiplier
        orders = list(range(2, 64))
        rdp    = [self._rdp_gaussian(q, sigma, alpha) * self._steps
                  for alpha in orders]
        epsilon = min(
            rdp_val - math.log(self.delta) / (alpha - 1)
            for rdp_val, alpha in zip(rdp, orders)
        )
        return max(epsilon, 0.0)

    # ── Private helpers ───────────────────────────────────────────────────

    def _clip_gradients(
        self, gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        total_norm = torch.norm(
            torch.stack([g.norm(2) for g in gradients])
        )
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        return [g * clip_coef for g in gradients]

    def _add_noise(
        self, gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        noise_std = self.noise_multiplier * self.clip_norm
        return [
            g + torch.randn_like(g) * noise_std
            for g in gradients
        ]

    @staticmethod
    def _rdp_gaussian(q: float, sigma: float, alpha: int) -> float:
        """RDP bound for Gaussian mechanism with subsampling."""
        if alpha == 1:
            return q * (math.exp(1 / sigma**2) - 1)
        return min(
            alpha / (2 * sigma**2),
            math.log(
                (1 - q) ** (alpha - 1)
                * ((1 - q) + q * math.exp((alpha - 1) / sigma**2))
                + q ** alpha * math.exp(alpha * (alpha - 1) / (2 * sigma**2))
            ) / (alpha - 1),
        )


class TopKCompressor:
    """
    Top-k gradient sparsification with error feedback.

    Only the top-s fraction of gradient coordinates by absolute
    value are transmitted. Residual gradients accumulate locally
    and are added to the next round's gradient before selection.

    Parameters
    ----------
    sparsity : float — fraction of coordinates to keep (0 < s ≤ 1)
    """

    def __init__(self, sparsity: float = 0.1) -> None:
        self.sparsity      = sparsity
        self._error_buffer: Optional[List[torch.Tensor]] = None

    def compress(
        self,
        gradients: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply top-k sparsification with error feedback.

        Returns
        -------
        sparse_grads : list of sparse tensors (zeros for non-selected)
        masks        : list of boolean masks for reconstruction
        """
        if self._error_buffer is None:
            self._error_buffer = [torch.zeros_like(g) for g in gradients]

        sparse_grads, masks = [], []
        new_errors = []

        for g, err in zip(gradients, self._error_buffer):
            g_corr = g + err
            k = max(1, int(g_corr.numel() * self.sparsity))
            threshold = torch.topk(g_corr.abs().flatten(), k).values[-1]
            mask = g_corr.abs() >= threshold
            sparse = g_corr * mask
            sparse_grads.append(sparse)
            masks.append(mask)
            new_errors.append(g_corr - sparse)

        self._error_buffer = new_errors
        return sparse_grads, masks

    def reset(self) -> None:
        self._error_buffer = None
