"""
federated/client.py
====================
Federated client: local training, DP perturbation, and gradient compression.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.fedctx_model import FedCTXModel
from privacy.dp_mechanism import DPMechanism, TopKCompressor
from xai.attention_xai import AttentionAttributor

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced multi-class classification.

    FL(p) = -α_c (1 - p_c)^γ log(p_c)
    """

    def __init__(
        self,
        n_classes:   int,
        gamma:       float = 2.0,
        class_counts: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if class_counts is not None:
            freq   = class_counts / class_counts.sum()
            alpha  = 1.0 / (freq + 1e-6)
            alpha /= alpha.sum()
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.ones(n_classes) / n_classes)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()
        targets_one_hot = nn.functional.one_hot(
            targets, num_classes=logits.size(-1)
        ).float()
        alpha_t = (self.alpha * targets_one_hot).sum(dim=-1)
        p_t     = (prob * targets_one_hot).sum(dim=-1)
        fl      = -alpha_t * (1 - p_t) ** self.gamma * (prob * targets_one_hot).sum(dim=-1).log()
        return fl.mean()


class FederatedClient:
    """
    A single federated client that trains locally and prepares
    backbone gradient updates for the aggregation server.

    Parameters
    ----------
    client_id  : int         — unique client identifier
    model      : FedCTXModel — local model instance
    X          : np.ndarray  — local training data (N, T, F)
    y          : np.ndarray  — local labels (N,)
    cfg        : dict        — full configuration dict
    device     : torch.device
    """

    def __init__(
        self,
        client_id: int,
        model:     FedCTXModel,
        X:         np.ndarray,
        y:         np.ndarray,
        cfg:       dict,
        device:    torch.device,
    ) -> None:
        self.client_id = client_id
        self.model     = model.to(device)
        self.device    = device
        self.cfg       = cfg
        self.n_samples = len(X)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        self.dataset    = TensorDataset(X_t, y_t)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )

        class_counts = np.bincount(y, minlength=cfg["model"]["n_classes"])
        self.criterion = FocalLoss(
            n_classes=cfg["model"]["n_classes"],
            gamma=cfg["training"]["focal_loss_gamma"],
            class_counts=class_counts,
        ).to(device)

        self.dp        = DPMechanism(
            clip_norm=cfg["privacy"]["clip_norm"],
            noise_multiplier=cfg["privacy"]["noise_multiplier"],
            delta=cfg["privacy"]["delta"],
        ) if cfg["privacy"]["enabled"] else None

        self.compressor = TopKCompressor(
            sparsity=cfg["compression"]["top_k_ratio"]
        ) if cfg["compression"]["enabled"] else None

        self.attributor = AttentionAttributor(self.model)

    # ── Public API ────────────────────────────────────────────────────────

    def local_train(
        self,
        global_backbone_sd: Dict[str, torch.Tensor],
        n_epochs: int,
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, float]:
        """
        Load global backbone, run local training, apply DP and compression,
        and return the gradient update.

        Parameters
        ----------
        global_backbone_sd : backbone state dict from server
        n_epochs           : local training epochs

        Returns
        -------
        sparse_update  : compressed backbone gradient
        importance     : local attention importance vector φ_k  (n_features,)
        train_loss     : average training loss this round
        """
        self.model.load_backbone_state_dict(global_backbone_sd)
        self.model.train()

        old_backbone = {
            k: v.clone()
            for k, v in self.model.cnn_backbone.state_dict().items()
        }

        optimiser = self._build_optimiser()
        total_loss = 0.0
        steps = 0

        for _ in range(n_epochs):
            for X_b, y_b in self.dataloader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimiser.zero_grad()
                logits = self.model(X_b)
                loss   = self.criterion(logits, y_b)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)

        # Compute backbone gradient (difference from start of round)
        new_backbone = self.model.cnn_backbone.state_dict()
        gradients = [
            (old_backbone[k] - new_backbone[k]).flatten()
            for k in old_backbone
        ]

        # DP noise
        if self.dp is not None:
            gradients = self.dp.clip_and_noise(gradients)

        # Gradient compression
        sparse_update = None
        if self.compressor is not None:
            gradients, _ = self.compressor.compress(gradients)

        # Reconstruct state-dict format
        sparse_update = {}
        idx = 0
        for k, v in old_backbone.items():
            numel = v.numel()
            sparse_update[k] = gradients[idx].view(v.shape)
            idx += 1

        # Compute local importance for Pathway 1
        importance = self.attributor.compute_local_importance(
            self.dataloader, self.device
        )

        logger.debug(
            "Client %d | loss=%.4f | dp_eps=%.2f",
            self.client_id, avg_loss,
            self.dp.compute_epsilon(self.n_samples,
                                    self.cfg["training"]["batch_size"])
            if self.dp else 0.0,
        )

        return sparse_update, importance, avg_loss

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run inference on (X, y) and return loss, predictions, probabilities."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            loss   = self.criterion(logits, y_t).item()
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=-1)

        return loss, preds, probs

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_optimiser(self) -> torch.optim.Optimizer:
        params = list(self.model.cnn_backbone.parameters()) + \
                 list(self.model.transformer.parameters()) + \
                 list(self.model.classifier.parameters())
        lr  = self.cfg["training"]["learning_rate"]
        wd  = self.cfg["training"]["weight_decay"]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        if self.cfg["training"]["lr_schedule"] == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg["federated"]["n_rounds"]
            )
        return opt
