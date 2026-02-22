"""
federated/server.py
====================
Aggregation server: orchestrates communication rounds,
client selection, backbone aggregation, and logging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from federated.aggregation import fedavg_backbone, aggregate_importance
from federated.client import FederatedClient
from models.fedctx_model import FedCTXModel
from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Central aggregation server for FedCTX-IoT.

    Parameters
    ----------
    global_model : FedCTXModel — global model instance
    clients      : list of FederatedClient
    X_test, y_test : held-out global test set for round-level evaluation
    cfg          : full configuration dict
    out_dir      : directory for checkpoints and logs
    device       : torch.device
    """

    def __init__(
        self,
        global_model: FedCTXModel,
        clients:      List[FederatedClient],
        X_test:       np.ndarray,
        y_test:       np.ndarray,
        cfg:          dict,
        out_dir:      str | Path,
        device:       torch.device,
    ) -> None:
        self.global_model = global_model.to(device)
        self.clients      = clients
        self.X_test       = X_test
        self.y_test       = y_test
        self.cfg          = cfg
        self.out_dir      = Path(out_dir)
        self.device       = device

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rng              = np.random.default_rng(cfg["dataset"]["random_seed"])
        self.round_metrics:   List[dict] = []
        self.global_importance: Optional[np.ndarray] = None

    # ── Main training loop ────────────────────────────────────────────────

    def run(self) -> None:
        """Execute T communication rounds of FedCTX-IoT training."""
        n_rounds       = self.cfg["federated"]["n_rounds"]
        client_frac    = self.cfg["federated"]["client_fraction"]
        local_epochs   = self.cfg["federated"]["local_epochs"]
        checkpoint_freq = self.cfg["output"]["checkpoint_freq"]

        logger.info(
            "Starting FedCTX-IoT training: %d rounds, %d clients, "
            "fraction=%.2f, local_epochs=%d",
            n_rounds, len(self.clients), client_frac, local_epochs,
        )

        for t in range(1, n_rounds + 1):
            selected = self._select_clients(client_frac)
            logger.info("Round %d/%d — %d clients selected",
                        t, n_rounds, len(selected))

            current_backbone_sd = self.global_model.backbone_state_dict()

            updates, importances, losses, n_samples = [], [], [], []
            for client in selected:
                update, imp, loss = client.local_train(
                    current_backbone_sd, local_epochs
                )
                updates.append(update)
                importances.append(imp)
                losses.append(loss)
                n_samples.append(client.n_samples)

            # Weighted aggregation
            weights = self._compute_weights(n_samples)
            new_backbone_sd = fedavg_backbone(
                updates, weights, current_backbone_sd
            )
            self.global_model.load_backbone_state_dict(new_backbone_sd)

            # Aggregate global importance Φ
            self.global_importance = aggregate_importance(importances, weights)

            # Broadcast updated backbone to all clients
            for client in self.clients:
                client.model.load_backbone_state_dict(new_backbone_sd)

            # Evaluate on global test set
            metrics = self._evaluate()
            metrics["round"]    = t
            metrics["avg_loss"] = float(np.mean(losses))
            self.round_metrics.append(metrics)

            logger.info(
                "Round %d | Acc=%.4f | F1=%.4f | Loss=%.4f",
                t, metrics["accuracy"], metrics["f1_macro"], metrics["avg_loss"]
            )

            # Checkpoint
            if t % checkpoint_freq == 0 or t == n_rounds:
                self._save_checkpoint(t)

        self._save_metrics()
        logger.info("Training complete. Results saved to %s", self.out_dir)

    # ── Private methods ───────────────────────────────────────────────────

    def _select_clients(
        self, fraction: float
    ) -> List[FederatedClient]:
        k = max(1, int(len(self.clients) * fraction))
        return self.rng.choice(self.clients, size=k, replace=False).tolist()

    @staticmethod
    def _compute_weights(n_samples: List[int]) -> List[float]:
        total = sum(n_samples)
        return [n / total for n in n_samples]

    def _evaluate(self) -> dict:
        self.global_model.eval()
        X_t = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.global_model(X_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=-1)
        return compute_metrics(self.y_test, preds, probs)

    def _save_checkpoint(self, round_num: int) -> None:
        path = self.out_dir / f"checkpoint_round_{round_num:04d}.pt"
        torch.save({
            "round":            round_num,
            "model_state_dict": self.global_model.state_dict(),
            "round_metrics":    self.round_metrics,
            "global_importance": self.global_importance,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    def _save_metrics(self) -> None:
        import json
        path = self.out_dir / "training_metrics.json"
        with open(path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)
        if self.global_importance is not None:
            np.save(self.out_dir / "global_importance.npy",
                    self.global_importance)
