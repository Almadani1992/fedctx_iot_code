"""
federated/aggregation.py
=========================
Weighted FedAvg aggregation applied only to CNN backbone parameters.
Transformer heads and classifier are never aggregated.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def fedavg_backbone(
    updates:  List[Dict[str, torch.Tensor]],
    weights:  List[float],
    current_backbone_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Aggregate backbone gradient updates using weighted averaging,
    then apply the aggregate update to the current backbone state dict.

    Parameters
    ----------
    updates  : list of backbone gradient dicts from selected clients
    weights  : normalised client weights (n_k / n_S)
    current_backbone_sd : current global backbone state dict

    Returns
    -------
    new_backbone_sd : updated backbone state dict
    """
    assert len(updates) == len(weights), \
        "Number of updates must equal number of weights."
    assert abs(sum(weights) - 1.0) < 1e-5, \
        "Weights must sum to 1."

    aggregate: Dict[str, torch.Tensor] = {}
    for key in updates[0]:
        aggregate[key] = sum(
            w * u[key] for w, u in zip(weights, updates)
        )

    new_sd = {
        k: current_backbone_sd[k] - aggregate[k]
        for k in current_backbone_sd
    }
    return new_sd


def aggregate_importance(
    importances: List[np.ndarray],
    weights:     List[float],
) -> np.ndarray:
    """
    Federated aggregation of per-client attention importance vectors.

    Φ = Σ_k (n_k / n) φ_k
    """
    return sum(
        w * phi for w, phi in zip(weights, importances)
    )
