"""
models/fedctx_model.py
=======================
Full FedCTX-IoT model: CNN backbone → Transformer encoder → classifier.

The model exposes two parameter groups:
  - backbone_params()  : CNN parameters  (globally aggregated)
  - local_params()     : Transformer + classifier parameters (kept local)

This split is consumed by federated/client.py and federated/aggregation.py.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from models.cnn_backbone import MultiScaleCNNBackbone
from models.transformer_encoder import PersonalisedTransformerEncoder


class FedCTXModel(nn.Module):
    """
    FedCTX-IoT: Multi-Scale CNN-Transformer for IoT Intrusion Detection.

    Parameters
    ----------
    in_features    : int   — number of input traffic features d*
    n_classes      : int   — number of output classes C
    embedding_dim  : int   — shared embedding dimension D
    cnn_kernels    : list  — kernel sizes for CNN branches
    n_heads        : int   — attention heads
    n_layers       : int   — Transformer layers
    seq_len        : int   — input sequence length N
    rank           : int   — Linformer rank r
    ffn_mult       : int   — FFN expansion factor
    dropout        : float — dropout probability
    """

    def __init__(
        self,
        in_features:   int,
        n_classes:     int,
        embedding_dim: int        = 128,
        cnn_kernels:   List[int]  = [3, 5, 7],
        n_heads:       int        = 8,
        n_layers:      int        = 4,
        seq_len:       int        = 32,
        rank:          int        = 64,
        ffn_mult:      int        = 4,
        dropout:       float      = 0.1,
    ) -> None:
        super().__init__()

        # ── Globally shared component ─────────────────────────────────────
        self.cnn_backbone = MultiScaleCNNBackbone(
            in_features=in_features,
            embedding_dim=embedding_dim,
            kernels=cnn_kernels,
            dropout=dropout,
        )

        # ── Locally adapted components (never transmitted) ────────────────
        self.transformer = PersonalisedTransformerEncoder(
            embed_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            rank=rank,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

        classifier_hidden = embedding_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )

        self._init_weights()

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, T, in_features)
        mask : optional (B, T)

        Returns
        -------
        logits : (B, n_classes)
        """
        h = self.cnn_backbone(x)             # (B, T, D)
        h = self.transformer(h, mask)         # (B, T, D)
        h = h.mean(dim=1)                     # global average pooling → (B, D)
        return self.classifier(h)             # (B, n_classes)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings for t-SNE visualisation."""
        with torch.no_grad():
            h = self.cnn_backbone(x)
            h = self.transformer(h)
            return h.mean(dim=1)              # (B, D)

    # ── Parameter group accessors ─────────────────────────────────────────

    def backbone_params(self) -> Iterator[nn.Parameter]:
        """CNN backbone parameters — globally aggregated."""
        return self.cnn_backbone.parameters()

    def local_params(self) -> Iterator[nn.Parameter]:
        """Transformer + classifier parameters — kept local."""
        return list(self.transformer.parameters()) + \
               list(self.classifier.parameters())

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            f"cnn_backbone.{k}": v
            for k, v in self.cnn_backbone.state_dict().items()
        }

    def load_backbone_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        backbone_sd = {
            k.replace("cnn_backbone.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("cnn_backbone.")
        }
        self.cnn_backbone.load_state_dict(backbone_sd)

    def get_attention_weights(self) -> List[torch.Tensor]:
        """Retrieve attention weights from all Transformer layers."""
        return self.transformer.get_all_attention_weights()

    # ── Utility ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> Dict[str, int]:
        backbone = sum(p.numel() for p in self.cnn_backbone.parameters())
        local    = sum(
            p.numel()
            for p in list(self.transformer.parameters()) +
                     list(self.classifier.parameters())
        )
        return {
            "backbone_params": backbone,
            "local_params":    local,
            "total_params":    backbone + local,
            "transmitted_pct": round(100 * backbone / (backbone + local), 1),
        }


def build_model_from_config(cfg: dict, in_features: int) -> FedCTXModel:
    """Convenience factory from a YAML config dict."""
    m = cfg["model"]
    return FedCTXModel(
        in_features   = in_features,
        n_classes     = m["n_classes"],
        embedding_dim = m["cnn_embedding_dim"],
        cnn_kernels   = m["cnn_kernels"],
        n_heads       = m["n_heads"],
        n_layers      = m["transformer_layers"],
        seq_len       = cfg["dataset"]["sequence_length"],
        rank          = m["linformer_rank"],
        ffn_mult      = m["ffn_expansion"],
        dropout       = m["cnn_dropout"],
    )
