"""
models/transformer_encoder.py
==============================
Personalised Transformer encoder with linear-complexity attention.

Uses the Linformer low-rank approximation (Wang et al., 2020) to
reduce self-attention from O(N²) to O(Nr), where r ≪ N.
Attention weight matrices are stored per forward pass for
Pathway 1 explainability in xai/attention_xai.py.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinformerAttention(nn.Module):
    """
    Multi-head linear attention (Linformer approximation).

    Keys and values are projected to rank r using learnable matrices
    E, F ∈ ℝ^{N×r}, reducing per-layer complexity from O(N²d) to O(Nrd).

    Parameters
    ----------
    embed_dim  : int   — model dimension D
    n_heads    : int   — number of attention heads M
    seq_len    : int   — input sequence length N (needed for E, F init)
    rank       : int   — low-rank projection dimension r
    dropout    : float — attention dropout
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads:   int,
        seq_len:   int,
        rank:      int   = 64,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, \
            "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.rank      = rank
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # Low-rank projection matrices (E for keys, F for values)
        self.E = nn.Parameter(torch.randn(seq_len, rank) * 0.02)
        self.F = nn.Parameter(torch.randn(seq_len, rank) * 0.02)

        self.attn_drop = nn.Dropout(dropout)

        # Storage for explainability
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, N, D)
        mask : optional (B, N)

        Returns
        -------
        out : (B, N, D)

        Side-effect
        -----------
        self._attn_weights stores (B, M, N, r) for explainability.
        """
        B, N, D = x.shape

        Q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Low-rank projection of K and V: (B, M, r, head_dim)
        E = self.E[:N, :].unsqueeze(0).unsqueeze(0)   # (1, 1, N, r)
        F = self.F[:N, :].unsqueeze(0).unsqueeze(0)

        K_low = torch.matmul(E.transpose(-2, -1), K)   # (B, M, r, head_dim)
        V_low = torch.matmul(F.transpose(-2, -1), V)   # (B, M, r, head_dim)

        # Scaled dot-product attention with projected K
        scores = torch.matmul(Q, K_low.transpose(-2, -1)) / self.scale
        # scores: (B, M, N, r)

        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1) == 0, float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        self._attn_weights = attn.detach()   # stored for XAI

        # Weighted sum over projected values
        out = torch.matmul(attn, V_low)      # (B, M, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """Last computed attention weights (B, M, N, r)."""
        return self._attn_weights


class TransformerLayer(nn.Module):
    """
    Single Transformer encoder layer with pre-layer normalisation.

    Architecture (pre-norm):
        x = x + MHA(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads:   int,
        seq_len:   int,
        rank:      int   = 64,
        ffn_mult:  int   = 4,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = LinformerAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            seq_len=seq_len,
            rank=rank,
            dropout=dropout,
        )

        ffn_dim = embed_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        return self.attn.attention_weights


class PersonalisedTransformerEncoder(nn.Module):
    """
    Stack of L Transformer layers with personalised attention heads.

    In the split-aggregation FL protocol, the parameters of this
    module are **never transmitted** to the aggregation server —
    they remain local to each client, enabling device-specific
    adaptation without sharing attention patterns.

    Parameters
    ----------
    embed_dim : int   — input/output dimension D
    n_heads   : int   — attention heads per layer
    n_layers  : int   — number of Transformer layers L
    seq_len   : int   — input sequence length N
    rank      : int   — Linformer projection rank r
    ffn_mult  : int   — FFN hidden-layer expansion factor
    dropout   : float — dropout probability

    Input  shape: (B, T, D)
    Output shape: (B, T, D)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads:   int = 8,
        n_layers:  int = 4,
        seq_len:   int = 32,
        rank:      int = 64,
        ffn_mult:  int = 4,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                seq_len=seq_len,
                rank=rank,
                ffn_mult=ffn_mult,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def get_all_attention_weights(self) -> List[torch.Tensor]:
        """
        Return attention weights from all layers.
        Each element has shape (B, M, N, r).
        Used by AttentionAttributor in xai/attention_xai.py.
        """
        return [
            layer.attention_weights
            for layer in self.layers
            if layer.attention_weights is not None
        ]
