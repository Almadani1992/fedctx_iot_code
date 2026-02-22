"""
models/cnn_backbone.py
======================
Multi-scale 1D-CNN backbone with depthwise separable convolutions.

Three parallel branches with kernel sizes k ∈ {3, 5, 7} extract
local traffic features at packet, flow, and session granularities.
Branch outputs are concatenated and projected to a common embedding
dimension D, producing a token sequence for the Transformer encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution.

    Decomposes a standard Conv1d(in, out, k) into:
      1. Depthwise  Conv1d(in, in, k, groups=in)
      2. Pointwise  Conv1d(in, out, 1)

    Parameters reduce from k·in·out  →  k·in + in·out.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        padding:      int = 0,
        bias:         bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConvBranch(nn.Module):
    """
    Single CNN branch: depthwise-separable conv → BN → GELU → residual.

    Input  shape: (B, T, F)   — batch, sequence, features
    Output shape: (B, T, D)   — batch, sequence, embedding_dim
    """

    def __init__(
        self,
        in_features:   int,
        embedding_dim: int,
        kernel_size:   int,
        dropout:       float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2   # same-length output

        self.conv = DepthwiseSeparableConv1d(
            in_channels=in_features,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn      = nn.BatchNorm1d(embedding_dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection when dimensions differ
        self.residual = (
            nn.Conv1d(in_features, embedding_dim, kernel_size=1)
            if in_features != embedding_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → transpose to (B, F, T) for Conv1d
        x_t = x.transpose(1, 2)
        out  = self.conv(x_t)
        out  = self.bn(out)
        out  = self.act(out)
        out  = self.dropout(out)
        res  = self.residual(x_t)
        out  = out + res
        return out.transpose(1, 2)   # (B, T, D)


class MultiScaleCNNBackbone(nn.Module):
    """
    Multi-scale 1D-CNN backbone.

    Three parallel ConvBranch modules with kernel sizes k ∈ {3, 5, 7}
    process the input sequence independently. Their outputs are
    concatenated along the feature axis and projected back to
    ``embedding_dim`` via a learnable linear layer.

    Parameters
    ----------
    in_features : int
        Number of input traffic features d*.
    embedding_dim : int
        Output embedding dimension D (default 128).
    kernels : list of int
        Kernel sizes for each parallel branch.
    dropout : float
        Dropout probability.

    Input  shape: (B, T, in_features)
    Output shape: (B, T, embedding_dim)
    """

    def __init__(
        self,
        in_features:   int,
        embedding_dim: int = 128,
        kernels:       List[int] = [3, 5, 7],
        dropout:       float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.branches = nn.ModuleList([
            ConvBranch(in_features, embedding_dim, k, dropout)
            for k in kernels
        ])

        concat_dim = embedding_dim * len(kernels)
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, T, in_features)

        Returns
        -------
        torch.Tensor, shape (B, T, embedding_dim)
        """
        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=-1)   # (B, T, D*n_branches)
        return self.projection(concat)             # (B, T, D)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim
