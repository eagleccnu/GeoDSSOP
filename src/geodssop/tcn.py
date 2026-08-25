"""Temporal-convolution block used by the frozen GeoDSSOP-PDB head."""

from __future__ import annotations

import torch
from torch import nn


def _apply_mask(value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return value * attention_mask.unsqueeze(-1).to(dtype=value.dtype)


class ResidualTCNBlock(nn.Module):
    """Two same-length dilated convolutions with per-residue normalization."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            dilation=self.dilation,
            padding=self.dilation,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            dilation=self.dilation,
            padding=self.dilation,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _convolution_stage(
        self,
        value: torch.Tensor,
        convolution: nn.Conv1d,
        attention_mask: torch.Tensor,
        normalization: nn.LayerNorm | None = None,
    ) -> torch.Tensor:
        value = convolution(value.transpose(1, 2)).transpose(1, 2)
        if normalization is not None:
            value = normalization(value)
        value = self.activation(value)
        value = self.dropout(value)
        return _apply_mask(value, attention_mask)

    def forward(self, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = _apply_mask(value, attention_mask)
        value = self._convolution_stage(residual, self.conv1, attention_mask, self.norm1)
        value = self._convolution_stage(value, self.conv2, attention_mask)
        value = self.norm2(residual + value)
        return _apply_mask(value, attention_mask)


__all__ = ["ResidualTCNBlock"]
