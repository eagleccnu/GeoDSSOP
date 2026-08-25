"""Small inference-time tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProteinBatch:
    """Residue layout needed by GeoDSSOP during inference."""

    record_id: tuple[str, ...]
    sequence_length: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.record_id)

    @property
    def padded_length(self) -> int:
        return int(self.attention_mask.shape[1])

    @property
    def device(self) -> torch.device:
        return self.attention_mask.device

    def validate(self) -> None:
        if self.batch_size == 0:
            raise ValueError("protein batch cannot be empty")
        if self.sequence_length.shape != (self.batch_size,):
            raise ValueError("sequence_length must have shape [batch]")
        if self.sequence_length.dtype != torch.int64:
            raise TypeError("sequence_length must use torch.int64")
        if self.attention_mask.ndim != 2 or self.attention_mask.shape[0] != self.batch_size:
            raise ValueError("attention_mask must have shape [batch, length]")
        if self.attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must use bool")
        positions = torch.arange(self.padded_length, device=self.device).unsqueeze(0)
        expected = positions < self.sequence_length.unsqueeze(1)
        if not torch.equal(expected, self.attention_mask):
            raise ValueError("attention mask is not a contiguous sequence prefix")

    def to(self, device: str | torch.device) -> "ProteinBatch":
        target = torch.device(device)
        result = ProteinBatch(
            record_id=self.record_id,
            sequence_length=self.sequence_length.to(target),
            attention_mask=self.attention_mask.to(target),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ResiduePrediction:
    """Bounded per-residue order-parameter predictions."""

    s2_mean: torch.Tensor

    def validate_against(self, batch: ProteinBatch, range_tolerance: float = 1.0e-6) -> None:
        if self.s2_mean.shape != batch.attention_mask.shape:
            raise ValueError("prediction shape does not match padded residue shape")
        if not self.s2_mean.is_floating_point() or self.s2_mean.device != batch.device:
            raise TypeError("prediction dtype/device does not match the batch")
        valid = self.s2_mean[batch.attention_mask]
        if not torch.isfinite(valid).all():
            raise ValueError("valid prediction contains NaN or infinity")
        if torch.any(valid < -range_tolerance) or torch.any(valid > 1.0 + range_tolerance):
            raise ValueError("valid prediction is outside [0, 1]")


__all__ = ["ProteinBatch", "ResiduePrediction"]
