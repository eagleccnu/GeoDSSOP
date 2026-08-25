"""GeoDSSOP-PDB model architecture.

The implementation preserves the parameter names and forward path used by the
frozen B4-PDB training checkpoints.  ``B4-PDB`` is retained only as the legacy
development identifier; the public method name is GeoDSSOP-PDB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from geodssop.batch import CachedGraphFeatureBatch
from geodssop.tcn import ResidualTCNBlock
from geodssop.types import ResiduePrediction


def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value * mask.unsqueeze(-1).to(dtype=value.dtype)


class VectorLinear(nn.Module):
    """Channel mixing only; therefore it is equivariant to 3-D rotations."""

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, 1)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != 3 or value.shape[1] != self.weight.shape[1]:
            raise ValueError("vector tensor does not match VectorLinear channels")
        result = torch.einsum("nvc,ov->noc", value, self.weight)
        return result if self.bias is None else result + self.bias.unsqueeze(0)


class VectorChannelDropout(nn.Module):
    """Drop whole vector channels so train-time dropout preserves vector geometry."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return value
        keep = torch.empty(value.shape[:2] + (1,), device=value.device, dtype=value.dtype).bernoulli_(1.0 - self.probability)
        return value * keep / (1.0 - self.probability)


def vector_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(value.square(), dim=-1), min=1.0e-12))


class GVPMessageLayer(nn.Module):
    """A compact scalar/vector message layer using only invariant scalar readouts."""

    def __init__(self, scalar_dim: int, vector_dim: int, edge_scalar_dim: int, edge_vector_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_scalar = nn.Sequential(nn.Linear(edge_scalar_dim, scalar_dim), nn.GELU(), nn.Linear(scalar_dim, scalar_dim))
        self.source_vector = VectorLinear(vector_dim, vector_dim)
        self.edge_vector = VectorLinear(edge_vector_dim, vector_dim)
        self.message_scalar = nn.Sequential(
            nn.Linear(scalar_dim * 2 + vector_dim, scalar_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(scalar_dim, scalar_dim),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(scalar_dim * 2, scalar_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(scalar_dim, scalar_dim),
        )
        self.scalar_norm = nn.LayerNorm(scalar_dim)
        self.vector_update = VectorLinear(vector_dim, vector_dim)
        self.vector_gate = nn.Linear(scalar_dim, vector_dim)
        self.vector_dropout = VectorChannelDropout(dropout)

    def forward(
        self, scalar: torch.Tensor, vector: torch.Tensor, edge_index: torch.Tensor,
        edge_scalar: torch.Tensor, edge_vector: torch.Tensor, edge_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scalar.ndim != 2 or vector.ndim != 3 or edge_index.shape[0] != 2:
            raise ValueError("invalid GVP layer input rank")
        source, destination = edge_index
        edge_s = self.edge_scalar(edge_scalar)
        geometric_message = self.source_vector(vector[source]) + self.edge_vector(edge_vector)
        message_s = self.message_scalar(torch.cat((scalar[source], edge_s, vector_norm(geometric_message)), dim=-1))
        gate = torch.sigmoid(self.vector_gate(message_s)).unsqueeze(-1)
        message_v = geometric_message * gate
        valid = edge_valid.to(dtype=scalar.dtype).unsqueeze(-1)
        message_s = message_s * valid
        message_v = message_v * valid.unsqueeze(-1)
        node_count = scalar.shape[0]
        summed_s = scalar.new_zeros((node_count, scalar.shape[1]))
        summed_v = vector.new_zeros(vector.shape)
        degree = scalar.new_zeros((node_count, 1))
        summed_s.index_add_(0, destination, message_s)
        summed_v.index_add_(0, destination, message_v)
        degree.index_add_(0, destination, valid)
        degree = degree.clamp_min(1.0)
        aggregate_s = summed_s / degree
        aggregate_v = summed_v / degree.unsqueeze(-1)
        updated_scalar = self.scalar_norm(scalar + self.scalar_update(torch.cat((scalar, aggregate_s), dim=-1)))
        updated_vector = vector + self.vector_dropout(self.vector_update(aggregate_v))
        updated_vector = updated_vector * torch.sigmoid(self.vector_gate(updated_scalar)).unsqueeze(-1)
        return updated_scalar, updated_vector


@dataclass(frozen=True)
class B4PDBConfig:
    esm_input_dim: int = 1280
    hidden_dim: int = 128
    vector_dim: int = 16
    graph_layers: int = 3
    dropout: float = 0.10
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)

    def validate(self) -> None:
        if (self.esm_input_dim, self.hidden_dim, self.vector_dim, self.graph_layers) != (1280, 128, 16, 3):
            raise ValueError("B4-PDB contract is ESM=1280, scalar=128, vector=16, layers=3")
        if self.dropout != 0.10 or self.dilations != (1, 2, 4, 8, 16):
            raise ValueError("B4-PDB dropout/dilations differ from the preregistered contract")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class B4PDBResidueGraphModel(nn.Module):
    """Frozen ESM-2 features fused with a GVP-style residue graph encoder."""

    model_id = "B4_PDB_frozen_ESM2_GVP3_gate_LiteTCN_v1"

    def __init__(self, config: B4PDBConfig | None = None) -> None:
        super().__init__()
        self.config = config or B4PDBConfig()
        self.config.validate()
        self.sequence_projection = nn.Linear(self.config.esm_input_dim, self.config.hidden_dim)
        self.node_scalar_projection = nn.Sequential(nn.Linear(40, self.config.hidden_dim), nn.GELU(), nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
        self.node_vector_projection = VectorLinear(1, self.config.vector_dim)
        self.layers = nn.ModuleList([
            GVPMessageLayer(self.config.hidden_dim, self.config.vector_dim, 21, 3, self.config.dropout)
            for _ in range(self.config.graph_layers)
        ])
        self.structure_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2 + 1, self.config.hidden_dim), nn.GELU(), nn.Dropout(self.config.dropout), nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(self.config.hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualTCNBlock(self.config.hidden_dim, dilation, self.config.dropout)
            for dilation in self.config.dilations
        ])
        self.prediction_head = nn.Linear(self.config.hidden_dim, 1)

    def _sequence(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.esm_input_dim:
            raise ValueError("B4-PDB ESM features must have shape [batch, length, 1280]")
        lengths = attention_mask.sum(dim=1)
        projected = [self.sequence_projection(features[index, :int(length)]) for index, length in enumerate(lengths.tolist())]
        return _masked(pad_sequence(projected, batch_first=True, padding_value=0.0), attention_mask)

    def _structure(self, batch: CachedGraphFeatureBatch) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = self.node_scalar_projection(batch.node_scalar)
        vector = self.node_vector_projection(batch.node_vector * batch.node_vector_mask.unsqueeze(-1).to(batch.node_vector.dtype))
        coordinate_mask = batch.coordinate_mask
        source, destination = batch.edge_index
        edge_valid = coordinate_mask[source] & coordinate_mask[destination]
        edge_vector = batch.edge_vector * batch.edge_vector_mask.unsqueeze(-1).to(batch.edge_vector.dtype)
        for layer in self.layers:
            scalar, vector = layer(scalar, vector, batch.edge_index, batch.edge_scalar, edge_vector, edge_valid)
        scalar = scalar * coordinate_mask.unsqueeze(-1).to(dtype=scalar.dtype)
        # P2 tensors are FP32 even while the immutable ESM cache is stored as FP16.
        # The structural branch therefore owns the dtype of this padded representation.
        padded = scalar.new_zeros((batch.protein.batch_size, batch.protein.padded_length, self.config.hidden_dim))
        padded[batch.node_batch_index, batch.node_position] = scalar
        padded_mask = torch.zeros_like(batch.protein.attention_mask)
        padded_mask[batch.node_batch_index, batch.node_position] = coordinate_mask
        return padded, padded_mask

    @staticmethod
    def _padded_coordinate_mask(batch: CachedGraphFeatureBatch) -> torch.Tensor:
        padded_mask = torch.zeros_like(batch.protein.attention_mask)
        padded_mask[batch.node_batch_index, batch.node_position] = batch.coordinate_mask
        return padded_mask

    def forward(
        self,
        batch: CachedGraphFeatureBatch,
        *,
        structure_enabled: bool = True,
        zero_structure_messages: bool = False,
    ) -> ResiduePrediction:
        batch.validate()
        if zero_structure_messages and not structure_enabled:
            raise ValueError("zero_structure_messages requires the fusion path to be enabled")
        sequence = self._sequence(batch.features.to(dtype=self.sequence_projection.weight.dtype), batch.protein.attention_mask)
        if structure_enabled:
            if zero_structure_messages:
                # G1 B4-NoStruct keeps the full fusion/prediction path and the
                # real missing-coordinate fallback, but fixes every structural
                # message immediately before fusion to exactly zero.  Creating
                # the zero tensor here also excludes structure-projection bias.
                structural = torch.zeros_like(sequence)
                coordinate_mask = self._padded_coordinate_mask(batch)
            else:
                structural, coordinate_mask = self._structure(batch)
                structural = self.structure_projection(structural)
            gate = torch.sigmoid(self.fusion_gate(torch.cat((sequence, structural, coordinate_mask.unsqueeze(-1).to(sequence.dtype)), dim=-1)))
            candidate = self.fusion_norm(sequence + gate * structural)
            value = torch.where(coordinate_mask.unsqueeze(-1), candidate, sequence)
        else:
            value = sequence
        value = _masked(value, batch.protein.attention_mask)
        for block in self.blocks:
            value = block(value, batch.protein.attention_mask)
        prediction = torch.sigmoid(self.prediction_head(value).squeeze(-1))
        prediction = prediction * batch.protein.attention_mask.to(dtype=prediction.dtype)
        result = ResiduePrediction(prediction)
        result.validate_against(batch.protein)
        return result

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


# Public names.  Aliasing rather than subclassing keeps the exact state-dict
# contract of the frozen research checkpoints.
GeoDSSOPConfig = B4PDBConfig
GeoDSSOPModel = B4PDBResidueGraphModel

__all__ = [
    "B4PDBConfig",
    "B4PDBResidueGraphModel",
    "GeoDSSOPConfig",
    "GeoDSSOPModel",
]
