"""Batch construction for ESM residue features and P2-v2 structure graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from geodssop.graph import validate_graph_arrays
from geodssop.types import ProteinBatch


NODE_SCALAR_DIM = 40
NODE_VECTOR_CHANNELS = 1
EDGE_SCALAR_DIM = 21
EDGE_VECTOR_CHANNELS = 3
CA_COORDINATE_MASK_CHANNEL = 37


@dataclass(frozen=True)
class CachedGraphFeatureBatch:
    """Whole-protein ESM features plus packed residue graphs."""

    protein: ProteinBatch
    features: torch.Tensor
    node_scalar: torch.Tensor
    node_vector: torch.Tensor
    node_vector_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_scalar: torch.Tensor
    edge_vector: torch.Tensor
    edge_vector_mask: torch.Tensor
    node_batch_index: torch.Tensor
    node_position: torch.Tensor
    graph_ptr: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.features.device

    @property
    def coordinate_mask(self) -> torch.Tensor:
        return self.node_scalar[:, CA_COORDINATE_MASK_CHANNEL] > 0.5

    def validate(self) -> None:
        self.protein.validate()
        node_count = int(self.node_scalar.shape[0])
        edge_count = int(self.edge_index.shape[1]) if self.edge_index.ndim == 2 else -1
        expected = {
            "features": (self.features, (self.protein.batch_size, self.protein.padded_length, 1280), torch.float32),
            "node_scalar": (self.node_scalar, (node_count, NODE_SCALAR_DIM), torch.float32),
            "node_vector": (self.node_vector, (node_count, NODE_VECTOR_CHANNELS, 3), torch.float32),
            "node_vector_mask": (self.node_vector_mask, (node_count, NODE_VECTOR_CHANNELS), torch.bool),
            "edge_index": (self.edge_index, (2, edge_count), torch.int64),
            "edge_type": (self.edge_type, (edge_count, 2), torch.bool),
            "edge_scalar": (self.edge_scalar, (edge_count, EDGE_SCALAR_DIM), torch.float32),
            "edge_vector": (self.edge_vector, (edge_count, EDGE_VECTOR_CHANNELS, 3), torch.float32),
            "edge_vector_mask": (self.edge_vector_mask, (edge_count, EDGE_VECTOR_CHANNELS), torch.bool),
            "node_batch_index": (self.node_batch_index, (node_count,), torch.int64),
            "node_position": (self.node_position, (node_count,), torch.int64),
            "graph_ptr": (self.graph_ptr, (self.protein.batch_size + 1,), torch.int64),
        }
        for name, (value, shape, dtype) in expected.items():
            if tuple(value.shape) != shape or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"invalid batch tensor: {name}")
        if self.graph_ptr[0].item() != 0 or self.graph_ptr[-1].item() != node_count:
            raise ValueError("graph pointers do not span all nodes")
        if not torch.equal(self.graph_ptr[1:] - self.graph_ptr[:-1], self.protein.sequence_length):
            raise ValueError("each graph must retain every sequence residue")
        if edge_count and (self.edge_index.min().item() < 0 or self.edge_index.max().item() >= node_count):
            raise ValueError("edge index is out of bounds")
        for value in (self.features, self.node_scalar, self.node_vector, self.edge_scalar, self.edge_vector):
            if not torch.isfinite(value).all():
                raise FloatingPointError("batch contains NaN or infinity")

    def to(self, device: str | torch.device) -> "CachedGraphFeatureBatch":
        target = torch.device(device)
        result = CachedGraphFeatureBatch(
            protein=self.protein.to(target),
            features=self.features.to(target, dtype=torch.float32),
            node_scalar=self.node_scalar.to(target),
            node_vector=self.node_vector.to(target),
            node_vector_mask=self.node_vector_mask.to(target),
            edge_index=self.edge_index.to(target),
            edge_type=self.edge_type.to(target),
            edge_scalar=self.edge_scalar.to(target),
            edge_vector=self.edge_vector.to(target),
            edge_vector_mask=self.edge_vector_mask.to(target),
            node_batch_index=self.node_batch_index.to(target),
            node_position=self.node_position.to(target),
            graph_ptr=self.graph_ptr.to(target),
        )
        result.validate()
        return result


def collate_graph_items(items: Sequence[Mapping[str, object]]) -> CachedGraphFeatureBatch:
    """Pack residue graphs while retaining canonical sequence order."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = [len(str(item["sequence"])) for item in items]
    maximum = max(lengths)
    attention = torch.arange(maximum).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    protein = ProteinBatch(
        record_id=tuple(str(item["record_id"]) for item in items),
        sequence_length=torch.tensor(lengths, dtype=torch.int64),
        attention_mask=attention,
    )
    feature_tensors = []
    node_scalar = []
    node_vector = []
    node_vector_mask = []
    edge_index = []
    edge_type = []
    edge_scalar = []
    edge_vector = []
    edge_vector_mask = []
    node_batch_index = []
    node_position = []
    pointers = [0]
    offset = 0
    for batch_index, (item, length) in enumerate(zip(items, lengths)):
        graph = item["graph"]
        if not isinstance(graph, Mapping):
            raise TypeError("graph must be a mapping of NumPy arrays")
        validate_graph_arrays(graph, expected_length=length)
        features = torch.as_tensor(np.asarray(item["features"]), dtype=torch.float32)
        if tuple(features.shape) != (length, 1280):
            raise ValueError("ESM features must have shape [length, 1280]")
        feature_tensors.append(features)
        node_scalar.append(torch.as_tensor(np.asarray(graph["node_scalar"]), dtype=torch.float32))
        node_vector.append(torch.as_tensor(np.asarray(graph["node_vector"]), dtype=torch.float32))
        node_vector_mask.append(torch.as_tensor(np.asarray(graph["node_vector_mask"]), dtype=torch.bool))
        local_edges = torch.as_tensor(np.asarray(graph["edge_index"]), dtype=torch.int64)
        edge_index.append(local_edges + offset)
        edge_type.append(torch.as_tensor(np.asarray(graph["edge_type"]), dtype=torch.bool))
        edge_scalar.append(torch.as_tensor(np.asarray(graph["edge_scalar"]), dtype=torch.float32))
        edge_vector.append(torch.as_tensor(np.asarray(graph["edge_vector"]), dtype=torch.float32))
        edge_vector_mask.append(torch.as_tensor(np.asarray(graph["edge_vector_mask"]), dtype=torch.bool))
        node_batch_index.append(torch.full((length,), batch_index, dtype=torch.int64))
        node_position.append(torch.arange(length, dtype=torch.int64))
        offset += length
        pointers.append(offset)
    result = CachedGraphFeatureBatch(
        protein=protein,
        features=pad_sequence(feature_tensors, batch_first=True, padding_value=0.0),
        node_scalar=torch.cat(node_scalar),
        node_vector=torch.cat(node_vector),
        node_vector_mask=torch.cat(node_vector_mask),
        edge_index=torch.cat(edge_index, dim=1),
        edge_type=torch.cat(edge_type),
        edge_scalar=torch.cat(edge_scalar),
        edge_vector=torch.cat(edge_vector),
        edge_vector_mask=torch.cat(edge_vector_mask),
        node_batch_index=torch.cat(node_batch_index),
        node_position=torch.cat(node_position),
        graph_ptr=torch.tensor(pointers, dtype=torch.int64),
    )
    result.validate()
    return result


__all__ = ["CachedGraphFeatureBatch", "collate_graph_items"]
