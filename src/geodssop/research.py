"""Portable cached-data contract used by the public training/evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from geodssop.batch import CachedGraphFeatureBatch, collate_graph_items
from geodssop.esm import load_feature_file
from geodssop.io import sha256_text


ALLOWED_SOURCES = ("md_ired", "soppe_legacy")
ALLOWED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class CachedLabeledRecord:
    record_id: str
    source: str
    split: str
    sequence: str
    feature_path: Path
    graph_path: Path
    target_path: Path


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def load_cached_manifest(path: str | Path) -> list[CachedLabeledRecord]:
    """Load the documented JSONL contract without reading labels yet."""
    source = Path(path).resolve()
    result: list[CachedLabeledRecord] = []
    identifiers: set[str] = set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        required = {
            "record_id", "source", "split", "sequence", "sequence_sha256",
            "feature_path", "graph_path", "target_path",
        }
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"manifest line {line_number} is missing {missing}")
        record_id = str(item["record_id"])
        if record_id in identifiers:
            raise ValueError(f"duplicate record_id: {record_id}")
        identifiers.add(record_id)
        sequence = str(item["sequence"]).upper()
        if sha256_text(sequence) != str(item["sequence_sha256"]):
            raise ValueError(f"sequence SHA mismatch for {record_id}")
        if item["source"] not in ALLOWED_SOURCES or item["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported source/split for {record_id}")
        result.append(CachedLabeledRecord(
            record_id=record_id,
            source=str(item["source"]),
            split=str(item["split"]),
            sequence=sequence,
            feature_path=_resolve(source.parent, str(item["feature_path"])),
            graph_path=_resolve(source.parent, str(item["graph_path"])),
            target_path=_resolve(source.parent, str(item["target_path"])),
        ))
    if not result:
        raise ValueError("cached-data manifest is empty")
    return result


def load_record_item(record: CachedLabeledRecord) -> dict[str, object]:
    features = load_feature_file(record.feature_path, record.sequence)
    with np.load(record.graph_path, allow_pickle=False) as archive:
        graph = {name: archive[name].copy() for name in archive.files}
    with np.load(record.target_path, allow_pickle=False) as archive:
        if set(archive.files) != {"s2", "mask"}:
            raise ValueError(f"target file for {record.record_id} must contain only s2 and mask")
        target = np.asarray(archive["s2"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=bool)
    length = len(record.sequence)
    if target.shape != (length,) or mask.shape != (length,):
        raise ValueError(f"target shape mismatch for {record.record_id}")
    if not np.isfinite(target[mask]).all():
        raise ValueError(f"non-finite target for {record.record_id}")
    return {
        "record_id": record.record_id,
        "sequence": record.sequence,
        "features": features,
        "graph": graph,
        "target": torch.from_numpy(target),
        "target_mask": torch.from_numpy(mask),
        "source": record.source,
    }


def collate_labeled_records(
    records: Sequence[CachedLabeledRecord],
    *,
    device: str | torch.device,
) -> tuple[CachedGraphFeatureBatch, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    items = [load_record_item(record) for record in records]
    batch = collate_graph_items(items).to(device)
    maximum = batch.protein.padded_length
    target = torch.zeros((len(items), maximum), dtype=torch.float32, device=device)
    mask = torch.zeros((len(items), maximum), dtype=torch.bool, device=device)
    for index, item in enumerate(items):
        length = len(str(item["sequence"]))
        target[index, :length] = item["target"].to(device)
        mask[index, :length] = item["target_mask"].to(device)
    return batch, target, mask, tuple(str(item["source"]) for item in items)


def source_balanced_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sources: Sequence[str],
    *,
    delta: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    element = functional.huber_loss(prediction, target, reduction="none", delta=delta)
    per_protein = []
    for index in range(prediction.shape[0]):
        selected = mask[index]
        if not selected.any():
            raise ValueError("every training protein must have at least one valid target")
        per_protein.append(element[index, selected].mean())
    stacked = torch.stack(per_protein)
    source_losses = {
        source: stacked[torch.tensor([value == source for value in sources], device=stacked.device)].mean()
        for source in sorted(set(sources))
    }
    if set(source_losses) != set(ALLOWED_SOURCES):
        raise ValueError("each source-balanced batch must contain both training sources")
    total = torch.stack([source_losses[source] for source in ALLOWED_SOURCES]).mean()
    return total, {source: float(value.detach().cpu()) for source, value in source_losses.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_predictions(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    records = list(rows)
    by_record: dict[str, list[dict[str, object]]] = {}
    for row in records:
        by_record.setdefault(str(row["record_id"]), []).append(row)
    per_protein = []
    for record_id, selected in sorted(by_record.items()):
        observed = np.asarray([float(row["observed_s2"]) for row in selected])
        predicted = np.asarray([float(row["predicted_s2"]) for row in selected])
        per_protein.append({
            "record_id": record_id,
            "source": str(selected[0]["source"]),
            "labels": len(selected),
            "pcc": pearson(observed, predicted),
            "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "mae": float(np.mean(np.abs(predicted - observed))),
        })
    finite_pcc = [float(row["pcc"]) for row in per_protein if np.isfinite(float(row["pcc"]))]
    return {
        "proteins": len(per_protein),
        "labels": len(records),
        "macro_pcc": float(np.mean(finite_pcc)) if finite_pcc else None,
        "macro_rmse": float(np.mean([float(row["rmse"]) for row in per_protein])),
        "macro_mae": float(np.mean([float(row["mae"]) for row in per_protein])),
        "per_protein": per_protein,
    }


__all__ = [
    "ALLOWED_SOURCES",
    "CachedLabeledRecord",
    "collate_labeled_records",
    "load_cached_manifest",
    "source_balanced_huber",
    "summarize_predictions",
]
