"""High-level GeoDSSOP-PDB prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from geodssop.batch import collate_graph_items
from geodssop.config import DEFAULT_GRAPH_CONFIG, ORIGINAL_GRAPH_CONFIG_SHA256
from geodssop.esm import extract_esm2_features, load_feature_file, resolve_device
from geodssop.graph import build_graph_arrays, load_mmcif_dict
from geodssop.io import sha256_file, sha256_text
from geodssop.structure_mapping import chain_candidates, choose_candidate, values
from geodssop.weights import load_ensemble


@dataclass(frozen=True)
class PredictionResult:
    record_id: str
    sequence: str
    s2_mean: np.ndarray
    ensemble_std: np.ndarray
    coordinate_available: np.ndarray
    structure_metadata: Mapping[str, object]
    weight_manifest: Mapping[str, object]


def configure_deterministic_inference() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _model_sort_key(value: str) -> tuple[bool, int | str]:
    return (not value.isdigit(), int(value) if value.isdigit() else value)


def _select_coordinate_model(data: Mapping[str, Any], auth_chain: str, label_chains: set[str]) -> str:
    auth = values(data, "_atom_site.auth_asym_id")
    labels = values(data, "_atom_site.label_asym_id")
    models = values(data, "_atom_site.pdbx_PDB_model_num")
    row_count = min(len(auth), len(labels))
    selected = {
        models[index] if index < len(models) else "1"
        for index in range(row_count)
        if auth[index] == auth_chain and (not label_chains or labels[index] in label_chains)
    }
    if not selected:
        raise ValueError(f"no coordinates found for chain {auth_chain!r}")
    return sorted(selected, key=_model_sort_key)[0]


def build_inference_graph(
    sequence: str,
    structure_path: str | Path,
    *,
    chain_id: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    source = Path(structure_path)
    data = load_mmcif_dict(source)
    selection = choose_candidate(chain_candidates(data), sequence, chain_id.strip())
    if selection.get("status") != "selected":
        detail = selection.get("tied_auth_chains", [])
        raise ValueError(
            f"unable to select a structure chain: {selection.get('status')}; candidates={detail}"
        )
    candidate = selection["candidate"]
    alignment = selection["alignment"]
    if float(alignment["identity"]) < 0.95 or float(alignment["query_coverage"]) < 0.95:
        raise ValueError(
            "sequence/structure alignment is below the frozen 0.95 identity and coverage thresholds"
        )
    selected_model = _select_coordinate_model(
        data, candidate.auth_chain, set(candidate.label_chains)
    )
    structure_sha = sha256_file(source)
    sequence_sha = sha256_text(sequence)
    entry_ids = values(data, "_entry.id")
    pdb_id = entry_ids[0].lower() if entry_ids else source.stem.split(".")[0].lower()
    graph_key = hashlib.sha256(
        f"{sequence_sha}|{structure_sha}|{candidate.auth_chain}|{selected_model}".encode("utf-8")
    ).hexdigest()
    record = {"sequence": sequence}
    task = {
        "mmcif_path": str(source),
        "sequence_sha256": sequence_sha,
        "auth_chain_id": candidate.auth_chain,
        "entity_id": candidate.entity_id,
        "label_chain_ids": ",".join(candidate.label_chains),
        "selected_model": selected_model,
        "graph_key": graph_key,
        "pdb_id": pdb_id,
        "mmcif_sha256": structure_sha,
        "p2_graph_config_sha256": ORIGINAL_GRAPH_CONFIG_SHA256,
        "graph_builder_code_sha256": "release-geodssop-graph-v1",
    }
    graph = build_graph_arrays(record, task, DEFAULT_GRAPH_CONFIG)
    coordinate_count = int(np.asarray(graph["ca_mask"], dtype=bool).sum())
    metadata = {
        "pdb_id": pdb_id,
        "auth_chain_id": candidate.auth_chain,
        "entity_id": candidate.entity_id,
        "label_chain_ids": list(candidate.label_chains),
        "selected_model": selected_model,
        "sequence_identity": float(alignment["identity"]),
        "query_coverage": float(alignment["query_coverage"]),
        "target_coverage": float(alignment["target_coverage"]),
        "coordinate_residues": coordinate_count,
        "coordinate_coverage": coordinate_count / len(sequence),
        "structure_sha256": structure_sha,
        "graph_key": graph_key,
    }
    return graph, metadata


def predict(
    *,
    record_id: str,
    sequence: str,
    structure_path: str | Path,
    weights_dir: str | Path,
    chain_id: str = "",
    features_path: str | Path | None = None,
    device: str = "auto",
    esm_cache_dir: str | Path | None = None,
) -> PredictionResult:
    """Predict residue-level backbone N-H S2 from one sequence and one structure."""
    configure_deterministic_inference()
    target = resolve_device(device)
    features = (
        load_feature_file(features_path, sequence)
        if features_path is not None
        else extract_esm2_features(sequence, device=target, cache_dir=esm_cache_dir)
    )
    graph, structure_metadata = build_inference_graph(
        sequence, structure_path, chain_id=chain_id
    )
    batch = collate_graph_items(
        [{
            "record_id": record_id,
            "sequence": sequence,
            "features": features,
            "graph": graph,
        }]
    ).to(target)
    ensemble, weight_manifest = load_ensemble(weights_dir, device=target)
    with torch.inference_mode():
        members = torch.stack([model(batch).s2_mean[0, : len(sequence)] for model in ensemble])
    mean = members.mean(dim=0).cpu().numpy().astype(np.float32)
    standard_deviation = members.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    coordinate_available = batch.coordinate_mask.cpu().numpy().astype(bool)
    return PredictionResult(
        record_id=record_id,
        sequence=sequence,
        s2_mean=mean,
        ensemble_std=standard_deviation,
        coordinate_available=coordinate_available,
        structure_metadata=structure_metadata,
        weight_manifest=weight_manifest,
    )


__all__ = [
    "PredictionResult",
    "build_inference_graph",
    "configure_deterministic_inference",
    "predict",
]
