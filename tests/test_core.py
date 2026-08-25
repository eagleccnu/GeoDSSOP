from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from geodssop.io import normalize_sequence, sha256_file
from geodssop.model import GeoDSSOPModel
from geodssop.weights import load_weight_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_sequence_normalization() -> None:
    assert normalize_sequence(" acd uoz-* \n") == "ACDXXXXX"


def test_frozen_model_contract_and_finite_output(demo_inputs) -> None:
    _record_id, sequence, _graph, _metadata, batch = demo_inputs
    torch.manual_seed(7)
    model = GeoDSSOPModel().eval()
    assert model.trainable_parameter_count == 1_116_753
    with torch.inference_mode():
        prediction = model(batch).s2_mean[0, : len(sequence)]
    assert prediction.shape == (76,)
    assert torch.isfinite(prediction).all()
    assert bool(((prediction >= 0.0) & (prediction <= 1.0)).all())


def test_model_is_rigid_rotation_invariant(demo_inputs) -> None:
    _record_id, sequence, _graph, _metadata, batch = demo_inputs
    rotation = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = replace(
        batch,
        node_vector=torch.einsum("...c,dc->...d", batch.node_vector, rotation),
        edge_vector=torch.einsum("...c,dc->...d", batch.edge_vector, rotation),
    )
    torch.manual_seed(9)
    model = GeoDSSOPModel().eval()
    with torch.inference_mode():
        expected = model(batch).s2_mean
        observed = model(rotated).s2_mean
    assert torch.allclose(expected, observed, atol=2.0e-5, rtol=0.0)


def test_demo_structure_mapping_and_graph_contract(demo_inputs) -> None:
    _record_id, sequence, graph, metadata, _batch = demo_inputs
    assert len(sequence) == 76
    assert metadata["pdb_id"] == "1ubq"
    assert metadata["auth_chain_id"] == "A"
    assert metadata["sequence_identity"] == 1.0
    assert metadata["query_coverage"] == 1.0
    assert metadata["coordinate_coverage"] == 1.0
    assert graph["node_scalar"].shape == (76, 40)
    assert graph["node_vector"].shape == (76, 1, 3)
    assert graph["edge_scalar"].shape[1] == 21
    assert graph["edge_vector"].shape[1:] == (3, 3)
    pairs = [tuple(value) for value in graph["edge_index"].T.tolist()]
    assert len(pairs) == len(set(pairs))


def test_committed_weight_manifest_identity() -> None:
    manifest = load_weight_manifest(ROOT / "checkpoints")
    observed = [member["sha256"] for member in manifest["members"]]
    assert observed == [
        "2f16536f395a763a2f83842c8e605d855c52c3f05d13006e050bfc23fc67171a",
        "cb088fc6ef7f9089eb0b3a4a03f758cfb923ef6a7e055688b2d7eb4580f73809",
        "bf9c3c12c55c8909fd5c1e4be40f97b17c39dcd970f1a1bd6ab4169992792724",
    ]
    summary = json.loads((ROOT / "data" / "manifests" / "split-summary.json").read_text())
    assert summary["records"] == 4150
    assert summary["manifest_sha256"] == sha256_file(
        ROOT / "data" / "manifests" / "split-manifest.csv"
    )


def test_expected_demo_file_is_bounded_and_complete() -> None:
    table = np.genfromtxt(
        ROOT / "examples" / "minimal_example" / "expected_predictions.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert len(table) == 76
    assert np.array_equal(table["position"], np.arange(1, 77))
    assert np.all((table["s2_mean"] >= 0.0) & (table["s2_mean"] <= 1.0))
