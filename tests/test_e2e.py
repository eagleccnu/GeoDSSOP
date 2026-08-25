from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from geodssop.inference import predict
from geodssop.io import read_single_fasta


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "minimal_example"


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.environ.get("GEODSSOP_TEST_WEIGHTS"),
    reason="set GEODSSOP_TEST_WEIGHTS to run the released-weight regression",
)
def test_released_ensemble_matches_frozen_demo() -> None:
    record_id, sequence = read_single_fasta(DEMO / "1ubq.fasta")
    device = os.environ.get("GEODSSOP_TEST_DEVICE", "cpu")
    result = predict(
        record_id=record_id,
        sequence=sequence,
        structure_path=DEMO / "1ubq.cif.gz",
        chain_id="A",
        features_path=DEMO / "1ubq-esm2-features.safetensors",
        weights_dir=Path(os.environ["GEODSSOP_TEST_WEIGHTS"]),
        device=device,
    )
    with (DEMO / "expected_predictions.csv").open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    expected_mean = np.asarray([float(row["s2_mean"]) for row in expected])
    expected_std = np.asarray([float(row["ensemble_std"]) for row in expected])
    tolerance = 2.0e-6 if device == "cpu" else 2.0e-5
    assert np.max(np.abs(result.s2_mean - expected_mean)) <= tolerance
    assert np.max(np.abs(result.ensemble_std - expected_std)) <= tolerance
    assert result.coordinate_available.all()
