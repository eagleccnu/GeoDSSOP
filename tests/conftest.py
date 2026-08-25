from __future__ import annotations

from pathlib import Path

import pytest

from geodssop.batch import collate_graph_items
from geodssop.esm import load_feature_file
from geodssop.inference import build_inference_graph
from geodssop.io import read_single_fasta


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "minimal_example"


@pytest.fixture(scope="session")
def demo_inputs():
    record_id, sequence = read_single_fasta(DEMO / "1ubq.fasta")
    graph, metadata = build_inference_graph(
        sequence, DEMO / "1ubq.cif.gz", chain_id="A"
    )
    features = load_feature_file(DEMO / "1ubq-esm2-features.safetensors", sequence)
    batch = collate_graph_items([{
        "record_id": record_id,
        "sequence": sequence,
        "features": features,
        "graph": graph,
    }])
    return record_id, sequence, graph, metadata, batch
