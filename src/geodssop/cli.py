"""Command-line interface for GeoDSSOP-PDB."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Sequence

import numpy as np
import torch

from geodssop.config import (
    ESM_REPOSITORY,
    ESM_REVISION,
    METHOD_ID,
    VERSION,
)
from geodssop.esm import extract_esm2_features, save_feature_file
from geodssop.inference import build_inference_graph, predict
from geodssop.io import (
    atomic_write_json,
    read_single_fasta,
    sha256_file,
    sha256_text,
    write_prediction_csv,
)


def _predict_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("predict", help="predict residue-level N-H S2")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True, help="PDBx/mmCIF (.cif or .cif.gz)")
    parser.add_argument("--chain", default="", help="author chain ID; required if mapping is ambiguous")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="optional precomputed ESM feature safetensors")
    parser.add_argument("--esm-cache-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--record-id")
    parser.add_argument("--output-dir", type=Path, required=True)


def _extract_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("extract-esm", help="extract pinned ESM-2 residue features")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--esm-cache-dir", type=Path)


def _inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("inspect-structure", help="validate sequence-to-structure mapping")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--chain", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geodssop",
        description="GeoDSSOP-PDB: residue-level protein backbone N-H order-parameter prediction",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _predict_parser(subparsers)
    _extract_parser(subparsers)
    _inspect_parser(subparsers)
    return parser


def _run_predict(args: argparse.Namespace) -> int:
    fasta_id, sequence = read_single_fasta(args.fasta)
    record_id = args.record_id or fasta_id
    output_dir = args.output_dir
    predictions_path = output_dir / "predictions.csv"
    manifest_path = output_dir / "prediction-manifest.json"
    if predictions_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing prediction artifact")
    result = predict(
        record_id=record_id,
        sequence=sequence,
        structure_path=args.structure,
        weights_dir=args.weights_dir,
        chain_id=args.chain,
        features_path=args.features,
        device=args.device,
        esm_cache_dir=args.esm_cache_dir,
    )
    rows = [
        {
            "record_id": result.record_id,
            "position": index + 1,
            "amino_acid": amino_acid,
            "s2_mean": f"{float(result.s2_mean[index]):.8f}",
            "ensemble_std": f"{float(result.ensemble_std[index]):.8f}",
            "coordinate_available": bool(result.coordinate_available[index]),
        }
        for index, amino_acid in enumerate(sequence)
    ]
    write_prediction_csv(predictions_path, rows)
    manifest = {
        "schema": "geodssop_prediction_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD_ID,
        "software_version": VERSION,
        "record_id": record_id,
        "sequence_length": len(sequence),
        "sequence_sha256": sha256_text(sequence),
        "fasta_filename": args.fasta.name,
        "fasta_sha256": sha256_file(args.fasta),
        "structure_filename": args.structure.name,
        "features_filename": args.features.name if args.features is not None else None,
        "features_sha256": sha256_file(args.features) if args.features is not None else None,
        "esm_repository": ESM_REPOSITORY,
        "esm_revision": ESM_REVISION,
        "esm_fp16_cache_compatibility_quantization": True,
        "structure": dict(result.structure_metadata),
        "ensemble": {
            "strategy": result.weight_manifest["ensemble_strategy"],
            "members": [
                {
                    "seed": member["seed"],
                    "filename": member["filename"],
                    "sha256": member["sha256"],
                }
                for member in result.weight_manifest["members"]
            ],
        },
        "device": args.device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "prediction_file": predictions_path.name,
        "prediction_sha256": sha256_file(predictions_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "record_id": record_id,
        "residues": len(sequence),
        "coordinate_coverage": result.structure_metadata["coordinate_coverage"],
        "predictions": str(predictions_path),
        "manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


def _run_extract(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    _identifier, sequence = read_single_fasta(args.fasta)
    features = extract_esm2_features(
        sequence, device=args.device, cache_dir=args.esm_cache_dir
    )
    save_feature_file(args.output, features, sequence)
    print(json.dumps({
        "status": "complete",
        "residues": len(sequence),
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }, sort_keys=True))
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    _identifier, sequence = read_single_fasta(args.fasta)
    _graph, metadata = build_inference_graph(
        sequence, args.structure, chain_id=args.chain
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "extract-esm":
        return _run_extract(args)
    if args.command == "inspect-structure":
        return _run_inspect(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
