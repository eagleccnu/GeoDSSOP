#!/usr/bin/env python3
"""Evaluate the frozen W3 ensemble on a portable cached-data split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from geodssop.inference import configure_deterministic_inference
from geodssop.io import atomic_write_json, sha256_file
from geodssop.research import (
    collate_labeled_records,
    load_cached_manifest,
    summarize_predictions,
)
from geodssop.weights import load_ensemble


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists; refusing to overwrite")
    args.output_dir.mkdir(parents=True)
    configure_deterministic_inference()
    records = [record for record in load_cached_manifest(args.manifest) if record.split == args.split]
    if not records:
        raise ValueError(f"manifest has no {args.split!r} records")
    ensemble, weight_manifest = load_ensemble(args.weights_dir, device=args.device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for record in records:
            batch, target, mask, sources = collate_labeled_records([record], device=args.device)
            member = torch.stack([model(batch).s2_mean[0] for model in ensemble])
            prediction = member.mean(dim=0)
            uncertainty = member.std(dim=0, unbiased=False)
            valid_positions = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
            for position in valid_positions:
                rows.append({
                    "record_id": record.record_id,
                    "source": sources[0],
                    "position": position + 1,
                    "amino_acid": record.sequence[position],
                    "observed_s2": f"{float(target[0, position].cpu()):.8f}",
                    "predicted_s2": f"{float(prediction[position].cpu()):.8f}",
                    "ensemble_std": f"{float(uncertainty[position].cpu()):.8f}",
                })
    summary = summarize_predictions(rows)
    prediction_path = args.output_dir / "predictions.csv"
    per_protein_path = args.output_dir / "per-protein.csv"
    write_csv(
        prediction_path,
        rows,
        ["record_id", "source", "position", "amino_acid", "observed_s2", "predicted_s2", "ensemble_std"],
    )
    write_csv(
        per_protein_path,
        summary.pop("per_protein"),
        ["record_id", "source", "labels", "pcc", "rmse", "mae"],
    )
    payload = {
        "schema": "geodssop_evaluation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "metrics": summary,
        "aggregation": "unweighted mean across proteins; PCC excludes undefined per-protein values",
        "input_manifest_sha256": sha256_file(args.manifest),
        "weight_members": [
            {"seed": item["seed"], "sha256": item["sha256"]}
            for item in weight_manifest["members"]
        ],
        "predictions_sha256": sha256_file(prediction_path),
        "per_protein_sha256": sha256_file(per_protein_path),
        "device": args.device,
    }
    atomic_write_json(args.output_dir / "summary.json", payload)
    print(json.dumps(payload["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
