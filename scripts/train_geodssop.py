#!/usr/bin/env python3
"""Train one GeoDSSOP seed from portable cached inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Sequence

from safetensors.torch import save_file
import torch

from geodssop.inference import configure_deterministic_inference
from geodssop.io import atomic_write_json, sha256_file
from geodssop.model import GeoDSSOPModel
from geodssop.research import (
    ALLOWED_SOURCES,
    CachedLabeledRecord,
    collate_labeled_records,
    load_cached_manifest,
    source_balanced_huber,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--proteins-per-source", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--huber-delta", type=float, default=0.10)
    return parser.parse_args(argv)


class SourceCycler:
    def __init__(self, records: Sequence[CachedLabeledRecord], seed: int) -> None:
        self.records = list(records)
        self.random = random.Random(seed)
        self.pool: list[CachedLabeledRecord] = []

    def draw(self, count: int) -> list[CachedLabeledRecord]:
        result = []
        while len(result) < count:
            if not self.pool:
                self.pool = list(self.records)
                self.random.shuffle(self.pool)
            result.append(self.pool.pop())
        return result


def validate(
    model: GeoDSSOPModel,
    records: Sequence[CachedLabeledRecord],
    *,
    device: str,
    delta: float,
) -> dict[str, object]:
    values: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for record in records:
            batch, target, mask, sources = collate_labeled_records([record], device=device)
            prediction = model(batch).s2_mean
            element = torch.nn.functional.huber_loss(
                prediction, target, reduction="none", delta=delta
            )
            values[sources[0]].append(float(element[mask].mean().cpu()))
    means = {source: sum(values[source]) / len(values[source]) for source in ALLOWED_SOURCES}
    return {"source_huber": means, "equal_source_mean_huber": sum(means.values()) / 2.0}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_dir.exists():
        raise FileExistsError("run directory already exists; refusing to overwrite")
    args.run_dir.mkdir(parents=True)
    configure_deterministic_inference()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    records = load_cached_manifest(args.manifest)
    train_by_source = {
        source: [record for record in records if record.split == "train" and record.source == source]
        for source in ALLOWED_SOURCES
    }
    validation_records = [record for record in records if record.split == "validation"]
    if any(not train_by_source[source] for source in ALLOWED_SOURCES):
        raise ValueError("training manifest must contain both sources")
    if any(not any(record.source == source for record in validation_records) for source in ALLOWED_SOURCES):
        raise ValueError("validation manifest must contain both sources")
    model = GeoDSSOPModel().to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    cyclers = {
        source: SourceCycler(train_by_source[source], args.seed + index * 100003)
        for index, source in enumerate(ALLOWED_SOURCES)
    }
    batches_per_epoch = max(
        math.ceil(len(train_by_source[source]) / args.proteins_per_source)
        for source in ALLOWED_SOURCES
    )
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for _batch_index in range(batches_per_epoch):
            selected = []
            for source in ALLOWED_SOURCES:
                selected.extend(cyclers[source].draw(args.proteins_per_source))
            batch, target, mask, sources = collate_labeled_records(selected, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch).s2_mean
            loss, _source_loss = source_balanced_huber(
                prediction, target, mask, sources, delta=args.huber_delta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation = validate(
            model, validation_records, device=args.device, delta=args.huber_delta
        )
        score = float(validation["equal_source_mean_huber"])
        improved = score < best - 1.0e-5
        if improved:
            best = score
            stale = 0
            save_file(
                {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                str(args.run_dir / "best.safetensors"),
                metadata={
                    "schema": "geodssop_inference_weights_v1",
                    "method": "GeoDSSOP-PDB",
                    "seed": str(args.seed),
                    "training_epoch": str(epoch),
                    "training_manifest_sha256": sha256_file(args.manifest),
                },
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_huber": sum(train_losses) / len(train_losses),
            "validation": validation,
            "best_equal_source_mean_huber": best,
            "improved": improved,
        }
        history.append(row)
        atomic_write_json(args.run_dir / "history.json", {"epochs": history})
        print(json.dumps(row, sort_keys=True))
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    atomic_write_json(args.run_dir / "run-manifest.json", {
        "schema": "geodssop_portable_training_run_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_manifest_sha256": sha256_file(args.manifest),
        "train_records": {source: len(train_by_source[source]) for source in ALLOWED_SOURCES},
        "validation_records": len(validation_records),
        "batches_per_epoch": batches_per_epoch,
        "epochs_completed": len(history),
        "best_equal_source_mean_huber": best,
        "best_weights_sha256": sha256_file(args.run_dir / "best.safetensors"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
