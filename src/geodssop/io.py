"""Input and output helpers for GeoDSSOP."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def normalize_sequence(value: str) -> str:
    compact = "".join(value.split()).upper()
    if not compact:
        raise ValueError("protein sequence is empty")
    return "".join(amino_acid if amino_acid in CANONICAL_AA else "X" for amino_acid in compact)


def read_single_fasta(path: str | Path) -> tuple[str, str]:
    source = Path(path)
    identifier = ""
    residues: list[str] = []
    records = 0
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            records += 1
            if records > 1:
                raise ValueError("the prediction CLI accepts exactly one FASTA record")
            identifier = line[1:].strip().split()[0] if line[1:].strip() else source.stem
            continue
        if records == 0:
            records = 1
            identifier = source.stem
        residues.append(line)
    if records != 1:
        raise ValueError("the FASTA file does not contain exactly one record")
    return identifier, normalize_sequence("".join(residues))


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


def write_prediction_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "position",
        "amino_acid",
        "s2_mean",
        "ensemble_std",
        "coordinate_available",
    ]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


__all__ = [
    "atomic_write_json",
    "normalize_sequence",
    "read_single_fasta",
    "sha256_file",
    "sha256_text",
    "write_prediction_csv",
]
