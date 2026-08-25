"""Safe loading and identity verification for the released W3 ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch

from geodssop.io import sha256_file
from geodssop.model import GeoDSSOPModel


WEIGHT_SCHEMA = "geodssop_inference_weights_v1"
MANIFEST_SCHEMA = "geodssop_weight_manifest_v1"


def load_weight_manifest(weights_dir: str | Path) -> dict[str, Any]:
    root = Path(weights_dir)
    path = root / "weights-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported GeoDSSOP weight manifest")
    members = payload.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise ValueError("the W3 prediction ensemble requires exactly three members")
    seeds = [int(member["seed"]) for member in members]
    if seeds != [20260801, 20260802, 20260803]:
        raise ValueError("weight manifest does not contain the frozen W3 seeds")
    return payload


def load_ensemble(
    weights_dir: str | Path,
    *,
    device: str | torch.device,
    verify_sha256: bool = True,
) -> tuple[list[GeoDSSOPModel], dict[str, Any]]:
    root = Path(weights_dir)
    manifest = load_weight_manifest(root)
    models: list[GeoDSSOPModel] = []
    for member in manifest["members"]:
        path = root / str(member["filename"])
        if not path.is_file():
            raise FileNotFoundError(
                f"missing ensemble weight {path}; see checkpoints/README.md"
            )
        if verify_sha256 and sha256_file(path) != str(member["sha256"]):
            raise RuntimeError(f"SHA-256 mismatch for {path.name}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if metadata.get("schema") != WEIGHT_SCHEMA:
            raise ValueError(f"unsupported weight schema in {path.name}")
        if int(metadata.get("seed", "-1")) != int(member["seed"]):
            raise ValueError(f"seed metadata mismatch in {path.name}")
        model = GeoDSSOPModel()
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        model.eval().requires_grad_(False).to(device)
        models.append(model)
    return models, manifest


__all__ = [
    "MANIFEST_SCHEMA",
    "WEIGHT_SCHEMA",
    "load_ensemble",
    "load_weight_manifest",
]
