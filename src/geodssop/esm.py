"""Pinned ESM-2 residue feature extraction and safe feature serialization."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from geodssop.config import ESM_HIDDEN_DIM, ESM_MAX_RESIDUES, ESM_REPOSITORY, ESM_REVISION
from geodssop.io import sha256_text


FEATURE_TENSOR_NAME = "features"
FEATURE_SCHEMA = "geodssop_esm2_features_v1"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def extract_esm2_features(
    sequence: str,
    *,
    device: str | torch.device = "auto",
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    """Return cache-compatible ESM-2 layer-33 features as CPU float32.

    The float32 representation is quantized through float16 because the frozen
    GeoDSSOP checkpoints were trained from the immutable FP16 ESM cache.
    """
    if len(sequence) > ESM_MAX_RESIDUES:
        raise ValueError(
            f"sequence has {len(sequence)} residues; the frozen ESM-2 contract allows at most {ESM_MAX_RESIDUES}"
        )
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError as error:  # pragma: no cover - exercised by install extras
        raise RuntimeError("install GeoDSSOP with the 'esm' extra to extract ESM-2 features") from error
    target = resolve_device(str(device))
    common = {
        "pretrained_model_name_or_path": ESM_REPOSITORY,
        "revision": ESM_REVISION,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(**common)
    model = EsmModel.from_pretrained(**common, add_pooling_layer=False)
    model.requires_grad_(False).eval().to(target)
    tokenized = tokenizer(
        sequence,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    expected_tokens = len(sequence) + 2
    if tuple(tokenized["input_ids"].shape) != (1, expected_tokens):
        raise ValueError("ESM tokenizer did not preserve one token per residue")
    with torch.inference_mode():
        output = model(
            input_ids=tokenized["input_ids"].to(target),
            attention_mask=tokenized["attention_mask"].to(target),
            return_dict=True,
        )
        online = output.last_hidden_state[0, 1:-1].float().cpu().contiguous()
    if tuple(online.shape) != (len(sequence), ESM_HIDDEN_DIM):
        raise RuntimeError("ESM-2 representation has an unexpected shape")
    return online.to(torch.float16).to(torch.float32)


def save_feature_file(path: str | Path, features: torch.Tensor, sequence: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = features.detach().cpu().to(torch.float16).contiguous()
    if tuple(value.shape) != (len(sequence), ESM_HIDDEN_DIM):
        raise ValueError("feature tensor and sequence lengths differ")
    save_file(
        {FEATURE_TENSOR_NAME: value},
        str(destination),
        metadata={
            "schema": FEATURE_SCHEMA,
            "sequence_sha256": sha256_text(sequence),
            "repository": ESM_REPOSITORY,
            "revision": ESM_REVISION,
            "storage_dtype": "float16",
        },
    )


def load_feature_file(path: str | Path, sequence: str) -> torch.Tensor:
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata.get("schema") != FEATURE_SCHEMA:
            raise ValueError("unsupported ESM feature schema")
        if metadata.get("sequence_sha256") != sha256_text(sequence):
            raise ValueError("ESM features do not match the input sequence")
        value = handle.get_tensor(FEATURE_TENSOR_NAME)
    if tuple(value.shape) != (len(sequence), ESM_HIDDEN_DIM):
        raise ValueError("ESM feature tensor has an unexpected shape")
    return value.to(torch.float32)


__all__ = [
    "extract_esm2_features",
    "load_feature_file",
    "resolve_device",
    "save_feature_file",
]
