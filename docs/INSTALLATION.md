# Installation

## Reference environment

The frozen validation environment used:

| Component | Version |
|---|---:|
| Python | 3.9.20 |
| PyTorch | 2.5.1 |
| CUDA runtime reported by PyTorch | 11.8 |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| Biopython | 1.84 |
| safetensors | 0.5.2 |
| transformers | 4.48.3 |

Create it with `environment.yml`, or install a PyTorch build appropriate for
the host and then install GeoDSSOP.

## Inference with precomputed features

This is the smallest supported installation and is sufficient for the bundled
example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Inference with online ESM-2 extraction

```bash
pip install -e ".[esm]"
```

The first run downloads the pinned `facebook/esm2_t33_650M_UR50D` revision.
The upstream model file is approximately 2.6 GB. Use `--esm-cache-dir` when a
specific Hugging Face cache location is required.

## Development and plotting

```bash
pip install -e ".[dev,plot]"
pytest
```

## Determinism

The CLI enables deterministic PyTorch algorithms and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA inference. Exact CPU/GPU bit
patterns are not guaranteed to match. The frozen example is a CPU reference;
same-device regression uses `2e-6`, CPU-CUDA comparison uses `2e-5`, and the
weight-conversion test requires exact tensors.
