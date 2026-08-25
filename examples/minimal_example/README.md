# Minimal example: PDB 1UBQ chain A

This example predicts the 76-residue ubiquitin sequence using the experimental
structure in PDB entry 1UBQ.

```bash
geodssop predict \
  --fasta examples/minimal_example/1ubq.fasta \
  --structure examples/minimal_example/1ubq.cif.gz \
  --chain A \
  --features examples/minimal_example/1ubq-esm2-features.safetensors \
  --weights-dir checkpoints \
  --device cpu \
  --output-dir example-output
```

The bundled ESM file avoids downloading the upstream 650M-parameter encoder for
this smoke test. GeoDSSOP verifies that its embedded sequence SHA-256 matches
the FASTA. The expected output was generated on CPU with the frozen three-seed
W3 ensemble and PyTorch 2.5.1. CPU values must agree within `2e-6`; CUDA values
must agree with the CPU reference within `2e-5`.

Files:

- `1ubq.fasta`: SHA-256 `0629ca1c6496ff15c99ba5a50f71c2de358d236a916175c5e3aa5dbf37f51b04`;
- `1ubq.cif.gz`: SHA-256 `b1a85fb2761c9d2e36734e48706645878036d6c848ebc386611a1957c697d6b3`;
- `1ubq-esm2-features.safetensors`: SHA-256 `bd3acabe1bc16f6720b56c0af6b7bfe21de4f71b668e5611c542c2703c7dc7b1`;
- `expected_predictions.csv`: SHA-256 `6da3f4dda4ef2eb3fb7447f94c6b056ab044656ecc801a80752703d961af4e37`.

The PDB coordinate file is CC0. Cite PDB ID 1UBQ and the original structure
authors where possible. See `docs/DATA_AND_LICENSES.md`.
