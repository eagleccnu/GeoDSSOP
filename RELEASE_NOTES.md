# GeoDSSOP v0.1.0

This research-preview release freezes the **GeoDSSOP-PDB** implementation used
for the manuscript. The internal development identifier **B4-PDB** maps exactly
to GeoDSSOP-PDB.

## Included

- The sequence–structure model, P2-v2 residue-graph builder, and command-line
  inference interface.
- Frozen W3 configurations and a three-seed ensemble weight manifest.
- A self-contained 1UBQ example with precomputed ESM-2 residue features and
  expected predictions.
- Portable training, evaluation, and plotting entry points.
- Label-free dataset/split metadata and machine-readable frozen result tables.
- Unit tests plus CPU and CUDA end-to-end regression tests.

The three inference-only `safetensors` files are release assets rather than Git
objects. `SHA256SUMS` and `weights-manifest.json` identify each member and its
original training checkpoint.

## Frozen headline results

| Evaluation | Proteins | Protein-macro PCC | Macro RMSE |
|---|---:|---:|---:|
| Preregistered independent MD-iRED test | 136 | 0.8250 | 0.1115 |
| Historical NMR10 benchmark | 10 | 0.8424 | 0.1320 |

The NMR10 set is a historical benchmark rather than a newly prospective blind
test. See `docs/MODEL_CARD.md` and the frozen result ledger for evidence
boundaries, aggregation definitions, and limitations.

## Reproducibility identity

- Training code commit: `0f008bb9ab944d7a1523b5a89c0dd19411fe7146`
- Evaluation code commit: `dc0f4a36f43c3dd4a83b6fa5a10e08738028203c`
- Training configuration SHA-256:
  `845c16898ccab6d5e16cecf2d1b165669ea8a0200174352b496f876b8204237c`
- Graph configuration SHA-256:
  `588e1b1a6e9fdc0cd30b7b4a6ffceaaabfb01fed13f2f90f4e296967960ac248`

## Scope

This version predicts backbone N–H order parameters from a protein sequence and
one experimental structure. AlphaFold-derived inputs are a planned validation
route, not a validated claim of this release. Experimental labels, pseudo-label
values, MD trajectories, and third-party material without established
redistribution rights are intentionally excluded.

