# Data boundary

This directory contains metadata needed to understand and reconstruct the
frozen data split. It does not contain supervision values.

## Included

- `manifests/split-manifest.csv`: one row per registered protein with accession,
  source, PDB/chain identity, sequence SHA-256, length, joint homology group,
  canonical role, and optional experimental CV fold.
- `manifests/split-summary.json`: counts and immutable source hashes.
- `demo/README.md`: provenance for the example data stored under `examples`.

The manifest has exactly 4,150 records:

| Canonical role | Records |
|---|---:|
| weak train | 3,490 |
| weak validation | 450 |
| protected MD test metadata only | 136 |
| experimental grouped CV | 26 |
| historical test metadata only | 10 |
| quarantine metadata only | 38 |

Training and validation were separated at the joint homology-group level. The
role column is evidence provenance, not a suggestion that protected labels are
available in this repository.

## Excluded

Sequences, residue targets, trajectories, structure caches, experimental-label
tables, and other third-party files are excluded pending their own access and
redistribution terms. See [the license record](../docs/DATA_AND_LICENSES.md).

## Local reconstruction

After obtaining source data lawfully, produce ESM feature files, graph archives,
and target archives following [docs/TRAINING.md](../docs/TRAINING.md). Use the
record ID and sequence SHA-256 in the split manifest to guard against accidental
reassignment or sequence drift.
