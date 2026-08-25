# Model weights

The paper model is the three-member W3 ensemble. Weight binaries are attached to
GitHub release `v0.1.0`, not committed to Git.

Download and verify them:

```bash
python scripts/download_weights.py --output-dir checkpoints
```

Expected files:

| Seed | Filename | Bytes | SHA-256 |
|---:|---|---:|---|
| 20260801 | `geodssop-pdb-w3-seed20260801.safetensors` | 4,477,940 | `2f16536f395a763a2f83842c8e605d855c52c3f05d13006e050bfc23fc67171a` |
| 20260802 | `geodssop-pdb-w3-seed20260802.safetensors` | 4,477,940 | `cb088fc6ef7f9089eb0b3a4a03f758cfb923ef6a7e055688b2d7eb4580f73809` |
| 20260803 | `geodssop-pdb-w3-seed20260803.safetensors` | 4,477,940 | `bf9c3c12c55c8909fd5c1e4be40f97b17c39dcd970f1a1bd6ab4169992792724` |

The committed `weights-manifest.json` records original checkpoint hashes,
epochs, steps, and training provenance. Only tensor data and non-sensitive
metadata are stored in the release files; optimizer, sampler, and RNG pickle
states are not distributed.

Do not rename or modify a weight while retaining its manifest entry. Loading
verifies SHA-256 before any tensor is used.
