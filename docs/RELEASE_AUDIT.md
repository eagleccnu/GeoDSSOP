# Release audit

**Release:** GeoDSSOP v0.1.0

**Audit date:** 2026-08-25

**Status:** **PASS**

**Repository:** `https://github.com/eagleccnu/GeoDSSOP` (private at audit time)

This record covers the isolated public package, the frozen W3 inference
weights, the minimal prediction example, the public result ledger, and the
remote GitHub copy. It does not certify excluded supervision files, raw MD
trajectories, or third-party datasets represented only by accession metadata.

## Release identity

- Package version: `0.1.0`
- Release tag: `v0.1.0`
- Release-source commit verified before the audit record was added:
  `980a6d4d427d7a0ac589615d88124f6f14ffb49e`
- Original training-code commit:
  `0f008bb9ab944d7a1523b5a89c0dd19411fe7146`
- Original evaluation-code commit:
  `dc0f4a36f43c3dd4a83b6fa5a10e08738028203c`
- Original training-config SHA-256:
  `845c16898ccab6d5e16cecf2d1b165669ea8a0200174352b496f876b8204237c`
- Original graph-config SHA-256:
  `588e1b1a6e9fdc0cd30b7b4a6ffceaaabfb01fed13f2f90f4e296967960ac248`

The immutable release tag is the authoritative identity for the final public
tree. `release-manifest.sha256` gives a content checksum for every tracked file
other than the manifest itself.

## Validation results

Validation was performed in a newly created virtual environment under the
isolated WQQ release-validation directory; no historical training directory or
checkpoint was modified.

- [x] Python source and all JSON/CSV configuration artifacts parse.
- [x] `pip check` reports no broken requirements.
- [x] Unit suite: 6 passed, 1 end-to-end test deselected.
- [x] CPU weight-backed 1UBQ regression: 1 passed, 6 unit tests deselected.
- [x] CUDA weight-backed 1UBQ regression: 1 passed, 6 unit tests deselected.
- [x] Repeated CPU predictions satisfy the frozen `2e-6` per-residue tolerance.
- [x] CPU–CUDA predictions agree within `2e-5` (observed maximum difference was
  approximately `1e-5`).
- [x] A one-epoch portable-training smoke test produced a reloadable
  114-tensor `safetensors` checkpoint.
- [x] Portable evaluation and prediction-plot smoke tests completed.
- [x] The isolated-build wheel
  `geodssop-0.1.0-py3-none-any.whl` contained 21 members, was 39,156 bytes,
  and contained no checkpoint or protected-data asset.

## Model-weight provenance

Each inference-only `safetensors` export was reloaded and compared tensor by
tensor with its original training checkpoint. Every tensor was exactly equal.
The original research model and the public package then produced bitwise-equal
CPU predictions for every seed and for the three-member ensemble.

| Release asset | Bytes | SHA-256 |
|---|---:|---|
| `geodssop-pdb-w3-seed20260801.safetensors` | 4,477,940 | `2f16536f395a763a2f83842c8e605d855c52c3f05d13006e050bfc23fc67171a` |
| `geodssop-pdb-w3-seed20260802.safetensors` | 4,477,940 | `cb088fc6ef7f9089eb0b3a4a03f758cfb923ef6a7e055688b2d7eb4580f73809` |
| `geodssop-pdb-w3-seed20260803.safetensors` | 4,477,940 | `bf9c3c12c55c8909fd5c1e4be40f97b17c39dcd970f1a1bd6ab4169992792724` |
| `SHA256SUMS` | 409 | `ea258cc34b9f083bedc7dd494c31c44cd7b54a41a7157f00303c3f9271a7f3a6` |
| `weights-manifest.json` | 1,775 | `7208ebc91f78c2f978207ba5a1fbc6f2c29d88226e44b01781244f88b797fdb3` |

All five assets were uploaded to the authenticated GitHub draft release,
downloaded into a new verification directory, and independently SHA-256
checked against the local release staging files. GitHub's own asset digests
reported the same values.

## Data and result checks

- [x] The public split manifest contains 4,150 records and contains no sequence
  or label values.
- [x] Its SHA-256 is
  `3fed87e5ccc2899d3ab212b2561b351c7aadca9780cec58d55ef4161d86b7dc3`.
- [x] All seven frozen result-ledger CSV files parse and the required headline,
  structural-ablation, and source-ablation values are present.
- [x] The 1UBQ FASTA, structure, precomputed ESM features, and expected output
  are sequence/hash consistent.
- [x] README commands, Markdown relative links, and the package entry points
  were checked.

## Security, size, and licensing checks

- [x] No credential, private key, supplied password, private workstation
  address, private absolute path, or token-shaped string is tracked.
- [x] No `.pt`, `.pth`, `.ckpt`, PDF, document, private key, or model-weight
  asset is tracked in Git.
- [x] The audited pre-release tree contained 62 blobs; the largest tracked file
  was 740,623 bytes and all tracked content totaled approximately 1.15 MB.
- [x] The local tree and private GitHub `main` tree at commit
  `980a6d4d427d7a0ac589615d88124f6f14ffb49e` matched exactly by path and Git
  blob hash.
- [x] Code licensing, the bundled wwPDB example, ESM-2 dependencies, and
  non-redistributed data boundaries are documented in `docs/DATA_AND_LICENSES.md`.

## Acceptance decision

GeoDSSOP v0.1.0 satisfies the planned reproducibility, provenance, regression,
privacy, file-size, licensing, and remote-integrity gates. The repository
remains private pending the manuscript/publication timing decision; changing
visibility requires a separate release-time review but does not require model
retraining.
