# GeoDSSOP

**GeoDSSOP-PDB** is a geometry-aware deep sequence–structure model for
residue-level prediction of protein backbone N–H order parameters, \(S^2\).
The method combines frozen ESM-2 residue representations with a GVP-style
encoder of a single protein structure, gated sequence–structure fusion, and a
multiscale dilated temporal-convolution head.

The internal development identifier **B4-PDB** maps exactly to the released
model **GeoDSSOP-PDB**. In text where the structure source is unambiguous, the
method is shortened to **GeoDSSOP**.

> Release status: public research preview (`v0.1.0`). The code, frozen result
> tables, minimal example, and release weights are openly accessible.

## What is released

- A portable Python package and command-line interface.
- The exact P2-v2 structure graph and GeoDSSOP-PDB model contracts.
- A three-seed W3 weight manifest; inference-only `safetensors` are attached to
  GitHub release `v0.1.0` and kept out of Git history.
- A self-contained 1UBQ example with precomputed, sequence-verified ESM-2
  features.
- Label-free accession, split, homology-group, and sequence-hash metadata for
  all 4,150 registered proteins.
- Machine-readable frozen result tables used by the manuscript.
- Portable training, evaluation, and plotting entry points for locally prepared
  cached data.
- Unit tests and an optional weight-backed end-to-end regression test.

Weak-label values, experimental labels, MD trajectories, copyrighted papers,
and third-party data without an established redistribution right are not
included. See [data/README.md](data/README.md).

## Model at a glance

```text
protein sequence ── ESM-2 t33 650M (frozen; 1280/residue) ── linear 128 ─┐
                                                                         ├─ gated fusion
single structure ─ P2-v2 residue graph ─ GVP-style encoder (3 layers) ──┘
       │
       ├─ 40 scalar node channels + 1 vector node channel
       └─ sequence ±1/±2 edges ∪ spatial Cα edges (12 Å, max 16/neighborhood)

gated representation ─ dilated LiteTCN (1,2,4,8,16) ─ sigmoid ─ residue S²
```

The model has 1,116,753 trainable parameters after ESM features are cached.
The released point estimate is the equal arithmetic mean of three seed models;
the reported uncertainty is their population standard deviation (`ddof=0`).

## Frozen headline results

| Evaluation | Proteins | Labels | Protein-macro PCC | Macro RMSE |
|---|---:|---:|---:|---:|
| Preregistered independent MD-iRED test | 136 | 17,483 | 0.8250 | 0.1115 |
| Historical NMR10 benchmark | 10 | 709 | 0.8424 | 0.1320 |

On the matched structural-branch validation ablation, the mean-of-seed
MD-iRED macro PCC was 0.80263 for full GeoDSSOP and 0.72199 for NoStruct.
ShuffleEdge gave 0.79972, so the experiment supports a contribution from the
structural branch but did not detect an additional benefit of the tested
long-range edge topology. These aggregation definitions are not interchangeable
with the prediction-ensemble values above. Full tables and evidence boundaries
are in [results/final_results_ledger](results/final_results_ledger).

## Installation

Python 3.9 and the locked CUDA 11.8 environment are the reference setup. CPU
inference is supported but ESM-2 feature extraction is much slower and requires
several gigabytes of memory.

```bash
conda env create -f environment.yml
conda activate geodssop
pip install -e .
```

For a platform-specific PyTorch installation, install PyTorch first and then:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[esm,dev]"
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for tested versions and a
smaller inference-only setup.

## Obtain the W3 weights

The three inference files are release assets, not Git objects:

```bash
python scripts/download_weights.py --output-dir checkpoints
```

The public release can be downloaded anonymously. An optional GitHub token may
be supplied through the standard `GITHUB_TOKEN` environment variable for API
rate-limit management or use with a private fork. Never place a token in this
repository or in a command committed to shell history. Every file is verified
against [checkpoints/weights-manifest.json](checkpoints/weights-manifest.json).

## Run the minimal example

The bundled ESM feature file makes this example independent of a 2.6 GB ESM-2
download while still verifying the input sequence hash:

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

Compare the result with
[expected_predictions.csv](examples/minimal_example/expected_predictions.csv).
The frozen CSV is the CPU reference. Repeated CPU validation uses a `2e-6`
per-residue tolerance; CPU-CUDA comparison uses `2e-5`.

To extract ESM-2 features from the FASTA instead:

```bash
geodssop extract-esm \
  --fasta examples/minimal_example/1ubq.fasta \
  --device cuda:0 \
  --output 1ubq-features.safetensors
```

Then provide that file through `--features`, or omit `--features` from
`geodssop predict` to run extraction in memory.

## Input contract

GeoDSSOP-PDB requires:

1. Exactly one protein sequence in FASTA format.
2. One structure in PDBx/mmCIF (`.cif` or `.cif.gz`) format.
3. An author chain ID when sequence-based chain selection is ambiguous.

The sequence-to-structure alignment must have at least 0.95 identity and 0.95
query coverage. Missing coordinates are retained as canonical sequence nodes;
those positions follow the sequence-path fallback. Structure B-factors,
occupancies, resolution, experimental method, ligands, waters, labels, and
dataset-role fields are not model features.

Use the read-only mapping check before a long prediction:

```bash
geodssop inspect-structure \
  --fasta protein.fasta \
  --structure protein.cif.gz \
  --chain A
```

## Output contract

`predictions.csv` contains:

- `position`: one-based position in the input FASTA;
- `amino_acid`: normalized input residue;
- `s2_mean`: W3 ensemble point prediction in `[0,1]`;
- `ensemble_std`: three-member population standard deviation;
- `coordinate_available`: whether a mapped Cα coordinate was present.

`prediction-manifest.json` records hashes, model members, structure mapping,
software versions, and the output checksum. It stores input filenames rather
than private absolute paths.

## Training and evaluation

The paper model used equal-source batches of two MD-iRED and two S-OPPE
proteins, protein-normalized Huber loss (`delta=0.10`), AdamW, and early
stopping on equal-source validation Huber. The complete path-neutral protocol
is [configs/training-w3-v1.yml](configs/training-w3-v1.yml).

For locally prepared caches that follow [docs/TRAINING.md](docs/TRAINING.md):

```bash
python scripts/train_geodssop.py \
  --manifest local-data/cached-records.jsonl \
  --seed 20260801 \
  --device cuda:0 \
  --run-dir runs/seed20260801

python scripts/evaluate_geodssop.py \
  --manifest local-data/cached-records.jsonl \
  --split test \
  --weights-dir checkpoints \
  --device cuda:0 \
  --output-dir evaluation/test

pip install -e ".[plot]"
python scripts/plot_predictions.py \
  --predictions evaluation/test/predictions.csv \
  --output evaluation/test/observed-vs-predicted.png
```

The public cached-data interface is intentionally separate from protected test
labels. Recreating the exact historical run additionally requires the
source-controlled data release identified in the provenance files; unavailable
supervision files are never silently substituted.

## Tests

```bash
pytest
```

To add the weight-backed end-to-end test:

```bash
GEODSSOP_TEST_WEIGHTS=checkpoints pytest -m e2e
```

On Windows PowerShell:

```powershell
$env:GEODSSOP_TEST_WEIGHTS = "checkpoints"
pytest -m e2e
```

## Scope and limitations

- This release predicts backbone N–H \(S^2\); it is not a general flexibility
  score and should not be interpreted as a thermodynamic observable for every
  residue type or timescale.
- The released model was validated with experimental PDB coordinates. Use with
  AlphaFold or other predicted structures is plausible but has not yet been
  established as the validated GeoDSSOP-AF model.
- The NMR10 set is a historical external benchmark, not a new blind test.
- The NMR26 set is related to the OPPE/S-OPPE teacher lineage and is treated as
  exploratory rather than independent confirmation.
- Low-variance proteins make per-protein PCC unstable; consult RMSE, MAE, and
  the number of valid labels alongside PCC.
- Ensemble spread describes disagreement among three training seeds; it is not
  calibrated experimental uncertainty.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for intended use and known failure
modes.

## Licensing and citation

GeoDSSOP code is released under the [MIT License](LICENSE). The bundled PDB 1UBQ
data file is from the wwPDB archive and is available under CC0; attribution is
retained in the example documentation. ESM-2 code and model files are MIT
licensed and are downloaded from the pinned upstream revision. Other datasets
retain their original terms and are represented here by metadata only.

Until the manuscript bibliographic record is final, cite the software using
[CITATION.cff](CITATION.cff) and include the frozen release tag/commit.

## Reproducibility identity

- Paper method: `GeoDSSOP-PDB` (`B4-PDB` during development)
- Training code commit: `0f008bb9ab944d7a1523b5a89c0dd19411fe7146`
- Evaluation code commit: `dc0f4a36f43c3dd4a83b6fa5a10e08738028203c`
- Original training config SHA-256:
  `845c16898ccab6d5e16cecf2d1b165669ea8a0200174352b496f876b8204237c`
- Original graph config SHA-256:
  `588e1b1a6e9fdc0cd30b7b4a6ffceaaabfb01fed13f2f90f4e296967960ac248`

The public release audit is recorded in
[docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md).
