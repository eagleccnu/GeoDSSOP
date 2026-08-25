# Inference guide

## 1. Prepare the sequence

Provide one FASTA record. Whitespace is removed, letters are uppercased, and
noncanonical symbols are mapped to `X`. The frozen ESM-2 contract supports at
most 1,022 residues per invocation.

## 2. Prepare the structure

Provide PDBx/mmCIF as plain `.cif` or gzip-compressed `.cif.gz`. GeoDSSOP maps
the polymer sequence in the file to the input FASTA. When multiple chains are
equally compatible, specify the author chain ID explicitly with `--chain`.

The accepted alignment thresholds are:

- sequence identity ≥ 0.95;
- input-query coverage ≥ 0.95.

The first coordinate model is used. Alternate conformers are selected by
highest occupancy, then blank/A/lexical alternate-location priority. Occupancy
is used only for conformer selection and is not passed to the network.

## 3. Inspect the mapping

```bash
geodssop inspect-structure \
  --fasta protein.fasta \
  --structure protein.cif.gz \
  --chain A
```

Review identity, coverage, selected coordinate model, and coordinate coverage.

## 4. Obtain or calculate ESM features

For ordinary use, omit `--features`; the prediction command extracts pinned
ESM-2 features in memory. To reuse features across runs:

```bash
geodssop extract-esm \
  --fasta protein.fasta \
  --device cuda:0 \
  --output protein-esm2.safetensors
```

Feature files contain only a float16 residue tensor and provenance metadata.
The input sequence SHA-256 is checked before use. Values are promoted to
float32 after loading, matching the paper model's training cache.

## 5. Predict

```bash
geodssop predict \
  --fasta protein.fasta \
  --structure protein.cif.gz \
  --chain A \
  --features protein-esm2.safetensors \
  --weights-dir checkpoints \
  --device cuda:0 \
  --output-dir prediction-protein
```

The command refuses to overwrite an existing prediction CSV or manifest.

## Residue numbering

Output positions are one-based positions in the FASTA, not author residue
numbers from the structure. This preserves a canonical node for every sequence
residue, including positions with missing coordinates. The manifest records the
selected structure chain and coverage. Use the FASTA as the authoritative
position map when joining predictions to experimental labels.

## Missing coordinates

Sequence nodes with no mapped Cα coordinate have
`coordinate_available=False`. The model uses its sequence-path fallback at
those nodes. Long missing-coordinate segments or an alignment near the 0.95
threshold should be treated as lower-confidence use cases.
