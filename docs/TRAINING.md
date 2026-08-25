# Portable training interface

The public trainer implements the core frozen W3 protocol using caches supplied
by the user. It does not redistribute the paper's label values.

## JSONL record schema

Each line of `cached-records.jsonl` is one protein:

```json
{
  "record_id": "MD__example_A",
  "source": "md_ired",
  "split": "train",
  "sequence": "ACDEFG...",
  "sequence_sha256": "...",
  "feature_path": "features/example.safetensors",
  "graph_path": "graphs/example.npz",
  "target_path": "targets/example.npz"
}
```

Allowed sources are `md_ired` and `soppe_legacy`; allowed splits are `train`,
`validation`, and `test`. Relative paths are resolved from the JSONL directory.

### ESM feature file

Create it with `geodssop extract-esm`. It contains a float16 tensor named
`features` with shape `[L,1280]` and sequence-hash metadata.

### Graph file

The graph is an `np.savez_compressed` archive produced by the P2-v2 graph
contract. Required model arrays include:

- `node_scalar [L,40]`;
- `node_vector [L,1,3]` and `node_vector_mask [L,1]`;
- `edge_index [2,E]`;
- `edge_type [E,2]`;
- `edge_scalar [E,21]`;
- `edge_vector [E,3,3]` and `edge_vector_mask [E,3]`;
- the validation arrays documented in `configs/graph-p2-v2.yml`.

The graph builder in `geodssop.graph` produces this schema. No array may use
pickled objects.

### Target file

The target is an `np.savez_compressed` archive with exactly:

- `s2`: float array `[L]`;
- `mask`: boolean/0–1 array `[L]` indicating valid supervised positions.

The target file is read only by the training/evaluation scripts, never by the
feature or graph builders.

## W3 training command

Run each preregistered seed into a new directory:

```bash
python scripts/train_geodssop.py \
  --manifest local-data/cached-records.jsonl \
  --seed 20260801 \
  --device cuda:0 \
  --run-dir runs/w3-seed20260801
```

The defaults reproduce the core protocol:

- two proteins per source and four per batch;
- smaller-source cycling without changing source weight;
- Huber `delta=0.10`, normalized first within protein and then within source;
- equal mean of the two source losses;
- AdamW, learning rate `3e-4`, weight decay `0.01`;
- global gradient clipping at 1.0;
- maximum 50 epochs, minimum 10, patience 8, improvement threshold `1e-5`.

The trainer refuses an existing run directory. It writes safe model tensors,
history, and a provenance manifest. Exact recovery/resume of interrupted paper
runs remains tied to the original SHA-identified research pipeline because the
portable trainer deliberately does not serialize executable optimizer pickle.

## Historical exactness boundary

`configs/training-w3-v1.yml` is a path-neutral transcription. The original
training preregistration has SHA-256
`845c16898ccab6d5e16cecf2d1b165669ea8a0200174352b496f876b8204237c`
and training commit
`0f008bb9ab944d7a1523b5a89c0dd19411fe7146`. Those identities, rather than an
unavailable third-party label file, define the exact historical run.
