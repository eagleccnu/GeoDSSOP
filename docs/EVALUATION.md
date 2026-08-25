# Evaluation and metric definitions

Run the frozen ensemble on a labeled cached split:

```bash
python scripts/evaluate_geodssop.py \
  --manifest local-data/cached-records.jsonl \
  --split test \
  --weights-dir checkpoints \
  --device cuda:0 \
  --output-dir evaluation/test
```

The output directory contains residue predictions, per-protein metrics, and a
summary manifest with input/output hashes.

## Macro PCC

For each protein, Pearson correlation is calculated across its valid labeled
residues. The reported protein-macro PCC is the unweighted mean across proteins
with a defined PCC. Proteins with fewer than two labels or effectively constant
observed/predicted values have undefined PCC and are excluded from the PCC
mean, but not from RMSE/MAE reporting.

This is different from pooled/micro PCC, which concatenates residues across
proteins. Always report the aggregation, protein count, label count, and
undefined-PCC handling.

## Ensemble aggregation

The paper's principal W3 result averages the three seed predictions residue by
residue and then calculates metrics. Some development ablation tables report
the mean of three separately calculated seed metrics. These values answer
different questions and are explicitly separated in the frozen results ledger.

## Evidence roles

- `weak_validation`: development selection; not independent testing.
- `protected_md_test_metadata_only`: preregistered independent MD-iRED test.
- `experimental_grouped_cv`: NMR26, teacher-lineage-related exploratory set.
- `protected_historical_test_metadata_only`: NMR10 historical benchmark; not a
  newly collected blind test.

Do not tune a frozen model based on protected-test or historical-benchmark
labels and then report the same set as confirmation.
