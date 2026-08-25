# Frozen public results ledger

These CSV files are the path-sanitized, machine-readable truth source for the
GeoDSSOP manuscript's frozen numerical results.

| File | Purpose |
|---|---|
| `datasets.csv` | dataset roles, sizes, and evidence tiers |
| `models.csv` | model identities, aggregation, and original checkpoint hashes |
| `primary_metrics.csv` | principal W3 development, protected, and experimental results |
| `source_ablation_metrics.csv` | MD-only, S-OPPE-only, and joint-training comparisons |
| `structural_ablation.csv` | Full, NoStruct, and ShuffleEdge gate |
| `contextual_baselines.csv` | B0 and SOPPCL context with evidence limitations |
| `claim-register.csv` | statements allowed or forbidden by the current evidence |

Important distinctions:

1. `prediction_ensemble` means average residue predictions first, then score.
2. `mean_of_seed_metrics` means score each seed first, then average metrics.
3. `weak_validation` was used for model selection and is not independent test
   evidence.
4. NMR10 is a historical external benchmark; NMR26 is exploratory and related
   to the OPPE/S-OPPE teacher lineage.
5. The independent MD-test136 result was evaluated once after the model and
   expert outputs were frozen.

The private ledger additionally records workstation artifact paths and is
intentionally not published. Its exported values and file hashes were validated
before this release subset was created.
