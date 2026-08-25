# Model card: GeoDSSOP-PDB W3

## Model details

- **Method:** Geometry-aware Deep Sequence–Structure Order-Parameter Predictor
- **Released version:** GeoDSSOP-PDB W3 v0.1.0
- **Legacy development ID:** B4-PDB
- **Task:** residue-level protein backbone N–H order parameter \(S^2\)
- **Inputs:** one amino-acid sequence and one mapped PDBx/mmCIF structure
- **Output:** bounded value in `[0,1]` per input-sequence residue
- **Weights:** three independently seeded models, equal prediction ensemble
- **Trainable parameters/member:** 1,116,753; ESM-2 is frozen and excluded

## Training data

The frozen W3 model used 3,490 weak-training proteins:

- 1,070 with MD-iRED labels derived from atomistic MD trajectories;
- 2,420 with S-OPPE pseudo-labels derived from structural-ensemble predictions.

It used 450 development-validation proteins (153 MD-iRED and 297 S-OPPE) for
early stopping. Source-balanced sampling and loss prevented the larger S-OPPE
domain from receiving proportional weight solely because it contained more
proteins.

Experimental NMR26, historical NMR10, and protected MD-test136 labels were not
used for W3 gradient updates or early stopping.

## Intended use

- Rapid hypothesis generation for residue-resolved fast-timescale backbone
  rigidity when a sequence and compatible single structure are available.
- Comparative profiling across proteins under the same input and preprocessing
  protocol.
- Research benchmarking of sequence–structure fusion for order-parameter
  prediction.

## Out-of-scope use

- Clinical decisions or safety-critical biological conclusions.
- Treating \(S^2\) as a universal scalar measure of every protein motion.
- Claiming calibrated uncertainty from three-seed ensemble spread.
- Claiming validated sequence-only deployment: the current model needs a
  structure.
- Claiming validated AlphaFold performance before a direct paired evaluation.

## Known limitations and failure modes

1. Single structures do not explicitly encode populations or exchange between
   conformational substates.
2. Structure and sequence mismatches below 0.95 identity/coverage are rejected;
   near-threshold accepted mappings still require review.
3. Long missing-coordinate segments rely on the sequence fallback.
4. Training supervision mixes two label-generating processes with different
   error structures and timescale assumptions.
5. NMR experimental labels are limited and heterogeneous; NMR26 is related to
   the pseudo-label teacher lineage.
6. Per-protein PCC is unstable for low-variance targets. Inspect RMSE/MAE and
   traces rather than ranking such proteins by PCC alone.
7. Correct long-range spatial-edge topology did not show a detectable added
   benefit in the preregistered ShuffleEdge gate, although the structural branch
   as a whole did contribute.

## Evaluation summary

- MD-test136: protein-macro PCC 0.824973, macro RMSE 0.111519.
- Historical NMR10: protein-macro PCC 0.842368, macro RMSE 0.132007.
- NMR10 comparison: B0 0.699628, GeoDSSOP 0.842368, SOPPCL 0.845475 using the
  same 709 mapped labels. GeoDSSOP is comparable to, not demonstrably better
  than, SOPPCL on this benchmark.

See the machine-readable results ledger for all aggregation definitions and
claim constraints.
