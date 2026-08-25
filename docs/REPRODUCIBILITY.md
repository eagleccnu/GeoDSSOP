# Reproducibility map

## Identity chain

| Object | Frozen identity |
|---|---|
| Training code | `0f008bb9ab944d7a1523b5a89c0dd19411fe7146` |
| Evaluation/ablation code | `dc0f4a36f43c3dd4a83b6fa5a10e08738028203c` |
| Training preregistration | SHA-256 `845c16898ccab6d5e16cecf2d1b165669ea8a0200174352b496f876b8204237c` |
| P2-v2 graph config | SHA-256 `588e1b1a6e9fdc0cd30b7b4a6ffceaaabfb01fed13f2f90f4e296967960ac248` |
| Data acceptance | SHA-256 `9cfcae2ef78a649c36291cb954aae9a98639d9b198fb9d2314edfdcb864442e1` |
| Split manifest | SHA-256 `3fed87e5ccc2899d3ab212b2561b351c7aadca9780cec58d55ef4161d86b7dc3` |

The current evaluation implementation differs from the training commit only by
the optional `zero_structure_messages` ablation path and a coordinate-mask
helper. The default full-model forward path was not rewritten. Training-history
reproduction should cite the training commit; released inference and structural
ablation should cite the publication commit/tag.

## Weight conversion

Each original `.pt` checkpoint was SHA-verified, its `model_state` tensors were
copied to CPU, and those tensors were written to a `safetensors` file. Reloaded
tensors were required to be exactly equal to every original tensor.

| Seed | Original checkpoint SHA-256 | Released weight SHA-256 |
|---:|---|---|
| 20260801 | `a5bccb7725292ddf303c905e6c144fde1b38458067f4f828f63eafa3901fafac` | `2f16536f395a763a2f83842c8e605d855c52c3f05d13006e050bfc23fc67171a` |
| 20260802 | `aa78f9e833adebd4ef19d91fa424b33648df8dba0fad05f69bb924d65d6002b3` | `cb088fc6ef7f9089eb0b3a4a03f758cfb923ef6a7e055688b2d7eb4580f73809` |
| 20260803 | `52f8155e99e9164db3f94d15becb2a14b68d5c4a4273cf3f6f50546eecb3401c` | `bf9c3c12c55c8909fd5c1e4be40f97b17c39dcd970f1a1bd6ab4169992792724` |

On the 1UBQ regression input, the public implementation and the original
research implementation were compared in one CPU process. Every member output
and the ensemble output were bitwise identical.

## Demo identity

| File | SHA-256 |
|---|---|
| `1ubq.fasta` | `0629ca1c6496ff15c99ba5a50f71c2de358d236a916175c5e3aa5dbf37f51b04` |
| `1ubq.cif.gz` | `b1a85fb2761c9d2e36734e48706645878036d6c848ebc386611a1957c697d6b3` |
| `1ubq-esm2-features.safetensors` | `bd3acabe1bc16f6720b56c0af6b7bfe21de4f71b668e5611c542c2703c7dc7b1` |
| `expected_predictions.csv` | `6da3f4dda4ef2eb3fb7447f94c6b056ab044656ecc801a80752703d961af4e37` |

## Result ledger

The CSV files under `results/final_results_ledger` are the public,
path-sanitized subset of the manuscript's frozen result ledger. The private
source-artifact audit remains outside this repository because it enumerates
workstation paths; all public numerical fields were validated before export.
