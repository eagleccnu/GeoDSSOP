"""Frozen GeoDSSOP-PDB inference constants."""

from __future__ import annotations


METHOD_NAME = "Geometry-aware Deep Sequence-Structure Order-Parameter Predictor"
METHOD_ID = "GeoDSSOP-PDB"
LEGACY_MODEL_ID = "B4-PDB"
VERSION = "0.1.0"

ESM_REPOSITORY = "facebook/esm2_t33_650M_UR50D"
ESM_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
ESM_HIDDEN_DIM = 1280
ESM_MAX_RESIDUES = 1022
ESM_TRAINING_CACHE_WEIGHT_SHA256 = (
    "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
)

TRAINING_GIT_COMMIT = "0f008bb9ab944d7a1523b5a89c0dd19411fe7146"
EVALUATION_GIT_COMMIT = "dc0f4a36f43c3dd4a83b6fa5a10e08738028203c"
TRAINING_CONFIG_SHA256 = (
    "845c16898ccab6d5e16cecf2d1b165669ea8a0200174352b496f876b8204237c"
)
ORIGINAL_GRAPH_CONFIG_SHA256 = (
    "588e1b1a6e9fdc0cd30b7b4a6ffceaaabfb01fed13f2f90f4e296967960ac248"
)

# Exact P2-v2 feature contract used for the released checkpoints.  Output and
# cache locations from the research workspace are deliberately absent.
DEFAULT_GRAPH_CONFIG = {
    "node_features": {
        "rsa": {"sphere_points": 96, "probe_radius_angstrom": 1.4},
        "secondary_structure": {
            "helix_phi_degrees": [-160.0, -30.0],
            "helix_psi_degrees": [-80.0, 45.0],
            "strand_phi_degrees": [-180.0, -40.0],
            "strand_psi_union_degrees": [[-180.0, -120.0], [90.0, 180.0]],
        },
    },
    "edge_features": {
        "spatial_ca_cutoff_angstrom": 12.0,
        "spatial_max_neighbors_per_source": 16,
        "sequence_separation_clip": 32,
        "rbf": {"channels": 16, "width_angstrom": 0.8},
    },
}


__all__ = [
    "DEFAULT_GRAPH_CONFIG",
    "ESM_HIDDEN_DIM",
    "ESM_MAX_RESIDUES",
    "ESM_REPOSITORY",
    "ESM_REVISION",
    "METHOD_ID",
    "VERSION",
]
