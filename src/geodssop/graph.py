"""Deterministic, label-free residue graph construction from a selected PDB chain."""

from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from geodssop.structure_mapping import (
    REQUIRED_BACKBONE,
    altloc_rank,
    chain_candidates,
    choose_candidate,
    values,
)


AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ORDER)}
BACKBONE_ATOMS = ("N", "CA", "C", "O")
MAXIMUM_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
    "X": 220.0,
}
VDW_RADII = {
    "H": 1.20, "D": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.80, "ZN": 1.39, "MG": 1.73, "CA": 1.94,
}


def load_mmcif_dict(path: str | Path) -> MMCIF2Dict:
    """Read a PDBx/mmCIF file, accepting plain text or gzip transparently."""
    source = Path(path)
    if source.suffix.lower() == ".gz":
        with gzip.open(source, "rt", encoding="utf-8", errors="replace") as handle:
            return MMCIF2Dict(handle)
    with source.open("rt", encoding="utf-8", errors="replace") as handle:
        return MMCIF2Dict(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_vector(vector: np.ndarray, epsilon: float = 1.0e-8) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if norm <= epsilon:
        return np.zeros(3, dtype=np.float64), False
    return np.asarray(vector, dtype=np.float64) / norm, True


def dihedral_radians(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed dihedral in radians using only rigid-invariant inner products."""
    first = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    second = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    third = np.asarray(d, dtype=np.float64) - np.asarray(c, dtype=np.float64)
    second_unit, valid = unit_vector(second)
    if not valid:
        return 0.0
    first_rejected = first - np.dot(first, second_unit) * second_unit
    third_rejected = third - np.dot(third, second_unit) * second_unit
    first_unit, first_valid = unit_vector(first_rejected)
    third_unit, third_valid = unit_vector(third_rejected)
    if not first_valid or not third_valid:
        return 0.0
    x_value = float(np.dot(first_unit, third_unit))
    y_value = float(np.dot(np.cross(second_unit, first_unit), third_unit))
    return math.atan2(y_value, x_value)


def fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z_values = 1.0 - 2.0 * (indices + 0.5) / count
    radii = np.sqrt(np.maximum(0.0, 1.0 - z_values * z_values))
    angles = golden_angle * indices
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles), z_values))


def equivariant_global_frame(coordinates: np.ndarray) -> np.ndarray:
    """Create a stable equivariant frame from canonical atom ordering."""
    if len(coordinates) < 3:
        return np.eye(3, dtype=np.float64)
    origin = coordinates[0]
    first = None
    first_index = None
    for index in range(1, len(coordinates)):
        candidate, valid = unit_vector(coordinates[index] - origin)
        if valid:
            first = candidate
            first_index = index
            break
    if first is None:
        return np.eye(3, dtype=np.float64)
    second = None
    for index in range(1, len(coordinates)):
        if index == first_index:
            continue
        vector = coordinates[index] - origin
        rejected = vector - np.dot(vector, first) * first
        candidate, valid = unit_vector(rejected)
        if valid:
            second = candidate
            break
    if second is None:
        # A fully collinear atom set is cylindrically symmetric about the first axis.
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, first))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        second, _ = unit_vector(reference - np.dot(reference, first) * first)
    third, _ = unit_vector(np.cross(first, second))
    second, _ = unit_vector(np.cross(third, first))
    return np.column_stack((first, second, third))


def solvent_accessible_area_by_position(
    atoms: list[dict[str, Any]], sphere_points: int, probe_radius: float
) -> dict[int, float]:
    heavy_atoms = [atom for atom in atoms if atom["element"].upper() not in {"H", "D"}]
    if not heavy_atoms:
        return {}
    coordinates = np.asarray([atom["coordinate"] for atom in heavy_atoms], dtype=np.float64)
    expanded_radii = np.asarray(
        [VDW_RADII.get(atom["element"].upper(), 1.80) + probe_radius for atom in heavy_atoms],
        dtype=np.float64,
    )
    tree = cKDTree(coordinates)
    sampling_frame = equivariant_global_frame(coordinates)
    unit_points = fibonacci_sphere(sphere_points)
    maximum_radius = float(expanded_radii.max())
    result: dict[int, float] = {}
    for index, (coordinate, radius) in enumerate(zip(coordinates, expanded_radii)):
        points = coordinate + radius * (unit_points @ sampling_frame.T)
        candidates = tree.query_ball_point(coordinate, radius + maximum_radius)
        accessible = np.ones(sphere_points, dtype=bool)
        for neighbor in candidates:
            if neighbor == index:
                continue
            squared = np.sum((points - coordinates[neighbor]) ** 2, axis=1)
            accessible &= squared >= expanded_radii[neighbor] ** 2 - 1.0e-10
            if not accessible.any():
                break
        surface = float(accessible.mean()) * 4.0 * math.pi * float(radius * radius)
        position = int(heavy_atoms[index]["entity_position"])
        result[position] = result.get(position, 0.0) + surface
    return result


def selected_chain_atoms(
    data: Mapping[str, Any], auth_chain: str, label_chains: set[str], selected_model: str
) -> list[dict[str, Any]]:
    atom_names = values(data, "_atom_site.label_atom_id")
    auth_chains = values(data, "_atom_site.auth_asym_id")
    labels = values(data, "_atom_site.label_asym_id")
    sequence_ids = values(data, "_atom_site.label_seq_id")
    model_numbers = values(data, "_atom_site.pdbx_PDB_model_num")
    alternative_ids = values(data, "_atom_site.label_alt_id")
    occupancies = values(data, "_atom_site.occupancy")
    x_values = values(data, "_atom_site.Cartn_x")
    y_values = values(data, "_atom_site.Cartn_y")
    z_values = values(data, "_atom_site.Cartn_z")
    elements = values(data, "_atom_site.type_symbol")
    component_ids = values(data, "_atom_site.label_comp_id")
    row_count = min(
        len(atom_names), len(auth_chains), len(labels), len(sequence_ids),
        len(x_values), len(y_values), len(z_values),
    )
    chosen: dict[tuple[int, str], dict[str, Any]] = {}
    for index in range(row_count):
        if auth_chains[index] != auth_chain:
            continue
        if label_chains and labels[index] not in label_chains:
            continue
        model = model_numbers[index] if index < len(model_numbers) else "1"
        if model != selected_model or not sequence_ids[index].isdigit():
            continue
        try:
            coordinate = np.array(
                [float(x_values[index]), float(y_values[index]), float(z_values[index])],
                dtype=np.float64,
            )
        except ValueError:
            continue
        atom_name = atom_names[index].strip()
        position = int(sequence_ids[index])
        alternative = alternative_ids[index] if index < len(alternative_ids) else "."
        try:
            occupancy = float(occupancies[index] if index < len(occupancies) else "1")
        except ValueError:
            occupancy = 0.0
        priority = altloc_rank(alternative)
        key = (position, atom_name)
        current = chosen.get(key)
        if current is not None and not (
            occupancy > current["occupancy"]
            or (occupancy == current["occupancy"] and priority < current["altloc_priority"])
        ):
            continue
        element = elements[index].strip().upper() if index < len(elements) else atom_name[:1].upper()
        chosen[key] = {
            "entity_position": position,
            "atom_name": atom_name,
            "coordinate": coordinate,
            "element": element,
            "component_id": component_ids[index].upper() if index < len(component_ids) else "",
            "occupancy": occupancy,
            "altloc_priority": priority,
        }
    return [chosen[key] for key in sorted(chosen)]


def apply_rigid_transform(
    atoms: list[dict[str, Any]], rotation: np.ndarray | None, translation: np.ndarray | None
) -> list[dict[str, Any]]:
    if rotation is None and translation is None:
        return atoms
    selected_rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    selected_translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    transformed = []
    for atom in atoms:
        copy = dict(atom)
        copy["coordinate"] = selected_rotation @ np.asarray(atom["coordinate"]) + selected_translation
        transformed.append(copy)
    return transformed


def build_edges(
    ca_coordinates: np.ndarray,
    ca_mask: np.ndarray,
    backbone_complete: np.ndarray,
    backbone_direction: np.ndarray,
    backbone_direction_mask: np.ndarray,
    cutoff: float,
    maximum_neighbors: int,
    separation_clip: int,
    rbf_channels: int,
    rbf_width: float,
) -> dict[str, np.ndarray]:
    length = len(ca_coordinates)
    edge_flags: dict[tuple[int, int], list[int]] = {}
    for source in range(length):
        for offset in (-2, -1, 1, 2):
            destination = source + offset
            if 0 <= destination < length:
                edge_flags.setdefault((source, destination), [0, 0])[0] = 1
    spatial_neighbor_count = np.zeros(length, dtype=np.int32)
    valid_indices = np.flatnonzero(ca_mask.astype(bool))
    if len(valid_indices):
        tree = cKDTree(ca_coordinates[valid_indices])
        for local_source, source in enumerate(valid_indices):
            neighbors = tree.query_ball_point(ca_coordinates[source], cutoff + 1.0e-8)
            ranked = []
            for local_destination in neighbors:
                destination = int(valid_indices[local_destination])
                if destination == source:
                    continue
                distance = float(np.linalg.norm(ca_coordinates[destination] - ca_coordinates[source]))
                ranked.append((distance, destination))
            for _distance, destination in sorted(ranked)[:maximum_neighbors]:
                edge_flags.setdefault((int(source), destination), [0, 0])[1] = 1
                spatial_neighbor_count[source] += 1
    ordered_edges = sorted(edge_flags)
    edge_index = np.asarray(ordered_edges, dtype=np.int32).T
    edge_count = len(ordered_edges)
    edge_type = np.asarray([edge_flags[edge] for edge in ordered_edges], dtype=np.uint8)
    scalar = np.zeros((edge_count, rbf_channels + 5), dtype=np.float32)
    vectors = np.zeros((edge_count, 3, 3), dtype=np.float32)
    vector_mask = np.zeros((edge_count, 3), dtype=np.uint8)
    distances = np.zeros(edge_count, dtype=np.float32)
    separations = np.zeros(edge_count, dtype=np.int16)
    centers = np.linspace(0.0, cutoff, rbf_channels, dtype=np.float64)
    for edge_index_value, (source, destination) in enumerate(ordered_edges):
        separation = destination - source
        separations[edge_index_value] = separation
        scalar[edge_index_value, rbf_channels] = np.clip(
            separation, -separation_clip, separation_clip
        ) / separation_clip
        scalar[edge_index_value, rbf_channels + 1:rbf_channels + 3] = edge_type[edge_index_value]
        ca_pair = bool(ca_mask[source] and ca_mask[destination])
        complete_pair = bool(backbone_complete[source] and backbone_complete[destination])
        scalar[edge_index_value, rbf_channels + 3] = float(ca_pair)
        scalar[edge_index_value, rbf_channels + 4] = float(complete_pair)
        if ca_pair:
            displacement = ca_coordinates[destination] - ca_coordinates[source]
            direction, valid = unit_vector(displacement)
            distance = float(np.linalg.norm(displacement))
            distances[edge_index_value] = distance
            scalar[edge_index_value, :rbf_channels] = np.exp(
                -((distance - centers) / rbf_width) ** 2
            ).astype(np.float32)
            if valid:
                vectors[edge_index_value, 0] = direction
                vector_mask[edge_index_value, 0] = 1
        if backbone_direction_mask[source]:
            vectors[edge_index_value, 1] = backbone_direction[source]
            vector_mask[edge_index_value, 1] = 1
        if backbone_direction_mask[destination]:
            vectors[edge_index_value, 2] = backbone_direction[destination]
            vector_mask[edge_index_value, 2] = 1
    return {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_scalar": scalar,
        "edge_vector": vectors,
        "edge_vector_mask": vector_mask,
        "edge_distance": distances,
        "edge_sequence_separation": separations,
        "spatial_neighbor_count": spatial_neighbor_count,
    }


def build_graph_arrays(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    mmcif_path = Path(str(task["mmcif_path"]))
    data = load_mmcif_dict(mmcif_path)
    sequence = str(record["sequence"])
    if hashlib.sha256(sequence.encode("ascii")).hexdigest() != str(task["sequence_sha256"]):
        raise ValueError("sequence SHA does not match the P2 task")
    selection = choose_candidate(
        chain_candidates(data), sequence, str(task["auth_chain_id"])
    )
    if selection["status"] != "selected":
        raise ValueError(f"P1-selected chain is no longer selectable: {selection['status']}")
    candidate = selection["candidate"]
    alignment = selection["alignment"]
    if str(candidate.entity_id) != str(task["entity_id"]):
        raise ValueError("selected entity changed after P1")
    if alignment["identity"] < 0.95 or alignment["query_coverage"] < 0.95:
        raise ValueError("alignment is below the preregistered P1 threshold")
    atoms = selected_chain_atoms(
        data,
        str(task["auth_chain_id"]),
        set(str(task["label_chain_ids"]).split(",")),
        str(task["selected_model"]),
    )
    atoms = apply_rigid_transform(atoms, rotation, translation)
    atoms_by_position: dict[int, dict[str, dict[str, Any]]] = {}
    for atom in atoms:
        atoms_by_position.setdefault(int(atom["entity_position"]), {})[str(atom["atom_name"])] = atom

    length = len(sequence)
    backbone_coordinates = np.zeros((length, 4, 3), dtype=np.float64)
    backbone_atom_mask = np.zeros((length, 4), dtype=np.uint8)
    mapped_positions = alignment["query_to_target"]
    for query_index, entity_position in enumerate(mapped_positions):
        if entity_position is None:
            continue
        residue_atoms = atoms_by_position.get(int(entity_position), {})
        for atom_index, atom_name in enumerate(BACKBONE_ATOMS):
            atom = residue_atoms.get(atom_name)
            if atom is not None:
                backbone_coordinates[query_index, atom_index] = atom["coordinate"]
                backbone_atom_mask[query_index, atom_index] = 1
    ca_coordinates = backbone_coordinates[:, 1].copy()
    ca_mask = backbone_atom_mask[:, 1].copy()
    backbone_complete = np.all(backbone_atom_mask == 1, axis=1).astype(np.uint8)
    backbone_direction = np.zeros((length, 3), dtype=np.float64)
    backbone_direction_mask = np.zeros(length, dtype=np.uint8)
    for index in range(length):
        if backbone_atom_mask[index, 0] and backbone_atom_mask[index, 2]:
            direction, valid = unit_vector(
                backbone_coordinates[index, 2] - backbone_coordinates[index, 0]
            )
            if valid:
                backbone_direction[index] = direction
                backbone_direction_mask[index] = 1

    torsions = np.zeros((length, 6), dtype=np.float32)
    torsion_masks = np.zeros((length, 3), dtype=np.uint8)
    torsion_angles = np.zeros((length, 3), dtype=np.float64)
    for index, entity_position in enumerate(mapped_positions):
        if entity_position is None:
            continue
        current = int(entity_position)
        if index > 0 and mapped_positions[index - 1] == current - 1:
            if backbone_atom_mask[index - 1, 2] and np.all(backbone_atom_mask[index, :3]):
                torsion_angles[index, 0] = dihedral_radians(
                    backbone_coordinates[index - 1, 2], backbone_coordinates[index, 0],
                    backbone_coordinates[index, 1], backbone_coordinates[index, 2],
                )
                torsion_masks[index, 0] = 1
            if (
                backbone_atom_mask[index - 1, 1]
                and backbone_atom_mask[index - 1, 2]
                and backbone_atom_mask[index, 0]
                and backbone_atom_mask[index, 1]
            ):
                torsion_angles[index, 2] = dihedral_radians(
                    backbone_coordinates[index - 1, 1], backbone_coordinates[index - 1, 2],
                    backbone_coordinates[index, 0], backbone_coordinates[index, 1],
                )
                torsion_masks[index, 2] = 1
        if index + 1 < length and mapped_positions[index + 1] == current + 1:
            if np.all(backbone_atom_mask[index, :3]) and backbone_atom_mask[index + 1, 0]:
                torsion_angles[index, 1] = dihedral_radians(
                    backbone_coordinates[index, 0], backbone_coordinates[index, 1],
                    backbone_coordinates[index, 2], backbone_coordinates[index + 1, 0],
                )
                torsion_masks[index, 1] = 1
    for torsion_index in range(3):
        torsions[:, 2 * torsion_index] = np.sin(torsion_angles[:, torsion_index]) * torsion_masks[:, torsion_index]
        torsions[:, 2 * torsion_index + 1] = np.cos(torsion_angles[:, torsion_index]) * torsion_masks[:, torsion_index]

    rsa_surface = solvent_accessible_area_by_position(
        atoms,
        int(config["node_features"]["rsa"]["sphere_points"]),
        float(config["node_features"]["rsa"]["probe_radius_angstrom"]),
    )
    rsa = np.zeros(length, dtype=np.float32)
    rsa_mask = np.zeros(length, dtype=np.uint8)
    for index, entity_position in enumerate(mapped_positions):
        if entity_position is None or int(entity_position) not in rsa_surface:
            continue
        amino_acid = sequence[index] if sequence[index] in MAXIMUM_ASA else "X"
        rsa[index] = np.clip(rsa_surface[int(entity_position)] / MAXIMUM_ASA[amino_acid], 0.0, 1.0)
        rsa_mask[index] = 1

    secondary_structure = np.zeros((length, 3), dtype=np.float32)
    secondary_structure_mask = np.zeros(length, dtype=np.uint8)
    secondary_config = config["node_features"]["secondary_structure"]
    for index in range(length):
        if not (torsion_masks[index, 0] and torsion_masks[index, 1]):
            continue
        phi = math.degrees(float(torsion_angles[index, 0]))
        psi = math.degrees(float(torsion_angles[index, 1]))
        helix = (
            secondary_config["helix_phi_degrees"][0] <= phi <= secondary_config["helix_phi_degrees"][1]
            and secondary_config["helix_psi_degrees"][0] <= psi <= secondary_config["helix_psi_degrees"][1]
        )
        strand = (
            secondary_config["strand_phi_degrees"][0] <= phi <= secondary_config["strand_phi_degrees"][1]
            and any(lower <= psi <= upper for lower, upper in secondary_config["strand_psi_union_degrees"])
        )
        secondary_structure[index, 1 if helix else 2 if strand else 0] = 1.0
        secondary_structure_mask[index] = 1

    edges = build_edges(
        ca_coordinates,
        ca_mask,
        backbone_complete,
        backbone_direction,
        backbone_direction_mask,
        float(config["edge_features"]["spatial_ca_cutoff_angstrom"]),
        int(config["edge_features"]["spatial_max_neighbors_per_source"]),
        int(config["edge_features"]["sequence_separation_clip"]),
        int(config["edge_features"]["rbf"]["channels"]),
        float(config["edge_features"]["rbf"]["width_angstrom"]),
    )
    amino_acid_index = np.asarray(
        [AA_TO_INDEX.get(amino_acid, AA_TO_INDEX["X"]) for amino_acid in sequence], dtype=np.uint8
    )
    amino_acid_one_hot = np.eye(len(AA_ORDER), dtype=np.float32)[amino_acid_index]
    local_contact_density = np.minimum(
        edges.pop("spatial_neighbor_count"),
        int(config["edge_features"]["spatial_max_neighbors_per_source"]),
    ).astype(np.float32) / int(config["edge_features"]["spatial_max_neighbors_per_source"])
    node_scalar = np.concatenate(
        (
            amino_acid_one_hot,
            torsions,
            torsion_masks.astype(np.float32),
            rsa[:, None],
            rsa_mask[:, None].astype(np.float32),
            secondary_structure,
            secondary_structure_mask[:, None].astype(np.float32),
            local_contact_density[:, None],
            ca_mask[:, None].astype(np.float32),
            backbone_complete[:, None].astype(np.float32),
            backbone_direction_mask[:, None].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "graph_schema_version": np.asarray([1], dtype=np.int16),
        "graph_key": np.asarray(str(task["graph_key"])),
        "sequence_sha256": np.asarray(str(task["sequence_sha256"])),
        "pdb_id": np.asarray(str(task["pdb_id"])),
        "auth_chain_id": np.asarray(str(task["auth_chain_id"])),
        "selected_model": np.asarray(str(task["selected_model"])),
        "mmcif_sha256": np.asarray(str(task["mmcif_sha256"])),
        "p2_graph_config_sha256": np.asarray(str(task["p2_graph_config_sha256"])),
        "graph_builder_code_sha256": np.asarray(str(task["graph_builder_code_sha256"])),
        "sequence_length": np.asarray([length], dtype=np.int32),
        "amino_acid_index": amino_acid_index,
        "node_scalar": node_scalar,
        "node_vector": backbone_direction[:, None, :].astype(np.float32),
        "node_vector_mask": backbone_direction_mask[:, None],
        "backbone_coordinates": backbone_coordinates.astype(np.float32),
        "backbone_atom_mask": backbone_atom_mask,
        "ca_coordinates": ca_coordinates.astype(np.float32),
        "ca_mask": ca_mask,
        **edges,
    }
    validate_graph_arrays(arrays, expected_length=length)
    return arrays


def validate_graph_arrays(arrays: Mapping[str, np.ndarray], expected_length: int | None = None) -> None:
    length = int(np.asarray(arrays["sequence_length"]).reshape(-1)[0])
    if expected_length is not None and length != expected_length:
        raise ValueError(f"sequence length {length} != expected {expected_length}")
    required_shapes = {
        "amino_acid_index": (length,),
        "node_scalar": (length, 40),
        "node_vector": (length, 1, 3),
        "node_vector_mask": (length, 1),
        "backbone_coordinates": (length, 4, 3),
        "backbone_atom_mask": (length, 4),
        "ca_coordinates": (length, 3),
        "ca_mask": (length,),
    }
    for name, shape in required_shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")
    edge_count = int(arrays["edge_index"].shape[1])
    edge_shapes = {
        "edge_index": (2, edge_count),
        "edge_type": (edge_count, 2),
        "edge_scalar": (edge_count, 21),
        "edge_vector": (edge_count, 3, 3),
        "edge_vector_mask": (edge_count, 3),
        "edge_distance": (edge_count,),
        "edge_sequence_separation": (edge_count,),
    }
    for name, shape in edge_shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")
    if edge_count:
        if int(arrays["edge_index"].min()) < 0 or int(arrays["edge_index"].max()) >= length:
            raise ValueError("edge index is out of bounds")
        edge_pairs = [tuple(pair) for pair in arrays["edge_index"].T.tolist()]
        if len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("duplicate directed edges")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
