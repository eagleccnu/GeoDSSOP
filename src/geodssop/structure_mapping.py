"""Label-free mmCIF chain mapping and coordinate inventory helpers."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from Bio import pairwise2
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


STANDARD_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
REQUIRED_BACKBONE = {"N", "CA", "C", "O"}
WATER_COMPONENTS = {"HOH", "DOD", "WAT"}
METAL_ELEMENTS = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "AL", "GA", "IN",
    "SN", "PB", "SB", "BI", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI",
    "CU", "ZN", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
}


def values(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def load_mmcif_dict(path: str | Path) -> MMCIF2Dict:
    """Read a PDBx/mmCIF file, accepting plain text or gzip transparently."""
    source = Path(path)
    if source.suffix.lower() == ".gz":
        with gzip.open(source, "rt", encoding="utf-8", errors="replace") as handle:
            return MMCIF2Dict(handle)
    with source.open("rt", encoding="utf-8", errors="replace") as handle:
        return MMCIF2Dict(handle)


def clean_polymer_sequence(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "X", value.upper())
    return "".join(character for character in value if character.isalpha())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ChainCandidate:
    entity_id: str
    auth_chain: str
    label_chains: tuple[str, ...]
    sequence: str
    polymer_type: str


def chain_candidates(data: Mapping[str, Any]) -> list[ChainCandidate]:
    entity_ids = values(data, "_entity_poly.entity_id")
    sequences = values(data, "_entity_poly.pdbx_seq_one_letter_code_can")
    polymer_types = values(data, "_entity_poly.type")
    strand_fields = values(data, "_entity_poly.pdbx_strand_id")
    label_ids = values(data, "_struct_asym.id")
    label_entities = values(data, "_struct_asym.entity_id")
    labels_by_entity: dict[str, list[str]] = {}
    for label, entity in zip(label_ids, label_entities):
        labels_by_entity.setdefault(entity, []).append(label)
    result: list[ChainCandidate] = []
    for index, entity_id in enumerate(entity_ids):
        sequence = clean_polymer_sequence(sequences[index] if index < len(sequences) else "")
        polymer_type = polymer_types[index] if index < len(polymer_types) else ""
        if not sequence or "polypeptide" not in polymer_type.lower():
            continue
        strand_text = strand_fields[index] if index < len(strand_fields) else ""
        auth_chains = [item.strip() for item in strand_text.split(",") if item.strip() not in {"", ".", "?"}]
        labels = tuple(labels_by_entity.get(entity_id, []))
        if not auth_chains:
            auth_chains = list(labels)
        for auth_chain in auth_chains:
            result.append(ChainCandidate(entity_id, auth_chain, labels, sequence, polymer_type))
    unique: dict[tuple[str, str, str], ChainCandidate] = {}
    for candidate in result:
        unique[(candidate.entity_id, candidate.auth_chain, candidate.sequence)] = candidate
    return sorted(unique.values(), key=lambda item: (item.auth_chain, item.entity_id, item.sequence))


def align_sequences(query: str, target: str) -> dict[str, Any]:
    if query == target:
        return {
            "score": float(2 * len(query)),
            "query_coverage": 1.0,
            "target_coverage": 1.0,
            "identity": 1.0,
            "exact": True,
            "query_to_target": list(range(1, len(query) + 1)),
        }
    alignments = pairwise2.align.globalms(
        query, target, 2.0, -1.0, -4.0, -0.5, one_alignment_only=True
    )
    if not alignments:
        return {
            "score": float("-inf"), "query_coverage": 0.0, "target_coverage": 0.0,
            "identity": 0.0, "exact": False, "query_to_target": [None] * len(query),
        }
    aligned_query, aligned_target, score, _start, _end = alignments[0]
    query_index = 0
    target_index = 0
    mapping: list[int | None] = [None] * len(query)
    aligned_pairs = 0
    matches = 0
    target_covered: set[int] = set()
    for query_char, target_char in zip(aligned_query, aligned_target):
        current_query = None
        current_target = None
        if query_char != "-":
            query_index += 1
            current_query = query_index
        if target_char != "-":
            target_index += 1
            current_target = target_index
        if current_query is not None and current_target is not None:
            mapping[current_query - 1] = current_target
            aligned_pairs += 1
            target_covered.add(current_target)
            if query_char == target_char or query_char == "X" or target_char == "X":
                matches += 1
    return {
        "score": float(score),
        "query_coverage": aligned_pairs / len(query) if query else 0.0,
        "target_coverage": len(target_covered) / len(target) if target else 0.0,
        "identity": matches / aligned_pairs if aligned_pairs else 0.0,
        "exact": False,
        "query_to_target": mapping,
    }


def choose_candidate(
    candidates: Iterable[ChainCandidate],
    query: str,
    declared_chain: str,
    *,
    ambiguity_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    candidates = list(candidates)
    considered = [item for item in candidates if not declared_chain or item.auth_chain == declared_chain]
    if not considered:
        return {"status": "declared_chain_not_found" if declared_chain else "no_protein_chain"}
    scored = [(candidate, align_sequences(query, candidate.sequence)) for candidate in considered]
    scored.sort(
        key=lambda item: (
            item[1]["score"], item[1]["identity"], item[1]["query_coverage"],
            -abs(len(item[0].sequence) - len(query)), item[0].auth_chain,
        ),
        reverse=True,
    )
    best_candidate, best_alignment = scored[0]
    tied = [
        item for item in scored
        if abs(item[1]["score"] - best_alignment["score"]) <= ambiguity_tolerance
        and abs(item[1]["identity"] - best_alignment["identity"]) <= ambiguity_tolerance
        and abs(item[1]["query_coverage"] - best_alignment["query_coverage"]) <= ambiguity_tolerance
    ]
    tied_chains = sorted({item[0].auth_chain for item in tied})
    if not declared_chain and len(tied_chains) > 1:
        return {
            "status": "ambiguous_sequence_chain",
            "tied_auth_chains": tied_chains,
            "candidate_count": len(considered),
            "best_alignment": best_alignment,
        }
    return {
        "status": "selected",
        "candidate": best_candidate,
        "alignment": best_alignment,
        "candidate_count": len(considered),
        "tied_auth_chains": tied_chains,
    }


def altloc_rank(value: str) -> tuple[int, str]:
    normalized = "" if value in {".", "?"} else value
    if normalized == "":
        return (0, "")
    if normalized == "A":
        return (1, "A")
    return (2, normalized)


def first_value(data: Mapping[str, Any], key: str, default: str = "") -> str:
    selected = values(data, key)
    return selected[0] if selected else default


def inventory_record(record: Mapping[str, Any], mmcif_path: Path) -> dict[str, Any]:
    data = load_mmcif_dict(mmcif_path)
    query = str(record["sequence"])
    declared_chain = str(record.get("chain_id") or "").strip()
    candidates = chain_candidates(data)
    selection = choose_candidate(candidates, query, declared_chain)
    protein_auth_chains = {candidate.auth_chain for candidate in candidates}

    entity_ids = values(data, "_entity_poly.entity_id")
    polymer_types = values(data, "_entity_poly.type")
    strand_fields = values(data, "_entity_poly.pdbx_strand_id")
    label_ids = values(data, "_struct_asym.id")
    label_entities = values(data, "_struct_asym.entity_id")
    labels_by_entity: dict[str, list[str]] = {}
    for label, entity in zip(label_ids, label_entities):
        labels_by_entity.setdefault(entity, []).append(label)
    nucleic_acid_chains: set[str] = set()
    polymer_entity_ids = set(entity_ids)
    for index, entity_id in enumerate(entity_ids):
        polymer_type = polymer_types[index] if index < len(polymer_types) else ""
        if "ribonucleotide" not in polymer_type.lower() and "deoxyribonucleotide" not in polymer_type.lower():
            continue
        strand_text = strand_fields[index] if index < len(strand_fields) else ""
        strands = [
            item.strip() for item in strand_text.split(",") if item.strip() not in {"", ".", "?"}
        ]
        nucleic_acid_chains.update(strands or labels_by_entity.get(entity_id, []))

    atom_names = values(data, "_atom_site.label_atom_id")
    auth_chains = values(data, "_atom_site.auth_asym_id")
    label_chains = values(data, "_atom_site.label_asym_id")
    label_seq_ids = values(data, "_atom_site.label_seq_id")
    auth_seq_ids = values(data, "_atom_site.auth_seq_id")
    alt_ids = values(data, "_atom_site.label_alt_id")
    occupancies = values(data, "_atom_site.occupancy")
    model_numbers = values(data, "_atom_site.pdbx_PDB_model_num")
    insertion_codes = values(data, "_atom_site.pdbx_PDB_ins_code")
    comp_ids = values(data, "_atom_site.label_comp_id")
    group_pdb = values(data, "_atom_site.group_PDB")
    atom_entity_ids = values(data, "_atom_site.label_entity_id")
    elements = values(data, "_atom_site.type_symbol")
    global_row_count = min(len(atom_names), len(auth_chains), len(comp_ids))
    global_models = sorted(
        {
            model_numbers[index] if index < len(model_numbers) else "1"
            for index in range(global_row_count)
        },
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    global_selected_model = global_models[0] if global_models else ""
    water_residues: set[tuple[str, str, str, str]] = set()
    ligand_residues: set[tuple[str, str, str, str]] = set()
    metal_atoms = 0
    for index in range(global_row_count):
        model = model_numbers[index] if index < len(model_numbers) else "1"
        if model != global_selected_model:
            continue
        component = comp_ids[index].upper()
        auth_sequence = auth_seq_ids[index] if index < len(auth_seq_ids) else "?"
        insertion = insertion_codes[index] if index < len(insertion_codes) else "?"
        residue_key = (auth_chains[index], auth_sequence, insertion, component)
        element = elements[index].upper() if index < len(elements) else ""
        entity_id = atom_entity_ids[index] if index < len(atom_entity_ids) else ""
        group = group_pdb[index].upper() if index < len(group_pdb) else ""
        if component in WATER_COMPONENTS:
            water_residues.add(residue_key)
        elif element in METAL_ELEMENTS:
            metal_atoms += 1
        elif group == "HETATM" and entity_id not in polymer_entity_ids:
            ligand_residues.add(residue_key)

    digest = hashlib.sha256()
    with mmcif_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    base: dict[str, Any] = {
        "record_id": str(record["record_id"]),
        "source": str(record["source"]),
        "canonical_dataset_role": str(record["canonical_dataset_role"]),
        "joint_homology_group": str(record["joint_homology_group"]),
        "pdb_id": str(record["pdb_id"]).strip().lower(),
        "declared_chain_id": declared_chain,
        "sequence_sha256": str(record["sequence_sha256"]),
        "sequence_length": len(query),
        "unknown_residue_count": int(record["unknown_residue_count"]),
        "mmcif_path": str(mmcif_path),
        "mmcif_sha256": digest.hexdigest(),
        "experimental_method": " | ".join(values(data, "_exptl.method")),
        "selection_status": selection["status"],
        "selected_auth_chain": "",
        "selected_entity_id": "",
        "selected_label_chains": "",
        "chain_inferred": not bool(declared_chain),
        "protein_chain_candidate_count": int(selection.get("candidate_count", 0)),
        "tied_auth_chains": ",".join(selection.get("tied_auth_chains", [])),
        "polymer_sequence_length": 0,
        "polymer_sequence_sha256": "",
        "alignment_score": None,
        "query_coverage": 0.0,
        "target_coverage": 0.0,
        "sequence_identity": 0.0,
        "sequence_exact": False,
        "alignment_accepted": False,
        "coordinate_model_count": 0,
        "selected_model": "",
        "pdb_protein_chain_count": len(protein_auth_chains),
        "other_protein_chain_count": 0,
        "nucleic_acid_chain_count": len(nucleic_acid_chains),
        "ligand_residue_count": len(ligand_residues),
        "water_residue_count": len(water_residues),
        "metal_atom_count": metal_atoms,
        "coordinate_residue_count": 0,
        "complete_backbone_residue_count": 0,
        "coordinate_coverage": 0.0,
        "complete_backbone_coverage": 0.0,
        "maximum_missing_coordinate_run": len(query),
        "altloc_residue_count": 0,
        "insertion_code_residue_count": 0,
        "negative_auth_seq_id_residue_count": 0,
        "duplicate_auth_numbering_count": 0,
        "modified_residue_count": 0,
        "inventory_status": selection["status"],
        "error": "",
    }
    if selection["status"] != "selected":
        return base
    candidate: ChainCandidate = selection["candidate"]
    alignment = selection["alignment"]
    base.update({
        "selected_auth_chain": candidate.auth_chain,
        "selected_entity_id": candidate.entity_id,
        "selected_label_chains": ",".join(candidate.label_chains),
        "other_protein_chain_count": len(protein_auth_chains - {candidate.auth_chain}),
        "polymer_sequence_length": len(candidate.sequence),
        "polymer_sequence_sha256": sha256_text(candidate.sequence),
        "alignment_score": alignment["score"],
        "query_coverage": alignment["query_coverage"],
        "target_coverage": alignment["target_coverage"],
        "sequence_identity": alignment["identity"],
        "sequence_exact": alignment["exact"],
    })
    accepted = alignment["query_coverage"] >= 0.95 and alignment["identity"] >= 0.95
    base["alignment_accepted"] = bool(accepted)

    lengths = [len(atom_names), len(auth_chains), len(label_chains), len(label_seq_ids)]
    row_count = min(lengths) if lengths else 0
    matching_rows = []
    label_set = set(candidate.label_chains)
    for index in range(row_count):
        if auth_chains[index] != candidate.auth_chain:
            continue
        if label_set and label_chains[index] not in label_set:
            continue
        model = model_numbers[index] if index < len(model_numbers) else "1"
        matching_rows.append((index, model))
    models = sorted(
        {item[1] for item in matching_rows},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    selected_model = models[0] if models else ""
    base["coordinate_model_count"] = len(models)
    base["selected_model"] = selected_model
    selected_atoms: dict[tuple[int, str], tuple[float, tuple[int, str], int]] = {}
    alt_positions: set[int] = set()
    insertion_positions: set[int] = set()
    negative_auth_number_positions: set[int] = set()
    auth_number_to_entity_positions: dict[tuple[str, str], set[int]] = {}
    modified_positions: set[int] = set()
    for index, model in matching_rows:
        if model != selected_model:
            continue
        seq_text = label_seq_ids[index]
        if not seq_text.isdigit():
            continue
        entity_position = int(seq_text)
        atom = atom_names[index].strip()
        alt = alt_ids[index] if index < len(alt_ids) else "."
        occupancy_text = occupancies[index] if index < len(occupancies) else "1"
        try:
            occupancy = float(occupancy_text)
        except ValueError:
            occupancy = 0.0
        priority = altloc_rank(alt)
        key = (entity_position, atom)
        current = selected_atoms.get(key)
        if current is None or occupancy > current[0] or (occupancy == current[0] and priority < current[1]):
            selected_atoms[key] = (occupancy, priority, index)
        if alt not in {".", "?", ""}:
            alt_positions.add(entity_position)
        insertion = insertion_codes[index] if index < len(insertion_codes) else "?"
        if insertion not in {".", "?", ""}:
            insertion_positions.add(entity_position)
        auth_sequence = auth_seq_ids[index] if index < len(auth_seq_ids) else "?"
        try:
            if int(auth_sequence) < 0:
                negative_auth_number_positions.add(entity_position)
        except ValueError:
            pass
        auth_number_to_entity_positions.setdefault((auth_sequence, insertion), set()).add(entity_position)
        comp = comp_ids[index].upper() if index < len(comp_ids) else ""
        if comp and comp not in STANDARD_AA3:
            modified_positions.add(entity_position)

    atoms_by_position: dict[int, set[str]] = {}
    for entity_position, atom in selected_atoms:
        atoms_by_position.setdefault(entity_position, set()).add(atom)
    coordinate_flags = []
    backbone_flags = []
    mapped_entity_positions = []
    for entity_position in alignment["query_to_target"]:
        mapped_entity_positions.append(entity_position)
        atoms = atoms_by_position.get(entity_position, set()) if entity_position is not None else set()
        coordinate_flags.append(bool(atoms))
        backbone_flags.append(REQUIRED_BACKBONE.issubset(atoms))
    missing_run = 0
    maximum_missing_run = 0
    for flag in coordinate_flags:
        if flag:
            missing_run = 0
        else:
            missing_run += 1
            maximum_missing_run = max(maximum_missing_run, missing_run)
    mapped_set = {value for value in mapped_entity_positions if value is not None}
    coordinate_count = sum(coordinate_flags)
    backbone_count = sum(backbone_flags)
    base.update({
        "coordinate_residue_count": coordinate_count,
        "complete_backbone_residue_count": backbone_count,
        "coordinate_coverage": coordinate_count / len(query) if query else 0.0,
        "complete_backbone_coverage": backbone_count / len(query) if query else 0.0,
        "maximum_missing_coordinate_run": maximum_missing_run,
        "altloc_residue_count": len(alt_positions & mapped_set),
        "insertion_code_residue_count": len(insertion_positions & mapped_set),
        "negative_auth_seq_id_residue_count": len(negative_auth_number_positions & mapped_set),
        "duplicate_auth_numbering_count": sum(
            len(entity_positions & mapped_set) > 1
            for entity_positions in auth_number_to_entity_positions.values()
        ),
        "modified_residue_count": len(modified_positions & mapped_set),
        "inventory_status": "accepted" if accepted and coordinate_count else (
            "alignment_below_threshold" if not accepted else "no_selected_chain_coordinates"
        ),
    })
    return base
