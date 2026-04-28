"""Container-level tests for ``constellation.massspec.library.Library``."""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.massspec.library import Library, assign_ids
from constellation.massspec.library.schemas import (
    LIBRARY_FRAGMENT_TABLE,
    PEPTIDE_TABLE,
    PRECURSOR_TABLE,
    PROTEIN_PEPTIDE_EDGE,
    PROTEIN_TABLE,
)
from constellation.massspec.peptide.ions import IonType


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _minimal_library() -> Library:
    return assign_ids(
        proteins=[
            {"accession": "P00001", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"},
            {"accession": "P00002", "sequence": "MAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT"},
        ],
        peptides=[
            {"modified_sequence": "PEPTIDE"},
            {"modified_sequence": "PEP[UNIMOD:4]TIDE"},
        ],
        precursors=[
            {"modified_sequence": "PEPTIDE", "charge": 2, "precursor_mz": 400.20},
            {
                "modified_sequence": "PEP[UNIMOD:4]TIDE",
                "charge": 2,
                "precursor_mz": 428.70,
                "rt_predicted": 1234.5,
            },
        ],
        fragments=[
            {
                "modified_sequence": "PEPTIDE",
                "precursor_charge": 2,
                "ion_type": int(IonType.B),
                "position": 2,
                "charge": 1,
                "mz_theoretical": 227.10,
                "intensity_predicted": 0.42,
            },
            {
                "modified_sequence": "PEPTIDE",
                "precursor_charge": 2,
                "ion_type": int(IonType.Y),
                "position": 5,
                "charge": 1,
                "mz_theoretical": 600.30,
                "intensity_predicted": 1.0,
            },
        ],
        protein_peptide=[
            ("P00001", "PEPTIDE"),
            ("P00002", "PEPTIDE"),  # shared between proteoforms
            ("P00001", "PEP[UNIMOD:4]TIDE"),
        ],
    )


# ──────────────────────────────────────────────────────────────────────
# Construction + validation
# ──────────────────────────────────────────────────────────────────────


def test_minimal_library_constructs() -> None:
    lib = _minimal_library()
    assert lib.n_proteins == 2
    assert lib.n_peptides == 2
    assert lib.n_precursors == 2
    assert lib.n_fragments == 2


def test_validate_rejects_orphan_peptide_fk() -> None:
    lib = _minimal_library()
    bad_precursors = pa.table(
        {
            "precursor_id": [0],
            "peptide_id": [9999],  # nonexistent
            "charge": pa.array([2], type=pa.int32()),
            "precursor_mz": [400.0],
            "rt_predicted": [-1.0],
            "ccs_predicted": [-1.0],
        },
        schema=PRECURSOR_TABLE,
    )
    with pytest.raises(ValueError, match="references ids not present"):
        Library(
            proteins=lib.proteins,
            peptides=lib.peptides,
            precursors=bad_precursors,
            fragments=LIBRARY_FRAGMENT_TABLE.empty_table(),
            protein_peptide=lib.protein_peptide,
        )


def test_validate_rejects_orphan_fragment_fk() -> None:
    lib = _minimal_library()
    bad_fragments = pa.table(
        {
            "precursor_id": [9999],
            "ion_type": pa.array([int(IonType.B)], type=pa.int8()),
            "position": pa.array([1], type=pa.int32()),
            "charge": pa.array([1], type=pa.int32()),
            "loss_id": pa.array([None], type=pa.string()),
            "mz_theoretical": [100.0],
            "intensity_predicted": pa.array([0.5], type=pa.float32()),
            "annotation": pa.array([None], type=pa.string()),
        },
        schema=LIBRARY_FRAGMENT_TABLE,
    )
    with pytest.raises(ValueError, match="references ids not present"):
        Library(
            proteins=lib.proteins,
            peptides=lib.peptides,
            precursors=lib.precursors,
            fragments=bad_fragments,
            protein_peptide=lib.protein_peptide,
        )


def test_validate_rejects_duplicate_protein_id() -> None:
    proteins = pa.table(
        {
            "protein_id": [0, 0],
            "accession": ["A", "B"],
            "sequence": ["M", "M"],
            "description": pa.array([None, None], type=pa.string()),
        },
        schema=PROTEIN_TABLE,
    )
    with pytest.raises(ValueError, match="duplicate"):
        Library(
            proteins=proteins,
            peptides=PEPTIDE_TABLE.empty_table(),
            precursors=PRECURSOR_TABLE.empty_table(),
            fragments=LIBRARY_FRAGMENT_TABLE.empty_table(),
            protein_peptide=PROTEIN_PEPTIDE_EDGE.empty_table(),
        )


# ──────────────────────────────────────────────────────────────────────
# Proteoform sharing
# ──────────────────────────────────────────────────────────────────────


def test_proteoform_sharing_peptides_topology() -> None:
    """One peptide → two proteins. Subsetting either protein keeps the
    peptide; both protein-peptide edges survive validate."""
    lib = _minimal_library()
    # PEPTIDE is shared between P00001 and P00002 — verify both edges.
    pp = lib.protein_peptide.to_pylist()
    pep_id = lib.peptides.column("peptide_id").to_pylist()[
        lib.peptides.column("modified_sequence").to_pylist().index("PEPTIDE")
    ]
    edges_to_pep = [e for e in pp if e["peptide_id"] == pep_id]
    assert len(edges_to_pep) == 2
    assert {e["protein_id"] for e in edges_to_pep} == {0, 1}


def test_subset_proteins_cascades() -> None:
    lib = _minimal_library()
    sub = lib.subset_proteins([0])  # P00001 only
    assert sub.n_proteins == 1
    # Both PEPTIDE and modified PEPTIDE are reachable from P00001.
    assert sub.n_peptides == 2
    assert sub.n_precursors == 2
    # Fragments only attach to PEPTIDE precursor in fixture.
    assert sub.n_fragments == 2


def test_subset_proteins_drops_unreferenced_peptides() -> None:
    lib = _minimal_library()
    sub = lib.subset_proteins([1])  # P00002 → only PEPTIDE
    assert sub.n_proteins == 1
    assert sub.n_peptides == 1
    modseqs = sub.peptides.column("modified_sequence").to_pylist()
    assert modseqs == ["PEPTIDE"]


# ──────────────────────────────────────────────────────────────────────
# Network projection
# ──────────────────────────────────────────────────────────────────────


def test_to_network_protein_peptide_bipartite() -> None:
    lib = _minimal_library()
    g = lib.to_network("protein-peptide")
    assert g.n_nodes() == lib.n_proteins + lib.n_peptides
    assert g.n_edges() == lib.protein_peptide.num_rows
    assert not g.directed
    # Connected components: PEPTIDE connects P00001 and P00002, and via
    # P00001 reaches PEP[UNIMOD:4]TIDE — so all four nodes form one
    # component.
    components = g.connected_components()
    assert len(components) == 1
    assert len(components[0]) == 4


def test_to_network_peptide_precursor_bipartite() -> None:
    lib = _minimal_library()
    g = lib.to_network("peptide-precursor")
    assert g.n_nodes() == lib.n_peptides + lib.n_precursors
    assert g.n_edges() == lib.n_precursors
    assert not g.directed


def test_to_network_full_directed() -> None:
    lib = _minimal_library()
    g = lib.to_network("full")
    assert g.n_nodes() == lib.n_proteins + lib.n_peptides + lib.n_precursors
    assert g.n_edges() == (lib.protein_peptide.num_rows + lib.n_precursors)
    assert g.directed
    kinds = set(g.edges.column("kind").to_pylist())
    assert kinds == {"protein->peptide", "peptide->precursor"}


def test_to_network_unknown_tier_raises() -> None:
    lib = _minimal_library()
    with pytest.raises(ValueError, match="unknown tier"):
        lib.to_network("nonsense")  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────


def test_metadata_round_trip() -> None:
    lib = _minimal_library()
    lib2 = lib.with_metadata({"x.library.predictor": "Prosit-2024"})
    assert lib2.metadata["x.library.predictor"] == "Prosit-2024"
    # original unchanged (frozen dataclass)
    assert "x.library.predictor" not in lib.metadata


def test_unset_predicted_sentinel() -> None:
    lib = _minimal_library()
    rt = lib.precursors.column("rt_predicted").to_pylist()
    # First precursor passed no rt_predicted → -1.0 sentinel
    assert rt[0] == -1.0
    # Second passed 1234.5 explicitly
    assert rt[1] == 1234.5


def test_modified_sequence_strips_to_canonical() -> None:
    lib = _minimal_library()
    rows = lib.peptides.to_pylist()
    by_modseq = {r["modified_sequence"]: r["sequence"] for r in rows}
    assert by_modseq["PEPTIDE"] == "PEPTIDE"
    assert by_modseq["PEP[UNIMOD:4]TIDE"] == "PEPTIDE"
