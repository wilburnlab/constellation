"""Unit tests for ``constellation.core.structure.selection``."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from constellation.core.structure import (
    STRUCTURE_TABLE,
    select_atom_names,
    select_backbone,
    select_chain,
    select_element,
    select_hetatm,
    select_protein,
    select_resname,
    select_residues,
    select_sidechain,
    select_water,
)


# ──────────────────────────────────────────────────────────────────────
# Mini fixture: one alanine residue + one water
# ──────────────────────────────────────────────────────────────────────


def _ala_water_fixture() -> pa.Table:
    return pa.table(
        {
            "serial": pa.array([1, 2, 3, 4, 5, 6, 7], type=pa.int32()),
            "name": pa.array(["N", "CA", "C", "O", "CB", "O", "H1"], type=pa.string()),
            "alt_loc": pa.array([None] * 7, type=pa.string()),
            "res_name": pa.array(["ALA"] * 5 + ["HOH"] * 2, type=pa.string()),
            "chain_id": pa.array(["A"] * 5 + ["W"] * 2, type=pa.string()),
            "res_seq": pa.array([1, 1, 1, 1, 1, 100, 100], type=pa.int32()),
            "ins_code": pa.array([None] * 7, type=pa.string()),
            "segment_id": pa.array([None] * 7, type=pa.string()),
            "element": pa.array(["N", "C", "C", "O", "C", "O", "H"], type=pa.string()),
            "occupancy": pa.array([1.0] * 7, type=pa.float32()),
            "b_factor": pa.array([10.0] * 7, type=pa.float32()),
            "formal_charge": pa.array([0] * 7, type=pa.int8()),
            "is_hetatm": pa.array([False] * 5 + [True] * 2, type=pa.bool_()),
            "model_id": pa.array([1] * 7, type=pa.int32()),
        },
        schema=STRUCTURE_TABLE,
    )


# ──────────────────────────────────────────────────────────────────────
# Type / composition
# ──────────────────────────────────────────────────────────────────────


def test_helpers_return_expressions():
    for f in (
        select_chain("A"),
        select_residues(1, 1),
        select_resname({"ALA"}),
        select_atom_names({"CA"}),
        select_backbone(),
        select_protein(),
        select_sidechain(),
        select_water(),
        select_hetatm(),
        select_element("C"),
    ):
        assert isinstance(f, pc.Expression)


def test_compose_with_and_or_not():
    expr = select_protein() & select_backbone() & ~select_resname({"PRO"})
    assert isinstance(expr, pc.Expression)


# ──────────────────────────────────────────────────────────────────────
# Filter behavior on the fixture
# ──────────────────────────────────────────────────────────────────────


def test_select_chain_single():
    t = _ala_water_fixture()
    out = t.filter(select_chain("A"))
    assert out.num_rows == 5


def test_select_chain_set():
    t = _ala_water_fixture()
    out = t.filter(select_chain(["A", "W"]))
    assert out.num_rows == 7


def test_select_residues_inclusive_range():
    t = _ala_water_fixture()
    out = t.filter(select_residues(1, 100))
    assert out.num_rows == 7
    out = t.filter(select_residues(1, 1))
    assert out.num_rows == 5


def test_select_residues_with_chain_filter():
    t = _ala_water_fixture()
    out = t.filter(select_residues(1, 100, chain_id="A"))
    assert out.num_rows == 5  # excludes the water rows


def test_select_residues_rejects_inverted_range():
    with pytest.raises(ValueError):
        select_residues(10, 5)


def test_select_resname_filters():
    t = _ala_water_fixture()
    assert t.filter(select_resname("ALA")).num_rows == 5
    assert t.filter(select_resname({"HOH"})).num_rows == 2


def test_select_atom_names_ca_per_residue():
    t = _ala_water_fixture()
    out = t.filter(select_atom_names({"CA"}))
    assert out.num_rows == 1


def test_select_backbone_extracts_n_ca_c_o():
    t = _ala_water_fixture()
    out = t.filter(select_backbone() & select_protein())
    # The fixture has N, CA, C, O for ALA — the water 'O' is not in protein.
    assert out.num_rows == 4
    assert sorted(out.column("name").to_pylist()) == ["C", "CA", "N", "O"]


def test_select_protein_excludes_water():
    t = _ala_water_fixture()
    assert t.filter(select_protein()).num_rows == 5


def test_select_sidechain_excludes_backbone():
    t = _ala_water_fixture()
    out = t.filter(select_sidechain())
    # ALA sidechain is just CB.
    assert out.num_rows == 1
    assert out.column("name").to_pylist() == ["CB"]


def test_select_water_finds_hoh():
    t = _ala_water_fixture()
    assert t.filter(select_water()).num_rows == 2


def test_select_hetatm_finds_water_records():
    t = _ala_water_fixture()
    assert t.filter(select_hetatm()).num_rows == 2


def test_select_element_filters():
    t = _ala_water_fixture()
    out = t.filter(select_element("O"))
    assert out.num_rows == 2


def test_empty_iterable_rejected():
    with pytest.raises(ValueError):
        select_resname([])


def test_negation_works():
    t = _ala_water_fixture()
    out = t.filter(~select_water())
    assert out.num_rows == 5
