"""Inclusion-list projection: Compound naming, CSV shape, collisions."""

from __future__ import annotations

import pyarrow.csv as pa_csv
import pytest

from constellation.massspec.library.digest import PrecursorSpec, precursors_from_fasta
from constellation.massspec.library.inclusion import (
    build_inclusion_list,
    compound_name,
    count_levels,
    dedupe_specs,
    find_mz_collisions,
    write_inclusion_list,
)

# Two proteins sharing LVNELTEFAK, so the shared-peptide path is covered.
_FASTA = """\
>sp|TEST1|ONE Test protein one
MKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEK
>sp|TEST2|TWO Test protein two
MKLVNELTEFAKDDSPDLPKMDEMKSLGKVGTR
"""


def _spec(modseq: str, charge: int, mz: float, seq: str | None = None) -> PrecursorSpec:
    return PrecursorSpec(
        modified_sequence=modseq,
        sequence=seq if seq is not None else modseq,
        charge=charge,
        precursor_mz=mz,
    )


# ── Compound naming ────────────────────────────────────────────────────


def test_compound_name_unmodified() -> None:
    assert compound_name(_spec("PEPTIDEK", 2, 500.0)) == "PEPTIDEK_2"


def test_compound_name_renders_unimod_accession() -> None:
    spec = _spec("PEPTIDEC[UNIMOD:4]K", 2, 500.0, seq="PEPTIDECK")
    assert compound_name(spec) == "PEPTIDEC[UNIMOD:4]K_2"


def test_compound_name_terminal_mod() -> None:
    spec = _spec("[UNIMOD:1]-QPEPTIDEK", 2, 500.0, seq="QPEPTIDEK")
    assert compound_name(spec) == "[UNIMOD:1]-QPEPTIDEK_2"


def test_compound_name_strips_proforma_charge_suffix() -> None:
    """``PrecursorSpec.charge`` is the authority; a ProForma ``/N`` suffix
    (which the peptide-list and Library paths can carry) must not leave
    the charge stamped twice."""
    assert compound_name(_spec("PEPTIDEK/2", 2, 500.0)) == "PEPTIDEK_2"
    assert compound_name(_spec("PEPTIDEK/[+Na+]", 2, 500.0)) == "PEPTIDEK_2"


def test_compound_name_separator_is_configurable() -> None:
    assert compound_name(_spec("PEPTIDEK", 3, 400.0), separator="+") == "PEPTIDEK+3"


# ── table shape ────────────────────────────────────────────────────────


def test_columns_are_exactly_compound_and_mz() -> None:
    table = build_inclusion_list([_spec("PEPTIDEK", 2, 500.0)])
    assert table.column_names == ["Compound", "m/z"]


def test_build_dedupes_on_compound() -> None:
    specs = [
        _spec("PEPTIDEK", 2, 500.0),
        _spec("PEPTIDEK", 2, 500.0),
        _spec("PEPTIDEK", 3, 333.7),
    ]
    assert build_inclusion_list(specs).num_rows == 2
    assert build_inclusion_list(specs, dedupe=False).num_rows == 3


def test_dedupe_specs_keeps_first_seen_order() -> None:
    specs = [_spec("BK", 2, 200.0), _spec("AK", 2, 100.0), _spec("BK", 2, 200.0)]
    assert [s.modified_sequence for s in dedupe_specs(specs)] == ["BK", "AK"]


def test_build_sorts_by_mz_ascending() -> None:
    specs = [_spec("CK", 2, 700.0), _spec("AK", 2, 500.0), _spec("BK", 2, 600.0)]
    table = build_inclusion_list(specs)
    assert table.column("m/z").to_pylist() == [500.0, 600.0, 700.0]


def test_build_sort_modes() -> None:
    specs = [_spec("CK", 2, 500.0), _spec("AK", 2, 700.0)]
    assert build_inclusion_list(specs, sort="compound").column("Compound").to_pylist() == [
        "AK_2",
        "CK_2",
    ]
    assert build_inclusion_list(specs, sort="input").column("Compound").to_pylist() == [
        "CK_2",
        "AK_2",
    ]
    with pytest.raises(ValueError, match="unknown sort"):
        build_inclusion_list(specs, sort="nonsense")  # type: ignore[arg-type]


def test_build_keeps_full_precision() -> None:
    """Rounding is a write-time concern only."""
    table = build_inclusion_list([_spec("PEPTIDEK", 2, 582.3190123456)])
    assert table.column("m/z")[0].as_py() == pytest.approx(582.3190123456, abs=1e-12)


def test_count_levels_separates_peptides_from_entries() -> None:
    specs = [
        _spec("PEPTIDEMK", 2, 500.0, seq="PEPTIDEMK"),
        _spec("PEPTIDEMK", 3, 333.7, seq="PEPTIDEMK"),
        _spec("PEPTIDEM[UNIMOD:35]K", 2, 508.0, seq="PEPTIDEMK"),
    ]
    assert count_levels(specs) == {"entries": 3, "peptides": 1, "peptidoforms": 2}


# ── CSV writing ────────────────────────────────────────────────────────


def test_write_header_is_unquoted_and_exact(tmp_path) -> None:
    """Method-editor importers match header text literally, and
    ``pyarrow.csv.write_csv`` always quotes it — this pins the hand-rolled
    writer in place."""
    path = write_inclusion_list(
        build_inclusion_list([_spec("PEPTIDEK", 2, 500.0)]), tmp_path / "list.csv"
    )
    assert path.read_bytes().split(b"\r\n")[0] == b"Compound,m/z"


def test_write_leaves_modified_compound_unquoted(tmp_path) -> None:
    spec = _spec("PEPTIDEC[UNIMOD:4]K", 2, 500.0, seq="PEPTIDECK")
    path = write_inclusion_list(build_inclusion_list([spec]), tmp_path / "list.csv")
    assert b"PEPTIDEC[UNIMOD:4]K_2,500.000\r\n" in path.read_bytes()


def test_write_uses_crlf(tmp_path) -> None:
    path = write_inclusion_list(
        build_inclusion_list([_spec("PEPTIDEK", 2, 500.0)]), tmp_path / "list.csv"
    )
    raw = path.read_bytes()
    assert raw.endswith(b"\r\n")
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_write_line_terminator_override(tmp_path) -> None:
    path = write_inclusion_list(
        build_inclusion_list([_spec("PEPTIDEK", 2, 500.0)]),
        tmp_path / "list.csv",
        line_terminator="\n",
    )
    assert b"\r" not in path.read_bytes()


def test_write_rounds_to_three_decimals_keeping_trailing_zeros(tmp_path) -> None:
    path = write_inclusion_list(
        build_inclusion_list([_spec("PEPTIDEK", 2, 1002.7)]), tmp_path / "list.csv"
    )
    assert b"PEPTIDEK_2,1002.700" in path.read_bytes()


def test_write_decimals_override(tmp_path) -> None:
    path = write_inclusion_list(
        build_inclusion_list([_spec("PEPTIDEK", 2, 582.3190123)]),
        tmp_path / "list.csv",
        decimals={"m/z": 5},
    )
    assert b"PEPTIDEK_2,582.31901" in path.read_bytes()


def test_write_creates_parent_directories(tmp_path) -> None:
    path = write_inclusion_list(
        build_inclusion_list([_spec("PEPTIDEK", 2, 500.0)]),
        tmp_path / "nested" / "deeper" / "list.csv",
    )
    assert path.is_file()


def test_write_roundtrips_through_pyarrow_csv_read(tmp_path) -> None:
    specs = [_spec("PEPTIDEK", 2, 582.319012), _spec("SAMPLERK", 3, 401.5551)]
    path = write_inclusion_list(build_inclusion_list(specs), tmp_path / "list.csv")
    back = pa_csv.read_csv(path)
    assert back.column_names == ["Compound", "m/z"]
    assert back.column("m/z").to_pylist() == pytest.approx(
        [401.555, 582.319], abs=1e-3
    )


# ── collisions ─────────────────────────────────────────────────────────


def test_find_mz_collisions_flags_near_isobars() -> None:
    table = build_inclusion_list(
        [
            _spec("AK", 2, 500.0000),
            _spec("BK", 2, 500.0020),  # 4 ppm away
            _spec("CK", 2, 700.0000),
        ]
    )
    hits = find_mz_collisions(table, tolerance_ppm=10.0)
    assert hits.num_rows == 1
    row = hits.to_pylist()[0]
    assert {row["compound_a"], row["compound_b"]} == {"AK_2", "BK_2"}
    assert row["delta_ppm"] == pytest.approx(4.0, abs=0.1)
    assert find_mz_collisions(table, tolerance_ppm=1.0).num_rows == 0


def test_find_mz_collisions_columns_on_empty_result() -> None:
    hits = find_mz_collisions(build_inclusion_list([_spec("AK", 2, 500.0)]))
    assert hits.num_rows == 0
    assert hits.column_names == [
        "compound_a",
        "compound_b",
        "mz_a",
        "mz_b",
        "delta_ppm",
    ]


# ── end to end from a FASTA ────────────────────────────────────────────


def test_end_to_end_from_fasta(tmp_path) -> None:
    fasta = tmp_path / "targets.fasta"
    fasta.write_text(_FASTA)

    specs = precursors_from_fasta(
        fasta,
        protease="Trypsin",
        missed_cleavages=0,
        charges=(2,),
        min_mz=350.0,
        max_mz=2000.0,
        fixed_mods={"C": "UNIMOD:4"},
    )
    table = build_inclusion_list(specs)
    path = write_inclusion_list(table, tmp_path / "inclusion_list.csv")

    rows = dict(
        zip(
            table.column("Compound").to_pylist(),
            table.column("m/z").to_pylist(),
            strict=True,
        )
    )
    assert rows["LVNELTEFAK_2"] == pytest.approx(582.319, abs=1e-3)
    # Shared between both proteins, but one target on the instrument.
    assert path.read_text().count("LVNELTEFAK_2,") == 1
    # Cysteines carry the fixed mod in UNIMOD form.
    assert any("C[UNIMOD:4]" in name for name in rows)
