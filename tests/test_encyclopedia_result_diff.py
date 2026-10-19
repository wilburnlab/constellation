"""``compare_results``: overlap and agreement of two EncyclopeDIA searches."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from constellation.massspec.io.encyclopedia import compare_results
from constellation.massspec.io.encyclopedia._modseq import encyclopedia_residue_form

# (modseq, charge, rt seconds, score, total intensity or None, protein)
_BASE = [
    ("PEPTIDEK", 2, 100.0, 0.001, 1.0e6, "P1"),
    ("PEPTIDEK", 3, 100.5, 0.002, 2.0e5, "P1"),
    ("M[+15.994915]ELVISK", 2, 250.0, 0.003, 4.0e6, "P2"),
    ("SAMPLER", 2, 400.0, 0.004, None, "P3"),
]


def _write_result(path: Path, rows, *, source: str = "run.raw", decoy_protein: str | None = None):
    """A result library with just the columns the comparison reads."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE entries (PeptideModSeq TEXT, PrecursorCharge INT, RTInSeconds REAL,
                              Score REAL, SourceFile TEXT);
        CREATE TABLE peptidequants (PeptideModSeq TEXT, PrecursorCharge INT,
                                    TotalIntensity REAL, SourceFile TEXT);
        CREATE TABLE peptidescores (PeptideModSeq TEXT, PrecursorCharge INT, SourceFile TEXT);
        CREATE TABLE proteinscores (ProteinAccession TEXT, IsDecoy INT, SourceFile TEXT);
        """
    )
    proteins = set()
    for i, (modseq, charge, rt, score, total, protein) in enumerate(rows):
        src = source if isinstance(source, str) else source[i % len(source)]
        con.execute("INSERT INTO entries VALUES (?,?,?,?,?)", (modseq, charge, rt, score, src))
        con.execute("INSERT INTO peptidescores VALUES (?,?,?)", (modseq, charge, src))
        if total is not None:
            # As the jar does: N-terminal modifications on the first residue here.
            con.execute(
                "INSERT INTO peptidequants VALUES (?,?,?,?)",
                (encyclopedia_residue_form(modseq), charge, total, src),
            )
        proteins.add(protein)
    for protein in sorted(proteins):
        con.execute("INSERT INTO proteinscores VALUES (?,0,?)", (protein, "run.raw"))
    if decoy_protein:
        con.execute("INSERT INTO proteinscores VALUES (?,1,?)", (decoy_protein, "run.raw"))
    con.commit()
    con.close()
    return path


def test_a_result_compared_with_itself_is_identical(tmp_path: Path) -> None:
    a = _write_result(tmp_path / "a.elib", _BASE)
    result = compare_results(a, a)
    assert result.identical_identifications
    assert (result.precursors.n_a, result.precursors.n_shared) == (4, 4)
    assert result.peptides.n_shared == 3 and result.proteins.n_shared == 3
    assert result.precursors.jaccard == 1.0
    assert result.rt_abs_s["max"] == 0.0 and result.score_abs["max"] == 0.0
    # Three of the four precursors are quantified; identical vectors give r = 1.
    assert result.intensity["n"] == 3 and result.intensity["pearson_log"] == 1.0
    assert result.intensity["frac_within_1pct"] == 1.0


def test_overlap_and_agreement_are_measured_over_the_shared_precursors(tmp_path: Path) -> None:
    a = _write_result(tmp_path / "a.elib", _BASE)
    other = [
        ("PEPTIDEK", 2, 101.0, 0.0015, 2.0e6, "P1"),  # shared: +1 s, 2x intensity
        ("PEPTIDEK", 3, 100.5, 0.002, 2.0e5, "P1"),  # shared: unchanged
        ("M[+15.994915]ELVISK", 2, 250.0, 0.003, 4.0e6, "P2"),  # shared: unchanged
        ("NEWPEPTIDER", 2, 500.0, 0.005, 1.0e5, "P4"),  # only in b; SAMPLER only in a
    ]
    b = _write_result(tmp_path / "b.elib", other, decoy_protein="DECOY_P9")
    result = compare_results(a, b)

    assert not result.identical_identifications
    assert (result.precursors.only_a, result.precursors.only_b, result.precursors.n_shared) == (
        1,
        1,
        3,
    )
    assert result.precursors.jaccard == pytest.approx(3 / 5)
    assert (result.peptides.n_a, result.peptides.n_b, result.peptides.n_shared) == (3, 3, 2)
    # Decoy proteins are not identifications.
    assert (result.proteins.n_a, result.proteins.n_b, result.proteins.n_shared) == (3, 3, 2)
    assert result.rt_abs_s["max"] == pytest.approx(1.0)
    assert result.rt_abs_s["median"] == 0.0
    assert result.score_abs["max"] == pytest.approx(0.0005)
    assert result.intensity["n"] == 3
    assert result.intensity["frac_within_1pct"] == pytest.approx(2 / 3)
    report = result.report()
    assert "precursors: 4 vs 4; shared 3" in report and "only a 1, only b 1" in report


def test_n_terminal_modifications_join_across_the_two_spellings(tmp_path: Path) -> None:
    """6.9.23 writes ``[+42.010565]MDFQ...`` in ``entries`` and
    ``M[+42.010565]DFQ...`` in ``peptidequants``; the quantity must
    still be found, also when the first residue has its own modification."""
    from constellation.massspec.io.encyclopedia._result_diff import _quant_join_key

    assert _quant_join_key("[+42.010565]MDFQHRPGGK", 3) == ("M[+42.010565]DFQHRPGGK", 3)
    assert _quant_join_key("[+42.010565]C[+57.0214635]DFTEDK", 2) == (
        "C[+42.010565][+57.0214635]DFTEDK",
        2,
    )
    assert _quant_join_key("M[+15.994915]ELVISK", 2) == ("M[+15.994915]ELVISK", 2)

    def _result(path: Path, total: float) -> Path:
        con = sqlite3.connect(_write_result(path, []))
        con.execute(
            "INSERT INTO entries VALUES ('[-17.026549]QLSALHR', 2, 300.0, 0.001, 'run.raw')"
        )
        con.execute(
            "INSERT INTO peptidequants VALUES ('Q[-17.026549]LSALHR', 2, ?, 'run.raw')", (total,)
        )
        con.commit()
        con.close()
        return path

    result = compare_results(_result(tmp_path / "a.elib", 1.0e6), _result(tmp_path / "b.elib", 2.0e6))
    assert result.precursors.n_shared == 1
    assert result.intensity["n"] == 1
    assert result.intensity["median_abs_log2"] == pytest.approx(1.0)


def test_the_run_name_is_not_compared(tmp_path: Path) -> None:
    """The same acquisition searched as ``run.raw`` and as ``run.mzML``."""
    a = _write_result(tmp_path / "a.elib", _BASE, source="run.raw")
    b = _write_result(tmp_path / "b.elib", _BASE, source="run.mzML")
    assert compare_results(a, b).identical_identifications


def test_multi_run_and_non_result_files_are_refused(tmp_path: Path) -> None:
    multi = _write_result(tmp_path / "multi.elib", _BASE, source=("one.raw", "two.raw"))
    single = _write_result(tmp_path / "single.elib", _BASE)
    with pytest.raises(ValueError, match="holds 2 runs"):
        compare_results(multi, single)

    not_a_result = tmp_path / "library.dlib"
    con = sqlite3.connect(not_a_result)
    con.execute("CREATE TABLE entries (PeptideModSeq TEXT)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="not a result library"):
        compare_results(not_a_result, single)
    with pytest.raises(FileNotFoundError):
        compare_results(tmp_path / "absent.elib", single)


def test_cli_exit_status_follows_min_overlap(tmp_path: Path, capsys) -> None:
    from constellation.cli.__main__ import main

    a = _write_result(tmp_path / "a.elib", _BASE)
    b = _write_result(tmp_path / "b.elib", _BASE[:3])  # 3 of 4: Jaccard 0.75
    base = ["massspec", "encyclopedia", "compare-results", str(a), str(b)]

    assert main(base) == 1
    assert "Jaccard 0.7500" in capsys.readouterr().out
    assert main([*base, "--min-overlap", "0.7"]) == 0
    assert main(["massspec", "encyclopedia", "compare-results", str(a), str(a)]) == 0
    capsys.readouterr()
    assert main([*base, "--min-overlap", "1.5"]) == 2
    assert main(["massspec", "encyclopedia", "compare-results", str(a), str(tmp_path / "x")]) == 2


# ── identity is the entry's own spelling, not the join key ─────────────
# (Regression from the review of b1fe1ca: the residue-form key used for
# joining quantities was also used as precursor identity, so N-terminal
# and lysine acetylation on a leading K were one precursor.)

_N_TERMINAL = ("[+42.010565]KVERPDK", 2, 300.0, 0.001, 1.0e6, "P1")
_ON_LYSINE = ("K[+42.010565]VERPDK", 2, 310.0, 0.002, 2.0e6, "P1")


def test_n_terminal_and_side_chain_modification_are_different_precursors(tmp_path: Path) -> None:
    a = _write_result(tmp_path / "a.elib", [_N_TERMINAL])
    b = _write_result(tmp_path / "b.elib", [_ON_LYSINE])
    result = compare_results(a, b)
    assert (result.precursors.n_a, result.precursors.n_b, result.precursors.n_shared) == (1, 1, 0)
    assert result.peptides.n_shared == 0
    assert not result.identical_identifications


def test_both_placements_in_one_file_are_both_counted(tmp_path: Path) -> None:
    """They used to overwrite each other in a dict."""
    both = _write_result(tmp_path / "both.elib", [_N_TERMINAL, _ON_LYSINE])
    only_one = _write_result(tmp_path / "one.elib", [_N_TERMINAL])
    result = compare_results(both, only_one)
    assert (result.precursors.n_a, result.precursors.n_b, result.precursors.n_shared) == (2, 1, 1)
    assert result.peptides.n_a == 2


def test_a_quantity_that_fits_two_precursors_is_left_out_and_reported(tmp_path: Path) -> None:
    """``peptidequants`` writes both placements as ``K[+42.010565]...``.
    When both precursors exist the quantity rows cannot be told apart;
    they are excluded from the quantity comparison and the report says so."""
    both = _write_result(tmp_path / "both.elib", [_N_TERMINAL, _ON_LYSINE, _BASE[0]])
    result = compare_results(both, both)
    assert result.precursors.n_shared == 3
    assert result.intensity["n"] == 1  # only PEPTIDEK 2+ is unambiguous
    assert result.ambiguous_quantities == (2, 2)
    assert "2 and 2 quantity rows fit more than one precursor" in result.report()
    # With one placement present the row is unambiguous and is used.
    one = _write_result(tmp_path / "one.elib", [_N_TERMINAL])
    alone = compare_results(one, one)
    assert alone.intensity["n"] == 1 and alone.ambiguous_quantities == (0, 0)
    assert "fit more than one precursor" not in alone.report()


def test_two_spellings_of_one_precursor_in_entries_are_refused(tmp_path: Path) -> None:
    path = _write_result(
        tmp_path / "dup.elib",
        [("M[+42.010565]DFQK", 2, 1.0, 0.1, 1.0, "P1"), ("[+42.010565]MDFQK", 2, 1.0, 0.1, 1.0, "P1")],
    )
    with pytest.raises(ValueError, match="the same precursor twice"):
        compare_results(path, path)
