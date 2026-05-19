"""Tier A tests for `constellation.massspec.search.collision`.

Builds minimal synthetic ``.elib`` + ``.dia`` SQLite fixtures in
``tmp_path`` so the test runs without real EncyclopeDIA data. Covers:

  * lowest-Score wins within a collision cluster
  * RT > ``rt_threshold_s`` → not paired
  * < ``min_shared_ions`` overlapping fragment m/z → not paired
  * outside any GPF window → no candidate pair
  * same modseq across SourceFiles → never paired
  * no collisions → empty loser set

Port of cartographer's ``filter_elib_by_collision`` from
``cartographer/data/collision.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch

from constellation.massspec.io.encyclopedia._codec import compress_mz
from constellation.massspec.search.collision import (
    _connected_components,
    _merge_isolation_windows,
    apply_collision_filter,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic .elib + .dia builders
# ──────────────────────────────────────────────────────────────────────


def _build_elib(path: Path, entries: list[dict]) -> None:
    """Write a minimal SQLite .elib with just the columns the collision
    filter touches (PrecursorMz / RTInSeconds / Score / PeptideModSeq /
    PeptideSeq / MassArray / MassEncodedLength + a SourceFile str)."""
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE entries ("
            "PrecursorMz REAL, PrecursorCharge INTEGER, "
            "PeptideModSeq TEXT, PeptideSeq TEXT, Copies INTEGER, "
            "RTInSeconds REAL, Score REAL, "
            "MassEncodedLength INTEGER, MassArray BLOB, "
            "IntensityEncodedLength INTEGER, IntensityArray BLOB, "
            "CorrelationEncodedLength INTEGER, CorrelationArray BLOB, "
            "QuantifiedIonsArray BLOB, "
            "RTInSecondsStart REAL, RTInSecondsStop REAL, "
            "MedianChromatogramEncodedLength INTEGER, "
            "MedianChromatogramArray BLOB, "
            "SourceFile TEXT)"
        )
        for e in entries:
            mz_arr = torch.tensor(e["mz"], dtype=torch.float64)
            blob, n = compress_mz(mz_arr)
            con.execute(
                "INSERT INTO entries (PrecursorMz, PrecursorCharge, "
                "PeptideModSeq, PeptideSeq, Copies, RTInSeconds, Score, "
                "MassEncodedLength, MassArray, "
                "IntensityEncodedLength, IntensityArray, "
                "CorrelationEncodedLength, CorrelationArray, "
                "QuantifiedIonsArray, "
                "RTInSecondsStart, RTInSecondsStop, "
                "MedianChromatogramEncodedLength, "
                "MedianChromatogramArray, "
                "SourceFile) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e["precursor_mz"], 2, e["modseq"], e["modseq"], 1,
                    e["rt"], e["score"],
                    n, blob,
                    0, b"",
                    0, b"",
                    None,
                    e["rt"] - 1.0, e["rt"] + 1.0,
                    0, b"",
                    e.get("source_file", "run.mzML"),
                ),
            )
        con.commit()
    finally:
        con.close()


def _build_dia(path: Path, windows: list[tuple[float, float]]) -> None:
    """Write a minimal SQLite .dia with just the ``ranges`` table."""
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE ranges ("
            "Start REAL, Stop REAL, DutyCycle REAL, NumWindows INTEGER)"
        )
        for lo, hi in windows:
            con.execute(
                "INSERT INTO ranges VALUES (?, ?, ?, ?)",
                (lo, hi, 1.0, len(windows)),
            )
        con.commit()
    finally:
        con.close()


# ──────────────────────────────────────────────────────────────────────
# Pure-function unit tests
# ──────────────────────────────────────────────────────────────────────


def test_merge_isolation_windows_dedups_near_duplicates() -> None:
    starts = [400.0, 400.0001, 400.0002, 500.0]
    stops = [410.0, 410.0001, 410.0002, 510.0]
    merged = _merge_isolation_windows(starts, stops)
    assert len(merged) == 2
    assert merged[0] == (400.0001, 410.0001)
    assert merged[1] == (500.0, 510.0)


def test_connected_components_simple_chain() -> None:
    pairs = [("a", "b"), ("b", "c"), ("d", "e")]
    components = _connected_components(pairs)
    components_as_frozen = {frozenset(c) for c in components}
    assert components_as_frozen == {
        frozenset({"a", "b", "c"}),
        frozenset({"d", "e"}),
    }


def test_connected_components_empty() -> None:
    assert _connected_components([]) == []


# ──────────────────────────────────────────────────────────────────────
# End-to-end against synthetic .elib + .dia
# ──────────────────────────────────────────────────────────────────────


def test_collision_filter_keeps_lowest_score(tmp_path: Path) -> None:
    """Two co-eluting peptides in same window sharing ≥4 fragment ions:
    the higher-Score (worse) one should be flagged as the loser."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    shared = [100.0, 200.0, 300.0, 400.0]
    _build_elib(
        elib,
        [
            {
                "modseq": "PEPTIDEAR", "precursor_mz": 500.0,
                "rt": 1000.0, "score": 0.01,  # better (lower)
                "mz": shared + [500.0],
            },
            {
                "modseq": "PEPTIDEBR", "precursor_mz": 500.5,
                "rt": 1002.0, "score": 0.1,  # worse (higher)
                "mz": shared + [600.0],
            },
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers = apply_collision_filter(elib, dia)
    assert losers == {"PEPTIDEBR"}


def test_collision_filter_returns_metadata(tmp_path: Path) -> None:
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    shared = [100.0, 200.0, 300.0, 400.0]
    _build_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.01, "mz": shared},
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1002.0,
             "score": 0.1, "mz": shared},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers, meta = apply_collision_filter(elib, dia, return_metadata=True)
    assert losers == {"BBBR"}
    assert meta["pairs"] == [("AAAR", "BBBR")]
    assert meta["clusters"] == [{"winner": "AAAR", "losers": ["BBBR"]}]
    assert meta["n_windows"] == 1
    assert meta["n_entries"] == 2


def test_collision_filter_rt_outside_threshold_not_paired(tmp_path: Path) -> None:
    """ΔRT > rt_threshold_s → no collision pair, no losers."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    shared = [100.0, 200.0, 300.0, 400.0]
    _build_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.01, "mz": shared},
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1010.0,
             "score": 0.1, "mz": shared},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers = apply_collision_filter(elib, dia, rt_threshold_s=5.0)
    assert losers == set()


def test_collision_filter_few_shared_ions_not_paired(tmp_path: Path) -> None:
    """< min_shared_ions overlapping fragments → no collision pair."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    _build_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.01, "mz": [100.0, 200.0, 300.0]},
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1002.0,
             "score": 0.1, "mz": [100.0, 200.0, 800.0]},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers = apply_collision_filter(elib, dia, min_shared_ions=4)
    assert losers == set()


def test_collision_filter_outside_any_window_skipped(tmp_path: Path) -> None:
    """PrecursorMz outside any GPF window → entry not paired against anything."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    shared = [100.0, 200.0, 300.0, 400.0]
    _build_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 700.0, "rt": 1000.0,
             "score": 0.01, "mz": shared},  # outside [450, 550]
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1002.0,
             "score": 0.1, "mz": shared},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers = apply_collision_filter(elib, dia)
    assert losers == set()


def test_collision_filter_same_modseq_across_sourcefiles_not_paired(
    tmp_path: Path,
) -> None:
    """Same modseq reported across two SourceFiles must not flag itself."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    shared = [100.0, 200.0, 300.0, 400.0]
    _build_elib(
        elib,
        [
            {"modseq": "SAMEMODSEQR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.01, "mz": shared, "source_file": "run1.mzML"},
            {"modseq": "SAMEMODSEQR", "precursor_mz": 500.0, "rt": 1002.0,
             "score": 0.02, "mz": shared, "source_file": "run2.mzML"},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers = apply_collision_filter(elib, dia)
    assert losers == set()


def test_collision_filter_no_collisions(tmp_path: Path) -> None:
    """Two unrelated peptides → empty loser set."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    _build_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.01, "mz": [100.0, 200.0]},
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1100.0,
             "score": 0.1, "mz": [700.0, 800.0]},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    assert apply_collision_filter(elib, dia) == set()


def test_collision_filter_three_way_cluster_keeps_best(tmp_path: Path) -> None:
    """A→B and B→C pair-flag; cluster {A, B, C} → only the lowest-Score
    survives, the other two are losers."""
    elib = tmp_path / "test.elib"
    dia = tmp_path / "test.dia"
    shared_ab = [100.0, 200.0, 300.0, 400.0]
    shared_bc = [500.0, 600.0, 700.0, 800.0]
    _build_elib(
        elib,
        [
            {"modseq": "AAAR", "precursor_mz": 500.0, "rt": 1000.0,
             "score": 0.5, "mz": shared_ab},
            {"modseq": "BBBR", "precursor_mz": 500.0, "rt": 1001.0,
             "score": 0.01, "mz": shared_ab + shared_bc},
            {"modseq": "CCCR", "precursor_mz": 500.0, "rt": 1002.0,
             "score": 0.3, "mz": shared_bc},
        ],
    )
    _build_dia(dia, [(450.0, 550.0)])
    losers, meta = apply_collision_filter(elib, dia, return_metadata=True)
    assert losers == {"AAAR", "CCCR"}
    assert len(meta["clusters"]) == 1
    assert meta["clusters"][0]["winner"] == "BBBR"
    assert set(meta["clusters"][0]["losers"]) == {"AAAR", "CCCR"}
