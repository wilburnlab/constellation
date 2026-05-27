"""Tier A tests for :func:`constellation.massspec.search.collision.
filter_elib_by_losers`.

Builds a synthetic ``.elib`` SQLite with the peptide-keyed tables the
filter touches, then drops a known loser set + asserts:

  * loser rows are removed from every modseq-keyed table
  * non-loser rows survive intact
  * the peptidetoprotein table (keyed on canonical PeptideSeq) is
    correctly filtered via the entries → modseq → peptideseq projection
  * empty loser set produces a byte-equivalent copy (idempotency path)
  * tables absent from the source are silently skipped (count = 0)
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from constellation.massspec.search.collision import filter_elib_by_losers


def _build_synthetic_elib(path: Path, *, omit: tuple[str, ...] = ()) -> None:
    """Write a SQLite with the tables ``filter_elib_by_losers`` touches.

    ``omit`` is a tuple of table names to skip (for testing the
    "absent table → silent skip" path).
    """
    schemas: dict[str, str] = {
        "entries": (
            "CREATE TABLE entries ("
            "PrecursorMz REAL, PrecursorCharge INTEGER, "
            "PeptideModSeq TEXT, PeptideSeq TEXT, Copies INTEGER, "
            "RTInSeconds REAL, Score REAL, "
            "MassEncodedLength INTEGER, MassArray BLOB, "
            "SourceFile TEXT)"
        ),
        "peptidescores": (
            "CREATE TABLE peptidescores ("
            "PrecursorCharge INTEGER, PeptideModSeq TEXT, PeptideSeq TEXT, "
            "SourceFile TEXT, QValue REAL, "
            "PosteriorErrorProbability REAL, IsDecoy INTEGER)"
        ),
        "peptidequants": (
            "CREATE TABLE peptidequants ("
            "PrecursorCharge INTEGER, PeptideModSeq TEXT, "
            "SourceFile TEXT, TotalIntensity REAL)"
        ),
        "peptidetoprotein": (
            "CREATE TABLE peptidetoprotein ("
            "PeptideSeq TEXT, isDecoy INTEGER, ProteinAccession TEXT)"
        ),
        "fragmentquants": (
            "CREATE TABLE fragmentquants ("
            "PeptideModSeq TEXT, SourceFile TEXT, "
            "QuantIonMassArray BLOB, QuantIonMassLength INTEGER)"
        ),
    }
    con = sqlite3.connect(str(path))
    try:
        for name, ddl in schemas.items():
            if name in omit:
                continue
            con.execute(ddl)

        # entries: AAAR / BBBR / CCCR
        if "entries" not in omit:
            con.executemany(
                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (500.0, 2, "AAAR", "AAAR", 1, 100.0, 0.5, 0, b"", "f1.mzML"),
                    (500.0, 2, "BBBR", "BBBR", 1, 101.0, 0.4, 0, b"", "f1.mzML"),
                    (500.0, 2, "CCCR", "CCCR", 1, 102.0, 0.3, 0, b"", "f1.mzML"),
                    (500.0, 2, "AAAR", "AAAR", 1, 100.0, 0.6, 0, b"", "f2.mzML"),
                ],
            )

        # peptidescores: one row per (modseq, sourcefile).
        if "peptidescores" not in omit:
            con.executemany(
                "INSERT INTO peptidescores VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (2, "AAAR", "AAAR", "f1.mzML", 0.01, 0.001, 0),
                    (2, "BBBR", "BBBR", "f1.mzML", 0.02, 0.002, 0),
                    (2, "CCCR", "CCCR", "f1.mzML", 0.03, 0.003, 0),
                ],
            )

        # peptidequants
        if "peptidequants" not in omit:
            con.executemany(
                "INSERT INTO peptidequants VALUES (?, ?, ?, ?)",
                [
                    (2, "AAAR", "f1.mzML", 1.5e6),
                    (2, "BBBR", "f1.mzML", 7.5e5),
                    (2, "CCCR", "f1.mzML", 3.2e5),
                ],
            )

        # peptidetoprotein — keyed on PeptideSeq
        if "peptidetoprotein" not in omit:
            con.executemany(
                "INSERT INTO peptidetoprotein VALUES (?, ?, ?)",
                [
                    ("AAAR", 0, "PROT_A"),
                    ("BBBR", 0, "PROT_B"),
                    ("CCCR", 0, "PROT_C"),
                ],
            )

        # fragmentquants
        if "fragmentquants" not in omit:
            con.executemany(
                "INSERT INTO fragmentquants VALUES (?, ?, ?, ?)",
                [
                    ("AAAR", "f1.mzML", b"", 0),
                    ("BBBR", "f1.mzML", b"", 0),
                    ("CCCR", "f1.mzML", b"", 0),
                ],
            )
        con.commit()
    finally:
        con.close()


def _count(path: Path, table: str, where: str | None = None) -> int:
    con = sqlite3.connect(str(path))
    try:
        sql = f"SELECT COUNT(*) FROM {table}"
        if where is not None:
            sql += f" WHERE {where}"
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


def test_filter_drops_loser_rows_from_modseq_tables(tmp_path: Path) -> None:
    src = tmp_path / "input.elib"
    out = tmp_path / "filtered.elib"
    _build_synthetic_elib(src)

    deleted = filter_elib_by_losers(src, {"BBBR", "CCCR"}, out)

    # entries: 4 rows → AAAR×2 survives (one per SourceFile), BBBR+CCCR drop.
    assert _count(out, "entries") == 2
    assert _count(out, "entries", "PeptideModSeq='AAAR'") == 2
    assert _count(out, "entries", "PeptideModSeq IN ('BBBR','CCCR')") == 0

    # peptidescores: 3 rows → 1 (AAAR) survives.
    assert _count(out, "peptidescores") == 1
    assert _count(out, "peptidescores", "PeptideModSeq='AAAR'") == 1

    # peptidequants: same.
    assert _count(out, "peptidequants") == 1

    # fragmentquants: same.
    assert _count(out, "fragmentquants") == 1

    # peptidetoprotein (PeptideSeq-keyed): 3 → 1 (AAAR seq survives).
    assert _count(out, "peptidetoprotein") == 1

    # Return counts match what was actually deleted.
    assert deleted["entries"] == 2
    assert deleted["peptidescores"] == 2
    assert deleted["peptidequants"] == 2
    assert deleted["fragmentquants"] == 2
    assert deleted["peptidetoprotein"] == 2


def test_filter_preserves_non_loser_rows(tmp_path: Path) -> None:
    """Drop just one loser; assert the other two modseqs survive intact
    across all tables."""
    src = tmp_path / "input.elib"
    out = tmp_path / "filtered.elib"
    _build_synthetic_elib(src)
    filter_elib_by_losers(src, {"BBBR"}, out)
    # entries: AAAR×2 + CCCR survives.
    assert _count(out, "entries", "PeptideModSeq='AAAR'") == 2
    assert _count(out, "entries", "PeptideModSeq='CCCR'") == 1
    assert _count(out, "entries", "PeptideModSeq='BBBR'") == 0
    # peptidetoprotein: AAAR + CCCR survive (BBBR seq dropped).
    rows = set()
    con = sqlite3.connect(str(out))
    try:
        for r in con.execute("SELECT PeptideSeq FROM peptidetoprotein"):
            rows.add(r[0])
    finally:
        con.close()
    assert rows == {"AAAR", "CCCR"}


def test_filter_empty_loser_set_is_byte_equivalent_copy(tmp_path: Path) -> None:
    """No losers → output is a plain copy of the source. Important
    for the orchestrator's filter-then-export plumbing (always-runs
    path even when nothing collides)."""
    src = tmp_path / "input.elib"
    out = tmp_path / "filtered.elib"
    _build_synthetic_elib(src)
    deleted = filter_elib_by_losers(src, set(), out)
    assert all(v == 0 for v in deleted.values())
    # Byte-equivalent copy.
    assert src.read_bytes() == out.read_bytes()


def test_filter_skips_absent_tables(tmp_path: Path) -> None:
    """If the source elib doesn't have peptidequants / fragmentquants,
    the filter silently skips them and reports count = 0."""
    src = tmp_path / "input.elib"
    out = tmp_path / "filtered.elib"
    _build_synthetic_elib(src, omit=("peptidequants", "fragmentquants"))
    deleted = filter_elib_by_losers(src, {"BBBR"}, out)
    assert deleted["peptidequants"] == 0
    assert deleted["fragmentquants"] == 0
    # The modseq-keyed tables that DO exist still get filtered.
    assert deleted["entries"] >= 1
    assert deleted["peptidescores"] >= 1


def test_filter_writes_to_new_path(tmp_path: Path) -> None:
    """Output path is independent of the source. The source is not
    modified — important so the orchestrator can keep raw outputs
    in _raw/ alongside filtered primaries."""
    src = tmp_path / "input.elib"
    out = tmp_path / "subdir" / "filtered.elib"   # nested, doesn't exist yet
    _build_synthetic_elib(src)
    original_size = src.stat().st_size

    filter_elib_by_losers(src, {"BBBR"}, out)

    # Source unchanged.
    assert src.stat().st_size == original_size
    # Output created with parent dir auto-made.
    assert out.is_file()
    # Source still has all 4 entries rows; output has fewer.
    assert _count(src, "entries") == 4
    assert _count(out, "entries") < 4


def test_filter_overwrites_existing_output(tmp_path: Path) -> None:
    """Output path can pre-exist — shutil.copyfile overwrites."""
    src = tmp_path / "input.elib"
    out = tmp_path / "filtered.elib"
    out.write_bytes(b"stale")
    _build_synthetic_elib(src)
    filter_elib_by_losers(src, {"BBBR"}, out)
    assert _count(out, "entries") == 3   # AAAR×2 + CCCR
