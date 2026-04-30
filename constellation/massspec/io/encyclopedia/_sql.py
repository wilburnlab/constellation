"""SQLite connection + safe row-iterator helpers for EncyclopeDIA files.

Cartographer's ``read_encyclopedia_table`` interpolates the table name
straight into a ``SELECT * FROM {table}`` query, which is a SQL-injection
hazard if the caller is ever exposed to untrusted input. Constellation's
port whitelists table names against the ``_schema`` constants and
refuses anything else.

Every ``iter_*`` function returns a generator yielding plain dicts
(column-name → value). Callers are responsible for closing the
connection (use the ``connect`` helpers in a context manager).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from constellation.massspec.io.encyclopedia._schema import (
    DIA_TABLE_NAMES,
    LIBRARY_TABLE_NAMES,
)


# ──────────────────────────────────────────────────────────────────────
# Connection helpers
# ──────────────────────────────────────────────────────────────────────


@contextmanager
def open_ro(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the SQLite file read-only via the ``mode=ro`` URI parameter.

    Read-only mode prevents accidental writes (e.g. from journal-replay
    on a partial open) and is the right contract for a reader.
    """
    uri = f"file:{path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        yield con
    finally:
        con.close()


@contextmanager
def open_rw(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the SQLite file read-write — used by the writer path."""
    con = sqlite3.connect(path)
    try:
        yield con
    finally:
        con.close()


# ──────────────────────────────────────────────────────────────────────
# Table-name whitelisting
# ──────────────────────────────────────────────────────────────────────


def _ensure_library_table(name: str) -> None:
    if name not in LIBRARY_TABLE_NAMES:
        raise ValueError(
            f"unknown encyclopedia library table {name!r}; "
            f"expected one of {sorted(LIBRARY_TABLE_NAMES)}"
        )


def _ensure_dia_table(name: str) -> None:
    if name not in DIA_TABLE_NAMES:
        raise ValueError(
            f"unknown encyclopedia .dia table {name!r}; "
            f"expected one of {sorted(DIA_TABLE_NAMES)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Generic iteration helpers
# ──────────────────────────────────────────────────────────────────────


def has_table(con: sqlite3.Connection, name: str) -> bool:
    """Whether the named table exists in the connected database."""
    cur = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def has_rows(con: sqlite3.Connection, name: str) -> bool:
    """``True`` iff the named library table exists *and* has rows.

    Used by the format-detection sniffer (e.g. "elib if peptidequants
    has any rows"). Whitelist-checked.
    """
    if name in LIBRARY_TABLE_NAMES:
        pass
    elif name in DIA_TABLE_NAMES:
        pass
    else:
        raise ValueError(f"unknown encyclopedia table {name!r}")
    if not has_table(con, name):
        return False
    cur = con.execute(f"SELECT 1 FROM {name} LIMIT 1")
    return cur.fetchone() is not None


def fetch_metadata(con: sqlite3.Connection) -> dict[str, str]:
    """Fetch the ``metadata`` table as a flat key→value dict."""
    if not has_table(con, "metadata"):
        return {}
    cur = con.execute("SELECT Key, Value FROM metadata")
    return {row[0]: row[1] for row in cur.fetchall()}


def _iter_dicts(
    con: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> Iterator[dict[str, Any]]:
    """Cursor → dict iterator with whitelisted table + columns."""
    col_list = ", ".join(columns)
    cur = con.execute(f"SELECT {col_list} FROM {table}")
    for row in cur:
        yield dict(zip(columns, row, strict=True))


# ──────────────────────────────────────────────────────────────────────
# Library-side iterators (entries / peptidetoprotein / scores / quants)
# ──────────────────────────────────────────────────────────────────────


_ENTRIES_COLS: tuple[str, ...] = (
    "PrecursorMz",
    "PrecursorCharge",
    "PeptideModSeq",
    "PeptideSeq",
    "Copies",
    "RTInSeconds",
    "Score",
    "MassEncodedLength",
    "MassArray",
    "IntensityEncodedLength",
    "IntensityArray",
    "CorrelationEncodedLength",
    "CorrelationArray",
    "QuantifiedIonsArray",
    "RTInSecondsStart",
    "RTInSecondsStop",
    "MedianChromatogramEncodedLength",
    "MedianChromatogramArray",
    "SourceFile",
)


def iter_entries(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_library_table("entries")
    return _iter_dicts(con, "entries", _ENTRIES_COLS)


_PEPTIDETOPROTEIN_COLS = ("PeptideSeq", "isDecoy", "ProteinAccession")


def iter_peptidetoprotein(
    con: sqlite3.Connection,
) -> Iterator[dict[str, Any]]:
    _ensure_library_table("peptidetoprotein")
    return _iter_dicts(con, "peptidetoprotein", _PEPTIDETOPROTEIN_COLS)


_PEPTIDESCORES_COLS = (
    "PrecursorCharge",
    "PeptideModSeq",
    "PeptideSeq",
    "SourceFile",
    "QValue",
    "PosteriorErrorProbability",
    "IsDecoy",
)


def iter_peptidescores(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_library_table("peptidescores")
    return _iter_dicts(con, "peptidescores", _PEPTIDESCORES_COLS)


_PROTEINSCORES_COLS = (
    "ProteinGroup",
    "ProteinAccession",
    "SourceFile",
    "QValue",
    "MinimumPeptidePEP",
    "IsDecoy",
)


def iter_proteinscores(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_library_table("proteinscores")
    return _iter_dicts(con, "proteinscores", _PROTEINSCORES_COLS)


_PEPTIDEQUANTS_COLS = (
    "PrecursorCharge",
    "PeptideModSeq",
    "PeptideSeq",
    "SourceFile",
    "RTInSecondsCenter",
    "RTInSecondsStart",
    "RTInSecondsStop",
    "TotalIntensity",
    "NumberOfQuantIons",
    "BestFragmentCorrelation",
    "BestFragmentDeltaMassPPM",
    "IdentifiedTICRatio",
)


def iter_peptidequants(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_library_table("peptidequants")
    return _iter_dicts(con, "peptidequants", _PEPTIDEQUANTS_COLS)


# ──────────────────────────────────────────────────────────────────────
# .dia iterators (precursor / spectra / ranges)
# ──────────────────────────────────────────────────────────────────────


_DIA_PRECURSOR_COLS = (
    "Fraction",
    "SpectrumName",
    "SpectrumIndex",
    "ScanStartTime",
    "IonInjectionTime",
    "IsolationWindowLower",
    "IsolationWindowUpper",
    "MassEncodedLength",
    "MassArray",
    "IntensityEncodedLength",
    "IntensityArray",
    "TIC",
)


def iter_dia_precursor(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_dia_table("precursor")
    return _iter_dicts(con, "precursor", _DIA_PRECURSOR_COLS)


_DIA_SPECTRA_COLS = (
    "Fraction",
    "SpectrumName",
    "PrecursorName",
    "SpectrumIndex",
    "ScanStartTime",
    "IonInjectionTime",
    "IsolationWindowLower",
    "IsolationWindowCenter",
    "IsolationWindowUpper",
    "PrecursorCharge",
    "MassEncodedLength",
    "MassArray",
    "IntensityEncodedLength",
    "IntensityArray",
)


def iter_dia_spectra(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_dia_table("spectra")
    return _iter_dicts(con, "spectra", _DIA_SPECTRA_COLS)


_DIA_RANGES_COLS = ("Start", "Stop", "DutyCycle", "NumWindows")


def iter_dia_ranges(con: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    _ensure_dia_table("ranges")
    return _iter_dicts(con, "ranges", _DIA_RANGES_COLS)


# ──────────────────────────────────────────────────────────────────────
# Helpers used by the writer path
# ──────────────────────────────────────────────────────────────────────


def insert_many(
    con: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    rows: Iterable[tuple[Any, ...]],
) -> None:
    """Bulk INSERT via parameter binding — table name is whitelisted."""
    if table in LIBRARY_TABLE_NAMES:
        pass
    elif table in DIA_TABLE_NAMES:
        pass
    else:
        raise ValueError(f"unknown encyclopedia table {table!r}")
    placeholders = ", ".join("?" * len(columns))
    col_list = ", ".join(columns)
    con.executemany(
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", rows
    )


__all__ = [
    "open_ro",
    "open_rw",
    "has_table",
    "has_rows",
    "fetch_metadata",
    "iter_entries",
    "iter_peptidetoprotein",
    "iter_peptidescores",
    "iter_proteinscores",
    "iter_peptidequants",
    "iter_dia_precursor",
    "iter_dia_spectra",
    "iter_dia_ranges",
    "insert_many",
]
