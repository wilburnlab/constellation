"""Tests for the .dia raw-spectra reader.

Uses a session-scoped subset of the full .dia (~120k MS2 + ~7k MS1
scans → 500 + 100). The full file would take 3+ minutes to decode.
Sample-file location is configurable via
``$CONSTELLATION_ENCYCLOPEDIA_SAMPLES`` so lab-specific filenames stay
out of the test source. The fixture also scrubs run-identifying
metadata + spectrum names before tests touch the file.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from constellation.core.io.readers import find_reader
from constellation.massspec.io.encyclopedia import DiaReader


SAMPLES = Path(
    os.environ.get(
        "CONSTELLATION_ENCYCLOPEDIA_SAMPLES",
        "~/WilburnLab/MS_Quant/test_encyclopedia_files",
    )
).expanduser()


def _first_dia() -> Path | None:
    if not SAMPLES.exists():
        return None
    matches = sorted(SAMPLES.glob("*.dia"))
    return matches[0] if matches else None


_SCRUBBED_RUN_NAME = "test-acquisition"


@pytest.fixture(scope="session")
def dia_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Subset the full .dia: 100 MS1 precursor scans + 500 MS2 spectra,
    metadata + ranges tables verbatim. Scrub run-identifying metadata.
    Decodes in seconds."""
    src = _first_dia()
    if src is None:
        pytest.skip(
            "no .dia sample present; "
            "set $CONSTELLATION_ENCYCLOPEDIA_SAMPLES to a directory containing one"
        )
    out = tmp_path_factory.mktemp("encyc-dia") / "subset.dia"
    shutil.copy(src, out)
    con = sqlite3.connect(out)
    try:
        con.executescript(f"""
DELETE FROM precursor WHERE rowid NOT IN
    (SELECT rowid FROM precursor ORDER BY rowid LIMIT 100);
DELETE FROM spectra WHERE rowid NOT IN
    (SELECT rowid FROM spectra ORDER BY rowid LIMIT 500);

UPDATE precursor SET SpectrumName = '{_SCRUBBED_RUN_NAME}.scan_' || SpectrumIndex;
UPDATE spectra
   SET SpectrumName = '{_SCRUBBED_RUN_NAME}.scan_' || SpectrumIndex,
       PrecursorName = '{_SCRUBBED_RUN_NAME}.precursor_' || SpectrumIndex;

DELETE FROM metadata;
INSERT INTO metadata (Key, Value) VALUES ('version', '0.5.0');
INSERT INTO metadata (Key, Value) VALUES ('sourcename', '{_SCRUBBED_RUN_NAME}');
INSERT INTO metadata (Key, Value) VALUES ('filename', '{_SCRUBBED_RUN_NAME}.mzML');

VACUUM;
""")
        con.commit()
    finally:
        con.close()
    return out


# ──────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────


def test_dia_reader_registered_for_ms_modality(tmp_path: Path):
    """find_reader('foo.dia', modality='ms') resolves to DiaReader."""
    reader = find_reader(tmp_path / "foo.dia", modality="ms")
    assert isinstance(reader, DiaReader)


# ──────────────────────────────────────────────────────────────────────
# End-to-end against the .dia subset
# ──────────────────────────────────────────────────────────────────────


def test_dia_subset_returns_spectra_precursor_and_ranges(dia_subset: Path):
    reader = DiaReader()
    result = reader.read(dia_subset)
    # We capped subset to 500 MS2 + 100 MS1; ranges is the verbatim
    # full table because it's small (~115k rows but quick to decode —
    # no blobs).
    assert result.primary.num_rows == 500
    assert result.companions["precursor"].num_rows == 100
    assert result.companions["ranges"].num_rows > 0
    # Schema columns
    assert "mz_list" in result.primary.schema.names
    assert "intensity_list" in result.primary.schema.names
    assert "isolation_window_lower" in result.primary.schema.names


def test_dia_subset_first_spectrum_decodes_cleanly(dia_subset: Path):
    reader = DiaReader()
    result = reader.read(dia_subset)
    first_mz = result.primary.column("mz_list")[0].as_py()
    first_int = result.primary.column("intensity_list")[0].as_py()
    # Same length on the two arrays — they're paired peak lists.
    assert len(first_mz) == len(first_int)
    # m/z values are physically plausible (Da, not garbage from byte-order)
    assert all(50.0 <= m <= 5000.0 for m in first_mz)


def test_dia_subset_run_metadata_scrubbed(dia_subset: Path):
    """The fixture scrubbed the metadata table; the reader should pick
    up the synthetic keys we wrote, not the original lab values."""
    reader = DiaReader()
    result = reader.read(dia_subset)
    md = result.run_metadata
    assert md["x.encyclopedia.version"] == "0.5.0"
    assert md["x.encyclopedia.sourcename"] == _SCRUBBED_RUN_NAME
