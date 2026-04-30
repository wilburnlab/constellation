"""Reader/Writer + Parquet round-trip tests for ``massspec.quant``."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.quant import (
    QUANT_READERS,
    QUANT_WRITERS,
    ParquetDirReader,
    ParquetDirWriter,
    Quant,
    QuantReader,
    QuantWriter,
    assemble_quant,
    load_quant,
    save_quant,
)


def _quant() -> Quant:
    acq = Acquisitions.from_records(
        [
            {
                "acquisition_id": 0,
                "source_file": "r0.raw",
                "source_kind": "thermo_raw",
                "acquisition_datetime": None,
            }
        ]
    )
    return assemble_quant(
        acquisitions=acq,
        protein_quant=[
            {"protein_id": 0, "acquisition_id": 0, "abundance": 1.0e6, "score": 0.9},
        ],
        transmission_protein_peptide=[
            {
                "protein_id": 0,
                "peptide_id": 0,
                "acquisition_id": None,
                "efficiency": 0.42,
            }
        ],
        metadata={"library_id": "lib-abc"},
    )


# ──────────────────────────────────────────────────────────────────────
# Protocol conformance + registry
# ──────────────────────────────────────────────────────────────────────


def test_writers_satisfy_protocol() -> None:
    for w in QUANT_WRITERS.values():
        assert isinstance(w, QuantWriter)


def test_readers_satisfy_protocol() -> None:
    for r in QUANT_READERS.values():
        assert isinstance(r, QuantReader)


def test_parquet_and_encyclopedia_registered() -> None:
    """encyclopedia.dlib + encyclopedia.elib adapters self-register on
    package import (via massspec/io/encyclopedia/adapters.py)."""
    assert "parquet_dir" in QUANT_WRITERS
    assert "parquet_dir" in QUANT_READERS
    assert "encyclopedia.dlib" in QUANT_WRITERS
    assert "encyclopedia.dlib" in QUANT_READERS
    assert "encyclopedia.elib" in QUANT_WRITERS
    assert "encyclopedia.elib" in QUANT_READERS


def test_parquet_writer_lossless() -> None:
    assert ParquetDirWriter().lossy is False


def test_encyclopedia_writers_marked_lossy() -> None:
    assert QUANT_WRITERS["encyclopedia.dlib"].lossy is True
    assert QUANT_WRITERS["encyclopedia.elib"].lossy is True


# ──────────────────────────────────────────────────────────────────────
# Parquet round-trip
# ──────────────────────────────────────────────────────────────────────


def test_parquet_round_trip(tmp_path: Path) -> None:
    q = _quant()
    out = tmp_path / "quant"
    save_quant(q, out, format="parquet_dir")
    assert (out / "manifest.json").exists()
    assert (out / "acquisitions.parquet").exists()

    q2 = load_quant(out, format="parquet_dir")
    assert q2.acquisitions.ids == q.acquisitions.ids
    assert q2.protein_quant.num_rows == 1
    assert q2.transmission_protein_peptide.num_rows == 1
    assert q2.metadata == q.metadata


def test_parquet_round_trip_preserves_run_agnostic_null(tmp_path: Path) -> None:
    """Transmission rows with null acquisition_id (run-agnostic
    calibration) must round-trip as null, not silently coerce to 0."""
    q = _quant()
    out = tmp_path / "quant"
    save_quant(q, out, format="parquet_dir")
    q2 = load_quant(out, format="parquet_dir")
    acqs = q2.transmission_protein_peptide.column("acquisition_id").to_pylist()
    assert acqs == [None]


# ──────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────


def test_elib_suffix_dispatch_resolves_to_encyclopedia(tmp_path: Path) -> None:
    from constellation.massspec.quant.io import _resolve_writer

    writer = _resolve_writer(None, tmp_path / "out.elib")
    assert writer.format_name == "encyclopedia.elib"


def test_dlib_suffix_dispatch_resolves_to_encyclopedia(tmp_path: Path) -> None:
    from constellation.massspec.quant.io import _resolve_writer

    writer = _resolve_writer(None, tmp_path / "out.dlib")
    assert writer.format_name == "encyclopedia.dlib"


def test_save_unknown_format_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no writer registered"):
        save_quant(_quant(), tmp_path / "x", format="not_a_format")
