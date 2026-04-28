"""Reader/Writer + Parquet round-trip tests for ``massspec.library``."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.library import (
    LIBRARY_READERS,
    LIBRARY_WRITERS,
    DlibReader,
    DlibWriter,
    Library,
    LibraryReader,
    LibraryWriter,
    ParquetDirReader,
    ParquetDirWriter,
    assign_ids,
    load_library,
    save_library,
)
from constellation.massspec.peptide.ions import IonType


def _library() -> Library:
    return assign_ids(
        proteins=[{"accession": "P00001", "sequence": "MKAA"}],
        peptides=[{"modified_sequence": "PEPTIDE"}],
        precursors=[
            {
                "modified_sequence": "PEPTIDE",
                "charge": 2,
                "precursor_mz": 400.2,
                "rt_predicted": 12.3,
            }
        ],
        fragments=[
            {
                "modified_sequence": "PEPTIDE",
                "precursor_charge": 2,
                "ion_type": int(IonType.Y),
                "position": 5,
                "charge": 1,
                "mz_theoretical": 600.3,
                "intensity_predicted": 1.0,
                "annotation": "y5+",
            }
        ],
        protein_peptide=[("P00001", "PEPTIDE")],
        metadata={"x.library.predictor": "Prosit-2024"},
    )


# ──────────────────────────────────────────────────────────────────────
# Protocol conformance + registry
# ──────────────────────────────────────────────────────────────────────


def test_writers_satisfy_protocol() -> None:
    for writer in LIBRARY_WRITERS.values():
        assert isinstance(writer, LibraryWriter)


def test_readers_satisfy_protocol() -> None:
    for reader in LIBRARY_READERS.values():
        assert isinstance(reader, LibraryReader)


def test_parquet_and_dlib_registered() -> None:
    assert "parquet_dir" in LIBRARY_WRITERS
    assert "parquet_dir" in LIBRARY_READERS
    assert "dlib" in LIBRARY_WRITERS
    assert "dlib" in LIBRARY_READERS


def test_parquet_dir_writer_metadata() -> None:
    w = ParquetDirWriter()
    assert w.format_name == "parquet_dir"
    assert w.lossy is False


def test_dlib_writer_marked_lossy() -> None:
    w = DlibWriter()
    assert w.lossy is True


# ──────────────────────────────────────────────────────────────────────
# Round-trip
# ──────────────────────────────────────────────────────────────────────


def test_parquet_dir_round_trip(tmp_path: Path) -> None:
    lib = _library()
    out = tmp_path / "lib"
    save_library(lib, out, format="parquet_dir")
    assert (out / "manifest.json").exists()
    assert (out / "proteins.parquet").exists()

    lib2 = load_library(out, format="parquet_dir")
    assert lib2.n_proteins == lib.n_proteins
    assert lib2.n_peptides == lib.n_peptides
    assert lib2.n_precursors == lib.n_precursors
    assert lib2.n_fragments == lib.n_fragments
    assert lib2.metadata == lib.metadata


def test_parquet_dir_round_trip_preserves_predicted_sentinels(
    tmp_path: Path,
) -> None:
    """Precursors with no rt/ccs predictions carry -1.0 sentinels —
    confirm those survive the float64 Parquet round-trip exactly."""
    lib = assign_ids(
        proteins=[{"accession": "X", "sequence": "M"}],
        peptides=[{"modified_sequence": "PEP"}],
        precursors=[
            {"modified_sequence": "PEP", "charge": 1, "precursor_mz": 300.1}
        ],
    )
    out = tmp_path / "lib"
    save_library(lib, out, format="parquet_dir")
    lib2 = load_library(out, format="parquet_dir")
    assert lib2.precursors.column("rt_predicted").to_pylist() == [-1.0]
    assert lib2.precursors.column("ccs_predicted").to_pylist() == [-1.0]


# ──────────────────────────────────────────────────────────────────────
# Dispatch + dlib stub
# ──────────────────────────────────────────────────────────────────────


def test_save_unknown_format_raises(tmp_path: Path) -> None:
    lib = _library()
    with pytest.raises(KeyError, match="no writer registered"):
        save_library(lib, tmp_path / "lib", format="not_a_format")


def test_dlib_writer_raises_with_pointer(tmp_path: Path) -> None:
    lib = _library()
    with pytest.raises(NotImplementedError, match="EncyclopeDIA"):
        save_library(lib, tmp_path / "out.dlib", format="dlib")


def test_dlib_reader_raises_with_pointer(tmp_path: Path) -> None:
    p = tmp_path / "out.dlib"
    p.touch()
    with pytest.raises(NotImplementedError, match="EncyclopeDIA"):
        load_library(p, format="dlib")


def test_format_dispatch_via_extension(tmp_path: Path) -> None:
    """A path ending in .dlib should dispatch to the dlib stub
    when no explicit format= is given."""
    lib = _library()
    with pytest.raises(NotImplementedError):
        save_library(lib, tmp_path / "out.dlib")
