"""Reader/Writer + Parquet round-trip tests for ``massspec.library``."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.library import (
    LIBRARY_READERS,
    LIBRARY_WRITERS,
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


def test_parquet_and_encyclopedia_registered() -> None:
    """encyclopedia.dlib + encyclopedia.elib adapters self-register on
    package import (via massspec/io/encyclopedia/adapters.py)."""
    assert "parquet_dir" in LIBRARY_WRITERS
    assert "parquet_dir" in LIBRARY_READERS
    assert "encyclopedia.dlib" in LIBRARY_WRITERS
    assert "encyclopedia.dlib" in LIBRARY_READERS
    assert "encyclopedia.elib" in LIBRARY_WRITERS
    assert "encyclopedia.elib" in LIBRARY_READERS


def test_parquet_dir_writer_metadata() -> None:
    w = ParquetDirWriter()
    assert w.format_name == "parquet_dir"
    assert w.lossy is False


def test_encyclopedia_writers_marked_lossy() -> None:
    """Terminal mods collapse onto residue 0 in EncyclopeDIA's grammar."""
    assert LIBRARY_WRITERS["encyclopedia.dlib"].lossy is True
    assert LIBRARY_WRITERS["encyclopedia.elib"].lossy is True


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
# Dispatch
# ──────────────────────────────────────────────────────────────────────


def test_save_unknown_format_raises(tmp_path: Path) -> None:
    lib = _library()
    with pytest.raises(KeyError, match="no writer registered"):
        save_library(lib, tmp_path / "lib", format="not_a_format")


def test_dlib_suffix_dispatch_resolves_to_encyclopedia(tmp_path: Path) -> None:
    """A path ending in .dlib must dispatch to the encyclopedia adapter
    when no explicit format= is given. Round-trip is exercised in
    tests/test_encyclopedia_roundtrip.py."""
    from constellation.massspec.library.io import _resolve_writer

    writer = _resolve_writer(None, tmp_path / "out.dlib")
    assert writer.format_name == "encyclopedia.dlib"


def test_elib_suffix_dispatch_resolves_to_encyclopedia(tmp_path: Path) -> None:
    from constellation.massspec.library.io import _resolve_writer

    writer = _resolve_writer(None, tmp_path / "out.elib")
    assert writer.format_name == "encyclopedia.elib"
