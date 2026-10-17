"""``_mirror``: any table of an EncyclopeDIA library file as Arrow, nothing lost."""

from __future__ import annotations

import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from constellation.massspec.io.encyclopedia import _mirror
from constellation.massspec.io.encyclopedia._codec import encode_f4, encode_f8

from _encyclopedia_fixtures import Entry, build_result_library, result_library_ddl


@pytest.fixture
def library(tmp_path: Path) -> Path:
    entries = [
        Entry("PEPTIDEK", "PEPTIDEK", 2, 464.7, mz=[200.1, 300.2, 400.3], intensity=[1.0, 2.0, 3.0]),
        Entry("ELVISK", "ELVISK", 3, 230.5, mz=[150.0], intensity=[9.0], total_intensity=0.0),
    ]
    return build_result_library(tmp_path / "run.elib", entries, decoy_peptides=[("KEDITPEP", "D1")])


def test_every_table_and_every_row(library: Path) -> None:
    con = _mirror.open_library(library)
    names = _mirror.table_names(con)
    assert len(names) == 10 and {"entries", "decoyentries", "peptidequants", "metadata"} <= set(names)
    for name in names:
        table = _mirror.read_table(con, name)
        assert table.num_rows == _mirror.count_rows(con, name), name
        declared = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
        # Every declared column is there, in order, except the byte-length
        # siblings of decoded arrays.
        lengths = {length for _d, length in _mirror.ARRAY_COLUMNS.get(name, {}).values() if length}
        assert table.schema.names == [c for c in declared if c not in lengths], name


def test_arrays_are_decoded_and_typed(library: Path) -> None:
    con = _mirror.open_library(library)
    entries = _mirror.read_table(con, "entries")
    assert entries.schema.field("MassArray").type == pa.large_list(pa.float64())
    assert entries.schema.field("IntensityArray").type == pa.large_list(pa.float32())
    assert entries.schema.field("QuantifiedIonsArray").type == pa.large_list(pa.uint8())
    assert entries.column("MassArray").to_pylist() == [[200.1, 300.2, 400.3], [150.0]]
    assert entries.column("IntensityArray").to_pylist() == [[1.0, 2.0, 3.0], [9.0]]
    # Absent arrays are null, not empty.
    assert entries.column("CorrelationArray").null_count == 2
    assert entries.schema.field("PrecursorCharge").type == pa.int64()
    assert entries.schema.field("PeptideModSeq").type == pa.string()
    scores = _mirror.read_table(con, "peptidescores")
    assert scores.schema.field("IsDecoy").type == pa.bool_()
    assert scores.column("IsDecoy").to_pylist() == [False, False, True]


def test_undecoded_is_a_byte_faithful_copy(library: Path) -> None:
    con = _mirror.open_library(library)
    raw = _mirror.read_table(con, "entries", decode=False)
    assert raw.schema.field("MassArray").type == pa.binary()
    assert "MassEncodedLength" in raw.schema.names
    stored = con.execute("SELECT MassArray, MassEncodedLength FROM entries").fetchall()
    assert raw.column("MassArray").to_pylist() == [r[0] for r in stored]
    assert raw.column("MassEncodedLength").to_pylist() == [r[1] for r in stored]


def test_batches_cover_the_table_in_order(tmp_path: Path) -> None:
    entries = [
        Entry(f"PEPTIDE{'A' * i}K", f"PEPTIDE{'A' * i}K", 2, 400.0 + i, mz=[100.0 + i], intensity=[1.0])
        for i in range(25)
    ]
    con = _mirror.open_library(build_result_library(tmp_path / "many.elib", entries))
    batches = list(_mirror.iter_table(con, "entries", columns=["PrecursorMz", "MassArray"], batch_rows=10))
    assert [b.num_rows for b in batches] == [10, 10, 5]
    assert pa.Table.from_batches(batches).column("PrecursorMz").to_pylist() == [400.0 + i for i in range(25)]
    assert batches[0].schema.names == ["PrecursorMz", "MassArray"]


def test_a_column_added_by_a_newer_version_is_carried(tmp_path: Path) -> None:
    """The reader this replaces listed its columns by hand and did not
    see what container 0.1.18 added."""
    path = tmp_path / "future.elib"
    con = sqlite3.connect(path)
    con.executescript(result_library_ddl())
    con.execute("ALTER TABLE proteinscores ADD COLUMN AddedNextYear double")
    con.execute("INSERT INTO proteinscores VALUES (1, 'P1', 'run.raw', 0.01, 0.02, 0, 3.5)")
    con.commit()
    con.close()
    table = _mirror.read_table(_mirror.open_library(path), "proteinscores")
    assert table.column("AddedNextYear").to_pylist() == [3.5]


def test_text_columns_with_numeric_looking_values_are_text(tmp_path: Path) -> None:
    """EncyclopeDIA declares text as ``string``, which SQLite gives
    numeric affinity: ``0.35`` in ``metadata.Value`` is stored as a
    number."""
    path = build_result_library(
        tmp_path / "run.elib", [], metadata={"-ptol": "0.35", "-numberOfThreadsUsed": "12"}
    )
    con = _mirror.open_library(path)
    assert isinstance(con.execute("SELECT Value FROM metadata WHERE Key='-ptol'").fetchone()[0], float)
    meta = _mirror.read_metadata(con)
    assert meta["-ptol"] == "0.35" and meta["-numberOfThreadsUsed"] == "12"
    assert _mirror.read_table(con, "metadata").schema.field("Value").type == pa.string()


def test_a_file_that_contradicts_itself_is_refused(tmp_path: Path) -> None:
    path = build_result_library(
        tmp_path / "run.elib", [Entry("PEPTIDEK", "PEPTIDEK", 2, 464.7, mz=[200.1, 300.2], intensity=[1.0, 2.0])]
    )
    con = sqlite3.connect(path)
    con.execute("UPDATE entries SET MassEncodedLength = 24")  # the array holds 16 bytes
    con.commit()
    con.close()
    with pytest.raises(_mirror.MirrorError, match="decodes to 16 bytes but the file records 24"):
        _mirror.read_table(_mirror.open_library(path), "entries")

    con = sqlite3.connect(path)
    con.execute("UPDATE entries SET MassEncodedLength = 16, IntensityArray = ?", (zlib.compress(b"abc"),))
    con.commit()
    con.close()
    with pytest.raises(_mirror.MirrorError, match="do not divide into 4-byte values"):
        _mirror.read_table(_mirror.open_library(path), "entries")


def test_unknown_tables_columns_and_types_are_named(library: Path, tmp_path: Path) -> None:
    con = _mirror.open_library(library)
    with pytest.raises(_mirror.MirrorError, match="no table named 'nonsense'"):
        _mirror.read_table(con, "nonsense")
    with pytest.raises(_mirror.MirrorError, match=r"entries has no column\(s\) \['Nope'\]"):
        _mirror.read_table(con, "entries", columns=["Nope"])
    with pytest.raises(FileNotFoundError):
        _mirror.open_library(tmp_path / "absent.elib")

    odd = tmp_path / "odd.db"
    other = sqlite3.connect(odd)
    other.execute("CREATE TABLE things (When_ timestamp)")
    other.commit()
    other.close()
    with pytest.raises(_mirror.MirrorError, match="declared 'timestamp'"):
        _mirror.read_table(_mirror.open_library(odd), "things")


def test_array_codec_round_trip_at_scale() -> None:
    """The list assembly (offsets over many rows, nulls in between)."""
    rng = np.random.default_rng(0)
    arrays = [rng.random(int(n)) for n in rng.integers(0, 40, size=500)]
    blobs: list[bytes | None] = [encode_f8(a)[0] for a in arrays]
    lengths: list[int | None] = [8 * len(a) for a in arrays]
    blobs[7] = None
    lengths[7] = None
    out = _mirror._decode_arrays(blobs, ">f8", lengths, "t.c")
    assert out.null_count == 1 and out[7].as_py() is None
    for i in (0, 6, 8, 499):
        assert np.array_equal(np.array(out[i].as_py()), arrays[i])
    assert pc.sum(pc.list_value_length(out)).as_py() == sum(len(a) for i, a in enumerate(arrays) if i != 7)
    floats = _mirror._decode_arrays([encode_f4(np.float32([1.5, 2.5]))[0]], ">f4", [8], "t.c")
    assert floats.type == pa.large_list(pa.float32()) and floats.to_pylist() == [[1.5, 2.5]]
