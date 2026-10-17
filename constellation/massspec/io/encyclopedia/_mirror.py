"""Read any table of an EncyclopeDIA library file as Arrow, losing nothing.

A ``.dlib`` / ``.elib`` is a SQLite file. The reader this replaces walked
a hand-picked list of columns row by row in Python and built every
output row as a dict, which made it slow, memory-bound, and blind to
whatever it had not been told about (when the container grew from
0.1.15 to 0.1.18 it simply did not see the new columns).

This module is the layer underneath a reader. It knows nothing about
peptides or libraries. It turns a table into a stream of Arrow record
batches:

- **every column the file has**, discovered from the file itself, so a
  column a newer EncyclopeDIA adds is carried without a code change;
- **typed from the declared SQLite type** (``double``, ``int``,
  ``string``, ``boolean``, ``blob``);
- **array columns decoded**. EncyclopeDIA stores arrays as
  zlib-compressed big-endian numbers in ``blob`` columns, usually with a
  sibling column giving the uncompressed byte length. ``ARRAY_COLUMNS``
  says which blobs are arrays of what; they come out as Arrow lists, the
  length is checked against the sibling, and the sibling (a detail of
  the serialisation) is dropped. A blob that is not listed stays a blob,
  untouched;
- **in batches**, so a file with millions of entries is never held in
  memory at once.

What each array holds was read off real 6.9.23 files (a peptide-centric
search result and a ScribeTwo DDA result) by decoding them every
plausible way and checking the lengths against the sibling columns and
against each other:

==================  ==========================  =====  =================================
table               column                      type   length column
==================  ==========================  =====  =================================
entries,            MassArray                   f8     MassEncodedLength
decoyentries        IntensityArray              f4     IntensityEncodedLength
                    CorrelationArray            f4     CorrelationEncodedLength
                    QuantifiedIonsArray         u1     (none; one byte per stored peak)
                    MedianChromatogramArray     f4     MedianChromatogramEncodedLength
peptidequants       QuantIonMassArray           f8     QuantIonMassLength
                    QuantIonIntensityArray      f4     QuantIonIntensityLength
                    QuantIonCorrelationArray    f4     QuantIonCorrelationLength
                    MedianChromatogramArray     f4     MedianChromatogramEncodedLength
                    MedianChromatogramRTArray   f4     MedianChromatogramRTEncodedLength
==================  ==========================  =====  =================================

Three facts about the values, from the same files, that a reader needs:
``peptidequants.TotalIntensity`` is the sum of that row's
``QuantIonIntensityArray`` and ``NumberOfQuantIons`` its length (zero
for a precursor that was identified but could not be quantified);
``entries.Score`` is the posterior error probability, the same number
as ``peptidescores.PosteriorErrorProbability``; and ``entries`` and
``peptidequants`` carry the same chromatogram.
"""

from __future__ import annotations

import sqlite3
import zlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa

#: ``table -> {blob column: (stored dtype, length column or None)}``.
_ENTRY_ARRAYS: dict[str, tuple[str, str | None]] = {
    "MassArray": (">f8", "MassEncodedLength"),
    "IntensityArray": (">f4", "IntensityEncodedLength"),
    "CorrelationArray": (">f4", "CorrelationEncodedLength"),
    "QuantifiedIonsArray": ("u1", None),
    "MedianChromatogramArray": (">f4", "MedianChromatogramEncodedLength"),
}
ARRAY_COLUMNS: Mapping[str, Mapping[str, tuple[str, str | None]]] = {
    "entries": _ENTRY_ARRAYS,
    "decoyentries": _ENTRY_ARRAYS,
    "peptidequants": {
        "QuantIonMassArray": (">f8", "QuantIonMassLength"),
        "QuantIonIntensityArray": (">f4", "QuantIonIntensityLength"),
        "QuantIonCorrelationArray": (">f4", "QuantIonCorrelationLength"),
        "MedianChromatogramArray": (">f4", "MedianChromatogramEncodedLength"),
        "MedianChromatogramRTArray": (">f4", "MedianChromatogramRTEncodedLength"),
    },
}

_SCALAR_TYPES: dict[str, pa.DataType] = {
    "double": pa.float64(),
    "real": pa.float64(),
    "float": pa.float64(),
    "int": pa.int64(),
    "integer": pa.int64(),
    "long": pa.int64(),
    "bigint": pa.int64(),
    "string": pa.string(),
    "text": pa.string(),
    "varchar": pa.string(),
    "boolean": pa.bool_(),
    "bool": pa.bool_(),
    "blob": pa.binary(),
}


class MirrorError(ValueError):
    """The file does not hold what its own declarations say it holds."""


def open_library(path: Path | str) -> sqlite3.Connection:
    """Open a library file read-only.

    ``immutable=1``: the file is a finished artefact, and without it
    SQLite wants to create journal files beside it, which fails on a
    read-only mount and litters a shared one.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such library file: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def table_names(con: sqlite3.Connection) -> list[str]:
    return [
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]


def _declared(con: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """``(column, declared type lower-cased)`` in the table's own order."""
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not rows:
        raise MirrorError(f"no table named {table!r} (the file has: {', '.join(table_names(con))})")
    return [(r[1], (r[2] or "").strip().lower()) for r in rows]


def _plan(
    con: sqlite3.Connection, table: str, columns: Sequence[str] | None, decode: bool
) -> tuple[list[str], list[pa.Field], dict[str, tuple[str, str | None]]]:
    """What to SELECT, the Arrow fields that come out, and the arrays to decode."""
    declared = _declared(con, table)
    arrays = dict(ARRAY_COLUMNS.get(table, {})) if decode else {}
    arrays = {c: spec for c, spec in arrays.items() if c in {name for name, _ in declared}}
    length_columns = {length for _dtype, length in arrays.values() if length}

    known = {name for name, _ in declared}
    if columns is not None:
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise MirrorError(f"{table} has no column(s) {unknown}; it has {sorted(known)}")
    wanted = [name for name, _ in declared if columns is None or name in set(columns)]
    if columns is None:
        wanted = [c for c in wanted if c not in length_columns]

    select: list[str] = list(wanted)
    fields: list[pa.Field] = []
    types = dict(declared)
    for name in wanted:
        if name in arrays:
            dtype = np.dtype(arrays[name][0]).newbyteorder("=")
            fields.append(pa.field(name, pa.large_list(pa.from_numpy_dtype(dtype))))
            length = arrays[name][1]
            if length and length not in select:
                select.append(length)  # fetched to check the decode, not emitted
        else:
            arrow_type = _SCALAR_TYPES.get(types[name].split("(")[0])
            if arrow_type is None:
                raise MirrorError(
                    f"{table}.{name} is declared {types[name]!r}, a type this reader has no "
                    f"mapping for"
                )
            fields.append(pa.field(name, arrow_type))
    return select, fields, {c: arrays[c] for c in wanted if c in arrays}


def table_schema(
    con: sqlite3.Connection,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    decode: bool = True,
) -> pa.Schema:
    """The Arrow schema :func:`iter_table` yields for ``table``."""
    _select, fields, _arrays = _plan(con, table, columns, decode)
    return pa.schema(fields)


def count_rows(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _as_bool(values: Sequence[object]) -> list[bool | None]:
    """SQLite has no boolean; EncyclopeDIA writes 0/1, and JDBC drivers
    have been known to write the words."""
    out: list[bool | None] = []
    for v in values:
        if v is None:
            out.append(None)
        elif isinstance(v, str):
            out.append(v.strip().lower() in ("1", "true", "t", "yes"))
        else:
            out.append(bool(v))
    return out


def _as_text(values: Sequence[object]) -> list[str | None] | Sequence[object]:
    """EncyclopeDIA declares text columns ``string``, which SQLite does
    not know and gives numeric affinity: a value that looks like a
    number (a tolerance of ``0.35`` in ``metadata``, a run named
    ``20261006``) is stored as one and comes back as an int or float.
    The declaration is what was meant, so such values are read as text."""
    if all(v is None or isinstance(v, str) for v in values):
        return values
    return [v if v is None or isinstance(v, str) else str(v) for v in values]


def _decode_arrays(
    blobs: Sequence[bytes | None],
    stored_dtype: str,
    lengths: Sequence[int | None] | None,
    where: str,
) -> pa.Array:
    """A column of compressed arrays as one Arrow list column."""
    dtype = np.dtype(stored_dtype)
    parts: list[np.ndarray] = []
    sizes = np.zeros(len(blobs), dtype=np.int64)
    nulls = np.zeros(len(blobs), dtype=bool)
    for i, blob in enumerate(blobs):
        if blob is None:
            nulls[i] = True
            continue
        raw = zlib.decompress(blob)
        if len(raw) % dtype.itemsize:
            raise MirrorError(
                f"{where}, row {i} of its batch: {len(raw)} bytes do not divide into "
                f"{dtype.itemsize}-byte values"
            )
        if lengths is not None and lengths[i] is not None and lengths[i] != len(raw):
            raise MirrorError(
                f"{where}, row {i} of its batch: decodes to {len(raw)} bytes but the file "
                f"records {lengths[i]}"
            )
        parts.append(np.frombuffer(raw, dtype=dtype))
        sizes[i] = len(raw) // dtype.itemsize
    native = dtype.newbyteorder("=")
    values = np.concatenate(parts).astype(native) if parts else np.zeros(0, dtype=native)
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return pa.LargeListArray.from_arrays(
        pa.array(offsets, type=pa.int64()),
        pa.array(values),
        mask=pa.array(nulls) if nulls.any() else None,
    )


def iter_table(
    con: sqlite3.Connection,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    decode: bool = True,
    batch_rows: int = 8192,
) -> Iterator[pa.RecordBatch]:
    """Stream ``table`` as record batches, in the file's row order.

    ``columns`` restricts and orders nothing: the result always follows
    the table's own column order, limited to the names given. With
    ``decode=False`` array blobs are left as they are stored and their
    length columns are kept, which is a byte-faithful copy.
    """
    select, fields, arrays = _plan(con, table, columns, decode)
    schema = pa.schema(fields)
    position = {name: i for i, name in enumerate(select)}
    cursor = con.execute(f"SELECT {', '.join(f'"{c}"' for c in select)} FROM \"{table}\"")
    while True:
        rows = cursor.fetchmany(batch_rows)
        if not rows:
            return
        by_column = list(zip(*rows, strict=True))
        out: list[pa.Array] = []
        for f in fields:
            values = by_column[position[f.name]]
            if f.name in arrays:
                stored_dtype, length_column = arrays[f.name]
                lengths = by_column[position[length_column]] if length_column else None
                out.append(_decode_arrays(values, stored_dtype, lengths, f"{table}.{f.name}"))
            elif pa.types.is_boolean(f.type):
                out.append(pa.array(_as_bool(values), type=f.type))
            elif pa.types.is_string(f.type):
                out.append(pa.array(_as_text(values), type=f.type))
            else:
                try:
                    out.append(pa.array(values, type=f.type))
                except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                    raise MirrorError(
                        f"{table}.{f.name} is declared {f.type} but holds a value that is "
                        f"not one ({exc})"
                    ) from None
        yield pa.RecordBatch.from_arrays(out, schema=schema)


def read_table(
    con: sqlite3.Connection,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    decode: bool = True,
    batch_rows: int = 8192,
) -> pa.Table:
    """The whole of ``table`` in memory. For files of ordinary size; a
    proteome-scale library's ``entries`` should be streamed."""
    schema = table_schema(con, table, columns=columns, decode=decode)
    batches = list(iter_table(con, table, columns=columns, decode=decode, batch_rows=batch_rows))
    return pa.Table.from_batches(batches, schema=schema)


def read_metadata(con: sqlite3.Connection) -> dict[str, str]:
    """The ``metadata`` table as a dict (empty when the file has none)."""
    if "metadata" not in table_names(con):
        return {}
    return {str(k): str(v) for k, v in con.execute("SELECT Key, Value FROM metadata")}


__all__ = [
    "ARRAY_COLUMNS",
    "MirrorError",
    "count_rows",
    "iter_table",
    "open_library",
    "read_metadata",
    "read_table",
    "table_names",
    "table_schema",
]
