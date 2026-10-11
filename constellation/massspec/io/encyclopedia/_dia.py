"""``.dia`` raw-spectra reader.

A ``.dia`` is the SQLite spectra cache EncyclopeDIA searches: MS1 scans
in ``precursor``, MS2 scans in ``spectra``, the isolation windows in
``ranges``, and a ``metadata`` key/value table the search jar reads to
decide how to treat the run. Each scan is one row carrying its own
zlib-compressed m/z and intensity arrays.

Two generations of the format exist (see ``_schema.py``): 0.5.0 from
EncyclopeDIA 2.12.30 and 0.8.0 from MSRawJava, the library inside
EncyclopeDIA 6.x and MSForest. This reader accepts both and presents one
shape, so callers do not branch on the version:

    ============================  ================================
    0.5.0 column                  presented as
    ============================  ================================
    ``IsolationWindowCenter``     ``isolation_window_target``
    (no ``spectra.TIC``)          ``tic`` null
    (no ``fractions`` table)      ``fractions`` empty
    (no RT bounds on ``ranges``)  ``rt_start`` / ``rt_stop`` null
    ============================  ================================

Two entry points:

- :class:`DiaFile` streams. Scans come back as Arrow record batches with
  ``list`` columns for the peak arrays, a few hundred scans at a time, so
  a multi-gigabyte file never has to fit in memory. Use this.
- :class:`DiaReader` is the ``core.io.RawReader`` registered for
  ``.dia``. It materialises the whole file and exists for small files
  and the reader registry.

Ion-mobility arrays (0.8.0, timsTOF) are not decoded: their encoding has
not been characterised against a real file, and a guess would be silently
wrong. A file that carries them raises ``NotImplementedError`` rather
than dropping the dimension.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pyarrow as pa

from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.massspec.io.encyclopedia._codec import decode_f4, decode_f8


# ──────────────────────────────────────────────────────────────────────
# Arrow schemas — one shape for both format generations.
# ──────────────────────────────────────────────────────────────────────


DIA_SPECTRA_SCHEMA = pa.schema(
    [
        pa.field("spectrum_index", pa.int64(), nullable=False),
        pa.field("spectrum_name", pa.string(), nullable=False),
        pa.field("precursor_name", pa.string(), nullable=True),
        pa.field("fraction", pa.int32(), nullable=False),
        pa.field("scan_start_time_s", pa.float64(), nullable=False),
        pa.field("ion_injection_time", pa.float64(), nullable=True),  # seconds
        pa.field("isolation_window_lower", pa.float64(), nullable=False),
        pa.field("isolation_window_target", pa.float64(), nullable=False),
        pa.field("isolation_window_upper", pa.float64(), nullable=False),
        pa.field("precursor_charge", pa.int32(), nullable=False),
        pa.field("tic", pa.float64(), nullable=True),
        pa.field("mz_list", pa.list_(pa.float64()), nullable=False),
        pa.field("intensity_list", pa.list_(pa.float32()), nullable=False),
    ],
    metadata={b"schema_name": b"DiaSpectraTable"},
)


DIA_PRECURSOR_SCHEMA = pa.schema(
    [
        pa.field("spectrum_index", pa.int64(), nullable=False),
        pa.field("spectrum_name", pa.string(), nullable=False),
        pa.field("fraction", pa.int32(), nullable=False),
        pa.field("scan_start_time_s", pa.float64(), nullable=False),
        pa.field("ion_injection_time", pa.float64(), nullable=True),  # seconds
        pa.field("isolation_window_lower", pa.float64(), nullable=False),
        pa.field("isolation_window_upper", pa.float64(), nullable=False),
        pa.field("tic", pa.float64(), nullable=True),
        pa.field("mz_list", pa.list_(pa.float64()), nullable=False),
        pa.field("intensity_list", pa.list_(pa.float32()), nullable=False),
    ],
    metadata={b"schema_name": b"DiaPrecursorTable"},
)


DIA_RANGES_SCHEMA = pa.schema(
    [
        pa.field("start", pa.float64(), nullable=False),
        pa.field("stop", pa.float64(), nullable=False),
        pa.field("duty_cycle", pa.float64(), nullable=False),
        pa.field("num_windows", pa.int32(), nullable=True),
        pa.field("ion_mobility_start", pa.float64(), nullable=True),
        pa.field("ion_mobility_stop", pa.float64(), nullable=True),
        pa.field("rt_start", pa.float64(), nullable=True),
        pa.field("rt_stop", pa.float64(), nullable=True),
    ],
    metadata={b"schema_name": b"DiaRangesTable"},
)

# Pre-0.8.0 private names, kept for importers of the old spelling.
_DIA_SPECTRA_SCHEMA = DIA_SPECTRA_SCHEMA
_DIA_PRECURSOR_SCHEMA = DIA_PRECURSOR_SCHEMA
_DIA_RANGES_SCHEMA = DIA_RANGES_SCHEMA

_SCAN_RE = re.compile(r"scan=(\d+)")


def scan_number(spectrum_name: str) -> int | None:
    """The instrument scan number in a ``SpectrumName``, if it has one.

    Both generations embed it as ``scan=N`` (0.8.0: ``scan=42``; 0.5.0:
    ``controllerType=0 controllerNumber=1 scan=42``). Names without it
    (a hand-built or scrubbed file) return ``None``.
    """
    m = _SCAN_RE.search(spectrum_name)
    return int(m.group(1)) if m else None


def _list_array(arrays: list[np.ndarray], value_type: pa.DataType) -> pa.ListArray:
    """Pack per-scan numpy arrays into one Arrow list column, no Python lists."""
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    if arrays:
        np.cumsum([a.size for a in arrays], out=offsets[1:])
        values = np.concatenate(arrays)
    else:
        values = np.empty(0)
    return pa.ListArray.from_arrays(
        pa.array(offsets, pa.int32()), pa.array(values, value_type)
    )


# ──────────────────────────────────────────────────────────────────────
# Streaming handle
# ──────────────────────────────────────────────────────────────────────


class DiaFile:
    """Read-only, streaming view of one ``.dia`` file.

    Use as a context manager. ``metadata`` keeps the values as SQLite
    returns them (the ``Value`` column has NUMERIC affinity, so counts
    come back as ``int`` and ``totalPrecursorTIC`` as ``float``).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no such .dia file: {self.path}")
        self._con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        tables = {
            r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = {"metadata", "precursor", "ranges", "spectra"} - tables
        if missing:
            self._con.close()
            raise ValueError(
                f"{self.path} is not a .dia file: missing table(s) {sorted(missing)}"
            )
        self._columns = {
            t: tuple(r[1] for r in self._con.execute(f"PRAGMA table_info({t})"))
            for t in ("precursor", "ranges", "spectra")
        }
        self.metadata: dict[str, Any] = dict(
            self._con.execute("SELECT Key, Value FROM metadata")
        )
        self.fractions: tuple[tuple[int, str], ...] = (
            tuple(
                (int(f), str(n))
                for f, n in self._con.execute(
                    "SELECT Fraction, Name FROM fractions ORDER BY Fraction"
                )
            )
            if "fractions" in tables
            else ()
        )

    def __enter__(self) -> "DiaFile":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._con.close()

    @property
    def format_version(self) -> str | None:
        """The file's ``version`` metadata value (``"0.5.0"``, ``"0.8.0"``)."""
        v = self.metadata.get("version")
        return None if v is None else str(v)

    def fraction_ids(self) -> tuple[int, ...]:
        """Distinct ``Fraction`` values present in the scan tables, ascending.

        A single-run file has one (0) and an empty ``fractions`` table; a
        merged file has one per source run.
        """
        ids = {
            int(r[0])
            for t in ("precursor", "spectra")
            for r in self._con.execute(f"SELECT DISTINCT Fraction FROM {t}")
        }
        return tuple(sorted(ids))

    def count(
        self, table: Literal["precursor", "spectra"], *, fraction: int | None = None
    ) -> int:
        where, params = self._where(fraction)
        return int(self._con.execute(f"SELECT count(*) FROM {table}{where}", params).fetchone()[0])

    def ranges(self) -> pa.Table:
        """The ``ranges`` table (small: one row per isolation window)."""
        have = self._columns["ranges"]
        wanted = (
            ("Start", "start"),
            ("Stop", "stop"),
            ("DutyCycle", "duty_cycle"),
            ("NumWindows", "num_windows"),
            ("IonMobilityStart", "ion_mobility_start"),
            ("IonMobilityStop", "ion_mobility_stop"),
            ("RtStart", "rt_start"),
            ("RtStop", "rt_stop"),
        )
        select = ", ".join(sql if sql in have else "NULL" for sql, _ in wanted)
        rows = self._con.execute(f"SELECT {select} FROM ranges").fetchall()
        cols = list(zip(*rows)) if rows else [()] * len(wanted)
        return pa.table(
            {
                name: pa.array(list(col), type=DIA_RANGES_SCHEMA.field(name).type)
                for (_, name), col in zip(wanted, cols)
            },
            schema=DIA_RANGES_SCHEMA,
        )

    def iter_precursor(
        self, *, fraction: int | None = None, batch_rows: int = 256
    ) -> Iterator[pa.RecordBatch]:
        """MS1 scans in ``SpectrumIndex`` order, as :data:`DIA_PRECURSOR_SCHEMA` batches."""
        return self._iter("precursor", fraction, batch_rows)

    def iter_spectra(
        self, *, fraction: int | None = None, batch_rows: int = 256
    ) -> Iterator[pa.RecordBatch]:
        """MS2 scans in ``SpectrumIndex`` order, as :data:`DIA_SPECTRA_SCHEMA` batches."""
        return self._iter("spectra", fraction, batch_rows)

    # ── internals ──────────────────────────────────────────────────────

    @staticmethod
    def _where(fraction: int | None) -> tuple[str, tuple[int, ...]]:
        return ("", ()) if fraction is None else (" WHERE Fraction = ?", (int(fraction),))

    def _iter(
        self, table: str, fraction: int | None, batch_rows: int
    ) -> Iterator[pa.RecordBatch]:
        have = self._columns[table]
        is_ms2 = table == "spectra"
        # (output name, SQL expression) — absent columns select NULL.
        target = "IsolationWindowTarget" if "IsolationWindowTarget" in have else "IsolationWindowCenter"
        select: list[tuple[str, str]] = [
            ("spectrum_index", "SpectrumIndex"),
            ("spectrum_name", "SpectrumName"),
            ("fraction", "Fraction"),
            ("scan_start_time_s", "ScanStartTime"),
            ("ion_injection_time", "IonInjectionTime"),
            ("isolation_window_lower", "IsolationWindowLower"),
            ("isolation_window_upper", "IsolationWindowUpper"),
            ("tic", "TIC" if "TIC" in have else "NULL"),
            ("_mz_len", "MassEncodedLength"),
            ("_mz", "MassArray"),
            ("_int_len", "IntensityEncodedLength"),
            ("_int", "IntensityArray"),
            ("_im", "IonMobilityArray" if "IonMobilityArray" in have else "NULL"),
        ]
        if is_ms2:
            select += [
                ("precursor_name", "PrecursorName"),
                ("isolation_window_target", target),
                ("precursor_charge", "PrecursorCharge"),
            ]
        names = [n for n, _ in select]
        where, params = self._where(fraction)
        cur = self._con.execute(
            f"SELECT {', '.join(e for _, e in select)} FROM {table}{where} ORDER BY SpectrumIndex",
            params,
        )
        schema = DIA_SPECTRA_SCHEMA if is_ms2 else DIA_PRECURSOR_SCHEMA
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                return
            col = dict(zip(names, zip(*rows)))
            if any(b is not None for b in col["_im"]):
                raise NotImplementedError(
                    f"{self.path} carries ion-mobility arrays in {table!r}; their "
                    f"encoding has not been characterised, so they are not decoded"
                )
            data: dict[str, pa.Array] = {
                "mz_list": _list_array(
                    [decode_f8(b, n) for b, n in zip(col["_mz"], col["_mz_len"])], pa.float64()
                ),
                "intensity_list": _list_array(
                    [decode_f4(b, n) for b, n in zip(col["_int"], col["_int_len"])], pa.float32()
                ),
            }
            for field in schema:
                if field.name in data:
                    continue
                values: Any = col[field.name]
                if field.name == "precursor_name":
                    # NUMERIC affinity: 0.8.0 files hold the integer 0 here.
                    values = [None if v is None else str(v) for v in values]
                data[field.name] = pa.array(list(values), type=field.type)
            yield pa.record_batch([data[f.name] for f in schema], schema=schema)


# ──────────────────────────────────────────────────────────────────────
# RawReader registration (whole file in memory)
# ──────────────────────────────────────────────────────────────────────


@register_reader
class DiaReader(RawReader):
    """EncyclopeDIA ``.dia`` raw spectra container, read whole.

    For anything larger than a test file prefer :class:`DiaFile`, which
    streams.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".dia",)
    modality: ClassVar[str] = "ms"

    def read(self, source: Path | Any) -> ReadResult:
        # Resolve Bundle → primary path if needed.
        path = Path(source.path) if hasattr(source, "path") else Path(source)
        with DiaFile(path) as dia:
            spectra = pa.Table.from_batches(list(dia.iter_spectra()), schema=DIA_SPECTRA_SCHEMA)
            precursor = pa.Table.from_batches(
                list(dia.iter_precursor()), schema=DIA_PRECURSOR_SCHEMA
            )
            ranges = dia.ranges()
            run_metadata: dict[str, Any] = {
                f"x.encyclopedia.{k}": v for k, v in dia.metadata.items()
            }
        return ReadResult(
            primary=spectra,
            companions={"precursor": precursor, "ranges": ranges},
            run_metadata=run_metadata,
        )


__all__ = [
    "DIA_PRECURSOR_SCHEMA",
    "DIA_RANGES_SCHEMA",
    "DIA_SPECTRA_SCHEMA",
    "DiaFile",
    "DiaReader",
    "scan_number",
]
