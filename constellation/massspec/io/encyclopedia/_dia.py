"""``.dia`` raw-spectra reader — subclass of ``core.io.RawReader``.

The ``.dia`` file is the raw-spectra container EncyclopeDIA materialises
from an mzML on first load. Schema is distinct from dlib/elib: 4 tables
(``metadata``, ``precursor``, ``ranges``, ``spectra``) holding MS1
precursor scans, MS2 DIA-window spectra, and the DIA window definitions.

Each scan is one row carrying its own (mz, intensity) blob — peak m/z
domains aren't shared across rows, so the natural shape is row-per-scan
with ``list<float>`` columns, not a matrix. We don't add a new core
schema for this in v1; a proper ``MS_SPECTRA_TABLE`` core schema can
land when an mzML reader needs to share the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from constellation.core.io.readers import RawReader, ReadResult, register_reader
from constellation.massspec.io.encyclopedia import _sql
from constellation.massspec.io.encyclopedia._codec import (
    decompress_intensity,
    decompress_mz,
)


# ──────────────────────────────────────────────────────────────────────
# Arrow schemas — local to .dia for now, not promoted to core.io.
# ──────────────────────────────────────────────────────────────────────


_DIA_SPECTRA_SCHEMA = pa.schema(
    [
        pa.field("spectrum_index", pa.int64(), nullable=False),
        pa.field("spectrum_name", pa.string(), nullable=False),
        pa.field("precursor_name", pa.string(), nullable=True),
        pa.field("fraction", pa.int32(), nullable=False),
        pa.field("scan_start_time_s", pa.float64(), nullable=False),
        pa.field("ion_injection_time", pa.float64(), nullable=True),
        pa.field("isolation_window_lower", pa.float64(), nullable=False),
        pa.field("isolation_window_center", pa.float64(), nullable=False),
        pa.field("isolation_window_upper", pa.float64(), nullable=False),
        pa.field("precursor_charge", pa.int32(), nullable=False),
        pa.field("mz_list", pa.list_(pa.float64()), nullable=False),
        pa.field("intensity_list", pa.list_(pa.float32()), nullable=False),
    ],
    metadata={b"schema_name": b"DiaSpectraTable"},
)


_DIA_PRECURSOR_SCHEMA = pa.schema(
    [
        pa.field("spectrum_index", pa.int64(), nullable=False),
        pa.field("spectrum_name", pa.string(), nullable=False),
        pa.field("fraction", pa.int32(), nullable=False),
        pa.field("scan_start_time_s", pa.float64(), nullable=False),
        pa.field("ion_injection_time", pa.float64(), nullable=True),
        pa.field("isolation_window_lower", pa.float64(), nullable=False),
        pa.field("isolation_window_upper", pa.float64(), nullable=False),
        pa.field("tic", pa.float64(), nullable=True),
        pa.field("mz_list", pa.list_(pa.float64()), nullable=False),
        pa.field("intensity_list", pa.list_(pa.float32()), nullable=False),
    ],
    metadata={b"schema_name": b"DiaPrecursorTable"},
)


_DIA_RANGES_SCHEMA = pa.schema(
    [
        pa.field("start", pa.float64(), nullable=False),
        pa.field("stop", pa.float64(), nullable=False),
        pa.field("duty_cycle", pa.float64(), nullable=False),
        pa.field("num_windows", pa.int32(), nullable=True),
    ],
    metadata={b"schema_name": b"DiaRangesTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────


@register_reader
class DiaReader(RawReader):
    """EncyclopeDIA ``.dia`` raw spectra container."""

    suffixes: ClassVar[tuple[str, ...]] = (".dia",)
    modality: ClassVar[str] = "ms"

    def read(self, source: Path | Any) -> ReadResult:
        # Resolve Bundle → primary path if needed.
        if hasattr(source, "path"):
            path = Path(source.path)
        else:
            path = Path(source)
        with _sql.open_ro(path) as con:
            spectra = _read_spectra(con)
            precursor = _read_precursor(con)
            ranges = _read_ranges(con)
            metadata = _sql.fetch_metadata(con)

        run_metadata: dict[str, Any] = {
            f"x.encyclopedia.{k}": v for k, v in metadata.items()
        }

        return ReadResult(
            primary=spectra,
            companions={"precursor": precursor, "ranges": ranges},
            run_metadata=run_metadata,
        )


def _read_spectra(con) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for r in _sql.iter_dia_spectra(con):
        mz = decompress_mz(r["MassArray"], int(r["MassEncodedLength"]))
        intensity = decompress_intensity(
            r["IntensityArray"], int(r["IntensityEncodedLength"])
        )
        rows.append(
            {
                "spectrum_index": int(r["SpectrumIndex"]),
                "spectrum_name": r["SpectrumName"],
                "precursor_name": r["PrecursorName"],
                "fraction": int(r["Fraction"]),
                "scan_start_time_s": float(r["ScanStartTime"]),
                "ion_injection_time": (
                    float(r["IonInjectionTime"])
                    if r["IonInjectionTime"] is not None
                    else None
                ),
                "isolation_window_lower": float(r["IsolationWindowLower"]),
                "isolation_window_center": float(r["IsolationWindowCenter"]),
                "isolation_window_upper": float(r["IsolationWindowUpper"]),
                "precursor_charge": int(r["PrecursorCharge"]),
                "mz_list": mz.tolist(),
                "intensity_list": intensity.tolist(),
            }
        )
    if not rows:
        return _DIA_SPECTRA_SCHEMA.empty_table()
    return pa.Table.from_pylist(rows, schema=_DIA_SPECTRA_SCHEMA)


def _read_precursor(con) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for r in _sql.iter_dia_precursor(con):
        mz = decompress_mz(r["MassArray"], int(r["MassEncodedLength"]))
        intensity = decompress_intensity(
            r["IntensityArray"], int(r["IntensityEncodedLength"])
        )
        rows.append(
            {
                "spectrum_index": int(r["SpectrumIndex"]),
                "spectrum_name": r["SpectrumName"],
                "fraction": int(r["Fraction"]),
                "scan_start_time_s": float(r["ScanStartTime"]),
                "ion_injection_time": (
                    float(r["IonInjectionTime"])
                    if r["IonInjectionTime"] is not None
                    else None
                ),
                "isolation_window_lower": float(r["IsolationWindowLower"]),
                "isolation_window_upper": float(r["IsolationWindowUpper"]),
                "tic": float(r["TIC"]) if r["TIC"] is not None else None,
                "mz_list": mz.tolist(),
                "intensity_list": intensity.tolist(),
            }
        )
    if not rows:
        return _DIA_PRECURSOR_SCHEMA.empty_table()
    return pa.Table.from_pylist(rows, schema=_DIA_PRECURSOR_SCHEMA)


def _read_ranges(con) -> pa.Table:
    rows = list(_sql.iter_dia_ranges(con))
    if not rows:
        return _DIA_RANGES_SCHEMA.empty_table()
    return pa.Table.from_pylist(
        [
            {
                "start": float(r["Start"]),
                "stop": float(r["Stop"]),
                "duty_cycle": float(r["DutyCycle"]),
                "num_windows": (
                    int(r["NumWindows"]) if r["NumWindows"] is not None else None
                ),
            }
            for r in rows
        ],
        schema=_DIA_RANGES_SCHEMA,
    )


__all__ = ["DiaReader"]
