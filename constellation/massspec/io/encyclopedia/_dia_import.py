"""Turn an EncyclopeDIA ``.dia`` into Constellation peak bundles.

The reverse of :mod:`._dia_write`. A ``.dia`` from anywhere (MSRawJava,
MSForest, EncyclopeDIA's own converter, another lab) becomes the same
on-disk shape ``massspec convert`` produces from a ``.raw``::

    <out>/<name>/peaks.parquet                 MS_PEAK_TABLE, one row per peak
    <out>/<name>/scan_metadata.parquet         SCAN_METADATA_TABLE
    <out>/<name>/acquisition_metadata.parquet  ACQUISITION_METADATA_TABLE
    <out>/<name>/manifest.json

so XIC extraction, the peak index and everything downstream read it
without knowing where it came from.

A merged ``.dia`` holds several runs as fractions; each fraction becomes
its own bundle, named after the fraction's source file, because a bundle
is one acquisition and scan numbers repeat across fractions.

What a ``.dia`` does not carry, and so is null in the bundle: filter
strings, collision energy, analyzer and activation per scan, the
precursor scan link, and every Thermo trailer column. What it carries
that a Thermo bundle states differently:

- MS1 "isolation bounds" are the scan range → ``scan_window_lower`` /
  ``scan_window_upper`` (a bundle's ``isolation_*`` stay null for MS1).
- ``IsolationWindowTarget`` → ``precursor_mz``.
- ``PrecursorName`` (the parent MS1 scan) → ``precursor_scan``.
- ``PrecursorCharge`` 0 means "not reported" → null.
- Injection time is seconds → milliseconds.
- ``TIC`` is the sum of the stored intensities, not the instrument's, and
  the base peak is the most intense stored peak (the file has no column for
  it; without it a base-peak chromatogram could not be drawn).

Scan numbers come from ``SpectrumName`` (``scan=N``). A file whose names
do not carry one (hand-built, scrubbed) falls back to ``SpectrumIndex``.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.io.encyclopedia._dia import DiaFile, scan_number
from constellation.massspec.schemas import (
    ACQUISITION_METADATA_SCHEMA_VERSION,
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)

SOURCE_FORMAT = "encyclopedia_dia"
BUNDLE_KIND = "encyclopedia_dia_acquisition"
MANIFEST_SCHEMA_VERSION = 1

# One scan, as the merge and the two writers pass it around.
_Scan = dict[str, Any]


def _scans(batches: Iterator[pa.RecordBatch], level: int) -> Iterator[_Scan]:
    """Flatten record batches into per-scan dicts with numpy peak arrays."""
    for batch in batches:
        mz = batch.column("mz_list")
        inten = batch.column("intensity_list")
        mz_off = mz.offsets.to_numpy()
        mz_val = mz.values.to_numpy()
        int_val = inten.values.to_numpy()
        if not np.array_equal(mz_off, inten.offsets.to_numpy()):
            raise ValueError(
                "a scan's m/z and intensity arrays differ in length; the .dia is corrupt"
            )
        cols = {
            name: batch.column(name).to_pylist()
            for name in batch.schema.names
            if name not in ("mz_list", "intensity_list")
        }
        for i in range(batch.num_rows):
            name = cols["spectrum_name"][i]
            number = scan_number(name)
            yield {
                "scan": cols["spectrum_index"][i] if number is None else number,
                "level": level,
                "rt": cols["scan_start_time_s"][i],
                "iit_s": cols["ion_injection_time"][i],
                "lower": cols["isolation_window_lower"][i],
                "upper": cols["isolation_window_upper"][i],
                "target": cols["isolation_window_target"][i] if level == 2 else None,
                "charge": cols["precursor_charge"][i] if level == 2 else None,
                "parent": _parent_scan(cols["precursor_name"][i]) if level == 2 else None,
                "tic": cols["tic"][i],
                "mz": mz_val[mz_off[i] : mz_off[i + 1]],
                "intensity": int_val[mz_off[i] : mz_off[i + 1]],
            }


def _parent_scan(precursor_name: str | None) -> int | None:
    """The parent MS1 scan number held in ``PrecursorName``, if it holds one.

    0.8.0 stores the scan number itself (0 = none); 0.5.0 stores the
    parent's ``SpectrumName``.
    """
    if not precursor_name:
        return None
    number = int(precursor_name) if precursor_name.isdigit() else scan_number(precursor_name)
    return number or None


class _PeakWriter:
    """Buffer scans and flush one row group per RT bin, as ``convert`` does."""

    def __init__(self, path: Path, rt_bin_width_s: float):
        self._writer = pq.ParquetWriter(
            str(path), MS_PEAK_TABLE, compression="zstd", use_dictionary=False
        )
        self._width = rt_bin_width_s
        self._bin: int | None = None
        self._scans: list[_Scan] = []

    def add(self, scan: _Scan) -> None:
        if scan["mz"].size == 0:
            return
        this = int(scan["rt"] // self._width)
        if self._bin is not None and this != self._bin:
            self.flush()
        self._bin = this
        self._scans.append(scan)

    def flush(self) -> None:
        if not self._scans:
            return
        sizes = np.array([s["mz"].size for s in self._scans])
        n = int(sizes.sum())

        def repeat(key: str, dtype: pa.DataType, ms2_only: bool = False) -> pa.Array:
            values = [
                None if (ms2_only and s["level"] != 2) else s[key] for s in self._scans
            ]
            return pa.array(values, type=dtype).take(
                pa.array(np.repeat(np.arange(len(values)), sizes), pa.int64())
            )

        charge = [
            None if (s["level"] != 2 or not s["charge"]) else s["charge"] for s in self._scans
        ]
        index = pa.array(np.repeat(np.arange(len(self._scans)), sizes), pa.int64())
        columns: dict[str, pa.Array] = {
            "scan": repeat("scan", pa.int32()),
            "level": repeat("level", pa.int8()),
            "rt": repeat("rt", pa.float64()),
            "mz": pa.array(np.concatenate([s["mz"] for s in self._scans]), pa.float64()),
            "intensity": pa.array(
                np.concatenate([s["intensity"] for s in self._scans]).astype(np.float64),
                pa.float64(),
            ),
            "isolation_lower": repeat("lower", pa.float64(), ms2_only=True),
            "isolation_upper": repeat("upper", pa.float64(), ms2_only=True),
            "precursor_scan": repeat("parent", pa.float64(), ms2_only=True),
            "precursor_mz": repeat("target", pa.float64(), ms2_only=True),
            "precursor_charge": pa.array(charge, pa.int8()).take(index),
        }
        table = pa.table(
            {
                f.name: columns.get(f.name, pa.nulls(n, f.type))
                for f in MS_PEAK_TABLE
            },
            schema=MS_PEAK_TABLE,
        )
        self._writer.write_table(table)
        self._scans = []

    def close(self) -> None:
        self.flush()
        self._writer.close()


def _base_peak(scan: _Scan) -> dict[str, float | None]:
    """The most intense stored peak; the first of equals, as an argmax gives."""
    if scan["mz"].size == 0:
        return {"base_peak_mz": None, "base_peak_intensity": None}
    top = int(np.argmax(scan["intensity"]))
    return {
        "base_peak_mz": float(scan["mz"][top]),
        "base_peak_intensity": float(scan["intensity"][top]),
    }


def _scan_metadata_row(scan: _Scan) -> dict[str, Any]:
    ms2 = scan["level"] == 2
    return {
        "scan": scan["scan"],
        "level": scan["level"],
        "rt": scan["rt"],
        "tic": scan["tic"],
        # Seconds (held as float32 in the file) → milliseconds. Going through
        # the short decimal the float32 stands for (0.095544, not
        # 0.09554400295...) keeps the value exact when it is later written
        # back out as seconds.
        "iit": (
            None
            if scan["iit_s"] is None
            else float(str(np.float32(scan["iit_s"]))) * 1000.0
        ),
        "peak_count": int(scan["mz"].size),
        **_base_peak(scan),
        "isolation_lower": scan["lower"] if ms2 else None,
        "isolation_upper": scan["upper"] if ms2 else None,
        "scan_window_lower": None if ms2 else scan["lower"],
        "scan_window_upper": None if ms2 else scan["upper"],
        "precursor_scan": scan["parent"] if ms2 else None,
        "precursor_mz": scan["target"] if ms2 else None,
        "precursor_charge": (scan["charge"] or None) if ms2 else None,
    }


def import_dia(
    dia_path: Path | str,
    out_dir: Path | str,
    *,
    rt_bin_width_s: float = 60.0,
    overwrite: bool = False,
) -> list[Path]:
    """Write one bundle per fraction of ``dia_path`` under ``out_dir``.

    Returns the bundle directories, in fraction order. A single-run file
    gives one bundle named after the ``.dia``; a merged file gives one
    per fraction, named after the fraction's source.
    """
    from constellation import __version__ as constellation_version

    dia_path = Path(dia_path).resolve()
    out_dir = Path(out_dir)
    bundles: list[Path] = []
    with DiaFile(dia_path) as dia:
        names = dict(dia.fractions)
        fraction_ids = dia.fraction_ids()
        if len(fraction_ids) > 1 and set(fraction_ids) - set(names):
            raise ValueError(
                f"{dia_path} has fractions {fraction_ids} but its fractions table "
                f"names only {sorted(names)}; cannot name the bundles"
            )
        targets = {
            f: out_dir / Path(names[f] if len(fraction_ids) > 1 else dia_path.name).stem
            for f in fraction_ids
        }
        if len(set(targets.values())) != len(targets):
            raise ValueError(f"{dia_path}: fraction names are not distinct: {names}")
        for bundle in targets.values():
            if (bundle / "manifest.json").exists() and not overwrite:
                raise FileExistsError(
                    f"{bundle} already holds a bundle (pass overwrite=True to replace it)"
                )

        for fraction in fraction_ids:
            bundle = targets[fraction]
            bundle.mkdir(parents=True, exist_ok=True)
            # Remove the manifest first: it is written last, so its
            # absence marks a bundle that is not complete.
            (bundle / "manifest.json").unlink(missing_ok=True)
            # Within a fraction both tables are in scan order, so a
            # two-way merge restores the acquisition order.
            merged = heapq.merge(
                _scans(dia.iter_precursor(fraction=fraction), 1),
                _scans(dia.iter_spectra(fraction=fraction), 2),
                key=lambda s: s["scan"],
            )
            peaks = _PeakWriter(bundle / "peaks.parquet", rt_bin_width_s)
            rows: list[dict[str, Any]] = []
            previous: int | None = None
            try:
                for scan in merged:
                    if previous is not None and scan["scan"] <= previous:
                        raise ValueError(
                            f"{dia_path} fraction {fraction}: scan numbers are not "
                            f"strictly increasing at {scan['scan']} (after {previous}); "
                            f"the file is not in acquisition order"
                        )
                    previous = scan["scan"]
                    peaks.add(scan)
                    rows.append(_scan_metadata_row(scan))
            finally:
                peaks.close()

            scan_table = pa.Table.from_pylist(rows, schema=SCAN_METADATA_TABLE)
            pq.write_table(scan_table, bundle / "scan_metadata.parquet", compression="zstd")
            md = dia.metadata
            rts = [r["rt"] for r in rows]
            single = len(fraction_ids) == 1
            source = str(md.get("sourcename") or dia_path.name) if single else names[fraction]
            analyzers = md.get("acq.mass_analyzers") if single else None
            acquisition = {
                "source_format": SOURCE_FORMAT,
                "source_path": str(dia_path),
                "constellation_version": constellation_version,
                "schema_version": ACQUISITION_METADATA_SCHEMA_VERSION,
                "instrument_model": md.get("instrument.model") if single else None,
                "instrument_serial": md.get("instrument.serial_number") if single else None,
                "software_version": (
                    None
                    if not single or md.get("instrument.software_version") is None
                    else str(md["instrument.software_version"])
                ),
                "creation_date": md.get("thermo.creation_date") if single else None,
                "analyzers": (
                    [a.removeprefix("MassAnalyzer") for a in str(analyzers).split(",")]
                    if analyzers
                    else None
                ),
                "n_scans": len(rows),
                "rt_min_seconds": min(rts) if rts else None,
                "rt_max_seconds": max(rts) if rts else None,
                "profile_mode": False,
            }
            pq.write_table(
                pa.Table.from_pylist([acquisition], schema=ACQUISITION_METADATA_TABLE),
                bundle / "acquisition_metadata.parquet",
                compression="zstd",
            )
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": BUNDLE_KIND,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                # The run this bundle stands for: the name the .dia records
                # for it, which is what a search writes as SourceFile.
                "source_file": source,
                "source_sha256": None,
                "parameters": {
                    "rt_bin_width_s": rt_bin_width_s,
                    "dia_file": str(dia_path),
                    "dia_format_version": dia.format_version,
                    "fraction": fraction,
                },
                "outputs": {
                    "peaks": "peaks.parquet",
                    "scan_metadata": "scan_metadata.parquet",
                    "acquisition_metadata": "acquisition_metadata.parquet",
                },
                "run_metadata": {
                    k: acquisition[k]
                    for k in (
                        "instrument_model",
                        "instrument_serial",
                        "software_version",
                        "creation_date",
                        "n_scans",
                        "rt_min_seconds",
                        "rt_max_seconds",
                        "profile_mode",
                        "analyzers",
                    )
                },
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            bundles.append(bundle)
    return bundles


__all__ = ["BUNDLE_KIND", "SOURCE_FORMAT", "import_dia"]
