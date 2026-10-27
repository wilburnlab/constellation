"""Synthetic converted acquisitions (bundles) shared by the mass-spec tests.

Small DIA and DDA runs shaped like real ones (scan order, isolation
windows, precursor links, filter strings), and a writer that lays them out
as ``massspec convert`` does: ``peaks.parquet``, ``scan_metadata.parquet``,
``acquisition_metadata.parquet`` and, with :func:`write_bundle_manifest`, the
manifest that makes the directory a bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.massspec.schemas import (
    ACQUISITION_METADATA_TABLE,
    MS_PEAK_TABLE,
    SCAN_METADATA_TABLE,
)

MS1_FILTER = "ITMS + c NSI r Full ms [350.0000-1250.0000]"


def dia_scans(
    *,
    cycles: int = 12,
    windows: tuple[tuple[float, float], ...] = ((400.0, 402.0), (402.0, 404.0), (404.0, 406.0)),
    seed: int = 0,
    empty_scan: int | None = None,
) -> list[dict[str, Any]]:
    """A small DIA run: ``cycles`` × (1 MS1 + one MS2 per window)."""
    rng = np.random.default_rng(seed)
    scans: list[dict[str, Any]] = []
    number = 0
    for _cycle in range(cycles):
        for level, window in [(1, None), *[(2, w) for w in windows]]:
            number += 1
            n_peaks = 0 if number == empty_scan else int(rng.integers(3, 9))
            mz = np.sort(rng.uniform(200.0, 1500.0, n_peaks))
            scan = {
                "scan": number,
                "level": level,
                # Not representable in float32, so rounding is exercised.
                "rt": number * 0.731 + 0.0123456789,
                "iit": 31.043 if level == 2 else 0.753,
                "mz": mz,
                "intensity": rng.uniform(10.0, 5e5, n_peaks),
                "filter_string": MS1_FILTER,
            }
            if window is not None:
                lo, hi = window
                target = (lo + hi) / 2.0 + 0.4324
                scan.update(
                    isolation_lower=target - 1.0,
                    isolation_upper=target + 1.0,
                    precursor_mz=target,
                    ms2_isolation_offset=0.0,
                    filter_string=f"ITMS + c NSI r Full ms2 {target:.4f}@hcd26.00 [200.0000-1500.0000]",
                    activation_type="hcd",
                )
            scans.append(scan)
    return scans


def dda_scans(*, n_ms1: int = 4, per_ms1: int = 5, seed: int = 1) -> list[dict[str, Any]]:
    """A small DDA run: every MS2 isolates a different precursor."""
    rng = np.random.default_rng(seed)
    scans: list[dict[str, Any]] = []
    number = 0
    for _ in range(n_ms1):
        number += 1
        parent = number
        mz = np.sort(rng.uniform(350.0, 1250.0, 6))
        scans.append(
            {"scan": number, "level": 1, "rt": number * 0.4, "iit": 1.0, "mz": mz,
             "intensity": rng.uniform(1e3, 1e6, 6), "filter_string": MS1_FILTER}
        )  # fmt: skip
        for _ in range(per_ms1):
            number += 1
            # Five decimals of real precision; the filter string keeps four.
            target = round(float(rng.uniform(400.0, 1000.0)), 5)
            offset = 0.25
            mz = np.sort(rng.uniform(200.0, 1500.0, 5))
            scans.append(
                {"scan": number, "level": 2, "rt": number * 0.4, "iit": 11.333, "mz": mz,
                 "intensity": rng.uniform(10.0, 1e5, 5),
                 "isolation_lower": target + offset - 0.8,
                 "isolation_upper": target + offset + 0.8,
                 "precursor_mz": target, "precursor_charge": 2, "precursor_scan": float(parent),
                 "ms2_isolation_offset": offset, "activation_type": "hcd",
                 "filter_string": f"ITMS + c NSI r d Full ms2 {target:.4f}@hcd26.00 [200.0000-1500.0000]"}
            )  # fmt: skip
    return scans


def write_bundle(
    root: Path,
    name: str,
    scans: list[dict[str, Any]],
    *,
    profile: bool = False,
    drop_scan_window_columns: bool = False,
    layout: str = "parts",
) -> Path:
    """Write ``scans`` as a bundle shaped like ``massspec convert`` output.

    ``layout="parts"`` writes the peaks as two parts, the scans split
    between them (``peaks/part-0000{0,1}.parquet``, manifest v2);
    ``layout="file"`` as one ``peaks.parquet`` (bundles before v2).
    """
    bundle = root / name
    bundle.mkdir(parents=True)
    peak_rows: dict[str, list[Any]] = {f.name: [] for f in MS_PEAK_TABLE}
    meta_rows: list[dict[str, Any]] = []
    for s in scans:
        n = int(s["mz"].size)
        for f in MS_PEAK_TABLE:
            if f.name == "mz":
                peak_rows["mz"].extend(s["mz"].tolist())
            elif f.name == "intensity":
                peak_rows["intensity"].extend(s["intensity"].tolist())
            else:
                peak_rows[f.name].extend([s.get(f.name)] * n)
        row = {k: v for k, v in s.items() if k not in ("mz", "intensity")}
        row["peak_count"] = n
        if s["level"] == 1 and not drop_scan_window_columns:
            row["scan_window_lower"], row["scan_window_upper"] = 350.0, 1250.0
        meta_rows.append(row)
    peaks = pa.table(
        {k: pa.array(v, MS_PEAK_TABLE.field(k).type) for k, v in peak_rows.items()},
        schema=MS_PEAK_TABLE,
    )
    if layout == "file":
        pq.write_table(peaks, bundle / "peaks.parquet", row_group_size=17)
    else:
        # Split at a scan boundary near the middle; within each part, scans
        # still straddle row groups.
        scan_col = peaks.column("scan").to_numpy()
        cut = int(np.searchsorted(scan_col, scan_col[len(scan_col) // 2])) if len(scan_col) else 0
        (bundle / "peaks").mkdir()
        for i, part in enumerate((peaks.slice(0, cut), peaks.slice(cut))):
            pq.write_table(part, bundle / "peaks" / f"part-{i:05d}.parquet", row_group_size=17)
    scan_table = pa.Table.from_pylist(meta_rows, schema=SCAN_METADATA_TABLE)
    if drop_scan_window_columns:  # a bundle written before schema v3
        scan_table = scan_table.drop_columns(["scan_window_lower", "scan_window_upper"])
    pq.write_table(scan_table, bundle / "scan_metadata.parquet")
    acquisition = {
        "source_format": "thermo_raw",
        "source_path": f"/data/{name}.raw",
        "constellation_version": "0.0.0",
        "schema_version": 1,
        "instrument_model": "Stellar",
        "instrument_serial": "QLT-S-10199",
        "software_version": "1.0.4354",
        "creation_date": "10/07/2026 12:15:40",
        "analyzers": ["ITMS"],
        "n_scans": len(scans),
        "profile_mode": profile,
    }
    pq.write_table(
        pa.Table.from_pylist([acquisition], schema=ACQUISITION_METADATA_TABLE),
        bundle / "acquisition_metadata.parquet",
    )
    return bundle


def with_scan_stats(scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the TIC and base peak a Thermo conversion records per scan."""
    for s in scans:
        if s["mz"].size:
            top = int(np.argmax(s["intensity"]))
            s["tic"] = float(s["intensity"].sum())
            s["base_peak_mz"] = float(s["mz"][top])
            s["base_peak_intensity"] = float(s["intensity"][top])
    return scans


def write_bundle_manifest(bundle: Path, *, kind: str = "thermo_acquisition") -> Path:
    """Give a :func:`write_bundle` directory the manifest a real bundle has."""
    name = bundle.name
    parts = (bundle / "peaks").is_dir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2 if parts else 1,
                "kind": kind,
                "source_file": f"/data/{name}.raw",
                "source_sha256": f"sha-of-{name}",
                "parameters": {"rt_bin_width_s": 60.0, "profile_mode": False},
                "outputs": {
                    "peaks": "peaks" if parts else "peaks.parquet",
                    "scan_metadata": "scan_metadata.parquet",
                    "acquisition_metadata": "acquisition_metadata.parquet",
                },
                "run_metadata": {
                    "instrument_model": "Stellar",
                    "instrument_serial": "QLT-S-10199",
                    "creation_date": "10/07/2026 12:15:40",
                },
            }
        )
    )
    return bundle
