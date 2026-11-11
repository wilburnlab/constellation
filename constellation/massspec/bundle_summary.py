"""What a converted acquisition holds, and how a run's peak lists compare.

:func:`summarize_bundle` describes one bundle, names the run's other bundles
beside it, and sets the m/z curves they have side by side. A centroider's bias
shows as the difference between its peak list's curve and the profile's, with
no identifications: the profile's curve measures the instrument, and a peak
list's measures the instrument and what picked it.

Everything comes from what the bundles record about themselves. No curve is
measured here (``massspec mz-axis`` does that, and the summary says which are
missing) and no point is read.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.massspec.bundle import read_bundle_manifest
from constellation.massspec.mz_axis import MzAxis, load_mz_axis, scale_error_ppm
from constellation.massspec.scans import open_acquisition

#: The width of the m/z bins a curve is shown in, in Th. Its knots are finer.
BIN_TH = 100.0


def _size_bytes(bundle: Path) -> int:
    return sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())


def _levels(bundle: Path, info: Any) -> list[dict[str, Any]]:
    meta = pq.read_table(bundle / "scan_metadata.parquet", columns=["level", "peak_count"])
    levels = meta.column("level")
    rows = []
    for level in info.levels:
        counts = meta.column("peak_count").filter(pc.equal(levels, level))
        rows.append(
            {
                "level": level,
                "n_scans": info.scans_per_level[level],
                "n_points": int(pc.sum(counts).as_py() or 0),
                "analyzers": list(info.analyzers[level]),
            }
        )
    return rows


def _binned_curve(axis: MzAxis, level: int, profile: MzAxis | None) -> dict[float, dict[str, Any]]:
    """One level of a curve by m/z bin. ``bias_th`` is its difference from
    ``profile``, taken only where the profile's own curve reaches, because past
    its last knot that curve is held flat and would look like a bias."""
    at = np.flatnonzero(axis.table.column("level").to_numpy() == level)
    mz = axis.table.column("mz").to_numpy()[at]
    shift = axis.table.column("shift_th").to_numpy()[at]
    coherence = axis.table.column("coherence").to_numpy()[at]
    bias = np.full(mz.shape, np.nan)
    # A profile curve whose branch is undetermined at this level has no shift
    # to compare with (``summarize`` raised there; review A9).
    if profile is not None and profile.is_determined(level):
        knots, _ = profile.knots(level)
        if knots.size:
            inside = (mz >= knots.min()) & (mz <= knots.max())
            bias[inside] = shift[inside] - profile.shift(level, mz[inside])
    bins: dict[float, dict[str, Any]] = {}
    for low in np.unique(np.floor(mz / BIN_TH) * BIN_TH):
        members = (mz >= low) & (mz < low + BIN_TH)
        known = bias[members][np.isfinite(bias[members])]
        bins[float(low)] = {
            "shift_th": float(shift[members].mean()),
            "coherence": float(np.median(coherence[members])),
            "bias_th": float(known.mean()) if known.size else None,
        }
    return bins


def _curves(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The run's curves level by level, the profile's first."""
    loaded = {m["name"]: load_mz_axis(m["bundle"]) for m in members if m["mz_axes"]}
    profile = next(
        (loaded[m["name"]] for m in members if m["holds_profile"] and m["name"] in loaded), None
    )
    levels = sorted({int(v) for axis in loaded.values() for v in axis.table.column("level").to_pylist()})
    rows = []
    for level in levels:
        lists = []
        for member in members:
            axis = loaded.get(member["name"])
            if axis is None or axis.knots(level)[0].size == 0:
                continue
            against = None if member["holds_profile"] else profile
            lists.append(
                {
                    "name": member["name"],
                    "key": axis.key,
                    "scale_error_ppm": scale_error_ppm(axis, level),
                    "bins": {f"{low:.0f}": row for low, row in _binned_curve(axis, level, against).items()},
                }
            )
        rows.append({"level": level, "lists": lists})
    return rows


def summarize_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    """Plain data (it serialises as JSON as it is)::

        path, name, kind, holds_profile, source_file, source_sha256, instrument
        size_bytes, acquisition_type, staggered, rt_range_s
        levels                 [{level, n_scans, n_points, analyzers}]
        n_scans_without_injection_time   None when the bundle does not say
        derived_from, partial  how this list was made, if it was made later
        bundles                [{name, path, this, holds_profile, mz_axes}], the
                               run's bundles beside this one, this one included
        mz_axis                [{level, lists: [{name, key, scale_error_ppm,
                               bins: {low m/z: {shift_th, coherence, bias_th}}}]}]

    Raises what ``read_bundle_manifest`` raises for a directory that is not a
    current bundle.
    """
    bundle = Path(bundle_dir).resolve()
    manifest = read_bundle_manifest(bundle)
    info = open_acquisition(bundle)
    run = manifest.get("run_metadata") or {}
    parameters: Mapping[str, Any] = manifest.get("parameters") or {}
    this = {
        "name": info.peak_list,
        "bundle": bundle,
        "holds_profile": info.holds_profile,
        "mz_axes": list(info.mz_axes),
    }
    members = sorted(
        [this, *info.related],
        key=lambda m: (not m["holds_profile"], str(m["bundle"])),
    )
    return {
        "path": str(bundle),
        "name": info.peak_list,
        "kind": info.kind,
        "holds_profile": info.holds_profile,
        "source_file": info.source_file,
        "source_sha256": info.source_sha256,
        "instrument": run.get("instrument_model"),
        "size_bytes": _size_bytes(bundle),
        "acquisition_type": str(info.acquisition_type) if info.acquisition_type else None,
        "staggered": info.staggered,
        "rt_range_s": list(info.rt_range),
        "levels": _levels(bundle, info),
        "n_scans_without_injection_time": run.get("n_scans_without_injection_time"),
        "derived_from": manifest.get("derived_from"),
        "partial": bool(parameters.get("partial")),
        "bundles": [
            {
                "name": m["name"],
                "path": str(m["bundle"]),
                "this": Path(m["bundle"]) == bundle,
                "holds_profile": m["holds_profile"],
                "mz_axes": list(m["mz_axes"]),
            }
            for m in members
        ],
        "mz_axis": _curves(members),
    }


def _axis_lines(entry: Mapping[str, Any]) -> list[str]:
    lists = entry["lists"]
    lows = sorted({float(low) for item in lists for low in item["bins"]})
    has_bias = any(row["bias_th"] is not None for item in lists for row in item["bins"].values())

    head = f"    {'m/z':<6}"
    rule_names = ""
    for item in lists:
        width = 16 + (8 if has_bias else 0)
        rule_names += f"  {item['name'][:width]:<{width}}"
    lines = [f"    {'':<6}{rule_names}".rstrip()]
    for _item in lists:
        head += f"  {'shift':>7} {'coher':>6}" + (f" {'bias':>7}" if has_bias else "") + " "
    lines.append(head.rstrip())

    for low in lows:
        row = f"    {low:<6.0f}"
        for item in lists:
            cell = item["bins"].get(f"{low:.0f}")
            if cell is None:
                row += "  " + " " * (14 + (8 if has_bias else 0)) + " "
                continue
            row += f"  {cell['shift_th']:>+7.3f} {cell['coherence']:>6.2f}"
            if has_bias:
                row += f" {cell['bias_th']:>+7.3f}" if cell["bias_th"] is not None else " " * 8
            row += " "
        lines.append(row.rstrip())
    scales = [
        f"{item['name']} {item['scale_error_ppm']:+.0f} ppm"
        for item in lists
        if item["scale_error_ppm"] is not None
    ]
    if scales:
        lines.append(f"    overall scale: {', '.join(scales)}")
    return lines


def format_bundle_summary(summary: Mapping[str, Any]) -> str:
    """:func:`summarize_bundle` as text."""
    gib = summary["size_bytes"] / 2**30
    noun = "profile points" if summary["holds_profile"] else "peaks"
    total = sum(row["n_points"] for row in summary["levels"])
    scans = sum(row["n_scans"] for row in summary["levels"])
    lines = [
        summary["path"],
        f"  holds        the {summary['name']} list: {total:,} {noun} in {scans:,} scans, {gib:.2f} GiB",
        f"  source       {summary['source_file']}"
        + ("" if summary["source_sha256"] else " (no checksum recorded)"),
    ]
    if summary["instrument"]:
        lines.append(f"  instrument   {summary['instrument']}")
    if summary["acquisition_type"]:
        layout = "staggered " if summary["staggered"] else ""
        lines.append(f"  acquisition  {layout}{summary['acquisition_type']}")
    low, high = summary["rt_range_s"]
    lines.append(f"  retention    {low / 60:.1f}-{high / 60:.1f} min")
    for row in summary["levels"]:
        lines.append(
            f"  MS{row['level']:<10} {row['n_scans']:,} scans, {row['n_points']:,} {noun}"
            + (f" ({', '.join(row['analyzers'])})" if row["analyzers"] else "")
        )
    blank = summary["n_scans_without_injection_time"]
    if blank:
        lines.append(f"  {blank:,} scans have a blank trailer, so no injection time")
    derived = summary["derived_from"]
    if derived:
        lines.append(f"  made from    {derived.get('source')} by {derived.get('algorithm')}")
    if summary["partial"]:
        lines.append("  partial      only some scans were picked, so this list cannot be searched")

    lines += ["", "The run's bundles:"]
    width = max(len(b["name"]) for b in summary["bundles"])
    missing = []
    for b in summary["bundles"]:
        curve = f"m/z curve {b['mz_axes'][-1]}" if b["mz_axes"] else "no m/z curve"
        mark = "*" if b["this"] else " "
        lines.append(f"  {mark} {b['name']:<{width}}  {curve:<20} {b['path']}")
        if not b["mz_axes"]:
            missing.append(b["path"])

    for entry in summary["mz_axis"]:
        lines += ["", f"m/z axis, MS{entry['level']} (Th; bias is a list's shift minus the profile's):"]
        lines += _axis_lines(entry)
    if missing:
        lines += ["", "To measure what is missing:"]
        lines += [f"  constellation massspec mz-axis {path}" for path in missing]
    return "\n".join(lines) + "\n"


__all__ = ["BIN_TH", "format_bundle_summary", "summarize_bundle"]
