"""Compare two ``.dia`` files by content.

Two ``.dia`` files that hold the same run are almost never byte-equal:
zlib settings, row order in ``ranges``, SQLite page layout and metadata
order all differ between writers. What matters is whether a search
would see the same thing, so this compares the logical content — every
scan's scalars and peak arrays, the isolation windows, the fractions and
the metadata — and says exactly where two files part ways.

It is the acceptance test for :mod:`._dia_write` (a Constellation-written
file against one MSRawJava wrote from the same ``.raw``), and a general
"are these the same spectra" tool.

Scans are streamed from both files in ``SpectrumIndex`` order, so two
multi-gigabyte files compare in bounded memory.

Scans are paired by **fraction and instrument scan number** (the
``scan=N`` in ``SpectrumName``), not by ``SpectrumIndex``. In a merged
file the index is a running count, so one scan present in only one file
shifts every index after it; pairing on the index would then compare
different scans for the rest of the file. ``SpectrumIndex`` is compared
as an ordinary value instead, so such a shift is reported once, as a
count. Files whose names carry no scan number fall back to the index.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from constellation.massspec.io.encyclopedia._dia import DiaFile, scan_number

# Metadata keys that name the file itself or the tool that wrote it, so
# they differ between any two writers without the spectra differing.
WRITER_SPECIFIC_METADATA = frozenset({"filename", "filelocation", "file.path"})

# Fractions are far fewer than 2**20 and scan numbers far smaller than
# 2**40, so one int64 orders (fraction, scan) pairs.
_FRACTION_SHIFT = 40

_MS1_SCALARS = (
    "spectrum_index",
    "spectrum_name",
    "fraction",
    "scan_start_time_s",
    "ion_injection_time",
    "isolation_window_lower",
    "isolation_window_upper",
    "tic",
)
_MS2_SCALARS = (
    *_MS1_SCALARS,
    "precursor_name",
    "isolation_window_target",
    "precursor_charge",
)


@dataclass
class FieldDiff:
    """How one column differs across the scans of a table."""

    n_scans: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    examples: list[str] = field(default_factory=list)


@dataclass
class DiaDiff:
    """Result of :func:`diff_dia`. ``identical`` is the verdict."""

    a: Path
    b: Path
    findings: list[str] = field(default_factory=list)
    #: ``"precursor.tic"`` → how that field differs.
    fields: dict[str, FieldDiff] = field(default_factory=dict)
    n_scans_compared: int = 0
    n_peaks_compared: int = 0

    @property
    def identical(self) -> bool:
        return not self.findings and not self.fields

    def report(self) -> str:
        lines = [f"a: {self.a}", f"b: {self.b}"]
        lines.append(
            f"compared {self.n_scans_compared:,} scans, {self.n_peaks_compared:,} peaks"
        )
        lines.extend(self.findings)
        for name, d in sorted(self.fields.items()):
            lines.append(
                f"{name}: differs in {d.n_scans:,} scans "
                f"(max abs {d.max_abs:.6g}, max rel {d.max_rel:.3g})"
            )
            lines.extend(f"    {e}" for e in d.examples)
        lines.append("IDENTICAL" if self.identical else "DIFFERENT")
        return "\n".join(lines)


def _close(x: Any, y: Any, rtol: float) -> bool:
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if x == y:
            return True
        return rtol > 0 and abs(x - y) <= rtol * max(abs(x), abs(y))
    return str(x) == str(y)


def _note(diff: DiaDiff, key: str, label: str, x: Any, y: Any, max_examples: int) -> None:
    d = diff.fields.setdefault(key, FieldDiff())
    d.n_scans += 1
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        delta = abs(float(x) - float(y))
        d.max_abs = max(d.max_abs, delta)
        scale = max(abs(float(x)), abs(float(y)))
        if scale > 0:
            d.max_rel = max(d.max_rel, delta / scale)
    if len(d.examples) < max_examples:
        d.examples.append(f"{label}: {x!r} vs {y!r}")


def _compare_scalars(
    diff: DiaDiff,
    table: str,
    a: pa.RecordBatch,
    b: pa.RecordBatch,
    names: Iterable[str],
    rtol: float,
    max_examples: int,
) -> None:
    labels = _labels(a)
    for name in names:
        xs = a.column(name)
        ys = b.column(name)
        if xs.equals(ys):
            continue
        for i, (x, y) in enumerate(zip(xs.to_pylist(), ys.to_pylist())):
            if not _close(x, y, rtol):
                _note(diff, f"{table}.{name}", labels[i], x, y, max_examples)


def _compare_peaks(
    diff: DiaDiff,
    table: str,
    a: pa.RecordBatch,
    b: pa.RecordBatch,
    mz_atol: float,
    intensity_rtol: float,
    max_examples: int,
) -> None:
    labels = _labels(a)
    for name, tol, relative in (
        ("mz_list", mz_atol, False),
        ("intensity_list", intensity_rtol, True),
    ):
        xa: pa.ListArray = a.column(name)
        xb: pa.ListArray = b.column(name)
        if name == "mz_list":
            diff.n_peaks_compared += len(xa.values)
        if xa.equals(xb):
            continue
        oa = xa.offsets.to_numpy()
        ob = xb.offsets.to_numpy()
        va = xa.values.to_numpy()
        vb = xb.values.to_numpy()
        for i in range(len(xa)):
            x = va[oa[i] : oa[i + 1]]
            y = vb[ob[i] : ob[i + 1]]
            if x.size != y.size:
                d = diff.fields.setdefault(f"{table}.n_peaks", FieldDiff())
                d.n_scans += 1
                d.max_abs = max(d.max_abs, float(abs(x.size - y.size)))
                if len(d.examples) < max_examples:
                    d.examples.append(f"{labels[i]}: {x.size} vs {y.size} peaks")
                continue
            if x.size == 0 or np.array_equal(x, y):
                continue
            delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
            if relative:
                scale = np.maximum(np.abs(x), np.abs(y)).astype(np.float64)
                rel = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
                worst = float(rel.max())
            else:
                worst = float(delta.max())
            if worst <= tol:
                continue
            d = diff.fields.setdefault(f"{table}.{name}", FieldDiff())
            d.n_scans += 1
            d.max_abs = max(d.max_abs, float(delta.max()))
            if relative:
                d.max_rel = max(d.max_rel, worst)
            if len(d.examples) < max_examples:
                j = int(np.argmax(delta))
                d.examples.append(f"{labels[i]}: peak {j} {float(x[j])!r} vs {float(y[j])!r}")


def _labels(batch: pa.RecordBatch) -> list[str]:
    """How to name each scan of ``batch`` in a report line."""
    return [
        f"fraction {f} {n}"
        for f, n in zip(batch.column("fraction").to_pylist(), batch.column("spectrum_name").to_pylist())
    ]


def _keys(batch: pa.RecordBatch, by_scan: bool) -> np.ndarray:
    """The value scans are paired on: ``(fraction, scan)`` or ``SpectrumIndex``."""
    if not by_scan:
        return batch.column("spectrum_index").to_numpy()
    fraction = batch.column("fraction").to_numpy().astype(np.int64)
    scans = np.array(
        [scan_number(n) for n in batch.column("spectrum_name").to_pylist()], dtype=np.int64
    )
    return (fraction << _FRACTION_SHIFT) | scans


def _describe(key: int, by_scan: bool) -> str:
    if not by_scan:
        return f"SpectrumIndex {key}"
    return f"fraction {key >> _FRACTION_SHIFT} scan={key & ((1 << _FRACTION_SHIFT) - 1)}"


def _pairs_by_scan(*batches: pa.RecordBatch | None) -> bool:
    """Whether every name in the first batch of each file carries a scan number."""
    return all(
        b is not None
        and all(scan_number(n) is not None for n in b.column("spectrum_name").to_pylist())
        for b in batches
    )


def _aligned(
    it_a: Iterator[pa.RecordBatch], it_b: Iterator[pa.RecordBatch]
) -> Iterator[tuple[pa.RecordBatch | None, pa.RecordBatch | None, list[str], list[str]]]:
    """Pair up two batch streams that are each in acquisition order.

    Yields ``(a, b, only_a, only_b)`` where ``a`` and ``b`` hold the same
    scans in the same order (or are ``None`` when nothing in this step
    is shared) and the two lists describe scans present on one side
    only. A scan missing from one file therefore costs one report line,
    not the rest of the comparison.
    """
    a, b = next(it_a, None), next(it_b, None)
    by_scan = _pairs_by_scan(a, b)
    while a is not None and b is not None:
        ka, kb = _keys(a, by_scan), _keys(b, by_scan)
        if ka.size == kb.size and np.array_equal(ka, kb):
            yield a, b, [], []
            a, b = next(it_a, None), next(it_b, None)
            continue
        # Both ascending: everything up to the smaller of the two last
        # keys can be settled now; the rest of the longer-reaching batch
        # waits for the other stream to catch up.
        horizon = min(int(ka[-1]), int(kb[-1]))
        ready_a, ready_b = ka <= horizon, kb <= horizon
        common = np.intersect1d(ka[ready_a], kb[ready_b], assume_unique=True)
        only_a = np.setdiff1d(ka[ready_a], common, assume_unique=True).tolist()
        only_b = np.setdiff1d(kb[ready_b], common, assume_unique=True).tolist()
        described = ([_describe(k, by_scan) for k in only_a], [_describe(k, by_scan) for k in only_b])
        if common.size:
            yield (
                a.take(pa.array(np.flatnonzero(np.isin(ka, common)))),
                b.take(pa.array(np.flatnonzero(np.isin(kb, common)))),
                *described,
            )
        else:
            yield None, None, *described
        a = a.take(pa.array(np.flatnonzero(~ready_a))) if not ready_a.all() else next(it_a, None)
        b = b.take(pa.array(np.flatnonzero(~ready_b))) if not ready_b.all() else next(it_b, None)
    while a is not None:
        yield None, None, [_describe(int(k), by_scan) for k in _keys(a, by_scan)], []
        a = next(it_a, None)
    while b is not None:
        yield None, None, [], [_describe(int(k), by_scan) for k in _keys(b, by_scan)]
        b = next(it_b, None)


def _compare_table(
    diff: DiaDiff,
    table: str,
    fa: DiaFile,
    fb: DiaFile,
    *,
    rtol: float,
    mz_atol: float,
    intensity_rtol: float,
    batch_rows: int,
    max_examples: int,
) -> None:
    names = _MS2_SCALARS if table == "spectra" else _MS1_SCALARS
    it_a = fa.iter_spectra if table == "spectra" else fa.iter_precursor
    it_b = fb.iter_spectra if table == "spectra" else fb.iter_precursor
    missing: dict[str, list[str]] = {"a": [], "b": []}
    for a, b, only_a, only_b in _aligned(
        it_a(batch_rows=batch_rows), it_b(batch_rows=batch_rows)
    ):
        missing["a"].extend(only_a)
        missing["b"].extend(only_b)
        if a is None or b is None:
            continue
        diff.n_scans_compared += a.num_rows
        _compare_scalars(diff, table, a, b, names, rtol, max_examples)
        _compare_peaks(diff, table, a, b, mz_atol, intensity_rtol, max_examples)
    for label, only in missing.items():
        if only:
            shown = ", ".join(only[:max_examples]) + (", ..." if len(only) > max_examples else "")
            diff.findings.append(f"{table}: {len(only):,} scans only in {label} ({shown})")


def _compare_ranges(diff: DiaDiff, fa: DiaFile, fb: DiaFile, rtol: float) -> None:
    def keyed(f: DiaFile) -> dict[tuple[float, float], list[dict[str, Any]]]:
        out: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in f.ranges().to_pylist():
            out.setdefault((row["start"], row["stop"]), []).append(row)
        return out

    ra, rb = keyed(fa), keyed(fb)
    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))
    if only_a:
        diff.findings.append(f"ranges: {len(only_a):,} windows only in a, e.g. {only_a[:3]}")
    if only_b:
        diff.findings.append(f"ranges: {len(only_b):,} windows only in b, e.g. {only_b[:3]}")
    worst: dict[str, tuple[int, float]] = {}
    for key in sorted(set(ra) & set(rb)):
        xs, ys = ra[key], rb[key]
        if len(xs) != len(ys):
            diff.findings.append(f"ranges: window {key} listed {len(xs)}x vs {len(ys)}x")
            continue
        for x, y in zip(xs, ys):
            for name in ("duty_cycle", "num_windows", "rt_start", "rt_stop"):
                if x[name] is None or y[name] is None:
                    # A 0.5.0 file has no RT bounds; absence is not a difference.
                    continue
                if not _close(x[name], y[name], rtol):
                    n, m = worst.get(name, (0, 0.0))
                    worst[name] = (n + 1, max(m, abs(float(x[name]) - float(y[name]))))
    for name, (n, m) in sorted(worst.items()):
        diff.findings.append(f"ranges.{name}: differs in {n:,} windows (max abs {m:.6g})")


def diff_dia(
    a: Path | str,
    b: Path | str,
    *,
    rtol: float = 0.0,
    mz_atol: float = 0.0,
    intensity_rtol: float = 0.0,
    ignore_metadata: Iterable[str] = WRITER_SPECIFIC_METADATA,
    metadata_keys: str = "common",
    batch_rows: int = 256,
    max_examples: int = 3,
) -> DiaDiff:
    """Compare ``a`` and ``b``. Defaults demand exact equality.

    ``rtol`` loosens scalar comparisons (scan times, window bounds, TIC,
    duty cycle); ``mz_atol`` and ``intensity_rtol`` loosen the peak
    arrays. ``metadata_keys`` is ``"common"`` to compare only keys both
    files have (and list the rest as informational counts), or ``"all"``
    to treat a key present in one file only as a difference.
    """
    diff = DiaDiff(Path(a), Path(b))
    with DiaFile(a) as fa, DiaFile(b) as fb:
        if fa.format_version != fb.format_version:
            diff.findings.append(f"format version: {fa.format_version} vs {fb.format_version}")
        if fa.fractions != fb.fractions:
            diff.findings.append(f"fractions: {fa.fractions} vs {fb.fractions}")

        skip = set(ignore_metadata) | {"version"}
        ka, kb = set(fa.metadata) - skip, set(fb.metadata) - skip
        for key in sorted(ka & kb):
            if not _close(fa.metadata[key], fb.metadata[key], rtol):
                diff.findings.append(
                    f"metadata[{key}]: {fa.metadata[key]!r} vs {fb.metadata[key]!r}"
                )
        if metadata_keys == "all":
            for label, only in (("a", ka - kb), ("b", kb - ka)):
                if only:
                    diff.findings.append(f"metadata keys only in {label}: {sorted(only)}")

        _compare_ranges(diff, fa, fb, rtol)
        for table in ("precursor", "spectra"):
            _compare_table(
                diff,
                table,
                fa,
                fb,
                rtol=rtol,
                mz_atol=mz_atol,
                intensity_rtol=intensity_rtol,
                batch_rows=batch_rows,
                max_examples=max_examples,
            )
    return diff


__all__ = ["WRITER_SPECIFIC_METADATA", "DiaDiff", "FieldDiff", "diff_dia"]
