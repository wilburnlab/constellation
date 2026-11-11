"""A bundle's points ordered by the precursor m/z each can be evidence for.

``peaks/`` is scan-major: a run's scans in the order they were acquired,
so every row group of it holds every isolation window (a window's scans
recur every cycle). A search reads the other way: a DIA query is matched
only in scans of its own window(s), over the whole gradient, and at MS1
only near its own m/z. This is the same points again, precursor-major,
so that what a query can match is contiguous on disk (PI, 2026-11-09;
``docs/plans/sweep.md``, "Precursor-major layout"). It is a copy, not an
index into ``peaks/``: Parquet reads pages, so pointers into a scan-major
file would read the whole run for each window.

Layout, one file per peak part (``part-RRRRR`` is ``peaks/part-RRRRR``'s
scans), so the build is the conversion's ranges and no scheme (DIA or
DDA, native windows or bins) has to be decided:

- ``ms<L>/part-RRRRR.parquet`` for L ≥ 2: ``lower`` (the scan's isolation
  lower bound; -inf for a scan missing either bound, which the sweep
  treats as unbounded), ``scan``, ``mz``, ``intensity``,
  stably sorted by ``lower``, so each scan's points stay together and in
  stored order (a point's position among its scan's rows is its stored
  position). A row group closes where ``lower`` changes once it holds
  :data:`MIN_GROUP_ROWS`, and never holds more than :data:`MAX_GROUP_ROWS`,
  so a window's rows are a few row groups found by their statistics.
- ``ms1/part-RRRRR.parquet``: ``mz``, ``intensity``, ``scan``, ``point``
  (the stored position), stably sorted by ``mz``, row groups of
  :data:`MAX_GROUP_ROWS`, so an m/z interval is a few row groups.
- ``scans.parquet``: every scan's ``scan``, ``level`` and ``n_points``,
  scans without points included.
- ``manifest.json`` and ``_SUCCESS``, whose stage key covers the layout
  version and a digest (xxh3-128) of every byte of each peak part and of
  ``scan_metadata.parquet`` (not the source's SHA-256, which every peak
  list of one ``.raw`` shares, nor modification times, which a copy
  changes; a file's size and footer were not enough, as two values swapped
  inside a page keep both). :func:`find_precursor_major` returns the
  layout only while that key still holds.

:func:`build_precursor_major` writes it from a bundle's peak parts, one
process per part at a time: ``massspec convert`` runs it as its last step
(unless ``--no-precursor-major``), as do ``centroid`` and the spectra-file
import, and ``massspec precursor-major BUNDLE`` runs it for any other
bundle. (Convert's range workers wrote it as they read until 2026-11-09:
no faster, and each held its range's points to sort them.) Values are the
bundle's own: no m/z correction is applied (a curve applies where named,
as everywhere).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core import stages
from constellation.massspec.bundle import peak_paths

__all__ = [
    "LAYOUT_VERSION",
    "MAX_BUILD_WORKERS",
    "MAX_GROUP_ROWS",
    "MIN_GROUP_ROWS",
    "PRECURSOR_MAJOR_DIR",
    "PrecursorMajor",
    "PrecursorMajorWriter",
    "build_precursor_major",
    "finish_precursor_major",
    "find_precursor_major",
    "layout_key",
]

PRECURSOR_MAJOR_DIR = "precursor_major"
#: Processes ``convert`` builds the layout in: on the wide-window run 8 were
#: no faster than 4 (memory bandwidth), and each holds at most
#: ``MAX_HELD_POINTS`` of a part's points at once (about 0.8 GB).
MAX_BUILD_WORKERS = 4
#: 2 (2026-11-09): a scan missing either isolation bound is stored as
#: unbounded (``lower`` -inf), as the sweep treats it; version 1 kept a
#: finite ``lower`` when only the upper bound was missing, and its points
#: were never read for that scan's queries.
LAYOUT_VERSION = 2
MIN_GROUP_ROWS = 1 << 13
MAX_GROUP_ROWS = 1 << 16
SCANS_FILENAME = "scans.parquet"
MANIFEST_FILENAME = "manifest.json"
KIND = "massspec_precursor_major"


def level_dir(level: int) -> str:
    return f"ms{int(level)}"


# ── writing ───────────────────────────────────────────────────────────


#: Points a layout writer holds before it writes what it has as files of
#: their own (``part-NNNNN-CCC.parquet``), at a scan's end: about 0.8 GB at
#: the ~100 bytes a held point costs while it is sorted. Until 2026-11-10 a
#: writer held its whole peak part, and a writer of one part a run
#: (``convert(workers=1)``, a ``.dia`` import) held the whole run.
MAX_HELD_POINTS = 1 << 23


class PrecursorMajorWriter:
    """One peak part's points, precursor-major. Fed the part's peaks in
    stored order (scans ascending, each scan's points together), it holds
    them and writes each level's file sorted on :meth:`close`, which returns
    each scan's point count, a row per run of a scan's points (summed by
    :func:`finish_precursor_major`). Past ``max_points`` held it writes what
    it holds, whole scans only, as files of their own: the sweep reads each
    file on its own, so a part's layout may be any number of them."""

    def __init__(self, root: Path | str, index: int, *, max_points: int | None = None) -> None:
        self.root = Path(root)
        self.index = int(index)
        self.max_points = MAX_HELD_POINTS if max_points is None else int(max_points)
        self._rows: dict[int, list[dict[str, np.ndarray]]] = {}
        self._types: dict[str, pa.DataType] = {}
        self._counts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._held = 0
        self._chunks = 0
        self._last_scan: int | None = None
        #: The most points held at once (for tests and measurement).
        self.peak_held = 0

    def write(self, peaks: pa.Table) -> None:
        """``peaks``: rows with ``scan``, ``level``, ``mz``, ``intensity``,
        ``isolation_lower`` and ``isolation_upper``, in stored order."""
        if peaks.num_rows == 0:
            return
        if self._held >= self.max_points:
            # Write out at a scan's end: the rows that finish the last scan
            # held go with it.
            scan = peaks.column("scan").to_numpy()
            if self._last_scan is not None and scan[0] == self._last_scan:
                other = scan != self._last_scan
                if not other.any():
                    # The whole batch is the scan held last, which may go on
                    # past it: hold it and write out where the scan ends.
                    # Until 2026-11-11 the flush came here, splitting a scan
                    # spanning three batches (review A8).
                    self._hold(peaks)
                    return
                k = int(np.argmax(other))
                self._hold(peaks.slice(0, k))
                peaks = peaks.slice(k)
            self._flush(final=False)
            if peaks.num_rows == 0:
                return
        self._hold(peaks)

    def _hold(self, peaks: pa.Table) -> None:
        if peaks.num_rows == 0:
            return
        scan = peaks.column("scan").to_numpy().astype(np.int64)
        self._last_scan = int(scan[-1])
        self._held += peaks.num_rows
        self.peak_held = max(self.peak_held, self._held)
        level = peaks.column("level").to_numpy().astype(np.int8)
        self._counts.append(_runs(scan, level))
        # Held as numpy columns (the bundle's own m/z and intensity types),
        # one list per level.
        for name in ("mz", "intensity"):
            self._types.setdefault(name, peaks.schema.field(name).type)
        lower = peaks.column("isolation_lower").to_numpy(zero_copy_only=False).astype(np.float64)
        upper = peaks.column("isolation_upper").to_numpy(zero_copy_only=False).astype(np.float64)
        cols = {
            "scan": scan.astype(np.int32),
            "mz": peaks.column("mz").to_numpy(),
            "intensity": peaks.column("intensity").to_numpy(),
            # Missing either bound is unbounded, as the sweep's schedule has it.
            "lower": np.where(np.isnan(lower) | np.isnan(upper), -np.inf, lower),
        }
        levels = np.unique(level)
        for lv in levels.tolist():
            sel = None if levels.size == 1 else level == lv
            self._rows.setdefault(int(lv), []).append(cols if sel is None else {k: v[sel] for k, v in cols.items()})

    def close(self) -> pa.Table:
        self._flush(final=True)
        return self._count_table()

    def _flush(self, *, final: bool) -> None:
        """Write what is held, each level sorted, as one file a level: the
        part's own name when it is all of the part, else a numbered piece."""
        name = (f"part-{self.index:05d}.parquet" if final and self._chunks == 0
                else f"part-{self.index:05d}-{self._chunks:03d}.parquet")  # fmt: skip
        for level, parts in sorted(self._rows.items()):
            if not parts:
                continue
            # Whole scans of one part: what it holds since the last piece.
            rows = {k: np.concatenate([p[k] for p in parts]) for k in ("scan", "mz", "intensity", "lower")}
            self._rows[level] = []
            path = self.root / level_dir(level) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if level == 1:
                _write_ms1(rows, path, self._types)
            else:
                _write_fragments(rows, path, self._types)
        self._held = 0
        self._chunks += 1

    def _count_table(self) -> pa.Table:
        """Rows to be summed by (scan, level): each run of a scan's points,
        with its rows."""
        parts = self._counts
        if not parts:
            return pa.table({"scan": pa.array([], pa.int32()), "level": pa.array([], pa.int8()),
                             "n_points": pa.array([], pa.int32())})  # fmt: skip
        return pa.table({
            "scan": pa.array(np.concatenate([s for s, _, _ in parts]).astype(np.int32)),
            "level": pa.array(np.concatenate([lv for _, lv, _ in parts]).astype(np.int8)),
            "n_points": pa.array(np.concatenate([n for _, _, n in parts]).astype(np.int32)),
        })  # fmt: skip


def _runs(scan: np.ndarray, level: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each run of one scan's rows: (scan, level, rows)."""
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    return scan[starts], level[starts], np.diff(np.r_[starts, scan.size]).astype(np.int64)


def _replace_into(table_writes, path: Path, schema: pa.Schema) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    # Encoded as the peak list is (plain floats under zstd): dictionaries of
    # m/z and intensity only grow the file.
    with pq.ParquetWriter(tmp, schema, compression="zstd", use_dictionary=False) as w:
        for t in table_writes:
            w.write_table(t, row_group_size=MAX_GROUP_ROWS)
    os.replace(tmp, path)


def _write_fragments(rows: dict[str, np.ndarray], path: Path, types: dict[str, pa.DataType]) -> None:
    lower = rows["lower"]
    # A run's scans have few distinct windows: sort their codes, which a
    # stable sort of small integers does in linear time.
    windows, code = np.unique(lower, return_inverse=True)
    code = code.astype(np.int16 if windows.size < (1 << 15) else np.int64)
    order = np.argsort(code, kind="stable")
    lower = lower[order]
    table = pa.table({
        "lower": pa.array(lower),
        "scan": pa.array(rows["scan"][order]),
        "mz": pa.array(rows["mz"][order], types.get("mz")),
        "intensity": pa.array(rows["intensity"][order], types.get("intensity")),
    })  # fmt: skip
    cuts = _group_cuts(lower)
    _replace_into((table.slice(a, b - a) for a, b in zip(cuts[:-1], cuts[1:], strict=True)), path, table.schema)


def _group_cuts(lower: np.ndarray) -> list[int]:
    """Row-group edges: a group closes where ``lower`` changes once it holds
    MIN_GROUP_ROWS, and holds at most MAX_GROUP_ROWS."""
    n = lower.size
    if n == 0:
        return [0, 0]
    changes = np.flatnonzero(lower[1:] != lower[:-1]) + 1
    cuts = [0]
    for edge in [*changes.tolist(), n]:
        while edge - cuts[-1] > MAX_GROUP_ROWS:
            cuts.append(cuts[-1] + MAX_GROUP_ROWS)
        if edge - cuts[-1] >= MIN_GROUP_ROWS or edge == n:
            if edge > cuts[-1]:
                cuts.append(edge)
    return cuts


def _write_ms1(rows: dict[str, np.ndarray], path: Path, types: dict[str, pa.DataType]) -> None:
    scan = rows["scan"]
    starts = np.flatnonzero(np.r_[True, scan[1:] != scan[:-1]])
    point = (np.arange(scan.size, dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, scan.size]))).astype(np.int32)
    order = np.argsort(rows["mz"], kind="stable")
    table = pa.table({
        "mz": pa.array(rows["mz"][order], types.get("mz")),
        "intensity": pa.array(rows["intensity"][order], types.get("intensity")),
        "scan": pa.array(scan[order]),
        "point": pa.array(point[order]),
    })  # fmt: skip
    _replace_into([table], path, table.schema)


# ── the whole layout ─────────────────────────────────────────────────


def _content_digest(path: Path) -> str:
    """xxh3-128 of a file's bytes: the project's digest of content too large
    for 64 bits (CLAUDE.md, "Content-derived ids"); about 2 GB/s here, where
    SHA-256 ran at 350 MB/s (7 s for the wide-window run's peaks, at every
    sweep's start)."""
    import xxhash

    digest = xxhash.xxh3_128()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return f"{path.stat().st_size}:{digest.hexdigest()}"


def layout_key(bundle: Path | str) -> str:
    """The stage key of a bundle's precursor-major layout: the layout
    version and a digest of every byte of each peak part and of the scan
    metadata. Until 2026-11-09 each file's size and Parquet footer stood in
    for its bytes, and two intensities swapped inside a page kept both, so
    an obsolete layout passed as current (review by Codex). About a second
    for 2 GB of peaks, paid once by a sweep's parent and once by a build."""
    bundle = Path(bundle)
    parts = [[p.name, _content_digest(p)] for p in peak_paths(bundle)]
    return stages.stage_key(
        layout=LAYOUT_VERSION, peaks=parts, scan_metadata=_content_digest(bundle / "scan_metadata.parquet")
    )


def finish_precursor_major(bundle: Path | str, counts: Iterable[pa.Table]) -> Path:
    """Write ``scans.parquet`` (from the parts' point counts), the manifest
    and ``_SUCCESS``, after every part is written and the scan metadata is.
    Refuses a layout whose rows are not the peak list's."""
    bundle = Path(bundle)
    root = bundle / PRECURSOR_MAJOR_DIR
    root.mkdir(parents=True, exist_ok=True)
    tables = [t for t in counts if t.num_rows]
    scans = (pa.concat_tables(tables) if tables else
             pa.table({"scan": pa.array([], pa.int32()), "level": pa.array([], pa.int8()),
                       "n_points": pa.array([], pa.int32())}))  # fmt: skip
    scans = scans.group_by(["scan", "level"]).aggregate([("n_points", "sum")]).rename_columns(
        ["scan", "level", "n_points"]).sort_by("scan")  # fmt: skip
    scans = scans.set_column(2, "n_points", scans.column("n_points").cast(pa.int32()))
    peak_rows = sum(pq.read_metadata(p).num_rows for p in peak_paths(bundle))
    held = sum(pq.read_metadata(p).num_rows for p in root.glob("ms*/part-*.parquet"))
    counted = int(pc.sum(scans.column("n_points")).as_py() or 0)
    if not (peak_rows == held == counted):
        raise ValueError(
            f"{root}: {held} rows written and {counted} counted for a peak list of {peak_rows}: not its points"
        )
    pq.write_table(scans, root / SCANS_FILENAME, compression="zstd")
    key = layout_key(bundle)
    levels = sorted(int(d.name[2:]) for d in root.glob("ms*") if d.is_dir())
    manifest = {
        "kind": KIND,
        "layout_version": LAYOUT_VERSION,
        "levels": levels,
        "parts": {level_dir(lv): sorted(p.name for p in (root / level_dir(lv)).glob("part-*.parquet")) for lv in levels},
        "rows": held,
        "min_group_rows": MIN_GROUP_ROWS,
        "max_group_rows": MAX_GROUP_ROWS,
    }
    tmp = root / f".{MANIFEST_FILENAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, root / MANIFEST_FILENAME)
    stages.mark_done(root, key)
    return root


@dataclass(frozen=True)
class PrecursorMajor:
    """A bundle's finished precursor-major layout."""

    root: Path
    levels: tuple[int, ...]
    key: str

    def files(self, level: int) -> list[Path]:
        return sorted((self.root / level_dir(level)).glob("part-*.parquet"))

    def scans(self) -> pa.Table:
        return pq.read_table(self.root / SCANS_FILENAME)


def find_precursor_major(bundle: Path | str) -> PrecursorMajor | None:
    """The bundle's precursor-major layout if it is finished and still the
    peak list's (its key holds); None otherwise."""
    bundle = Path(bundle)
    root = bundle / PRECURSOR_MAJOR_DIR
    if not (root / MANIFEST_FILENAME).exists():
        return None
    try:
        key = layout_key(bundle)
    except (OSError, ValueError):
        return None
    if not stages.is_done(root, key):
        return None
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    if manifest.get("kind") != KIND or manifest.get("layout_version") != LAYOUT_VERSION:
        return None
    return PrecursorMajor(root=root, levels=tuple(manifest["levels"]), key=key)


# ── building one for an existing bundle ──────────────────────────────

_COLUMNS = ["scan", "level", "mz", "intensity", "isolation_lower", "isolation_upper"]


#: Rows a build reads of a peak part at a time.
BUILD_BATCH_ROWS = 1 << 20


def _build_part(part: str, root: str, index: int) -> pa.Table:
    writer = PrecursorMajorWriter(root, index)
    f = pq.ParquetFile(part)
    for batch in f.iter_batches(columns=_COLUMNS, batch_size=BUILD_BATCH_ROWS):
        writer.write(pa.Table.from_batches([batch]))
    return writer.close()


def build_precursor_major(bundle: Path | str, *, workers: int = 1, force: bool = False) -> Path:
    """Write ``bundle``'s precursor-major layout from its peak parts (in
    ``workers`` processes); a layout that is finished and current is kept
    unless ``force``."""
    import shutil

    bundle = Path(bundle)
    if not force and find_precursor_major(bundle) is not None:
        return bundle / PRECURSOR_MAJOR_DIR
    root = bundle / PRECURSOR_MAJOR_DIR
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    parts = peak_paths(bundle)
    indices = [_part_index(p, i) for i, p in enumerate(parts)]
    jobs = [(str(p), str(root), k) for p, k in zip(parts, indices, strict=True)]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            counts = list(pool.map(_build_part, *zip(*jobs, strict=True)))
    else:
        counts = [_build_part(*job) for job in jobs]
    # Scans with no points are in no part: they come from the scan metadata.
    meta = pq.read_table(bundle / "scan_metadata.parquet", columns=["scan", "level"])
    every = pa.table({"scan": meta.column("scan").cast(pa.int32()), "level": meta.column("level").cast(pa.int8()),
                      "n_points": pa.array(np.zeros(meta.num_rows, np.int32))})  # fmt: skip
    return finish_precursor_major(bundle, [every, *counts])


def _part_index(path: Path, i: int) -> int:
    stem = path.stem
    if stem.startswith("part-") and stem[5:].isdigit():
        return int(stem[5:])
    return i


def describe(bundle: Path | str) -> dict[str, Any] | None:
    """What a bundle's layout holds (its manifest), or None without one."""
    root = Path(bundle) / PRECURSOR_MAJOR_DIR
    path = root / MANIFEST_FILENAME
    return json.loads(path.read_text()) if path.exists() else None
