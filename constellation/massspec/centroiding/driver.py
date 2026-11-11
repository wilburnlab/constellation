"""Make a peak list from a profile bundle: a complete bundle of its own.

Named for what made it: ``<run>.<algorithm>-<8 hex digits>``, the digest being of
the algorithm, its version and its parameters, so a changed setting sits beside
the earlier list and replaces nothing. It is written beside its source unless
told otherwise, and its manifest, written last, records where it came from.

One worker process per part of the source, each of which holds a contiguous range
of scans. A scan acquired as centroids has no profile to work on, so what it
stored passes through, and the list covers every scan. Without that, or with
``scans`` / ``stride`` asking for only some scans (for evaluation), a scan would
be empty; a partial list says so in its manifest and the spectra-file writer
refuses it, because a search would quietly lose what those scans hold.

Profile scans the algorithm was not made for are refused rather than guessed at:
today that means anything but an ion trap. An Orbitrap's instrument centroids are
in the run's peak bundle, not its profile bundle.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core import stages
from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    PEAKS_DIR,
    PROFILE_DIR,
    PeakWriter,
    is_profile_bundle,
    part_name,
    peak_rows,
    read_bundle_manifest,
)
from constellation.massspec.centroiding.apex import pick_apex
from constellation.massspec.precursor_major import PRECURSOR_MAJOR_DIR, build_precursor_major
from constellation.massspec.schemas import SCAN_METADATA_TABLE

SCAN_METADATA_FILENAME = "scan_metadata.parquet"
ACQUISITION_METADATA_FILENAME = "acquisition_metadata.parquet"
#: The analyzer an algorithm's profile scans must come from.
ION_TRAP = "ITMS"


@dataclass(frozen=True)
class Algorithm:
    """What a peak list is made with: from one scan's m/z, intensity and
    injection time (ms, None when unknown) to its peaks' m/z and areas."""

    name: str
    version: int
    pick: Callable[..., tuple[np.ndarray, np.ndarray]]


ALGORITHMS: dict[str, Algorithm] = {
    "apex": Algorithm("apex", 1, pick_apex),
}


def peak_list_key(algorithm: Algorithm, parameters: dict[str, Any] | None = None) -> str:
    """``<algorithm>-<8 hex digits of what made it>``: no run in it, so one
    name means one way of centroiding across runs."""
    digest = stages.stage_key(
        algorithm=algorithm.name, version=algorithm.version, parameters=parameters or {}
    )
    return f"{algorithm.name}-{digest[:8]}"


def _level_summary(scans: pa.Table) -> list[dict[str, Any]]:
    """Per MS level: scans, peaks, and how many scans each way of centroiding made."""
    levels: dict[int, dict[str, Any]] = {}
    grouped = scans.group_by(["level", "centroider"], use_threads=False).aggregate(
        [("scan", "count"), ("peak_count", "sum")]
    )
    for row in grouped.to_pylist():
        entry = levels.setdefault(
            int(row["level"]), {"level": int(row["level"]), "n_scans": 0, "n_peaks": 0, "peaks_from": {}}
        )
        entry["n_scans"] += int(row["scan_count"])
        entry["n_peaks"] += int(row["peak_count_sum"] or 0)
        entry["peaks_from"][row["centroider"]] = int(row["scan_count"])
    return [levels[k] for k in sorted(levels)]


def _chosen_scans(scans: pa.Table, chosen: Sequence[int] | None, stride: int | None) -> np.ndarray | None:
    numbers = scans.column("scan").to_numpy()
    if chosen is not None:
        return np.unique(np.asarray(list(chosen), dtype=np.int64))
    if stride is not None:
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        return numbers[::stride].astype(np.int64)
    return None


def _make_part(
    source: str,
    target: str,
    algorithm: str,
    parameters: dict[str, Any],
    scans: pa.Table,
    chosen: np.ndarray | None,
    rt_bin_width_s: float,
) -> pa.Table:
    """One part of the list; returns ``scan``, ``peak_count``,
    ``intensity_floor``, ``spectrum_type`` and ``centroider`` for the scans
    it held."""
    pick = ALGORITHMS[algorithm].pick
    final = Path(target)
    tmp = final.with_name(f".{final.name}.tmp")
    file = pq.ParquetFile(source)
    # What a scan was stored as, and its injection time, by scan number.
    numbers = scans.column("scan").to_numpy()
    stored_as = scans.column("spectrum_type").to_pylist()
    iits = scans.column("iit").to_pylist()
    where = {int(n): i for i, n in enumerate(numbers.tolist())}

    out_scan: list[int] = []
    out_count: list[int] = []
    out_floor: list[float | None] = []
    out_type: list[str] = []
    out_made: list[str | None] = []

    def flush(writer: PeakWriter, numbers_: list[int], mzs: list[np.ndarray], ints: list[np.ndarray]) -> None:
        if not numbers_:
            return
        rows = pa.array([where[n] for n in numbers_], pa.int64())
        writer.write(
            peak_rows(
                scans.take(rows),
                np.array([m.size for m in mzs], dtype=np.int64),
                np.concatenate(mzs),
                np.concatenate(ints),
            )
        )

    carry_scan: int | None = None
    carry_mz: list[np.ndarray] = []
    carry_int: list[np.ndarray] = []
    pending_numbers: list[int] = []
    pending_mz: list[np.ndarray] = []
    pending_int: list[np.ndarray] = []

    with PeakWriter(tmp, rt_bin_width_s=rt_bin_width_s, byte_stream_split=True) as writer:

        def finish_scan(number: int, mz: np.ndarray, intensity: np.ndarray) -> None:
            at = where[number]
            if stored_as[at] == "profile":
                if chosen is not None and not np.isin(number, chosen):
                    peaks = (np.empty(0), np.empty(0))
                    made = algorithm
                else:
                    peaks = pick(mz, intensity, iits[at], **parameters)
                    made = algorithm
                kind = "centroid"
            else:  # acquired as centroids: what it stored is its peaks
                peaks = (mz, intensity)
                made = "instrument"
                kind = "centroid"
            out_scan.append(number)
            out_count.append(int(peaks[0].size))
            out_floor.append(float(np.min(peaks[1])) if peaks[1].size else None)
            out_type.append(kind)
            out_made.append(made)
            pending_numbers.append(number)
            pending_mz.append(peaks[0])
            pending_int.append(peaks[1])
            if len(pending_numbers) >= 256:
                flush(writer, pending_numbers, pending_mz, pending_int)
                pending_numbers.clear()
                pending_mz.clear()
                pending_int.clear()

        for batch in file.iter_batches(batch_size=1 << 20, columns=["scan", "mz", "intensity"]):
            scan = batch.column(0).to_numpy()
            mz = batch.column(1).to_numpy()
            intensity = batch.column(2).to_numpy()
            cuts = np.flatnonzero(np.diff(scan)) + 1
            for a, b in zip((0, *cuts.tolist()), (*cuts.tolist(), scan.size)):
                number = int(scan[a])
                if carry_scan is not None and number != carry_scan:
                    finish_scan(carry_scan, np.concatenate(carry_mz), np.concatenate(carry_int))
                    carry_mz, carry_int = [], []
                carry_scan = number
                carry_mz.append(mz[a:b])
                carry_int.append(intensity[a:b])
        if carry_scan is not None:
            finish_scan(carry_scan, np.concatenate(carry_mz), np.concatenate(carry_int))
        flush(writer, pending_numbers, pending_mz, pending_int)
    os.replace(tmp, final)
    return pa.table(
        {
            "scan": pa.array(out_scan, pa.int32()),
            "peak_count": pa.array(out_count, pa.uint32()),
            "intensity_floor": pa.array(out_floor, pa.float64()),
            "spectrum_type": pa.array(out_type, pa.string()),
            "centroider": pa.array(out_made, pa.string()),
        }
    )


def _make_part_star(args: tuple) -> pa.Table:
    return _make_part(*args)


def _refuse_other_analyzers(scans: pa.Table, algorithm: Algorithm) -> None:
    stored = scans.column("spectrum_type").to_pylist()
    analyzers = scans.column("analyzer").to_pylist()
    others = sorted({a or "unknown" for s, a in zip(stored, analyzers) if s == "profile" and a != ION_TRAP})
    if others:
        raise ValueError(
            f"this run has profile scans from {', '.join(others)}; {algorithm.name} is for "
            f"ion-trap profile data, and a profile bundle does not hold an Orbitrap's "
            f"instrument centroids to pass through"
        )


def centroid_bundle(
    profile_bundle: Path | str,
    algorithm: str,
    *,
    output_dir: Path | str | None = None,
    parameters: dict[str, Any] | None = None,
    scans: Sequence[int] | None = None,
    stride: int | None = None,
    workers: int = 1,
    force: bool = False,
    rt_bin_width_s: float = 60.0,
    precursor_major: bool = True,
) -> Path:
    """Make a peak list from ``profile_bundle``; returns the new bundle's path.
    With ``precursor_major`` its peaks are also written ordered by isolation
    window (``massspec.precursor_major``), before its manifest."""
    source = Path(profile_bundle)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {', '.join(sorted(ALGORITHMS))}")
    chosen_algorithm = ALGORITHMS[algorithm]
    manifest = read_bundle_manifest(source)
    if not is_profile_bundle(source):
        raise ValueError(
            f"{source} holds peaks, not profile points; a peak list is made from a run's "
            f"profile bundle (<run>.profile, from `massspec convert`)"
        )
    parameters = dict(parameters or {})
    name = peak_list_key(chosen_algorithm, parameters)
    run = Path(str(manifest.get("source_file") or source.name).replace("\\", "/").rsplit("/", 1)[-1]).stem
    out = Path(output_dir) if output_dir is not None else source.parent / f"{run}.{name}"
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"peak list already there: {out} (pass --force to make it again)")
        shutil.rmtree(out)
    scan_table = pq.read_table(source / SCAN_METADATA_FILENAME)
    _refuse_other_analyzers(scan_table, chosen_algorithm)
    chosen = _chosen_scans(scan_table, scans, stride)
    partial = chosen is not None

    (out / PEAKS_DIR).mkdir(parents=True, exist_ok=True)
    parts = sorted((source / PROFILE_DIR).glob("part-*.parquet"))
    # What a worker reads per scan: the verbatim trailer is most of the table.
    narrow = scan_table.select([c for c in scan_table.column_names if c != "trailer_extras"])
    tasks = [
        (str(p), str(out / PEAKS_DIR / part_name(i)), algorithm, parameters, narrow, chosen, rt_bin_width_s)
        for i, p in enumerate(parts)
    ]
    try:
        if workers <= 1 or len(tasks) <= 1:
            results = [_make_part_star(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                results = list(pool.map(_make_part_star, tasks))
        made = pa.concat_tables(results) if results else None
        table = scan_table
        if made is not None and made.num_rows:
            order = {int(n): i for i, n in enumerate(made.column("scan").to_pylist())}
            rows = [order.get(int(n)) for n in scan_table.column("scan").to_pylist()]
            if "intensity_floor" not in table.column_names:  # a source written before v5
                table = cast_to_schema(table, SCAN_METADATA_TABLE)
            for column in ("peak_count", "intensity_floor", "spectrum_type", "centroider"):
                values = made.column(column).take(pa.array(rows, pa.int64()))
                if column == "peak_count":
                    values = values.fill_null(0)
                elif column == "intensity_floor":
                    pass  # a scan with no peaks has no floor
                else:
                    # A scan with no rows in the source has no points either way.
                    values = pc.coalesce(
                        values, pa.scalar("centroid" if column == "spectrum_type" else algorithm)
                    )
                at = table.schema.get_field_index(column)
                table = table.set_column(at, table.schema.field(column), values.cast(table.schema.field(column).type))
        pq.write_table(table, out / SCAN_METADATA_FILENAME, compression="zstd")
        shutil.copy2(source / ACQUISITION_METADATA_FILENAME, out / ACQUISITION_METADATA_FILENAME)
        total = int(pc.sum(table.column("peak_count")).as_py() or 0)
        record = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": "thermo_acquisition",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_file": manifest.get("source_file"),
            "source_sha256": manifest.get("source_sha256"),
            "parameters": {
                "rt_bin_width_s": rt_bin_width_s,
                "peaks": "centroid",
                "centroider": algorithm,
                "partial": partial,
                "n_scans_picked": None if chosen is None else int(chosen.size),
                "workers": workers,
            },
            "outputs": {
                "peaks": PEAKS_DIR,
                **({"precursor_major": PRECURSOR_MAJOR_DIR} if precursor_major else {}),
                "scan_metadata": SCAN_METADATA_FILENAME,
                "acquisition_metadata": ACQUISITION_METADATA_FILENAME,
            },
            "run_metadata": {
                **(manifest.get("run_metadata") or {}),
                "n_peaks": total,
                "levels": _level_summary(table),
            },
            "derived_from": {
                "name": name,
                "source": os.path.relpath(source, out.parent),
                "algorithm": algorithm,
                "version": chosen_algorithm.version,
                "parameters": parameters,
            },
        }
        if precursor_major:
            build_precursor_major(out, workers=workers)
        (out / MANIFEST_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out
