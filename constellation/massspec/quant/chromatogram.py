"""Extracted-ion-chromatogram (XIC) extraction.

The load-bearing MS primitive: a tolerance-gated match between expected
and observed m/z, carrying the observed intensity + the scan's retention
time through. One engine, ``massspec.sweep``: every target is compiled
into a query, and each scan is matched against all the queries active in
it at once, then the matches are reduced to trace rows by
:class:`XicTraceReducer`. Values are the canonical f64 peaks, so a trace is
always f64-exact.

  ``extract_xic`` — every level asked, in one sweep. It reads a bundle
  scan-major (its ``peaks/`` in scan ranges; Mode A) or, with
  ``layout='precursor_major'``, from the bundle's precursor-major layout a
  unit of targets at a time (``massspec.precursor_major``), which scales
  with the targets rather than with their matches; a target's rows are the
  same either way. ``extract_xic_scan_major`` is one level of it, with the
  extractor's original keywords. Its rows are those the earlier
  target-by-target implementation gave, bit for bit, except where that
  implementation was wrong: a target with a sequence and a charge but no
  precursor m/z has one computed, so its fragments are gated by isolation
  window, and the assigned-scan restriction applies to MS2 only (an MS1
  scan never has an MS2 scan's number).

  Mode B, ``extract_xic_indexed`` (slices of the m/z-sorted ``peak_index``,
  target by target), was retired on 2026-11-09: its rows did not equal
  Mode A's, and the precursor-major sweep is the per-target-scalable path
  it was meant to be. The browser still uses the index for single XICs
  (``massspec.scans.xic``).

Both consume the normalized ``XIC_TARGET_TABLE`` (see
``massspec.quant.targets``) and emit ``XIC_TRACE_TABLE`` (the deliberate
terminal artifact — trace → PRECURSOR_QUANT reduction is a separate
downstream step). Match semantics default to nearest-peak (one row per
(target, scan, ion); rectangular). ``all_in_window=True`` returns every
peak within tolerance (interference / chimeric / DIA work).

Channels (the m/z extracted at, with what each is) come from one
generator, ``massspec.peptide.channels`` through ``sweep.compile_queries``:
MS1 isotope envelopes from composition (or isotopes off a bare
``precursor_mz`` for the no-peptide m/z case), MS2 fragment ladders, with
abundances and top-N selection. Mass error is conventional (observed -
theoretical), signed, preserved so probabilistic scoring stays downstream.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.bundle import open_peaks
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.quant.schemas import XIC_TRACE_TABLE

if TYPE_CHECKING:
    from constellation.massspec.sweep import ChannelSettings
    from constellation.massspec.sweep.engine import SweepResult
    from constellation.massspec.sweep.evidence import EvidenceGate
    from constellation.massspec.sweep.reduce import QueryBatch

__all__ = [
    "extract_xic",
    "extraction_tolerance_metadata",
    "extract_xic_scan_major",
    "XicTraceReducer",
    "save_xic",
    "write_xic_manifest",
    "load_xic",
]

_XIC_MANIFEST = "manifest.json"
_XIC_TABLE_FILE = "xic_trace.parquet"

ToleranceUnit = Literal["ppm", "Da"]
_DEFAULT_ION_TYPES = (IonType.B, IonType.Y)


# ──────────────────────────────────────────────────────────────────────
# Target + query representations
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Target:
    target_id: int
    precursor_charge: int | None
    precursor_mz: float | None
    modified_sequence: str | None
    rt_center: float | None
    rt_start: float | None
    rt_end: float | None
    scan: int | None


@dataclass(slots=True)
class _QueryIon:
    """One theoretical m/z to extract, with its trace coordinates."""

    mz: float
    precursor_charge: int
    isotope: int | None = None  # MS1
    ion_type: int | None = None  # MS2 (IonType value)
    position: int | None = None
    fragment_charge: int | None = None
    loss_id: str | None = None


def _targets_to_records(targets: pa.Table) -> list[_Target]:
    d = targets.to_pydict()
    n = targets.num_rows
    return [
        _Target(
            target_id=d["target_id"][i],
            precursor_charge=d["precursor_charge"][i],
            precursor_mz=d["precursor_mz"][i],
            modified_sequence=d["modified_sequence"][i],
            rt_center=d["rt_center"][i],
            rt_start=d["rt_start"][i],
            rt_end=d["rt_end"][i],
            scan=d["scan"][i],
        )
        for i in range(n)
    ]


def _rt_bounds(t: _Target, rt_window: float | None) -> tuple[float, float]:
    """Resolve the [lo, hi] RT gate for a target. Explicit [start, end]
    wins; else center ± window; else unbounded."""
    if t.rt_start is not None and t.rt_end is not None:
        return t.rt_start, t.rt_end
    if t.rt_center is not None and rt_window is not None:
        return t.rt_center - rt_window, t.rt_center + rt_window
    return -np.inf, np.inf


def _query_ions(channels, level: int, q: int) -> list[_QueryIon]:
    """Query ``q``'s channels (``massspec.peptide.channels``) as the ions
    the index extraction walks (it once built its own: the sweep's
    generator is the one source of channels)."""
    a, b = int(channels.offsets[q]), int(channels.offsets[q + 1])
    names = [None, *channels.loss_ids]
    if level == 1:
        return [
            _QueryIon(mz=float(channels.mz[j]), precursor_charge=int(channels.precursor_charge[j]),
                      isotope=int(channels.isotope[j]))
            for j in range(a, b)
        ]  # fmt: skip
    return [
        _QueryIon(mz=float(channels.mz[j]), precursor_charge=int(channels.precursor_charge[j]),
                  ion_type=int(channels.ion_type[j]), position=int(channels.position[j]),
                  fragment_charge=int(channels.fragment_charge[j]), loss_id=names[int(channels.loss[j])])
        for j in range(a, b)
    ]  # fmt: skip


# ──────────────────────────────────────────────────────────────────────
# Output assembly
# ──────────────────────────────────────────────────────────────────────


class _TraceBuilder:
    """Accumulates matched (target, scan, ion) rows into XIC_TRACE_TABLE."""

    _NUM = (
        "acquisition_id",
        "target_id",
        "precursor_charge",
        "level",
        "scan",
        "rt",
        "isolation_lower",
        "isolation_upper",
        "isotope",
        "ion_type",
        "position",
        "fragment_charge",
        "mz_theoretical",
        "mz_observed",
        "intensity",
        "mz_error_da",
        "mz_error_ppm",
    )

    def __init__(self) -> None:
        self._cols: dict[str, list] = {k: [] for k in self._NUM}
        self._cols["modified_sequence"] = []
        self._cols["loss_id"] = []

    def add(self, **arrays) -> None:
        n = len(arrays["scan"])
        if n == 0:
            return
        for k in self._cols:
            v = arrays.get(k)
            if v is None:
                self._cols[k].extend([None] * n)
            elif np.isscalar(v) or isinstance(v, str):
                self._cols[k].extend([v] * n)
            else:
                self._cols[k].extend(list(v))

    def table(self) -> pa.Table:
        tbl = pa.table(self._cols)
        return cast_to_schema(tbl, XIC_TRACE_TABLE)


def _emit_grid(
    builder: _TraceBuilder,
    *,
    acquisition_id: int,
    target: _Target,
    level: int,
    ions: list[_QueryIon],
    scan_ids: np.ndarray,  # (n_scans,)
    rts: np.ndarray,
    iso_lo: np.ndarray,
    iso_hi: np.ndarray,
    observed_mz: np.ndarray,  # (n_scans, n_ions) NaN where unmatched
    intensity: np.ndarray,  # (n_scans, n_ions)
    within: np.ndarray,  # (n_scans, n_ions) bool
    drop_unmatched: bool,
) -> None:
    """Flatten a (scan × ion) match grid into trace rows."""
    n_scans, n_ions = within.shape
    keep = within if drop_unmatched else np.ones_like(within, dtype=bool)
    si, qi = np.nonzero(keep)
    if si.size == 0:
        return
    theo = np.array([ion.mz for ion in ions], dtype=np.float64)
    obs = observed_mz[si, qi]
    inten = np.where(within[si, qi], intensity[si, qi], 0.0)
    err_da = np.where(within[si, qi], obs - theo[qi], np.nan)
    builder.add(
        acquisition_id=acquisition_id,
        target_id=target.target_id,
        modified_sequence=target.modified_sequence,
        precursor_charge=np.array(
            [ions[q].precursor_charge for q in qi], dtype=np.int64
        ),
        level=level,
        scan=scan_ids[si],
        rt=rts[si],
        isolation_lower=iso_lo[si],
        isolation_upper=iso_hi[si],
        isotope=[ions[q].isotope for q in qi],
        ion_type=[ions[q].ion_type for q in qi],
        position=[ions[q].position for q in qi],
        fragment_charge=[ions[q].fragment_charge for q in qi],
        loss_id=[ions[q].loss_id for q in qi],
        mz_theoretical=theo[qi],
        mz_observed=np.where(within[si, qi], obs, np.nan),
        intensity=inten,
        mz_error_da=err_da,
        mz_error_ppm=np.where(within[si, qi], err_da / theo[qi] * 1e6, np.nan),
    )


def _finalize(table: pa.Table, output_path: Path | str | None) -> pa.Table | Path:
    if output_path is None:
        return table
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out


_EXTRACTION_TOL_KEY = b"x.massspec.extraction_tolerance"
_EXTRACTION_TOL_UNIT_KEY = b"x.massspec.extraction_tolerance_unit"


def extraction_tolerance_metadata(tolerances: dict[int, tuple[float, str]]) -> dict[bytes, bytes]:
    """Schema metadata recording the tolerances a trace was extracted at:
    ``x.massspec.extraction_tolerance`` (and its unit) when every level used
    one, and ``x.massspec.extraction_tolerance.ms<L>`` per level when the
    trace spans more than one level."""
    meta: dict[bytes, bytes] = {}
    distinct = {(float(t), str(u)) for t, u in tolerances.values()}
    if len(distinct) == 1:
        tol, unit = distinct.pop()
        meta[_EXTRACTION_TOL_KEY] = repr(tol).encode("utf-8")
        meta[_EXTRACTION_TOL_UNIT_KEY] = unit.encode("utf-8")
    if len(tolerances) > 1:
        for level, (tol, unit) in sorted(tolerances.items()):
            meta[_EXTRACTION_TOL_KEY + f".ms{level}".encode()] = repr(float(tol)).encode("utf-8")
            meta[_EXTRACTION_TOL_UNIT_KEY + f".ms{level}".encode()] = str(unit).encode("utf-8")
    return meta


def _stamp_extraction_tolerance(
    table: pa.Table, tolerance: float, tolerance_unit: str
) -> pa.Table:
    """Record the XIC match tolerance in the trace's schema metadata. A peak only
    enters a target's trace when it sits within this tolerance of a target channel,
    so a downstream consumer (e.g. counter component grouping) can read it to bound
    how far apart two co-fit candidates may be and still share extracted signal."""
    meta = dict(table.schema.metadata or {})
    meta[_EXTRACTION_TOL_KEY] = repr(float(tolerance)).encode("utf-8")
    meta[_EXTRACTION_TOL_UNIT_KEY] = str(tolerance_unit).encode("utf-8")
    return table.replace_schema_metadata(meta)


def save_xic(
    table: pa.Table,
    path: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write an XIC_TRACE_TABLE as a small bundle: ``xic_trace.parquet`` +
    ``manifest.json``. A derived artifact (not a ``Quant`` container), so
    it lives outside the QUANT_WRITERS registry."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, p / _XIC_TABLE_FILE)
    return write_xic_manifest(p, metadata=metadata)


def write_xic_manifest(path: Path | str, *, metadata: dict[str, Any] | None = None) -> Path:
    """Write the ``manifest.json`` that makes ``path`` (holding
    ``xic_trace.parquet``) an XIC trace bundle."""
    p = Path(path)
    manifest = {
        "format": "xic_trace",
        "schema_version": 1,
        "table": _XIC_TABLE_FILE,
        "metadata": metadata or {},
    }
    (p / _XIC_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return p


def load_xic(path: Path | str) -> pa.Table:
    """Read an XIC trace bundle written by :func:`save_xic`. Tolerates a
    bare ``xic_trace.parquet`` and back-fills schema metadata."""
    p = Path(path)
    table_file = p / _XIC_TABLE_FILE if p.is_dir() else p
    table = pq.read_table(table_file)
    if table.schema.metadata is None:
        table = table.replace_schema_metadata(XIC_TRACE_TABLE.metadata)
    return table


# ──────────────────────────────────────────────────────────────────────
# Mode A — scan-major sweep
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class XicTraceReducer:
    """Sweep matches as ``XIC_TRACE_TABLE`` rows, in Mode A's order: target
    input order, then (rt, scan), then the target's channels in the order
    its query lists them, then point. With ``drop_unmatched=False`` every
    cell the sweep could have matched is a row, intensity 0.0 and NaN m/z
    and errors where nothing was within tolerance (NaN, not null, as
    always). A rule that can match several points per cell (``all``) gives
    matches only."""

    drop_unmatched: bool = True
    name: str = "xic_trace"
    schema: pa.Schema = XIC_TRACE_TABLE
    #: Only the queries the sweep's evidence gate passed.
    gated: bool = False

    @property
    def zero_fills(self) -> bool:
        """A row for every eligible cell (the sweep sizes buckets by them)."""
        return not self.drop_unmatched

    def reduce(self, batch: "QueryBatch") -> pa.Table:
        plan, s = batch.plan, batch.scans
        fill = not self.drop_unmatched and batch.settings.rule.name != "all"
        parts = []
        for level in plan.levels:
            channels = plan.channels_for_level(level)
            if channels is None:
                continue
            c = batch.cells(level, fill=fill)
            parts.append((level, channels, c.query, c.row, c.channel, c.point_index, c.mz_observed, c.intensity,
                          c.matched))  # fmt: skip
        return _trace_table(plan, s, batch.scan_order, parts, batch.settings.acquisition_id)


def _trace_table(plan, s, scan_order, parts, acquisition_id) -> pa.Table:
    from constellation.massspec.sweep.schedule import order_by

    if not parts:
        return cast_to_schema(XIC_TRACE_TABLE.empty_table(), XIC_TRACE_TABLE)
    cols: dict[str, list] = {k: [] for k in (
        "query", "level", "row", "channel", "point", "charge", "isotope", "ion_type",
        "position", "fragment_charge", "loss_id", "theo", "obs", "intensity", "matched",
    )}  # fmt: skip
    for level, channels, q, r, ch, point, obs, inten, matched in parts:
        g = channels.offsets[q] + ch
        n = q.size
        frag = level != 1
        cols["query"].append(q)
        cols["level"].append(np.full(n, level, np.int8))
        cols["row"].append(r)
        cols["channel"].append(ch)
        cols["point"].append(point)
        cols["charge"].append(channels.precursor_charge[g])
        cols["isotope"].append(np.ma.masked_array(channels.isotope[g], mask=np.full(n, frag)))
        for name, values in (("ion_type", channels.ion_type), ("position", channels.position),
                             ("fragment_charge", channels.fragment_charge)):  # fmt: skip
            cols[name].append(np.ma.masked_array(values[g], mask=np.full(n, not frag)))
        names = np.array([None, *channels.loss_ids], dtype=object)
        cols["loss_id"].append(names[channels.loss[g]] if frag else np.full(n, None, dtype=object))
        cols["theo"].append(channels.mz[g])
        cols["obs"].append(obs)
        cols["intensity"].append(inten)
        cols["matched"].append(matched)
    c = {k: (np.ma.concatenate(v) if isinstance(v[0], np.ma.MaskedArray) else np.concatenate(v))
         for k, v in cols.items()}  # fmt: skip
    by = order_by(c["query"], c["level"], scan_order[c["row"]], c["channel"], c["point"])
    c = {k: v[by] for k, v in c.items()}
    row = c["row"]
    matched = c["matched"]
    theo = c["theo"]
    obs = np.where(matched, c["obs"], np.nan)
    err = np.where(matched, obs - theo, np.nan)
    query = pa.array(c["query"])
    targets = plan.targets
    n = int(c["query"].size)

    def masked(v, dtype):
        return pa.array(np.asarray(v.filled(0), dtype=dtype), mask=np.ma.getmaskarray(v))

    table = pa.table(
        {
            "acquisition_id": pa.array(np.full(n, acquisition_id, np.int64)),
            "target_id": targets.column("target_id").take(query),
            "modified_sequence": targets.column("modified_sequence").take(query),
            "precursor_charge": pa.array(c["charge"].astype(np.int8)),
            "level": pa.array(c["level"]),
            "scan": pa.array(s.scan[row].astype(np.int32)),
            "rt": pa.array(s.rt[row]),
            "isolation_lower": pa.array(s.lower[row]),
            "isolation_upper": pa.array(s.upper[row]),
            "isotope": masked(c["isotope"], np.int8),
            "ion_type": masked(c["ion_type"], np.int8),
            "position": masked(c["position"], np.int32),
            "fragment_charge": masked(c["fragment_charge"], np.int32),
            "loss_id": pa.array(c["loss_id"], pa.string()),
            "mz_theoretical": pa.array(theo),
            "mz_observed": pa.array(obs),
            "intensity": pa.array(np.where(matched, c["intensity"], 0.0)),
            "mz_error_da": pa.array(err),
            "mz_error_ppm": pa.array(np.where(matched, err / theo * 1e6, np.nan)),
        }
    )
    return cast_to_schema(table, XIC_TRACE_TABLE)


def extract_xic(
    peaks: pa.Table | Path | str,
    targets: pa.Table,
    *,
    acquisition_id: int,
    levels: Sequence[int],
    tolerances: dict[int, tuple[float, ToleranceUnit]],
    channels: "ChannelSettings | None" = None,
    rt_window: float | None = None,
    rule: str = "nearest",
    assigned_scans_only: bool = False,
    drop_unmatched: bool = True,
    mz_axis: str | None = None,
    gate: "EvidenceGate | None" = None,
    only_passing: bool = False,
    claims: bool = False,
    summary: bool = False,
    workers: int = 1,
    work_dir: Path | str | None = None,
    progress=None,
    fragment_abundance=None,
    layout: str = "scan_major",
    spill_dir: Path | str | None = None,
    acquired_ranges: Any = "bundle",
) -> "SweepResult":
    """Extract XIC traces for ``targets`` at every MS level in ``levels`` in
    one sweep of ``peaks`` (a bundle, streamed, or a peak table), each level
    at its own tolerance; ``channels`` says which isotopes and fragment ions
    each target is matched at. The trace is the result's ``xic_trace``
    output: ``.table("xic_trace")`` for it in memory,
    ``.write("xic_trace", path)`` to stream it to a file. ``rule`` is
    ``nearest`` (Mode A's), ``all`` (every point within tolerance) or
    ``sum`` (their sum, as one row).

    With a ``gate``, ``only_passing`` keeps only the targets it passed in
    the trace. ``claims`` adds the ``point_claims`` output (points matched
    by more than one target, or by two channels of one) and ``summary`` the
    ``query_summary`` output (per target: the gate's verdict, scans
    eligible, cells matched). ``fragment_abundance`` says where fragment
    channels' abundances come from (``massspec.peptide.channels``; a
    library's with ``library.fragment_abundance.LibraryFragments``), which
    ``channels.top_fragments`` selects on. Every output carries ``tolerances`` in its
    schema metadata (:func:`extraction_tolerance_metadata`), in memory and
    when written. ``layout='precursor_major'`` reads the bundle's
    precursor-major layout instead of its scan-ordered peaks (the same rows
    per target, in units of targets by precursor m/z; ``sweep``).
    ``spill_dir`` is where a sweep that outgrows memory carries on on disk
    (the system's temporary directory by default; ``sweep``).
    ``acquired_ranges`` is the MS2 acquired m/z range fragment channels are
    made within (``sweep.compile_queries``): ``"bundle"`` (the default) reads
    it from a bundle's scan metadata by isolation window, None makes every
    channel, and anything else is given to the compile as it is. Until
    2026-11-11 every channel was made, so the top N fragments counted ions
    the scans never acquired (review of 2026-11-10, C6)."""
    from constellation.massspec.sweep import ChannelSettings, compile_queries
    from constellation.massspec.sweep.engine import SweepSettings, sweep
    from constellation.massspec.sweep.reduce import QuerySummaryReducer
    from constellation.massspec.sweep.rules import rule_named

    if only_passing and gate is None:
        raise ValueError("only_passing needs a gate to pass targets")

    from constellation.massspec.peptide.channels import UNMODELLED

    if isinstance(acquired_ranges, str):
        if acquired_ranges != "bundle":
            raise ValueError(f"acquired_ranges is 'bundle', None or the ranges; got {acquired_ranges!r}")
        acquired_ranges = _bundle_ranges(peaks) if any(int(lv) >= 2 for lv in levels) else None
    plan = compile_queries(
        targets,
        levels=levels,
        channels=channels if channels is not None else ChannelSettings(),
        rt_window=rt_window,
        assigned_scans_only=assigned_scans_only,
        fragment_abundance=fragment_abundance if fragment_abundance is not None else UNMODELLED,
        acquired_ranges=acquired_ranges,
    )
    settings = SweepSettings(
        tolerances=dict(tolerances),
        rule=rule_named(rule),
        mz_axis=mz_axis,
        acquisition_id=acquisition_id,
        gate=gate,
        claims=claims,
    )
    if not isinstance(peaks, pa.Table):
        path = Path(peaks)
        if not (path / "manifest.json").is_file():
            peaks = open_peaks(path).to_table()
    reducers = [XicTraceReducer(drop_unmatched=drop_unmatched, gated=only_passing)]
    if summary:
        reducers.append(QuerySummaryReducer())
    result = sweep(peaks, plan, settings, reducers=reducers, workers=workers, work_dir=work_dir,
                   progress=progress, layout=layout, spill_dir=spill_dir)  # fmt: skip
    # The tolerances go with every output, read or written: Counter bounds
    # its collision grouping by the trace's.
    stamp = extraction_tolerance_metadata(dict(tolerances))
    for name in result.schemas:
        result.metadata[name] = {**result.metadata.get(name, {}), **stamp}
    return result


def _bundle_ranges(peaks) -> pa.Table | None:
    """A bundle's MS2 acquired ranges by isolation window, or None for a
    peak table or a directory that is no bundle."""
    if isinstance(peaks, pa.Table):
        return None
    path = Path(peaks)
    if not ((path / "manifest.json").is_file() and (path / "scan_metadata.parquet").is_file()):
        return None
    from constellation.massspec.acquisition_structure import read_acquired_ranges

    return read_acquired_ranges(path)


def extract_xic_scan_major(
    peaks: pa.Table | Path | str,
    targets: pa.Table,
    *,
    acquisition_id: int,
    level: int,
    n_isotopes: int = 3,
    ion_types: Sequence[IonType] = _DEFAULT_ION_TYPES,
    max_fragment_charge: int = 2,
    neutral_losses: Sequence[str] | None = None,
    charge_range: tuple[int, int] | None = None,
    rt_window: float | None = None,
    tolerance: float = 20.0,
    tolerance_unit: ToleranceUnit = "ppm",
    all_in_window: bool = False,
    assigned_scans_only: bool = False,
    drop_unmatched: bool = True,
    mz_axis: str | None = None,
    workers: int = 1,
    output_path: Path | str | None = None,
) -> pa.Table | Path:
    """Mode A scan-major XIC extraction. See module docstring.

    ``peaks`` is a converted acquisition (a bundle directory, streamed) or
    a peak table. ``all_in_window=True`` returns every peak within tolerance
    of each channel in each scan instead of the nearest (interference /
    chimeric work).
    ``mz_axis`` (a bundle's measured m/z axis key, or ``latest``) corrects
    the observed m/z before matching. ``workers > 1`` sweeps a bundle's scan
    ranges in that many processes; the rows are the same.
    """
    from constellation.massspec.sweep import ChannelSettings

    result = extract_xic(
        peaks,
        targets,
        acquisition_id=acquisition_id,
        levels=[level],
        tolerances={level: (tolerance, tolerance_unit)},
        channels=ChannelSettings(
            n_isotopes=n_isotopes,
            charge_range=charge_range,
            ion_types=tuple(ion_types),
            max_fragment_charge=max_fragment_charge,
            neutral_losses=tuple(neutral_losses) if neutral_losses else None,
        ),
        rt_window=rt_window,
        rule="all" if all_in_window else "nearest",
        assigned_scans_only=assigned_scans_only,
        drop_unmatched=drop_unmatched,
        mz_axis=mz_axis,
        workers=workers,
    )
    return _finalize(result.table("xic_trace"), output_path)
