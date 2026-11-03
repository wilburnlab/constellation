"""Extracted-ion-chromatogram (XIC) extraction.

The load-bearing MS primitive: a tolerance-gated match between expected
and observed m/z, carrying the observed intensity + the scan's retention
time through. Two modes share the ``core.signal`` matching kernel and the
same target contract / output schema:

  ``extract_xic_scan_major`` — Mode A (default). One pass over the
  canonical scan-major peaks through ``massspec.sweep``: every target is
  compiled into a query, and each scan is matched against all the queries
  active in it at once, then the matches are reduced to trace rows by
  :class:`XicTraceReducer`. Reads the canonical f64 peaks (a bundle is
  streamed, a table is taken whole), so it is always f64-exact. Its rows
  are those the earlier target-by-target implementation gave, bit for bit,
  except where that implementation was wrong: a target with a sequence and
  a charge but no precursor m/z has one computed, so its fragments are
  gated by isolation window, and the assigned-scan restriction applies to
  MS2 only (an MS1 scan never has an MS2 scan's number).

  ``extract_xic_indexed`` — Mode B. Slices a derived mass-sorted peak
  index (``massspec.quant.peak_index``): per query, a window bound-search
  on the partition's mz axis, then nearest-per-scan selection. Per-query
  independent → the path that scales to proteome / search query counts.
  Over an f32-m/z index the reported mass error is f32 unless
  ``exact_error=True`` re-derives it from the canonical table for matched
  survivors. NOTE the f32-m/z store also affects row *presence*: a peak
  sitting within ~0.06 ppm of the tolerance edge can round across it, so an
  f32-m/z index can match a hair more or fewer peaks than f64 Mode A.
  ``exact_error`` re-derives values for matched rows, not presence, so it does
  NOT recover such a boundary miss. Keep ``mz`` at f64 (the default) when exact
  Mode A parity matters; f32-m/z is an opt-in for ML / size where sub-ppm exact
  matching is not required.

Both consume the normalized ``XIC_TARGET_TABLE`` (see
``massspec.quant.targets``) and emit ``XIC_TRACE_TABLE`` (the deliberate
terminal artifact — trace → PRECURSOR_QUANT reduction is a separate
downstream step). Match semantics default to nearest-peak (one row per
(target, scan, ion); rectangular). ``all_in_window=True`` returns every
peak within tolerance (interference / chimeric / DIA work).

Query m/z come from the existing peptide chemistry: MS1 isotope envelopes
via ``peptide_envelope`` (or isotopes off a bare ``precursor_mz`` for the
no-peptide m/z case), MS2 fragment ladders via the batched
``fragment_ladder_indices_batch``. Mass error is conventional (observed -
theoretical), signed, preserved so probabilistic scoring stays downstream.

NOTE: Mode B still reads index partitions per target. MS1
``peptide_envelope`` is called per (sequence, charge) rather than batched —
a future optimization, not a correctness issue.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.core.signal import bounds_within_tolerance
from constellation.core.stats.units import ISOTOPE_MASS_DIFF
from constellation.massspec.bundle import open_peaks
from constellation.massspec.peptide.envelope import peptide_envelope
from constellation.massspec.peptide.ions import IonType, fragment_ladder_indices_batch
from constellation.massspec.quant.peak_index import read_manifest as read_index_manifest
from constellation.massspec.quant.schemas import XIC_TRACE_TABLE

if TYPE_CHECKING:
    from constellation.massspec.sweep import IonSettings
    from constellation.massspec.sweep.engine import SweepResult
    from constellation.massspec.sweep.evidence import EvidenceGate
    from constellation.massspec.sweep.reduce import QueryBatch

__all__ = [
    "extract_xic",
    "extraction_tolerance_metadata",
    "extract_xic_scan_major",
    "extract_xic_indexed",
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


def _peptidoform(modseq: str) -> Peptidoform:
    p = parse_proforma(modseq)
    if not isinstance(p, Peptidoform):
        raise ValueError(
            f"XIC extraction supports linear peptidoforms; {modseq!r} parsed to "
            f"{type(p).__name__}"
        )
    return p


def _ms1_queries(
    t: _Target,
    *,
    n_isotopes: int,
    charge_range: tuple[int, int] | None,
) -> list[_QueryIon]:
    if t.modified_sequence:
        pf = _peptidoform(t.modified_sequence)
        if charge_range is not None:
            charges = list(range(charge_range[0], charge_range[1] + 1))
        elif t.precursor_charge is not None:
            charges = [int(t.precursor_charge)]
        else:
            raise ValueError(
                f"target {t.target_id}: MS1 extraction needs a charge "
                "(charge_range or precursor_charge)"
            )
        out: list[_QueryIon] = []
        for z in charges:
            mz_iso, _ = peptide_envelope(pf, charge=z, n_peaks=n_isotopes)
            for k, mz in enumerate(mz_iso.tolist()):
                out.append(_QueryIon(mz=mz, precursor_charge=z, isotope=k))
        return out
    # No-peptide m/z target: isotopes off the bare precursor m/z.
    if t.precursor_mz is None:
        raise ValueError(
            f"target {t.target_id}: needs modified_sequence or precursor_mz"
        )
    z = int(t.precursor_charge) if t.precursor_charge is not None else 1
    n = n_isotopes if t.precursor_charge is not None else 1
    return [
        _QueryIon(
            mz=t.precursor_mz + k * ISOTOPE_MASS_DIFF / z, precursor_charge=z, isotope=k
        )
        for k in range(n)
    ]


def _ms2_queries_batch(
    targets: list[_Target],
    *,
    ion_types: Sequence[IonType],
    max_fragment_charge: int,
    neutral_losses: Sequence[str] | None,
) -> dict[int, list[_QueryIon]]:
    """Build MS2 fragment queries for all targets in one batched ladder call."""
    idxs, peptidoforms = [], []
    for i, t in enumerate(targets):
        if not t.modified_sequence:
            raise ValueError(
                f"target {t.target_id}: MS2 fragment extraction requires "
                "modified_sequence (a bare-m/z target cannot define fragments)"
            )
        if t.precursor_charge is None:
            raise ValueError(
                f"target {t.target_id}: MS2 extraction requires precursor_charge"
            )
        idxs.append(i)
        peptidoforms.append(_peptidoform(t.modified_sequence))

    ladders = fragment_ladder_indices_batch(
        peptidoforms,
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
        neutral_losses=neutral_losses,
    )
    out: dict[int, list[_QueryIon]] = {}
    for i, ladder in zip(idxs, ladders):
        t = targets[i]
        ceiling = min(max_fragment_charge, int(t.precursor_charge))
        ions = [
            _QueryIon(
                mz=mz,
                precursor_charge=int(t.precursor_charge),
                ion_type=ion_type,
                position=pos,
                fragment_charge=ch,
                loss_id=loss_id,
            )
            for (ion_type, pos, ch, loss_id), mz in ladder.items()
            if ch <= ceiling
        ]
        out[i] = ions
    return out


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
    input order, then (rt, scan), then the target's ions in the order its
    queries list them, then peak. With ``drop_unmatched=False`` every cell
    the sweep could have matched is a row, intensity 0.0 and NaN m/z and
    errors where nothing was within tolerance (NaN, not null, as always).
    A rule that can match several peaks per cell (``all``) gives matches
    only."""

    drop_unmatched: bool = True
    name: str = "xic_trace"
    schema: pa.Schema = XIC_TRACE_TABLE
    #: Only the queries the sweep's evidence gate passed.
    gated: bool = False

    def reduce(self, batch: "QueryBatch") -> pa.Table:
        plan, s, m = batch.plan, batch.scans, batch.matches
        rows = batch.match_rows()
        fill = not self.drop_unmatched and batch.settings.rule.name != "all"
        parts = []
        for level in plan.levels:
            ions = plan.ions_for_level(level)
            if ions is None:
                continue
            at = np.flatnonzero(m.level == level)
            q, r, ion = m.query[at], rows[at], m.ion[at]
            peak, obs, inten = m.peak_index[at], m.mz[at], m.intensity[at]
            matched = np.ones(at.size, dtype=bool)
            if fill:
                q, r, ion, idx = _left_join_cells(batch, level, q, r, ion)
                matched = idx >= 0
                peak, obs, inten = (
                    np.where(matched, np.r_[v, fallback][idx], fallback)
                    for v, fallback in ((peak, -1), (obs, np.nan), (inten, 0.0))
                )
            parts.append((level, ions, q, r, ion, peak, obs, inten, matched))
        return _trace_table(plan, s, batch.scan_order, parts, batch.settings.acquisition_id)


def _left_join_cells(batch, level, q, r, ion):
    """Every eligible cell of ``level`` in output order, with the index of
    its match among ``(q, r, ion)`` (-1 where none)."""
    cq, cr, ci = batch.eligible_cells(level)
    order = batch.scan_order
    by_cell = np.lexsort((ci, order[cr], cq))
    by_match = np.lexsort((ion, order[r], q))
    key_c = np.stack([cq[by_cell], order[cr][by_cell], ci[by_cell]], axis=1)
    key_m = np.stack([q[by_match], order[r][by_match], ion[by_match]], axis=1)
    idx = np.full(cq.size, -1, dtype=np.int64)
    if key_m.size:
        # Both sides sorted on the same key; a match is one of the cells.
        view = np.dtype([("q", np.int64), ("s", np.int64), ("i", np.int64)])
        kc = np.ascontiguousarray(key_c).view(view).reshape(-1)
        km = np.ascontiguousarray(key_m).view(view).reshape(-1)
        pos = np.searchsorted(kc, km)
        idx[by_cell[pos]] = by_match
    return cq, cr, ci, idx


def _trace_table(plan, s, scan_order, parts, acquisition_id) -> pa.Table:
    if not parts:
        return cast_to_schema(XIC_TRACE_TABLE.empty_table(), XIC_TRACE_TABLE)
    cols: dict[str, list] = {k: [] for k in (
        "query", "level", "row", "ion_order", "peak", "charge", "isotope", "ion_type",
        "position", "fragment_charge", "loss_id", "theo", "obs", "intensity", "matched",
    )}  # fmt: skip
    for level, ions, q, r, ion, peak, obs, inten, matched in parts:
        g = ions.offsets[q] + ion
        n = q.size
        frag = level != 1
        cols["query"].append(q)
        cols["level"].append(np.full(n, level, np.int8))
        cols["row"].append(r)
        cols["ion_order"].append(ion)
        cols["peak"].append(peak)
        cols["charge"].append(ions.precursor_charge[g])
        cols["isotope"].append(np.ma.masked_array(ions.isotope[g], mask=np.full(n, frag)))
        for name, values in (("ion_type", ions.ion_type), ("position", ions.position),
                             ("fragment_charge", ions.fragment_charge)):  # fmt: skip
            cols[name].append(np.ma.masked_array(values[g], mask=np.full(n, not frag)))
        names = np.array([None, *ions.loss_ids], dtype=object)
        cols["loss_id"].append(names[ions.loss[g]] if frag else np.full(n, None, dtype=object))
        cols["theo"].append(ions.mz[g])
        cols["obs"].append(obs)
        cols["intensity"].append(inten)
        cols["matched"].append(matched)
    c = {k: (np.ma.concatenate(v) if isinstance(v[0], np.ma.MaskedArray) else np.concatenate(v))
         for k, v in cols.items()}  # fmt: skip
    by = np.lexsort((c["peak"], c["ion_order"], scan_order[c["row"]], c["level"], c["query"]))
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
    ions: "IonSettings | None" = None,
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
) -> "SweepResult":
    """Extract XIC traces for ``targets`` at every MS level in ``levels`` in
    one sweep of ``peaks`` (a bundle, streamed, or a peak table), each level
    at its own tolerance. The trace is the result's ``xic_trace`` output:
    ``.table("xic_trace")`` for it in memory, ``.write("xic_trace", path)``
    to stream it to a file. ``rule`` is ``nearest`` (Mode A's), ``all`` (every
    peak within tolerance) or ``sum`` (their sum, as one row).

    With a ``gate``, ``only_passing`` keeps only the targets it passed in
    the trace. ``claims`` adds the ``peak_claims`` output (peaks matched by
    more than one target or ion) and ``summary`` the ``query_summary``
    output (per target: the gate's verdict, scans eligible, ions matched)."""
    from constellation.massspec.sweep import IonSettings, compile_queries
    from constellation.massspec.sweep.engine import SweepSettings, sweep
    from constellation.massspec.sweep.reduce import QuerySummaryReducer
    from constellation.massspec.sweep.rules import rule_named

    if only_passing and gate is None:
        raise ValueError("only_passing needs a gate to pass targets")

    plan = compile_queries(
        targets,
        levels=levels,
        ions=ions if ions is not None else IonSettings(),
        rt_window=rt_window,
        assigned_scans_only=assigned_scans_only,
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
    return sweep(peaks, plan, settings, reducers=reducers, workers=workers, work_dir=work_dir,
                 progress=progress)  # fmt: skip


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
    charge_range: tuple[int, int] | None = (1, 4),
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
    per (scan, ion) instead of the nearest (interference / chimeric work).
    ``mz_axis`` (a bundle's measured m/z axis key, or ``latest``) corrects
    the observed m/z before matching. ``workers > 1`` sweeps a bundle's scan
    ranges in that many processes; the rows are the same.
    """
    from constellation.massspec.sweep import IonSettings

    result = extract_xic(
        peaks,
        targets,
        acquisition_id=acquisition_id,
        levels=[level],
        tolerances={level: (tolerance, tolerance_unit)},
        ions=IonSettings(
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
    return _finalize(
        _stamp_extraction_tolerance(result.table("xic_trace"), tolerance, tolerance_unit),
        output_path,
    )


# ──────────────────────────────────────────────────────────────────────
# Mode B — mass-index slice
# ──────────────────────────────────────────────────────────────────────


def _select_partitions(
    manifest, level: int, precursor_mz: float | None, dda_width: float
):
    """Partition dirs (level, wid) whose window could cover the precursor.

    For binned (DDA) levels the bin id is ``floor(bin_value / dda_width)``
    where ``bin_value`` is the scan's own precursor m/z (or its window center
    when absent). A covering scan's narrow window can straddle the target
    precursor from several bins away, so the neighbor span must cover the
    worst-case bin distance: the scan's ``bin_value`` and the target precursor
    both lie inside that scan's window (≤ ``2 * max_halfwidth`` apart), so
    ``span = ceil(2 * max_halfwidth / dda_width) + 1`` (the +1 absorbs the
    floor boundary) guarantees every scan Mode A would gate in is loaded. The
    per-row ``_scan_gate`` then filters on the real bounds, so over-selecting
    bins only costs I/O, never correctness.
    """
    chosen = []
    binned_span = _binned_span(manifest, level, dda_width)
    for p in manifest.partitions:
        if p["level"] != level:
            continue
        if level == 1 or precursor_mz is None:
            chosen.append((p["level"], p["isolation_window_id"]))
            continue
        scheme = manifest.window_scheme.get(str(level), "native")
        if scheme == "binned":
            target_bin = int(precursor_mz // dda_width)
            if abs(p["isolation_window_id"] - target_bin) <= binned_span:
                chosen.append((p["level"], p["isolation_window_id"]))
        else:
            lo, hi = p["isolation_lower"], p["isolation_upper"]
            if lo is None or hi is None or (lo <= precursor_mz <= hi):
                chosen.append((p["level"], p["isolation_window_id"]))
    return chosen


def _binned_span(manifest, level: int, dda_width: float) -> int:
    """Bin-neighbor radius for a binned level (see :func:`_select_partitions`)."""
    halfwidths = getattr(manifest, "dda_max_halfwidth_mz", {}) or {}
    max_hw = float(halfwidths.get(str(level), 0.0))
    if max_hw <= 0.0 or dda_width <= 0.0:
        return 1
    return int(math.ceil(2.0 * max_hw / dda_width)) + 1


def extract_xic_indexed(
    index_root: Path | str,
    targets: pa.Table,
    *,
    acquisition_id: int,
    level: int,
    n_isotopes: int = 3,
    ion_types: Sequence[IonType] = _DEFAULT_ION_TYPES,
    max_fragment_charge: int = 2,
    neutral_losses: Sequence[str] | None = None,
    charge_range: tuple[int, int] | None = (1, 4),
    rt_window: float | None = None,
    tolerance: float = 20.0,
    tolerance_unit: ToleranceUnit = "ppm",
    all_in_window: bool = False,
    assigned_scans_only: bool = False,
    drop_unmatched: bool = True,
    exact_error: bool = False,
    peaks_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pa.Table | Path:
    """Mode B mass-index XIC extraction. See module docstring."""
    index_root = Path(index_root)
    manifest = read_index_manifest(index_root)
    if exact_error and _index_mz_is_f32(manifest) and peaks_dir is None:
        raise ValueError(
            "exact_error=True needs peaks_dir to re-derive f64 m/z error from the "
            "canonical peak table on an f32-downcast index"
        )
    dda_width = float(manifest.parameters.get("dda_window_width_mz", 10.0))
    records = _targets_to_records(targets)
    builder = _TraceBuilder()

    part_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def load_partition(lvl: int, wid: int) -> dict[str, np.ndarray] | None:
        key = (lvl, wid)
        if key in part_cache:
            return part_cache[key]
        pdir = index_root / f"ms_level={lvl}" / f"isolation_window={wid}"
        f = pdir / "part-00000.parquet"
        if not f.exists():
            part_cache[key] = None
            return None
        t = pq.read_table(f)
        cols = {
            "mz": t.column("mz").to_numpy(zero_copy_only=False).astype(np.float64),
            "rt": t.column("rt").to_numpy(zero_copy_only=False).astype(np.float64),
            "scan": t.column("scan").to_numpy(zero_copy_only=False).astype(np.int64),
            "intensity": t.column("intensity")
            .to_numpy(zero_copy_only=False)
            .astype(np.float64),
            "iso_lo": t.column("isolation_lower").to_numpy(zero_copy_only=False),
            "iso_hi": t.column("isolation_upper").to_numpy(zero_copy_only=False),
        }
        part_cache[key] = cols
        return cols

    ms2_queries = (
        _ms2_queries_batch(
            records,
            ion_types=ion_types,
            max_fragment_charge=max_fragment_charge,
            neutral_losses=neutral_losses,
        )
        if level != 1
        else {}
    )

    for i, t in enumerate(records):
        ions = (
            ms2_queries[i]
            if level != 1
            else _ms1_queries(t, n_isotopes=n_isotopes, charge_range=charge_range)
        )
        if not ions:
            continue
        rt_lo, rt_hi = _rt_bounds(t, rt_window)
        parts = _select_partitions(manifest, level, t.precursor_mz, dda_width)
        _extract_target_indexed(
            builder,
            t,
            level,
            ions,
            parts,
            load_partition,
            rt_lo,
            rt_hi,
            tolerance,
            tolerance_unit,
            assigned_scans_only,
            drop_unmatched,
            acquisition_id,
            all_in_window,
        )

    table = builder.table()
    if exact_error and _index_mz_is_f32(manifest) and peaks_dir is not None:
        table = _enrich_exact_error(table, peaks_dir, level)
    return _finalize(
        _stamp_extraction_tolerance(table, tolerance, tolerance_unit), output_path
    )


def _extract_target_indexed(
    builder,
    t,
    level,
    ions,
    parts,
    load_partition,
    rt_lo,
    rt_hi,
    tolerance,
    tolerance_unit,
    assigned_scans_only,
    drop_unmatched,
    acquisition_id,
    all_in_window=False,
) -> None:
    # Gather candidate peaks across partitions once.
    mz_parts, rt_parts, scan_parts, int_parts, lo_parts, hi_parts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for lvl, wid in parts:
        cols = load_partition(lvl, wid)
        if cols is None:
            continue
        mz_parts.append(cols["mz"])
        rt_parts.append(cols["rt"])
        scan_parts.append(cols["scan"])
        int_parts.append(cols["intensity"])
        lo_parts.append(cols["iso_lo"])
        hi_parts.append(cols["iso_hi"])
    if not mz_parts:
        return
    p_mz = np.concatenate(mz_parts)
    order = np.argsort(p_mz, kind="stable")  # merge of per-partition sorted arrays
    p_mz = p_mz[order]
    p_rt = np.concatenate(rt_parts)[order]
    p_scan = np.concatenate(scan_parts)[order]
    p_int = np.concatenate(int_parts)[order]
    p_lo = np.concatenate(lo_parts)[order]
    p_hi = np.concatenate(hi_parts)[order]

    # Scan-level gate (rt window + isolation coverage + assigned scan) over
    # ALL loaded partition rows. rt / iso / scan are per-scan properties, so a
    # scan is a *candidate* iff any of its peaks pass — this reconstructs the
    # exact gated scan set Mode A's `_candidate_scans` produces, independent of
    # query m/z, so `drop_unmatched=False` can zero-fill unmatched cells.
    scan_gate = _scan_gate(
        p_rt, p_lo, p_hi, p_scan, t, level, rt_lo, rt_hi, assigned_scans_only
    )
    if not scan_gate.any():
        return
    uscan, ui = np.unique(p_scan[scan_gate], return_index=True)
    g_rt, g_lo, g_hi = p_rt[scan_gate], p_lo[scan_gate], p_hi[scan_gate]
    cand_rt, cand_lo, cand_hi = g_rt[ui], g_lo[ui], g_hi[ui]
    scan_pos = {int(s): i for i, s in enumerate(uscan)}

    axis = torch.from_numpy(p_mz)
    query_mz = np.array([ion.mz for ion in ions], dtype=np.float64)
    lo_idx, hi_idx = bounds_within_tolerance(
        axis, torch.from_numpy(query_mz), tolerance, tolerance_unit
    )
    lo_idx = lo_idx.numpy()
    hi_idx = hi_idx.numpy()

    if all_in_window:
        _emit_all_in_window_indexed(
            builder,
            t,
            level,
            ions,
            query_mz,
            lo_idx,
            hi_idx,
            p_mz,
            p_rt,
            p_scan,
            p_int,
            p_lo,
            p_hi,
            scan_pos,
            rt_lo,
            rt_hi,
            assigned_scans_only,
            acquisition_id,
        )
        return

    # Nearest per scan: matched_by_qi[qi] = {scan: (observed_mz, intensity)}.
    matched_by_qi: dict[int, dict[int, tuple[float, float]]] = {}
    for qi, ion in enumerate(ions):
        a, b = int(lo_idx[qi]), int(hi_idx[qi])
        if b <= a:
            continue
        sl = slice(a, b)
        rgate = _scan_gate(
            p_rt[sl],
            p_lo[sl],
            p_hi[sl],
            p_scan[sl],
            t,
            level,
            rt_lo,
            rt_hi,
            assigned_scans_only,
        )
        if not rgate.any():
            continue
        w_mz, w_scan, w_int = p_mz[sl][rgate], p_scan[sl][rgate], p_int[sl][rgate]
        sgather = _argmin_per_group(w_scan, np.abs(w_mz - ion.mz))
        per_scan = matched_by_qi.setdefault(qi, {})
        for k in sgather:
            per_scan[int(w_scan[k])] = (float(w_mz[k]), float(w_int[k]))

    if drop_unmatched and not matched_by_qi:
        return

    out_qi, out_scan, out_obs, out_int = [], [], [], []
    out_rt, out_lo, out_hi = [], [], []
    for qi in range(len(ions)):
        per_scan = matched_by_qi.get(qi, {})
        scans = per_scan.keys() if drop_unmatched else (int(s) for s in uscan)
        for s in scans:
            obs, inten = per_scan.get(s, (np.nan, 0.0))
            pos = scan_pos[s]
            out_qi.append(qi)
            out_scan.append(s)
            out_rt.append(cand_rt[pos])
            out_lo.append(cand_lo[pos])
            out_hi.append(cand_hi[pos])
            out_obs.append(obs)
            out_int.append(inten)
    if not out_qi:
        return
    qi_arr = np.array(out_qi, dtype=np.int64)
    obs = np.array(out_obs, dtype=np.float64)
    theo = query_mz[qi_arr]
    matched = ~np.isnan(obs)
    err_da = np.where(matched, obs - theo, np.nan)
    builder.add(
        acquisition_id=acquisition_id,
        target_id=t.target_id,
        modified_sequence=t.modified_sequence,
        precursor_charge=np.array(
            [ions[q].precursor_charge for q in qi_arr], dtype=np.int64
        ),
        level=level,
        scan=np.array(out_scan, dtype=np.int64),
        rt=np.array(out_rt, dtype=np.float64),
        isolation_lower=np.array(out_lo, dtype=np.float64),
        isolation_upper=np.array(out_hi, dtype=np.float64),
        isotope=[ions[q].isotope for q in qi_arr],
        ion_type=[ions[q].ion_type for q in qi_arr],
        position=[ions[q].position for q in qi_arr],
        fragment_charge=[ions[q].fragment_charge for q in qi_arr],
        loss_id=[ions[q].loss_id for q in qi_arr],
        mz_theoretical=theo,
        mz_observed=obs,
        intensity=np.array(out_int, dtype=np.float64),
        mz_error_da=err_da,
        mz_error_ppm=np.where(matched, err_da / theo * 1e6, np.nan),
    )


def _scan_gate(
    rt, iso_lo, iso_hi, scan, t, level, rt_lo, rt_hi, assigned_scans_only
) -> np.ndarray:
    """Per-row mask for the scan-level gate (rt window + MS2 isolation
    coverage + assigned scan). Shared by Mode B candidate-scan discovery and
    per-window-slice filtering so both apply identical semantics."""
    gate = (rt >= rt_lo) & (rt <= rt_hi)
    if level != 1 and t.precursor_mz is not None:
        covers = (
            ~(np.isnan(iso_lo) | np.isnan(iso_hi))
            & (iso_lo <= t.precursor_mz)
            & (t.precursor_mz <= iso_hi)
        )
        passall = np.isnan(iso_lo) | np.isnan(iso_hi)
        gate = gate & (covers | passall)
    if assigned_scans_only and t.scan is not None:
        gate = gate & (scan == t.scan)
    return gate


def _emit_all_in_window_indexed(
    builder,
    t,
    level,
    ions,
    query_mz,
    lo_idx,
    hi_idx,
    p_mz,
    p_rt,
    p_scan,
    p_int,
    p_lo,
    p_hi,
    scan_pos,
    rt_lo,
    rt_hi,
    assigned_scans_only,
    acquisition_id,
) -> None:
    """All-peaks-in-window emit for Mode B (no zero-fill — mirrors Mode A's
    `_emit_all_scan_major`, where `drop_unmatched` does not apply)."""
    out_qi, out_scan, out_rt, out_lo, out_hi, out_obs, out_int = ([] for _ in range(7))
    for qi, ion in enumerate(ions):
        a, b = int(lo_idx[qi]), int(hi_idx[qi])
        if b <= a:
            continue
        sl = slice(a, b)
        rgate = _scan_gate(
            p_rt[sl],
            p_lo[sl],
            p_hi[sl],
            p_scan[sl],
            t,
            level,
            rt_lo,
            rt_hi,
            assigned_scans_only,
        )
        if not rgate.any():
            continue
        out_qi.append(np.full(int(rgate.sum()), qi, dtype=np.int64))
        out_scan.append(p_scan[sl][rgate])
        out_rt.append(p_rt[sl][rgate])
        out_lo.append(p_lo[sl][rgate])
        out_hi.append(p_hi[sl][rgate])
        out_obs.append(p_mz[sl][rgate])
        out_int.append(p_int[sl][rgate])
    if not out_qi:
        return
    qi_arr = np.concatenate(out_qi)
    obs = np.concatenate(out_obs)
    theo = query_mz[qi_arr]
    err_da = obs - theo
    builder.add(
        acquisition_id=acquisition_id,
        target_id=t.target_id,
        modified_sequence=t.modified_sequence,
        precursor_charge=np.array(
            [ions[q].precursor_charge for q in qi_arr], dtype=np.int64
        ),
        level=level,
        scan=np.concatenate(out_scan),
        rt=np.concatenate(out_rt),
        isolation_lower=np.concatenate(out_lo),
        isolation_upper=np.concatenate(out_hi),
        isotope=[ions[q].isotope for q in qi_arr],
        ion_type=[ions[q].ion_type for q in qi_arr],
        position=[ions[q].position for q in qi_arr],
        fragment_charge=[ions[q].fragment_charge for q in qi_arr],
        loss_id=[ions[q].loss_id for q in qi_arr],
        mz_theoretical=theo,
        mz_observed=obs,
        intensity=np.concatenate(out_int),
        mz_error_da=err_da,
        mz_error_ppm=err_da / theo * 1e6,
    )


def _argmin_per_group(group: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Row indices of the minimum ``value`` within each distinct ``group``."""
    order = np.lexsort((value, group))
    g_sorted = group[order]
    first = np.concatenate(([True], g_sorted[1:] != g_sorted[:-1]))
    return order[first]


def _index_mz_is_f32(manifest) -> bool:
    return manifest.precision.get("mz") == "f32"


def _enrich_exact_error(table: pa.Table, peaks_dir: Path | str, level: int) -> pa.Table:
    """Re-derive f64 mz_observed / mz_error for matched survivors from the
    canonical MS_PEAK_TABLE (joins on (scan, nearest mz))."""
    matched = pc.is_valid(table.column("mz_observed"))
    if not pc.any(matched).as_py():
        return table
    src = peaks_dir
    dataset = open_peaks(peaks_dir)
    canon = dataset.to_table(
        columns=["scan", "level", "mz", "intensity"],
        filter=ds.field("level") == level,
    )
    # Per (scan), nearest canonical mz to each row's theoretical mz.
    canon_by_scan: dict[int, np.ndarray] = {}
    cs = canon.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
    cm = canon.column("mz").to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.lexsort((cm, cs))
    cs, cm = cs[order], cm[order]
    starts = np.concatenate(([0], np.flatnonzero(cs[1:] != cs[:-1]) + 1, [cs.shape[0]]))
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        canon_by_scan[int(cs[a])] = cm[a:b]

    scan_col = table.column("scan").to_numpy(zero_copy_only=False)
    theo_col = table.column("mz_theoretical").to_numpy(zero_copy_only=False)
    obs_col = (
        table.column("mz_observed").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    new_obs = obs_col.copy()
    for r in range(table.num_rows):
        if np.isnan(obs_col[r]):
            continue
        arr = canon_by_scan.get(int(scan_col[r]))
        if arr is None or arr.size == 0:
            continue
        j = int(np.argmin(np.abs(arr - theo_col[r])))
        new_obs[r] = arr[j]
    err = new_obs - theo_col
    out = table.set_column(
        table.schema.get_field_index("mz_observed"),
        "mz_observed",
        pa.array(new_obs, pa.float64()),
    )
    out = out.set_column(
        out.schema.get_field_index("mz_error_da"),
        "mz_error_da",
        pa.array(np.where(np.isnan(new_obs), np.nan, err), pa.float64()),
    )
    out = out.set_column(
        out.schema.get_field_index("mz_error_ppm"),
        "mz_error_ppm",
        pa.array(
            np.where(np.isnan(new_obs), np.nan, err / theo_col * 1e6), pa.float64()
        ),
    )
    _ = src
    return out
