"""Frozen copy of the target-major Mode A XIC extractor, as it stood before
the sweep replaced it (``massspec.quant.chromatogram`` at ``79f54dd``).

A test oracle only: the sweep's trace reducer must reproduce this
function's output row for row, in the same order, with the same NaN
positions and the same float bits. Do not edit it to follow the package;
it is the behaviour being matched. Copied verbatim except for the module
docstring, the imports, and leaving out ``save_xic`` / ``load_xic`` and
Mode B.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.core.signal import bounds_within_tolerance, nearest_within_tolerance
from constellation.core.stats.units import ISOTOPE_MASS_DIFF
from constellation.massspec.peptide.envelope import peptide_envelope
from constellation.massspec.peptide.ions import IonType, fragment_ladder_indices_batch
from constellation.massspec.quant.schemas import XIC_TRACE_TABLE

__all__ = ["extract_xic_scan_major"]

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


# ──────────────────────────────────────────────────────────────────────
# Mode A — scan-major sweep
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _ScanIndex:
    """Per-scan peak store for one MS level: scans in RT order, peaks
    mz-sorted within each scan, addressed by offsets."""

    scan_ids: np.ndarray
    rts: np.ndarray
    iso_lo: np.ndarray
    iso_hi: np.ndarray
    flat_mz: np.ndarray
    flat_int: np.ndarray
    offsets: np.ndarray  # (n_scans + 1,)


def _build_scan_index(peaks: pa.Table, level: int) -> _ScanIndex:
    ms = peaks.filter(pc.equal(peaks["level"], level))
    ms = ms.sort_by([("rt", "ascending"), ("scan", "ascending"), ("mz", "ascending")])
    if ms.num_rows == 0:
        empty = np.array([], dtype=np.float64)
        return _ScanIndex(
            np.array([], np.int64),
            empty,
            empty,
            empty,
            empty,
            empty,
            np.array([0], np.int64),
        )
    scan = ms.column("scan").to_numpy(zero_copy_only=False).astype(np.int64)
    rt = ms.column("rt").to_numpy(zero_copy_only=False)
    mz = ms.column("mz").to_numpy(zero_copy_only=False)
    inten = ms.column("intensity").to_numpy(zero_copy_only=False)
    iso_lo = ms.column("isolation_lower").to_numpy(zero_copy_only=False)
    iso_hi = ms.column("isolation_upper").to_numpy(zero_copy_only=False)

    # Scan boundaries on the rt-then-scan sort.
    change = np.flatnonzero(scan[1:] != scan[:-1]) + 1
    starts = np.concatenate(([0], change))
    offsets = np.concatenate((starts, [scan.shape[0]])).astype(np.int64)
    return _ScanIndex(
        scan_ids=scan[starts],
        rts=rt[starts],
        iso_lo=iso_lo[starts],
        iso_hi=iso_hi[starts],
        flat_mz=mz,
        flat_int=inten,
        offsets=offsets,
    )


def _candidate_scans(
    idx: _ScanIndex,
    t: _Target,
    level: int,
    rt_window: float | None,
    assigned_scans_only: bool,
) -> np.ndarray:
    """Indices (into idx.scan_ids) of scans gating in for this target."""
    if idx.scan_ids.size == 0:
        return np.array([], dtype=np.int64)
    lo, hi = _rt_bounds(t, rt_window)
    left = bisect.bisect_left(idx.rts.tolist(), lo)
    right = bisect.bisect_right(idx.rts.tolist(), hi)
    cand = np.arange(left, right, dtype=np.int64)
    if cand.size == 0:
        return cand
    # MS2 isolation-window gating: keep scans whose window covers the precursor.
    if level != 1 and t.precursor_mz is not None:
        clo = idx.iso_lo[cand]
        chi = idx.iso_hi[cand]
        covers = (
            ~(np.isnan(clo) | np.isnan(chi))
            & (clo <= t.precursor_mz)
            & (t.precursor_mz <= chi)
        )
        # null bounds pass-all
        passall = np.isnan(clo) | np.isnan(chi)
        cand = cand[covers | passall]
    if assigned_scans_only and t.scan is not None:
        cand = cand[idx.scan_ids[cand] == t.scan]
    return cand


def _match_grid(
    idx: _ScanIndex,
    cand: np.ndarray,
    query_mz: np.ndarray,
    tolerance: float,
    tolerance_unit: ToleranceUnit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-within-tolerance match of queries against each candidate
    scan. Returns (observed_mz, intensity, within) each (n_cand, n_q)."""
    n_cand = cand.shape[0]
    counts = (idx.offsets[cand + 1] - idx.offsets[cand]).astype(np.int64)
    max_p = int(counts.max()) if n_cand else 0
    obs_mz = np.full((n_cand, max_p), np.inf, dtype=np.float64)
    obs_int = np.zeros((n_cand, max_p), dtype=np.float64)
    for j, s in enumerate(cand):
        a, b = int(idx.offsets[s]), int(idx.offsets[s] + counts[j])
        obs_mz[j, : counts[j]] = idx.flat_mz[a:b]
        obs_int[j, : counts[j]] = idx.flat_int[a:b]

    q = torch.from_numpy(query_mz)
    axis = torch.from_numpy(obs_mz)
    queries = q.unsqueeze(0).expand(n_cand, -1)
    best_idx, within, signed = nearest_within_tolerance(
        axis, queries, tolerance, tolerance_unit
    )
    inten = torch.from_numpy(obs_int).gather(1, best_idx)
    # observed mz = query - signed_delta (signed_delta = query - observed)
    observed = (queries.to(torch.float64) - signed).numpy()
    return observed, inten.numpy(), within.numpy()


def extract_xic_scan_major(
    peaks: pa.Table,
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
    output_path: Path | str | None = None,
) -> pa.Table | Path:
    """Mode A scan-major XIC extraction. See module docstring.

    ``all_in_window=True`` returns every peak within tolerance per
    (scan, ion) instead of the nearest (interference / chimeric work).
    """
    records = _targets_to_records(targets)
    idx = _build_scan_index(peaks, level)
    builder = _TraceBuilder()

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
        cand = _candidate_scans(idx, t, level, rt_window, assigned_scans_only)
        if cand.size == 0:
            continue
        if all_in_window:
            _emit_all_scan_major(
                builder,
                acquisition_id,
                t,
                level,
                ions,
                idx,
                cand,
                tolerance,
                tolerance_unit,
            )
            continue
        query_mz = np.array([ion.mz for ion in ions], dtype=np.float64)
        observed, inten, within = _match_grid(
            idx, cand, query_mz, tolerance, tolerance_unit
        )
        _emit_grid(
            builder,
            acquisition_id=acquisition_id,
            target=t,
            level=level,
            ions=ions,
            scan_ids=idx.scan_ids[cand],
            rts=idx.rts[cand],
            iso_lo=idx.iso_lo[cand],
            iso_hi=idx.iso_hi[cand],
            observed_mz=observed,
            intensity=inten,
            within=within,
            drop_unmatched=drop_unmatched,
        )
    return _finalize(
        _stamp_extraction_tolerance(builder.table(), tolerance, tolerance_unit),
        output_path,
    )


def _build_obs_matrices(
    idx: _ScanIndex, cand: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = (idx.offsets[cand + 1] - idx.offsets[cand]).astype(np.int64)
    max_p = int(counts.max()) if cand.size else 0
    obs_mz = np.full((cand.shape[0], max_p), np.inf, dtype=np.float64)
    obs_int = np.zeros((cand.shape[0], max_p), dtype=np.float64)
    for j, s in enumerate(cand):
        a, b = int(idx.offsets[s]), int(idx.offsets[s] + counts[j])
        obs_mz[j, : counts[j]] = idx.flat_mz[a:b]
        obs_int[j, : counts[j]] = idx.flat_int[a:b]
    return obs_mz, obs_int, counts


def _emit_all_scan_major(
    builder, acquisition_id, t, level, ions, idx, cand, tolerance, tolerance_unit
) -> None:
    """All-peaks-in-window emit for one target (scan-major)."""
    obs_mz, obs_int, _ = _build_obs_matrices(idx, cand)
    query_mz = np.array([ion.mz for ion in ions], dtype=np.float64)
    queries = torch.from_numpy(query_mz).unsqueeze(0).expand(cand.shape[0], -1)
    lo, hi = bounds_within_tolerance(
        torch.from_numpy(obs_mz), queries, tolerance, tolerance_unit
    )
    lo, hi = lo.numpy(), hi.numpy()
    o_scan, o_rt, o_lo, o_hi, o_obs, o_int, o_qi = ([] for _ in range(7))
    for j in range(cand.shape[0]):
        s = cand[j]
        for q in range(len(ions)):
            for pk in range(int(lo[j, q]), int(hi[j, q])):
                m = obs_mz[j, pk]
                if not np.isfinite(m):
                    continue
                o_scan.append(idx.scan_ids[s])
                o_rt.append(idx.rts[s])
                o_lo.append(idx.iso_lo[s])
                o_hi.append(idx.iso_hi[s])
                o_obs.append(m)
                o_int.append(obs_int[j, pk])
                o_qi.append(q)
    if not o_qi:
        return
    qi = np.array(o_qi, dtype=np.int64)
    obs = np.array(o_obs, dtype=np.float64)
    theo = query_mz[qi]
    err = obs - theo
    builder.add(
        acquisition_id=acquisition_id,
        target_id=t.target_id,
        modified_sequence=t.modified_sequence,
        precursor_charge=np.array(
            [ions[q].precursor_charge for q in qi], dtype=np.int64
        ),
        level=level,
        scan=np.array(o_scan),
        rt=np.array(o_rt),
        isolation_lower=np.array(o_lo),
        isolation_upper=np.array(o_hi),
        isotope=[ions[q].isotope for q in qi],
        ion_type=[ions[q].ion_type for q in qi],
        position=[ions[q].position for q in qi],
        fragment_charge=[ions[q].fragment_charge for q in qi],
        loss_id=[ions[q].loss_id for q in qi],
        mz_theoretical=theo,
        mz_observed=obs,
        intensity=np.array(o_int),
        mz_error_da=err,
        mz_error_ppm=err / theo * 1e6,
    )


# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────


def assert_traces_identical(actual: pa.Table, expected: pa.Table) -> None:
    """Assert two XIC trace tables are the same table: schema and schema
    metadata, row order, null bitmaps, and every float's bit pattern (so a
    NaN where the other has a null, or ``-0.0`` for ``0.0``, is a difference)."""
    assert actual.schema.equals(expected.schema, check_metadata=True), (
        f"schemas differ:\n{actual.schema}\nvs\n{expected.schema}"
    )
    assert actual.num_rows == expected.num_rows, (
        f"{actual.num_rows} rows, expected {expected.num_rows}"
    )
    for name in expected.column_names:
        a = actual.column(name).combine_chunks()
        e = expected.column(name).combine_chunks()
        a_valid = np.asarray(a.is_valid())
        e_valid = np.asarray(e.is_valid())
        bad = np.flatnonzero(a_valid != e_valid)
        assert bad.size == 0, f"{name}: null at different rows, first {bad[0]}"
        if pa.types.is_floating(e.type):
            a_bits = a.fill_null(0.0).to_numpy(zero_copy_only=False).view(np.int64)
            e_bits = e.fill_null(0.0).to_numpy(zero_copy_only=False).view(np.int64)
            bad = np.flatnonzero((a_bits != e_bits) & e_valid)
        else:
            bad = np.flatnonzero(~np.asarray(pc.fill_null(pc.equal(a, e), True)))
        assert bad.size == 0, (
            f"{name}: {bad.size} rows differ, first at {bad[0]}: "
            f"{a[int(bad[0])]} vs {e[int(bad[0])]}"
        )
