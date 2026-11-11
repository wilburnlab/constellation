"""What kind of acquisition a run is, read off its MS2 isolation windows.

Whether a run is DIA or DDA is a property of the data, not something a
user should have to assert: a DIA run cycles through a small, fixed set
of isolation windows and visits each one hundreds or thousands of times,
while a DDA run isolates a different precursor nearly every scan. The
number of MS2 scans per distinct window separates the two by orders of
magnitude (a 100-minute gas-phase-fraction run here: 80,213 scans over
49 windows; a DDA run on the same instrument: 179,533 scans over 134,180
windows).

The inference is used to label a ``.dia`` file's
``rawFileStructure.*`` metadata, which EncyclopeDIA reads to decide how
to search it, and to pick a default search mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

AcquisitionType = Literal["DIA", "DDA"]

#: A run is DIA when its MS2 scans revisit each isolation window at
#: least this many times on average. Real DIA is in the hundreds to
#: thousands; real DDA is between 1 and 2.
MIN_VISITS_PER_DIA_WINDOW = 10.0


@dataclass(frozen=True)
class AcquisitionStructure:
    """Summary of a run's scan structure."""

    acquisition_type: AcquisitionType
    #: DIA only: neighbouring windows overlap by about half their width,
    #: i.e. alternate cycles are offset (overlapped / staggered windows,
    #: which the search demultiplexes).
    is_staggered: bool
    n_ms1: int
    n_ms2: int
    #: Distinct ``(isolation_lower, isolation_upper)`` pairs among MS2 scans.
    n_windows: int


def isolation_windows(scan_metadata: pa.Table) -> np.ndarray:
    """Distinct MS2 isolation windows, as an ``(n, 2)`` float32 array sorted by lower bound.

    Bounds are compared at float32 precision: that is what a ``.dia``
    stores, and it keeps a window that the instrument reports with
    last-digit jitter from splitting in two.
    """
    ms2 = scan_metadata.filter(pc.equal(scan_metadata.column("level"), 2))
    lower = ms2.column("isolation_lower").to_numpy(zero_copy_only=False)
    upper = ms2.column("isolation_upper").to_numpy(zero_copy_only=False)
    keep = ~(np.isnan(lower) | np.isnan(upper))
    pairs = np.stack([lower[keep], upper[keep]], axis=1).astype(np.float32)
    return np.unique(pairs, axis=0) if pairs.size else pairs.reshape(0, 2)


def infer_acquisition_structure(scan_metadata: pa.Table) -> AcquisitionStructure:
    """Classify a run from a ``SCAN_METADATA_TABLE``-shaped table.

    Needs the ``level``, ``isolation_lower`` and ``isolation_upper``
    columns. Raises ``ValueError`` for a run with no MS2 scans or no
    usable isolation bounds, since neither label would mean anything.
    """
    level = scan_metadata.column("level")
    n_ms1 = int(pc.sum(pc.equal(level, 1)).as_py() or 0)
    n_ms2 = int(pc.sum(pc.equal(level, 2)).as_py() or 0)
    if n_ms2 == 0:
        raise ValueError("run has no MS2 scans; it is neither DIA nor DDA")
    windows = isolation_windows(scan_metadata)
    n_windows = int(windows.shape[0])
    if n_windows == 0:
        raise ValueError("no MS2 scan carries isolation bounds; cannot classify the run")

    is_dia = n_ms2 / n_windows >= MIN_VISITS_PER_DIA_WINDOW
    staggered = False
    if is_dia and n_windows > 1:
        # Overlap between each window and the next one up, as a fraction
        # of the narrower of the two. Contiguous schemes sit near 0 (a
        # small margin at most); staggered schemes sit near 0.5.
        lo, hi = windows[:, 0].astype(np.float64), windows[:, 1].astype(np.float64)
        overlap = np.clip(hi[:-1] - lo[1:], 0.0, None)
        narrower = np.minimum(hi[:-1] - lo[:-1], hi[1:] - lo[1:])
        frac = np.divide(overlap, narrower, out=np.zeros_like(overlap), where=narrower > 0)
        staggered = bool(np.median(frac) > 0.25)
    return AcquisitionStructure(
        acquisition_type="DIA" if is_dia else "DDA",
        is_staggered=staggered,
        n_ms1=n_ms1,
        n_ms2=n_ms2,
        n_windows=n_windows,
    )


def parent_scans(
    scan: np.ndarray,
    level: np.ndarray,
    recorded: np.ndarray | None = None,
    faims_cv: np.ndarray | None = None,
) -> np.ndarray:
    """Each scan's parent: the scan one level down it was taken from, as a
    scan number, -1 for none (an MS1 scan, or one with no lower-level scan
    before it).

    ``scan`` and ``level`` are in scan order. The parent is the scan the
    instrument records (``recorded``, the trailer's master scan; 0, NaN or
    negative is none), else the last scan of the level below before it.
    Where FAIMS compensation voltages are recorded (``faims_cv``, NaN for
    none), that is the last such scan *at the same voltage*: in a method
    alternating voltages, the scan just before is the other voltage's.
    The one rule for the browser (``massspec.scans``) and the sweep.

    Not every instrument records the master scan: the lab's Stellar DIA run
    RF30 records none for any of its 234,000 MS2 scans, which come exactly
    100 to an MS1 scan, so the derived parent is the usual case.

    A recorded master scan is taken only if it is a scan of the run, earlier,
    one level down and at the same voltage; otherwise the derived parent
    stands (until 2026-11-11 any number above 0 was taken: the scan itself,
    a later one, one not in the run). Voltages group scans only when there
    are several; a scan recorded without one serves every group, so a run
    that records its voltage on MS2 alone still finds MS1 parents (it found
    none; review A10).
    """
    scan = np.asarray(scan, dtype=np.int64)
    level = np.asarray(level, dtype=np.int64)
    parent = np.full(scan.size, -1, dtype=np.int64)
    if scan.size == 0:
        return parent
    cv = None if faims_cv is None else np.asarray(faims_cv, dtype=np.float64)
    voltages = np.unique(cv[~np.isnan(cv)]) if cv is not None else np.empty(0)
    if voltages.size <= 1:
        cv_group = np.zeros(scan.size, np.int64)
        free = np.zeros(scan.size, bool)
    else:
        free = np.isnan(cv)
        cv_group = np.where(free, -1, np.searchsorted(voltages, np.where(free, voltages[0], cv)))
    groups = np.unique(cv_group[~free]) if (~free).any() else np.array([0])
    for group in groups:
        in_group = (cv_group == group) | free
        for lvl in np.unique(level[(level > 1) & in_group]):
            below = np.flatnonzero((level == lvl - 1) & in_group)
            at = np.flatnonzero((level == lvl) & (cv_group == group) if voltages.size > 1 else (level == lvl))
            if below.size and at.size:
                # Rows are in scan order, so the last lower-level row before
                # each scan is found by position.
                before = np.searchsorted(below, at, side="left") - 1
                ok = before >= 0
                parent[at[ok]] = scan[below[before[ok]]]
    if voltages.size > 1 and free.any():
        # A higher-level scan without a voltage: the last scan below at all.
        for lvl in np.unique(level[(level > 1) & free]):
            below = np.flatnonzero(level == lvl - 1)
            at = np.flatnonzero((level == lvl) & free)
            if below.size:
                before = np.searchsorted(below, at, side="left") - 1
                ok = before >= 0
                parent[at[ok]] = scan[below[before[ok]]]
    if recorded is not None:
        given = np.asarray(recorded, dtype=np.float64)
        ok = (level > 1) & ~np.isnan(given) & (given > 0)
        claimed = np.where(ok, given, 0).astype(np.int64)
        # Rows are in scan order: find each recorded scan among them.
        pos = np.searchsorted(scan, claimed)
        pos_c = np.minimum(pos, max(scan.size - 1, 0))
        exists = ok & (pos < scan.size) & (scan[pos_c] == claimed) if scan.size else ok
        exists &= claimed < scan
        exists &= level[pos_c] == level - 1
        if voltages.size > 1:
            same = (cv_group[pos_c] == cv_group) | free[pos_c] | free
            exists &= same
        parent[exists] = claimed[exists]
    return parent


def unsupported(scan_metadata: pa.Table) -> list[str]:
    """Why a run cannot be read scan by scan as one acquisition, as
    sentences (none: it can): several FAIMS compensation voltages, whose
    scans a window keyed by isolation bounds would mix; multiplexed (MSX)
    isolation, several precursors a scan where a scan holds one window.
    The one test for the sweep, the browser and the ``.dia`` export
    (review A13: the export did not test MSX, the browser neither)."""
    out = []
    names = set(scan_metadata.column_names)
    if "faims_cv" in names:
        cv = scan_metadata.column("faims_cv").cast(pa.float64()).fill_null(np.nan).to_numpy(zero_copy_only=False)
        seen = np.unique(cv[~np.isnan(cv)])
        if seen.size > 1:
            out.append(
                f"it is a multi-CV FAIMS run (compensation voltages {', '.join(f'{v:g}' for v in seen)}), "
                "which is not separated by voltage yet; split it by CV first"
            )
    if "filter_string" in names:
        filters = scan_metadata.column("filter_string")
        msx = pc.fill_null(pc.match_substring_regex(filters, r"(^|\s)msx(\s|$)"), False)
        if pc.any(msx).as_py():
            out.append(
                f"it has {pc.sum(msx).as_py():,} multiplexed (MSX) scans, each isolating several precursors, "
                "where a scan holds one isolation window"
            )
    return out


def refuse_unsupported(scan_metadata: pa.Table, where: str) -> None:
    """Raise if the run is one :func:`unsupported` names a reason for."""
    reasons = unsupported(scan_metadata)
    if reasons:
        raise ValueError(f"{where}: " + "; and ".join(reasons))


def impute_injection_times(
    level: np.ndarray, iit: np.ndarray, tic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Injection times (ms) with the ones a run did not record estimated,
    and which were (``(iit, imputed)``).

    A scan's injection time is about the time its ion target took to fill:
    ``IIT = 1000 · N / TIC``, with ``N`` the level's median of
    ``IIT × TIC / 1000`` over the scans that record both, capped at the
    level's longest recorded time (the method's maximum). A scan with no TIC
    keeps NaN and is not flagged; so does a level with nothing to take ``N``
    from. On RF30, hiding the recorded times gave back estimates within
    25 % on 86 % of MS1 and 84 % of MS2 scans (island spec, decision E,
    ``docs/plans/profile-and-peak-lists.md``). 13.2 % of RF30's scans record
    none, and a count term was NaN on all of them (review S4)."""
    level = np.asarray(level)
    iit = np.asarray(iit, dtype=np.float64).copy()
    tic = np.asarray(tic, dtype=np.float64)
    imputed = np.zeros(iit.size, bool)
    for lvl in np.unique(level):
        at = level == lvl
        both = at & np.isfinite(iit) & np.isfinite(tic) & (tic > 0) & (iit > 0)
        missing = at & ~np.isfinite(iit) & np.isfinite(tic) & (tic > 0)
        if not both.any() or not missing.any():
            continue
        n = float(np.median(iit[both] * tic[both] / 1000.0))
        cap = float(iit[both].max())
        iit[missing] = np.minimum(1000.0 * n / tic[missing], cap)
        imputed[missing] = True
    return iit, imputed


__all__ = [
    "MIN_VISITS_PER_DIA_WINDOW",
    "AcquisitionStructure",
    "AcquisitionType",
    "impute_injection_times",
    "infer_acquisition_structure",
    "isolation_windows",
    "parent_scans",
    "refuse_unsupported",
    "unsupported",
]
