"""What a sweep looks for: extraction targets compiled into queries and their channels.

A query is one row of an :data:`~constellation.massspec.quant.schemas.XIC_TARGET_TABLE`
as the sweep resolves it: its retention-time interval, the precursor m/z
that decides which isolation windows it is matched in, and the scan its
fragments are restricted to, if any. Its channels are the m/z values it
expects signal at, each with what it is: precursor isotopes in MS1 scans,
fragment ions in MS2 and higher. Both are held as flat arrays in
compressed-row form (query ``q``'s channels are rows
``offsets[q]:offsets[q + 1]``), in the order output lists them, so nothing
exists per channel as a Python object.

The channels are the m/z values the scan-major extractor always queried, value for value:
isotopes from the binned :func:`~constellation.massspec.peptide.envelope.peptide_envelope`
at the target's own charge (or at each charge of ``charge_range`` when one
is asked for, and of ``fallback_charge_range`` for a target with no
charge), or spaced off a bare precursor m/z; fragments from the batched ladder, up to the
lower of ``max_fragment_charge`` and the precursor's charge. Two things
differ from that extractor, both on purpose: a target with a sequence and a
charge but no precursor m/z has one computed, so its fragments are gated by
isolation window rather than matched in every window; and the assigned-scan
restriction applies to fragments only, since an MS1 scan never has the
number of the MS2 scan a match was made in.
"""

from __future__ import annotations

import json
from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.core.sequence.protein import peptide_masses
from constellation.massspec.peptide.channels import (
    FRAGMENTS,
    ISOTOPES,
    UNMODELLED,
    Channels,
    ChannelSettings,
    FragmentAbundance,
    concat_channels,
    fragment_channels,
    isotope_channels,
)
from constellation.massspec.peptide.mz import PROTON_MASS
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep.schemas import SWEEP_CHANNEL_TABLE, SWEEP_QUERY_TABLE

__all__ = [
    "ISOTOPES",
    "FRAGMENTS",
    "ChannelSettings",
    "Channels",
    "QueryPlan",
    "check_rt_window",
    "compile_queries",
]

#: Where a written plan keeps its ``ChannelSettings`` (channel table metadata).
_SETTINGS_KEY = b"x.massspec.sweep.channel_settings"
#: And what its fragment abundances came from.
_ABUNDANCE_KEY = b"x.massspec.sweep.fragment_abundance"


@dataclass(frozen=True)
class QueryPlan:
    """Compiled queries: per-query arrays (index = position in the input)
    and their channels by kind, for the levels the sweep matches."""

    targets: pa.Table  # the input, cast to XIC_TARGET_TABLE
    levels: tuple[int, ...]
    precursor_charge: np.ndarray  # int16, -1 for none
    precursor_mz: np.ndarray  # float64, NaN for no isolation gate
    precursor_mz_derived: np.ndarray  # bool
    rt_lo: np.ndarray  # float64
    rt_hi: np.ndarray  # float64
    assigned_scan: np.ndarray  # int64, -1 for none
    isotopes: Channels | None
    fragments: Channels | None
    settings: ChannelSettings = field(default_factory=ChannelSettings)
    #: What the fragment abundances came from (``FragmentAbundance.describe``).
    fragment_abundance: dict = field(default_factory=lambda: dict(UNMODELLED.describe()))

    @property
    def n_queries(self) -> int:
        return int(self.rt_lo.size)

    @cached_property
    def precursor_order(self) -> np.ndarray:
        """Every query by precursor m/z (a stable sort, those with none
        last): computed once per plan, not per block."""
        return np.argsort(self.precursor_mz, kind="stable")

    def channels_for_level(self, level: int) -> Channels | None:
        return self.isotopes if level == 1 else self.fragments

    def to_tables(self) -> tuple[pa.Table, pa.Table]:
        """The plan as :data:`SWEEP_QUERY_TABLE` and :data:`SWEEP_CHANNEL_TABLE`."""
        t = self.targets
        charge = self.precursor_charge
        queries = pa.table(
            {
                "query": pa.array(np.arange(self.n_queries, dtype=np.int32)),
                "target_id": t.column("target_id"),
                "modified_sequence": t.column("modified_sequence"),
                "precursor_charge": pa.array(charge.astype(np.int8), mask=charge < 0),
                "precursor_mz": pa.array(self.precursor_mz, mask=np.isnan(self.precursor_mz)),
                "precursor_mz_derived": pa.array(self.precursor_mz_derived),
                "rt_lo": pa.array(self.rt_lo),
                "rt_hi": pa.array(self.rt_hi),
                "assigned_scan": pa.array(
                    self.assigned_scan.astype(np.int32), mask=self.assigned_scan < 0
                ),
                "precursor_id": t.column("precursor_id"),
                "peptide_id": t.column("peptide_id"),
                "is_decoy": t.column("is_decoy"),
                "decoy_set": t.column("decoy_set"),
            },
            schema=SWEEP_QUERY_TABLE,
        )
        tables = [_channel_table(c) for c in (self.isotopes, self.fragments) if c is not None]
        channels = (
            pa.concat_tables(tables) if tables else SWEEP_CHANNEL_TABLE.empty_table()
        )  # two small in-memory tables of one plan, not a stage hand-off
        meta = dict(channels.schema.metadata or {})
        meta[b"x.massspec.sweep.levels"] = ",".join(map(str, self.levels)).encode()
        meta[b"x.massspec.sweep.loss_ids"] = ",".join(
            self.fragments.loss_ids if self.fragments is not None else ()
        ).encode()
        meta[_SETTINGS_KEY] = json.dumps(self.settings.to_dict()).encode()
        meta[_ABUNDANCE_KEY] = json.dumps(self.fragment_abundance).encode()
        return queries, channels.replace_schema_metadata(meta)

    def write(self, directory) -> None:
        """Write the plan as ``queries.parquet``, ``channels.parquet`` and
        ``targets.parquet`` (the whole target table, every column a
        reducer may read) in ``directory``: what a sweep on disk hands its
        worker processes, so they see the plan a sweep in memory does."""
        from pathlib import Path

        import pyarrow.parquet as pq

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        queries, channels = self.to_tables()
        for name, table in (("queries", queries), ("channels", channels), ("targets", self.targets)):
            tmp = d / f".{name}.parquet.tmp"
            pq.write_table(table, tmp)
            tmp.replace(d / f"{name}.parquet")

    @classmethod
    def read(cls, directory) -> QueryPlan:
        """A plan written by :meth:`write`."""
        from pathlib import Path

        import pyarrow.parquet as pq

        d = Path(directory)
        return cls.from_tables(
            pq.read_table(d / "queries.parquet"),
            pq.read_table(d / "channels.parquet"),
            pq.read_table(d / "targets.parquet"),
        )

    @classmethod
    def from_tables(cls, queries: pa.Table, channels: pa.Table, targets: pa.Table | None = None) -> QueryPlan:
        """Rebuild a plan written by :meth:`to_tables`, with its targets
        (``plan.targets``) when given. Without them the targets are what
        the query table holds, which leaves out their retention-time
        columns, scan and anything else a reducer may read (as a plan on
        disk did until 2026-11-07, so reducers there saw another plan)."""
        meta = channels.schema.metadata or {}
        levels = tuple(int(x) for x in meta[b"x.massspec.sweep.levels"].decode().split(",") if x)
        loss_text = meta.get(b"x.massspec.sweep.loss_ids", b"").decode()
        loss_ids = tuple(x for x in loss_text.split(",") if x)
        settings = (
            ChannelSettings.from_dict(json.loads(meta[_SETTINGS_KEY].decode()))
            if _SETTINGS_KEY in meta
            else ChannelSettings()
        )
        n = queries.num_rows
        if targets is not None:
            if targets.num_rows != n:
                raise ValueError(f"{targets.num_rows} targets for a plan of {n} queries")
            targets = cast_to_schema(targets, XIC_TARGET_TABLE)
        else:
            targets = cast_to_schema(
                pa.table(
                    {
                        "target_id": queries.column("target_id"),
                        "modified_sequence": queries.column("modified_sequence"),
                        "precursor_charge": queries.column("precursor_charge"),
                        "precursor_mz": queries.column("precursor_mz"),
                        "precursor_id": queries.column("precursor_id"),
                        "peptide_id": queries.column("peptide_id"),
                        "is_decoy": queries.column("is_decoy"),
                        **(
                            {"decoy_set": queries.column("decoy_set")}
                            if "decoy_set" in queries.column_names else {}
                        ),
                    }
                ),
                XIC_TARGET_TABLE,
            )
        kinds = channels.column("kind").to_numpy()

        def of(kind: int, ids: tuple[str, ...]) -> Channels:
            return _channels_from_table(kind, channels.filter(pa.array(kinds == kind)), n, ids)

        return cls(
            targets=targets,
            levels=levels,
            precursor_charge=queries.column("precursor_charge").fill_null(-1).to_numpy().astype(np.int16),
            precursor_mz=queries.column("precursor_mz").fill_null(np.nan).to_numpy().astype(np.float64),
            precursor_mz_derived=queries.column("precursor_mz_derived").to_numpy(zero_copy_only=False),
            rt_lo=queries.column("rt_lo").to_numpy(),
            rt_hi=queries.column("rt_hi").to_numpy(),
            assigned_scan=queries.column("assigned_scan").fill_null(-1).to_numpy().astype(np.int64),
            isotopes=of(ISOTOPES, ()) if 1 in levels else None,
            fragments=of(FRAGMENTS, loss_ids) if any(level >= 2 for level in levels) else None,
            settings=settings,
            fragment_abundance=(
                json.loads(meta[_ABUNDANCE_KEY].decode()) if _ABUNDANCE_KEY in meta else dict(UNMODELLED.describe())
            ),
        )


def _channel_table(c: Channels) -> pa.Table:
    frag = c.kind == FRAGMENTS
    n = c.n
    none = np.ones(n, dtype=bool)
    loss_names = pa.array([None, *c.loss_ids], pa.string())
    return pa.table(
        {
            "kind": pa.array(np.full(n, c.kind, np.int8)),
            "query": pa.array(c.query_of().astype(np.int32)),
            "channel": pa.array(c.ordinal().astype(np.int32)),
            "mz_theoretical": pa.array(c.mz),
            "precursor_charge": pa.array(c.precursor_charge),
            "isotope": pa.array(c.isotope, mask=none if frag else None),
            "ion_type": pa.array(c.ion_type, mask=None if frag else none),
            "position": pa.array(c.position, mask=None if frag else none),
            "fragment_charge": pa.array(c.fragment_charge, mask=None if frag else none),
            "loss_id": loss_names.take(pa.array(c.loss.astype(np.int64))) if frag else pa.nulls(n, pa.string()),
            "abundance": pa.array(c.abundance, mask=np.isnan(c.abundance)),
            "evidence_key": pa.array(c.evidence_key),
        },
        schema=SWEEP_CHANNEL_TABLE,
    )


def _channels_from_table(kind: int, rows: pa.Table, n_queries: int, loss_ids: tuple[str, ...]) -> Channels:
    def col(name, dtype, fill):
        return rows.column(name).fill_null(fill).to_numpy().astype(dtype)

    query = rows.column("query").to_numpy()
    offsets = np.zeros(n_queries + 1, dtype=np.int64)
    np.cumsum(np.bincount(query, minlength=n_queries), out=offsets[1:])
    names = rows.column("loss_id")
    found = pc.index_in(names, value_set=pa.array(list(loss_ids), pa.string()))
    unknown = pc.and_(names.is_valid(), found.is_null())
    if pc.any(unknown).as_py():
        raise ValueError(f"channels name losses the plan does not list ({loss_ids}): "
                         f"{pc.unique(names.filter(unknown)).to_pylist()[:5]}")  # fmt: skip
    loss = found.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int16) + 1
    return Channels(
        kind=kind,
        offsets=offsets,
        mz=rows.column("mz_theoretical").to_numpy(),
        precursor_charge=col("precursor_charge", np.int8, 0),
        isotope=col("isotope", np.int8, 0),
        ion_type=col("ion_type", np.int8, -1),
        position=col("position", np.int32, -1),
        fragment_charge=col("fragment_charge", np.int32, -1),
        loss=loss,
        abundance=rows.column("abundance").to_numpy(zero_copy_only=False).astype(np.float32),
        evidence_key=col("evidence_key", np.int32, 0),
        loss_ids=loss_ids,
    )


# ── compiling ─────────────────────────────────────────────────────────


def _peptidoform(modseq: str) -> Peptidoform:
    p = parse_proforma(modseq)
    if not isinstance(p, Peptidoform):
        raise ValueError(
            f"XIC extraction supports linear peptidoforms; {modseq!r} parsed to "
            f"{type(p).__name__}"
        )
    return p


def _rt_bounds(t: pa.Table, rt_window: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Explicit [start, end] wins; else center ± window; else unbounded. A
    NaN bound bounds nothing, as it did not in the extractor, whose
    bisection over retention times put a NaN before the first scan (as a
    lower bound) and after the last (as an upper one)."""

    def values(name: str) -> tuple[np.ndarray, np.ndarray]:
        col = t.column(name)
        return (
            np.asarray(col.is_valid(), dtype=bool),
            col.fill_null(0.0).to_numpy(zero_copy_only=False).astype(np.float64),
        )

    (has_s, start), (has_e, end), (has_c, center) = (
        values(n) for n in ("rt_start", "rt_end", "rt_center")
    )
    explicit = has_s & has_e
    lo = np.where(explicit, start, -np.inf)
    hi = np.where(explicit, end, np.inf)
    if rt_window is not None:
        windowed = ~explicit & has_c
        lo = np.where(windowed, center - rt_window, lo)
        hi = np.where(windowed, center + rt_window, hi)
    return np.where(np.isnan(lo), -np.inf, lo), np.where(np.isnan(hi), np.inf, hi)


def check_rt_window(targets: pa.Table) -> None:
    """Raise ``ValueError`` when an RT window cannot be drawn for these
    targets. A window is drawn around ``rt_center``, which is in the
    run's own seconds; a target whose only time is a library's on a
    model's scale (``rt_library`` on ``rt_scale``) has none, and
    windowing it would gate on the wrong clock. Refused rather than done,
    or silently widened to the whole run."""
    import pyarrow.compute as pc

    t = cast_to_schema(targets, XIC_TARGET_TABLE)
    scale = t.column("rt_scale")
    model = pc.and_(
        pc.and_(pc.is_valid(scale), pc.not_equal(pc.fill_null(scale, ""), "observed")),
        pc.and_(pc.is_valid(t.column("rt_library")), pc.is_null(t.column("rt_center"))),
    )
    explicit = pc.and_(pc.is_valid(t.column("rt_start")), pc.is_valid(t.column("rt_end")))
    model = pc.and_(model, pc.invert(explicit))
    n = pc.sum(model).as_py() or 0
    if n:
        scales = sorted(set(scale.filter(model).to_pylist()))
        raise ValueError(
            f"{n} targets carry retention times on {', '.join(scales)}, not this run's seconds: an "
            f"RT window cannot be drawn around them until they are mapped onto the run (calibration "
            f"comes with the search); drop the RT window, or give rt_start / rt_end"
        )


#: Targets compiled at once. What a compile holds per target (its parsed
#: peptide, its whole candidate ladder of some 80 ions before the top N are
#: kept) is bounded by this, not by the number of targets: at 964k targets
#: the ladder alone took over 10 GB before 2026-11-09.
COMPILE_CHUNK = 1 << 16


def compile_queries(
    targets: pa.Table,
    *,
    levels: Sequence[int],
    channels: ChannelSettings = ChannelSettings(),
    rt_window: float | None = None,
    assigned_scans_only: bool = False,
    derive_precursor_mz: bool = True,
    fragment_abundance: FragmentAbundance = UNMODELLED,
    chunk_targets: int | None = COMPILE_CHUNK,
) -> QueryPlan:
    """Compile extraction targets into a sweep's queries.

    ``levels`` are the MS levels to match at: 1 for precursor isotopes,
    2 (or higher) for fragments; ``channels`` says which isotopes and
    fragment ions they are (``massspec.peptide.channels``), and
    ``fragment_abundance`` where fragment abundances come from (looked up
    by each target's ``precursor_id``). A target that can define no
    channels of a level asked for is an error when it would then be
    matched at nothing, as the scan-major extractor's were (a bare m/z at
    MS2 alone, a sequence with no charge); otherwise it simply has none of
    that kind.

    Vectorized over targets: each distinct sequence is parsed once and
    its envelope and ladder built once; nothing else runs per target in
    Python (1.3 ms a target at proteome scale until 2026-11-07). Targets
    are compiled ``chunk_targets`` at a time, in order of precursor id (so
    a fragment library sorted by precursor is read a few row groups per
    chunk), and the plan is the same, bit for bit, whatever the chunk.
    """
    levels = tuple(sorted(set(int(x) for x in levels)))
    if not levels or min(levels) < 1:
        raise ValueError(f"levels must be MS levels ≥ 1; got {levels!r}")
    t = cast_to_schema(targets, XIC_TARGET_TABLE)
    n = t.num_rows
    if rt_window is not None:
        check_rt_window(t)
    want_frag = any(level >= 2 for level in levels)
    if want_frag and 1 not in levels:
        _check_fragment_targets(t)
    key = t.column("precursor_id").fill_null(np.iinfo(np.int64).max).to_numpy().astype(np.int64)
    prepare = getattr(fragment_abundance, "prepare", None) if want_frag else None
    if prepare is not None:
        prepare(key[key != np.iinfo(np.int64).max])
    try:
        kwargs = dict(levels=levels, channels=channels, rt_window=rt_window, assigned_scans_only=assigned_scans_only,
                      derive_precursor_mz=derive_precursor_mz, fragment_abundance=fragment_abundance)  # fmt: skip
        if chunk_targets is None or n <= chunk_targets:
            return _compile(t, **kwargs)
        if chunk_targets < 1:
            raise ValueError(f"chunk_targets must be at least 1, got {chunk_targets}")
        order = np.argsort(key, kind="stable")
        plans = [_compile(t.take(pa.array(order[i:i + chunk_targets])), **kwargs) for i in range(0, n, chunk_targets)]
        return _join_plans(t, order, plans)
    finally:
        release = getattr(fragment_abundance, "release", None)
        if prepare is not None and release is not None:
            release()


def _check_fragment_targets(t: pa.Table) -> None:
    """At MS2 alone, a target that can define no fragments is an error, the
    first in target order named (as the extractor did before matching)."""
    modseq = t.column("modified_sequence")
    present = np.asarray(pc.fill_null(pc.not_equal(modseq, ""), False).to_numpy(zero_copy_only=False), dtype=bool)
    charge = t.column("precursor_charge").fill_null(-1).to_numpy().astype(np.int64)
    bad = np.flatnonzero(~present | (charge < 1))
    if bad.size:
        i = int(bad[0])
        target = t.column("target_id")[i].as_py()
        if not present[i]:
            raise ValueError(
                f"target {target}: MS2 fragment extraction requires "
                "modified_sequence (a bare-m/z target cannot define fragments)"
            )
        raise ValueError(f"target {target}: MS2 extraction requires precursor_charge")


def _join_plans(t: pa.Table, order: np.ndarray, plans: list[QueryPlan]) -> QueryPlan:
    """Plans compiled from ``t.take(order)`` chunk by chunk, as one plan of
    ``t`` in its own order."""
    inverse = np.empty(order.size, dtype=np.int64)
    inverse[order] = np.arange(order.size, dtype=np.int64)

    def per_query(name: str) -> np.ndarray:
        return np.concatenate([getattr(p, name) for p in plans])[inverse]

    def channels(name: str) -> Channels | None:
        parts = [getattr(p, name) for p in plans]
        return None if parts[0] is None else concat_channels(parts).take(inverse)

    described = dict(plans[0].fragment_abundance)
    if "report" in described:
        report: dict[str, Any] = {}
        for p in plans:
            _merge_report(report, p.fragment_abundance.get("report") or {})
        described["report"] = report
    first = plans[0]
    return QueryPlan(
        targets=t,
        levels=first.levels,
        precursor_charge=per_query("precursor_charge"),
        precursor_mz=per_query("precursor_mz"),
        precursor_mz_derived=per_query("precursor_mz_derived"),
        rt_lo=per_query("rt_lo"),
        rt_hi=per_query("rt_hi"),
        assigned_scan=per_query("assigned_scan"),
        isotopes=channels("isotopes"),
        fragments=channels("fragments"),
        settings=first.settings,
        fragment_abundance=described,
    )


def _merge_report(into: dict[str, Any], part: dict[str, Any]) -> None:
    """Add one chunk's abundance report to the whole compile's: counts
    sum, a ``max_`` value is the largest, examples gather (up to 5), and
    anything else keeps its first value."""
    for name, value in part.items():
        have = into.get(name)
        if have is None:
            into[name] = dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
        elif isinstance(value, dict):
            _merge_report(have, value)
        elif isinstance(value, list):
            have.extend(value[: max(0, 5 - len(have))])
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        elif name.startswith("max_"):
            into[name] = max(have, value)
        else:
            into[name] = have + value


def _compile(
    t: pa.Table,
    *,
    levels: tuple[int, ...],
    channels: ChannelSettings,
    rt_window: float | None,
    assigned_scans_only: bool,
    derive_precursor_mz: bool,
    fragment_abundance: FragmentAbundance,
) -> QueryPlan:
    """:func:`compile_queries` on one chunk of cast targets."""
    n = t.num_rows
    want_ms1 = 1 in levels
    want_frag = any(level >= 2 for level in levels)
    target_ids = t.column("target_id").to_pylist()

    # Each distinct sequence once, in order of first appearance.
    modseq = t.column("modified_sequence")
    present = np.asarray(pc.fill_null(pc.not_equal(modseq, ""), False).to_numpy(zero_copy_only=False), dtype=bool)
    distinct = pc.unique(modseq.filter(pa.array(present))).to_pylist() if present.any() else []
    peptide = np.full(n, -1, dtype=np.int64)
    if distinct:
        at = pc.index_in(modseq, value_set=pa.array(distinct, pa.string())).to_numpy(zero_copy_only=False)
        peptide = np.where(present, np.nan_to_num(np.asarray(at, dtype=np.float64), nan=-1), -1).astype(np.int64)
    charge_col = t.column("precursor_charge")
    precursor_charge = charge_col.fill_null(-1).to_numpy().astype(np.int16)
    # A charge below 1 is no charge (-1 here, null on disk). Until 2026-11-09
    # charge 0 counted as one: its precursor m/z came out infinite and the
    # query was never matched at MS2, with no word.
    precursor_charge[precursor_charge < 1] = -1
    has_charge = precursor_charge >= 1

    # Fragments are checked first and in target order, as the extractor did
    # before matching anything; they are the only errors raised before MS1's.
    frag = np.zeros(n, dtype=bool)
    if want_frag:
        if not want_ms1:
            bad = np.flatnonzero(~present | ~has_charge)
            if bad.size:
                i = int(bad[0])
                if not present[i]:
                    raise ValueError(
                        f"target {target_ids[i]}: MS2 fragment extraction requires "
                        "modified_sequence (a bare-m/z target cannot define fragments)"
                    )
                raise ValueError(f"target {target_ids[i]}: MS2 extraction requires precursor_charge")
        frag = present & has_charge
    parsed = [_peptidoform(m) for m in distinct]

    pmz = t.column("precursor_mz").fill_null(np.nan).to_numpy().astype(np.float64)
    derived = np.zeros(n, dtype=bool)
    if derive_precursor_mz:
        want = np.flatnonzero(np.isnan(pmz) & present & has_charge)
        if want.size:
            used = np.unique(peptide[want])
            masses = peptide_masses([parsed[p] for p in used.tolist()]).numpy()
            z = precursor_charge[want].astype(np.int64)
            # precursor_mz's (mass + z·proton) / z, elementwise.
            pmz[want] = (masses[np.searchsorted(used, peptide[want])] + z * PROTON_MASS) / z.astype(np.float64)
            derived[want] = True

    rt_lo, rt_hi = _rt_bounds(t, rt_window)
    assigned = (
        t.column("scan").fill_null(-1).to_numpy().astype(np.int64)
        if assigned_scans_only
        else np.full(n, -1, dtype=np.int64)
    )

    isotopes = (
        isotope_channels(parsed, peptide, precursor_charge.astype(np.int64), pmz, channels,
                         target_ids=target_ids)  # fmt: skip
        if want_ms1
        else None
    )
    fragments = None
    if want_frag:
        key = t.column("precursor_id").fill_null(-1).to_numpy().astype(np.int64)
        fragments = fragment_channels(
            parsed, np.where(frag, peptide, -1), precursor_charge.astype(np.int64), channels, fragment_abundance,
            key=key, target_ids=target_ids,
        )  # fmt: skip
    return QueryPlan(
        targets=t,
        levels=levels,
        precursor_charge=precursor_charge,
        precursor_mz=pmz,
        precursor_mz_derived=derived,
        rt_lo=rt_lo,
        rt_hi=rt_hi,
        assigned_scan=assigned,
        isotopes=isotopes,
        fragments=fragments,
        settings=channels,
        fragment_abundance=dict(fragment_abundance.describe()),
    )
