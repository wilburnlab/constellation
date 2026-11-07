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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import json

import numpy as np
import pyarrow as pa

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.core.stats.units import ISOTOPE_MASS_DIFF
from constellation.massspec.peptide.envelope import envelope_mz, neutral_envelope
from constellation.massspec.peptide.ions import IonType, fragment_ladder_arrays_batch
from constellation.massspec.peptide.mz import precursor_mz as _precursor_mz
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE
from constellation.massspec.sweep.schemas import SWEEP_CHANNEL_TABLE, SWEEP_QUERY_TABLE

if TYPE_CHECKING:
    import torch

__all__ = [
    "ISOTOPES",
    "FRAGMENTS",
    "ChannelSettings",
    "Channels",
    "QueryPlan",
    "check_rt_window",
    "compile_queries",
]

#: Channel kinds: precursor isotopes are matched in MS1 scans, fragments in the rest.
ISOTOPES = 1
FRAGMENTS = 2

#: Where a written plan keeps its ``ChannelSettings`` (channel table metadata).
_SETTINGS_KEY = b"x.massspec.sweep.channel_settings"


@dataclass(frozen=True)
class ChannelSettings:
    """Which channels a query is matched at: the isotopes and charges of its
    precursor in MS1, the fragment ions in MS2."""

    n_isotopes: int = 3
    #: Isotope offsets to use instead of ``range(n_isotopes)``. Negative
    #: offsets (M−1, M−2) are for labelled peptides, whose incomplete
    #: labelling puts intensity below the fully labelled mass.
    isotope_offsets: tuple[int, ...] | None = None
    #: Charges to look for every target at in MS1, for the evidence of an
    #: analyte across its charge states (a peptide's precursors then share
    #: their MS1 channels). None (the default) is each target's own charge
    #: (PI, 2026-11-07; until then (1, 4) was the default for every target).
    charge_range: tuple[int, int] | None = None
    #: The charges a target with no charge of its own is looked for at in
    #: MS1, when ``charge_range`` is not given.
    fallback_charge_range: tuple[int, int] = (1, 4)
    ion_types: tuple[IonType, ...] = (IonType.B, IonType.Y)
    max_fragment_charge: int = 2
    neutral_losses: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.offsets() == ():
            raise ValueError(
                "the isotope selection is empty (n_isotopes < 1 or no isotope_offsets): MS1 would have no "
                "channels; leave level 1 out of the levels instead"
            )
        for name in ("charge_range", "fallback_charge_range"):
            value = getattr(self, name)
            if value is not None and not (len(value) == 2 and 1 <= int(value[0]) <= int(value[1])):
                raise ValueError(f"{name} must be (low, high) with 1 <= low <= high; got {value!r}")
        if int(self.max_fragment_charge) < 1:
            raise ValueError(f"max_fragment_charge must be >= 1; got {self.max_fragment_charge}")
        if not self.ion_types:
            raise ValueError("ion_types must name at least one ion series")

    def offsets(self) -> tuple[int, ...]:
        if self.isotope_offsets is not None:
            return tuple(sorted(set(int(k) for k in self.isotope_offsets)))
        return tuple(range(int(self.n_isotopes)))

    def to_dict(self) -> dict:
        """The settings as plain data (JSON), for a plan written to disk."""
        return {
            "n_isotopes": int(self.n_isotopes),
            "isotope_offsets": None if self.isotope_offsets is None else [int(k) for k in self.isotope_offsets],
            "charge_range": None if self.charge_range is None else [int(z) for z in self.charge_range],
            "fallback_charge_range": [int(z) for z in self.fallback_charge_range],
            "ion_types": [IonType(t).name for t in self.ion_types],
            "max_fragment_charge": int(self.max_fragment_charge),
            "neutral_losses": None if self.neutral_losses is None else list(self.neutral_losses),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChannelSettings:
        def pair(v):
            return None if v is None else (int(v[0]), int(v[1]))

        return cls(
            n_isotopes=int(d["n_isotopes"]),
            isotope_offsets=None if d["isotope_offsets"] is None else tuple(int(k) for k in d["isotope_offsets"]),
            charge_range=pair(d["charge_range"]),
            fallback_charge_range=pair(d["fallback_charge_range"]),
            ion_types=tuple(IonType[name] for name in d["ion_types"]),
            max_fragment_charge=int(d["max_fragment_charge"]),
            neutral_losses=None if d["neutral_losses"] is None else tuple(d["neutral_losses"]),
        )


@dataclass(frozen=True)
class Channels:
    """One kind of channel for every query, in compressed-row form."""

    kind: int
    offsets: np.ndarray  # (n_queries + 1,) int64
    mz: np.ndarray  # float64
    precursor_charge: np.ndarray  # int8
    isotope: np.ndarray  # int8; isotopes only
    ion_type: np.ndarray  # int8 IonType value; fragments only
    position: np.ndarray  # int32; fragments only
    fragment_charge: np.ndarray  # int32; fragments only
    loss: np.ndarray  # int16: 0 for none, k for loss_ids[k - 1]; fragments only
    abundance: np.ndarray  # float32; NaN where not modelled
    evidence_key: np.ndarray  # int32
    loss_ids: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return int(self.mz.size)

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def query_of(self) -> np.ndarray:
        """The query each channel belongs to."""
        return np.repeat(np.arange(self.offsets.size - 1, dtype=np.int64), self.counts())

    def ordinal(self) -> np.ndarray:
        """Each channel's position among its query's channels of this kind."""
        return np.arange(self.n, dtype=np.int64) - np.repeat(self.offsets[:-1], self.counts())


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

    @property
    def n_queries(self) -> int:
        return int(self.rt_lo.size)

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
        )


def _channel_table(c: Channels) -> pa.Table:
    frag = c.kind == FRAGMENTS
    n = c.n
    none = np.ones(n, dtype=bool)
    loss_names = np.array([None, *c.loss_ids], dtype=object)
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
            "loss_id": pa.array(loss_names[c.loss] if frag else [None] * n, pa.string()),
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
    names = rows.column("loss_id").to_pylist()
    lookup = {name: k + 1 for k, name in enumerate(loss_ids)}
    return Channels(
        kind=kind,
        offsets=offsets,
        mz=rows.column("mz_theoretical").to_numpy(),
        precursor_charge=col("precursor_charge", np.int8, 0),
        isotope=col("isotope", np.int8, 0),
        ion_type=col("ion_type", np.int8, -1),
        position=col("position", np.int32, -1),
        fragment_charge=col("fragment_charge", np.int32, -1),
        loss=np.array([0 if x is None else lookup[x] for x in names], dtype=np.int16),
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


def compile_queries(
    targets: pa.Table,
    *,
    levels: Sequence[int],
    channels: ChannelSettings = ChannelSettings(),
    rt_window: float | None = None,
    assigned_scans_only: bool = False,
    derive_precursor_mz: bool = True,
) -> QueryPlan:
    """Compile extraction targets into a sweep's queries.

    ``levels`` are the MS levels to match at: 1 for precursor isotopes,
    2 (or higher) for fragments; ``channels`` says which isotopes and
    fragment ions they are. A target that can define no channels of a
    level asked for is an error when it would then be matched at nothing,
    as the scan-major extractor's were (a bare m/z at MS2 alone, a sequence
    with no charge); otherwise it simply has none of that kind.
    """
    levels = tuple(sorted(set(int(x) for x in levels)))
    if not levels or min(levels) < 1:
        raise ValueError(f"levels must be MS levels ≥ 1; got {levels!r}")
    t = cast_to_schema(targets, XIC_TARGET_TABLE)
    d = t.to_pydict()
    n = t.num_rows
    want_ms1 = 1 in levels
    want_frag = any(level >= 2 for level in levels)

    modseq = d["modified_sequence"]
    charge_in = d["precursor_charge"]
    pmz_in = d["precursor_mz"]
    parsed: dict[str, Peptidoform] = {}

    def pf(s: str) -> Peptidoform:
        p = parsed.get(s)
        if p is None:
            p = parsed[s] = _peptidoform(s)
        return p

    # Fragments are checked first and in target order, as the extractor did
    # before matching anything; they are the only errors raised before MS1's.
    frag_targets: list[int] = []
    if want_frag:
        for i in range(n):
            if not modseq[i]:
                if not want_ms1:
                    raise ValueError(
                        f"target {d['target_id'][i]}: MS2 fragment extraction requires "
                        "modified_sequence (a bare-m/z target cannot define fragments)"
                    )
                continue
            if charge_in[i] is None:
                if not want_ms1:
                    raise ValueError(
                        f"target {d['target_id'][i]}: MS2 extraction requires precursor_charge"
                    )
                continue
            pf(modseq[i])
            frag_targets.append(i)

    precursor_charge = np.array([-1 if z is None else int(z) for z in charge_in], dtype=np.int16)
    pmz = np.array([np.nan if m is None else float(m) for m in pmz_in], dtype=np.float64)
    derived = np.zeros(n, dtype=bool)
    if derive_precursor_mz:
        for i in range(n):
            if pmz_in[i] is None and modseq[i] and charge_in[i] is not None:
                pmz[i] = _precursor_mz(pf(modseq[i]), charge=int(charge_in[i]))
                derived[i] = True

    if rt_window is not None:
        check_rt_window(t)
    rt_lo, rt_hi = _rt_bounds(t, rt_window)
    scan_in = d["scan"]
    assigned = np.array(
        [-1 if (not assigned_scans_only or s is None) else int(s) for s in scan_in],
        dtype=np.int64,
    )

    isotopes = _isotopes(d, n, channels, pf) if want_ms1 else None
    fragments = (
        _fragments(d, n, frag_targets, precursor_charge, channels, pf) if want_frag else None
    )
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
    )


def _isotopes(d: dict, n: int, settings: ChannelSettings, pf) -> Channels:
    offsets_k = settings.offsets()
    n_peaks = max(max(offsets_k) + 1, 1)
    neutral: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
    envelopes: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    mz_l: list[np.ndarray] = []
    z_l: list[np.ndarray] = []
    k_l: list[np.ndarray] = []
    ab_l: list[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        modseq = d["modified_sequence"][i]
        charge = d["precursor_charge"][i]
        if modseq:
            if settings.charge_range is not None:
                charges = list(range(settings.charge_range[0], settings.charge_range[1] + 1))
            elif charge is not None:
                charges = [int(charge)]
            else:
                lo, hi = settings.fallback_charge_range
                charges = list(range(lo, hi + 1))
            for z in charges:
                key = (modseq, z)
                if key not in envelopes:
                    # One envelope per peptide, put at each charge (what
                    # peptide_envelope does at one charge).
                    if modseq not in neutral:
                        masses, ab_e = neutral_envelope(pf(modseq), n_peaks=n_peaks)
                        neutral[modseq] = (masses, ab_e.numpy())
                    masses, ab_np = neutral[modseq]
                    envelopes[key] = (envelope_mz(masses, z).numpy(), ab_np)
                mz_e, ab_e = envelopes[key]
                mz = np.empty(len(offsets_k))
                ab = np.full(len(offsets_k), np.nan, dtype=np.float32)
                for j, k in enumerate(offsets_k):
                    if 0 <= k < mz_e.size:
                        mz[j] = mz_e[k]
                        ab[j] = ab_e[k]
                    else:
                        mz[j] = mz_e[0] + k * ISOTOPE_MASS_DIFF / z
                mz_l.append(mz)
                z_l.append(np.full(len(offsets_k), z, np.int8))
                k_l.append(np.array(offsets_k, np.int8))
                ab_l.append(ab)
                counts[i] += len(offsets_k)
            continue
        pmz = d["precursor_mz"][i]
        if pmz is None:
            raise ValueError(f"target {d['target_id'][i]}: needs modified_sequence or precursor_mz")
        z = int(charge) if charge is not None else 1
        ks = offsets_k if charge is not None else (0,)
        mz_l.append(np.array([pmz + k * ISOTOPE_MASS_DIFF / z for k in ks], dtype=np.float64))
        z_l.append(np.full(len(ks), z, np.int8))
        k_l.append(np.array(ks, np.int8))
        ab_l.append(np.full(len(ks), np.nan, np.float32))
        counts[i] += len(ks)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype)

    isotope = cat(k_l, np.int8)
    return Channels(
        kind=ISOTOPES,
        offsets=offsets,
        mz=cat(mz_l, np.float64),
        precursor_charge=cat(z_l, np.int8),
        isotope=isotope,
        ion_type=np.full(total, -1, np.int8),
        position=np.full(total, -1, np.int32),
        fragment_charge=np.full(total, -1, np.int32),
        loss=np.zeros(total, np.int16),
        abundance=cat(ab_l, np.float32),
        evidence_key=isotope.astype(np.int32),
    )


def _fragments(
    d: dict,
    n: int,
    frag_targets: list[int],
    precursor_charge: np.ndarray,
    settings: ChannelSettings,
    pf,
) -> Channels:
    # One ladder per distinct sequence; targets sharing a sequence share it.
    seqs = list(dict.fromkeys(d["modified_sequence"][i] for i in frag_targets))
    ladder = fragment_ladder_arrays_batch(
        [pf(s) for s in seqs],
        ion_types=settings.ion_types,
        max_fragment_charge=settings.max_fragment_charge,
        neutral_losses=settings.neutral_losses,
    )
    which = {s: k for k, s in enumerate(seqs)}
    rows: list[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int64)
    for i in frag_targets:
        k = which[d["modified_sequence"][i]]
        a, b = ladder.offsets[k], ladder.offsets[k + 1]
        ceiling = min(settings.max_fragment_charge, int(precursor_charge[i]))
        keep = a + np.flatnonzero(ladder.charge[a:b] <= ceiling)
        rows.append(keep)
        counts[i] = keep.size
    take = np.concatenate(rows) if rows else np.empty(0, np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    query = np.repeat(np.arange(n, dtype=np.int64), counts)
    ion_type = ladder.ion_type[take]
    position = ladder.position[take]
    # Rank of (ion type, position) within each query: one fragment ion seen
    # at two charges or with a loss is one piece of evidence.
    ion = (query << 24) | (ion_type.astype(np.int64) << 16) | position.astype(np.int64)
    distinct, key = np.unique(ion, return_inverse=True)
    first = np.searchsorted(distinct, np.arange(n, dtype=np.int64) << 24)
    evidence = (key - first[query]).astype(np.int32)
    return Channels(
        kind=FRAGMENTS,
        offsets=offsets,
        mz=ladder.mz[take],
        precursor_charge=np.repeat(precursor_charge.astype(np.int8), counts),
        isotope=np.zeros(take.size, np.int8),
        ion_type=ion_type,
        position=position,
        fragment_charge=ladder.charge[take],
        loss=ladder.loss[take],
        abundance=np.full(take.size, np.nan, np.float32),
        evidence_key=evidence,
        loss_ids=ladder.loss_ids,
    )
