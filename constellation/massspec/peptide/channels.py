"""Channels: the m/z values a query expects signal at, at every MS level.

One generator for every consumer that needs a peptide's theoretical ions
as signal to look for: the sweep's queries (``massspec.sweep``), the mass
index's extraction (``quant.chromatogram``, Mode B) and spectrum
annotation (``spectra.annotate``). Until 2026-11-07 each built its own,
with its own defaults.

A *channel* is one m/z with what it is: an isotope of the precursor at a
charge (matched in MS1 scans), or a fragment ion (ion type, bond position,
fragment charge, neutral loss; matched in MS2 and higher). Channels are
held per query in compressed-row form (:class:`Channels`), so nothing
exists per channel as a Python object. The same contract serves every
level; a per-level choice of what is matched (reporter ions in MS3, say)
is where it would extend.

**Abundance**, one convention at every level: a channel's share of its
group's whole modelled distribution, computed before any selection, so
choosing the top N never changes a value. The group is the precursor at
one charge for isotopes (its share of the full isotope envelope, from its
elemental composition, labels included), and the query for fragments.
``0`` means modelled and predicted absent; ``NaN`` means not modelled.
Where fragment abundance comes from is the caller's choice
(:class:`FragmentAbundance`): :data:`UNMODELLED` (NaN), :class:`Uniform`
(equal intensity for every ion of the series asked, the assumption of the
classic pre-Prosit DDA scores), or a library's predicted or observed
intensities (``massspec.library.fragment_abundance``).

**Top N** keeps a group's N most abundant channels, in their own order:
the most abundant isotopes (which for a heavy peptide need not include
M+0), or a library's most intense fragments (EncyclopeDIA traces 10). It
needs abundances that rank: refused for unmodelled or uniform fragments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from constellation.core.sequence.proforma import Peptidoform
from constellation.core.stats.units import ISOTOPE_MASS_DIFF
from constellation.massspec.peptide.envelope import neutral_envelopes
from constellation.massspec.peptide.ions import IonType, fragment_ladder_arrays_batch
from constellation.massspec.peptide.mz import PROTON_MASS

__all__ = [
    "ISOTOPES",
    "FRAGMENTS",
    "ChannelSettings",
    "Channels",
    "CandidateIons",
    "FragmentAbundance",
    "Unmodelled",
    "UNMODELLED",
    "Uniform",
    "isotope_channels",
    "fragment_channels",
    "select_top",
]

#: Channel kinds: precursor isotopes are matched in MS1 scans, fragments in the rest.
ISOTOPES = 1
FRAGMENTS = 2

#: Envelope peaks computed when the most abundant isotopes are chosen: wide
#: enough that the envelope's maximum and its neighbours are among them for
#: peptides of up to some 10 kDa.
_TOP_ISOTOPE_WIDTH = 16


@dataclass(frozen=True)
class ChannelSettings:
    """Which channels a query is matched at: the isotopes and charges of its
    precursor in MS1, the fragment ions in MS2."""

    n_isotopes: int = 3
    #: Isotope offsets to use instead of ``range(n_isotopes)``. Negative
    #: offsets (M−1, M−2) are for labelled peptides, whose incomplete
    #: labelling puts intensity below the fully labelled mass.
    isotope_offsets: tuple[int, ...] | None = None
    #: Keep each precursor's N most abundant isotopes (from its envelope)
    #: instead of ``n_isotopes`` / ``isotope_offsets``.
    top_isotopes: int | None = None
    #: Charges to look for every target at in MS1, for the evidence of an
    #: analyte across its charge states (a peptide's precursors then share
    #: their MS1 channels). None (the default) is each target's own charge
    #: (PI, 2026-11-07; until then (1, 4) was the default for every target).
    charge_range: tuple[int, int] | None = None
    #: The charges a target with no charge of its own is looked for at in
    #: MS1, when ``charge_range`` is not given.
    fallback_charge_range: tuple[int, int] = (1, 4)
    ion_types: tuple[IonType, ...] = (IonType.B, IonType.Y)
    #: Fragment charges run to ``max(1, min(max_fragment_charge, z −
    #: fragment_charge_below_precursor))`` for a precursor of charge z
    #: (to ``max_fragment_charge`` where there is no precursor charge).
    max_fragment_charge: int = 2
    fragment_charge_below_precursor: int = 0
    neutral_losses: tuple[str, ...] | None = None
    #: Keep each query's N most abundant fragment ions (needs abundances
    #: that rank: a library's).
    top_fragments: int | None = None

    def __post_init__(self) -> None:
        if self.isotope_offsets is not None and self.top_isotopes is not None:
            raise ValueError("give isotope_offsets or top_isotopes, not both")
        if self.top_isotopes is None and self.offsets() == ():
            raise ValueError(
                "the isotope selection is empty (n_isotopes < 1 or no isotope_offsets): MS1 would have no "
                "channels; leave level 1 out of the levels instead"
            )
        for name in ("top_isotopes", "top_fragments"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be >= 1; got {value}")
        for name in ("charge_range", "fallback_charge_range"):
            value = getattr(self, name)
            if value is not None and not (len(value) == 2 and 1 <= int(value[0]) <= int(value[1])):
                raise ValueError(f"{name} must be (low, high) with 1 <= low <= high; got {value!r}")
        if int(self.max_fragment_charge) < 1:
            raise ValueError(f"max_fragment_charge must be >= 1; got {self.max_fragment_charge}")
        if int(self.fragment_charge_below_precursor) < 0:
            raise ValueError("fragment_charge_below_precursor must be >= 0")
        if not self.ion_types:
            raise ValueError("ion_types must name at least one ion series")

    def offsets(self) -> tuple[int, ...]:
        if self.isotope_offsets is not None:
            return tuple(sorted(set(int(k) for k in self.isotope_offsets)))
        return tuple(range(int(self.n_isotopes)))

    def fragment_charge_ceiling(self, precursor_charge: np.ndarray) -> np.ndarray:
        """The highest fragment charge for each precursor charge (< 1 for
        none)."""
        z = np.asarray(precursor_charge, dtype=np.int64)
        capped = np.maximum(1, np.minimum(int(self.max_fragment_charge), z - int(self.fragment_charge_below_precursor)))
        return np.where(z >= 1, capped, int(self.max_fragment_charge))

    def to_dict(self) -> dict:
        """The settings as plain data (JSON), for a plan written to disk."""
        return {
            "n_isotopes": int(self.n_isotopes),
            "isotope_offsets": None if self.isotope_offsets is None else [int(k) for k in self.isotope_offsets],
            "top_isotopes": self.top_isotopes,
            "charge_range": None if self.charge_range is None else [int(z) for z in self.charge_range],
            "fallback_charge_range": [int(z) for z in self.fallback_charge_range],
            "ion_types": [IonType(t).name for t in self.ion_types],
            "max_fragment_charge": int(self.max_fragment_charge),
            "fragment_charge_below_precursor": int(self.fragment_charge_below_precursor),
            "neutral_losses": None if self.neutral_losses is None else list(self.neutral_losses),
            "top_fragments": self.top_fragments,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChannelSettings:
        def pair(v):
            return None if v is None else (int(v[0]), int(v[1]))

        return cls(
            n_isotopes=int(d["n_isotopes"]),
            isotope_offsets=None if d["isotope_offsets"] is None else tuple(int(k) for k in d["isotope_offsets"]),
            top_isotopes=d.get("top_isotopes"),
            charge_range=pair(d["charge_range"]),
            fallback_charge_range=pair(d["fallback_charge_range"]),
            ion_types=tuple(IonType[name] for name in d["ion_types"]),
            max_fragment_charge=int(d["max_fragment_charge"]),
            fragment_charge_below_precursor=int(d.get("fragment_charge_below_precursor", 0)),
            neutral_losses=None if d["neutral_losses"] is None else tuple(d["neutral_losses"]),
            top_fragments=d.get("top_fragments"),
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
    abundance: np.ndarray  # float32; share of the group's modelled whole, NaN where not modelled
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


def _ragged(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(counts)
    return np.arange(total, dtype=np.int64) + np.repeat(np.asarray(starts, dtype=np.int64) - (ends - counts), counts)


def _offsets(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def select_top(group: np.ndarray, abundance: np.ndarray, n: int) -> np.ndarray:
    """Which rows to keep: each group's ``n`` most abundant (NaN last,
    ties in row order), as a boolean mask in row order. ``group`` is
    non-decreasing (rows grouped)."""
    rows = np.arange(group.size, dtype=np.int64)
    key = np.where(np.isnan(abundance), -np.inf, abundance)
    order = np.lexsort((rows, -key, group))
    g = group[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if g.size else np.empty(0, np.int64)
    rank = np.arange(g.size) - np.repeat(starts, np.diff(np.r_[starts, g.size]))
    keep = np.zeros(group.size, dtype=bool)
    keep[order[rank < int(n)]] = True
    return keep


# ── isotopes ──────────────────────────────────────────────────────────


def isotope_channels(
    peptidoforms: Sequence[Peptidoform],
    peptide: np.ndarray,
    charge: np.ndarray,
    precursor_mz: np.ndarray,
    settings: ChannelSettings,
    *,
    target_ids: Sequence[Any] | None = None,
) -> Channels:
    """Each query's precursor isotopes: for a query with a peptide
    (``peptide[q]`` an index into ``peptidoforms``, -1 for none), its
    envelope at its own charge (``charge[q]``, -1 for none), or at each
    charge of ``settings.charge_range``, or of ``fallback_charge_range``
    for a query without one; for a bare m/z (``precursor_mz[q]``), isotopes
    spaced off it (only M+0 without a charge). ``target_ids`` name queries
    in errors.

    The m/z are the binned envelope's, value for value what
    :func:`~constellation.massspec.peptide.envelope.peptide_envelope`
    gives at each charge; abundance is each isotope's share of the full
    envelope (NaN for a bare m/z, and for an offset below M+0)."""
    peptide = np.asarray(peptide, dtype=np.int64)
    charge = np.asarray(charge, dtype=np.int64)
    precursor_mz = np.asarray(precursor_mz, dtype=np.float64)
    n = peptide.size
    ids = list(target_ids) if target_ids is not None else list(range(n))
    has_seq = peptide >= 0
    bare = ~has_seq
    missing = np.flatnonzero(bare & np.isnan(precursor_mz))
    if missing.size:
        raise ValueError(f"target {ids[int(missing[0])]}: needs modified_sequence or precursor_mz")

    top = settings.top_isotopes
    offsets_k = np.array(settings.offsets() if top is None else range(int(top)), dtype=np.int64)
    width = max(int(offsets_k.max()) + 1, 1) if top is None else max(_TOP_ISOTOPE_WIDTH, int(top))

    # Which charges each sequence query is looked for at.
    if settings.charge_range is not None:
        lo, hi = settings.charge_range
        per = np.full(n, hi - lo + 1, dtype=np.int64)
        first = np.full(n, lo, dtype=np.int64)
    else:
        own = charge >= 1
        lo, hi = settings.fallback_charge_range
        per = np.where(own, 1, hi - lo + 1).astype(np.int64)
        first = np.where(own, charge, lo).astype(np.int64)
    per = np.where(has_seq, per, 0)
    pair_query = np.repeat(np.arange(n, dtype=np.int64), per)
    pair_z = np.repeat(first, per) + _ragged(np.zeros(n, np.int64), per)
    pair_peptide = peptide[pair_query]

    # One envelope per distinct peptide, all at once, put at each of its charges.
    used = np.unique(pair_peptide)
    masses, shares = neutral_envelopes([peptidoforms[p] for p in used.tolist()], n_peaks=width)
    env = np.searchsorted(used, pair_peptide)
    z = pair_z.astype(np.float64)
    # (mass + z·proton) / z, the envelope_mz arithmetic, elementwise.
    mz_all = (masses[env] + (pair_z * PROTON_MASS)[:, None]) / z[:, None]

    if top is None:
        k = np.broadcast_to(offsets_k, (pair_z.size, offsets_k.size))
        inside = (k >= 0) & (k < width)
        kk = np.clip(k, 0, width - 1)
        rows = np.arange(pair_z.size)[:, None]
        mz = np.where(inside, mz_all[rows, kk], mz_all[:, :1] + (k * ISOTOPE_MASS_DIFF) / z[:, None])
        ab = np.where(inside, shares[env][rows, kk], np.nan)
        seq_k = k
    else:
        # The N most abundant envelope bins of each (precursor, charge),
        # in isotope order.
        order = np.argsort(-shares[env], axis=1, kind="stable")[:, : int(top)]
        seq_k = np.sort(order, axis=1)
        rows = np.arange(pair_z.size)[:, None]
        mz = mz_all[rows, seq_k]
        ab = shares[env][rows, seq_k]
    nk = seq_k.shape[1]
    seq_query = np.repeat(pair_query, nk)
    seq_mz = mz.reshape(-1)
    seq_z = np.repeat(pair_z, nk)
    seq_iso = np.asarray(seq_k).reshape(-1)
    seq_ab = ab.reshape(-1)

    # Bare m/z: isotopes spaced off it, M+0 only without a charge.
    bare_q = np.flatnonzero(bare)
    with_z = charge[bare_q] >= 1
    bare_ks = offsets_k if top is None else np.arange(int(top), dtype=np.int64)
    nb = np.where(with_z, bare_ks.size, 1).astype(np.int64)
    b_query = np.repeat(bare_q, nb)
    b_ordinal = _ragged(np.zeros(bare_q.size, np.int64), nb)
    b_k = np.where(np.repeat(with_z, nb), bare_ks[np.minimum(b_ordinal, bare_ks.size - 1)], 0)
    b_z = np.repeat(np.where(with_z, charge[bare_q], 1), nb)
    b_mz = precursor_mz[b_query] + (b_k * ISOTOPE_MASS_DIFF) / b_z

    query = np.concatenate([seq_query, b_query])
    order = np.argsort(query, kind="stable")
    isotope = np.concatenate([seq_iso, b_k])[order].astype(np.int8)
    total = order.size
    return Channels(
        kind=ISOTOPES,
        offsets=_offsets(np.bincount(query, minlength=n).astype(np.int64)),
        mz=np.concatenate([seq_mz, b_mz])[order].astype(np.float64),
        precursor_charge=np.concatenate([seq_z, b_z])[order].astype(np.int8),
        isotope=isotope,
        ion_type=np.full(total, -1, np.int8),
        position=np.full(total, -1, np.int32),
        fragment_charge=np.full(total, -1, np.int32),
        loss=np.zeros(total, np.int16),
        abundance=np.concatenate([seq_ab, np.full(b_mz.size, np.nan)])[order].astype(np.float32),
        evidence_key=isotope.astype(np.int32),
    )


# ── fragments ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateIons:
    """Every fragment ion the settings allow for each query, before any
    selection: what a :class:`FragmentAbundance` scores. Per query:
    ``key`` (the id a source looks abundances up by, such as the
    precursor id; -1 for none), ``precursor_charge`` and the peptide's
    ``length``; per ion, compressed by query like :class:`Channels`."""

    offsets: np.ndarray
    key: np.ndarray
    precursor_charge: np.ndarray
    length: np.ndarray
    ion_type: np.ndarray
    position: np.ndarray
    fragment_charge: np.ndarray
    loss: np.ndarray
    mz: np.ndarray
    loss_ids: tuple[str, ...]

    def query_of(self) -> np.ndarray:
        return np.repeat(np.arange(self.offsets.size - 1, dtype=np.int64), np.diff(self.offsets))


class FragmentAbundance(Protocol):
    """Where fragment abundances come from. ``ranks`` says whether they
    say which ions are more intense (so top-N can select on them);
    ``model_losses`` the neutral losses whose ions the source itself names
    (which get channels whatever the loss rules say); ``abundance`` one
    value per candidate ion (float, NaN where not modelled);
    ``describe`` what it is, for a plan's metadata."""

    ranks: bool

    def model_losses(self) -> tuple[str, ...]: ...

    def abundance(self, ions: CandidateIons) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Unmodelled:
    """No abundance: NaN for every ion (what every fragment channel had
    until 2026-11-07)."""

    ranks: bool = False

    def model_losses(self) -> tuple[str, ...]:
        return ()

    def abundance(self, ions: CandidateIons) -> np.ndarray:
        return np.full(ions.mz.size, np.nan)

    def describe(self) -> dict[str, Any]:
        return {"source": "unmodelled"}


UNMODELLED = Unmodelled()


@dataclass(frozen=True)
class Uniform:
    """Equal intensity for every ion of the series asked for, at every
    charge, with and without its losses: the assumption of the classic
    pre-Prosit DDA scores. Optional weights per ion type, per fragment
    charge and for a neutral loss (SEQUEST, for instance, weighs a-ions
    and losses below b and y). Each query's abundances are shares of its
    whole ladder (they sum to 1). They do not rank ions, so top-N cannot
    select on them."""

    ion_type_weights: Mapping[IonType, float] | None = None
    charge_weights: Mapping[int, float] | None = None
    loss_weight: float = 1.0
    ranks: bool = field(default=False, init=False)

    def model_losses(self) -> tuple[str, ...]:
        return ()

    def abundance(self, ions: CandidateIons) -> np.ndarray:
        w = np.ones(ions.mz.size)
        if self.ion_type_weights:
            table = np.ones(max(int(t) for t in IonType) + 1)
            for t, v in self.ion_type_weights.items():
                table[int(t)] = float(v)
            w = w * table[ions.ion_type.astype(np.int64)]
        if self.charge_weights:
            top = int(ions.fragment_charge.max()) + 1 if ions.fragment_charge.size else 1
            table = np.ones(max(top, max(self.charge_weights) + 1))
            for z, v in self.charge_weights.items():
                table[int(z)] = float(v)
            w = w * table[ions.fragment_charge.astype(np.int64)]
        w = np.where(ions.loss > 0, w * float(self.loss_weight), w)
        q = ions.query_of()
        total = np.bincount(q, weights=w, minlength=ions.offsets.size - 1)
        return w / np.where(total[q] > 0, total[q], 1.0)

    def describe(self) -> dict[str, Any]:
        return {
            "source": "uniform",
            "ion_type_weights": None if not self.ion_type_weights
            else {IonType(t).name: float(v) for t, v in self.ion_type_weights.items()},
            "charge_weights": None if not self.charge_weights
            else {str(z): float(v) for z, v in self.charge_weights.items()},
            "loss_weight": float(self.loss_weight),
        }


def fragment_channels(
    peptidoforms: Sequence[Peptidoform],
    peptide: np.ndarray,
    charge: np.ndarray,
    settings: ChannelSettings,
    abundance: FragmentAbundance = UNMODELLED,
    *,
    key: np.ndarray | None = None,
    target_ids: Sequence[Any] | None = None,
) -> Channels:
    """Each query's fragment ions: for a query with a peptide
    (``peptide[q]`` into ``peptidoforms``, -1 for none), its ladder of the
    settings' ion types and losses, at fragment charges up to the ceiling
    for its precursor charge (``charge[q]``; -1 for no cap but
    ``max_fragment_charge``). One ladder per distinct peptide. Abundance
    from ``abundance`` (looked up by ``key[q]``, such as the precursor
    id); with ``settings.top_fragments`` only each query's most abundant,
    in ladder order. Ladder order is position, then ion type as given,
    then charge, then no loss before each loss."""
    peptide = np.asarray(peptide, dtype=np.int64)
    charge = np.asarray(charge, dtype=np.int64)
    n = peptide.size
    ids = list(target_ids) if target_ids is not None else list(range(n))
    top = settings.top_fragments
    if top is not None and not abundance.ranks:
        raise ValueError(
            f"top_fragments needs abundances that rank ions (a fragment library); {abundance.describe()['source']} "
            f"abundances do not"
        )
    used = np.unique(peptide[peptide >= 0])
    ladder = fragment_ladder_arrays_batch(
        [peptidoforms[p] for p in used.tolist()],
        ion_types=settings.ion_types,
        max_fragment_charge=settings.max_fragment_charge,
        neutral_losses=settings.neutral_losses,
    )
    lengths_used = np.fromiter((len(peptidoforms[p].sequence) for p in used.tolist()), np.int64, used.size)
    has = peptide >= 0
    which = np.where(has, np.searchsorted(used, np.where(has, peptide, used[0] if used.size else 0)), 0)
    starts = np.where(has, ladder.offsets[which] if used.size else 0, 0)
    sizes = np.where(has, (ladder.offsets[which + 1] - ladder.offsets[which]) if used.size else 0, 0)
    take = _ragged(starts, sizes)
    row_query = np.repeat(np.arange(n, dtype=np.int64), sizes)
    ceiling = settings.fragment_charge_ceiling(charge)
    keep = ladder.charge[take] <= ceiling[row_query]
    take, row_query = take[keep], row_query[keep]
    counts = np.bincount(row_query, minlength=n).astype(np.int64)
    offsets = _offsets(counts)
    ions = CandidateIons(
        offsets=offsets,
        key=np.full(n, -1, np.int64) if key is None else np.asarray(key, dtype=np.int64),
        precursor_charge=charge,
        length=np.where(has, lengths_used[which] if used.size else 0, 0),
        ion_type=ladder.ion_type[take],
        position=ladder.position[take],
        fragment_charge=ladder.charge[take],
        loss=ladder.loss[take],
        mz=ladder.mz[take],
        loss_ids=ladder.loss_ids,
    )
    values = np.asarray(abundance.abundance(ions), dtype=np.float64)
    if top is not None:
        keep = select_top(row_query, values, int(top)) & (values > 0)
        empty = np.flatnonzero((counts > 0) & (np.bincount(row_query[keep], minlength=n) == 0))
        if empty.size:
            raise ValueError(
                f"target {ids[int(empty[0])]} (and {empty.size - 1} more) has no fragment ion with a positive "
                f"abundance to keep"
            )
        take, row_query, values = take[keep], row_query[keep], values[keep]
        counts = np.bincount(row_query, minlength=n).astype(np.int64)
        offsets = _offsets(counts)
    ion_type = ladder.ion_type[take]
    position = ladder.position[take]
    # Rank of (ion type, position) within each query: one fragment ion seen
    # at two charges or with a loss is one piece of evidence.
    ion = (row_query << 24) | (ion_type.astype(np.int64) << 16) | position.astype(np.int64)
    distinct, rank = np.unique(ion, return_inverse=True)
    first = np.searchsorted(distinct, np.arange(n, dtype=np.int64) << 24)
    evidence = (rank - first[row_query]).astype(np.int32)
    return Channels(
        kind=FRAGMENTS,
        offsets=offsets,
        mz=ladder.mz[take],
        precursor_charge=np.repeat(charge.astype(np.int8), counts),
        isotope=np.zeros(take.size, np.int8),
        ion_type=ion_type,
        position=position,
        fragment_charge=ladder.charge[take],
        loss=ladder.loss[take],
        abundance=values.astype(np.float32),
        evidence_key=evidence,
        loss_ids=ladder.loss_ids,
    )


def settings_json(settings: ChannelSettings) -> str:
    return json.dumps(settings.to_dict())
