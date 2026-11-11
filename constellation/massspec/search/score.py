"""Scorer v1: a KL trace over MS2, its elution peaks found by a peak picker.

For each precursor query, on its MS2 dense view (``QueryBatch.dense``,
``[queries, scans, channels]``), the scorer (PI, 2026-11-10):

1. takes each channel's observed intensity where it was acquired
   (``Dense.observed``) and the library's predicted share of it (the
   channel's abundance; a channel the library does not model is left out);
2. optionally sums each channel over ``±half_window`` slots;
3. computes at every slot ``KL(observed shares ‖ library shares)`` over the
   channels acquired there (``core.stats.losses.kld``, both renormalized
   over those channels, smoothed by ``pseudocount``). That is the per-ion
   multinomial deviance: ``G² = 2·N·KL`` for N ions in the observed
   distribution. A slot where fewer than ``min_observed`` channels have
   any intensity has no evidence;
4. makes it a detection trace (``exp(−KL)``, 0 without evidence, or
   ``−KL`` clipped at ``kl_cap``) and finds its peaks with a
   ``core.signal.peaks.PeakPicker``, bounding them on an intensity
   consensus of the fragments (the k-th largest of each channel's intensity
   over its predicted share, among channels predicted at ``min_share`` of
   the query's strongest or more: k fragments must co-elute for the
   consensus to rise).

Every candidate gets a row of :data:`CANDIDATE_TABLE`, its ``score`` the
negative KL at the apex on the picker's smoothed trace: its KL at the apex
(with and without the sliding sum, and on the picker's smoothed trace), the
KL of the spectrum summed over its extent, the picker's measures, its
retention times and apex scan, its ions and mass error. The best by the
picker's score is ``chosen``; a query with no candidate still gets one
row, with null scores, so that the number of targets and of decoys scored
is known: also one with no scan to score (its RT interval outside the run,
its precursor in no isolation window) or no channel. ``candidate_fragments``
holds the chosen candidate's per-channel intensities as measured, for
quantities, whether the library models the channel or not.

It runs as a sweep reducer (:class:`KlScorer`, ``zero_fills``), best on
the precursor-major layout; :func:`score_acquisition` compiles the queries,
sweeps and returns the outputs with how they were made in their metadata.
The detection settings' defaults are set by the GPF benchmark
(``scripts/bench-scorer-v1.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F

from constellation.core.signal.consensus import collapse
from constellation.core.signal.peaks import GaussianSmooth, LocalMaxima, PeakPicker, WalkOut
from constellation.core.stats.losses import kld, pearson_correlation
from constellation.massspec.search.schemas import CANDIDATE_FRAGMENT_TABLE, CANDIDATE_TABLE

__all__ = ["ScorerSettings", "KlScorer", "score_acquisition", "FEATURE_SET"]

#: What the candidates' columns are, recorded in their metadata.
FEATURE_SET = {"name": "kl-trace", "version": 1}


def _default_picker() -> PeakPicker:
    # A Gaussian smoother, not Savitzky-Golay: where the pattern matches over
    # a stretch the trace is flat-topped, and a polynomial smoother rings at
    # the stretch's edges, making peaks of them.
    return PeakPicker(detect=LocalMaxima(prominence=0.05), smooth=GaussianSmooth(1.0),
                      boundaries=WalkOut(rise=2), max_peaks=5, recenter=3)  # fmt: skip


@dataclass(frozen=True)
class ScorerSettings:
    """How the KL trace is made and read (module docstring). Windows are
    in slots of the query's MS2 axis."""

    half_window: int = 0
    min_observed: int = 3
    pseudocount: float = 1e-3
    transform: str = "exp_neg"  # "exp_neg" or "neg"
    kl_cap: float = 5.0
    picker: PeakPicker = field(default_factory=_default_picker)
    k: int = 3
    min_share: float = 0.1
    min_coelution: float = 0.75
    max_cells: int = 1 << 22

    def as_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in ("half_window", "min_observed", "pseudocount", "transform", "kl_cap",
                                             "k", "min_share", "min_coelution", "max_cells")}  # fmt: skip
        out["picker"] = self.picker.identity()
        return out


def _box(values: torch.Tensor, h: int) -> torch.Tensor:
    """Sum over ``±h`` along dim 1 of ``[Q, S, C]``."""
    if h <= 0:
        return values
    q, s, c = values.shape
    x = values.permute(0, 2, 1).reshape(q * c, 1, s)
    k = torch.ones(1, 1, 2 * h + 1, dtype=values.dtype)
    return F.conv1d(x, k, padding=h).reshape(q, c, s).permute(0, 2, 1)


def _gather_slot(t: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    """``t [Q, S, ...]`` at slots ``at [Q, M]`` → ``[Q, M, ...]``."""
    index = at.clamp(min=0)
    shape = (*index.shape, *t.shape[2:])
    view = index.reshape(*index.shape, *([1] * (t.dim() - 2))).expand(shape)
    return torch.gather(t, 1, view)


@dataclass(frozen=True)
class KlScorer:
    """The scorer as a sweep reducer (module docstring)."""

    settings: ScorerSettings = field(default_factory=ScorerSettings)
    acquisition_id: int = 0
    pass_: int = 1
    role: str = "search"
    level: int = 2
    name: ClassVar[str] = "kl_scorer"
    schema: ClassVar[pa.Schema] = CANDIDATE_TABLE
    outputs: ClassVar[dict[str, pa.Schema]] = {"candidates": CANDIDATE_TABLE,
                                                "candidate_fragments": CANDIDATE_FRAGMENT_TABLE}  # fmt: skip
    gated: ClassVar[bool] = False
    zero_fills: ClassVar[bool] = True

    def reduce(self, batch) -> dict[str, pa.Table]:
        candidates, fragments = [], []
        for chunk in batch.dense_chunks(self.settings.max_cells, levels=[self.level]):
            d = batch.dense(self.level, queries=chunk, dtype=torch.float64)
            c, f = self._score(batch, d)
            candidates.append(c)
            fragments.append(f)
        return {
            "candidates": pa.concat_tables(candidates) if candidates else CANDIDATE_TABLE.empty_table(),
            "candidate_fragments": pa.concat_tables(fragments) if fragments else CANDIDATE_FRAGMENT_TABLE.empty_table(),
        }

    # ── one chunk ─────────────────────────────────────────────────────

    def _score(self, batch, d) -> tuple[pa.Table, pa.Table]:
        s = self.settings
        q, n_s, n_c = d.shape
        p = d.abundance.to(torch.float64)
        model = d.channel_valid & torch.isfinite(p) & (p > 0)
        # Per query, taken by query row: through the candidate axis, whose
        # width is the chunk's, a query with no candidate counted 0 alone
        # and its true count beside one with a candidate (Codex review).
        n_channels = model.sum(-1).numpy()
        if n_s == 0 or n_c == 0:
            # No query of the chunk has a scan (its RT interval outside the
            # run, its precursor in no isolation window), or none has a
            # channel (all removed by the channel settings): none has a
            # candidate, and each gets its row. One such query among others
            # has no valid slot or channel, so no peak, on the path below.
            return self._rows(batch, d, np.zeros((q, 0), bool), {}, n_channels), CANDIDATE_FRAGMENT_TABLE.empty_table()
        # What was measured, modelled or not: each channel's intensity where
        # its cell was acquired (the fragments' quantities). The KL and the
        # consensus take only the channels the library models (``o``).
        observed = torch.nan_to_num(d.observed(), nan=0.0)
        measured = torch.where(d.acquired & d.slot_valid.unsqueeze(-1), observed, torch.zeros_like(observed))
        kept = d.acquired & model.unsqueeze(1) & d.slot_valid.unsqueeze(-1)
        o = torch.where(kept, measured, torch.zeros_like(measured))
        lib = torch.where(model, p, torch.zeros_like(p)).unsqueeze(1).expand(q, n_s, n_c)

        def kl_at(h: int) -> tuple[torch.Tensor, torch.Tensor]:
            summed = _box(o, h)
            # A channel counts at a slot only if every valid slot of its window acquired it.
            if h > 0:
                have = _box(kept.to(torch.float64), h)
                slots = _box(d.slot_valid.to(torch.float64).unsqueeze(-1).expand(q, n_s, n_c), h)
                ok = kept & (have == slots)
            else:
                ok = kept
            n_obs = ((summed > 0) & ok).sum(-1)
            kl = kld(pred=lib, target=summed, mask=ok, pseudocount=s.pseudocount, pseudocount_units="probability",
                     reduce="none")  # fmt: skip
            evidence = d.slot_valid & (n_obs >= s.min_observed)
            return torch.where(evidence, kl, torch.full_like(kl, float("nan"))), n_obs

        kl, n_obs = kl_at(s.half_window)
        kl0, _ = kl_at(0) if s.half_window > 0 else (kl, n_obs)
        if s.transform == "exp_neg":
            trace = torch.where(torch.isnan(kl), torch.zeros_like(kl), torch.exp(-kl))
        elif s.transform == "neg":
            trace = torch.where(torch.isnan(kl), torch.full_like(kl, -s.kl_cap), (-kl).clamp(min=-s.kl_cap))
        else:
            raise ValueError(f"transform must be 'exp_neg' or 'neg', not {s.transform!r}")

        # The intensity consensus the peaks are bounded and measured on.
        # Channels predicted at least min_share of the query's strongest:
        # an absolute share would leave a long peptide's top ions none.
        top = torch.where(model, p, torch.zeros_like(p)).amax(-1, keepdim=True)
        strong = model & (p >= s.min_share * top)
        scale = torch.where(strong, 1.0 / torch.where(strong, p, torch.ones_like(p)), torch.full_like(p, float("nan")))
        consensus, _ = collapse(o.permute(0, 2, 1), valid=(kept & strong.unsqueeze(1)).permute(0, 2, 1),
                                method="kth_largest", k=s.k, scale=scale)  # fmt: skip
        consensus = torch.where(d.slot_valid, consensus, torch.full_like(consensus, float("nan")))
        peaks = s.picker(trace, x=d.rt, valid=d.slot_valid, boundary_trace=consensus, boundary_valid=d.slot_valid)
        m = peaks.valid.shape[-1]
        apex = peaks.apex.clamp(min=0)
        b_apex = peaks.boundary_apex.clamp(min=0)
        left, right = peaks.left.clamp(min=0), peaks.right.clamp(min=0)
        slot = torch.arange(n_s).view(1, 1, n_s)
        inside = (slot >= left.unsqueeze(-1)) & (slot <= right.unsqueeze(-1)) & d.slot_valid.unsqueeze(1)  # [Q,M,S]

        # KL of the spectrum summed over each candidate's extent.
        summed = torch.einsum("qms,qsc->qmc", inside.to(torch.float64), o)
        ok_ext = model.unsqueeze(1) & (torch.einsum("qms,qsc->qmc", inside.to(torch.float64), kept.to(torch.float64)) > 0)
        kl_ext = kld(pred=torch.where(model, p, torch.zeros_like(p)).unsqueeze(1).expand(q, m, n_c), target=summed,
                     mask=ok_ext, pseudocount=s.pseudocount, pseudocount_units="probability", reduce="none")  # fmt: skip
        if s.picker.smooth is not None:
            smooth_trace = s.picker.smooth(trace, d.slot_valid)
        else:
            smooth_trace = trace
        smoothed = torch.gather(torch.nan_to_num(smooth_trace, nan=0.0), 1, apex)
        if s.transform == "exp_neg":
            kl_smooth = -torch.log(smoothed.clamp(min=1e-300))
        else:
            kl_smooth = -smoothed

        # Co-elution: each channel's correlation with the consensus inside the extent.
        traces = o.permute(0, 2, 1).unsqueeze(1).expand(q, m, n_c, n_s)
        cons = torch.nan_to_num(consensus, nan=0.0).unsqueeze(1).unsqueeze(2).expand(q, m, n_c, n_s)
        mask = inside.unsqueeze(2) & kept.permute(0, 2, 1).unsqueeze(1)
        r = pearson_correlation(traces, cons, mask=mask)
        coeluting = ((r >= s.min_coelution) & model.unsqueeze(1)).sum(-1)

        # Abundance-weighted signed mass error at the apex.
        err = _gather_slot(d.mz_error_ppm.to(torch.float64), b_apex)  # [Q, M, C]
        hit = _gather_slot(d.matched, b_apex) & model.unsqueeze(1)
        weight = torch.where(hit, p.unsqueeze(1).expand(q, m, n_c), torch.zeros_like(err))
        total = weight.sum(-1)
        ppm = torch.where(total > 0, (torch.nan_to_num(err) * weight).sum(-1) / total.clamp(min=1e-300),
                          torch.full_like(total, float("nan")))  # fmt: skip

        # The KL features at the KL trace's own apex; the mass error and the
        # fragments' intensities at the intensity consensus's.
        features = {
            # The score is the KL on the picker's smoothed trace at the apex:
            # on the GPF bench it separated EncyclopeDIA's identifications
            # from decoys better than the raw KL there in every setting (AUC
            # 0.825 against 0.797 without an RT window, 0.874 against 0.867
            # with one).
            "score": -kl_smooth,
            "detector_score": peaks.score,
            "kl_apex": torch.gather(kl, 1, apex),
            "kl_apex_h0": torch.gather(kl0, 1, apex),
            "kl_apex_smoothed": kl_smooth,
            "kl_integrated": kl_ext,
            "rt_apex": torch.gather(torch.nan_to_num(d.rt, nan=float("nan")), 1, b_apex),
            "rt_left": peaks.left_x,
            "rt_right": peaks.right_x,
            "apex_scan": torch.gather(d.scan, 1, b_apex),
            "rt_kl_apex": peaks.apex_x,
            "kl_apex_scan": torch.gather(d.scan, 1, apex),
            "height": peaks.height,
            "prominence": peaks.prominence,
            "snr": peaks.snr,
            "width": peaks.width,
            "area": peaks.area,
            "n_observed": torch.gather(n_obs, 1, apex),
            "n_coeluting": coeluting,
            "ms2_ppm": ppm,
        }
        valid = peaks.valid.numpy()
        return (self._rows(batch, d, valid, features, n_channels),
                self._fragments(batch, d, valid, b_apex, inside, measured, model))  # fmt: skip

    def _rows(self, batch, d, valid: np.ndarray, features: dict[str, torch.Tensor], n_channels: np.ndarray) -> pa.Table:
        """The chunk's rows of :data:`CANDIDATE_TABLE`: one per candidate
        (``valid [Q, M]``, measured by ``features``, each ``[Q, M]``), and
        one, rank -1 with its measures null, for a query with none.
        ``n_channels [Q]`` is per query."""
        plan, targets, local = batch.plan, batch.plan.targets, d.queries
        n_cand = valid.sum(-1)
        qi, ri = np.nonzero(valid)
        none = np.flatnonzero(n_cand == 0)
        rows_q = np.concatenate([qi, none])
        rows_m = np.concatenate([ri, np.zeros(none.size, np.int64)])
        has = np.concatenate([np.ones(qi.size, bool), np.zeros(none.size, bool)])
        by = np.lexsort((rows_m, rows_q))
        rows_q, rows_m, has = rows_q[by], rows_m[by], has[by]
        at = pa.array(local[rows_q])

        def measure(name: str) -> pa.Array:
            typ = CANDIDATE_TABLE.field(name).type
            if not has.any():  # no query of the chunk has a candidate: ``features`` may be [Q, 0], or none
                return pa.nulls(rows_q.size, typ)
            values = features[name].numpy()[rows_q, rows_m]
            if pa.types.is_floating(typ):
                values = values.astype(np.float64)
                return pa.array(values, typ, mask=~has | np.isnan(values))
            return pa.array(values.astype(np.int64), typ, mask=~has)

        rt_library = targets.column("rt_library").take(at).to_numpy(zero_copy_only=False).astype(np.float64)
        col = {
            "acquisition_id": pa.array(np.full(rows_q.size, self.acquisition_id, np.int64)),
            "query": pa.array(d.global_queries[rows_q].astype(np.int64)),
            **{name: targets.column(name).take(at)
               for name in ("target_id", "precursor_id", "peptide_id", "modified_sequence", "is_decoy", "decoy_set")},
            "precursor_charge": pa.array(plan.precursor_charge[local][rows_q].astype(np.int8)),
            "pass": pa.array(np.full(rows_q.size, self.pass_, np.int8)),
            "role": pa.array([self.role] * rows_q.size, pa.string()),
            "rank": pa.array(np.where(has, rows_m, -1).astype(np.int32)),
            "chosen": pa.array((rows_m == 0)),
            "n_candidates": pa.array(n_cand[rows_q].astype(np.int32)),
            "rt_library": pa.array(rt_library, pa.float64(), mask=np.isnan(rt_library)),
            "n_channels": pa.array(n_channels[rows_q].astype(np.int32)),
        }  # fmt: skip
        return pa.table({f.name: col[f.name] if f.name in col else measure(f.name) for f in CANDIDATE_TABLE},
                        schema=CANDIDATE_TABLE)  # fmt: skip

    def _fragments(self, batch, d, valid, b_apex, inside, measured, model) -> pa.Table:
        """The chosen candidate's channels (:data:`CANDIDATE_FRAGMENT_TABLE`):
        each one's measured intensity at its apex and summed over its
        extent, whether the library models the channel or not."""
        chosen = np.flatnonzero(valid[:, 0]) if valid.shape[1] else np.empty(0, np.int64)
        if not chosen.size:
            return CANDIDATE_FRAGMENT_TABLE.empty_table()
        at = b_apex[chosen, 0]
        apex_int = measured[chosen, at]  # [k, C]
        ext = torch.einsum("ks,ksc->kc", inside[chosen, 0].to(torch.float64), measured[chosen])
        fq, fc = np.nonzero((d.channel_valid[chosen]).numpy())
        fq_full = chosen[fq]
        abundance = d.abundance.numpy()[fq_full, fc]
        return pa.table({
            "acquisition_id": pa.array(np.full(fq.size, self.acquisition_id, np.int64)),
            "query": pa.array(d.global_queries[fq_full].astype(np.int64)),
            "target_id": batch.plan.targets.column("target_id").take(pa.array(d.queries[fq_full])),
            "channel": pa.array(fc.astype(np.int32)),
            "mz_theoretical": pa.array(d.mz.numpy()[fq_full, fc]),
            "ion_type": pa.array(d.ion_type.numpy()[fq_full, fc].astype(np.int8)),
            "position": pa.array(d.position.numpy()[fq_full, fc].astype(np.int32)),
            "fragment_charge": pa.array(d.fragment_charge.numpy()[fq_full, fc].astype(np.int32)),
            "abundance": pa.array(abundance.astype(np.float32), mask=~np.isfinite(abundance)),
            "modelled": pa.array(model.numpy()[fq_full, fc]),
            "intensity_apex": pa.array(apex_int.numpy()[fq, fc]),
            "intensity_summed": pa.array(ext.numpy()[fq, fc]),
        }, schema=CANDIDATE_FRAGMENT_TABLE)  # fmt: skip


def score_acquisition(
    bundle: Path | str,
    targets: pa.Table,
    *,
    fragment_abundance,
    acquisition_id: int = 0,
    tolerance: tuple[float, str] = (0.35, "Da"),
    rule=None,
    channels=None,
    settings: ScorerSettings | None = None,
    rt_window: float | None = None,
    layout: str = "precursor_major",
    workers: int = 1,
    work_dir: Path | str | None = None,
    spill_dir: Path | str | None = None,
    pass_: int = 1,
    role: str = "search",
):
    """Score ``targets`` in one acquisition (module docstring): compile their
    MS2 queries with ``channels`` (``ChannelSettings.for_scoring()`` by
    default), the library's ``fragment_abundance`` and the bundle's acquired
    m/z range per window (top-N among the ions acquired), sweep ``bundle`` with
    :class:`KlScorer`, and return the ``SweepResult`` whose ``candidates``
    and ``candidate_fragments`` carry the feature set, the picker and the
    scorer's settings (and a ``core.stages`` key over them) in their metadata."""
    from constellation.core.stages import stage_key
    from constellation.massspec.peptide.channels import ChannelSettings
    from constellation.massspec.sweep import compile_queries
    from constellation.massspec.sweep.engine import SweepSettings, sweep
    from constellation.massspec.sweep.rules import Nearest

    from constellation.massspec.acquisition_structure import read_acquired_ranges

    settings = settings or ScorerSettings()
    channels = channels or ChannelSettings.for_scoring()
    rule = rule or Nearest()
    # Top-N among the ions each window acquired (review C6): a channel never
    # measured would take a library slot and give the KL nothing.
    plan = compile_queries(targets, levels=[2], channels=channels, rt_window=rt_window,
                           fragment_abundance=fragment_abundance,
                           acquired_ranges=read_acquired_ranges(bundle))  # fmt: skip
    sweep_settings = SweepSettings(tolerances={2: tuple(tolerance)}, rule=rule, acquisition_id=acquisition_id)
    scorer = KlScorer(settings=settings, acquisition_id=acquisition_id, pass_=pass_, role=role)
    result = sweep(bundle, plan, sweep_settings, reducers=[scorer], layout=layout, workers=workers,
                   work_dir=work_dir, spill_dir=spill_dir)  # fmt: skip
    record = {
        "feature_set": FEATURE_SET,
        "settings": settings.as_dict(),
        "channels": channels.as_dict() if hasattr(channels, "as_dict") else repr(channels),
        "tolerance": list(tolerance),
        "rule": rule.name,
        "pass": pass_,
        "role": role,
    }
    stamp = {
        b"x.massspec.feature_set": json.dumps(FEATURE_SET).encode(),
        b"x.constellation.peak_picker": json.dumps(settings.picker.identity()).encode(),
        b"x.massspec.scorer": json.dumps(record, default=str).encode(),
        b"x.massspec.scorer.key": stage_key(**json.loads(json.dumps(record, default=str))).encode(),
    }
    for name in ("candidates", "candidate_fragments"):
        result.metadata[name] = {**result.metadata.get(name, {}), **stamp}
    return result


def with_picker(settings: ScorerSettings, **changes) -> ScorerSettings:
    """``settings`` with its picker's stages replaced (``detect=...``, ...)."""
    return replace(settings, picker=replace(settings.picker, **changes))
