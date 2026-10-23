"""Retention-time alignment: one run's times onto another's, or a library's.

A monotone map from a *reference* time axis (a library's retention
times, or another run's) to a run's *observed* times, fitted through
where the paired times are densest so that wrong identifications barely
move it (``docs/plans/rt-alignment.md``).

Methods (``align(method=...)``):

  ``global``        the globally best strictly monotone path through the
                    density, as a smooth invertible spline, refined
                    against the pairs themselves (``refine="modal"``);
                    the default, the most accurate in the benchmark
                    (PI, 2026-10-22).
  ``encyclopedia``  EncyclopeDIA 6.9.23's fit, replayed bit for bit
                    (:mod:`.encyclopedia`).
  ``greedy``        the same algorithm in generic form, with padding,
                    trimming and a robust bandwidth available
                    (``core.signal.alignment``).

``refine`` defaults to each method's own (``modal`` for ``global``, none
for the others); ``refine=None`` gives the unrefined global path, and
``"density"`` refines against the density instead. Beyond the
calibration range the map follows an extrapolation policy, ``soft``
linear tails by default (``clamp`` is the jar's behaviour).

Pairs come from :mod:`.pairs`; :func:`align_to_library` and
:func:`align_runs` read them from a search experiment. Nothing here
writes into an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch

from constellation.core.signal.alignment import (
    GlobalPathOptions,
    GreedyOptions,
    MonotoneAlignment,
    RefineOptions,
    fit_monotone_alignment,
)
from constellation.core.stats.monotone import Extrapolation, TailedCurve
from constellation.massspec.rt.encyclopedia import EncyclopediaFit, fit_encyclopedia
from constellation.massspec.rt.pairs import (
    PAIR_SCHEMA,
    engine_pairs,
    library_pairs,
    run_pairs,
)

__all__ = [
    "RtAlignment",
    "align",
    "align_to_library",
    "align_runs",
    "to_rt_alignment_table",
    "PAIR_SCHEMA",
    "library_pairs",
    "run_pairs",
    "engine_pairs",
]

Method = Literal["encyclopedia", "greedy", "global"]


@dataclass(frozen=True)
class RtAlignment:
    """A fitted retention-time map, in seconds on both sides.

    ``alignment`` is the fit in its own units: reference and observed
    times divided by ``reference_scale`` / ``observed_scale`` (60 for the
    EncyclopeDIA replay, which fits in minutes as the jar does).
    ``reference_unit`` is ``"s"``, or ``"iRT"`` for a predicted scale with
    no unit. ``encyclopedia`` holds the replay's own record (its float32
    band, kernel and apex) when that method was used.
    """

    alignment: MonotoneAlignment
    method: str
    reference_unit: str
    reference_scale: float
    observed_scale: float
    n_pairs: int
    encyclopedia: EncyclopediaFit | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def to_observed(self, reference_rt) -> torch.Tensor:
        """Where each reference time falls in the run (seconds)."""
        x = torch.as_tensor(reference_rt, dtype=torch.float64) / self.reference_scale
        with torch.no_grad():
            return self.alignment.forward(x) * self.observed_scale

    def to_reference(self, observed_rt) -> torch.Tensor:
        """The reference time of each observed time."""
        y = torch.as_tensor(observed_rt, dtype=torch.float64) / self.observed_scale
        with torch.no_grad():
            return self.alignment.inverse(y) * self.reference_scale

    def residuals(self, reference_rt, observed_rt) -> torch.Tensor:
        """``observed − to_observed(reference)`` in seconds."""
        return torch.as_tensor(observed_rt, dtype=torch.float64) - self.to_observed(reference_rt)

    def within_band(self, reference_rt, observed_rt) -> torch.Tensor:
        """Inside the fit's middle-95 % residual band."""
        x = torch.as_tensor(reference_rt, dtype=torch.float64) / self.reference_scale
        y = torch.as_tensor(observed_rt, dtype=torch.float64) / self.observed_scale
        if self.encyclopedia is not None:
            return self.encyclopedia.within_band(x, y)
        with torch.no_grad():
            return self.alignment.within_band(x, y)

    def is_extrapolated(self, reference_rt) -> torch.Tensor:
        x = torch.as_tensor(reference_rt, dtype=torch.float64) / self.reference_scale
        return self.alignment.is_extrapolated(x)

    @property
    def band(self) -> tuple[float, float]:
        """The residual band in seconds."""
        lo, hi = self.alignment.band
        return lo * self.observed_scale, hi * self.observed_scale


def _encyclopedia_alignment(x, y, policy: Extrapolation, coverage: float) -> tuple[MonotoneAlignment, EncyclopediaFit]:
    fit = fit_encyclopedia(x, y, coverage=coverage)
    within = fit.within_band(x, y)
    tailed = TailedCurve.build(fit.curve, policy, x_support=x[within])
    diagnostics = {"model": fit.model, "sigma_cells": fit.sigma, "radius": fit.radius, "apex": fit.apex}
    return MonotoneAlignment(tailed, "encyclopedia", fit.band, int(x.numel()), diagnostics), fit


def align(
    pairs: pa.Table,
    *,
    method: Method = "global",
    refine: Literal["auto", "density", "modal"] | None = "auto",
    curve: Literal["linear", "rational_quadratic"] | None = None,
    extrapolation: Extrapolation | Literal["soft", "linear", "clamp"] = "soft",
    coverage: float = 0.95,
    weight_column: str | None = None,
    reference_unit: str = "s",
    options: GreedyOptions | GlobalPathOptions | None = None,
    refine_options: RefineOptions | None = None,
) -> RtAlignment:
    """Fit ``observed_rt ≈ f(reference_rt)`` to a table of pairs.

    By default the global path refined by the modal objective; see the
    module docstring for the methods. ``pairs`` has ``reference_rt`` and ``observed_rt`` (seconds; see
    :mod:`.pairs`), in the order the fit should see them — the
    EncyclopeDIA replay is order-sensitive, as the jar is.
    ``reference_unit="iRT"`` marks a reference on a predicted scale (it is
    then not converted). ``weight_column`` weights each pair's
    contribution to the density (not for ``encyclopedia``).
    """
    policy = Extrapolation.coerce(extrapolation)
    ref = torch.from_numpy(pairs.column("reference_rt").to_numpy(zero_copy_only=False).astype(np.float64))
    obs = torch.from_numpy(pairs.column("observed_rt").to_numpy(zero_copy_only=False).astype(np.float64))
    if method == "encyclopedia":
        if refine not in ("auto", None) or curve is not None or weight_column is not None or options is not None:
            raise ValueError("method='encyclopedia' is the jar's fit and takes no curve, refine, weights or options")
        ref_scale = 60.0 if reference_unit == "s" else 1.0
        alignment, fit = _encyclopedia_alignment(ref / ref_scale, obs / 60.0, policy, coverage)
        return RtAlignment(alignment, method, reference_unit, ref_scale, 60.0, pairs.num_rows, fit)
    weights = None
    if weight_column is not None:
        weights = torch.from_numpy(pairs.column(weight_column).to_numpy(zero_copy_only=False).astype(np.float64))
    alignment = fit_monotone_alignment(
        ref,
        obs,
        weights,
        method=method,
        curve=curve,
        refine=refine,
        extrapolation=policy,
        coverage=coverage,
        options=options,
        refine_options=refine_options,
    )
    return RtAlignment(alignment, method, reference_unit, 1.0, 1.0, pairs.num_rows)


def _read_rt_alignment(experiment_dir: Path, *, unit: str, acquisition_id: int) -> pa.Table:
    from constellation.massspec.manifest import open_table

    try:
        dataset = open_table(experiment_dir, "rt_alignment")
    except FileNotFoundError:
        return pa.table({"pass": pa.array([], pa.string())})
    condition = (ds.field("unit") == unit) & (ds.field("acquisition_id") == acquisition_id)
    return dataset.to_table(filter=condition)


def align_to_library(
    experiment_dir: Path | str,
    *,
    unit: str,
    acquisition_id: int,
    precursors: pa.Table | None = None,
    max_qvalue: float | None = 0.01,
    per: Literal["peptide", "precursor"] = "peptide",
    reference_unit: str = "s",
    **fit,
) -> RtAlignment:
    """A search's library → run alignment.

    Without ``precursors``, refits the points the engine itself fitted
    (the experiment's ``rt_alignment`` rows for that search and run, its
    final pass), so ``method="encyclopedia"`` reproduces the engine's fit
    and the other methods are compared on the engine's own evidence.
    With ``precursors`` (a ``PRECURSOR_TABLE`` with content-derived ids,
    e.g. a predicted library), pairs its times with the run's
    identifications at ``max_qvalue``.
    """
    root = Path(experiment_dir)
    if precursors is None:
        pairs = engine_pairs(_read_rt_alignment(root, unit=unit, acquisition_id=acquisition_id))
    else:
        from constellation.massspec.results import identifications

        ids = identifications(root, unit=unit, acquisition_id=acquisition_id, max_qvalue=max_qvalue)
        pairs = library_pairs(ids, precursors, max_qvalue=max_qvalue, per=per)
    if pairs.num_rows == 0:
        raise ValueError(f"no retention-time pairs for search {unit!r}, acquisition {acquisition_id}")
    return align(pairs, reference_unit=reference_unit, **fit)


def align_runs(
    experiment_dir: Path | str,
    *,
    reference_acquisition_id: int,
    acquisition_id: int,
    reference_search: str | None = None,
    search: str | None = None,
    max_qvalue: float | None = 0.01,
    per: Literal["peptide", "precursor"] = "peptide",
    **fit,
) -> RtAlignment:
    """One run's times onto another's, from both runs' identifications.

    ``reference_search`` / ``search`` name the search (unit) to read for
    each run when an acquisition was searched more than once.
    """
    from constellation.massspec.results import identifications

    root = Path(experiment_dir)
    ref = identifications(root, unit=reference_search, acquisition_id=reference_acquisition_id, max_qvalue=max_qvalue)
    tgt = identifications(root, unit=search, acquisition_id=acquisition_id, max_qvalue=max_qvalue)
    pairs = run_pairs(ref, tgt, max_qvalue=max_qvalue, per=per)
    if pairs.num_rows == 0:
        raise ValueError(f"acquisitions {reference_acquisition_id} and {acquisition_id} share no identifications")
    return align(pairs, **fit)


def to_rt_alignment_table(
    alignment: RtAlignment,
    pairs: pa.Table,
    *,
    acquisition_id: int | None,
    pass_name: str | None = None,
) -> pa.Table:
    """``RT_ALIGNMENT`` rows for a library → run fit (``engine="constellation"``).

    ``pass`` is ``pass_name`` or the method. A reference that is not in
    seconds is refused: the table's times are seconds.
    """
    from constellation.massspec.search.schemas import RT_ALIGNMENT

    if alignment.reference_unit != "s":
        raise ValueError(f"RT_ALIGNMENT holds seconds; this alignment's reference is {alignment.reference_unit!r}")
    ref = pairs.column("reference_rt").to_numpy(zero_copy_only=False).astype(np.float64)
    obs = pairs.column("observed_rt").to_numpy(zero_copy_only=False).astype(np.float64)
    fitted = alignment.to_observed(torch.from_numpy(ref)).numpy()
    within = alignment.within_band(torch.from_numpy(ref), torch.from_numpy(obs)).numpy()
    n = pairs.num_rows
    return pa.table(
        {
            "acquisition_id": pa.array([acquisition_id] * n, pa.int64()),
            "engine": pa.array(["constellation"] * n, pa.string()),
            "pass": pa.array([pass_name or alignment.method] * n, pa.string()),
            "modified_sequence": pairs.column("modified_sequence"),
            "peptide_id": pairs.column("peptide_id"),
            "charge": pairs.column("charge"),
            "library_rt": pa.array(ref),
            "observed_rt": pa.array(obs),
            "fitted_rt": pa.array(fitted),
            "residual": pa.array(obs - fitted),
            "within_95": pa.array(within),
            "is_decoy": pairs.column("is_decoy"),
        },
        schema=RT_ALIGNMENT,
    )

