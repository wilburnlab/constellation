"""Predictions put onto a library, property by property.

``predict_onto(library, {property: predictor}, collision_energy=...)``:

1. **Requests.** ``build_requests``: each precursor's unlabelled peptide
   and charge, asked once (``prediction.requests``).
2. **Checks before anything runs.** Every predictor says which requests
   it cannot take (``Predictor.check``). A precursor any of them refuses
   is dropped, and so is the rest of its group: the target precursor and
   every decoy made from it at that charge (PI, 2026-11-05), so targets
   and decoys stay matched for scoring and calibration. Each drop is
   counted by property and reason; a precursor dropped only because its
   partner was refused is counted apart.
3. **Predictions**, in chunks of requests in canonical order.
4. **One library**: the source library (its decoy pairing, spans,
   modification provenance) less the dropped precursors, with each
   predicted value put on the precursors that asked it: ``rt_predicted``
   (seconds on the predictor's scale, recorded), ``ccs_predicted`` (Å²),
   ``charge_probability``, and the fragments (``prediction.assemble``) in
   place of the source's. A value a predictor did not give (NaN) leaves
   the source's. The library's metadata records which predictor made each
   property, the collision energy per charge, the retention-time scale and
   the counts (``x.massspec.prediction.*``).

Deuterium shifts retention time and no model here knows it; such a
peptide is predicted as its light form (a caveat the report carries when
labels were removed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.library import Library
from constellation.massspec.library.prediction.assemble import Ladders, fragment_table
from constellation.massspec.library.prediction.base import (
    PROPERTIES,
    FragmentArrays,
    Predictor,
    Property,
    ValueArray,
)
from constellation.massspec.library.prediction.collision_energy import CollisionEnergy
from constellation.massspec.library.prediction.requests import RequestPlan, build_requests
from constellation.massspec.library.schemas import LIBRARY_FRAGMENT_TABLE

__all__ = ["DEFAULT_CHUNK", "PredictionReport", "predict_onto"]

#: Requests predicted at a time: bounds the memory of a chunk's model
#: outputs and fragments, not the result.
DEFAULT_CHUNK = 100_000
#: Library precursor columns each value property fills.
_COLUMN = {"retention_time": "rt_predicted", "ccs": "ccs_predicted", "charge_state": "charge_probability"}
_EXAMPLES = 5


@dataclass(frozen=True)
class PredictionReport:
    """What a prediction did: counts, drops by reason, the models and
    settings. ``to_dict()`` is what the library's metadata and a run's
    manifest record."""

    precursors: int
    precursors_kept: int
    requests: int
    requests_predicted: int
    fragments: int
    refused: dict[str, dict[str, int]]
    examples: dict[str, dict[str, list[str]]]
    dropped_with_partner: int
    labelled_peptides: int
    predictors: dict[str, dict[str, Any]]
    collision_energy: dict[str, Any] | None
    rt_scale: str | None
    caveats: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return self.precursors - self.precursors_kept

    def to_dict(self) -> dict[str, Any]:
        return {
            "precursors": self.precursors,
            "precursors_kept": self.precursors_kept,
            "dropped": self.dropped,
            "dropped_with_partner": self.dropped_with_partner,
            "requests": self.requests,
            "requests_predicted": self.requests_predicted,
            "fragments": self.fragments,
            "refused": self.refused,
            "examples": self.examples,
            "labelled_peptides": self.labelled_peptides,
            "predictors": self.predictors,
            "collision_energy": self.collision_energy,
            "rt_scale": self.rt_scale,
            "caveats": list(self.caveats),
        }


def _check_predictors(predictors: Mapping[str, Predictor], collision_energy: CollisionEnergy | None) -> None:
    if not predictors:
        raise ValueError("no predictors given")
    for prop, predictor in predictors.items():
        if prop not in PROPERTIES:
            raise ValueError(f"unknown property {prop!r} (one of {', '.join(PROPERTIES)})")
        if predictor.capabilities.property != prop:
            raise ValueError(f"{predictor.name} predicts {predictor.capabilities.property}, not {prop}")
        if predictor.capabilities.needs_collision_energy and collision_energy is None:
            raise ValueError(f"{predictor.name} needs a collision energy (collision_energy=)")


def predict_onto(
    library: Library,
    predictors: Mapping[Property, Predictor],
    *,
    collision_energy: CollisionEnergy | None = None,
    chunk: int = DEFAULT_CHUNK,
) -> tuple[Library, PredictionReport]:
    """``library`` with ``predictors``' properties put on its precursors
    (module docstring). Returns the library and the report."""
    _check_predictors(predictors, collision_energy)
    plan = build_requests(library, collision_energy)
    n = len(plan.precursor_id)

    # ── checks: which source precursors are refused, and why ────────
    refused = np.zeros(n, dtype=bool)
    counts: dict[str, dict[str, int]] = {}
    examples: dict[str, dict[str, list[str]]] = {}

    def note(prop: str, reason: str, detail: str) -> None:
        counts.setdefault(prop, {})[reason] = counts.get(prop, {}).get(reason, 0) + 1
        shown = examples.setdefault(prop, {}).setdefault(reason, [])
        if len(shown) < _EXAMPLES and detail not in shown:
            shown.append(detail)

    for row, (reason, detail) in plan.unrequestable.items():
        refused[row] = True
        note("input", reason, detail)
    has_request = plan.request >= 0
    for prop, predictor in predictors.items():
        if not len(plan.requests):
            break
        refusals = predictor.check(plan.requests)
        bad = refusals.refused
        hit = np.flatnonzero(has_request & ~refused & bad[np.where(has_request, plan.request, 0)])
        for row in hit.tolist():
            r = plan.request[row]
            note(prop, refusals.reasons[r], refusals.details[r])
        refused[hit] = True

    bad_groups = np.unique(plan.group[refused])
    dropped = np.isin(plan.group, bad_groups)
    kept_rows = np.flatnonzero(~dropped)

    # ── predictions, chunk by chunk ─────────────────────────────────
    kept_requests = np.unique(plan.request[kept_rows])
    values = {prop: np.full(len(kept_requests), np.nan) for prop in predictors if prop != "fragments"}
    fragment_parts: list[pa.Table] = []
    fragments_predictor = predictors.get("fragments")
    ladders: Ladders | None = None
    source_peptide = np.zeros(0, np.int64)
    if fragments_predictor is not None and len(kept_rows):
        ladders, source_peptide = _source_ladders(library, plan, kept_rows, fragments_predictor)
    kept_request_of_row = np.searchsorted(kept_requests, plan.request[kept_rows])
    for start in range(0, len(kept_requests), max(1, int(chunk))):
        stop = min(start + int(chunk), len(kept_requests))
        sub = plan.requests.take(kept_requests[start:stop])
        for prop, predictor in predictors.items():
            out = predictor.predict(sub)
            if prop == "fragments":
                rows = np.flatnonzero((kept_request_of_row >= start) & (kept_request_of_row < stop))
                fragment_parts.append(
                    fragment_table(
                        _as_fragments(out, predictor.name),
                        kept_request_of_row[rows] - start,
                        plan.precursor_id[kept_rows[rows]],
                        source_peptide[rows],
                        ladders,
                    )
                )
            else:
                values[prop][start:stop] = _as_values(out, predictor.name, len(sub)).values

    # ── one library ────────────────────────────────────────────────
    out = library.subset_precursors(pa.array(plan.precursor_id[kept_rows], type=pa.int64()))
    precursors = out.precursors
    at = pc.index_in(precursors.column("precursor_id"), value_set=pa.array(plan.precursor_id[kept_rows]))
    at = at.to_numpy(zero_copy_only=False).astype(np.int64)
    for prop, by_request in values.items():
        column = _COLUMN[prop]
        new = by_request[kept_request_of_row][at]
        old = precursors.column(column).to_numpy(zero_copy_only=False).astype(np.float64)
        # NaN is "not predicted": the source's value stays (null for a
        # charge probability, −1 for the others).
        merged = pa.array(np.where(np.isnan(new), old, new), type=pa.float64(), from_pandas=(prop == "charge_state"))
        precursors = precursors.set_column(precursors.schema.get_field_index(column), column, merged)
    if fragments_predictor is not None:
        fragments = pa.concat_tables(fragment_parts) if fragment_parts else LIBRARY_FRAGMENT_TABLE.empty_table()
        fragments = fragments.filter(
            pc.is_in(fragments.column("precursor_id"), value_set=precursors.column("precursor_id").combine_chunks())
        )
    else:
        fragments = out.fragments

    rt_predictor = predictors.get("retention_time")
    report = PredictionReport(
        precursors=n,
        precursors_kept=precursors.num_rows,
        requests=len(plan.requests),
        requests_predicted=len(kept_requests),
        fragments=fragments.num_rows,
        refused=counts,
        examples=examples,
        dropped_with_partner=int(n - precursors.num_rows - refused.sum()),
        labelled_peptides=plan.labelled_peptides,
        predictors={prop: p.identity() for prop, p in predictors.items()},
        collision_energy=(
            collision_energy.describe(sorted({int(z) for z in plan.requests.charge[kept_requests].tolist()}))
            if collision_energy is not None and fragments_predictor is not None
            else None
        ),
        rt_scale=rt_predictor.capabilities.rt_scale if rt_predictor is not None else None,
        caveats=(
            ["isotope labels were removed before prediction: fragment m/z are the labelled peptide's, "
             "intensities and retention times its light form's (deuterium's retention shift is not modelled)"]
            if plan.labelled_peptides
            else []
        ),
    )
    extras = {
        **out.metadata_extras,
        "x.massspec.prediction.predictors": report.predictors,
        "x.massspec.prediction.report": {k: v for k, v in report.to_dict().items() if k not in ("predictors", "examples")},
    }
    if report.collision_energy is not None:
        extras["x.massspec.prediction.collision_energy"] = report.collision_energy
    if report.rt_scale is not None:
        extras["x.massspec.prediction.rt_scale"] = report.rt_scale
    return replace(out, precursors=precursors, fragments=fragments, metadata_extras=extras), report


def _as_fragments(out: Any, name: str) -> FragmentArrays:
    if not isinstance(out, FragmentArrays):
        raise TypeError(f"{name} returned {type(out).__name__} for fragments, not FragmentArrays")
    return out


def _as_values(out: Any, name: str, n: int) -> ValueArray:
    if not isinstance(out, ValueArray) or len(out.values) != n:
        raise TypeError(f"{name} must return a ValueArray of {n} values")
    return out


def _source_ladders(
    library: Library, plan: RequestPlan, kept_rows: np.ndarray, predictor: Predictor
) -> tuple[Ladders, np.ndarray]:
    """Ladders of the kept precursors' own (labelled) peptides, and each
    kept precursor's index into them."""
    peptide_ids = plan.peptide_id[kept_rows]
    unique_ids, local = np.unique(peptide_ids, return_inverse=True)
    peptides = library.peptides
    at = pc.index_in(pa.array(unique_ids, type=pa.int64()), value_set=peptides.column("peptide_id").combine_chunks())
    sequences = pc.take(peptides.column("modified_sequence"), at).to_pylist()
    caps = predictor.capabilities
    ladders = Ladders([parse_proforma(s) for s in sequences], caps.ion_types, caps.max_fragment_charge)
    return ladders, local.astype(np.int64)
