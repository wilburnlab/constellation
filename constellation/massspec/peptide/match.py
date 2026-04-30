"""Peak-list ↔ peak-list matching with tolerance windows.

Two layers:

  ``match_mz(query_mz, ref_mz, ...)`` — a peptidoform-agnostic primitive.
  Pairs each query m/z to its closest reference m/z within a ppm/Da
  tolerance window. Reusable for any two-peak-list comparison: observed
  spectrum vs theoretical fragments, observed vs observed (replicate /
  inter-instrument), theoretical vs theoretical (e.g. comparing two
  isotope envelopes).

  ``assign_fragments(peptidoform, obs_mz, ...)`` — peptidoform-aware
  wrapper. Generates the theoretical fragment ladder via
  :func:`fragment_ladder`, calls ``match_mz``, and attaches fragment-ion
  identity (``ion_type`` / ``position`` / ``charge`` / ``loss_id``) to
  each match. The verb "assign" is field-canonical (Skyline,
  Spectronaut, mzPAF all use "ion assignment" for "say what fragment
  identity an observed peak corresponds to").

The matcher uses ``torch.searchsorted`` against the sorted reference
list — O((n_query + n_ref) log n_ref) — rather than the
O(n_query × n_ref) broadcast-grid version cartographer prototyped.
The grid version is fine on dlib-scale spectra (~200 peaks × ~50
theoretical) but scales catastrophically on full mzML scans
(thousands of peaks).

Degenerate-theoretical exception, by request: when two reference
peaks fall inside one query's tolerance window AND are themselves
within tolerance of each other (i.e. the references are mutually
indistinguishable at this resolution), the query intensity is
assigned to *both* matches. This is EncyclopeDIA's default behaviour
and a defensible "good enough" approximation in the absence of a
prior on per-ion contribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import ProFormaResult
from constellation.massspec.peptide.ions import IonType, fragment_ladder

# ──────────────────────────────────────────────────────────────────────
# Match records
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MzMatch:
    """One (query, reference) m/z pair within tolerance.

    ``query_idx`` and ``ref_idx`` index back into the *original* (input)
    arrays — sorting that ``match_mz`` does internally is hidden from
    the caller.
    """

    query_idx: int
    ref_idx: int
    query_mz: float
    ref_mz: float
    query_intensity: float  # NaN when no intensity passed
    error_da: float  # signed (query - ref); natural unit for low-res
    error_ppm: float  # signed (query - ref) / ref * 1e6; natural unit for high-res


@dataclass(frozen=True, slots=True)
class IonAssignment:
    """One observed peak ↔ theoretical fragment-ion match.

    Extends ``MzMatch`` with fragment identity. ``obs_*`` are aliased
    onto ``query_*`` for readability — when matching an observed
    spectrum against theoretical fragments, "obs" is more natural than
    "query".
    """

    obs_idx: int
    obs_mz: float
    obs_intensity: float
    theo_mz: float
    ion_type: IonType
    position: int
    charge: int
    loss_id: str | None
    error_da: float
    error_ppm: float


# ──────────────────────────────────────────────────────────────────────
# Layer 1 — generic m/z matcher
# ──────────────────────────────────────────────────────────────────────


def match_mz(
    query_mz: torch.Tensor,
    ref_mz: torch.Tensor,
    *,
    tolerance: float = 20.0,
    tolerance_unit: Literal["ppm", "Da"] = "ppm",
    query_intensity: torch.Tensor | None = None,
) -> list[MzMatch]:
    """Match each ``query_mz`` to the closest ``ref_mz`` within tolerance.

    Vectorized via ``torch.searchsorted`` — O((n_query + n_ref) log n_ref).
    The candidate slice for each query is found with two searchsorted
    calls (`query_mz - tol` and `query_mz + tol`), and the per-query
    Python loop selects the best match (or all matches in the
    mutually-degenerate case).

    Parameters
    ----------
    query_mz, ref_mz
        1-D float tensors. Either can be unsorted; sorting is done
        internally on a copy of ``ref_mz``.
    tolerance, tolerance_unit
        Tolerance window. ``"ppm"`` is the high-res default; ``"Da"``
        is the right call on low-res / unit-mass-resolution data
        where ppm windows are vacuously wide at low m/z.
    query_intensity
        Optional. Carried through to ``MzMatch.query_intensity``.
        ``NaN`` is filled in when omitted.

    Returns
    -------
    list[MzMatch]
        Zero or more matches per query. The query→reference assignment
        is injective except in the degenerate-reference case
        (two ref peaks within tolerance of each other AND both within
        tolerance of one query → both emit the same query intensity).
    """
    if query_mz.ndim != 1 or ref_mz.ndim != 1:
        raise ValueError("query_mz and ref_mz must be 1-D")
    if query_intensity is not None and query_intensity.shape != query_mz.shape:
        raise ValueError("query_intensity must match query_mz shape")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive; got {tolerance}")
    if tolerance_unit not in ("ppm", "Da"):
        raise ValueError(
            f"tolerance_unit must be 'ppm' or 'Da'; got {tolerance_unit!r}"
        )

    if query_mz.numel() == 0 or ref_mz.numel() == 0:
        return []

    # Work in float64 throughout — peak m/z is precision-sensitive.
    qmz = query_mz.to(torch.float64)
    rmz = ref_mz.to(torch.float64)

    # Sort references; track the inverse permutation so MzMatch.ref_idx
    # points back into the *input* array.
    rmz_sorted, ref_sort_idx = torch.sort(rmz)
    ref_sort_idx_list = ref_sort_idx.tolist()

    # Per-query tolerance vector.
    if tolerance_unit == "ppm":
        tol = qmz * (tolerance * 1e-6)
    else:  # Da
        tol = torch.full_like(qmz, float(tolerance))

    # Candidate slice [left[i], right[i]) for each query i.
    left = torch.searchsorted(rmz_sorted, qmz - tol, side="left")
    right = torch.searchsorted(rmz_sorted, qmz + tol, side="right")

    qmz_list = qmz.tolist()
    rmz_sorted_list = rmz_sorted.tolist()
    left_list = left.tolist()
    right_list = right.tolist()
    tol_list = tol.tolist()
    if query_intensity is None:
        qint_list = [float("nan")] * len(qmz_list)
    else:
        qint_list = query_intensity.to(torch.float64).tolist()

    out: list[MzMatch] = []
    for i, q in enumerate(qmz_list):
        lo, hi = left_list[i], right_list[i]
        n_cand = hi - lo
        if n_cand == 0:
            continue
        if n_cand == 1:
            sorted_j = lo
            _emit(out, i, sorted_j, q, rmz_sorted_list, ref_sort_idx_list, qint_list)
            continue
        # 2+ candidates: mutual-degeneracy check.
        span = rmz_sorted_list[hi - 1] - rmz_sorted_list[lo]
        if span <= tol_list[i]:
            for sorted_j in range(lo, hi):
                _emit(out, i, sorted_j, q, rmz_sorted_list, ref_sort_idx_list, qint_list)
        else:
            # Pick the closest reference.
            best_j = lo
            best_diff = abs(rmz_sorted_list[lo] - q)
            for sorted_j in range(lo + 1, hi):
                d = abs(rmz_sorted_list[sorted_j] - q)
                if d < best_diff:
                    best_diff = d
                    best_j = sorted_j
            _emit(out, i, best_j, q, rmz_sorted_list, ref_sort_idx_list, qint_list)
    return out


def _emit(
    out: list[MzMatch],
    query_idx: int,
    sorted_ref_idx: int,
    qmz: float,
    rmz_sorted_list: list[float],
    ref_sort_idx_list: list[int],
    qint_list: list[float],
) -> None:
    rmz = rmz_sorted_list[sorted_ref_idx]
    ref_idx = ref_sort_idx_list[sorted_ref_idx]
    err_da = qmz - rmz
    err_ppm = (err_da / rmz) * 1e6 if rmz != 0.0 else float("nan")
    out.append(
        MzMatch(
            query_idx=query_idx,
            ref_idx=ref_idx,
            query_mz=qmz,
            ref_mz=rmz,
            query_intensity=qint_list[query_idx],
            error_da=err_da,
            error_ppm=err_ppm,
        )
    )


# ──────────────────────────────────────────────────────────────────────
# Layer 2 — peptidoform-aware assignment
# ──────────────────────────────────────────────────────────────────────


def assign_fragments(
    peptidoform: ProFormaResult,
    obs_mz: torch.Tensor,
    obs_intensity: torch.Tensor | None = None,
    *,
    tolerance: float = 20.0,
    tolerance_unit: Literal["ppm", "Da"] = "ppm",
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
    vocab: ModVocab = UNIMOD,
) -> list[IonAssignment]:
    """Assign theoretical fragment ions to observed peaks.

    Calls :func:`fragment_ladder(peptidoform, ...)` to enumerate
    theoretical fragments, then ``match_mz`` to pair each observed peak
    to the closest theoretical ion within tolerance. Returns
    ``IonAssignment`` records carrying full fragment identity.
    """
    table, mz_tensor = fragment_ladder(
        peptidoform,
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
        neutral_losses=neutral_losses,
        return_tensor=False,
        vocab=vocab,
    )
    if mz_tensor is None:
        # Use the table's mz_theoretical column directly.
        theo_mz = torch.tensor(
            table.column("mz_theoretical").to_pylist(), dtype=torch.float64
        )
    else:  # pragma: no cover  (return_tensor=False above)
        theo_mz = mz_tensor

    if theo_mz.numel() == 0:
        return []

    ion_type_codes = table.column("ion_type").to_pylist()
    positions = table.column("position").to_pylist()
    charges = table.column("charge").to_pylist()
    loss_ids = table.column("loss_id").to_pylist()

    matches = match_mz(
        obs_mz,
        theo_mz,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        query_intensity=obs_intensity,
    )

    out: list[IonAssignment] = []
    for m in matches:
        out.append(
            IonAssignment(
                obs_idx=m.query_idx,
                obs_mz=m.query_mz,
                obs_intensity=m.query_intensity,
                theo_mz=m.ref_mz,
                ion_type=IonType(int(ion_type_codes[m.ref_idx])),
                position=int(positions[m.ref_idx]),
                charge=int(charges[m.ref_idx]),
                loss_id=loss_ids[m.ref_idx],
                error_da=m.error_da,
                error_ppm=m.error_ppm,
            )
        )
    return out


__all__ = [
    "MzMatch",
    "IonAssignment",
    "match_mz",
    "assign_fragments",
]
