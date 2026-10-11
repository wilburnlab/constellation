"""Encyclopedia-specific projection: peak list → LIBRARY_FRAGMENT_TABLE rows.

Thin wrapper around :func:`massspec.peptide.match.assign_fragments`.
The matching primitive is generic; this file's only job is the
schema projection (``IonAssignment`` → row dict for
``LIBRARY_FRAGMENT_TABLE``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from constellation.core.sequence.proforma import ProFormaResult
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.peptide.match import assign_fragments


def annotate_peaks_counted(
    peptidoform: ProFormaResult,
    *,
    precursor_charge: int,
    obs_mz: torch.Tensor,
    obs_intensity: torch.Tensor,
    tolerance_ppm: float = 20.0,
    max_fragment_charge: int | None = None,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
) -> tuple[list[dict[str, Any]], int]:
    """Match observed peaks to the theoretical b/y ladder.

    Returns the LIBRARY_FRAGMENT_TABLE rows and the number of distinct
    observed peaks they came from. The two differ when one peak lies
    within tolerance of two theoretical ions that are themselves within
    tolerance of each other (a singly and a doubly charged ion of the
    same m/z): :func:`~constellation.massspec.peptide.match.match_mz`
    emits a row for each, carrying the same intensity. So "peaks not
    annotated" is the stored count minus the second value, not minus
    the number of rows.

    Theoretical ions with no observed match are dropped (rows exist
    only for fragments that were actually observed). Observed peaks
    outside every tolerance window are dropped too.
    """
    if max_fragment_charge is None:
        max_fragment_charge = max(1, min(precursor_charge, 2))

    assignments = assign_fragments(
        peptidoform,
        obs_mz,
        obs_intensity,
        tolerance=tolerance_ppm,
        tolerance_unit="ppm",
        ion_types=ion_types,
        max_fragment_charge=max_fragment_charge,
    )

    rows = [
        {
            "ion_type": int(a.ion_type),
            "position": a.position,
            "charge": a.charge,
            "loss_id": a.loss_id,
            "mz_theoretical": a.theo_mz,
            "intensity_predicted": float(a.obs_intensity),
            "annotation": None,
        }
        for a in assignments
    ]
    return rows, len({a.obs_idx for a in assignments})


def annotate_peaks(
    peptidoform: ProFormaResult,
    *,
    precursor_charge: int,
    obs_mz: torch.Tensor,
    obs_intensity: torch.Tensor,
    tolerance_ppm: float = 20.0,
    max_fragment_charge: int | None = None,
    ion_types: Sequence[IonType] = (IonType.B, IonType.Y),
) -> list[dict[str, Any]]:
    """The rows of :func:`annotate_peaks_counted`, without the count."""
    rows, _n_peaks = annotate_peaks_counted(
        peptidoform,
        precursor_charge=precursor_charge,
        obs_mz=obs_mz,
        obs_intensity=obs_intensity,
        tolerance_ppm=tolerance_ppm,
        max_fragment_charge=max_fragment_charge,
        ion_types=ion_types,
    )
    return rows


__all__ = ["annotate_peaks", "annotate_peaks_counted"]
