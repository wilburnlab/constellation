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
    """Match observed peaks to theoretical b/y ladder; emit LIBRARY_FRAGMENT_TABLE rows.

    Theoretical ions with no observed match are dropped (we only emit
    rows for fragments that were actually observed). Orphan observed
    peaks (outside any tolerance window) are dropped.
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

    return [
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


__all__ = ["annotate_peaks"]
