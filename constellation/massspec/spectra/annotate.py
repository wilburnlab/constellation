"""Annotate an observed spectrum with a peptide's fragment ions.

:func:`annotate_spectrum` labels every peak of one spectrum that lies
within tolerance of a theoretical ion of the peptide (b/y by default,
optionally other series, higher charges and neutral losses) or of the
intact precursor. Labels are mzPAF (``y7``, ``b5^2``, ``y3-[H2O]``, ``p^2``);
the ion's identity also rides as columns, with the matching error. A peak
near several ions takes the nearest.

:func:`ion_ladder` lists the theoretical ions themselves, numbered as
labels number them, which is what an ion table or a sequence-coverage
view draws from.

Both take the peptide as a ProForma 2.0 string, the form experiment
tables store (``PEPTIDE_TABLE.modified_sequence``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pyarrow as pa

ANNOTATED_PEAK_SCHEMA = pa.schema(
    [
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        #: mzPAF, null for a peak no ion explains.
        pa.field("annotation", pa.string()),
        #: "a" / "b" / "c" / "x" / "y" / "z", or "p" for the precursor.
        pa.field("ion_type", pa.string()),
        #: The n of b_n / y_n; null for the precursor.
        pa.field("ion_number", pa.int16()),
        pa.field("fragment_charge", pa.int8()),
        pa.field("loss", pa.string()),
        pa.field("mz_theoretical", pa.float64()),
        #: Observed minus theoretical.
        pa.field("error_da", pa.float64()),
        pa.field("error_ppm", pa.float64()),
    ],
    metadata={b"schema_name": b"AnnotatedPeaks"},
)

ION_LADDER_SCHEMA = pa.schema(
    [
        pa.field("ion_type", pa.string(), nullable=False),
        pa.field("ion_number", pa.int16(), nullable=False),
        #: 0-based peptide bond the ion breaks (the fragment tables' ``position``).
        pa.field("position", pa.int16(), nullable=False),
        pa.field("fragment_charge", pa.int8(), nullable=False),
        pa.field("loss", pa.string()),
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("annotation", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"IonLadder"},
)

_LETTERS = "abcxyz"


def _ion_types(names: Sequence[str]):
    from constellation.massspec.peptide.ions import IonType

    try:
        return tuple(IonType[n.upper()] for n in names)
    except KeyError as exc:
        raise ValueError(f"unknown ion type {exc.args[0]!r}; use a, b, c, x, y or z") from None


def default_fragment_charge(precursor_charge: int) -> int:
    """Fragment charges to consider for a precursor: up to one below its
    own, at least 1 and at most 3."""
    return max(1, min(3, precursor_charge - 1))


def ion_ladder(
    modified_sequence: str,
    *,
    ion_types: Sequence[str] = ("b", "y"),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
) -> pa.Table:
    """The peptide's theoretical fragment ions (``ION_LADDER_SCHEMA``),
    sorted by series, number and charge."""
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.annotation.mzpaf import fragment_row_to_mzpaf
    from constellation.massspec.peptide.ions import fragment_ladder, ion_number

    peptide = parse_proforma(modified_sequence)
    length = len(peptide.sequence)
    table, _ = fragment_ladder(
        peptide,
        ion_types=_ion_types(ion_types),
        max_fragment_charge=max_fragment_charge,
        neutral_losses=neutral_losses,
        return_tensor=False,
    )
    rows = []
    for r in table.to_pylist():
        rows.append(
            {
                "ion_type": _LETTERS[r["ion_type"]],
                "ion_number": ion_number(r["ion_type"], r["position"], length),
                "position": r["position"],
                "fragment_charge": r["charge"],
                "loss": r["loss_id"],
                "mz": r["mz_theoretical"],
                "annotation": fragment_row_to_mzpaf(r, length=length),
            }
        )
    rows.sort(key=lambda r: (r["ion_type"], r["ion_number"], r["fragment_charge"], r["loss"] or ""))
    return pa.Table.from_pylist(rows, schema=ION_LADDER_SCHEMA)


def annotate_spectrum(
    mz: np.ndarray | pa.Array | Sequence[float],
    intensity: np.ndarray | pa.Array | Sequence[float],
    modified_sequence: str,
    *,
    charge: int,
    tolerance: float,
    tolerance_unit: Literal["ppm", "Da"],
    ion_types: Sequence[str] = ("b", "y"),
    max_fragment_charge: int | None = None,
    neutral_losses: Sequence[str] | None = None,
    include_precursor: bool = True,
) -> pa.Table:
    """One row per peak (``ANNOTATED_PEAK_SCHEMA``), in the order given.

    ``max_fragment_charge`` defaults to :func:`default_fragment_charge` of
    ``charge``. With ``include_precursor`` a peak at the intact precursor's
    m/z (at ``charge``) that no fragment explains is labelled ``p^charge``.
    """
    import torch

    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.annotation.mzpaf import fragment_row_to_mzpaf
    from constellation.massspec.peptide.ions import ion_number
    from constellation.massspec.peptide.match import assign_fragments, match_mz
    from constellation.massspec.peptide.mz import precursor_mz

    obs_mz = np.asarray(mz, dtype=np.float64)
    obs_int = np.asarray(intensity, dtype=np.float64)
    if obs_mz.shape != obs_int.shape:
        raise ValueError("mz and intensity differ in length")
    n = obs_mz.size
    columns: dict[str, list] = {name: [None] * n for name in ANNOTATED_PEAK_SCHEMA.names}
    columns["mz"] = obs_mz.tolist()
    columns["intensity"] = obs_int.tolist()
    if n == 0:
        return pa.table(columns, schema=ANNOTATED_PEAK_SCHEMA)

    peptide = parse_proforma(modified_sequence)
    length = len(peptide.sequence)
    assignments = assign_fragments(
        peptide,
        torch.from_numpy(obs_mz.copy()),
        torch.from_numpy(obs_int.copy()),
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        ion_types=_ion_types(ion_types),
        max_fragment_charge=max_fragment_charge or default_fragment_charge(charge),
        neutral_losses=neutral_losses,
    )
    best: dict[int, object] = {}
    for a in assignments:  # nearest ion for each peak
        held = best.get(a.obs_idx)
        if held is None or abs(a.error_da) < abs(held.error_da):  # type: ignore[attr-defined]
            best[a.obs_idx] = a
    for i, a in best.items():
        row = {"ion_type": int(a.ion_type), "position": a.position, "charge": a.charge, "loss_id": a.loss_id}
        columns["annotation"][i] = fragment_row_to_mzpaf(row, length=length)
        columns["ion_type"][i] = _LETTERS[int(a.ion_type)]
        columns["ion_number"][i] = ion_number(int(a.ion_type), a.position, length)
        columns["fragment_charge"][i] = a.charge
        columns["loss"][i] = a.loss_id
        columns["mz_theoretical"][i] = a.theo_mz
        columns["error_da"][i] = a.error_da
        columns["error_ppm"][i] = a.error_ppm

    if include_precursor:
        target = precursor_mz(peptide, charge=charge)
        for m in match_mz(
            torch.from_numpy(obs_mz.copy()),
            torch.tensor([target], dtype=torch.float64),
            tolerance=tolerance,
            tolerance_unit=tolerance_unit,
        ):
            i = m.query_idx
            if columns["annotation"][i] is not None:
                continue
            columns["annotation"][i] = "p" if charge == 1 else f"p^{charge}"
            columns["ion_type"][i] = "p"
            columns["fragment_charge"][i] = charge
            columns["mz_theoretical"][i] = target
            columns["error_da"][i] = m.error_da
            columns["error_ppm"][i] = m.error_ppm
    return pa.table(columns, schema=ANNOTATED_PEAK_SCHEMA)


__all__ = [
    "ANNOTATED_PEAK_SCHEMA",
    "ION_LADDER_SCHEMA",
    "annotate_spectrum",
    "default_fragment_charge",
    "ion_ladder",
]
