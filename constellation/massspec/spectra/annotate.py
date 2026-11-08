"""Annotate an observed spectrum with a peptide's fragment ions.

:func:`annotate_spectrum` labels every peak of one spectrum that lies
within tolerance of a theoretical ion of the peptide (b/y by default,
optionally other series, higher charges and neutral losses) or of the
intact precursor. Labels are mzPAF (``y7``, ``b5^2``, ``y3-[H2O]``, ``p^2``);
the ion's identity also rides as columns, with the matching error. A peak
near several ions takes the nearest.

:func:`ion_ladder` lists the theoretical ions themselves, numbered as
labels number them, which is what an ion table or a sequence-coverage
view draws from. :func:`precursor_isotopes` and
:func:`theoretical_fragment_targets` give the m/z values to trace for a
peptide: its precursor's isotopes, and fragments to follow when no
library spectrum says which are intense.

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


def _one_peptide_fragments(modified_sequence: str, *, ion_types, max_fragment_charge: int,
                            neutral_losses, precursor_charge: int = -1):
    """One peptide's fragment channels from the shared generator
    (``massspec.peptide.channels``), and the peptide."""
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.channels import ChannelSettings, fragment_channels

    peptide = parse_proforma(modified_sequence)
    settings = ChannelSettings(
        ion_types=_ion_types(ion_types), max_fragment_charge=max_fragment_charge,
        neutral_losses=tuple(neutral_losses) if neutral_losses else None,
    )  # fmt: skip
    channels = fragment_channels([peptide], np.array([0]), np.array([precursor_charge]), settings)
    return channels, peptide


def ion_ladder(
    modified_sequence: str,
    *,
    ion_types: Sequence[str] = ("b", "y"),
    max_fragment_charge: int = 1,
    neutral_losses: Sequence[str] | None = None,
) -> pa.Table:
    """The peptide's theoretical fragment ions (``ION_LADDER_SCHEMA``),
    sorted by series, number and charge: the channels the sweep would
    trace, from the same generator."""
    from constellation.massspec.annotation.mzpaf import fragment_row_to_mzpaf
    from constellation.massspec.peptide.ions import ion_number

    channels, peptide = _one_peptide_fragments(modified_sequence, ion_types=ion_types,
                                               max_fragment_charge=max_fragment_charge,
                                               neutral_losses=neutral_losses)  # fmt: skip
    length = len(peptide.sequence)
    names = [None, *channels.loss_ids]
    rows = []
    for j in range(channels.n):
        r = {"ion_type": int(channels.ion_type[j]), "position": int(channels.position[j]),
             "charge": int(channels.fragment_charge[j]), "loss_id": names[int(channels.loss[j])]}  # fmt: skip
        rows.append(
            {
                "ion_type": _LETTERS[r["ion_type"]],
                "ion_number": ion_number(r["ion_type"], r["position"], length),
                "position": r["position"],
                "fragment_charge": r["charge"],
                "loss": r["loss_id"],
                "mz": float(channels.mz[j]),
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

    from constellation.massspec.annotation.mzpaf import fragment_row_to_mzpaf
    from constellation.massspec.peptide.ions import ion_number
    from constellation.massspec.peptide.match import match_mz
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

    channels, peptide = _one_peptide_fragments(
        modified_sequence, ion_types=ion_types,
        max_fragment_charge=max_fragment_charge or default_fragment_charge(charge),
        neutral_losses=neutral_losses,
    )  # fmt: skip
    length = len(peptide.sequence)
    names = [None, *channels.loss_ids]
    matches = (
        match_mz(torch.from_numpy(obs_mz.copy()), torch.from_numpy(channels.mz.copy()), tolerance=tolerance,
                 tolerance_unit=tolerance_unit, query_intensity=torch.from_numpy(obs_int.copy()))  # fmt: skip
        if channels.n
        else []
    )
    best: dict[int, object] = {}
    for m in matches:  # nearest ion for each peak
        held = best.get(m.query_idx)
        if held is None or abs(m.error_da) < abs(held.error_da):  # type: ignore[attr-defined]
            best[m.query_idx] = m
    for i, m in best.items():
        j = m.ref_idx
        row = {"ion_type": int(channels.ion_type[j]), "position": int(channels.position[j]),
               "charge": int(channels.fragment_charge[j]), "loss_id": names[int(channels.loss[j])]}  # fmt: skip
        columns["annotation"][i] = fragment_row_to_mzpaf(row, length=length)
        columns["ion_type"][i] = _LETTERS[row["ion_type"]]
        columns["ion_number"][i] = ion_number(row["ion_type"], row["position"], length)
        columns["fragment_charge"][i] = row["charge"]
        columns["loss"][i] = row["loss_id"]
        columns["mz_theoretical"][i] = float(channels.mz[j])
        columns["error_da"][i] = m.error_da
        columns["error_ppm"][i] = m.error_ppm

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


PRECURSOR_ISOTOPE_SCHEMA = pa.schema(
    [
        pa.field("isotope", pa.int8(), nullable=False),
        pa.field("mz", pa.float64(), nullable=False),
        #: Relative abundance, the most abundant isotope 1.
        pa.field("abundance", pa.float64(), nullable=False),
        #: "M", "M+1", ...
        pa.field("label", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"PrecursorIsotopes"},
)


def precursor_isotopes(modified_sequence: str, *, charge: int, n_isotopes: int = 3) -> pa.Table:
    """The precursor's first ``n_isotopes`` isotope peaks at ``charge``
    (``PRECURSOR_ISOTOPE_SCHEMA``), from its elemental composition."""
    from constellation.core.sequence.proforma import parse_proforma
    from constellation.massspec.peptide.channels import ChannelSettings, isotope_channels

    if charge is None or int(charge) <= 0:
        raise ValueError(f"charge must be positive; got {charge}")
    channels = isotope_channels([parse_proforma(modified_sequence)], np.array([0]), np.array([int(charge)]),
                                np.array([np.nan]), ChannelSettings(n_isotopes=n_isotopes))  # fmt: skip
    mz_np = channels.mz.astype(np.float64)
    # Shares of the whole envelope from the generator, shown with the most
    # abundant isotope as 1.
    ab = channels.abundance.astype(np.float64)
    ab = ab / ab.max() if ab.size and ab.max() > 0 else ab
    return pa.table(
        {
            "isotope": pa.array(range(mz_np.size), pa.int8()),
            "mz": mz_np,
            "abundance": ab,
            "label": ["M" if i == 0 else f"M+{i}" for i in range(mz_np.size)],
        },
        schema=PRECURSOR_ISOTOPE_SCHEMA,
    )


def theoretical_fragment_targets(
    modified_sequence: str,
    *,
    top_n: int = 8,
    mz_range: tuple[float, float] = (150.0, 2000.0),
) -> pa.Table:
    """Fragment ions worth tracing when no spectrum says which are intense
    (``ION_LADDER_SCHEMA`` rows, singly charged, no losses): y ions from the
    longest down, then b ions, both from ion 3 up (shorter ones are often
    below the scan range or shared by many peptides), within ``mz_range``,
    at most ``top_n``."""
    ladder = ion_ladder(modified_sequence)
    rows = [
        r
        for r in ladder.to_pylist()
        if r["ion_number"] >= 3 and mz_range[0] <= r["mz"] <= mz_range[1] and r["loss"] is None
    ]
    rows.sort(key=lambda r: (r["ion_type"] != "y", -r["ion_number"]))
    return pa.Table.from_pylist(rows[:top_n], schema=ION_LADDER_SCHEMA)


__all__ = [
    "ANNOTATED_PEAK_SCHEMA",
    "ION_LADDER_SCHEMA",
    "PRECURSOR_ISOTOPE_SCHEMA",
    "annotate_spectrum",
    "default_fragment_charge",
    "ion_ladder",
    "precursor_isotopes",
    "theoretical_fragment_targets",
]
