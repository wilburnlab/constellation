"""Koina fragment-annotation strings → fragment-ladder lookup keys.

Prosit-family intensity models return an ``annotation`` array of ASCII
**bytes** in the form ``b"y1+1"`` / ``b"b12+2"``: an ion letter, the
1-indexed Roepstorff position, then ``+`` and the fragment charge. All
174 tokens in the captured fixtures match that grammar exactly, with no
neutral-loss forms present — but the loss group is parsed anyway, since
newer PTM-aware models emit ``b"y3-H2O+1"``.

The position convention here is the one thing worth getting right:
Prosit's ordinal is 1-indexed Roepstorff, while Constellation keys
fragments by 0-indexed peptide-bond position from the N-terminus. The
conversion is identical to ``massspec.io.msp._annotate._resolve_annotation``
and is deliberately duplicated rather than imported, because that helper
is welded to the mzPAF parser — Koina's grammar is a fixed 3-field form
that a 40-character regex handles without the mzPAF round trip.
"""

from __future__ import annotations

import re

from constellation.massspec.peptide.ions import IonType
from constellation.massspec.peptide.neutral_losses import LOSS_REGISTRY

_ANNOTATION_RE = re.compile(
    r"^(?P<letter>[abcxyz])(?P<ordinal>\d+)"
    r"(?:-(?P<loss>[A-Za-z0-9]+))?"
    r"\+(?P<charge>\d+)$"
)

_LETTER_TO_ION_TYPE: dict[str, IonType] = {
    "a": IonType.A,
    "b": IonType.B,
    "c": IonType.C,
    "x": IonType.X,
    "y": IonType.Y,
    "z": IonType.Z,
}

#: y/x/z count from the C-terminus, so their ordinal inverts.
_C_SIDE_TYPES = frozenset({IonType.X, IonType.Y, IonType.Z})

_LOSS_IDS = frozenset(LOSS_REGISTRY.ids())


def decode_annotation(raw: bytes | str) -> str:
    """Koina returns ``bytes``; normalise to ``str`` for storage."""
    return raw.decode("ascii") if isinstance(raw, bytes) else str(raw)


def parse_koina_annotation(
    raw: bytes | str,
    seq_len: int,
) -> tuple[int, int, int, str | None] | None:
    """Parse one annotation into ``(ion_type, bond_position, charge, loss_id)``.

    Returns ``None`` when the token doesn't parse, names an unregistered
    neutral loss, or resolves to a bond position outside the peptide —
    all of which are kept as partial-ID rows carrying the raw string,
    per the ``LIBRARY_FRAGMENT_TABLE`` nullability contract, rather than
    being dropped.

    ``seq_len`` is the peptide's canonical residue count; a peptide of
    length L has L-1 fragmentable bonds, indexed 0..L-2.
    """
    match = _ANNOTATION_RE.match(decode_annotation(raw))
    if match is None:
        return None

    ion_type = _LETTER_TO_ION_TYPE[match["letter"]]

    loss = match["loss"]
    if loss is not None and loss not in _LOSS_IDS:
        return None

    ordinal = int(match["ordinal"])
    if ion_type in _C_SIDE_TYPES:
        bond_pos = seq_len - 1 - ordinal
    else:
        bond_pos = ordinal - 1

    if bond_pos < 0 or bond_pos > seq_len - 2:
        return None

    return int(ion_type), bond_pos, int(match["charge"]), loss


__all__ = ["decode_annotation", "parse_koina_annotation"]
