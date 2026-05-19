"""NIST MSP ``Mods=`` field parsing + ProForma 2.0 builder.

NIST proteomics MSPs encode modifications in the Comment line as::

    Mods=N/pos,aa,name/pos,aa,name/...

where ``N`` is the modification count, ``pos`` is the **0-indexed**
residue position, ``aa`` is the affected residue letter, and ``name``
is the canonical UNIMOD short name (e.g. ``Oxidation``,
``Carbamidomethyl``, ``Phospho``, ``Acetyl``). ``Mods=0`` means
unmodified.

This module:

  1. Parses the ``Mods=`` field into a list of (pos, aa, name) tuples.
  2. Builds a ProForma 2.0 string from the parsed mods + canonical
     sequence + charge, then routes through ``parse_proforma`` so the
     resulting ``Peptidoform`` flows through the project's single
     canonical sequence-construction path.

Per the project rule (constellation/core/CLAUDE.md):
``UNIMOD`` is the universe of known modifications; the bundled
``ModVocab`` already populates short-name aliases ("Ox",
"Carbamidomethyl", ...) so ``UNIMOD.get("Oxidation")`` resolves the
canonical accession directly — no parallel name index needed.

Unresolved mod names are reported via ``MspModResolutionError`` so the
driver can decide whether to skip the entry or fall back on
``Modstring=`` mass deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.core.sequence.proforma import ModRef, Peptidoform, TaggedMod


@dataclass(frozen=True, slots=True)
class MspMod:
    """One parsed entry from a NIST ``Mods=`` field."""

    position: int  # 0-indexed residue offset
    residue: str  # affected residue letter
    name: str  # canonical UNIMOD short name as written in the file


class MspModResolutionError(KeyError):
    """Raised when a NIST ``Mods=`` name has no match in the supplied vocab."""


def parse_mods_field(value: str) -> list[MspMod]:
    """Parse a NIST ``Mods=`` field value into ``MspMod`` entries.

    ``Mods=0`` → ``[]`` (unmodified). Otherwise the value is
    ``N/pos,aa,name/pos,aa,name/...`` with the leading count ``N``
    redundant against the trailing list — we honour the trailing list.
    """
    value = value.strip()
    if not value or value == "0":
        return []

    parts = value.split("/")
    # First element is the count; subsequent elements are pos,aa,name triples.
    out: list[MspMod] = []
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        bits = part.split(",")
        if len(bits) != 3:
            raise ValueError(
                f"malformed Mods entry {part!r} — expected pos,aa,name"
            )
        pos_str, aa, name = bits
        try:
            pos = int(pos_str)
        except ValueError as e:
            raise ValueError(
                f"malformed Mods position {pos_str!r} in {part!r}"
            ) from e
        if len(aa) != 1:
            raise ValueError(
                f"malformed Mods residue {aa!r} in {part!r} — expected one letter"
            )
        out.append(MspMod(position=pos, residue=aa, name=name.strip()))
    return out


def build_proforma_string(
    sequence: str,
    charge: int | None,
    mods: list[MspMod],
    *,
    vocab: ModVocab = UNIMOD,
) -> str:
    """Compose a ProForma 2.0 string from canonical sequence + parsed mods.

    Mod names are resolved via ``vocab.get(name)`` which accepts the
    canonical id (``UNIMOD:35``), the full UNIMOD name, or any
    registered alias (``Oxidation``, ``Ox``, ...). The resulting
    ProForma string uses the canonical accession form
    ``M[UNIMOD:35]`` so the modseq is interoperable with any
    PSI-compliant consumer.

    Validates that ``sequence[pos] == residue`` for each mod — catches
    misalignment between the NIST 0-indexed position and the canonical
    sequence early.
    """
    if not sequence:
        raise ValueError("sequence must be non-empty")

    # Map each position to its mod accession(s). Multiple mods on the
    # same residue stack as nested brackets per ProForma.
    by_pos: dict[int, list[str]] = {}
    unresolved: list[str] = []
    for m in mods:
        if not (0 <= m.position < len(sequence)):
            raise ValueError(
                f"Mods position {m.position} out of range "
                f"for sequence length {len(sequence)}"
            )
        if sequence[m.position] != m.residue:
            raise ValueError(
                f"Mods residue mismatch at position {m.position}: "
                f"sequence has {sequence[m.position]!r}, Mods says {m.residue!r}"
            )
        try:
            modification = vocab.get(m.name)
        except KeyError:
            unresolved.append(m.name)
            continue
        by_pos.setdefault(m.position, []).append(modification.id)

    if unresolved:
        raise MspModResolutionError(
            f"unresolved Mods names against vocab: {unresolved!r}"
        )

    pieces: list[str] = []
    for i, residue in enumerate(sequence):
        pieces.append(residue)
        for accession in by_pos.get(i, ()):
            pieces.append(f"[{accession}]")
    proforma = "".join(pieces)
    if charge is not None:
        proforma = f"{proforma}/{charge}"
    return proforma


def build_peptidoform(
    sequence: str,
    charge: int | None,
    mods: list[MspMod],
    *,
    vocab: ModVocab = UNIMOD,
) -> Peptidoform:
    """Construct a ``Peptidoform`` directly from NIST-parsed mods.

    NIST .msp library entries are always linear peptides with residue
    modifications + optional charge — no terminal mods, labile mods,
    ranges, cross-links, or global isotopes. We can build the
    ``Peptidoform`` dataclass directly without round-tripping through
    ``parse_proforma`` — that saves ~25 ms/entry of Lark/Earley parse
    time, which dominates the read cost on large MSPs.

    Mod names are resolved via ``vocab.get(name)`` which accepts the
    canonical id (``UNIMOD:35``), the full UNIMOD name, or any
    registered alias (``Oxidation``, ``Ox``, ...). The resulting
    ``Peptidoform`` is byte-identical to one obtained via
    ``parse_proforma(build_proforma_string(...))`` — verified by
    round-trip tests.

    The bond-by-bond residue check from ``build_proforma_string`` is
    duplicated here so direct construction has the same validation
    floor as the string path.
    """
    if not sequence:
        raise ValueError("sequence must be non-empty")

    by_pos: dict[int, list[TaggedMod]] = {}
    unresolved: list[str] = []
    for m in mods:
        if not (0 <= m.position < len(sequence)):
            raise ValueError(
                f"Mods position {m.position} out of range "
                f"for sequence length {len(sequence)}"
            )
        if sequence[m.position] != m.residue:
            raise ValueError(
                f"Mods residue mismatch at position {m.position}: "
                f"sequence has {sequence[m.position]!r}, Mods says {m.residue!r}"
            )
        try:
            modification = vocab.get(m.name)
        except KeyError:
            unresolved.append(m.name)
            continue
        # Modification.id is "UNIMOD:35"; ProForma ModRef stores cv +
        # accession as the bare numeric part ("35"), so round-trip
        # via format_proforma renders correctly.
        cv, _, accession = modification.id.partition(":")
        by_pos.setdefault(m.position, []).append(
            TaggedMod(mod=ModRef(cv=cv or None, accession=accession or None))
        )

    if unresolved:
        raise MspModResolutionError(
            f"unresolved Mods names against vocab: {unresolved!r}"
        )

    residue_mods = {
        pos: tuple(taglist) for pos, taglist in by_pos.items()
    }
    return Peptidoform(
        sequence=sequence,
        residue_mods=residue_mods,
        charge=charge,
    )


__all__ = [
    "MspMod",
    "MspModResolutionError",
    "parse_mods_field",
    "build_proforma_string",
    "build_peptidoform",
]
