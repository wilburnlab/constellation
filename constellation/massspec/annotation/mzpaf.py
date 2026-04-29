"""mzPAF v1.0 — HUPO-PSI peak-annotation format (ratified Aug 2025).

mzPAF is the standard text grammar for annotating which fragment ion a
mass-spectrum peak corresponds to. Constellation's relational fragment
schemas (LIBRARY_FRAGMENT_TABLE, FRAGMENT_ION_TABLE) carry
``ion_type`` / ``position`` / ``charge`` / ``loss_id`` as canonical
columns; mzPAF strings are emitted at write time via the projection
helpers in this module (`fragment_row_to_mzpaf`,
`fragment_table_to_mzpaf`).

This independent implementation mirrors the rule that bans pyteomics
from `core.sequence.proforma`: the parser is built from the spec,
not by wrapping the upstream `mzpaf` PyPI package, so the two
implementations cross-check each other in
`tests/test_mzpaf_pypi_parity.py`.

Scope (Phase 1):

  - **Peptide ion series** (a/b/c/x/y/z) with position — full support.
  - **Decorations** — neutral loss (formula or named bracket), isotope,
    adduct, charge, mass error (Da or ppm), confidence — full support.
  - **Alternates** — comma-separated alternative interpretations of the
    same peak — full support.
  - **Multi-analyte prefix** (``N@``) — parsed and round-tripped, but
    chimeric-spectrum consumption is downstream work.
  - **Unknown** (``?``) — full support.

Deferred (raise NotImplementedError on parse, with a clear message):

  - Immonium ions (``IY``, ``IY[Phospho]``)
  - Internal fragments (``m3:6``)
  - Precursor (``p``)
  - Reporter ions (``r[TMT127N]``)
  - Named ions (``_{Urocanic Acid}``)
  - Formula tokens (``f{C16H22O}``)
  - SMILES tokens (``s{...}``)

These are valid mzPAF v1.0 syntax but not on the EncyclopeDIA-port
critical path — they land alongside the mzSpecLib reader (Phase 3) and
the chimeric-spectrum quant work that needs them.

Public API:

    Annotation                One single-analyte interpretation
    PeakAnnotation            Full mzPAF parse for one peak
    parse_mzpaf(s)            -> PeakAnnotation
    format_mzpaf(p)           -> str
    MzPAFError / MzPAFSyntaxError
    fragment_row_to_mzpaf(row)            -> str
    fragment_table_to_mzpaf(table)        -> pa.Array (vectorized)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping

import pyarrow as pa

from constellation.massspec.annotation._grammar import (
    RE_ADDUCT,
    RE_ANALYTE_PREFIX,
    RE_CHARGE,
    RE_CONFIDENCE,
    RE_FORMULA,
    RE_IMMONIUM,
    RE_INTERNAL,
    RE_ISOTOPE,
    RE_LOSS_BRACKET,
    RE_LOSS_FORMULA,
    RE_MASS_ERROR,
    RE_NAMED,
    RE_PEPTIDE_ION,
    RE_PRECURSOR,
    RE_REPORTER,
    RE_SMILES,
    RE_UNKNOWN,
    IonClass,
    MassErrorUnit,
)


# ──────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────


class MzPAFError(Exception):
    """Base for all mzPAF errors."""


class MzPAFSyntaxError(MzPAFError):
    """Raised when input does not conform to the mzPAF grammar."""


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NeutralLoss:
    """One neutral-loss / gain term: ``+CO`` or ``-H2O`` or ``-[Phospho]``.

    ``sign`` is +1 for addition (rare; e.g. ``+CO`` for x→y) or -1 for
    loss. ``token`` is the chemical formula (``H2O``, ``CO``,
    ``CH4O``) when ``is_named`` is False, or the bracket-payload name
    (``Phospho``, ``Hex``) when ``is_named`` is True.
    """

    sign: int  # +1 or -1
    token: str
    is_named: bool = False  # True for [Name], False for chemical formula


@dataclass(frozen=True, slots=True)
class MassError:
    """Observed-vs-theoretical mass error attached to an annotation."""

    value: float
    unit: MassErrorUnit


@dataclass(frozen=True, slots=True)
class Annotation:
    """One single-analyte mzPAF interpretation.

    ``ion_class`` discriminates the prefix family. For PEPTIDE,
    ``ion_letter`` is one of ``a/b/c/x/y/z`` and ``position`` is the
    fragment index (1-based, per spec). Other ion classes leave those
    fields None.

    Decorations apply uniformly:

      losses          : tuple of NeutralLoss in source-order
      isotope         : signed integer offset (``+2i`` → 2, ``-1i`` → -1)
                        or 0 when absent
      adduct          : raw bracket body (``M+H``, ``M+H+Na``) or None
      charge          : integer charge (positive in CID; sign preserved
                        for negative-mode) or None when absent
      mass_error      : MassError or None
      confidence      : non-negative float or None
      analyte_idx     : multi-analyte index (``N@`` prefix) — 1-based
                        per spec; None when annotation refers to the
                        sole analyte.
    """

    ion_class: IonClass
    ion_letter: str | None = None
    position: int | None = None
    losses: tuple[NeutralLoss, ...] = ()
    isotope: int = 0
    adduct: str | None = None
    charge: int | None = None
    mass_error: MassError | None = None
    confidence: float | None = None
    analyte_idx: int | None = None


@dataclass(frozen=True, slots=True)
class PeakAnnotation:
    """Full mzPAF interpretation for one peak.

    A peak may carry multiple alternate interpretations (comma-
    separated in the source string). For chimeric spectra the
    alternates may belong to different analytes (``analyte_idx``
    attribute on each Annotation) — same flat list, discriminated by
    the per-Annotation prefix.
    """

    annotations: tuple[Annotation, ...] = ()


# ──────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────


def parse_mzpaf(s: str) -> PeakAnnotation:
    """Parse an mzPAF v1.0 annotation string.

    Raises:
        MzPAFSyntaxError: input does not conform to the grammar.
        NotImplementedError: input uses a v1.0 token (immonium /
            internal / precursor / reporter / named / formula /
            SMILES) deferred to a later phase.
    """
    if not s:
        raise MzPAFSyntaxError("empty input")
    # Top-level: comma separates alternates / multi-analyte interpretations.
    # Commas only appear at this top level — bracket payloads are unicode-
    # safe but never contain commas in v1.0 syntax.
    parts = _split_top_level(s, ",")
    annotations = tuple(_parse_one(p.strip()) for p in parts)
    if not annotations:
        raise MzPAFSyntaxError(f"no annotations parsed from {s!r}")
    return PeakAnnotation(annotations=annotations)


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split on `sep` at depth 0 (outside of brackets / braces)."""
    depth = 0
    out: list[str] = []
    start = 0
    for i, c in enumerate(s):
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth < 0:
                raise MzPAFSyntaxError(
                    f"unbalanced bracket at index {i} in {s!r}"
                )
        elif c == sep and depth == 0:
            out.append(s[start:i])
            start = i + 1
    if depth != 0:
        raise MzPAFSyntaxError(f"unbalanced brackets in {s!r}")
    out.append(s[start:])
    return out


def _parse_one(s: str) -> Annotation:
    """Parse a single annotation (one analyte, no top-level commas)."""
    if not s:
        raise MzPAFSyntaxError("empty annotation")

    rest = s
    analyte_idx: int | None = None
    m = RE_ANALYTE_PREFIX.match(rest)
    if m is not None:
        analyte_idx = int(m.group(1))
        rest = rest[m.end():]

    # Ion-class dispatch (mutually exclusive prefixes).
    ion_class, ion_letter, position, rest = _parse_ion_prefix(rest)

    losses: list[NeutralLoss] = []
    isotope = 0
    adduct: str | None = None
    charge: int | None = None
    mass_error: MassError | None = None
    confidence: float | None = None

    # Decorations — fixed order per spec, but we accept reorderings for
    # tolerance to author drift. Loop until rest is empty or no progress.
    while rest:
        # 1. neutral-loss tokens accumulate at the start of the suffix.
        m = RE_LOSS_BRACKET.match(rest)
        if m is not None:
            sign = 1 if m.group(1) == "+" else -1
            losses.append(
                NeutralLoss(sign=sign, token=m.group(2), is_named=True)
            )
            rest = rest[m.end():]
            continue
        m = RE_LOSS_FORMULA.match(rest)
        if m is not None:
            sign = 1 if m.group(1) == "+" else -1
            losses.append(
                NeutralLoss(sign=sign, token=m.group(2), is_named=False)
            )
            rest = rest[m.end():]
            continue
        # 2. isotope — anchored "+Ni" / "-Ni"
        m = RE_ISOTOPE.match(rest)
        if m is not None and isotope == 0:
            sign = 1 if m.group(1) == "+" else -1
            isotope = sign * int(m.group(2))
            rest = rest[m.end():]
            continue
        # 3. adduct
        if rest.startswith("[M"):
            m = RE_ADDUCT.match(rest)
            if m is None:
                raise MzPAFSyntaxError(f"malformed adduct in {s!r}")
            if adduct is not None:
                raise MzPAFSyntaxError(
                    f"multiple adducts in {s!r}; max 1 per annotation"
                )
            adduct = m.group(1)
            rest = rest[m.end():]
            continue
        # 4. charge
        if rest.startswith("^"):
            m = RE_CHARGE.match(rest)
            if m is None:
                raise MzPAFSyntaxError(f"malformed charge in {s!r}")
            if charge is not None:
                raise MzPAFSyntaxError(f"multiple charges in {s!r}")
            charge = int(m.group(1))
            rest = rest[m.end():]
            continue
        # 5. mass error
        if rest.startswith("/"):
            m = RE_MASS_ERROR.match(rest)
            if m is None:
                raise MzPAFSyntaxError(f"malformed mass error in {s!r}")
            if mass_error is not None:
                raise MzPAFSyntaxError(f"multiple mass errors in {s!r}")
            unit = (
                MassErrorUnit.PPM if m.group(2) == "ppm" else MassErrorUnit.DA
            )
            mass_error = MassError(value=float(m.group(1)), unit=unit)
            rest = rest[m.end():]
            continue
        # 6. confidence
        if rest.startswith("*"):
            m = RE_CONFIDENCE.match(rest)
            if m is None:
                raise MzPAFSyntaxError(f"malformed confidence in {s!r}")
            if confidence is not None:
                raise MzPAFSyntaxError(f"multiple confidences in {s!r}")
            confidence = float(m.group(1))
            rest = rest[m.end():]
            continue

        raise MzPAFSyntaxError(
            f"unrecognized decoration starting at {rest!r} (in {s!r})"
        )

    return Annotation(
        ion_class=ion_class,
        ion_letter=ion_letter,
        position=position,
        losses=tuple(losses),
        isotope=isotope,
        adduct=adduct,
        charge=charge,
        mass_error=mass_error,
        confidence=confidence,
        analyte_idx=analyte_idx,
    )


def _parse_ion_prefix(
    s: str,
) -> tuple[IonClass, str | None, int | None, str]:
    """Consume the ion-prefix token. Returns (class, letter, position, rest)."""
    m = RE_PEPTIDE_ION.match(s)
    if m is not None:
        return IonClass.PEPTIDE, m.group(1), int(m.group(2)), s[m.end():]
    m = RE_UNKNOWN.match(s)
    if m is not None:
        return IonClass.UNKNOWN, None, None, s[m.end():]
    # Deferred ion classes — recognize and reject with a useful message
    # rather than letting the suffix parser fail confusingly.
    if RE_IMMONIUM.match(s):
        raise NotImplementedError(
            f"immonium ion parsing deferred (Phase 3 / mzSpecLib reader): {s!r}"
        )
    if RE_INTERNAL.match(s):
        raise NotImplementedError(
            f"internal fragment parsing deferred: {s!r}"
        )
    if RE_REPORTER.match(s):
        raise NotImplementedError(
            f"reporter ion parsing deferred: {s!r}"
        )
    if RE_NAMED.match(s):
        raise NotImplementedError(
            f"named ion parsing deferred: {s!r}"
        )
    if RE_FORMULA.match(s):
        raise NotImplementedError(
            f"formula ion parsing deferred: {s!r}"
        )
    if RE_SMILES.match(s):
        raise NotImplementedError(
            f"SMILES ion parsing deferred: {s!r}"
        )
    if RE_PRECURSOR.match(s):
        raise NotImplementedError(
            f"precursor ion parsing deferred: {s!r}"
        )
    raise MzPAFSyntaxError(f"unrecognized ion prefix in {s!r}")


# ──────────────────────────────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────────────────────────────


def format_mzpaf(p: PeakAnnotation) -> str:
    """Render a PeakAnnotation back to mzPAF text. Lossless for parsed input."""
    return ",".join(_format_one(a) for a in p.annotations)


def _format_one(a: Annotation) -> str:
    parts: list[str] = []
    if a.analyte_idx is not None:
        parts.append(f"{a.analyte_idx}@")

    if a.ion_class is IonClass.PEPTIDE:
        if a.ion_letter is None or a.position is None:
            raise MzPAFError(
                "PEPTIDE annotation missing ion_letter or position"
            )
        parts.append(f"{a.ion_letter}{a.position}")
    elif a.ion_class is IonClass.UNKNOWN:
        parts.append("?")
    else:
        raise NotImplementedError(
            f"format for ion_class {a.ion_class.value} deferred"
        )

    for loss in a.losses:
        sign = "+" if loss.sign > 0 else "-"
        if loss.is_named:
            parts.append(f"{sign}[{loss.token}]")
        else:
            parts.append(f"{sign}{loss.token}")
    if a.isotope != 0:
        sign = "+" if a.isotope > 0 else "-"
        parts.append(f"{sign}{abs(a.isotope)}i")
    if a.adduct is not None:
        parts.append(f"[{a.adduct}]")
    if a.charge is not None:
        parts.append(f"^{a.charge}")
    if a.mass_error is not None:
        unit = "ppm" if a.mass_error.unit is MassErrorUnit.PPM else ""
        parts.append(f"/{a.mass_error.value}{unit}")
    if a.confidence is not None:
        parts.append(f"*{a.confidence}")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Projection: Constellation fragment-row -> mzPAF
# ──────────────────────────────────────────────────────────────────────

# Mirrors massspec.peptide.ions.IonType. Duplicated as a literal map so
# this module stays a leaf of `peptide.ions` (the inverse import would
# create a cycle since `ions.py` may want to call back into mzPAF
# rendering once Phase 3 ships per-fragment annotations).
_ION_LETTERS: tuple[str, ...] = ("a", "b", "c", "x", "y", "z")


def fragment_row_to_mzpaf(row: Mapping) -> str:
    """Render one fragment-row (dict-like) to its canonical mzPAF string.

    Required keys (per LIBRARY_FRAGMENT_TABLE / FRAGMENT_ION_TABLE):
        ion_type    int (0..5 → a/b/c/x/y/z)
        position    int   (1-based)
        charge      int

    Optional keys:
        loss_id     str | None — when present, rendered as a named
                    bracket loss (``-[H2O]``, ``-[Phospho]``).
                    Absent / None / empty → no loss decoration.
    """
    ion_type = int(row["ion_type"])
    if not 0 <= ion_type < len(_ION_LETTERS):
        raise MzPAFError(f"invalid ion_type {ion_type}; expected 0..5")
    letter = _ION_LETTERS[ion_type]
    position = int(row["position"])
    charge = int(row["charge"])
    loss = row.get("loss_id")

    parts: list[str] = [f"{letter}{position}"]
    if loss:
        # Constellation's LOSS_REGISTRY ids ("H2O", "NH3", "HPO3", "H3PO4")
        # are bare chemical formulae — render in the bracket-named form
        # for round-trip stability with mzSpecLib readers that recover
        # the loss as a named token.
        parts.append(f"-[{loss}]")
    if charge != 1:
        parts.append(f"^{charge}")
    return "".join(parts)


def fragment_table_to_mzpaf(table: pa.Table) -> pa.Array:
    """Vectorized projection of a fragment table to an mzPAF string column.

    Returns a pa.Array (string) of the same length as ``table``. The
    table must carry ``ion_type``, ``position``, ``charge`` columns;
    ``loss_id`` is read when present.
    """
    ion_types = table.column("ion_type").to_pylist()
    positions = table.column("position").to_pylist()
    charges = table.column("charge").to_pylist()
    if "loss_id" in table.column_names:
        losses: Iterable = table.column("loss_id").to_pylist()
    else:
        losses = (None,) * len(ion_types)

    out: list[str] = []
    for it, pos, ch, loss in zip(ion_types, positions, charges, losses):
        out.append(
            fragment_row_to_mzpaf(
                {"ion_type": it, "position": pos, "charge": ch, "loss_id": loss}
            )
        )
    return pa.array(out, type=pa.string())


__all__ = [
    "Annotation",
    "MzPAFError",
    "MzPAFSyntaxError",
    "NeutralLoss",
    "MassError",
    "PeakAnnotation",
    "format_mzpaf",
    "fragment_row_to_mzpaf",
    "fragment_table_to_mzpaf",
    "parse_mzpaf",
]
