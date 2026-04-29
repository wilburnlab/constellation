"""Universal Spectrum Identifier (USI) — HUPO-PSI standard, ratified Jul 2021.

USI is a multi-part key for referencing a specific spectrum in a public
repository. The grammar (informal):

    mzspec:<collection>:<run_name>:<index_type>:<index>[:<interpretation>]

where:

    <collection>      ProteomeXchange / repo accession (e.g. "PXD000561",
                      "MSV000079514", "PXL000001"), or "USI000000" for
                      synthetic / private collections.
    <run_name>        MS run identifier — usually the source filename
                      stem (e.g. "Adult_Frontalcortex_bRP_Elite_85_f09").
                      Colons in run names are not permitted by the spec.
    <index_type>      One of: scan, index, nativeId, trace.
    <index>           The index value (typically an integer for "scan"
                      or "index"; an opaque string for "nativeId" /
                      "trace").
    <interpretation>  Optional. ProForma 2.0 modseq + "/" + charge,
                      e.g. "VLHPLEGAVVIIFK/2" or
                      "[iTRAQ4plex]-LHFFM[Oxidation]PGFAPLTSR/3".
                      The charge is encoded by ProForma's `/N` suffix —
                      we delegate parsing to `core.sequence.proforma`.

Reference resolvers: USICentral (ProteomeCentral), PRIDE, PeptideAtlas,
MassIVE, jPOST.

Spec: https://psidev.info/usi
Reference impl (Python): part of the `pyteomics` and `mzspeclib`
packages. We do not import either here — the parser is independent so
that grammar drift is detectable in the parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from constellation.core.sequence.proforma import (
    Peptidoform,
    ProFormaError,
    parse_proforma,
)

IndexType = Literal["scan", "index", "nativeId", "trace"]
_VALID_INDEX_TYPES: frozenset[str] = frozenset(
    {"scan", "index", "nativeId", "trace"}
)


# ──────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────


class USIError(Exception):
    """Base for all USI errors."""


class USISyntaxError(USIError):
    """Raised when input does not conform to the USI grammar."""


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class USI:
    """Parsed USI. Lossless round-trip via `parse_usi` / `format()`.

    The interpretation slot, when present, embeds a ProForma 2.0 string
    plus a `/charge` suffix — both are normalized into a single
    `Peptidoform` (whose `.charge` carries the precursor charge per
    ProForma's serialization convention).
    """

    collection: str
    run_name: str
    index_type: IndexType
    index: str
    interpretation: Peptidoform | None = None

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def parse(cls, s: str) -> "USI":
        """Parse a USI string. Equivalent to module-level ``parse_usi(s)``."""
        return parse_usi(s)

    @classmethod
    def from_spectrum(
        cls,
        *,
        collection: str,
        run_name: str,
        scan: int | None = None,
        index: int | None = None,
        native_id: str | None = None,
        trace: str | None = None,
        peptidoform: Peptidoform | None = None,
    ) -> "USI":
        """Build a USI by naming exactly one of (scan, index, native_id, trace).

        Per the USI spec, exactly one index discriminator must be supplied.
        """
        provided = [
            (k, v)
            for k, v in (
                ("scan", scan),
                ("index", index),
                ("nativeId", native_id),
                ("trace", trace),
            )
            if v is not None
        ]
        if len(provided) != 1:
            raise ValueError(
                "from_spectrum requires exactly one of "
                "(scan, index, native_id, trace); "
                f"got {[k for k, _ in provided]}"
            )
        index_type, value = provided[0]
        return cls(
            collection=collection,
            run_name=run_name,
            index_type=index_type,  # type: ignore[arg-type]
            index=str(value),
            interpretation=peptidoform,
        )

    # ── serialization ───────────────────────────────────────────────
    def format(self) -> str:
        """Render to canonical USI form. Lossless when input was canonical."""
        parts = [
            "mzspec",
            self.collection,
            self.run_name,
            self.index_type,
            self.index,
        ]
        head = ":".join(parts)
        if self.interpretation is None:
            return head
        # ProForma's `format_proforma` already includes the `/charge`
        # suffix when the peptidoform carries a charge. The USI
        # interpretation slot REQUIRES the charge — without it, the
        # interpretation is malformed per spec.
        from constellation.core.sequence.proforma import format_proforma

        if self.interpretation.charge is None:
            raise USISyntaxError(
                "USI interpretation requires a charge state "
                "(set Peptidoform.charge before formatting)"
            )
        return f"{head}:{format_proforma(self.interpretation)}"

    def __str__(self) -> str:  # convenience
        return self.format()


# ──────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────


def parse_usi(s: str) -> USI:
    """Parse a USI string into a :class:`USI`.

    Raises :class:`USISyntaxError` on malformed input.
    """
    if not s:
        raise USISyntaxError("empty input")
    # The interpretation slot may itself contain ":" (rarely — but
    # ProForma ranges like `(0..3)[+15]` don't introduce ":", and
    # CV-tagged mods like `[U:Phospho]` use ":" inside square brackets,
    # which split() over the whole string would mishandle). Cap the
    # split at 5 so the interpretation is recovered intact.
    parts = s.split(":", 5)
    if len(parts) < 5:
        raise USISyntaxError(
            f"USI requires at least 5 colon-separated fields, got "
            f"{len(parts)}: {s!r}"
        )
    if parts[0] != "mzspec":
        raise USISyntaxError(
            f"USI must start with 'mzspec:', got {parts[0]!r}"
        )
    collection, run_name, index_type, index = parts[1:5]
    interpretation_str = parts[5] if len(parts) == 6 else None

    if not collection:
        raise USISyntaxError("USI collection is empty")
    if not run_name:
        raise USISyntaxError("USI run_name is empty")
    if index_type not in _VALID_INDEX_TYPES:
        raise USISyntaxError(
            f"USI index_type must be one of {sorted(_VALID_INDEX_TYPES)}, "
            f"got {index_type!r}"
        )
    if not index:
        raise USISyntaxError("USI index is empty")

    interpretation: Peptidoform | None = None
    if interpretation_str:
        try:
            parsed = parse_proforma(interpretation_str)
        except ProFormaError as e:
            raise USISyntaxError(
                f"USI interpretation is not valid ProForma 2.0: "
                f"{interpretation_str!r}: {e}"
            ) from e
        # USI interpretation must be a single peptidoform (not a
        # multi-chain cross-linked complex — that's an mzSpecLib /
        # mzPAF concern). Reject multi-peptidoform inputs explicitly.
        if not isinstance(parsed, Peptidoform):
            raise USISyntaxError(
                "USI interpretation must be a single peptidoform; "
                "multi-chain cross-links are not supported in USI"
            )
        if parsed.charge is None:
            raise USISyntaxError(
                "USI interpretation must include a charge state "
                "(append `/N` per ProForma 2.0 §4.6)"
            )
        interpretation = parsed

    return USI(
        collection=collection,
        run_name=run_name,
        index_type=index_type,  # type: ignore[arg-type]
        index=index,
        interpretation=interpretation,
    )


__all__ = [
    "USI",
    "USIError",
    "USISyntaxError",
    "parse_usi",
]
