"""Concrete ``LibraryDesign`` instances + design registry.

The lab's in-house TSO-based PCR-cDNA design is the only one shipped
today; the registry pattern lets future designs (ONT-stock kits,
direct-RNA, lab variants) drop in as additional JSON files without
code changes here.

Designs are loaded at import time from JSON files under
``constellation/data/sequencing/`` — the build script
``scripts/build-sequencing-primers-json.py`` regenerates them.

Public surface:

    load_design(name)           — fetch a design by canonical name
    available_designs()         — list registered design names

The registry is populated lazily on first access — JSON I/O is cheap
but importing for its side effects shouldn't pay it unless something
asks for a design.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Final

from constellation.sequencing.transcriptome.adapters import (
    Adapter,
    AdapterSlot,
    Barcode,
    BarcodeSlot,
    LibraryDesign,
    PolyASlot,
    TranscriptSlot,
)


# ──────────────────────────────────────────────────────────────────────
# Defaults matching NanoporeAnalysis's hard-threshold parity settings
# ──────────────────────────────────────────────────────────────────────

# These match `NanoporeAnalysis/align.py` constants byte-for-byte —
# changing them breaks parity validation.
_POLYA_MIN_LEN: Final = 15
_POLYA_MAX_LEN: Final = 40
_POLYA_EDGE_DISTANCE: Final = 1
_ADAPTER_MAX_DISTANCE: Final = 2
_BARCODE_MAX_DISTANCE: Final = 2
_TRANSCRIPT_MIN_LEN: Final = 200


# Normalized acceptance score for BarcodeSlot in case probabilistic
# mode wants to consult it; HardThresholdScorer ignores this and uses
# the raw edit-distance threshold above.
def _normalized_min_score(max_distance: int, barcode_length: int) -> float:
    return 1.0 - (max_distance / barcode_length)


# ──────────────────────────────────────────────────────────────────────
# JSON → LibraryDesign
# ──────────────────────────────────────────────────────────────────────


def _load_design_json(name: str) -> dict:
    """Read a design JSON from packaged data."""
    pkg = resources.files("constellation.data.sequencing")
    path = pkg / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"design {name!r} has no JSON at {path}")
    return json.loads(path.read_text())


def _design_from_dict(design: dict) -> LibraryDesign:
    """Materialize a design dict into a ``LibraryDesign``.

    The 5' → 3' read layout for a TSO-based PCR-cDNA design:

        [5' SSP] [transcript] [polyA] [barcode] [3' primer]

    ``barcode`` here is what NanoporeAnalysis's ``score_umi`` aligns
    against — it sits between the polyA tail and the 3' primer match.
    Reads in the on-disk orientation carry the reverse complement of
    the barcode sequence in this slot; scorers do the RC at scoring
    time so the BarcodeSlot stores the canonical sequence.

    Boundary note. PolyA detection greedily absorbs the leading A of
    any barcode whose RC begins with A (BC05, BC06, BC11, BC12 in this
    design), so those reads' ``putative_umi`` regions are 1 nt shorter
    and match the design at ed=1. This is a real boundary ambiguity —
    the leading A is information-theoretically confounded with the
    polyA tail under hard-threshold scoring, but is *recoverable* under
    a probabilistic scorer that consults a polyA-length distribution
    (a 21-nt polyA + 16-nt ed=0 match has different joint likelihood
    than a 20-nt polyA + 15-nt ed=1 match). Resolving it lives in the
    scorer (see ``transcriptome.scoring``), not in design preprocessing
    — boundary uncertainty is a scoring concept, not a data property.
    """
    name = design["name"]

    ssp_adapter = Adapter(name=f"{name}_ssp", sequence=design["ssp"], kind="5p")
    primer3_adapter = Adapter(
        name=f"{name}_primer3", sequence=design["primer_3"], kind="3p"
    )

    barcode_length = design["barcode_length"]

    barcodes: tuple[Barcode, ...] = tuple(
        Barcode(name=b["name"], sequence=b["sequence"], kit=name)
        for b in design["barcodes"]
    )

    layout = (
        AdapterSlot(adapters=(ssp_adapter,), max_distance=_ADAPTER_MAX_DISTANCE),
        TranscriptSlot(min_length=_TRANSCRIPT_MIN_LEN),
        PolyASlot(
            min_length=_POLYA_MIN_LEN,
            max_length=_POLYA_MAX_LEN,
            edge_distance=_POLYA_EDGE_DISTANCE,
        ),
        BarcodeSlot(
            barcodes=barcodes,
            min_score=_normalized_min_score(_BARCODE_MAX_DISTANCE, barcode_length),
        ),
        AdapterSlot(adapters=(primer3_adapter,), max_distance=_ADAPTER_MAX_DISTANCE),
    )

    return LibraryDesign(
        name=name,
        layout=layout,
        allow_reverse_complement=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


# Names of designs we ship JSON for. Adding a new design: drop a JSON
# file under constellation/data/sequencing/ and append its name here.
_KNOWN_DESIGNS: tuple[str, ...] = ("cdna_wilburn_v1",)


_DESIGN_CACHE: dict[str, LibraryDesign] = {}


def load_design(name: str) -> LibraryDesign:
    """Fetch a design's ``LibraryDesign`` by canonical name.

    Cached on first load; subsequent calls return the same instance.
    """
    if name not in _DESIGN_CACHE:
        if name not in _KNOWN_DESIGNS:
            raise KeyError(
                f"design {name!r} is not registered. Known designs: "
                f"{', '.join(_KNOWN_DESIGNS) or '(none)'}"
            )
        design = _load_design_json(name)
        _DESIGN_CACHE[name] = _design_from_dict(design)
    return _DESIGN_CACHE[name]


def available_designs() -> tuple[str, ...]:
    """Names of all registered designs."""
    return _KNOWN_DESIGNS


# ──────────────────────────────────────────────────────────────────────
# Module-level shortcut for the in-house design
# ──────────────────────────────────────────────────────────────────────


# Eagerly construct so callers can `from .designs import CDNA_WILBURN_V1`
# rather than going through `load_design("cdna_wilburn_v1")`. The JSON
# load is one filesystem read at import time; cheap.
CDNA_WILBURN_V1: LibraryDesign = load_design("cdna_wilburn_v1")


__all__ = [
    "CDNA_WILBURN_V1",
    "available_designs",
    "load_design",
]
