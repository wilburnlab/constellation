"""Adapter / Barcode / LibraryConstruct — composable read-structure model.

A nanopore cDNA read is a sequence of *segments*: 5' adapter, optional
UMI, transcript, polyA tail, 3' adapter, barcode. Different chemistries
arrange these segments in different orders with different sequences; a
flexible ``LibraryConstruct`` lets demux work uniformly across:

    - the lab's in-house SMARTer-derived chemistry
    - standard ONT cDNA kits (SQK-PCS111, SQK-PCB111)
    - direct-RNA chemistry (RCB114) — fewer segments, no PCR
    - future kits as ONT releases them

Segments are independent first-class objects so adapters and barcodes
shared between kits aren't duplicated. Demux is then a function over
``LibraryConstruct``; new kits add a construct rather than new code.

Status: STUB. Concrete sequence registries (the lab's in-house adapter
sequences, the ONT-published 96-barcode set, etc.) populate in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal


# ──────────────────────────────────────────────────────────────────────
# Atoms — Adapter, Barcode
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Adapter:
    """A short adapter / primer sequence introduced during library prep.

    ``kind`` distinguishes adapters by their role in the construct, not
    their chemistry. ``5p`` / ``3p`` adapters bracket the insert;
    ``polyA`` is the canonical 'AAAA' anchor sequence; ``umi`` is a
    unique-molecular-identifier scaffold.
    """

    name: str
    sequence: str
    kind: Literal["5p", "3p", "polyA", "umi"]


@dataclass(frozen=True, slots=True)
class Barcode:
    """One barcode within a kit's barcode panel.

    ``kit`` groups barcodes into the panel they belong to ('SQK-NBD114-24',
    'SQK-PCB111-96', 'lab-custom-v3'). Demux only considers barcodes
    in the panel attached to the LibraryConstruct's BarcodeSlot.
    """

    name: str
    sequence: str
    kit: str


# ──────────────────────────────────────────────────────────────────────
# Segment ABCs (compose into LibraryConstruct.layout)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Segment:
    """Base for a slot in a LibraryConstruct's read-layout sequence.

    Concrete subclasses below — kept as discriminated dataclasses
    rather than a Protocol so Arrow / pickling can round-trip the
    construct definitions.
    """

    kind: ClassVar[str] = "abstract"


@dataclass(frozen=True, slots=True)
class AdapterSlot(Segment):
    """Slot expecting one of the listed adapters (logical OR).

    Single-adapter slots use a one-element tuple; OR-of-adapters slots
    accept any adapter in the tuple at demux time.
    """

    adapters: tuple[Adapter, ...] = ()
    max_distance: int = 1
    kind: ClassVar[str] = "adapter"


@dataclass(frozen=True, slots=True)
class BarcodeSlot(Segment):
    """Slot containing a barcode from the supplied panel.

    ``min_score`` is the demux acceptance threshold (1 - normalized
    edit distance); below this the barcode is unassigned.
    """

    barcodes: tuple[Barcode, ...] = ()
    min_score: float = 0.7
    kind: ClassVar[str] = "barcode"


@dataclass(frozen=True, slots=True)
class PolyASlot(Segment):
    """PolyA / polyT tail slot — used as the demux anchor.

    ``min_length`` is the minimum number of consecutive A's (or T's
    on antisense) required to call a polyA region. ``edge_distance``
    permits this many edit operations at the boundaries (homopolymer
    error tolerance).
    """

    min_length: int = 20
    edge_distance: int = 1
    kind: ClassVar[str] = "polyA"


@dataclass(frozen=True, slots=True)
class UMISlot(Segment):
    """Fixed-length unique-molecular-identifier scaffold."""

    length: int = 12
    kind: ClassVar[str] = "umi"


@dataclass(frozen=True, slots=True)
class TranscriptSlot(Segment):
    """The variable-length insert — the actual cDNA / RNA we care about."""

    min_length: int = 100
    max_length: int | None = None
    kind: ClassVar[str] = "transcript"


# ──────────────────────────────────────────────────────────────────────
# Composable construct
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LibraryConstruct:
    """Composable description of a sequencing library's read structure.

    ``layout`` describes the segment order from 5' → 3' on the sense
    strand. ``allow_reverse_complement`` lets demux check both
    orientations (nanopore reads aren't strand-specific by default).

    Phase 5 ships built-in constructs for:

        SMARTER_CDNA_INHOUSE   the lab's adapter chemistry
        ONT_PCS111             SQK-PCS111 cDNA-PCR sequencing
        ONT_PCB111             SQK-PCB111 cDNA-PCR-barcoded
        ONT_RCB114             SQK-RCB114 direct-RNA + 96-barcode
    """

    name: str
    layout: tuple[Segment, ...] = field(default_factory=tuple)
    allow_reverse_complement: bool = True


__all__ = [
    "Adapter",
    "Barcode",
    "Segment",
    "AdapterSlot",
    "BarcodeSlot",
    "PolyASlot",
    "UMISlot",
    "TranscriptSlot",
    "LibraryConstruct",
]
