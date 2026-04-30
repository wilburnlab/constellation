"""Telomere / centromere / tandem-repeat detection.

Telomeric repeats (vertebrate ``TTAGGG``, plant ``TTTAGGG``,
nematode ``TTAGGC``, yeast ``TG(1-3)``) at contig termini are a
strong assembly-completeness signal — chromosome-level scaffolds
should have telomeric repeats at both ends. Centromeres are harder
(species-specific consensus repeats; satellite-DNA arrays).

Pure-functional helpers — no subprocess wrappers needed for telomere
scanning, which is just k-mer enumeration over the contig ends.

Status: STUB. Pending Phase 9.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.sequencing.reference.reference import Reference


_PHASE = "Phase 9 (annotation/telomeres)"


# Common telomeric repeats by clade (5' → 3' on the G-strand).
TELOMERE_MOTIFS: dict[str, str] = {
    "vertebrate": "TTAGGG",
    "plant": "TTTAGGG",
    "nematode": "TTAGGC",
    "drosophila": "TTAGG",   # variable; Drosophila uses retrotransposons too
    "ciliate": "TTGGGG",     # Tetrahymena
}


def find_telomeres(
    reference: Reference,
    *,
    motif: str = "TTAGGG",
    min_repeat_count: int = 4,
    edge_window: int = 10_000,
) -> pa.Table:
    """Locate telomeric-repeat clusters at contig ends.

    Returns ``FEATURE_TABLE``-shaped rows with ``type='telomere'``.
    ``motif`` is the canonical repeat (often picked from
    :data:`TELOMERE_MOTIFS`). ``min_repeat_count`` is the minimum
    number of consecutive motif occurrences to call a telomere.
    """
    raise NotImplementedError(f"find_telomeres pending {_PHASE}")


def find_tandem_repeats(
    reference: Reference,
    *,
    min_period: int = 3,
    max_period: int = 200,
    min_copies: int = 5,
) -> pa.Table:
    """Generic tandem-repeat detection (centromeric satellite DNA,
    microsatellites, minisatellites). Returns ``FEATURE_TABLE``-shaped
    rows with ``type='tandem_repeat'``."""
    raise NotImplementedError(f"find_tandem_repeats pending {_PHASE}")


__all__ = [
    "TELOMERE_MOTIFS",
    "find_telomeres",
    "find_tandem_repeats",
]
