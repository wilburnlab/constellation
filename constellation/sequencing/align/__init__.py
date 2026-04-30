"""Sequence alignment — pairwise, mapping, locate.

Modules named by *verb* (operation), not by backend library, so the
choice of edlib / parasail / minimap2 / mappy is a kwarg of the
operation rather than visible at the import level.

    pairwise    pairwise_align(query, ref, *, backend='edlib'|'parasail')
    map         many-to-one mapping (minimap2 / mappy) — the workhorse
                for long-read genome / transcriptome alignment
    locate      error-tolerant substring search anchoring demux
                (edlib edit-distance scan over short queries)

Status: STUB. Pending Phase 4.
"""

from __future__ import annotations

from constellation.sequencing.align.locate import locate_substring
from constellation.sequencing.align.map import map_reads
from constellation.sequencing.align.pairwise import pairwise_align

__all__ = [
    "pairwise_align",
    "map_reads",
    "locate_substring",
]
