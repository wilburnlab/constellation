"""Sequence alignment — pairwise, mapping, locate.

Two-layer split for the mapping verbs: ``minimap2`` houses the generic
subprocess wrapper (use-case-agnostic), and ``map`` houses the use-case
orchestrators (``map_to_genome`` for splice-aware cDNA → genome; future
``map_assembly`` and ``map_dna_to_genome`` compose the same primitive).

    pairwise    pairwise_align(query, ref, *, backend='edlib')
    map         use-case orchestrators on top of minimap2_run
    minimap2    generic minimap2 subprocess runner + index builder
    locate      error-tolerant substring search (edlib)
"""

from __future__ import annotations

from constellation.sequencing.align.locate import locate_substring
from constellation.sequencing.align.map import map_to_genome
from constellation.sequencing.align.minimap2 import minimap2_build_index, minimap2_run
from constellation.sequencing.align.pairwise import pairwise_align

__all__ = [
    "pairwise_align",
    "map_to_genome",
    "minimap2_run",
    "minimap2_build_index",
    "locate_substring",
]
