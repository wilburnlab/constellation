"""Genome track kernels.

Each module holds one kernel, self-registered via ``@register_track``;
importing this package pulls them all in exactly once.

Kernels (reading slot names per ``genome.session.GenomeSource``):

- ``reference_sequence`` — vector; one binding for the session's reference.
- ``gene_annotation``    — vector; the reference annotation, plus one binding
                           per align source's ``derived_annotation/``.
- ``coverage_histogram`` — vector; one binding per align source's
                           ``coverage.parquet``.
- ``read_pileup``        — vector / hybrid; one binding per align source's
                           ``alignments/`` + ``alignment_blocks/``.
- ``cluster_pileup``     — vector / hybrid; one binding per cluster source.
- ``splice_junctions``   — vector; one binding per align source's
                           ``introns.parquet``.

``_alignment_view`` holds the block-attach and mismatch-position helpers
shared by the two pile-up kernels.

The mirror-symmetric TS renderer for each kernel lives under
``constellation/viz/frontend/src/track_renderers/<kind>.ts``.
"""

from constellation.viz.modalities.genome.tracks import (  # noqa: F401
    cluster_pileup,
    coverage_histogram,
    gene_annotation,
    read_pileup,
    reference_sequence,
    splice_junctions,
)
