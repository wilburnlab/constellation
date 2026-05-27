"""S2 align subpackage — transcriptome-specific align-stage logic.

Currently a thin placeholder: the heavy lifting for ``constellation
transcriptome align`` lives in :mod:`constellation.sequencing.align`
(the minimap2 orchestrator + cigar parsers) and
:mod:`constellation.sequencing.quant` (filter + gene-overlap kernels).
This subpackage exists as the home for align-stage *diagnostic*
reports (``diagnostics.py``); it may absorb more transcriptome-specific
align-stage code as that surface grows.
"""

from __future__ import annotations
