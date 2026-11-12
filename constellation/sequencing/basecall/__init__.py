"""Basecaller integration — Dorado today, in-house physics-based later.

``DoradoRunner`` is a thin subprocess wrapper around the ``dorado``
binary (resolved via :func:`constellation.thirdparty.find('dorado')`).
Subcommand methods (``basecaller``, ``duplex``, ``aligner``, ``demux``,
``summary``) all return :class:`RunHandle` objects rather than blocking,
so 3–4-day basecalls support detached execution / re-attach / resume.

``DoradoModel`` is Dorado's model argument, checked by
:meth:`DoradoModel.parse` and passed through as written: a variant
(``"sup"``, ``"hac@v5.0.0,5mCG_5hmCG"``), a full model name
(``"dna_r10.4.1_e8.2_400bps_sup@v5.2.0"``) or a model directory.

License: Dorado is distributed under the Oxford Nanopore Technologies
PLC. Public License Version 1.0 — Research Purposes only.
Constellation is Apache 2.0; the wrapper is permissible, but anyone
*invoking* the wrapper independently accepts ONT's terms. See LICENCE
discussion in CLAUDE.md / plan file.
"""

from __future__ import annotations

from constellation.sequencing.basecall.dorado import DoradoRunner, RunHandle, RunStatus
from constellation.sequencing.basecall.models import DoradoModel

__all__ = [
    "DoradoRunner",
    "DoradoModel",
    "RunHandle",
    "RunStatus",
]
