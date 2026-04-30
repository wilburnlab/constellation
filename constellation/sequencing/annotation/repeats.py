"""Repeat-element detection — RepeatMasker, EDTA, or in-house.

Repeat masking is a notorious tooling rabbit hole. The choice between
RepeatMasker (Smit / Hubley; needs a repeat library), EDTA (Ou et al.
2019; de novo TE library construction), and lighter k-mer-based
heuristics is per-project. Phase 9 ships a runner abstraction that
hides the choice — concrete backends (RepeatMaskerBackend,
EDTABackend, KmerBackend) plug in below.

Status: STUB. Pending Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa

from constellation.sequencing.reference.reference import Reference


_PHASE = "Phase 9 (annotation/repeats)"


@dataclass(frozen=True)
class RepeatsRunner:
    """Repeat-element detection over a Reference.

    Returns ``FEATURE_TABLE``-shaped rows with ``type='repeat_region'``
    and per-repeat-class metadata in ``attributes_json``.
    """

    backend: Literal["repeatmasker", "edta", "kmer"] = "repeatmasker"
    threads: int = 8
    repeat_library: Path | None = None  # Required for RepeatMasker

    def run(
        self,
        reference: Reference,
        output_dir: Path,
    ) -> pa.Table:
        raise NotImplementedError(f"RepeatsRunner.run pending {_PHASE}")


__all__ = ["RepeatsRunner"]
