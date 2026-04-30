"""BUSCO subprocess wrapper — completeness + ortholog gene calls.

BUSCO (Benchmarking Universal Single-Copy Orthologs) is the standard
genome-completeness metric for eukaryotic assemblies. Lineage-specific
ortholog datasets (eukaryota_odb10, vertebrata_odb12, etc.) live
outside the repo; the runner respects ``BUSCO_DOWNLOADS_PATH`` for the
on-disk lineage cache.

Status: STUB. Pending Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from constellation.sequencing.assembly.assembly import Assembly


_PHASE = "Phase 9 (annotation/busco + thirdparty/busco)"


@dataclass(frozen=True)
class BuscoRunner:
    """Runs ``busco`` against an assembly and emits both:

    - busco_* completeness columns for ASSEMBLY_STATS
    - per-ortholog FEATURE_TABLE rows (gene predictions for the
      complete + duplicated orthologs)
    """

    lineage: str  # 'vertebrata_odb12', 'eukaryota_odb10', ...
    threads: int = 8
    mode: str = "genome"  # 'genome' | 'transcriptome' | 'protein'

    def run(
        self,
        assembly: Assembly,
        output_dir: Path,
    ) -> tuple[pa.Table, pa.Table]:
        """Returns ``(stats_row, feature_rows)``.

        ``stats_row`` is a one-row Arrow table populating the busco_*
        columns of ``ASSEMBLY_STATS``. ``feature_rows`` are
        ``FEATURE_TABLE``-shaped rows for the predicted ortholog
        genes — caller appends to ``Reference.features``.
        """
        raise NotImplementedError(f"BuscoRunner.run pending {_PHASE}")


__all__ = ["BuscoRunner"]
