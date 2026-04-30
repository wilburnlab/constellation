"""Dorado-based polishing — minimap2 alignment + Dorado polish.

Polish loop: align reads back to draft contigs with minimap2, then
``dorado polish`` consumes the alignment and emits a polished FASTA.
We wrap the two-step pipeline as one runner so the project layer
sees a single verb.

Status: STUB. Pending Phase 8 (assembly/{polish, ragtag} +
thirdparty/{minimap2, samtools, ragtag}).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constellation.sequencing.assembly.assembly import Assembly


_PHASE = "Phase 8 (assembly/polish + minimap2/samtools/ragtag thirdparty)"


@dataclass(frozen=True)
class PolishRunner:
    """Polish a draft Assembly with Dorado + minimap2.

    ``run(assembly, read_paths, rounds=N)`` executes N polishing
    rounds, each one aligning reads with minimap2, sorting + indexing
    with samtools, then invoking ``dorado polish``. Returns a new
    ``Assembly`` with ``polish_rounds`` incremented per contig.
    """

    rounds: int = 1
    threads: int = 8

    def run(
        self,
        assembly: Assembly,
        read_paths: list[Path],
        output_dir: Path,
        *,
        rounds: int | None = None,
    ) -> Assembly:
        raise NotImplementedError(f"PolishRunner.run pending {_PHASE}")


__all__ = ["PolishRunner"]
