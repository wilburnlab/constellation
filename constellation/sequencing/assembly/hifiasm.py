"""HiFiASM subprocess wrapper.

HiFiASM (Cheng et al. 2021) is the lab's primary genome assembler for
nanopore long reads — produces dual-haplotype assemblies + p_ctg
(primary contigs) GFA output that we project to ``Assembly``.
Resolved via :func:`constellation.thirdparty.find('hifiasm')`.

Status: STUB. Pending Phase 7 (assembly/hifiasm + thirdparty/hifiasm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constellation.sequencing.assembly.assembly import Assembly


_PHASE = "Phase 7 (assembly + thirdparty/hifiasm)"


@dataclass(frozen=True)
class HiFiAsmRunner:
    """Thin subprocess wrapper around the ``hifiasm`` binary.

    ``run(reads_paths, output_prefix, threads=...)`` invokes hifiasm,
    waits for completion, parses the resulting GFA + FASTA outputs,
    and returns an ``Assembly`` projecting the primary-contig output.
    For dual-haplotype workflows, ``run_diploid`` returns a tuple of
    Assembly objects (haplotype 1, haplotype 2).
    """

    threads: int = 16
    extra_args: tuple[str, ...] = ()

    def run(
        self,
        read_paths: list[Path],
        output_prefix: Path,
        *,
        threads: int | None = None,
    ) -> Assembly:
        raise NotImplementedError(f"HiFiAsmRunner.run pending {_PHASE}")

    def run_diploid(
        self,
        read_paths: list[Path],
        output_prefix: Path,
        *,
        threads: int | None = None,
    ) -> tuple[Assembly, Assembly]:
        raise NotImplementedError(
            f"HiFiAsmRunner.run_diploid pending {_PHASE}"
        )


__all__ = ["HiFiAsmRunner"]
