"""RagTag subprocess wrapper — reference-guided scaffolding.

RagTag uses a reference genome to scaffold a draft assembly: aligns
contigs to the reference, infers their order + orientation, and
inserts gap-Ns where needed. Output is a scaffolded FASTA + AGP map.

Status: STUB. Pending Phase 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.reference.reference import Reference


_PHASE = "Phase 8 (assembly/ragtag + thirdparty/ragtag)"


@dataclass(frozen=True)
class RagTagRunner:
    """Reference-guided scaffolding via the ``ragtag.py scaffold``
    subcommand. Produces a ``SCAFFOLD_TABLE`` populating the
    ``Assembly.scaffolds`` slot.
    """

    threads: int = 8
    extra_args: tuple[str, ...] = ()

    def run(
        self,
        assembly: Assembly,
        reference: Reference,
        output_dir: Path,
    ) -> Assembly:
        raise NotImplementedError(f"RagTagRunner.run pending {_PHASE}")


__all__ = ["RagTagRunner"]
