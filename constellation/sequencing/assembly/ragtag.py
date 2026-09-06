"""RagTag subprocess orchestrator — reference-guided scaffolding.

RagTag aligns a draft assembly's contigs to a reference genome, infers
their order + orientation, and joins them with gap-Ns into scaffolds
(``ragtag.scaffold.fasta`` + ``ragtag.scaffold.agp``). The resulting
``Assembly`` carries the scaffold sequences as its contigs (so polish /
alignment downstream operate on the joined sequences) and a
``SCAFFOLD_TABLE`` recording how the draft contigs composed each scaffold
(parsed from the AGP — a provenance id space, see ``Assembly.validate``).

Composes the generic :func:`ragtag_run` wrapper. RagTag (MIT) resolves
via :func:`constellation.thirdparty.find('ragtag')`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from constellation.core.progress import ProgressCallback
from constellation.sequencing.assembly.agp import parse_agp
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.ragtag_run import ragtag_run
from constellation.sequencing.readers.fastx import read_fasta_genome
from constellation.sequencing.reference.materialize import materialise_genome_fasta
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.assembly import SCAFFOLD_TABLE
from constellation.thirdparty.registry import ToolNotFoundError, find


@dataclass(frozen=True)
class RagTagRunner:
    """Reference-guided scaffolding via ``ragtag.py scaffold``."""

    threads: int = 8
    extra_args: tuple[str, ...] = ()

    def run(
        self,
        assembly: Assembly,
        reference: GenomeReference,
        output_dir: Path,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> Assembly:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ref_fasta = materialise_genome_fasta(
            reference,
            output_dir / "reference.fasta",
            output_dir / "reference.fasta.meta.json",
        )
        draft = assembly.to_genome_reference()
        query_fasta = materialise_genome_fasta(
            draft,
            output_dir / "query.fasta",
            output_dir / "query.fasta.meta.json",
        )

        ragtag_dir = output_dir / "ragtag"
        ragtag_run(
            ref_fasta,
            query_fasta,
            ragtag_dir,
            args=tuple(self.extra_args),
            threads=self.threads,
            progress_cb=progress_cb,
        )

        agp_path = ragtag_dir / "ragtag.scaffold.agp"
        scaffold_fasta = ragtag_dir / "ragtag.scaffold.fasta"

        # SCAFFOLD_TABLE from the AGP (references the draft contig id space).
        name_to_id = dict(
            zip(
                assembly.contigs.column("name").to_pylist(),
                assembly.contigs.column("contig_id").to_pylist(),
            )
        )
        scaffold_rows = parse_agp(agp_path, name_to_id)
        scaffolds_table = (
            pa.Table.from_pylist(scaffold_rows, schema=SCAFFOLD_TABLE)
            if scaffold_rows
            else None
        )

        # New contigs = the scaffold sequences themselves.
        scaffold_genome = read_fasta_genome(scaffold_fasta)
        provenance = json.dumps(
            {
                "stage": "scaffold",
                "tool": "ragtag",
                "version": _ragtag_version(),
                "reference_contigs": int(reference.contigs.num_rows),
                "args": list(self.extra_args),
            }
        )
        return Assembly.from_genome_reference(
            scaffold_genome,
            scaffolds=scaffolds_table,
            haplotype="scaffold",
            provenance_json=provenance,
            metadata_extras={
                "stage": "scaffold",
                "scaffolded_fasta": str(scaffold_fasta),
                "agp": str(agp_path),
            },
        )


def _ragtag_version() -> str | None:
    try:
        return find("ragtag").version
    except ToolNotFoundError:
        return None


__all__ = ["RagTagRunner"]
