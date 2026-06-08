"""Materialize a ``GenomeReference`` to a FASTA file on disk.

Several external tools (minimap2 / dorado aligner index a reference,
RagTag scaffolds against one, dorado polish reads the draft) need the
sequence as a plain FASTA, not the in-memory Arrow container. This is
the single shared writer — promoted out of ``align/map.py`` so the align,
scaffold, and polish stages all use one implementation.

The cache key is contig count only (coarse): adequate for the
"import once, align/scaffold many" workflow. If a caller mutates a
``GenomeReference`` between invocations against the same ``fasta_path``,
delete the meta file to force a rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constellation.sequencing.reference.reference import GenomeReference


def materialise_genome_fasta(
    genome: "GenomeReference",
    fasta_path: Path,
    meta_path: Path,
) -> Path:
    """Write ``genome``'s contigs to ``fasta_path``, caching by contig count.

    Skips the rewrite when ``fasta_path`` exists and ``meta_path`` records
    a matching contig count. Returns ``fasta_path``.
    """
    fasta_path = Path(fasta_path)
    meta_path = Path(meta_path)
    expected_n = int(genome.contigs.num_rows)
    if fasta_path.exists() and meta_path.exists():
        try:
            stamp = json.loads(meta_path.read_text())
            if int(stamp.get("n_contigs", -1)) == expected_n:
                return fasta_path
        except (OSError, json.JSONDecodeError):
            pass

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    contig_ids = genome.contigs.column("contig_id").to_pylist()
    names = genome.contigs.column("name").to_pylist()
    with fasta_path.open("w", encoding="utf-8") as fh:
        for contig_id, name in zip(contig_ids, names):
            fh.write(f">{name}\n")
            fh.write(genome.sequence_of(int(contig_id)))
            fh.write("\n")
    meta_path.write_text(
        json.dumps({"n_contigs": expected_n, "fasta": fasta_path.name}, indent=2)
    )
    return fasta_path


__all__ = ["materialise_genome_fasta"]
