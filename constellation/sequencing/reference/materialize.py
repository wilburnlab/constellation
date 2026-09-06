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


def _contig_identity(genome: "GenomeReference") -> str:
    """Digest of (name, length, SEQUENCE) over all contigs, in table order.

    Sequence content is load-bearing, not an optimisation to skip. An
    earlier version hashed names and lengths only, on the reasoning that
    "any real edit moves at least one length" — which is false for
    exactly the operation this cache serves. Polishing is
    substitution-heavy: correcting a SNP rewrites bases while leaving
    every name and length identical, so a polished assembly and its
    draft hashed the same and the stale FASTA was reused.

    The cost is one pass over the sequences, which is what writing them
    costs anyway, so a miss is no more expensive than it was.
    """
    import hashlib

    contigs = genome.contigs
    names = contigs.column("name").to_pylist()
    lengths = (
        contigs.column("length").to_pylist()
        if "length" in contigs.column_names
        else [None] * len(names)
    )
    contig_ids = contigs.column("contig_id").to_pylist()
    h = hashlib.blake2b(digest_size=16)
    for contig_id, nm, ln in zip(contig_ids, names, lengths):
        h.update(f"{nm}\t{ln}\n".encode())
        h.update(genome.sequence_of(int(contig_id)).encode())
    return h.hexdigest()


def materialise_genome_fasta(
    genome: "GenomeReference",
    fasta_path: Path,
    meta_path: Path,
) -> Path:
    """Write ``genome``'s contigs to ``fasta_path``, caching by identity.

    Skips the rewrite when ``fasta_path`` exists and ``meta_path`` records
    a matching contig *identity* — names and lengths, not merely the
    count. Counting alone made a rerun into an existing directory treat a
    DIFFERENT assembly with the same number of contigs as a cache hit, so
    polishing aligned and called consensus against the previous
    assembly and produced a plausible but entirely wrong result.

    Names + lengths are read from the contig table and cost nothing
    relative to writing the sequences, and any real edit (a polish round,
    a rescaffold) moves at least one length. Returns ``fasta_path``.
    """
    fasta_path = Path(fasta_path)
    meta_path = Path(meta_path)
    expected_n = int(genome.contigs.num_rows)
    key = _contig_identity(genome)
    if fasta_path.exists() and meta_path.exists():
        try:
            stamp = json.loads(meta_path.read_text())
            if stamp.get("contig_identity") == key:
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
        json.dumps(
            {
                "n_contigs": expected_n,
                "contig_identity": key,
                "fasta": fasta_path.name,
            },
            indent=2,
        )
    )
    return fasta_path


__all__ = ["materialise_genome_fasta"]
