"""Per-read splicing-topology fingerprints.

The Phase-2 cluster key. For each primary alignment, project its
``ALIGNMENT_BLOCK_TABLE`` rows into a canonical junction sequence,
quantise donor/acceptor coordinates by ``intron_quantum_bp`` (default
10 — absorbs small cryptic-splice excursions; see CLAUDE.md
"PolyA-barcode boundary ambiguity" / "scoring-domain rule" precedents
for the lab's history of preserving observable basecaller artifacts),
hash via ``blake2b`` to a uint64, and emit ``READ_FINGERPRINT_TABLE``.

Two reads with the same ``fingerprint_hash`` describe the same
internal splicing topology — they have the same intron chain at the
quantum's resolution. Terminal-exon (TSS / polyA) drift is
intentionally NOT in the hash; the genome-guided clusterer applies
``--max-5p-drift`` / ``--max-3p-drift`` filters as a downstream
secondary step (see ``cluster_genome.py`` when it lands in Phase 2).

Primary-alignment-only — secondary + supplementary alignments are
excluded so a single read maps to at most one fingerprint row.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.schemas.alignment import READ_FINGERPRINT_TABLE


def _build_contig_id_lookup(contigs: pa.Table) -> dict[str, int]:
    """``CONTIG_TABLE.name`` → ``contig_id`` dict."""
    return {
        str(name): int(cid)
        for name, cid in zip(
            contigs.column("name").to_pylist(),
            contigs.column("contig_id").to_pylist(),
            strict=True,
        )
    }


def _hash_junction_sequence(
    contig_id: int,
    strand: str,
    junctions: list[tuple[int, int]],
) -> int:
    """blake2b-8 over the canonical tuple. Stdlib only; no xxhash dep."""
    h = hashlib.blake2b(digest_size=8)
    h.update(contig_id.to_bytes(8, "little", signed=True))
    h.update(strand.encode("ascii"))
    h.update(len(junctions).to_bytes(4, "little", signed=False))
    for donor, acceptor in junctions:
        h.update(donor.to_bytes(8, "little", signed=True))
        h.update(acceptor.to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest(), "little", signed=False)


def _format_signature(
    contig_name: str,
    junctions: list[tuple[int, int]],
) -> str:
    """Human-readable signature: ``"chrN:donor1-acceptor1,donor2-acceptor2"``.

    Diagnostic-only; not the cluster key. Empty intron-chain (single
    block) renders as ``"chrN:."``.
    """
    if not junctions:
        return f"{contig_name}:."
    parts = ",".join(f"{d}-{a}" for d, a in junctions)
    return f"{contig_name}:{parts}"


def compute_read_fingerprints(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    contigs: pa.Table,
    *,
    intron_quantum_bp: int = 10,
) -> pa.Table:
    """Build the ``READ_FINGERPRINT_TABLE`` from per-read alignment blocks.

    Primary alignments only — rows where ``is_secondary or
    is_supplementary`` are dropped. The hash absorbs small cryptic-
    splice-site excursions (within ±``intron_quantum_bp``) into the
    same fingerprint bucket while preserving alt-splicing topology
    differences.

    Parameters
    ----------
    alignment_blocks : pa.Table
        ``ALIGNMENT_BLOCK_TABLE``-shaped. Blocks are grouped by
        ``alignment_id`` and ordered by ``block_index`` internally.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies ``read_id``, ``ref_name``,
        ``strand``, and the secondary/supplementary flags.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — supplies the ``ref_name → contig_id``
        resolution.
    intron_quantum_bp : int, default 10
        Donor/acceptor coordinates are quantised to multiples of this
        value before hashing.

    Returns
    -------
    pa.Table conforming to ``READ_FINGERPRINT_TABLE``. Empty if no
    primary alignment has any aligned block.
    """
    if intron_quantum_bp < 1:
        raise ValueError(
            f"intron_quantum_bp must be ≥ 1, got {intron_quantum_bp}"
        )
    if alignment_blocks.num_rows == 0 or alignments.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    # Filter to primary alignments — the cardinality contract is one
    # fingerprint row per read, and primary alignments are how we
    # disambiguate when minimap2 emits secondary/supplementary.
    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask).select(
        ["alignment_id", "read_id", "ref_name", "strand"]
    )
    if primary.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    # Hash-join blocks to primary on alignment_id; secondary/supplementary
    # rows in alignment_blocks fall away naturally.
    joined = alignment_blocks.join(primary, keys="alignment_id", join_type="inner")
    if joined.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    name_to_id = _build_contig_id_lookup(contigs)

    # Group blocks per alignment, ordered by block_index.
    by_aid: dict[int, list[tuple[int, int, int]]] = {}
    aid_meta: dict[int, dict[str, object]] = {}
    aids = joined.column("alignment_id").to_pylist()
    bidx = joined.column("block_index").to_pylist()
    rstart = joined.column("ref_start").to_pylist()
    rend = joined.column("ref_end").to_pylist()
    rid = joined.column("read_id").to_pylist()
    rname = joined.column("ref_name").to_pylist()
    strand = joined.column("strand").to_pylist()

    for a, bi, rs, re_, r_id, ref_n, st in zip(
        aids, bidx, rstart, rend, rid, rname, strand, strict=True
    ):
        by_aid.setdefault(int(a), []).append((int(bi), int(rs), int(re_)))
        if int(a) not in aid_meta:
            aid_meta[int(a)] = {
                "read_id": str(r_id),
                "ref_name": str(ref_n),
                "strand": str(st),
            }

    q = int(intron_quantum_bp)
    rows: list[dict[str, object]] = []
    for aid, blocks in by_aid.items():
        blocks.sort(key=lambda t: t[0])
        meta = aid_meta[aid]
        contig_id = name_to_id.get(meta["ref_name"])
        if contig_id is None:
            # Alignment to a contig the genome doesn't list — skip.
            continue
        ordered_starts = [b[1] for b in blocks]
        ordered_ends = [b[2] for b in blocks]
        # Quantised junction sequence (donor = prev block end,
        # acceptor = next block start).
        raw_junctions = list(zip(ordered_ends[:-1], ordered_starts[1:]))
        quant_junctions = [((d // q) * q, (a // q) * q) for d, a in raw_junctions]
        h = _hash_junction_sequence(
            contig_id=int(contig_id),
            strand=meta["strand"],
            junctions=quant_junctions,
        )
        sig = _format_signature(meta["ref_name"], raw_junctions)
        rows.append(
            {
                "read_id": meta["read_id"],
                "contig_id": int(contig_id),
                "strand": meta["strand"],
                "n_blocks": int(len(blocks)),
                "span_start": int(ordered_starts[0]),
                "span_end": int(ordered_ends[-1]),
                "fingerprint_hash": int(h),
                "junction_signature": sig,
            }
        )

    if not rows:
        return READ_FINGERPRINT_TABLE.empty_table()
    return pa.Table.from_pylist(rows, schema=READ_FINGERPRINT_TABLE)


__all__ = ["compute_read_fingerprints"]
