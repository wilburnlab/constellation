"""Per-read splicing-topology fingerprints.

The Phase-2 cluster key. For each primary alignment, project its
``ALIGNMENT_BLOCK_TABLE`` rows into a per-read junction sequence,
look up each ``(donor, acceptor)`` against the supplied
``INTRON_TABLE`` to substitute its canonical ``intron_id``, hash via
``blake2b`` over the resulting ``(contig_id, strand, [intron_id, ...])``
tuple, and emit ``READ_FINGERPRINT_TABLE``.

Two reads with the same ``fingerprint_hash`` describe the same
canonical splicing topology — their junctions map to the same intron
clusters in the same order. Donor / acceptor positions don't enter the
hash directly; the ``intron_id`` is the cluster identifier (assigned
by :func:`constellation.sequencing.quant.junctions.cluster_junctions`),
so basecaller-driven single-bp jitter is absorbed automatically as long
as the introns table was clustered with a matching tolerance.

Terminal-exon (TSS / polyA) drift is NOT in the hash; the genome-guided
clusterer applies ``--max-5p-drift`` / ``--max-3p-drift`` filters as a
downstream secondary step (see ``cluster_genome.py``).

Primary-alignment-only — secondary + supplementary alignments are
excluded so a single read maps to at most one fingerprint row.

Reads whose per-read junctions don't have an exact ``(contig_id, strand,
donor_pos, acceptor_pos)`` match in the supplied introns table are
**skipped** (no fingerprint row emitted). This is the same silent-drop
policy used elsewhere in the resolve stage (e.g. unknown-contig drops);
in normal pipeline use the introns table is built from the same
``alignment_blocks`` input so every junction has a match. Mismatched
inputs surface as "fewer fingerprints than reads."
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


def _build_intron_lookup(
    introns: pa.Table,
) -> dict[tuple[int, str, int, int], int]:
    """``(contig_id, strand, donor_pos, acceptor_pos)`` → ``intron_id``.

    Multiple rows in INTRON_TABLE can share an ``intron_id`` (cluster
    membership), but the four-tuple key is unique per row by construction
    (``aggregate_junctions`` produces one row per distinct observed
    position pair, and ``cluster_junctions`` preserves row count).
    """
    contig_ids = introns.column("contig_id").to_pylist()
    strands = introns.column("strand").to_pylist()
    donors = introns.column("donor_pos").to_pylist()
    acceptors = introns.column("acceptor_pos").to_pylist()
    intron_ids = introns.column("intron_id").to_pylist()
    return {
        (int(c), str(s), int(d), int(a)): int(iid)
        for c, s, d, a, iid in zip(
            contig_ids, strands, donors, acceptors, intron_ids,
            strict=True,
        )
    }


def _hash_intron_sequence(
    contig_id: int,
    strand: str,
    intron_ids: list[int],
) -> int:
    """blake2b-8 over the canonical tuple. Stdlib only."""
    h = hashlib.blake2b(digest_size=8)
    h.update(contig_id.to_bytes(8, "little", signed=True))
    h.update(strand.encode("ascii"))
    h.update(len(intron_ids).to_bytes(4, "little", signed=False))
    for iid in intron_ids:
        h.update(iid.to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest(), "little", signed=False)


def _format_signature(
    contig_name: str,
    junctions: list[tuple[int, int]],
) -> str:
    """Human-readable signature: ``"chrN:donor1-acceptor1,donor2-acceptor2"``.

    Uses raw observed donor / acceptor positions (not seed positions or
    intron_ids) so per-read splicing variation stays visible in the
    diagnostic output even when reads cluster together. Diagnostic-only;
    not the cluster key. Empty intron-chain (single block) renders as
    ``"chrN:."``.
    """
    if not junctions:
        return f"{contig_name}:."
    parts = ",".join(f"{d}-{a}" for d, a in junctions)
    return f"{contig_name}:{parts}"


def compute_read_fingerprints(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    contigs: pa.Table,
    introns: pa.Table,
) -> pa.Table:
    """Build the ``READ_FINGERPRINT_TABLE`` from per-read alignment blocks
    + a clustered ``INTRON_TABLE``.

    Primary alignments only — rows where ``is_secondary or
    is_supplementary`` are dropped. Per-read junction positions are
    substituted with the canonical ``intron_id`` from the supplied
    introns table before hashing, so reads sharing a clustered intron
    chain (regardless of single-bp jitter at individual splice sites)
    bucket together.

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
    introns : pa.Table
        ``INTRON_TABLE``-shaped — supplies the per-position-pair →
        ``intron_id`` mapping. Typically the output of
        ``cluster_junctions(aggregate_junctions(...))`` against the
        same ``alignment_blocks``.

    Returns
    -------
    pa.Table conforming to ``READ_FINGERPRINT_TABLE``. Empty if no
    primary alignment has any aligned block.
    """
    if (
        alignment_blocks.num_rows == 0
        or alignments.num_rows == 0
        or introns.num_rows == 0
    ):
        # When the introns table is empty but blocks/alignments aren't,
        # single-block alignments still produce zero-junction
        # fingerprints — they don't need an introns lookup at all.
        # Fall through to the main path so this case is handled below.
        if introns.num_rows == 0 and (
            alignment_blocks.num_rows == 0 or alignments.num_rows == 0
        ):
            return READ_FINGERPRINT_TABLE.empty_table()
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
    intron_lookup = _build_intron_lookup(introns)

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
        # Per-read raw junction sequence: donor = prev block end,
        # acceptor = next block start.
        raw_junctions = list(zip(ordered_ends[:-1], ordered_starts[1:]))

        # Substitute each (donor, acceptor) → intron_id via the lookup.
        # Skip the read entirely if any junction has no exact match
        # (mismatched inputs).
        strand_str = str(meta["strand"])
        intron_id_seq: list[int] = []
        missing = False
        for donor, acceptor in raw_junctions:
            key = (int(contig_id), strand_str, int(donor), int(acceptor))
            iid = intron_lookup.get(key)
            if iid is None:
                missing = True
                break
            intron_id_seq.append(iid)
        if missing:
            continue

        h = _hash_intron_sequence(
            contig_id=int(contig_id),
            strand=strand_str,
            intron_ids=intron_id_seq,
        )
        sig = _format_signature(meta["ref_name"], raw_junctions)
        rows.append(
            {
                "read_id": meta["read_id"],
                "contig_id": int(contig_id),
                "strand": strand_str,
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
