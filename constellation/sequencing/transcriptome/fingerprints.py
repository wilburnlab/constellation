"""Per-read splicing-topology fingerprints.

The Phase-2 cluster key. For each primary alignment, project its
``ALIGNMENT_BLOCK_TABLE`` rows into a per-read junction sequence,
look up each ``(donor, acceptor)`` against the supplied
``INTRON_TABLE`` to substitute its canonical ``intron_id``, hash via
``xxh64`` over the resulting ``(contig_id, strand, [intron_id, ...])``
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

Implementation: Arrow joins + group_by for the per-block work, numpy
shift for vectorized junction derivation, hash-join for intron_id
substitution, and a tight per-alignment xxh64 loop bounded by
N_alignments (not N_blocks). String columns (``read_id``, ``ref_name``)
stay out of the heavy block-cardinality sort — they re-attach via a
small post-aggregation join. See ``sequencing/CLAUDE.md`` "Resolve-stage
pitfalls" for the doctrine.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import xxhash

from constellation.sequencing.schemas.alignment import READ_FINGERPRINT_TABLE


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
    if alignment_blocks.num_rows == 0 or alignments.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    # ── Filter primaries + resolve contig_id (inner-join drops unknown contigs)
    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask).select(
        ["alignment_id", "read_id", "ref_name", "strand"]
    )
    if primary.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    contigs_min = contigs.select(["contig_id", "name"]).rename_columns(
        ["contig_id", "ref_name"]
    )
    primary = primary.join(contigs_min, keys="ref_name", join_type="inner")
    if primary.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    # ── Join blocks ⨝ primary on numeric-only columns; strings stay out
    # of the block-cardinality sort. read_id (~22 GB at PromethION scale,
    # int32 offset-overflow risk) lives only on the alignment-cardinality
    # `primary` table; we re-attach it via a small join at the end.
    primary_numeric = primary.select(["alignment_id", "contig_id", "strand"])
    blocks_proj = alignment_blocks.select(
        ["alignment_id", "block_index", "ref_start", "ref_end"]
    )
    joined = blocks_proj.join(
        primary_numeric, keys="alignment_id", join_type="inner"
    )
    if joined.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    # ── Sort by (alignment_id, block_index). strand carries as a small
    # string column (1-char values), well under int32 offset limit at
    # block cardinality. Mirrors the aggregate_junctions blueprint at
    # quant/junctions.py:188-206.
    sorted_blocks = joined.sort_by(
        [("alignment_id", "ascending"), ("block_index", "ascending")]
    ).combine_chunks()

    # ── Per-alignment span + n_blocks + contig_id + strand (group_by aggregate).
    # contig_id and strand are constant within alignment; "min" is a cheap
    # constant-value extractor.
    span_agg = (
        sorted_blocks.group_by("alignment_id")
        .aggregate(
            [
                ("ref_start", "min"),
                ("ref_end", "max"),
                ("block_index", "count"),
                ("contig_id", "min"),
                ("strand", "min"),
            ]
        )
        .rename_columns(
            [
                "alignment_id",
                "span_start",
                "span_end",
                "n_blocks",
                "contig_id",
                "strand",
            ]
        )
    )

    # ── Vectorized junction derivation via numpy shift.
    # For each row except the first of its alignment, donor = prev row's
    # ref_end, acceptor = this row's ref_start. The mask drops first-of-
    # alignment rows (no prev row in the same alignment).
    aids = sorted_blocks.column("alignment_id").to_numpy(zero_copy_only=False)
    starts = sorted_blocks.column("ref_start").to_numpy(zero_copy_only=False)
    ends = sorted_blocks.column("ref_end").to_numpy(zero_copy_only=False)
    contig_arr = sorted_blocks.column("contig_id").to_numpy(zero_copy_only=False)

    # Dictionary-encode strand to skip per-row string materialization at
    # block cardinality. The dict has ~3 entries ({'+', '-', '?'}) so the
    # round-trip back to strings at junction cardinality is cheap.
    strand_da = pc.dictionary_encode(sorted_blocks.column("strand"))
    if isinstance(strand_da, pa.ChunkedArray):
        strand_da = strand_da.combine_chunks()
    strand_values = strand_da.dictionary.to_pylist()
    strand_idx_arr = strand_da.indices.to_numpy(zero_copy_only=False)

    n_rows = len(aids)
    if n_rows > 1:
        same_alignment = aids[1:] == aids[:-1]
        donor_arr = ends[:-1][same_alignment]
        acceptor_arr = starts[1:][same_alignment]
        j_aid = aids[1:][same_alignment]
        j_contig = contig_arr[1:][same_alignment]
        j_strand_idx = strand_idx_arr[1:][same_alignment]
        # junction_order = global position, used to re-sort after the
        # left-outer-join (which doesn't guarantee input order).
        junction_order = np.arange(len(j_aid), dtype=np.int64)
    else:
        donor_arr = np.zeros(0, dtype=np.int64)
        acceptor_arr = np.zeros(0, dtype=np.int64)
        j_aid = np.zeros(0, dtype=np.int64)
        j_contig = np.zeros(0, dtype=np.int64)
        j_strand_idx = np.zeros(0, dtype=np.int32)
        junction_order = np.zeros(0, dtype=np.int64)

    # Project dict-encoded strand indices back to strings for the join
    # with introns.strand (which is plain string). np.take is fast even at
    # junction cardinality; the dictionary is ~3 entries.
    strand_lookup = np.asarray(strand_values, dtype=object)
    if len(j_strand_idx) > 0:
        j_strand_str = strand_lookup[j_strand_idx.astype(np.intp, copy=False)]
        j_strand_pa = pa.array(list(j_strand_str), type=pa.string())
    else:
        j_strand_pa = pa.array([], type=pa.string())

    junctions_table = pa.table(
        {
            "alignment_id": pa.array(j_aid),
            "contig_id": pa.array(j_contig),
            "strand": j_strand_pa,
            "donor_pos": pa.array(donor_arr),
            "acceptor_pos": pa.array(acceptor_arr),
            "junction_order": pa.array(junction_order),
        }
    )

    # ── Lookup intron_id per junction via hash-join with introns.
    # Drop alignments where any junction has no exact (contig_id, strand,
    # donor_pos, acceptor_pos) match — silent-drop policy.
    bad_aid_set: pa.Array | None = None
    if junctions_table.num_rows > 0:
        introns_min = introns.select(
            [
                "intron_id",
                "contig_id",
                "strand",
                "donor_pos",
                "acceptor_pos",
            ]
        )
        j_with_iid = junctions_table.join(
            introns_min,
            keys=["contig_id", "strand", "donor_pos", "acceptor_pos"],
            join_type="left outer",
        )
        missing_mask = pc.is_null(j_with_iid.column("intron_id"))
        bad_aids_col = j_with_iid.filter(missing_mask).column("alignment_id")
        if len(bad_aids_col) > 0:
            bad_aid_set = pc.unique(bad_aids_col)
            keep = pc.invert(
                pc.is_in(
                    j_with_iid.column("alignment_id"), value_set=bad_aid_set
                )
            )
            j_with_iid = j_with_iid.filter(keep)
        # Re-sort to restore (alignment_id, junction_order) order — the join
        # is not order-preserving in general.
        j_with_iid = j_with_iid.sort_by(
            [
                ("alignment_id", "ascending"),
                ("junction_order", "ascending"),
            ]
        ).combine_chunks()
        j_aids_sorted = j_with_iid.column("alignment_id").to_numpy(
            zero_copy_only=False
        )
        j_iids_sorted = j_with_iid.column("intron_id").to_numpy(
            zero_copy_only=False
        )
        j_donor_sorted = j_with_iid.column("donor_pos").to_numpy(
            zero_copy_only=False
        )
        j_accept_sorted = j_with_iid.column("acceptor_pos").to_numpy(
            zero_copy_only=False
        )
    else:
        j_aids_sorted = np.zeros(0, dtype=np.int64)
        j_iids_sorted = np.zeros(0, dtype=np.int64)
        j_donor_sorted = np.zeros(0, dtype=np.int64)
        j_accept_sorted = np.zeros(0, dtype=np.int64)

    # Drop alignments with missing junctions from span_agg too — they'd
    # otherwise emit a zero-junction fingerprint by the empty-slice path.
    if bad_aid_set is not None and len(bad_aid_set) > 0:
        span_agg = span_agg.filter(
            pc.invert(
                pc.is_in(span_agg.column("alignment_id"), value_set=bad_aid_set)
            )
        )

    if span_agg.num_rows == 0:
        return READ_FINGERPRINT_TABLE.empty_table()

    # ── Per-alignment hash + signature loop.
    # Aligned ordering: span_sorted and j_aids_sorted are both sorted by
    # alignment_id ascending. np.searchsorted gives per-alignment slice
    # boundaries in O(N_alignments × log N_junctions).
    span_sorted = span_agg.sort_by(
        [("alignment_id", "ascending")]
    ).combine_chunks()
    sp_aids = span_sorted.column("alignment_id").to_numpy(zero_copy_only=False)
    sp_span_start = span_sorted.column("span_start").to_numpy(
        zero_copy_only=False
    )
    sp_span_end = span_sorted.column("span_end").to_numpy(zero_copy_only=False)
    sp_n_blocks = span_sorted.column("n_blocks").to_numpy(zero_copy_only=False)
    sp_contig = span_sorted.column("contig_id").to_numpy(zero_copy_only=False)
    sp_strand = span_sorted.column("strand").to_pylist()

    n_alignments = len(sp_aids)
    if len(j_aids_sorted) > 0:
        starts_idx = np.searchsorted(j_aids_sorted, sp_aids, side="left")
        ends_idx = np.searchsorted(j_aids_sorted, sp_aids, side="right")
    else:
        starts_idx = np.zeros(n_alignments, dtype=np.int64)
        ends_idx = np.zeros(n_alignments, dtype=np.int64)

    # ref_name lookup from contig_id, used only for the diagnostic signature
    # column. Alignment-cardinality — cheap.
    contig_name_map = {
        int(cid): str(name)
        for cid, name in zip(
            contigs.column("contig_id").to_pylist(),
            contigs.column("name").to_pylist(),
            strict=True,
        )
    }

    out_hashes = np.empty(n_alignments, dtype=np.uint64)
    out_sigs: list[str] = [""] * n_alignments

    xxh = xxhash.xxh64_intdigest
    for i in range(n_alignments):
        s = int(starts_idx[i])
        e = int(ends_idx[i])
        contig_id_i = int(sp_contig[i])
        strand_i = str(sp_strand[i])
        contig_bytes = int(contig_id_i).to_bytes(8, "little", signed=True)
        strand_bytes = strand_i.encode("ascii")
        if e > s:
            iid_slice = j_iids_sorted[s:e].astype(np.int64, copy=False)
            len_bytes = int(e - s).to_bytes(4, "little", signed=False)
            out_hashes[i] = xxh(
                contig_bytes + strand_bytes + len_bytes + iid_slice.tobytes()
            )
            donor_slice = j_donor_sorted[s:e]
            accept_slice = j_accept_sorted[s:e]
            contig_name = contig_name_map.get(
                contig_id_i, f"contig_{contig_id_i}"
            )
            parts = ",".join(
                f"{int(d)}-{int(a)}"
                for d, a in zip(donor_slice, accept_slice, strict=True)
            )
            out_sigs[i] = f"{contig_name}:{parts}"
        else:
            zero_len_bytes = (0).to_bytes(4, "little", signed=False)
            out_hashes[i] = xxh(contig_bytes + strand_bytes + zero_len_bytes)
            contig_name = contig_name_map.get(
                contig_id_i, f"contig_{contig_id_i}"
            )
            out_sigs[i] = f"{contig_name}:."

    # ── Attach read_id via small post-aggregation join (alignment-cardinality
    # with the read_id string column — small).
    summary = pa.table(
        {
            "alignment_id": pa.array(sp_aids, type=pa.int64()),
            "contig_id": pa.array(sp_contig, type=pa.int64()),
            "strand": pa.array(sp_strand, type=pa.string()),
            "n_blocks": pa.array(
                sp_n_blocks.astype(np.int32, copy=False), type=pa.int32()
            ),
            "span_start": pa.array(sp_span_start, type=pa.int64()),
            "span_end": pa.array(sp_span_end, type=pa.int64()),
            "fingerprint_hash": pa.array(out_hashes, type=pa.uint64()),
            "junction_signature": pa.array(out_sigs, type=pa.string()),
        }
    )
    primary_rid = primary.select(["alignment_id", "read_id"])
    summary = summary.join(primary_rid, keys="alignment_id", join_type="inner")

    return pa.table(
        {
            "read_id": summary.column("read_id"),
            "contig_id": summary.column("contig_id"),
            "strand": summary.column("strand"),
            "n_blocks": summary.column("n_blocks"),
            "span_start": summary.column("span_start"),
            "span_end": summary.column("span_end"),
            "fingerprint_hash": summary.column("fingerprint_hash"),
            "junction_signature": summary.column("junction_signature"),
        },
        schema=READ_FINGERPRINT_TABLE,
    )


__all__ = ["compute_read_fingerprints"]
