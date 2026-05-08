"""Phase 2 — genome-guided fingerprint clustering.

Consumes ``READ_FINGERPRINT_TABLE`` (re-derived at runtime from
``alignment_blocks/`` + ``introns.parquet``, so the
``--intron-tolerance-bp`` knob is a clustering parameter — sweeping it
re-runs ``cluster_junctions`` against the raw per-position rows
in ``introns.parquet`` and rebuilds fingerprints in one shot), the
trimmed transcript-window sequences from S1 demux, and per-alignment
block summaries; produces the splicing-topology-resolved clusters that
Phase 4 will measure de novo Phase 3 against.

Granularity is the **transcript / isoform** level — finer than gene,
coarser than per-base variant. A gene with N isoforms produces ≥ N
clusters (truncated reads, novel isoforms, and basecaller-junction-
errors can yield more); reads agreeing on canonical-intron-id splicing
topology collapse into one cluster regardless of substitution noise.

Algorithm (roadmap §2.1):

1. Group reads by ``fingerprint_hash`` per (contig, strand).
2. Strand-aware 5'/3' span-coherence drift filter — separates
   alt-TSS / alt-polyA outliers from the within-cluster majority.
   Drift-filtered reads are *kept* in the membership table with
   ``role='drift_filtered'`` so Phase 4 can ask where Phase 3
   landed them.
3. Layer-0 dereplicate trimmed transcript-window sequences within
   each surviving cluster; pick the most-abundant unique sequence's
   read as the representative (tiebreak by mean ``dorado_quality``).
4. Compute per-member fit-to-cluster scores (drift_5p_bp,
   drift_3p_bp, match_rate, indel_rate, n_aligned_bp). Multi-
   parameter — downstream consumers can build any weighted
   goodness-of-fit on top.
5. Optional ``build_consensus``: per-position weighted-PWM majority
   vote on the genome coordinate frame via the shared
   ``align/consensus.py`` primitive.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.align.consensus import build_consensus
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)


_LAYER0_DIGEST_SIZE = 8


def _seq_hash(sequence: str) -> int:
    """blake2b-8 of an ASCII trimmed-window sequence."""
    h = hashlib.blake2b(digest_size=_LAYER0_DIGEST_SIZE)
    h.update(sequence.encode("ascii", errors="replace"))
    return int.from_bytes(h.digest(), "little", signed=False)


def _strand_signed_drift(
    span_start: int,
    span_end: int,
    median_start: float,
    median_end: float,
    strand: str,
) -> tuple[int, int]:
    """Strand-aware (5'_drift_bp, 3'_drift_bp).

    On ``+`` strand: 5' = span_start, 3' = span_end.
    On ``-`` strand: 5' = span_end, 3' = span_start.
    Sign is read - median (positive = read extends further than the
    median in that orientation). On ``?`` strand we treat as ``+``;
    minimap2 in splice-aware mode uses ``+``/``-``, so this only
    matters on assembly-vs-genome runs without splice-site disambiguation.
    """
    if strand == "-":
        drift_5p = int(span_end - median_end)
        drift_3p = int(span_start - median_start)
    else:
        drift_5p = int(span_start - median_start)
        drift_3p = int(span_end - median_end)
    return drift_5p, drift_3p


def _build_alignment_metrics(
    alignment_blocks: pa.Table,
) -> dict[int, dict[str, int]]:
    """Sum n_match / n_mismatch / n_insert / n_delete per alignment_id.

    Returns ``{alignment_id: {'n_match', 'n_mismatch', 'n_insert',
    'n_delete'}}``. ``n_match`` / ``n_mismatch`` may be 0 when only
    CIGAR-derived (cs:long absent) — caller flags ``match_rate`` /
    ``indel_rate`` as null in that case.
    """
    if alignment_blocks.num_rows == 0:
        return {}
    aids = alignment_blocks.column("alignment_id").to_pylist()
    nm = alignment_blocks.column("n_match").to_pylist()
    nx = alignment_blocks.column("n_mismatch").to_pylist()
    ni = alignment_blocks.column("n_insert").to_pylist()
    nd = alignment_blocks.column("n_delete").to_pylist()
    out: dict[int, dict[str, int]] = {}
    for aid, m, x, i, d in zip(aids, nm, nx, ni, nd, strict=True):
        rec = out.setdefault(
            int(aid),
            {"n_match": 0, "n_mismatch": 0, "n_insert": 0, "n_delete": 0,
             "cs_aware": True},
        )
        if m is None or x is None:
            rec["cs_aware"] = False
        else:
            rec["n_match"] += int(m)
            rec["n_mismatch"] += int(x)
        rec["n_insert"] += int(i)
        rec["n_delete"] += int(d)
    return out


def _build_alignment_lookup(alignments: pa.Table) -> dict[str, dict[str, Any]]:
    """``read_id → {alignment_id, ref_start}`` for primary alignments only.

    Phase 2's clustering operates per-read; the (read_id → primary
    alignment_id) lookup keys cluster membership back into the
    alignment_blocks / alignment_cs row space.
    """
    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask).select(
        ["read_id", "alignment_id", "ref_start"]
    )
    return {
        str(rid): {"alignment_id": int(aid), "ref_start": int(rs)}
        for rid, aid, rs in zip(
            primary.column("read_id").to_pylist(),
            primary.column("alignment_id").to_pylist(),
            primary.column("ref_start").to_pylist(),
            strict=True,
        )
    }


def _layer0_derep(
    read_ids: list[str],
    sequences: dict[str, str],
    qualities: dict[str, float],
) -> tuple[str, int, dict[str, int], dict[str, int]]:
    """Pick the representative read by Layer-0 derep.

    Returns ``(rep_read_id, n_unique_sequences, hash_of_read,
    count_per_hash)``. ``hash_of_read`` maps each read_id to its
    sequence hash so the caller can role-tag duplicates. Ties on
    abundance break on mean ``dorado_quality``. If a read has no
    sequence in ``sequences``, it's skipped (the cluster's
    representative comes from the surviving sequenced reads).
    """
    hash_of_read: dict[str, int] = {}
    count_per_hash: dict[int, int] = {}
    quality_sum_per_hash: dict[int, float] = {}
    quality_count_per_hash: dict[int, int] = {}
    exemplar_per_hash: dict[int, str] = {}
    for rid in read_ids:
        seq = sequences.get(rid)
        if seq is None or not seq:
            continue
        h = _seq_hash(seq)
        hash_of_read[rid] = h
        count_per_hash[h] = count_per_hash.get(h, 0) + 1
        q = qualities.get(rid)
        if q is not None:
            quality_sum_per_hash[h] = quality_sum_per_hash.get(h, 0.0) + float(q)
            quality_count_per_hash[h] = quality_count_per_hash.get(h, 0) + 1
        exemplar_per_hash.setdefault(h, rid)
    if not count_per_hash:
        # No sequences available — fall back to first read by id.
        return read_ids[0], 0, hash_of_read, {}
    # Pick rep: max count, tiebreak by mean quality.
    def _score(h: int) -> tuple[int, float]:
        cnt = count_per_hash[h]
        if quality_count_per_hash.get(h, 0) > 0:
            mean_q = quality_sum_per_hash[h] / quality_count_per_hash[h]
        else:
            mean_q = 0.0
        return (cnt, mean_q)
    best_hash = max(count_per_hash.keys(), key=_score)
    rep_read_id = exemplar_per_hash[best_hash]
    n_unique = len(count_per_hash)
    # Map hash → str-keyed count for caller convenience (they don't
    # care about the int hash, just the abundances per role tag).
    count_per_hash_str = {str(h): cnt for h, cnt in count_per_hash.items()}
    return rep_read_id, n_unique, hash_of_read, count_per_hash_str


def _make_membership_row(
    *,
    cluster_id: int,
    read_id: str,
    role: str,
    drift_5p_bp: int | None,
    drift_3p_bp: int | None,
    metrics: dict[str, int] | None,
) -> dict[str, Any]:
    if metrics is None:
        match_rate = None
        indel_rate = None
        n_aligned = 0
    else:
        n_match = int(metrics.get("n_match", 0))
        n_mismatch = int(metrics.get("n_mismatch", 0))
        n_insert = int(metrics.get("n_insert", 0))
        n_delete = int(metrics.get("n_delete", 0))
        n_aligned = n_match + n_mismatch + n_insert + n_delete
        cs_aware = bool(metrics.get("cs_aware", False))
        if cs_aware and (n_match + n_mismatch) > 0:
            match_rate = float(n_match) / float(n_match + n_mismatch)
        else:
            match_rate = None
        if cs_aware and n_aligned > 0:
            indel_rate = float(n_insert + n_delete) / float(n_aligned)
        else:
            indel_rate = None
    return {
        "cluster_id": int(cluster_id),
        "read_id": str(read_id),
        "role": role,
        "drift_5p_bp": (None if drift_5p_bp is None else int(drift_5p_bp)),
        "drift_3p_bp": (None if drift_3p_bp is None else int(drift_3p_bp)),
        "match_rate": match_rate,
        "indel_rate": indel_rate,
        "n_aligned_bp": int(n_aligned),
    }


def cluster_by_fingerprint(
    fingerprints: pa.Table,
    reads: pa.Table,
    *,
    alignments: pa.Table,
    alignment_blocks: pa.Table | None = None,
    alignment_cs: pa.Table | None = None,
    genome: GenomeReference | None = None,
    read_to_sample: Mapping[str, int] | None = None,
    max_5p_drift: int = 25,
    max_3p_drift: int = 75,
    min_cluster_size: int = 1,
    build_consensus_seq: bool = False,
    drop_drift_filtered: bool = False,
    per_sample_clusters: bool = False,
    cluster_id_seed: int = 0,
) -> tuple[pa.Table, pa.Table]:
    """Group reads into clusters by quantised splicing-topology fingerprint.

    Parameters
    ----------
    fingerprints : pa.Table
        ``READ_FINGERPRINT_TABLE``-shaped. Already derived against the
        clustering-time ``INTRON_TABLE`` (which carries the per-row
        ``intron_id`` cluster assignments computed at the chosen
        ``intron_tolerance_bp``).
    reads : pa.Table
        Must carry columns ``read_id``, ``sequence`` (the **trimmed
        transcript window**, NOT the raw read), and optionally
        ``dorado_quality`` (mean phred over the trimmed window;
        Layer-0 tiebreak).
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies primary-alignment lookup
        from ``read_id`` to ``(alignment_id, ref_start)``.
    alignment_blocks : pa.Table, optional
        ``ALIGNMENT_BLOCK_TABLE``-shaped. When provided, populates
        ``match_rate`` / ``indel_rate`` / ``n_aligned_bp`` on
        cluster_membership rows. Required when
        ``build_consensus_seq=True``.
    alignment_cs : pa.Table, optional
        ``ALIGNMENT_CS_TABLE``-shaped. Required when
        ``build_consensus_seq=True``.
    genome : GenomeReference, optional
        Required when ``build_consensus_seq=True``; supplies the
        reference window for PWM accumulation.
    read_to_sample : Mapping[str, int], optional
        ``read_id → sample_id`` for ``per_sample_clusters`` mode.
        Reads without a sample mapping fall into the "unassigned"
        partition (``sample_id`` column null in the output).
    max_5p_drift, max_3p_drift : int
        Strand-aware bp tolerance for the span-coherence filter.
        Defaults 25 / 75 mirror the lab observations on TSS / polyA
        scatter.
    min_cluster_size : int, default 1
        Drop clusters whose surviving (post-drift-filter) member
        count is below this. Singletons are kept by default.
    build_consensus_seq : bool, default False
        Populate ``consensus_sequence`` per cluster via the shared
        weighted-PWM kernel. Requires ``alignment_blocks``,
        ``alignment_cs``, and ``genome``.
    drop_drift_filtered : bool, default False
        When True, drift-filtered reads are excluded from
        ``cluster_membership``. Default keeps them with
        ``role='drift_filtered'``.
    per_sample_clusters : bool, default False
        When True, partition by sample_id in addition to (contig,
        strand). Default treats clusters as spanning samples.
    cluster_id_seed : int, default 0
        Starting cluster_id. Caller packs ``(worker_idx << 32)`` here
        when running parallel partitions.

    Returns
    -------
    (clusters, membership) : (pa.Table, pa.Table)
        Schemas: ``TRANSCRIPT_CLUSTER_TABLE``, ``CLUSTER_MEMBERSHIP_TABLE``.
    """
    if build_consensus_seq:
        if alignment_blocks is None or alignment_cs is None or genome is None:
            raise ValueError(
                "build_consensus_seq=True requires alignment_blocks, "
                "alignment_cs, and genome"
            )
    if fingerprints.num_rows == 0:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        )

    # ── Precompute lookups ──────────────────────────────────────────
    seq_lookup: dict[str, str] = {
        str(rid): str(seq)
        for rid, seq in zip(
            reads.column("read_id").to_pylist(),
            reads.column("sequence").to_pylist(),
            strict=True,
        )
    }
    quality_lookup: dict[str, float] = {}
    if "dorado_quality" in reads.schema.names:
        for rid, q in zip(
            reads.column("read_id").to_pylist(),
            reads.column("dorado_quality").to_pylist(),
            strict=True,
        ):
            if q is not None:
                quality_lookup[str(rid)] = float(q)
    primary_lookup = _build_alignment_lookup(alignments)
    if alignment_blocks is not None:
        metrics_by_aid = _build_alignment_metrics(alignment_blocks)
    else:
        metrics_by_aid = {}

    # Build (alignment_id → contig sequence) lazy cache for consensus.
    contig_id_to_name: dict[int, str] = {}
    if genome is not None:
        for cid, name in zip(
            genome.contigs.column("contig_id").to_pylist(),
            genome.contigs.column("name").to_pylist(),
            strict=True,
        ):
            contig_id_to_name[int(cid)] = str(name)

    # ── Partition fingerprints ──────────────────────────────────────
    # Iterate as Python rows once; partitioning logic is small enough
    # not to need pa.compute group_by gymnastics, and the per-group
    # work is row-shaped anyway.
    rows = fingerprints.to_pylist()
    partitions: dict[
        tuple[int, str, int | None],
        list[dict[str, Any]],
    ] = {}
    for r in rows:
        sample_id = None
        if per_sample_clusters and read_to_sample is not None:
            sample_id = read_to_sample.get(str(r["read_id"]))
        key = (int(r["contig_id"]), str(r["strand"]), sample_id)
        partitions.setdefault(key, []).append(r)

    # ── Cluster each partition ──────────────────────────────────────
    cluster_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    next_cluster_id = int(cluster_id_seed)

    for (contig_id, strand, sample_id), partition_rows in partitions.items():
        # Group by fingerprint_hash inside the partition.
        groups: dict[int, list[dict[str, Any]]] = {}
        for r in partition_rows:
            groups.setdefault(int(r["fingerprint_hash"]), []).append(r)

        for fp_hash, members in groups.items():
            n_total = len(members)
            # Drift filter (singletons skip — no median to compare).
            if n_total > 1:
                med_start = statistics.median(
                    int(m["span_start"]) for m in members
                )
                med_end = statistics.median(
                    int(m["span_end"]) for m in members
                )
                kept: list[dict[str, Any]] = []
                drifted: list[dict[str, Any]] = []
                for m in members:
                    drift_5p, drift_3p = _strand_signed_drift(
                        int(m["span_start"]),
                        int(m["span_end"]),
                        med_start,
                        med_end,
                        strand,
                    )
                    m["_drift_5p"] = drift_5p
                    m["_drift_3p"] = drift_3p
                    if (abs(drift_5p) > max_5p_drift
                            or abs(drift_3p) > max_3p_drift):
                        drifted.append(m)
                    else:
                        kept.append(m)
            else:
                kept = members
                drifted = []
                for m in members:
                    m["_drift_5p"] = None
                    m["_drift_3p"] = None

            if len(kept) < min_cluster_size:
                # Whole cluster (including drift_filtered) drops out.
                continue

            # Layer-0 derep + representative selection.
            kept_ids = [str(m["read_id"]) for m in kept]
            rep_id, n_unique, hash_of_read, _counts = _layer0_derep(
                kept_ids, seq_lookup, quality_lookup
            )

            cluster_id = next_cluster_id
            next_cluster_id += 1

            # Optional consensus.
            consensus_seq: str | None = None
            if build_consensus_seq:
                # Use the cluster's median span window for the PWM.
                kept_starts = [int(m["span_start"]) for m in kept]
                kept_ends = [int(m["span_end"]) for m in kept]
                window_start = min(kept_starts)
                window_end = max(kept_ends)
                contig_name = contig_id_to_name.get(int(contig_id))
                if contig_name is not None and genome is not None:
                    contig_seq = genome.sequence_of(int(contig_id))
                    window_seq = contig_seq[window_start:window_end]
                    member_aids: list[int] = []
                    member_weights: list[float] = []
                    member_ref_starts: list[int] = []
                    for m in kept:
                        rid = str(m["read_id"])
                        info = primary_lookup.get(rid)
                        if info is None:
                            continue
                        member_aids.append(int(info["alignment_id"]))
                        # Weight = Layer-0 abundance = 1 per read for
                        # genome-anchored consensus (consistent with the
                        # most-abundant-unique-sequence representative
                        # already chosen). Phase 3 plugs in de novo
                        # abundance here.
                        member_weights.append(1.0)
                        member_ref_starts.append(int(info["ref_start"]))
                    if member_aids and alignment_cs is not None:
                        consensus_seq = build_consensus(
                            member_alignment_ids=member_aids,
                            member_weights=member_weights,
                            member_ref_starts=member_ref_starts,
                            alignment_cs=alignment_cs,
                            reference_sequence=window_seq,
                            reference_start=window_start,
                        )

            # Cluster row.
            kept_starts = [int(m["span_start"]) for m in kept]
            kept_ends = [int(m["span_end"]) for m in kept]
            cluster_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "representative_read_id": str(rep_id),
                    "n_reads": int(len(kept)),
                    "identity_threshold": None,
                    "consensus_sequence": consensus_seq,
                    "predicted_protein": None,
                    "orf_start": None,
                    "orf_end": None,
                    "orf_strand": None,
                    "codon_table": None,
                    "mode": "genome-guided",
                    "contig_id": int(contig_id),
                    "strand": str(strand),
                    "span_start": int(min(kept_starts)),
                    "span_end": int(max(kept_ends)),
                    "fingerprint_hash": int(fp_hash),
                    "n_unique_sequences": int(n_unique) if n_unique > 0 else 1,
                    "sample_id": (None if sample_id is None else int(sample_id)),
                }
            )

            # Membership rows — kept members.
            rep_seq_hash = hash_of_read.get(rep_id)
            for m in kept:
                rid = str(m["read_id"])
                role = "member"
                if rid == rep_id:
                    role = "representative"
                elif rep_seq_hash is not None and hash_of_read.get(rid) == rep_seq_hash:
                    role = "duplicate"
                info = primary_lookup.get(rid)
                metrics = (
                    metrics_by_aid.get(int(info["alignment_id"]))
                    if info is not None
                    else None
                )
                drift_5p = m.get("_drift_5p") if role == "member" else None
                drift_3p = m.get("_drift_3p") if role == "member" else None
                membership_rows.append(
                    _make_membership_row(
                        cluster_id=cluster_id,
                        read_id=rid,
                        role=role,
                        drift_5p_bp=drift_5p,
                        drift_3p_bp=drift_3p,
                        metrics=metrics,
                    )
                )

            # Membership rows — drift-filtered.
            if not drop_drift_filtered:
                for m in drifted:
                    rid = str(m["read_id"])
                    info = primary_lookup.get(rid)
                    metrics = (
                        metrics_by_aid.get(int(info["alignment_id"]))
                        if info is not None
                        else None
                    )
                    membership_rows.append(
                        _make_membership_row(
                            cluster_id=cluster_id,
                            read_id=rid,
                            role="drift_filtered",
                            drift_5p_bp=m.get("_drift_5p"),
                            drift_3p_bp=m.get("_drift_3p"),
                            metrics=metrics,
                        )
                    )

    if not cluster_rows:
        return (
            TRANSCRIPT_CLUSTER_TABLE.empty_table(),
            CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        )
    clusters = pa.Table.from_pylist(cluster_rows, schema=TRANSCRIPT_CLUSTER_TABLE)
    membership = pa.Table.from_pylist(
        membership_rows, schema=CLUSTER_MEMBERSHIP_TABLE
    )
    return clusters, membership


__all__ = ["cluster_by_fingerprint"]
