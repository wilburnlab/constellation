"""Splice-junction aggregation + intron clustering.

Two-step reduction from per-read alignment blocks to canonical introns:

    aggregate_junctions   ALIGNMENT_BLOCK_TABLE  →  INTRON_TABLE (raw form)
                          One row per distinct (contig, donor, acceptor,
                          strand) tuple observed across the input. Each
                          row is its own singleton cluster — ``intron_id``
                          is assigned sequentially and ``is_intron_seed``
                          is True for every row. ``read_count`` is the
                          number of distinct reads supporting that exact
                          position pair.

    cluster_junctions     INTRON_TABLE (raw)  →  INTRON_TABLE (clustered)
                          Greedy support-ranked clustering with an
                          ``±tolerance_bp`` absorption window. Re-assigns
                          ``intron_id`` and ``is_intron_seed`` so multiple
                          rows can share an ``intron_id`` (the cluster).
                          Highest-``read_count`` member wins as the seed,
                          with GT-AG > GC-AG > AT-AC > other motif tiebreak.

Both functions return the same ``INTRON_TABLE`` schema. The pre-cluster
form has the property ``n_distinct_intron_ids == n_rows``; the post-cluster
form has ``n_distinct_intron_ids ≤ n_rows`` and exactly one
``is_intron_seed=True`` row per ``intron_id``.

Why two steps? The CLI default flow runs both: ``aggregate_junctions``
to count exact-position support, then ``cluster_junctions`` to consolidate
basecaller-driven single-bp jitter into the same canonical cluster while
preserving real alt-splicing topology at coarser scales. The
``--no-cluster-junctions`` opt-out skips the second step (equivalent to
``tolerance_bp=0``); each observed position pair stays its own cluster.

``motif`` is the strand-naive +-strand dinucleotide pair
``"<donor[0:2]>-<acceptor[-2:]>"`` (e.g. ``"GT-AG"``, ``"GC-AG"``,
``"AT-AC"``, ``"other"``). ``annotated`` is True iff this exact
``(donor_pos, acceptor_pos)`` pair appears as an annotated intron — both
splice sites must match annotated exon ends/starts on the same contig
(STAR's ``SJ.out.tab`` ``annotated`` column semantics).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from constellation.sequencing.schemas.alignment import INTRON_TABLE


# ──────────────────────────────────────────────────────────────────────
# Helpers (shared)
# ──────────────────────────────────────────────────────────────────────


def _contig_id_lookup(contigs: pa.Table) -> dict[str, int]:
    """``CONTIG_TABLE.name`` → ``contig_id``."""
    return {
        str(name): int(cid)
        for name, cid in zip(
            contigs.column("name").to_pylist(),
            contigs.column("contig_id").to_pylist(),
            strict=True,
        )
    }


def _annotation_splice_sites(
    annotation_features: pa.Table,
    *,
    exon_type: str = "exon",
) -> tuple[dict[int, frozenset[int]], dict[int, frozenset[int]]]:
    """Build (donors, acceptors) sets per ``contig_id`` from exons."""
    if annotation_features.num_rows == 0:
        return {}, {}
    type_mask = pc.equal(annotation_features.column("type"), exon_type)
    exons = annotation_features.filter(type_mask)
    if exons.num_rows == 0:
        return {}, {}

    contig_ids = exons.column("contig_id").to_pylist()
    starts = exons.column("start").to_pylist()
    ends = exons.column("end").to_pylist()
    donors_by_contig: dict[int, set[int]] = {}
    acceptors_by_contig: dict[int, set[int]] = {}
    for cid, s, e in zip(contig_ids, starts, ends, strict=True):
        donors_by_contig.setdefault(int(cid), set()).add(int(e))
        acceptors_by_contig.setdefault(int(cid), set()).add(int(s))
    return (
        {k: frozenset(v) for k, v in donors_by_contig.items()},
        {k: frozenset(v) for k, v in acceptors_by_contig.items()},
    )


def _motif_at(
    genome,
    contig_id: int,
    donor_pos: int,
    acceptor_pos: int,
) -> str:
    """Strand-naive +-strand dinucleotide pair."""
    try:
        seq = genome.sequence_of(int(contig_id))
    except (KeyError, ValueError, IndexError):
        return "other"
    n = len(seq)
    if donor_pos < 0 or donor_pos + 2 > n:
        return "other"
    if acceptor_pos - 2 < 0 or acceptor_pos > n:
        return "other"
    donor = seq[donor_pos : donor_pos + 2].upper()
    acceptor = seq[acceptor_pos - 2 : acceptor_pos].upper()
    return f"{donor}-{acceptor}"


# ──────────────────────────────────────────────────────────────────────
# Step 1 — raw aggregation
# ──────────────────────────────────────────────────────────────────────


def aggregate_junctions(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    genome,
    *,
    annotation=None,
    exon_type: str = "exon",
    primary_only: bool = True,
) -> pa.Table:
    """Reduce per-read alignment blocks to one row per observed splice
    position pair.

    The output is ``INTRON_TABLE``-shaped. Each row represents a distinct
    exact ``(contig_id, donor_pos, acceptor_pos, strand)`` tuple with the
    number of supporting reads. ``intron_id`` is assigned sequentially
    (each row is its own singleton cluster pending downstream
    ``cluster_junctions``); ``is_intron_seed`` is True on every row.

    Parameters
    ----------
    alignment_blocks : pa.Table
        ``ALIGNMENT_BLOCK_TABLE``-shaped. Junctions are derived from
        adjacent block pairs within each ``alignment_id``.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies ``read_id``, ``ref_name``,
        ``strand``, plus the secondary/supplementary flags used when
        ``primary_only=True``.
    genome : GenomeReference
        Read 2 bp at donor / acceptor for the ``motif`` column.
    annotation : Annotation | None
        When provided, populates the ``annotated`` column.
    exon_type : str, default "exon"
        Feature type that encodes annotated exons.
    primary_only : bool, default True
        Drop secondary / supplementary alignments before aggregation.

    Returns
    -------
    pa.Table conforming to ``INTRON_TABLE``. Rows are sorted by
    ``(contig_id, donor_pos, acceptor_pos, strand)`` for determinism;
    ``intron_id`` matches the row index.
    """
    if alignment_blocks.num_rows == 0 or alignments.num_rows == 0:
        return INTRON_TABLE.empty_table()

    if primary_only:
        primary_mask = pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
        alignments = alignments.filter(primary_mask)
        if alignments.num_rows == 0:
            return INTRON_TABLE.empty_table()

    # Arrow-native strategy — at mouse / PromethION scale the 1.6B-row
    # alignment_blocks table is the dominant working set. We keep
    # `read_id` (a ~30-byte string column) OUT of the heavy sort by
    # deferring the primary-table enrichment until AFTER junction
    # detection: sort blocks-numeric-only, find junctions in torch,
    # then attach `read_id` / `ref_name` / `strand` to the much smaller
    # junctions table via two small Arrow joins. Counting supporting
    # reads per junction becomes `count(alignment_id)` since alignment_id
    # is 1:1 with read_id for primary alignments.
    primary_min = alignments.select(["alignment_id", "ref_name", "strand"])

    # Filter blocks to those belonging to primary alignments. Using
    # `pc.is_in` as a semi-join keeps the blocks table as a single
    # input to the sort instead of carrying primary's columns along.
    primary_aids = primary_min.column("alignment_id")
    blocks_min = alignment_blocks.select(
        ["alignment_id", "block_index", "ref_start", "ref_end"]
    ).filter(pc.is_in(alignment_blocks.column("alignment_id"), primary_aids))
    if blocks_min.num_rows < 2:
        return INTRON_TABLE.empty_table()

    # Sort blocks-only (4 int64/int32 columns, no strings). This is the
    # operation the wide-table predecessor pinned on at mouse scale —
    # the prior version sorted 14 columns including a ~50 GB `read_id`
    # string column and then `combine_chunks`'d the entire result,
    # whose offsets-rewrite is partially single-threaded.
    blocks_sorted = blocks_min.sort_by(
        [("alignment_id", "ascending"), ("block_index", "ascending")]
    ).combine_chunks()

    def _col_to_torch(col_name: str) -> torch.Tensor:
        return torch.from_numpy(
            blocks_sorted.column(col_name).chunk(0).to_numpy(zero_copy_only=True)
        )

    aid_t = _col_to_torch("alignment_id")
    rstart_t = _col_to_torch("ref_start")
    rend_t = _col_to_torch("ref_end")

    same_aid = aid_t[1:] == aid_t[:-1]
    if not bool(same_aid.any().item()):
        return INTRON_TABLE.empty_table()

    # Per-junction: alignment_id from donor row (== acceptor row's
    # alignment_id, by same_aid mask), donor_pos = donor row's ref_end,
    # acceptor_pos = next row's ref_start.
    junction_aid = aid_t[:-1][same_aid].numpy()
    donor_pos = rend_t[:-1][same_aid].numpy()
    acceptor_pos = rstart_t[1:][same_aid].numpy()

    junctions = pa.table(
        {
            "alignment_id": pa.array(junction_aid, type=pa.int64()),
            "donor_pos": pa.array(donor_pos, type=pa.int64()),
            "acceptor_pos": pa.array(acceptor_pos, type=pa.int64()),
        }
    )

    # Attach ref_name + strand from primary; contig_id from contigs.
    # Both joins are now on the (smaller) junctions table — `ref_name`
    # and `strand` strings land here, not on the 1.6B-row blocks table.
    junctions = junctions.join(
        primary_min, keys="alignment_id", join_type="inner"
    )
    contig_lookup = genome.contigs.select(["contig_id", "name"]).rename_columns(
        ["contig_id", "ref_name"]
    )
    junctions = junctions.join(
        contig_lookup, keys="ref_name", join_type="inner"
    )
    if junctions.num_rows == 0:
        return INTRON_TABLE.empty_table()

    # Each row of `junctions` is a unique (alignment_id, donor, acceptor)
    # tuple by construction (one junction per consecutive same-aid block
    # pair in the sorted table). For primary alignments alignment_id is
    # 1:1 with read_id, so `count(alignment_id)` per (contig, donor,
    # acceptor, strand) equals `count_distinct(read_id)` — a plain
    # `count` aggregation is much cheaper than `count_distinct`.
    aggregated = junctions.group_by(
        ["contig_id", "donor_pos", "acceptor_pos", "strand"]
    ).aggregate([("alignment_id", "count")])
    if aggregated.num_rows == 0:
        return INTRON_TABLE.empty_table()
    aggregated = aggregated.rename_columns(
        [c if c != "alignment_id_count" else "read_id_count_distinct"
         for c in aggregated.column_names]
    )

    # Sort to match today's deterministic output order:
    # (contig_id, donor_pos, acceptor_pos, strand) ascending.
    aggregated = aggregated.sort_by(
        [
            ("contig_id", "ascending"),
            ("donor_pos", "ascending"),
            ("acceptor_pos", "ascending"),
            ("strand", "ascending"),
        ]
    )

    n_unique = aggregated.num_rows
    contig_ids_out = aggregated.column("contig_id").to_pylist()
    donors_out = aggregated.column("donor_pos").to_pylist()
    acceptors_out = aggregated.column("acceptor_pos").to_pylist()

    # motif lookup is bounded by unique-junction count, not by block count
    # — a Python loop here is fine at mouse scale (tens of thousands of
    # junctions max).
    motifs = [
        _motif_at(genome, int(cid), int(d), int(a))
        for cid, d, a in zip(contig_ids_out, donors_out, acceptors_out, strict=True)
    ]

    have_annotation = annotation is not None
    if have_annotation:
        annotated_donors, annotated_acceptors = _annotation_splice_sites(
            annotation.features, exon_type=exon_type
        )
        annotateds: list[bool | None] = []
        for cid, d, a in zip(contig_ids_out, donors_out, acceptors_out, strict=True):
            d_set = annotated_donors.get(int(cid), frozenset())
            a_set = annotated_acceptors.get(int(cid), frozenset())
            annotateds.append((int(d) in d_set) and (int(a) in a_set))
    else:
        annotateds = [None] * n_unique

    return pa.table(
        {
            "intron_id": pa.array(np.arange(n_unique, dtype=np.int64)),
            "contig_id": aggregated.column("contig_id"),
            "strand": aggregated.column("strand"),
            "donor_pos": aggregated.column("donor_pos"),
            "acceptor_pos": aggregated.column("acceptor_pos"),
            "read_count": aggregated.column("read_id_count_distinct").cast(
                pa.int64()
            ),
            "motif": pa.array(motifs, type=pa.string()),
            "is_intron_seed": pa.array([True] * n_unique, type=pa.bool_()),
            "annotated": pa.array(annotateds, type=pa.bool_()),
        },
        schema=INTRON_TABLE,
    )


# ──────────────────────────────────────────────────────────────────────
# Step 2 — greedy support-ranked clustering
# ──────────────────────────────────────────────────────────────────────


# Fallback motif rank for any motif not in the explicit priority list.
_FALLBACK_MOTIF_RANK: int = 999


def _seed_donor_key(seed: tuple[int, int, int]) -> int:
    """Extract donor_pos from a (donor, acceptor, intron_id) tuple."""
    return seed[0]


def _motif_rank_map(motif_priority: Sequence[str]) -> dict[str, int]:
    """Build a motif → rank lookup. Earlier entries win on ties."""
    return {str(m): i for i, m in enumerate(motif_priority)}


def cluster_junctions(
    introns: pa.Table,
    *,
    tolerance_bp: int = 5,
    motif_priority: Sequence[str] = ("GT-AG", "GC-AG", "AT-AC"),
) -> pa.Table:
    """Greedy support-ranked clustering of observed junctions.

    Re-assigns ``intron_id`` and ``is_intron_seed`` so multiple input rows
    sharing nearby splice positions on the same (contig, strand) collapse
    into one cluster. The highest-``read_count`` member wins as the
    cluster seed, with the supplied ``motif_priority`` (defaulting to
    GT-AG > GC-AG > AT-AC > other) as tiebreak. Subsequent members
    absorb if both ``|donor - seed_donor| ≤ tolerance_bp`` AND
    ``|acceptor - seed_acceptor| ≤ tolerance_bp`` (L∞ ball, not L₂).

    Parameters
    ----------
    introns : pa.Table
        ``INTRON_TABLE``-shaped — typically the output of
        :func:`aggregate_junctions`. The ``intron_id`` and
        ``is_intron_seed`` columns are *recomputed*; existing values
        are ignored.
    tolerance_bp : int, default 5
        ``±W`` absorption window per axis. Set to 0 to disable
        clustering — each row stays its own singleton cluster.
    motif_priority : sequence of str, default ("GT-AG", "GC-AG", "AT-AC")
        Motif tiebreak order. Earlier entries win on ``read_count``
        ties. Motifs absent from this list rank below all listed ones.

    Returns
    -------
    pa.Table conforming to ``INTRON_TABLE``. Same row count and
    per-row ``donor_pos`` / ``acceptor_pos`` / ``read_count`` / ``motif``
    / ``annotated`` as the input; ``intron_id`` may now repeat across
    rows, and exactly one row per distinct ``intron_id`` has
    ``is_intron_seed = True``. Output rows are sorted by
    ``(intron_id, is_intron_seed DESC, donor_pos, acceptor_pos)`` —
    seed first, members after, deterministic.
    """
    if tolerance_bp < 0:
        raise ValueError(
            f"tolerance_bp must be >= 0, got {tolerance_bp}"
        )
    if introns.num_rows == 0:
        return INTRON_TABLE.empty_table()

    motif_rank = _motif_rank_map(motif_priority)

    contig_ids = introns.column("contig_id").to_pylist()
    strands = introns.column("strand").to_pylist()
    donors = introns.column("donor_pos").to_pylist()
    acceptors = introns.column("acceptor_pos").to_pylist()
    read_counts = introns.column("read_count").to_pylist()
    motifs = introns.column("motif").to_pylist()
    annotated = introns.column("annotated").to_pylist()

    n = introns.num_rows

    # Build sort order: (read_count desc, motif_rank asc, contig asc,
    # donor asc, acceptor asc, strand asc) for deterministic seeding.
    indices = list(range(n))
    indices.sort(
        key=lambda i: (
            -int(read_counts[i]),
            motif_rank.get(str(motifs[i]) if motifs[i] is not None else "",
                           _FALLBACK_MOTIF_RANK),
            int(contig_ids[i]),
            int(donors[i]),
            int(acceptors[i]),
            str(strands[i]),
        )
    )

    # Per-(contig, strand) seed lists. Each entry stores
    # (donor_pos, acceptor_pos, intron_id) sorted by donor_pos so we
    # can bisect for window candidates.
    seeds_by_partition: dict[
        tuple[int, str], list[tuple[int, int, int]]
    ] = {}

    # Output: intron_id assignment + is_intron_seed flag per input row.
    out_intron_id: list[int] = [0] * n
    out_is_seed: list[bool] = [False] * n
    next_intron_id = 0
    W = int(tolerance_bp)

    for src_i in indices:
        partition = (int(contig_ids[src_i]), str(strands[src_i]))
        d = int(donors[src_i])
        a = int(acceptors[src_i])

        seed_list = seeds_by_partition.get(partition)
        match_id: int | None = None
        if seed_list:
            # bisect for seeds with donor in [d-W, d+W]. ``key=`` lets
            # bisect index the (donor, acceptor, id) tuples directly
            # without rebuilding a parallel donor-keys list.
            lo = bisect.bisect_left(seed_list, d - W, key=_seed_donor_key)
            hi = bisect.bisect_right(seed_list, d + W, key=_seed_donor_key)
            for k in range(lo, hi):
                seed_d, seed_a, seed_id = seed_list[k]
                if abs(seed_a - a) <= W:
                    match_id = seed_id
                    break

        if match_id is None:
            # New seed — sorted insertion into partition's seed list.
            new_id = next_intron_id
            next_intron_id += 1
            out_intron_id[src_i] = new_id
            out_is_seed[src_i] = True
            if seed_list is None:
                seed_list = []
                seeds_by_partition[partition] = seed_list
            bisect.insort(seed_list, (d, a, new_id), key=_seed_donor_key)
        else:
            out_intron_id[src_i] = match_id
            out_is_seed[src_i] = False

    # Build the output table preserving the input row count + per-row
    # donor / acceptor / etc; only intron_id + is_intron_seed change.
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "intron_id": int(out_intron_id[i]),
                "contig_id": int(contig_ids[i]),
                "strand": str(strands[i]),
                "donor_pos": int(donors[i]),
                "acceptor_pos": int(acceptors[i]),
                "read_count": int(read_counts[i]),
                "motif": (None if motifs[i] is None else str(motifs[i])),
                "is_intron_seed": bool(out_is_seed[i]),
                "annotated": annotated[i],
            }
        )

    out = pa.Table.from_pylist(rows, schema=INTRON_TABLE)
    # Final stable sort: (intron_id asc, is_intron_seed desc, donor asc,
    # acceptor asc) — seed first within each cluster, deterministic.
    sort_keys = [
        ("intron_id", "ascending"),
        ("is_intron_seed", "descending"),
        ("donor_pos", "ascending"),
        ("acceptor_pos", "ascending"),
    ]
    return out.sort_by(sort_keys)


__all__ = ["aggregate_junctions", "cluster_junctions"]
