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

    # Arrow-native: join blocks × primary on alignment_id (multithreaded
    # C++), attach contig_id via second join with the contigs table, then
    # sort once on (alignment_id, block_index). Junction pairs fall out of
    # a torch boolean mask on consecutive rows.
    primary = alignments.select(
        ["alignment_id", "read_id", "ref_name", "strand"]
    )
    joined = alignment_blocks.join(primary, keys="alignment_id", join_type="inner")
    if joined.num_rows == 0:
        return INTRON_TABLE.empty_table()

    contig_lookup = genome.contigs.select(["contig_id", "name"]).rename_columns(
        ["contig_id", "ref_name"]
    )
    joined = joined.join(contig_lookup, keys="ref_name", join_type="inner")
    if joined.num_rows < 2:
        return INTRON_TABLE.empty_table()

    joined = joined.sort_by(
        [("alignment_id", "ascending"), ("block_index", "ascending")]
    ).combine_chunks()

    # Bridge numeric columns to torch (zero-copy via numpy). After
    # combine_chunks() each column has a single chunk so to_numpy()
    # zero-copies the Arrow buffer into a numpy view; torch.from_numpy
    # then borrows that view. String columns (strand, read_id) stay in
    # Arrow — pc.take on the donor-row slice picks the per-junction
    # values without a Python loop.
    def _col_to_torch(col_name: str) -> torch.Tensor:
        return torch.from_numpy(
            joined.column(col_name).chunk(0).to_numpy(zero_copy_only=True)
        )

    aid_t = _col_to_torch("alignment_id")
    rstart_t = _col_to_torch("ref_start")
    rend_t = _col_to_torch("ref_end")
    contig_t = _col_to_torch("contig_id")

    same_aid = aid_t[1:] == aid_t[:-1]
    n_junctions = int(same_aid.sum().item())
    if n_junctions == 0:
        return INTRON_TABLE.empty_table()

    donor_pos = rend_t[:-1][same_aid].numpy()
    acceptor_pos = rstart_t[1:][same_aid].numpy()
    junction_contig = contig_t[:-1][same_aid].numpy()

    donor_row_idx = torch.nonzero(same_aid, as_tuple=False).squeeze(1)
    donor_row_idx_pa = pa.array(donor_row_idx.numpy())
    n_rows = joined.num_rows
    strand_donor_rows = pc.take(
        joined.column("strand").slice(0, n_rows - 1), donor_row_idx_pa
    )
    read_id_donor_rows = pc.take(
        joined.column("read_id").slice(0, n_rows - 1), donor_row_idx_pa
    )

    junctions = pa.table(
        {
            "contig_id": pa.array(junction_contig),
            "donor_pos": pa.array(donor_pos),
            "acceptor_pos": pa.array(acceptor_pos),
            "strand": strand_donor_rows,
            "read_id": read_id_donor_rows,
        }
    )

    # group_by + count_distinct replaces today's per-junction set[str_read_id].
    aggregated = junctions.group_by(
        ["contig_id", "donor_pos", "acceptor_pos", "strand"]
    ).aggregate([("read_id", "count_distinct")])
    if aggregated.num_rows == 0:
        return INTRON_TABLE.empty_table()

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
