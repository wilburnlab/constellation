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

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.schemas.alignment import INTRON_TABLE


# ──────────────────────────────────────────────────────────────────────
# Helpers (shared)
# ──────────────────────────────────────────────────────────────────────


def _strings_to_large_string(table: pa.Table) -> pa.Table:
    """Cast every ``string`` column to ``large_string``.

    Arrow's ``string`` type uses 32-bit offsets, so a single column whose
    total bytes exceed 2 GiB overflows during ``sort_by`` / ``take``
    ("offset overflow while concatenating arrays"). At PromethION scale
    the per-block ``read_id`` / ``ref_name`` columns cross that limit.
    ``large_string`` uses 64-bit offsets.
    """
    for i, field in enumerate(table.schema):
        if pa.types.is_string(field.type):
            table = table.set_column(
                i, field.name, pc.cast(table.column(i), pa.large_string())
            )
    return table


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

    Implementation note: this is the **vectorized** path. The earlier
    Python-row-loop implementation hung for hours at PromethION scale
    (tens of millions of `joined` rows + per-row dict.setdefault calls).
    The vectorized path uses Arrow's ``sort_by`` + slice-shift +
    ``group_by(...).aggregate([("read_id", "count_distinct")])`` to
    compute the cross-read junction-support count in seconds. The
    pre-vectorized implementation is kept as
    :func:`_aggregate_junctions_legacy` for parity testing.
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

    primary = alignments.select(
        ["alignment_id", "read_id", "ref_name", "strand"]
    )
    joined = alignment_blocks.join(primary, keys="alignment_id", join_type="inner")
    if joined.num_rows < 2:
        return INTRON_TABLE.empty_table()

    # At PromethION scale the read_id / ref_name string columns exceed the
    # 2 GiB int32 offset budget; sort_by's internal `take` then raises
    # "offset overflow while concatenating arrays". Cast string columns to
    # large_string (64-bit offsets) before the sort. The output table is
    # rebuilt with plain string types below, so this stays internal.
    joined = _strings_to_large_string(joined)

    # Sort by (alignment_id, block_index) so adjacent rows within each
    # alignment form a donor/acceptor candidate pair via a one-row shift.
    joined = joined.sort_by(
        [("alignment_id", "ascending"), ("block_index", "ascending")]
    )

    n = joined.num_rows
    # Slice columns into "current" (rows 0..n-2) and "next" (rows 1..n-1).
    # Pairs where current.alignment_id == next.alignment_id correspond to
    # adjacent blocks within the same alignment — i.e. one observed
    # splice junction. Pairs across alignment boundaries get masked out.
    aid_curr = joined.column("alignment_id").slice(0, n - 1)
    aid_next = joined.column("alignment_id").slice(1, n - 1)
    same_alignment = pc.equal(aid_curr, aid_next)

    obs = pa.table(
        {
            "alignment_id": aid_curr,
            "read_id": joined.column("read_id").slice(0, n - 1),
            "ref_name": joined.column("ref_name").slice(0, n - 1),
            "strand": joined.column("strand").slice(0, n - 1),
            "donor": joined.column("ref_end").slice(0, n - 1),
            "acceptor": joined.column("ref_start").slice(1, n - 1),
        }
    ).filter(same_alignment)
    if obs.num_rows == 0:
        return INTRON_TABLE.empty_table()

    # Map ref_name → contig_id. The lookup table is small; do it as a
    # Python list-comprehension once across all rows, then drop the
    # unknown-contig ones via an Arrow filter. (Vectorized via Arrow
    # would need a join against contigs; the Python loop is fine since
    # we iterate once across the post-mask junction observations.)
    name_to_id = _contig_id_lookup(genome.contigs)
    contig_id_list = [name_to_id.get(n) for n in obs.column("ref_name").to_pylist()]
    contig_id_arr = pa.array(contig_id_list, type=pa.int64())
    obs = obs.append_column("contig_id", contig_id_arr).filter(
        pc.is_valid(contig_id_arr)
    )
    if obs.num_rows == 0:
        return INTRON_TABLE.empty_table()

    # Group by (contig_id, donor, acceptor, strand) and count distinct
    # read_ids. This is the cross-read aggregation step — replaces the
    # old `counts: dict[..., set[str]]` Python build.
    grouped = obs.group_by(
        ["contig_id", "donor", "acceptor", "strand"]
    ).aggregate([("read_id", "count_distinct")])

    # Sort for deterministic intron_id assignment (matches the legacy
    # path, which sorted on (contig_id, donor, acceptor, strand) via
    # ``sorted(counts.keys())``).
    grouped = grouped.sort_by(
        [
            ("contig_id", "ascending"),
            ("donor", "ascending"),
            ("acceptor", "ascending"),
            ("strand", "ascending"),
        ]
    )

    n_unique = grouped.num_rows
    contig_ids = grouped.column("contig_id").to_pylist()
    donors = grouped.column("donor").to_pylist()
    acceptors = grouped.column("acceptor").to_pylist()
    strands = grouped.column("strand").to_pylist()
    read_counts = grouped.column("read_id_count_distinct").to_pylist()

    # Motif lookup + annotation flag — both per-unique-junction (small N
    # after dedup), so Python loops here are fine. Avoids re-implementing
    # 2 bp genome-sequence lookup in vectorized form.
    motifs = [
        _motif_at(genome, c, d, a)
        for c, d, a in zip(contig_ids, donors, acceptors, strict=True)
    ]

    if annotation is not None:
        annotated_donors, annotated_acceptors = _annotation_splice_sites(
            annotation.features, exon_type=exon_type
        )
        annotated: list[bool | None] = []
        for c, d, a in zip(contig_ids, donors, acceptors, strict=True):
            d_set = annotated_donors.get(c, frozenset())
            a_set = annotated_acceptors.get(c, frozenset())
            annotated.append((d in d_set) and (a in a_set))
    else:
        annotated = [None] * n_unique

    return pa.table(
        {
            "intron_id": pa.array(range(n_unique), type=pa.int64()),
            "contig_id": pa.array(contig_ids, type=pa.int64()),
            "strand": pa.array(strands, type=pa.string()),
            "donor_pos": pa.array(donors, type=pa.int64()),
            "acceptor_pos": pa.array(acceptors, type=pa.int64()),
            "read_count": pa.array(read_counts, type=pa.int64()),
            "motif": pa.array(motifs, type=pa.string()),
            "is_intron_seed": pa.array([True] * n_unique, type=pa.bool_()),
            "annotated": pa.array(annotated, type=pa.bool_()),
        },
        schema=INTRON_TABLE,
    )


def _aggregate_junctions_legacy(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    genome,
    *,
    annotation=None,
    exon_type: str = "exon",
    primary_only: bool = True,
) -> pa.Table:
    """Pre-vectorisation implementation, kept for parity testing.

    Iterates the post-join table row-by-row in Python: builds an
    ``alignment_id → list[(block_index, ref_start, ref_end)]`` dict,
    then a ``(contig, donor, acceptor, strand) → set[read_id]`` dict.
    Correct but unusable at PromethION scale (tens of millions of
    rows in the post-join table → multi-hour hangs). The current
    :func:`aggregate_junctions` is the vectorised replacement; this
    function exists so the parity test can directly compare outputs.
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

    primary = alignments.select(
        ["alignment_id", "read_id", "ref_name", "strand"]
    )
    joined = alignment_blocks.join(primary, keys="alignment_id", join_type="inner")
    if joined.num_rows == 0:
        return INTRON_TABLE.empty_table()

    name_to_id = _contig_id_lookup(genome.contigs)

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
        aid_meta.setdefault(
            int(a),
            {
                "read_id": str(r_id),
                "ref_name": str(ref_n),
                "strand": str(st),
            },
        )

    counts: dict[tuple[int, int, int, str], set[str]] = {}
    for aid, blocks in by_aid.items():
        blocks.sort(key=lambda t: t[0])
        meta = aid_meta[aid]
        contig_id = name_to_id.get(meta["ref_name"])
        if contig_id is None:
            continue
        ends = [b[2] for b in blocks]
        starts = [b[1] for b in blocks]
        for donor, acceptor in zip(ends[:-1], starts[1:]):
            key = (int(contig_id), int(donor), int(acceptor), str(meta["strand"]))
            counts.setdefault(key, set()).add(str(meta["read_id"]))

    if not counts:
        return INTRON_TABLE.empty_table()

    annotated_donors: dict[int, frozenset[int]] = {}
    annotated_acceptors: dict[int, frozenset[int]] = {}
    have_annotation = annotation is not None
    if have_annotation:
        annotated_donors, annotated_acceptors = _annotation_splice_sites(
            annotation.features, exon_type=exon_type
        )

    sorted_keys = sorted(counts.keys())
    rows: list[dict[str, object]] = []
    for iid, key in enumerate(sorted_keys):
        contig_id, donor, acceptor, st = key
        n_reads = len(counts[key])
        motif = _motif_at(genome, contig_id, donor, acceptor)
        if have_annotation:
            d_set = annotated_donors.get(contig_id, frozenset())
            a_set = annotated_acceptors.get(contig_id, frozenset())
            ann = (donor in d_set) and (acceptor in a_set)
        else:
            ann = None
        rows.append(
            {
                "intron_id": int(iid),
                "contig_id": int(contig_id),
                "strand": st,
                "donor_pos": int(donor),
                "acceptor_pos": int(acceptor),
                "read_count": int(n_reads),
                "motif": motif,
                "is_intron_seed": True,
                "annotated": ann,
            }
        )

    return pa.Table.from_pylist(rows, schema=INTRON_TABLE)


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
