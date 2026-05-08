"""Data-informed exon discovery + gene rollup from clustered introns.

Given the canonical ``INTRON_TABLE`` (post-:func:`cluster_junctions`)
and the per-position ``COVERAGE_TABLE``, derive an ``Annotation`` whose
``FEATURE_TABLE`` contains data-supported gene + exon features. Both
inputs originate from the always-on resolve-stage outputs of
``transcriptome align``; the function chain is:

    aggregate_junctions  →  cluster_junctions  →  INTRON_TABLE
                                                          \\
    build_pileup  →  COVERAGE_TABLE                        \\
                            \\                              \\
                             ▼                               ▼
                              build_derived_annotation(...)
                                              │
                                              ▼
                                      (Annotation,
                                       BLOCK_EXON_ASSIGNMENT_TABLE,
                                       EXON_PSI_TABLE)

Granularity contract:
- Exons are minimal segments — each cut introduced by a trusted intron
  donor or acceptor inside a covered region produces a new exon
  segment. Reads agreeing on splice topology share segment membership.
- Boundary calls are threshold-driven: a position is a boundary iff
  it appears as donor_pos or acceptor_pos of an intron whose total
  cluster ``read_count`` (summed across all member position pairs)
  meets ``min_intron_read_count``.
- Gene rollup is connected-components over exons, where edges follow
  trusted-intron splice junctions (donor==exon_a.end AND
  acceptor==exon_b.start, same contig + strand).
- Coverage thresholds the candidate exon body but does NOT contribute
  to boundary determination — basecaller noise on coverage doesn't
  introduce false boundaries.

v1 limitations (deferred to v2):
- Pure single-exon genes (no trusted introns) are not discovered.
  Multi-exon genes with at least one trusted intron pull in their
  flanking exons as expected.
- Coverage is unstranded (build_pileup does not stratify by alignment
  strand); per-strand exon derivation uses unstranded coverage as a
  depth-thresholding substrate. Loci with reads on both strands at
  similar depth could over-emit. v2 lifts this with stranded pile-up.
- Differential exon-abundance (negbin LRT across samples) is deferred;
  v1 emits the per-(exon, sample) inclusion / exclusion / PSI substrate.
"""

from __future__ import annotations

from typing import Mapping

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.graph.network import Network
from constellation.sequencing.schemas.alignment import (
    BLOCK_EXON_ASSIGNMENT_TABLE,
)
from constellation.sequencing.schemas.quant import EXON_PSI_TABLE
from constellation.sequencing.schemas.reference import FEATURE_TABLE


# Sentinel sample_id when no read_to_sample mapping is supplied or a
# read has no resolved sample. Matches COVERAGE_TABLE convention.
_UNSTRATIFIED_SAMPLE_ID: int = -1


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _trusted_seed_rows(
    introns: pa.Table, *, min_intron_read_count: int
) -> pa.Table:
    """Return seed rows of intron clusters whose total read_count
    meets the threshold.

    Filters in two steps:
      1. ``group_by(intron_id).sum(read_count)`` → trusted intron_ids.
      2. ``filter(is_intron_seed=True AND intron_id IN trusted_ids)``.
    """
    if introns.num_rows == 0:
        return introns.schema.empty_table()
    totals = introns.group_by("intron_id").aggregate(
        [("read_count", "sum")]
    )
    trusted_ids_arr = totals.filter(
        pc.greater_equal(
            totals.column("read_count_sum"),
            int(min_intron_read_count),
        )
    ).column("intron_id")
    seeds = introns.filter(pc.equal(introns.column("is_intron_seed"), True))
    return seeds.filter(pc.is_in(seeds.column("intron_id"), trusted_ids_arr))


def _coalesce_covered_intervals(
    coverage: pa.Table, *, min_exon_depth: int
) -> dict[int, list[tuple[int, int]]]:
    """Per-contig list of [start, end) intervals with depth >= threshold.

    Coalesces abutting / overlapping intervals across sample partitions —
    a region covered at depth N in sample A and depth M in sample B
    both pass the threshold and merge into one covered span.
    """
    if coverage.num_rows == 0:
        return {}
    filt = coverage.filter(
        pc.greater_equal(coverage.column("depth"), int(min_exon_depth))
    )
    if filt.num_rows == 0:
        return {}
    contig_ids = filt.column("contig_id").to_pylist()
    starts = filt.column("start").to_pylist()
    ends = filt.column("end").to_pylist()
    by_contig: dict[int, list[tuple[int, int]]] = {}
    for c, s, e in zip(contig_ids, starts, ends, strict=True):
        by_contig.setdefault(int(c), []).append((int(s), int(e)))
    coalesced: dict[int, list[tuple[int, int]]] = {}
    for c, ivs in by_contig.items():
        ivs.sort()
        merged: list[tuple[int, int]] = []
        cur_start, cur_end = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_end:  # abutting or overlapping
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        coalesced[c] = merged
    return coalesced


def _empty_features_table() -> pa.Table:
    return FEATURE_TABLE.empty_table()


def _empty_block_assignments() -> pa.Table:
    return BLOCK_EXON_ASSIGNMENT_TABLE.empty_table()


def _empty_exon_psi() -> pa.Table:
    return EXON_PSI_TABLE.empty_table()


# ──────────────────────────────────────────────────────────────────────
# derive_exons
# ──────────────────────────────────────────────────────────────────────


def derive_exons(
    coverage: pa.Table,
    introns: pa.Table,
    contigs: pa.Table,
    *,
    min_exon_depth: int = 5,
    min_intron_read_count: int = 3,
) -> pa.Table:
    """Build a FEATURE_TABLE-shaped ``type='exon'`` table from coverage
    + clustered introns.

    Parameters
    ----------
    coverage : pa.Table
        ``COVERAGE_TABLE``-shaped — RLE depth per (contig, sample).
    introns : pa.Table
        ``INTRON_TABLE``-shaped — clustered intron site evidence.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — accepted as a placeholder for v1.5
        per-strand validation; not used in v1.
    min_exon_depth : int, default 5
        Minimum coverage depth (per RLE row) required for a region
        to be considered as part of a candidate exon.
    min_intron_read_count : int, default 3
        Minimum total reads supporting an intron cluster (summed across
        cluster member positions) to treat it as a trusted boundary.

    Returns
    -------
    pa.Table conforming to ``FEATURE_TABLE`` with one row per derived
    exon (``type='exon'``). ``feature_id`` is assigned sequentially
    starting from 0; ``parent_id`` is null and gets populated by
    :func:`roll_up_genes`. ``source='constellation_derived'``.
    """
    del contigs  # reserved for v1.5 stranded coverage validation

    if introns.num_rows == 0 or coverage.num_rows == 0:
        return _empty_features_table()

    trusted_seeds = _trusted_seed_rows(
        introns, min_intron_read_count=min_intron_read_count
    )
    if trusted_seeds.num_rows == 0:
        # No trusted introns → no boundaries → no exons in v1.
        return _empty_features_table()

    coalesced = _coalesce_covered_intervals(
        coverage, min_exon_depth=min_exon_depth
    )
    if not coalesced:
        return _empty_features_table()

    # Group trusted introns by (contig_id, strand). Keep both the set
    # of (donor, acceptor) tuples (for the "drop pieces == intron"
    # rule) and the set of cut positions (donors ∪ acceptors).
    intron_pairs_by_partition: dict[tuple[int, str], set[tuple[int, int]]] = {}
    cuts_by_partition: dict[tuple[int, str], set[int]] = {}
    for r in trusted_seeds.to_pylist():
        key = (int(r["contig_id"]), str(r["strand"]))
        d = int(r["donor_pos"])
        a = int(r["acceptor_pos"])
        intron_pairs_by_partition.setdefault(key, set()).add((d, a))
        cuts = cuts_by_partition.setdefault(key, set())
        cuts.add(d)
        cuts.add(a)

    exon_records: list[dict] = []
    next_feature_id = 0

    # Iterate per (contig, strand). For each covered interval on the
    # contig, segment using cuts of that strand's intron set; drop
    # pieces whose [start, end) exactly matches a trusted intron's span.
    for (contig_id, strand) in sorted(intron_pairs_by_partition.keys()):
        intron_pairs = intron_pairs_by_partition[(contig_id, strand)]
        cut_positions = cuts_by_partition[(contig_id, strand)]
        intervals = coalesced.get(contig_id, [])
        for iv_start, iv_end in intervals:
            interior_cuts = sorted(
                p for p in cut_positions if iv_start < p < iv_end
            )
            edges = [iv_start] + interior_cuts + [iv_end]
            for i in range(len(edges) - 1):
                p_start = edges[i]
                p_end = edges[i + 1]
                if p_end <= p_start:
                    continue
                if (p_start, p_end) in intron_pairs:
                    # This piece exactly matches a trusted intron span
                    # — drop. Pieces inside larger introns (e.g. nested
                    # alt-3'SS subspans) are NOT dropped; they may be
                    # exons in some other isoform.
                    continue
                exon_records.append(
                    {
                        "feature_id": next_feature_id,
                        "contig_id": int(contig_id),
                        "start": int(p_start),
                        "end": int(p_end),
                        "strand": str(strand),
                        "type": "exon",
                        "name": None,
                        "parent_id": None,  # set by roll_up_genes
                        "source": "constellation_derived",
                        "score": None,
                        "phase": None,
                        "attributes_json": None,
                    }
                )
                next_feature_id += 1

    if not exon_records:
        return _empty_features_table()
    return pa.Table.from_pylist(exon_records, schema=FEATURE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# roll_up_genes
# ──────────────────────────────────────────────────────────────────────


def roll_up_genes(
    exons: pa.Table,
    introns: pa.Table,
    *,
    min_intron_read_count: int = 3,
) -> pa.Table:
    """Combine exons + connected-component-derived genes into a single
    FEATURE_TABLE.

    Parameters
    ----------
    exons : pa.Table
        ``FEATURE_TABLE``-shaped, ``type='exon'`` only — the output of
        :func:`derive_exons`.
    introns : pa.Table
        ``INTRON_TABLE``-shaped — used to build splice-junction edges
        between exons. Only trusted intron seeds (above
        ``min_intron_read_count``) contribute edges.
    min_intron_read_count : int, default 3
        Same threshold as :func:`derive_exons`.

    Returns
    -------
    pa.Table conforming to ``FEATURE_TABLE`` with ``type='gene'`` rows
    appended to the input exons. Each exon's ``parent_id`` is rewritten
    to point at its gene's ``feature_id``. Gene ``feature_id``s are
    assigned starting after the highest exon ``feature_id``.
    """
    if exons.num_rows == 0:
        return _empty_features_table()

    trusted_seeds = _trusted_seed_rows(
        introns, min_intron_read_count=min_intron_read_count
    )

    # Index exons by (contig, strand, end) for donor matching and by
    # (contig, strand, start) for acceptor matching.
    exon_meta: dict[int, dict] = {
        int(r["feature_id"]): r for r in exons.to_pylist()
    }
    exon_by_end: dict[tuple[int, str, int], list[int]] = {}
    exon_by_start: dict[tuple[int, str, int], list[int]] = {}
    for fid, r in exon_meta.items():
        cid = int(r["contig_id"])
        strand = str(r["strand"])
        exon_by_end.setdefault((cid, strand, int(r["end"])), []).append(fid)
        exon_by_start.setdefault((cid, strand, int(r["start"])), []).append(fid)

    edge_pairs: list[tuple[int, int]] = []
    for r in trusted_seeds.to_pylist():
        cid = int(r["contig_id"])
        strand = str(r["strand"])
        d = int(r["donor_pos"])
        a = int(r["acceptor_pos"])
        srcs = exon_by_end.get((cid, strand, d), [])
        dsts = exon_by_start.get((cid, strand, a), [])
        for src in srcs:
            for dst in dsts:
                if src != dst:
                    edge_pairs.append((src, dst))

    nodes_table = pa.table(
        {"id": pa.array(list(exon_meta.keys()), type=pa.int64())}
    )
    edges_table = pa.table(
        {
            "src": pa.array(
                [int(e[0]) for e in edge_pairs], type=pa.int64()
            ),
            "dst": pa.array(
                [int(e[1]) for e in edge_pairs], type=pa.int64()
            ),
        }
    )
    net = Network(nodes_table, edges_table, directed=False)
    components = net.connected_components()

    max_exon_id = (
        max(exon_meta.keys()) if exon_meta else -1
    )
    next_gene_id = max_exon_id + 1

    gene_records: list[dict] = []
    exon_to_gene: dict[int, int] = {}
    for component in components:
        gene_id = next_gene_id
        next_gene_id += 1
        first_exon = exon_meta[int(component[0])]
        contig_id = int(first_exon["contig_id"])
        strand = str(first_exon["strand"])
        starts = [int(exon_meta[int(e)]["start"]) for e in component]
        ends = [int(exon_meta[int(e)]["end"]) for e in component]
        gene_records.append(
            {
                "feature_id": int(gene_id),
                "contig_id": contig_id,
                "start": int(min(starts)),
                "end": int(max(ends)),
                "strand": strand,
                "type": "gene",
                "name": None,
                "parent_id": None,
                "source": "constellation_derived",
                "score": None,
                "phase": None,
                "attributes_json": None,
            }
        )
        for e in component:
            exon_to_gene[int(e)] = int(gene_id)

    # Update exon parent_ids and emit combined table.
    new_exon_records: list[dict] = []
    for r in exons.to_pylist():
        new_r = dict(r)
        new_r["parent_id"] = exon_to_gene.get(int(r["feature_id"]))
        new_exon_records.append(new_r)

    combined = gene_records + new_exon_records
    combined.sort(key=lambda r: int(r["feature_id"]))
    return pa.Table.from_pylist(combined, schema=FEATURE_TABLE)


# ──────────────────────────────────────────────────────────────────────
# assign_blocks_to_exons
# ──────────────────────────────────────────────────────────────────────


def assign_blocks_to_exons(
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    derived_features: pa.Table,
    contigs: pa.Table,
) -> pa.Table:
    """Build ``BLOCK_EXON_ASSIGNMENT_TABLE`` — long-form M:N edge table
    between alignment blocks and derived exons.

    Parameters
    ----------
    alignment_blocks : pa.Table
        ``ALIGNMENT_BLOCK_TABLE``-shaped.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped — supplies ``ref_name`` for the
        contig lookup. Secondary / supplementary alignments are
        excluded so each block contributes at most once per exon.
    derived_features : pa.Table
        ``FEATURE_TABLE``-shaped output of :func:`roll_up_genes`. Only
        ``type='exon'`` rows are used.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — supplies the ``ref_name → contig_id``
        resolution.

    Returns
    -------
    pa.Table conforming to ``BLOCK_EXON_ASSIGNMENT_TABLE``. One row
    per (alignment_id, block_index, data_exon_id) overlap edge with
    ``overlap_bp > 0``.
    """
    if alignment_blocks.num_rows == 0 or derived_features.num_rows == 0:
        return _empty_block_assignments()

    exons = derived_features.filter(
        pc.equal(derived_features.column("type"), "exon")
    )
    if exons.num_rows == 0:
        return _empty_block_assignments()

    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask).select(
        ["alignment_id", "ref_name"]
    )

    name_to_cid: dict[str, int] = {
        str(n): int(c)
        for n, c in zip(
            contigs.column("name").to_pylist(),
            contigs.column("contig_id").to_pylist(),
            strict=True,
        )
    }
    aid_to_cid: dict[int, int] = {}
    for aid, ref_n in zip(
        primary.column("alignment_id").to_pylist(),
        primary.column("ref_name").to_pylist(),
        strict=True,
    ):
        cid = name_to_cid.get(str(ref_n))
        if cid is not None:
            aid_to_cid[int(aid)] = cid

    exons_by_contig: dict[int, list[tuple[int, int, int]]] = {}
    for r in exons.to_pylist():
        c = int(r["contig_id"])
        exons_by_contig.setdefault(c, []).append(
            (int(r["start"]), int(r["end"]), int(r["feature_id"]))
        )
    for c in exons_by_contig:
        exons_by_contig[c].sort()

    rows: list[dict] = []
    aids = alignment_blocks.column("alignment_id").to_pylist()
    bidxs = alignment_blocks.column("block_index").to_pylist()
    bstarts = alignment_blocks.column("ref_start").to_pylist()
    bends = alignment_blocks.column("ref_end").to_pylist()

    for aid, bidx, bs, be in zip(aids, bidxs, bstarts, bends, strict=True):
        contig_id = aid_to_cid.get(int(aid))
        if contig_id is None:
            continue
        exon_list = exons_by_contig.get(contig_id, [])
        if not exon_list:
            continue
        bs_int = int(bs)
        be_int = int(be)
        block_len = be_int - bs_int
        if block_len <= 0:
            continue
        # Linear scan from leftmost candidate. Exons sorted by start;
        # we early-exit when an exon's start >= block's end.
        for e_start, e_end, e_id in exon_list:
            if e_start >= be_int:
                break
            if e_end <= bs_int:
                continue
            ovl_start = max(e_start, bs_int)
            ovl_end = min(e_end, be_int)
            ovl = ovl_end - ovl_start
            if ovl <= 0:
                continue
            exon_len = e_end - e_start
            rows.append(
                {
                    "alignment_id": int(aid),
                    "block_index": int(bidx),
                    "data_exon_id": int(e_id),
                    "overlap_bp": int(ovl),
                    "block_fraction": (
                        float(ovl) / float(block_len) if block_len > 0 else 0.0
                    ),
                    "exon_fraction": (
                        float(ovl) / float(exon_len) if exon_len > 0 else 0.0
                    ),
                }
            )

    if not rows:
        return _empty_block_assignments()
    return pa.Table.from_pylist(rows, schema=BLOCK_EXON_ASSIGNMENT_TABLE)


# ──────────────────────────────────────────────────────────────────────
# compute_exon_psi
# ──────────────────────────────────────────────────────────────────────


def compute_exon_psi(
    block_assignments: pa.Table,
    alignments: pa.Table,
    derived_features: pa.Table,
    contigs: pa.Table,
    read_to_sample: Mapping[str, int] | None = None,
) -> pa.Table:
    """Per-(derived_exon, sample) inclusion / exclusion / PSI substrate.

    Definitions:

    - **Gene-spanning alignment**: ``alignment.ref_start <= gene.start AND
      alignment.ref_end >= gene.end`` on the same contig.
    - **Inclusion**: gene-spanning alignment with at least one block
      overlapping the exon (a row in ``block_assignments`` with
      ``overlap_bp > 0`` for ``(alignment_id, exon_id)``).
    - **Exclusion**: gene-spanning alignment with NO block overlapping
      the exon. By construction such an alignment must have spliced
      past the exon (a single un-spliced block spanning the gene
      would intersect every exon).
    - **Ambiguous**: alignment that overlaps the gene but doesn't
      fully span it — diagnostic only; not in the PSI ratio.

    PSI = ``n_inclusion_reads / (n_inclusion_reads + n_exclusion_reads)``.

    Parameters
    ----------
    block_assignments : pa.Table
        ``BLOCK_EXON_ASSIGNMENT_TABLE``-shaped — output of
        :func:`assign_blocks_to_exons`.
    alignments : pa.Table
        ``ALIGNMENT_TABLE``-shaped.
    derived_features : pa.Table
        ``FEATURE_TABLE``-shaped output of :func:`roll_up_genes` —
        contains both gene + exon rows.
    contigs : pa.Table
        ``CONTIG_TABLE``-shaped — supplies the ``ref_name → contig_id``
        resolution.
    read_to_sample : Mapping[str, int] | None
        Optional ``read_id → sample_id``. When omitted, every alignment
        contributes to a single ``sample_id = -1`` partition (matches
        ``COVERAGE_TABLE``'s unstratified sentinel).

    Returns
    -------
    pa.Table conforming to ``EXON_PSI_TABLE``. One row per
    ``(data_exon_id, sample_id)`` pair with at least one
    gene-spanning or partial-spanning alignment.
    """
    exon_filter = pc.equal(derived_features.column("type"), "exon")
    exons = derived_features.filter(exon_filter)
    gene_filter = pc.equal(derived_features.column("type"), "gene")
    genes = derived_features.filter(gene_filter)
    if exons.num_rows == 0 or genes.num_rows == 0:
        return _empty_exon_psi()

    gene_meta: dict[int, dict] = {
        int(r["feature_id"]): {
            "contig_id": int(r["contig_id"]),
            "start": int(r["start"]),
            "end": int(r["end"]),
        }
        for r in genes.to_pylist()
    }
    exon_meta: dict[int, dict] = {
        int(r["feature_id"]): {
            "gene_id": (
                int(r["parent_id"]) if r["parent_id"] is not None else None
            ),
            "contig_id": int(r["contig_id"]),
            "start": int(r["start"]),
            "end": int(r["end"]),
        }
        for r in exons.to_pylist()
    }

    exons_by_gene: dict[int, list[int]] = {}
    for eid, m in exon_meta.items():
        gid = m.get("gene_id")
        if gid is not None:
            exons_by_gene.setdefault(gid, []).append(eid)

    name_to_cid: dict[str, int] = {
        str(n): int(c)
        for n, c in zip(
            contigs.column("name").to_pylist(),
            contigs.column("contig_id").to_pylist(),
            strict=True,
        )
    }

    primary_mask = pc.and_(
        pc.invert(alignments.column("is_secondary")),
        pc.invert(alignments.column("is_supplementary")),
    )
    primary = alignments.filter(primary_mask)

    # alignment_id → metadata.
    aid_meta: dict[int, dict] = {}
    for r in primary.to_pylist():
        cid = name_to_cid.get(str(r["ref_name"]))
        if cid is None:
            continue
        aid_meta[int(r["alignment_id"])] = {
            "contig_id": cid,
            "ref_start": int(r["ref_start"]),
            "ref_end": int(r["ref_end"]),
            "read_id": str(r["read_id"]),
        }

    # alignment_id → set of exon_ids it overlaps (any block, any overlap_bp > 0).
    aid_exon_set: dict[int, set[int]] = {}
    if block_assignments.num_rows > 0:
        ba_aids = block_assignments.column("alignment_id").to_pylist()
        ba_eids = block_assignments.column("data_exon_id").to_pylist()
        ba_ovls = block_assignments.column("overlap_bp").to_pylist()
        for aid, eid, ovl in zip(ba_aids, ba_eids, ba_ovls, strict=True):
            if int(ovl) > 0:
                aid_exon_set.setdefault(int(aid), set()).add(int(eid))

    # alignments grouped by contig for gene-spanning check.
    aids_by_contig: dict[int, list[tuple[int, int, int]]] = {}
    for aid, m in aid_meta.items():
        aids_by_contig.setdefault(m["contig_id"], []).append(
            (m["ref_start"], m["ref_end"], aid)
        )

    if read_to_sample is None:
        read_to_sample = {}

    counts: dict[tuple[int, int], dict[str, int]] = {}

    for gene_id, exon_ids in exons_by_gene.items():
        gene = gene_meta[gene_id]
        gene_contig = gene["contig_id"]
        gene_start = gene["start"]
        gene_end = gene["end"]
        contig_aids = aids_by_contig.get(gene_contig, [])
        spanning_aids: list[int] = []
        partial_aids: list[int] = []
        for aref_start, aref_end, aid in contig_aids:
            if aref_start <= gene_start and aref_end >= gene_end:
                spanning_aids.append(aid)
            elif aref_end > gene_start and aref_start < gene_end:
                partial_aids.append(aid)

        for aid in spanning_aids:
            read_id = aid_meta[aid]["read_id"]
            sample_id = int(
                read_to_sample.get(read_id, _UNSTRATIFIED_SAMPLE_ID)
            )
            overlapping_exons = aid_exon_set.get(aid, set())
            for eid in exon_ids:
                key = (eid, sample_id)
                rec = counts.setdefault(key, {"inc": 0, "exc": 0, "amb": 0})
                if eid in overlapping_exons:
                    rec["inc"] += 1
                else:
                    rec["exc"] += 1
        for aid in partial_aids:
            read_id = aid_meta[aid]["read_id"]
            sample_id = int(
                read_to_sample.get(read_id, _UNSTRATIFIED_SAMPLE_ID)
            )
            for eid in exon_ids:
                key = (eid, sample_id)
                rec = counts.setdefault(key, {"inc": 0, "exc": 0, "amb": 0})
                rec["amb"] += 1

    if not counts:
        return _empty_exon_psi()

    rows: list[dict] = []
    for (eid, sid), rec in counts.items():
        inc = rec["inc"]
        exc = rec["exc"]
        denom = inc + exc
        psi = (float(inc) / float(denom)) if denom > 0 else None
        rows.append(
            {
                "data_exon_id": int(eid),
                "sample_id": int(sid),
                "n_inclusion_reads": int(inc),
                "n_exclusion_reads": int(exc),
                "n_ambiguous_reads": int(rec["amb"]),
                "psi": psi,
            }
        )
    # Sort for determinism: (data_exon_id asc, sample_id asc).
    rows.sort(key=lambda r: (r["data_exon_id"], r["sample_id"]))
    return pa.Table.from_pylist(rows, schema=EXON_PSI_TABLE)


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────


def build_derived_annotation(
    coverage: pa.Table,
    introns: pa.Table,
    alignment_blocks: pa.Table,
    alignments: pa.Table,
    contigs: pa.Table,
    *,
    read_to_sample: Mapping[str, int] | None = None,
    min_exon_depth: int = 5,
    min_intron_read_count: int = 3,
):
    """Resolve-stage entry point — chains :func:`derive_exons` →
    :func:`roll_up_genes` → :func:`assign_blocks_to_exons` →
    :func:`compute_exon_psi`.

    Returns
    -------
    (annotation, block_assignments, exon_psi)
        - ``annotation`` : :class:`Annotation` with FEATURE_TABLE
          containing gene + exon rows. Empty when no trusted introns
          are present.
        - ``block_assignments`` : ``BLOCK_EXON_ASSIGNMENT_TABLE``-shaped.
        - ``exon_psi`` : ``EXON_PSI_TABLE``-shaped.
    """
    # Local import to avoid circular import at module load time.
    from constellation.sequencing.annotation.annotation import Annotation

    metadata = {
        "min_exon_depth": int(min_exon_depth),
        "min_intron_read_count": int(min_intron_read_count),
        "source": "constellation_derived",
    }

    exons = derive_exons(
        coverage, introns, contigs,
        min_exon_depth=min_exon_depth,
        min_intron_read_count=min_intron_read_count,
    )
    if exons.num_rows == 0:
        empty_annotation = Annotation(
            features=_empty_features_table(),
            metadata_extras=metadata,
        )
        return (
            empty_annotation,
            _empty_block_assignments(),
            _empty_exon_psi(),
        )

    full_features = roll_up_genes(
        exons, introns, min_intron_read_count=min_intron_read_count
    )
    block_assignments = assign_blocks_to_exons(
        alignment_blocks, alignments, full_features, contigs
    )
    exon_psi = compute_exon_psi(
        block_assignments, alignments, full_features, contigs,
        read_to_sample=read_to_sample,
    )
    annotation = Annotation(
        features=full_features,
        metadata_extras=metadata,
    )
    return annotation, block_assignments, exon_psi


__all__ = [
    "derive_exons",
    "roll_up_genes",
    "assign_blocks_to_exons",
    "compute_exon_psi",
    "build_derived_annotation",
]
