"""Splice-junction aggregator.

Reduces ``ALIGNMENT_BLOCK_TABLE`` to ``SPLICE_JUNCTION_TABLE`` —
one row per distinct ``(contig_id, donor_pos, acceptor_pos, strand)``
tuple observed across the input. ``donor_pos`` is the previous block's
``ref_end`` (intron 5'); ``acceptor_pos`` is the next block's
``ref_start`` (intron 3'). Both are 0-based half-open coordinates,
matching the conventions in :mod:`constellation.sequencing.schemas`.

``motif`` is the strand-naive +-strand dinucleotide pair
``"<donor[0:2]>-<acceptor[-2:]>"`` (e.g. ``"GT-AG"``,
``"CT-AC"``, ``"AT-AC"``, ``"GC-AG"``, ``"other"``). Strand-aware
canonical-motif checks live downstream of this function — we keep the
raw dinucleotide pair so basecaller-drift / non-canonical-splicing
diagnostics remain visible (per the CLAUDE.md scoring-domain rule:
preserve observable artefacts; don't collapse them at the
aggregator).

``annotated`` is populated when an :class:`Annotation` is provided —
a junction is flagged True when both ``donor_pos`` and ``acceptor_pos``
appear in the annotation's exon-end / exon-start sets on the same
contig (the "both splice sites are known" definition; matches
STAR's ``SJ.out.tab`` ``annotated`` column semantics). When no
Annotation is passed (e.g. against a freshly-assembled genome), the
column is left null.

Why aggregate introns rather than exons: see the rationale in the
plan file at ``~/.claude/plans/in-our-last-few-agile-nygaard.md`` §1.2,
or in the docstring of ``SPLICE_JUNCTION_TABLE`` itself.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.schemas.alignment import SPLICE_JUNCTION_TABLE


# ──────────────────────────────────────────────────────────────────────
# Helpers
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
    """Build (donors, acceptors) sets per ``contig_id`` from exons.

    Donors = exon ``end`` (intron starts after the exon ends).
    Acceptors = exon ``start`` (intron ends where the next exon begins).
    Both are 0-based half-open positions, matching the conventions
    used elsewhere in this codebase.
    """
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
    """Strand-naive +-strand dinucleotide pair.

    Out-of-bounds reads (e.g. junction near contig start/end where
    we'd index off the contig) return ``"other"`` so we never raise
    on legitimate-but-rare alignments.
    """
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
# Public API
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
    """Build ``SPLICE_JUNCTION_TABLE`` from per-read alignment blocks.

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
        Read 2 bp at donor / acceptor for the ``motif`` column. Same
        object the rest of the resolve stage already has in scope.
    annotation : Annotation | None
        When provided, populates the ``annotated`` column. ``gene_id``
        stays null in v1 — multi-gene junctions are real (overlapping
        loci) and a single ``gene_id`` would lie; lift this when
        downstream callers actually need it.
    exon_type : str, default "exon"
        Feature type that encodes annotated exons.
    primary_only : bool, default True
        Drop secondary / supplementary alignments before aggregation.
        Off lets you measure the rare-junction tail across all
        chimeric / multi-mapper variants — diagnostic-only mode.

    Returns
    -------
    pa.Table conforming to ``SPLICE_JUNCTION_TABLE``. ``junction_id``
    is assigned sequentially in the order
    ``(contig_id, donor_pos, acceptor_pos, strand)`` is encountered.
    """
    if alignment_blocks.num_rows == 0 or alignments.num_rows == 0:
        return SPLICE_JUNCTION_TABLE.empty_table()

    if primary_only:
        primary_mask = pc.and_(
            pc.invert(alignments.column("is_secondary")),
            pc.invert(alignments.column("is_supplementary")),
        )
        alignments = alignments.filter(primary_mask)
        if alignments.num_rows == 0:
            return SPLICE_JUNCTION_TABLE.empty_table()

    primary = alignments.select(
        ["alignment_id", "read_id", "ref_name", "strand"]
    )
    joined = alignment_blocks.join(primary, keys="alignment_id", join_type="inner")
    if joined.num_rows == 0:
        return SPLICE_JUNCTION_TABLE.empty_table()

    name_to_id = _contig_id_lookup(genome.contigs)

    # Group blocks by alignment_id; build per-alignment junction list
    # ordered by block_index. We collect (contig_id, donor, acceptor,
    # strand, read_id) per junction so the cross-read aggregation can
    # count distinct read_ids per (contig, donor, acceptor, strand).
    by_aid: dict[int, list[tuple[int, int]]] = {}
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

    # Aggregate: (contig_id, donor, acceptor, strand) → set of read_ids.
    # Using a set of read_ids (vs a count) makes the "distinct reads
    # supporting this junction" semantics exact when an alignment has
    # the same junction recorded twice (shouldn't happen, but cheap).
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
        return SPLICE_JUNCTION_TABLE.empty_table()

    annotated_donors: dict[int, frozenset[int]] = {}
    annotated_acceptors: dict[int, frozenset[int]] = {}
    have_annotation = annotation is not None
    if have_annotation:
        annotated_donors, annotated_acceptors = _annotation_splice_sites(
            annotation.features, exon_type=exon_type
        )

    # Stable ordering: sort by (contig_id, donor_pos, acceptor_pos, strand).
    sorted_keys = sorted(counts.keys())
    rows: list[dict[str, object]] = []
    for jid, key in enumerate(sorted_keys):
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
                "junction_id": int(jid),
                "contig_id": int(contig_id),
                "donor_pos": int(donor),
                "acceptor_pos": int(acceptor),
                "strand": st,
                "read_count": int(n_reads),
                "motif": motif,
                "annotated": ann,
                "gene_id": None,
            }
        )

    return pa.Table.from_pylist(rows, schema=SPLICE_JUNCTION_TABLE)


__all__ = ["aggregate_junctions"]
