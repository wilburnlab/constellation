"""Stage 3 — M-step: consensus, haplotypes, and the ORF support gate.

For one template and the reads the E-step assigned to it:

1. **Pooled folded consensus** over every assigned read. Round 1 needs no
   alignment work at all — minimap2's own ``cg:Z`` is already a read→template
   alignment in exactly the convention ``project_member`` wants (query = read,
   ``I`` consumes the read, ``D`` the template), so it feeds straight in with
   ``frame_start = t_start`` and ``member_start = q_start``, and the
   soft-clipped read flanks become the end-extension candidates. Only the
   post-splice passes call edlib.
2. **Variants**, unchanged. The existing null is already homopolymer-run-length
   aware, which is what a 5-G run with a 22% indel rate needs.
3. **Haplotypes** from the FDR-supported, in-core, base-substitution columns.
4. **Per-haplotype folded consensus**, then the ORF re-predicted under a
   support gate.

The order matters and is not obvious: the pooled PWM has to come first,
because in round 1 there are no declared columns to group by — the columns
come out of the variant caller.

**The support gate.** A template's flanks are one read's sequence and carry no
certificate, so an ORF must not silently extend through them. Certification is
evaluated on *consensus* columns (where gap-winning columns do not exist,
which resolves an ambiguity in the spec): a column is certified when its base
coverage and its agreement both clear thresholds. An ORF reaching past the
seed ORF's boundary is truncated at the last certified column and flagged.
This is only meaningful *because* the consensus kernel can now extend past the
frame at all — before that, an ORF could never reach uncertified ground.

**The minor-haplotype floor.** The spec says "≥3 reads **and** ≥1%", but 1% of
a template with 20,000 assigned reads is 200, which discards exactly the
~5-read minority proteoform the whole design exists to preserve. The haplotype
columns are already FDR-gated by the context-conditional null, so the fraction
floor is redundant at low depth and harmful at high depth: it defaults to 0.0
here, and ``--overdispersion`` is the right instrument at high depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    ConsensusResult,
    MemberSpec,
    frame_consensus,
    member_alleles,
)
from constellation.sequencing.transcriptome.cluster.denovo.haplotypes import (
    build_haplotypes,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import (
    ORF_CODON_TABLE,
    best_sense_orf,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (
    ErrorModel,
    call_variants,
)


@dataclass(slots=True)
class RefinedTemplate:
    """One node: a haplotype's consensus, its ORF, and its read support."""

    parent_template_id: int
    haplotype_id: int
    consensus: str
    n_reads: int
    node_weight: float
    protein: str | None = None
    orf_start: int = -1
    orf_end: int = -1
    orf_certified_end: int = -1
    orf_is_truncated_by_support: bool = False
    allele_string: str = ""
    declared_variants: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    n_inserted_columns: int = 0
    n_extended_5p: int = 0
    n_extended_3p: int = 0
    variants: list[tuple] = field(default_factory=list, repr=False)


def certified_columns(
    cres: ConsensusResult, *, min_depth: float, min_agreement: float
) -> np.ndarray:
    """Per **consensus** position: is this column supported by real reads?

    ``base_cov`` counts A/C/G/T votes only. At a ragged end many members vote
    a deletion, so total coverage stays flat there while base coverage ramps
    down — which is exactly the distinction the gate needs.
    """
    f = cres.frame_of_cons
    if f is None or f.size == 0:
        return np.zeros(0, dtype=bool)
    rows = cres.pwm[f]
    base_cov = rows[:, :4].sum(axis=1)
    total = rows.sum(axis=1)
    agreement = rows.max(axis=1) / np.maximum(total, 1.0)
    return (base_cov >= min_depth) & (agreement >= min_agreement)


def gated_orf(
    consensus: str,
    certified: np.ndarray,
    *,
    seed_orf_end: int,
    min_aa_length: int = 30,
):
    """Predict the ORF, truncating it where read support runs out.

    ``seed_orf_end`` is the previous round's ORF end in *consensus*
    coordinates; anything at or before it is already certified by
    construction. Returns ``(protein, start, end, certified_end, truncated)``
    or ``None``.
    """
    hit = best_sense_orf(consensus, min_aa_length=min_aa_length)
    if hit is None:
        return None
    prot, st, en = hit
    if en <= max(seed_orf_end, st) or certified.size == 0:
        return prot, st, en, en, False

    # Walk forward from wherever the certificate ends and find the first
    # uncertified column the ORF would have to pass through.
    lo = min(max(seed_orf_end, st), len(consensus))
    limit = lo
    while limit < min(en, certified.size) and certified[limit]:
        limit += 1
    if limit >= en:
        return prot, st, en, en, False

    # Report the ORF as ending at the last certified column, on a codon
    # boundary. It no longer ends in a stop — that is the point of the flag.
    trunc_end = st + 3 * ((limit - st) // 3)
    if trunc_end - st < min_aa_length * 3:
        return None
    from constellation.core.sequence.nucleic import translate

    prot_t = translate(
        consensus[st:trunc_end], codon_table=ORF_CODON_TABLE, partial="discard"
    )
    return prot_t, st, trunc_end, limit, True


def specs_from_assignments(
    rows,
    read_seq: dict[str, str],
) -> list[MemberSpec]:
    """Build PWM members straight from the E-step's minimap2 alignments.

    ``rows`` are dicts with ``read_id, cigar, t_start, q_start, weight``. The
    orientation is fixed: minimap2 emits query→target, so the template is the
    reference and ``centroid_is_query`` is False.
    """
    out: list[MemberSpec] = []
    for i, r in enumerate(rows):
        cig = r.get("cigar")
        if not cig:
            continue
        out.append(
            MemberSpec(
                member_seq=read_seq[r["read_id"]],
                weight=float(r.get("weight", 1.0)),
                cigar=cig,
                centroid_is_query=False,
                ref_start=int(r["t_start"]),
                member_start=int(r["q_start"]),
                member_id=i,
            )
        )
    return out


def _haplotype_columns(
    vrows: list[tuple], *, min_indel_reads: float = 3.0, max_columns: int = 64
) -> list[int]:
    """Indices of the variants allowed to define haplotypes.

    ``in_core`` always applies — the ragged 5'/3' coverage ramps are length
    classes, not alleles. On top of that a column is admitted when either:

    * ``call == 'real'`` — the FDR gate against the context-conditional null,
      which stops one error read fragmenting a clean template; or
    * it is an **indel** carrying at least ``min_indel_reads``.

    That second clause is a deliberate departure from the components path,
    which admits only base-substitution minor alleles. An in-core indel
    changes the *reading frame*, so its consequence is categorical rather
    than quantitative: a 1-nt deletion in a 5-G run three codons before the
    stop bypasses the stop and reads a further ~60 residues out of the 3' UTR.

    Under the **default** error model the clause is close to a no-op — the
    prior homopolymer rate is ~1%, so a 20% deletion at depth 125 clears FDR
    on its own (measured p ≈ 1e-24). It earns its place under
    ``--error-model empirical``, where ε is refit from the data: the real
    5-G run carries a ~22% indel rate, a fitted null therefore *expects* the
    deletion, and the readthrough proteoform would be dropped as error. That
    is the case ledger #22 names — the node must be kept and flagged, not
    believed or deleted. The read floor keeps stray 1-2 read indels out.
    """
    idx = []
    for i, vr in enumerate(vrows):
        if not vr[11]:  # in_core
            continue
        is_indel = vr[2] == "-"
        if vr[10] == "real" or (is_indel and vr[6] >= min_indel_reads):
            idx.append(i)
    idx.sort(key=lambda i: vrows[i][8])  # most significant first
    return sorted(idx[:max_columns])


def refine_template(
    frame: str,
    members: list[MemberSpec],
    *,
    template_id: int = 0,
    seed_orf_end: int = 0,
    error_model: ErrorModel | None = None,
    overdispersion: float = 0.0,
    min_aa_length: int = 30,
    min_haplotype_reads: int = 3,
    min_haplotype_frac: float = 0.0,
    support_min_depth: float = 3.0,
    support_min_agreement: float = 0.6,
    max_haplotype_columns: int = 64,
    **kernel_kwargs,
) -> list[RefinedTemplate]:
    """Refine one template into its haplotype nodes."""
    if not members:
        return []
    model = error_model or ErrorModel()
    pooled = frame_consensus(frame, members, frame_weight=0.0, **kernel_kwargs)
    total_reads = float(sum(m.weight for m in members))

    vrows = call_variants(pooled, model=model, overdispersion=overdispersion)
    sel = (
        _haplotype_columns(
            vrows,
            min_indel_reads=min_haplotype_reads,
            max_columns=max_haplotype_columns,
        )
        if vrows
        else []
    )

    def _node(
        hid: int, hap_members: list[MemberSpec], allele: str, cols: np.ndarray
    ) -> RefinedTemplate | None:
        cres = (
            pooled
            if hid == 0 and len(hap_members) == len(members)
            else frame_consensus(frame, hap_members, frame_weight=0.0, **kernel_kwargs)
        )
        certified = certified_columns(
            cres, min_depth=support_min_depth, min_agreement=support_min_agreement
        )
        node = RefinedTemplate(
            parent_template_id=template_id,
            haplotype_id=hid,
            consensus=cres.consensus,
            n_reads=int(round(sum(m.weight for m in hap_members))),
            node_weight=float(sum(m.weight for m in hap_members)),
            allele_string=allele,
            declared_variants=cols,
            n_inserted_columns=cres.n_inserted_columns,
            n_extended_5p=cres.n_extended_5p,
            n_extended_3p=cres.n_extended_3p,
            variants=vrows if hid == 0 else [],
        )
        orf = gated_orf(
            cres.consensus,
            certified,
            seed_orf_end=seed_orf_end,
            min_aa_length=min_aa_length,
        )
        if orf is not None:
            node.protein, node.orf_start, node.orf_end, node.orf_certified_end, node.orf_is_truncated_by_support = orf
        return node

    if not sel:
        return [_node(0, members, "", np.empty(0, dtype=np.int64))]

    # Place every member on the selected columns and collapse to haplotypes.
    sub = [vrows[i] for i in sel]
    var_cons = np.array([r[0] for r in sub], dtype=np.int64)
    # Read alleles at the PWM columns the caller reported. An insertion
    # column has no consensus position to map back from, so going via
    # `frame_of_cons[var_cons]` would silently address the wrong column.
    var_frame = np.array([r[12] for r in sub], dtype=np.int64)
    A = member_alleles(pooled, var_frame)
    weights = [m.weight for m in members]
    member_of_row = list(range(len(members)))
    w = np.asarray(weights, dtype=np.float64)
    # Phasing r² per selected column; the caller attaches it to the variant
    # table so a reader can tell a linked allele pair from scattered error.
    hres = build_haplotypes(
        A,
        w,
        [int(x) for x in var_cons],
        [r[2] for r in sub],
        [r[1] for r in sub],
    )

    # Group members by their allele row, most-supported first.
    uniq_rows, inverse = np.unique(A, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    mass = np.zeros(uniq_rows.shape[0], dtype=np.float64)
    np.add.at(mass, inverse, w)
    ranked = np.argsort(-mass, kind="stable")

    keep: list[tuple[int, np.ndarray]] = []
    absorbed: list[int] = []
    for rank, h in enumerate(ranked.tolist()):
        idx = np.flatnonzero(inverse == h)
        # The most-supported haplotype is always a node; a minor one needs
        # real support. min_haplotype_frac defaults to 0 — see the module
        # docstring for why a 1% floor is actively harmful at depth.
        if rank == 0 or (
            mass[h] >= min_haplotype_reads
            and mass[h] >= min_haplotype_frac * total_reads
        ):
            keep.append((h, idx))
        else:
            absorbed.extend(idx.tolist())

    # Unsupported haplotypes fold into the major node rather than vanishing,
    # so the node read counts still sum to the template's total.
    if absorbed:
        h0, idx0 = keep[0]
        keep[0] = (h0, np.concatenate([idx0, np.asarray(absorbed, dtype=np.int64)]))

    out: list[RefinedTemplate] = []
    for hid, (h, idx) in enumerate(keep):
        allele = "".join("." if v < 0 else "ACGT-"[v] for v in uniq_rows[h])
        out.append(
            _node(
                hid,
                [members[member_of_row[int(i)]] for i in idx],
                allele,
                var_cons,
            )
        )
    if out:
        # The full variant catalogue hangs off the major node, each row
        # carrying its phasing r² (0.0 for positions that defined no column).
        # vr[12] is the PWM column — internal, not a table field.
        r2_of = {i: float(hres.max_r2[j]) for j, i in enumerate(sel)}
        out[0].variants = [(*vr[:12], r2_of.get(i, 0.0)) for i, vr in enumerate(vrows)]
    return out


__all__ = [
    "RefinedTemplate",
    "certified_columns",
    "gated_orf",
    "refine_template",
    "specs_from_assignments",
]
