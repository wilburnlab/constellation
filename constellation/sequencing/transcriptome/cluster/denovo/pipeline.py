"""De novo first-round assembly orchestrator.

``assemble_clusters`` is the pure in-memory core (dereplicate → minimizers
→ candidates → verify → connected-components → consensus → ORF → quant);
it takes a reads table and returns Arrow tables, so it's testable without
a demux dir. ``cluster_transcripts`` wraps it with demux-window loading +
output writing + the cluster manifest.

Each component member is aligned to the component centroid for both the
consensus PWM and the membership metrics — reusing the cached verify
CIGAR for directly-verified edges and re-aligning only the members that
reach the centroid through the component (so edlib still never runs twice
on the same edge).
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.sequence.nucleic import STANDARD, CodonTable, translate
from constellation.sequencing.schemas.quant import FEATURE_QUANT
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo._cigar import cigar_stats
from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
    generate_candidates,
)
from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
    connected_components,
)
from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    MemberSpec,
    centroid_consensus,
)
from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
    dereplicate,
)
from constellation.sequencing.transcriptome.cluster.denovo.haplotypes import (
    build_haplotypes,
)
from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
    extract_minimizers,
)
from constellation.sequencing.transcriptome.cluster.denovo.quant import (
    cluster_feature_quant,
)
from constellation.sequencing.transcriptome.cluster.denovo.schemas import (
    CLUSTER_ALIGNMENT_TABLE,
    CLUSTER_HAPLOTYPE_TABLE,
    CLUSTER_VARIANT_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (
    ErrorModel,
    call_variants,
    disagreement_stats,
    estimate_error_rates,
    merge_disagreements,
    reclassify_variants,
)
from constellation.sequencing.transcriptome.cluster.denovo.verify import (
    verify_candidates,
)


_ORF_CODON_TABLE: CodonTable = CodonTable(
    transl_table=STANDARD.transl_table,
    name=f"{STANDARD.name} (ATG-only starts)",
    forward=STANDARD.forward,
    starts=frozenset({"ATG"}),
    stops=STANDARD.stops,
)


@dataclass(slots=True)
class AssembledClusters:
    clusters: pa.Table  # TRANSCRIPT_CLUSTER_TABLE
    membership: pa.Table  # CLUSTER_MEMBERSHIP_TABLE
    feature_quant: pa.Table  # FEATURE_QUANT(feature_origin='cluster_id')
    variants: pa.Table  # CLUSTER_VARIANT_TABLE
    haplotypes: pa.Table  # CLUSTER_HAPLOTYPE_TABLE
    alignments: pa.Table  # CLUSTER_ALIGNMENT_TABLE (empty unless emit_alignments)
    n_input_reads: int
    n_unique: int


@functools.lru_cache(maxsize=8)
def _orf_regex(min_aa_length: int):
    import regex

    # ATG start … ≥ min_aa_length codons … stop. Compiled once per length
    # (vs once per call inside find_orfs — at 10M+ clusters that recompile
    # dominated the per-cluster cost).
    return regex.compile(
        f"(?:ATG)(?:[ACGT]{{3}}){{{min_aa_length - 1},}}?(?:TAA|TAG|TGA)"
    )


def _is_low_complexity(seq: str) -> bool:
    """Cheap guard: a consensus dominated by one base or a long homopolymer
    is almost always a connected-components chaining artifact. Such sequences
    make the overlapped-ORF regex pathological, so skip ORF prediction."""
    n = len(seq)
    if n < 30:
        return False
    counts = (seq.count("A"), seq.count("C"), seq.count("G"), seq.count("T"))
    return max(counts) > 0.8 * n


def _best_clean_orf(seq: str, *, min_aa_length: int):
    """Longest internal-stop-free ATG ORF as ``(protein, start, end, strand,
    transl_table)``, or ``None``. Cached regex + low-complexity guard make it
    cheap enough to run on every one of millions of cluster consensuses."""
    if len(seq) < min_aa_length * 3 or _is_low_complexity(seq):
        return None
    s = seq.upper().replace("U", "T")
    pat = _orf_regex(min_aa_length)
    # Best (longest) internal-stop-free protein per (frame, stop position).
    best: dict[tuple[int, int], tuple] = {}
    for m in pat.finditer(s, overlapped=True):
        nt = s[m.start() : m.end()]
        prot = translate(nt[:-3], codon_table=_ORF_CODON_TABLE, partial="discard")
        if "*" in prot or len(prot) < min_aa_length:
            continue
        key = (m.start() % 3, m.end())
        prior = best.get(key)
        if prior is None or len(prot) > len(prior[0]):
            best[key] = (prot, m.start(), m.end())
    if not best:
        return None
    prot, st, en = max(best.values(), key=lambda x: len(x[0]))
    return prot, int(st), int(en), "+", _ORF_CODON_TABLE.transl_table


# A single cluster holding more unique sequences than this is almost
# certainly a connected-components chaining artifact (a low-complexity or
# repeat minimizer bridging unrelated reads) rather than a real transcript.
_MEGA_CLUSTER_UNIQUES = 2_000_000

# Edlib budget (fraction of the shorter length) for re-aligning a member to
# its centroid when the edge wasn't directly verified.
_REALIGN_MAX_FRAC = 0.15

# Consensus / variant / haplotype statistics saturate well before this many
# unique members; capping the members that feed the per-cluster PWM bounds
# memory + edlib work for both genuinely high-expression clusters and
# chaining artifacts. Quant + membership still cover every read.
_CONSENSUS_MAX_MEMBERS = 20_000

# Variant positions feeding the haplotype matrix + pairwise r². The variant
# table keeps every position; only the covariance build is capped.
_HAPLOTYPE_MAX_VARIANTS = 64

# ── Per-cluster consensus + ORF + member metrics (fork-shared globals) ──
_W: dict[str, Any] = {}


_CIGAR_TRANSPOSE = str.maketrans("ID", "DI")


def _transpose_cigar(cigar: str) -> str:
    """Swap I↔D so an edlib ``query=centroid → ref=member`` CIGAR reads as
    ``member → centroid`` (member as query) — used to emit every member's
    alignment in one consistent centroid-as-reference frame."""
    return cigar.translate(_CIGAR_TRANSPOSE)


def _member_alignment(c: int, m: int):
    """Aligned (short, long, ref_start, cigar) for the (centroid, member)
    edge — cache hit when directly verified, else a fresh edlib re-align."""
    seqs = _W["seqs"]
    seqlen = _W["seqlen"]
    aln = _W["aln"]
    lc, lm = int(seqlen[c]), int(seqlen[m])
    if lm < lc or (lm == lc and m < c):
        s, lng = m, c
    else:
        s, lng = c, m
    hit = aln.get((s << 32) | lng)
    if hit is not None:
        return s, lng, int(hit[0]), hit[1]
    import edlib

    # Bounded re-align: members reach the centroid through the component but
    # may be many edits away. Cap the edlib budget so a chain-far member exits
    # fast (returns None → default metrics, excluded from the consensus PWM by
    # the consensus_gate anyway) instead of doing unbounded path-construction.
    budget = int(_REALIGN_MAX_FRAC * min(int(seqlen[s]), int(seqlen[lng])))
    res = edlib.align(seqs[s], seqs[lng], mode="HW", task="path", k=budget)
    if res["editDistance"] < 0 or not res["locations"]:
        return None
    return s, lng, int(res["locations"][0][0]), (res["cigar"] or "")


def _cluster_chunk(
    bounds: tuple[int, int],
    *,
    predict_orfs: bool,
    min_aa_length: int,
    identity: float,
    overdispersion: float,
    emit_alignments: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[tuple], list[tuple], list[tuple], list[tuple], dict, list[tuple]]:
    seqs = _W["seqs"]
    abund = _W["abund"]
    seqlen = _W["seqlen"]
    ptr = _W["ptr"]
    members = _W["members"]
    centroid = _W["centroid"]
    model = _W["error_model"]
    # Members beyond this identity to the centroid still belong to the
    # component but don't vote in the consensus (protects against chain-end
    # divergence corrupting the centroid frame).
    consensus_gate = 2.0 * (1.0 - identity)
    lo, hi = bounds
    cluster_out: list[tuple] = []
    metric_out: list[tuple] = []
    variant_out: list[tuple] = []
    haplotype_out: list[tuple] = []
    align_out: list[tuple] = []
    disagree: dict[tuple[int, int], tuple[float, float]] = {}
    for cid in range(lo, hi):
        c = int(centroid[cid])
        mem = members[ptr[cid] : ptr[cid + 1]]
        nonc = mem[mem != c]
        # Cap the members that get aligned + vote in the consensus to the
        # most-abundant N (statistics saturate; bounds memory + edlib on
        # high-expression or chained clusters). Overflow members keep their
        # default membership metrics and still count toward quant.
        if nonc.shape[0] > _CONSENSUS_MAX_MEMBERS:
            top = nonc[np.argsort(-abund[nonc], kind="stable")[:_CONSENSUS_MAX_MEMBERS]]
        else:
            top = nonc
        if emit_alignments:
            # Centroid self-row: the consensus frame is anchored on it.
            lc = int(seqlen[c])
            align_out.append(
                (cid, c, "representative", 0, f"{lc}=", lc, 0, 0, 0, 0, 0, True)
            )
        specs: list[MemberSpec] = []
        for m in top:
            m = int(m)
            al = _member_alignment(c, m)
            if al is None:
                metric_out.append((m, int(seqlen[m]), 0, 0, 0, 0, 0, 0))
                continue
            s, lng, ref_start, cigar = al
            nm, nx, ni, nd = cigar_stats(cigar)
            ref_end = ref_start + nm + nx + nd
            oh5 = ref_start
            oh3 = int(seqlen[lng]) - ref_end
            m_is_long = 1 if m == lng else 0
            metric_out.append((m, nm, nx, ni, nd, oh5, oh3, m_is_long))
            len_short = int(seqlen[s])
            edit = nx + ni + nd
            in_cons = len_short > 0 and (edit / len_short) <= consensus_gate
            if in_cons:
                specs.append(
                    MemberSpec(
                        member_seq=seqs[m],
                        weight=float(abund[m]),
                        cigar=cigar,
                        centroid_is_query=(s == c),
                        ref_start=ref_start,
                    )
                )
            if emit_alignments:
                # Normalise to a member→centroid alignment (centroid as ref).
                # When the centroid was the shorter query, transpose the CIGAR
                # (I↔D) and swap insert/delete into the member frame.
                if s == c:
                    cig_out, rs_out, ins_m, del_m = _transpose_cigar(cigar), 0, nd, ni
                else:  # centroid was the longer ref → already member→centroid
                    cig_out, rs_out, ins_m, del_m = cigar, ref_start, ni, nd
                sd5 = oh5 if m_is_long else -oh5
                sd3 = oh3 if m_is_long else -oh3
                align_out.append(
                    (
                        cid,
                        m,
                        "member",
                        rs_out,
                        cig_out,
                        nm,
                        nx,
                        ins_m,
                        del_m,
                        sd5,
                        sd3,
                        bool(in_cons),
                    )
                )
        if specs:
            cres = centroid_consensus(seqs[c], float(abund[c]), specs)
            consensus = cres.consensus
            merge_disagreements(disagree, disagreement_stats(cres))
            vrows = call_variants(cres, model=model, overdispersion=overdispersion)
            if vrows:
                keep = cres.winner < 4
                centroid_of_cons = np.flatnonzero(keep)
                member_weights = np.array(
                    [float(abund[c])] + [s.weight for s in specs], dtype=np.float64
                )
                # Cap the variant positions feeding the haplotype matrix +
                # pairwise r² to the most-supported ones — bounds the (M, V)
                # allele matrix and the list<int32> column (a real offset-
                # overflow risk if a chained cluster has thousands of variants).
                if len(vrows) > _HAPLOTYPE_MAX_VARIANTS:
                    sel = sorted(
                        sorted(range(len(vrows)), key=lambda i: -vrows[i][6])[
                            :_HAPLOTYPE_MAX_VARIANTS
                        ]
                    )
                else:
                    sel = list(range(len(vrows)))
                sub = [vrows[i] for i in sel]
                var_cons = np.array([r[0] for r in sub], dtype=np.int64)
                var_centroid = centroid_of_cons[var_cons]
                hres = build_haplotypes(
                    cres,
                    var_centroid,
                    var_cons,
                    [r[2] for r in sub],
                    [r[1] for r in sub],
                    member_weights,
                )
                r2_by_idx = {sel[j]: float(hres.max_r2[j]) for j in range(len(sub))}
                for i, vr in enumerate(vrows):
                    variant_out.append((cid, *vr, r2_by_idx.get(i, 0.0)))
                for hid, (astr, ab, nu, comp) in enumerate(hres.haplotypes):
                    haplotype_out.append(
                        (cid, hid, astr, hres.variant_positions, ab, nu, comp)
                    )
        else:
            consensus = seqs[c]
        protein = orf_start = orf_end = orf_strand = codon_tbl = None
        if predict_orfs:
            orf = _best_clean_orf(consensus, min_aa_length=min_aa_length)
            if orf is not None:
                protein, orf_start, orf_end, orf_strand, codon_tbl = orf
        cluster_out.append(
            (cid, consensus, protein, orf_start, orf_end, orf_strand, codon_tbl)
        )
        if progress is not None and (cid - lo) % 500_000 == 499_999:
            progress(f"  …consensus/variants: {cid - lo + 1:,} clusters processed")
    return cluster_out, metric_out, variant_out, haplotype_out, disagree, align_out


def assemble_clusters(
    reads: pa.Table,
    *,
    identity: float = 0.98,
    max_5p_overhang: int = 30,
    max_3p_overhang: int = 30,
    kmer: int = 15,
    window: int = 10,
    minimizers_per_seq: int | None = 50,
    min_cluster_size: int = 1,
    min_abundance: int = 1,
    predict_orfs: bool = True,
    min_aa_length: int = 60,
    overdispersion: float = 0.0,
    error_model: ErrorModel | None = None,
    fit_empirical: bool = False,
    emit_alignments: bool = False,
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> AssembledClusters:
    """Pure in-memory de novo first-round assembly over a reads table.

    ``reads`` carries ``(read_id, sequence, sample_id)`` — the trimmed
    transcript windows of Complete, non-fragment reads. ``progress`` is an
    optional ``str -> None`` sink called at each stage boundary (the CLI
    wires it to a flushing stderr printer so the last line before a crash
    pinpoints the failing stage).
    """
    log = progress or (lambda _m: None)
    n_input = reads.num_rows
    log(f"dereplicating {n_input:,} reads…")
    uniq_table, read_map = dereplicate(reads)
    n_uniq = uniq_table.num_rows
    empty = AssembledClusters(
        TRANSCRIPT_CLUSTER_TABLE.empty_table(),
        CLUSTER_MEMBERSHIP_TABLE.empty_table(),
        FEATURE_QUANT.empty_table(),
        CLUSTER_VARIANT_TABLE.empty_table(),
        CLUSTER_HAPLOTYPE_TABLE.empty_table(),
        CLUSTER_ALIGNMENT_TABLE.empty_table(),
        n_input,
        n_uniq,
    )
    if n_uniq == 0:
        log("no unique sequences — nothing to cluster")
        return empty

    seqs = uniq_table.column("sequence").to_pylist()
    abundance = (
        uniq_table.column("abundance").to_numpy(zero_copy_only=False).astype(np.int64)
    )
    seqlen = (
        uniq_table.column("seq_len").to_numpy(zero_copy_only=False).astype(np.int64)
    )

    log(f"{n_uniq:,} unique sequences — extracting minimizers (k={kmer}, w={window})…")
    index = extract_minimizers(
        uniq_table.column("sequence"),
        k=kmer,
        w=window,
        max_per_seq=minimizers_per_seq,
    )
    log(f"{index.mini_hash.shape[0]:,} minimizers — generating candidate pairs…")
    candidates = generate_candidates(index, abundance)
    del index  # free the minimizer index (~tens of GB) before verify
    log(
        f"{candidates.num_rows:,} candidate pairs — verifying (edlib, identity≥{identity})…"
    )
    accepted = verify_candidates(
        candidates,
        seqs,
        identity=identity,
        max_5p=max_5p_overhang,
        max_3p=max_3p_overhang,
        threads=threads,
    )
    del candidates  # free the candidate table before grouping
    log(f"{accepted.num_rows:,} accepted edges — grouping (connected components)…")
    comp = connected_components(
        n_uniq,
        abundance,
        seqlen,
        accepted.column("uniq_short").to_numpy(zero_copy_only=False),
        accepted.column("uniq_long").to_numpy(zero_copy_only=False),
    )
    cluster_of = comp.cluster_of
    centroid_uniq = comp.centroid_uniq
    n_clusters_raw = centroid_uniq.shape[0]

    # ── size filter (total reads; lone sub-min_abundance singletons) + remap ─
    n_reads_raw = np.zeros(n_clusters_raw, dtype=np.int64)
    np.add.at(n_reads_raw, cluster_of, abundance)
    n_uniq_raw = np.bincount(cluster_of, minlength=n_clusters_raw)
    survive = (n_reads_raw >= int(min_cluster_size)) & ~(
        (n_uniq_raw == 1) & (n_reads_raw < int(min_abundance))
    )
    new_id = np.full(n_clusters_raw, -1, dtype=np.int64)
    new_id[survive] = np.arange(int(survive.sum()), dtype=np.int64)
    cluster_of = new_id[cluster_of]
    surviving_centroids = centroid_uniq[survive]
    n_clusters = surviving_centroids.shape[0]
    if n_clusters == 0:
        log("no clusters survived the size filter")
        return empty
    largest = int(n_uniq_raw[survive].max()) if n_clusters else 0
    log(
        f"{n_clusters:,} clusters (largest holds {largest:,} unique sequences) — "
        "building consensus + variants…"
    )
    if largest > _MEGA_CLUSTER_UNIQUES:
        log(
            f"WARNING: a cluster holds {largest:,} unique sequences — likely a "
            "connected-components chaining artifact (low-complexity / repeat). "
            "Consensus uses abundance-weighted chunked accumulation; if this "
            "stalls, raise --identity or --min-cluster-size."
        )

    # ── CSR members per surviving cluster ──
    uniq_ids = np.arange(n_uniq, dtype=np.int64)
    valid = cluster_of >= 0
    cu = cluster_of[valid]
    uu = uniq_ids[valid]
    order = np.argsort(cu, kind="stable")
    cu_s = cu[order]
    uu_s = uu[order]
    ptr = np.zeros(n_clusters + 1, dtype=np.int64)
    np.cumsum(np.bincount(cu_s, minlength=n_clusters), out=ptr[1:])

    aln_lookup = _build_alignment_lookup(accepted)
    _W.clear()
    _W.update(
        seqs=seqs,
        abund=abundance,
        seqlen=seqlen,
        aln=aln_lookup,
        ptr=ptr,
        members=uu_s,
        centroid=surviving_centroids,
        error_model=error_model or ErrorModel(),
    )
    worker_kw = dict(
        predict_orfs=predict_orfs,
        min_aa_length=min_aa_length,
        identity=identity,
        overdispersion=overdispersion,
        emit_alignments=emit_alignments,
    )
    disagree: dict = {}
    align_res: list[tuple] = []
    try:
        if threads <= 1 or n_clusters < 256:
            if threads <= 1 and n_clusters >= 1_000_000:
                log(
                    f"  consensus/variants for {n_clusters:,} clusters "
                    "single-threaded — pass --threads N to parallelise"
                )
            (
                cluster_res,
                metric_res,
                variant_res,
                hap_res,
                disagree,
                align_res,
            ) = _cluster_chunk((0, n_clusters), progress=log, **worker_kw)
        else:
            step = max(1, (n_clusters + threads * 8 - 1) // (threads * 8))
            bounds = [
                (lo, min(lo + step, n_clusters)) for lo in range(0, n_clusters, step)
            ]
            ctx = mp.get_context("fork")
            cluster_res, metric_res, variant_res, hap_res = [], [], [], []
            done = 0
            with ProcessPoolExecutor(max_workers=threads, mp_context=ctx) as ex:
                futs = [ex.submit(_cluster_chunk, b, **worker_kw) for b in bounds]
                for fut in futs:
                    cr, mr, vr, hr, dg, ar = fut.result()
                    cluster_res.extend(cr)
                    metric_res.extend(mr)
                    variant_res.extend(vr)
                    hap_res.extend(hr)
                    merge_disagreements(disagree, dg)
                    align_res.extend(ar)
                    done += 1
                    if done % max(1, len(bounds) // 20) == 0:
                        log(f"  …consensus/variants: {done}/{len(bounds)} chunks done")
    finally:
        _W.clear()
    cluster_res.sort(key=lambda r: r[0])

    # Empirical error model: re-fit ε from the cluster-wide high-confidence
    # disagreements and re-test every variant position (no second consensus).
    if fit_empirical and variant_res:
        fitted = estimate_error_rates(disagree, base=error_model or ErrorModel())
        variant_res = reclassify_variants(
            variant_res, fitted, overdispersion=overdispersion
        )

    clusters = _build_cluster_table(
        cluster_res,
        surviving_centroids=surviving_centroids,
        uniq_table=uniq_table,
        abundance=abundance,
        cluster_of=cluster_of,
        n_clusters=n_clusters,
        identity=identity,
    )
    membership = _build_membership_table(
        read_map=read_map,
        uniq_table=uniq_table,
        cluster_of=cluster_of,
        surviving_centroids=surviving_centroids,
        seqlen=seqlen,
        metric_res=metric_res,
    )
    feature_quant = cluster_feature_quant(read_map, cluster_of)
    variants = _build_variant_table(variant_res)
    haplotypes = _build_haplotype_table(hap_res)
    alignments = _build_alignment_table(align_res, uniq_table)
    log(
        f"assembled {clusters.num_rows:,} clusters, "
        f"{membership.num_rows:,} membership rows, {variants.num_rows:,} variants"
    )
    return AssembledClusters(
        clusters,
        membership,
        feature_quant,
        variants,
        haplotypes,
        alignments,
        n_input,
        n_uniq,
    )


def _build_variant_table(variant_res: list[tuple]) -> pa.Table:
    if not variant_res:
        return CLUSTER_VARIANT_TABLE.empty_table()
    cols = list(zip(*variant_res))
    return pa.table(
        {
            "cluster_id": pa.array(cols[0], pa.int64()),
            "consensus_pos": pa.array(cols[1], pa.int32()),
            "consensus_allele": pa.array(cols[2], pa.string()),
            "minor_allele": pa.array(cols[3], pa.string()),
            "variant_class": pa.array(cols[4], pa.string()),
            "homopolymer_run": pa.array(cols[5], pa.int32()),
            "depth_total": pa.array(cols[6], pa.int32()),
            "depth_minor": pa.array(cols[7], pa.int32()),
            "minor_fraction": pa.array(cols[8], pa.float32()),
            "p_error": pa.array(cols[9], pa.float64()),
            "epsilon_class": pa.array(cols[10], pa.float32()),
            "call": pa.array(cols[11], pa.string()),
            "max_linkage_r2": pa.array(cols[12], pa.float32()),
        },
        schema=CLUSTER_VARIANT_TABLE,
    )


def _build_haplotype_table(hap_res: list[tuple]) -> pa.Table:
    if not hap_res:
        return CLUSTER_HAPLOTYPE_TABLE.empty_table()
    cols = list(zip(*hap_res))
    n = len(hap_res)
    return pa.table(
        {
            "cluster_id": pa.array(cols[0], pa.int64()),
            "haplotype_id": pa.array(cols[1], pa.int32()),
            "allele_string": pa.array(cols[2], pa.string()),
            "variant_positions": pa.array(cols[3], pa.list_(pa.int32())),
            "abundance": pa.array(cols[4], pa.int64()),
            "n_unique_sequences": pa.array(cols[5], pa.int32()),
            "per_sample_abundance": pa.nulls(n, pa.list_(pa.int64())),
            "is_complete": pa.array(cols[6], pa.bool_()),
        },
        schema=CLUSTER_HAPLOTYPE_TABLE,
    )


def _build_alignment_table(align_res: list[tuple], uniq_table: pa.Table) -> pa.Table:
    """Resolve per-member alignment rows (keyed on uniq_id) into the
    CLUSTER_ALIGNMENT_TABLE — read_id + abundance taken from uniq_table."""
    if not align_res:
        return CLUSTER_ALIGNMENT_TABLE.empty_table()
    cols = list(zip(*align_res))
    uid = np.asarray(cols[1], dtype=np.int64)
    nm = np.asarray(cols[5], dtype=np.int64)
    nx = np.asarray(cols[6], dtype=np.int64)
    ni = np.asarray(cols[7], dtype=np.int64)
    nd = np.asarray(cols[8], dtype=np.int64)
    aligned = nm + nx + ni + nd
    identity = np.where(aligned > 0, nm / np.maximum(aligned, 1), 1.0).astype(
        np.float32
    )
    return pa.table(
        {
            "cluster_id": pa.array(cols[0], pa.int64()),
            "read_id": pc.take(
                uniq_table.column("representative_read_id"), pa.array(uid)
            ).cast(pa.string()),
            "role": pa.array(cols[2], pa.string()),
            "abundance": pc.take(uniq_table.column("abundance"), pa.array(uid)).cast(
                pa.int64()
            ),
            "ref_start": pa.array(cols[3], pa.int32()),
            "cigar": pa.array(cols[4], pa.string()),
            "n_match": pa.array(nm, pa.int32()),
            "n_mismatch": pa.array(nx, pa.int32()),
            "n_insert": pa.array(ni, pa.int32()),
            "n_delete": pa.array(nd, pa.int32()),
            "identity": pa.array(identity),
            "overhang_5p": pa.array(cols[9], pa.int32()),
            "overhang_3p": pa.array(cols[10], pa.int32()),
            "in_consensus": pa.array(cols[11], pa.bool_()),
        },
        schema=CLUSTER_ALIGNMENT_TABLE,
    )


def _build_alignment_lookup(accepted: pa.Table) -> dict[int, tuple[int, str]]:
    short = accepted.column("uniq_short").to_numpy(zero_copy_only=False)
    long = accepted.column("uniq_long").to_numpy(zero_copy_only=False)
    ref_start = accepted.column("ref_start").to_numpy(zero_copy_only=False)
    cigar = accepted.column("cigar").to_pylist()
    keys = (short.astype(np.int64) << 32) | long.astype(np.int64)
    return {int(k): (int(rs), cg) for k, rs, cg in zip(keys, ref_start, cigar)}


def _build_cluster_table(
    results: list[tuple],
    *,
    surviving_centroids: np.ndarray,
    uniq_table: pa.Table,
    abundance: np.ndarray,
    cluster_of: np.ndarray,
    n_clusters: int,
    identity: float,
) -> pa.Table:
    rep_per_uniq = uniq_table.column("representative_read_id").to_pylist()
    valid = cluster_of >= 0
    n_unique = np.bincount(cluster_of[valid], minlength=n_clusters)
    n_reads = np.zeros(n_clusters, dtype=np.int64)
    np.add.at(n_reads, cluster_of[valid], abundance[valid])

    cid = [r[0] for r in results]
    rep = [rep_per_uniq[int(surviving_centroids[c])] for c in cid]
    return pa.table(
        {
            "cluster_id": pa.array(cid, pa.int64()),
            "representative_read_id": pa.array(rep, pa.string()),
            "n_reads": pa.array([int(n_reads[c]) for c in cid], pa.int32()),
            "identity_threshold": pa.array([float(identity)] * len(cid), pa.float32()),
            "consensus_sequence": pa.array([r[1] for r in results], pa.string()),
            "predicted_protein": pa.array([r[2] for r in results], pa.string()),
            "orf_start": pa.array([r[3] for r in results], pa.int32()),
            "orf_end": pa.array([r[4] for r in results], pa.int32()),
            "orf_strand": pa.array([r[5] for r in results], pa.string()),
            "codon_table": pa.array([r[6] for r in results], pa.int32()),
            "mode": pa.array(["de-novo"] * len(cid), pa.string()),
            "contig_id": pa.nulls(len(cid), pa.int64()),
            "strand": pa.nulls(len(cid), pa.string()),
            "span_start": pa.nulls(len(cid), pa.int64()),
            "span_end": pa.nulls(len(cid), pa.int64()),
            "fingerprint_hash": pa.nulls(len(cid), pa.uint64()),
            "n_unique_sequences": pa.array([int(n_unique[c]) for c in cid], pa.int32()),
            "sample_id": pa.nulls(len(cid), pa.int64()),
        },
        schema=TRANSCRIPT_CLUSTER_TABLE,
    )


def _build_membership_table(
    *,
    read_map: pa.Table,
    uniq_table: pa.Table,
    cluster_of: np.ndarray,
    surviving_centroids: np.ndarray,
    seqlen: np.ndarray,
    metric_res: list[tuple],
) -> pa.Table:
    n_uniq = uniq_table.num_rows
    is_centroid = np.zeros(n_uniq, dtype=bool)
    if surviving_centroids.shape[0]:
        is_centroid[surviving_centroids] = True

    # Per-uniq metrics — centroid self-values by default.
    match_rate = np.ones(n_uniq, dtype=np.float64)
    indel_rate = np.zeros(n_uniq, dtype=np.float64)
    n_aligned = seqlen.astype(np.int64).copy()
    drift5 = np.zeros(n_uniq, dtype=np.int64)
    drift3 = np.zeros(n_uniq, dtype=np.int64)

    if metric_res:
        arr = np.asarray(metric_res, dtype=np.int64)
        mu = arr[:, 0]
        nm, nx, ni, nd = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        oh5, oh3, m_is_long = arr[:, 5], arr[:, 6], arr[:, 7].astype(bool)
        aligned = nm + nx + ni + nd
        with np.errstate(divide="ignore", invalid="ignore"):
            mr = np.where((nm + nx) > 0, nm / (nm + nx), 1.0)
            ir = np.where(aligned > 0, (ni + nd) / aligned, 0.0)
        d5 = np.where(m_is_long, oh5, -oh5)
        d3 = np.where(m_is_long, oh3, -oh3)
        match_rate[mu] = mr
        indel_rate[mu] = ir
        n_aligned[mu] = np.where(aligned > 0, aligned, seqlen[mu])
        drift5[mu] = d5
        drift3[mu] = d3

    r_uniq = read_map.column("uniq_id").to_numpy(zero_copy_only=False)
    r_cluster = cluster_of[r_uniq]
    keep = r_cluster >= 0
    keep_arr = pa.array(keep)

    # role — Arrow-native so read_id never round-trips through a Python list /
    # a giant numpy '<U' array (the read-cardinality anti-pattern + a real
    # offset-overflow / OOM hazard at PromethION scale).
    expected_rep = pc.take(
        uniq_table.column("representative_read_id"), pa.array(r_uniq)
    )
    is_uniq_rep = pc.equal(read_map.column("read_id"), expected_rep).to_numpy(
        zero_copy_only=False
    )
    uniq_is_centroid = is_centroid[r_uniq]
    role = np.where(
        ~is_uniq_rep,
        "duplicate",
        np.where(uniq_is_centroid, "representative", "member"),
    )

    return pa.table(
        {
            "cluster_id": pa.array(r_cluster[keep].astype(np.int64)),
            "read_id": pc.filter(read_map.column("read_id"), keep_arr).cast(
                pa.string()
            ),
            "role": pa.array(role[keep]),
            "drift_5p_bp": pa.array(drift5[r_uniq][keep].astype(np.int32)),
            "drift_3p_bp": pa.array(drift3[r_uniq][keep].astype(np.int32)),
            "match_rate": pa.array(match_rate[r_uniq][keep].astype(np.float32)),
            "indel_rate": pa.array(indel_rate[r_uniq][keep].astype(np.float32)),
            "n_aligned_bp": pa.array(n_aligned[r_uniq][keep].astype(np.int32)),
        },
        schema=CLUSTER_MEMBERSHIP_TABLE,
    )


def cluster_transcripts(
    demux_dir: Path,
    *,
    output_dir: Path,
    identity: float = 0.98,
    max_5p_overhang: int = 30,
    max_3p_overhang: int = 30,
    kmer: int = 15,
    window: int = 10,
    minimizers_per_seq: int | None = 50,
    min_cluster_size: int = 1,
    min_abundance: int = 1,
    max_cluster_rounds: int = 1,
    error_model: Literal["default", "empirical"] = "default",
    overdispersion: float = 0.0,
    predict_orfs: bool = True,
    min_aa_length: int = 60,
    emit_cluster_detail: bool = False,
    detail_top_n: int = 50,
    emit_alignments: bool = False,
    threads: int = 1,
    samples: Any | None = None,
    write_fasta: bool = True,
    report: bool = True,
    progress_cb: Any | None = None,
    verbose: bool = False,
    resume: bool = False,
) -> dict[str, Path]:
    """Run de novo first-round assembly over an S1 demux output dir.

    ``verbose`` prints per-stage breadcrumbs to stderr (flushed immediately,
    so the last line survives even a native-library abort); ``progress_cb``,
    if given, additionally receives a ``ProgressEvent`` per stage.
    """
    from constellation.sequencing.transcriptome.cluster.denovo._io import (
        load_demux_windows,
        write_outputs,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage logger — flush immediately so the last line is visible even if a
    # native library aborts the process (the std::length_error class of crash
    # bypasses Python's normal stderr flush).
    log: Callable[[str], None] | None = None
    if verbose or progress_cb is not None:

        def _log(msg: str) -> None:
            if verbose:
                print(f"[de-novo] {msg}", file=sys.stderr, flush=True)
            if progress_cb is not None:
                try:
                    from constellation.core.progress import ProgressEvent

                    progress_cb(
                        ProgressEvent(
                            kind="stage_progress", stage="de-novo", message=msg
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass

        log = _log
        log(f"loading Complete demux windows from {demux_dir}…")

    reads = load_demux_windows(Path(demux_dir))
    result = assemble_clusters(
        reads,
        identity=identity,
        max_5p_overhang=max_5p_overhang,
        max_3p_overhang=max_3p_overhang,
        kmer=kmer,
        window=window,
        minimizers_per_seq=minimizers_per_seq,
        min_cluster_size=min_cluster_size,
        min_abundance=min_abundance,
        predict_orfs=predict_orfs,
        min_aa_length=min_aa_length,
        overdispersion=overdispersion,
        fit_empirical=(error_model == "empirical"),
        emit_alignments=emit_alignments,
        threads=threads,
        progress=log,
    )
    paths = write_outputs(
        result,
        output_dir=output_dir,
        demux_dir=Path(demux_dir),
        samples=samples,
        write_fasta=write_fasta,
        predict_orfs=predict_orfs,
        emit_cluster_detail=emit_cluster_detail,
        detail_top_n=detail_top_n,
        parameters={
            "mode": "de-novo",
            "identity": float(identity),
            "max_5p_overhang": int(max_5p_overhang),
            "max_3p_overhang": int(max_3p_overhang),
            "kmer": int(kmer),
            "window": int(window),
            "minimizers_per_seq": (
                int(minimizers_per_seq) if minimizers_per_seq is not None else None
            ),
            "min_cluster_size": int(min_cluster_size),
            "min_abundance": int(min_abundance),
            "max_cluster_rounds": int(max_cluster_rounds),
            "error_model": str(error_model),
            "overdispersion": float(overdispersion),
            "predict_orfs": bool(predict_orfs),
            "min_aa_length": int(min_aa_length),
            "emit_alignments": bool(emit_alignments),
            "threads": int(threads),
        },
    )
    if report:
        try:
            from constellation.sequencing.transcriptome.cluster.denovo.diagnostics import (  # noqa: E501
                build_denovo_diagnostics_report,
            )

            paths["report"] = build_denovo_diagnostics_report(output_dir)
        except Exception:  # noqa: BLE001 — report never breaks a successful run
            pass
    return paths


__all__ = ["cluster_transcripts", "assemble_clusters", "AssembledClusters"]
