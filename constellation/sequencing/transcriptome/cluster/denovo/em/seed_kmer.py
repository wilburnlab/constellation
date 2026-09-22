"""Stage 1, alternative — seeding by read-level kmer clustering.

The ORF seeder (:mod:`.seed`) makes one template per distinct ORF, and
distinct ORFs scale ~linearly with reads, so it compresses ~1.4x: 9.39M reads
became 3,778,760 templates / 6.12 Gb, and round 1's E-step against that set
cost **13.95 h and 127 GB** — after which only 11.6% of the templates had
recruited a read.

This seeder runs the shipped ``--mode kmer`` kernels on the **reads
themselves** and elects one read per cluster:

    dereplicate -> extract_minimizers -> generate_candidates
                -> verify_candidates -> connected_components -> elect

Measured on the same 9.39M reads at ``identity 0.93``, ends ``5' unbounded /
3' <= 100``, k15 w10 m100: **850,450 templates in 1.26 Gb**, and the partition
it produces *before any alignment to a template* is better than the one the
ORF path reaches *after* its 14 h E-step (gene dominance 0.866 vs 0.821,
purity 0.985 vs 0.978, ARI 0.931 vs 0.560). Round 1's E-step drops to 2.89 h
at 26 GB while assigning **more** reads (98.6% vs 96.2%) at the same gene
agreement.

Three things about this that are easy to get backwards:

* **The grouping order and the template election are different decisions.**
  The bench swept one "rule" that drove the minimizer bucket anchor, the
  component centroid order *and* which read became the template, which forced
  a choice between the best partition (``abundance``) and the best-sequenced
  representative (``longest-above-quality``). They are separable: the
  grouping keeps the shipped abundance order, and the template is elected by
  the EM's own ``--seed-representative`` policy over the cluster's members.
  So ``comp.centroid_uniq`` is computed and **deliberately discarded**.

* **linclust's longest-read centre is the wrong anchor here.** Swept at
  9.39M reads it loses decisively (gene dominance 0.866 ``abundance`` vs
  0.815 ``length``): anchoring on the longest read pairs the most
  length-discordant reads in each bucket, which are exactly the pairs the
  overhang gate rejects. At 1M the rules converge, so the experiment that
  would have "settled" this at small scale gets it backwards.

* **The ORF is annotation here, not part of the algorithm.** Under ORF
  seeding the ORF is the *key* — a template exists because a distinct ORF
  did. This seeder clusters on read sequence similarity, the M-step splits on
  column covariance, and round 1 ranks on cluster support and seed quality,
  so nothing in the loop consults an ORF. Templates therefore ship
  ``orf_start == orf_end == 0`` and no ORF is predicted at seed time. That is
  load-bearing rather than an omission: ``mstep.gated_orf`` treats the seed
  ORF interval as **certified by construction** and declines to judge it, so
  predicting one here would silently *relax* the M-step's support gate on the
  strength of a claim this seeder never made. The M-step predicts each node's
  ORF on its own consensus, under its own ``min_aa_length``, which is where
  the protein annotation belongs.

* **A 3' cap is what makes 5'-unbounded recruitment safe, and it only shows
  at depth.** With unbounded ends, connected components put **7,096,457 reads
  (76% of the corpus) into one component at purity 0.154** — invisible below
  full scale, where the same setting peaks at 328k. Hence
  :func:`seed_by_kmer_clustering`'s ``max_cluster_read_frac`` guard, which
  counts *reads* (the existing ``_MEGA_CLUSTER_UNIQUES`` warning counts
  uniques and never fired on this).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
    generate_candidates,
)
from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
    connected_components,
    greedy_set_cover,
)
from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
    dereplicate,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.elect import (
    best_quality_per_uniq,
    elect_representatives,
    resolve_policy,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
    TEMPLATE_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
    extract_minimizers,
)
from constellation.sequencing.transcriptome.cluster.denovo.verify import (
    UNBOUNDED_OVERHANG,
    verify_candidates,
)


#: One row per read. ``cluster_id`` is ``-1`` for a read whose cluster fell
#: below ``min_seed_reads`` — the row is kept rather than dropped so the table
#: stays one-per-read and the diagnostics can count what the filter cost.
#: (The ORF path's read->orf map drops unmapped reads instead, because "this
#: read carries no ORF" is a property of the read, not of a filter.)
READ_CLUSTER_MAP_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("cluster_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=True),
    ]
)

Grouping = Literal["components", "greedy"]

#: Defaults, from the 9.39M-read sweep. `identity` and the asymmetric ends are
#: the whole result: `0.98 / 30:30` (the `--mode kmer` shipped gate) costs the
#: same 14 h as ORF seeding, and relaxing to these buys the 4.8x cut.
DEFAULT_SEED_IDENTITY = 0.93
DEFAULT_SEED_MAX_5P = UNBOUNDED_OVERHANG
DEFAULT_SEED_MAX_3P = 100
#: Fraction of the corpus in one cluster above which seeding FAILS. A quarter
#: of the corpus is never one transcript, and the failure it catches costs a
#: multi-hour round: the E-step would index one mega-template and the M-step's
#: member cap would quietly bound its PWM while the partition stays garbage.
#: Deliberately far above the 4.4% `500:500` case the sweep calls "already
#: unsafe" — that one gets the warning below, because 4% of a corpus is within
#: reach of a genuinely dominant transcript and a hard failure has to be
#: something no biology explains.
DEFAULT_MAX_CLUSTER_READ_FRAC = 0.25
#: ...but a *fraction* is only meaningful once there is a corpus to be a
#: fraction OF. Three transcripts sequenced 20x each put 33% of the reads in
#: each cluster, correctly. So both thresholds also require this many reads in
#: the cluster: below it there is no corpus-scale chain to catch, and the
#: entire evidence base for these numbers is at 9.4M reads. Not a measured
#: constant — a floor chosen to keep a guard from firing where it has nothing
#: to say.
DEFAULT_MIN_CHAIN_CLUSTER_READS = 10_000
#: Reported, not fatal. On real data at the shipped operating point the largest
#: cluster is 77,591 reads of 9.39M — 0.83% — so 1% is already past what any
#: transcript's depth explains, while 4.4% (`500:500`) and 76% (`inf:inf`) are
#: the two chaining failures the sweep found.
_CHAIN_WARN_FRAC = 0.01


@dataclass(frozen=True, slots=True)
class KmerSeedResult:
    """Round-1 templates from read-level kmer clustering."""

    templates: pa.Table  # TEMPLATE_TABLE
    read_cluster: pa.Table  # READ_CLUSTER_MAP_SCHEMA
    stats: dict[str, Any] = field(default_factory=dict)


class ChainedClusterError(RuntimeError):
    """One connected component swallowed too much of the corpus."""


def seed_by_kmer_clustering(
    reads: pa.Table,
    *,
    identity: float = DEFAULT_SEED_IDENTITY,
    max_5p_overhang: int = DEFAULT_SEED_MAX_5P,
    max_3p_overhang: int = DEFAULT_SEED_MAX_3P,
    kmer: int = 15,
    window: int = 10,
    minimizers_per_seq: int | None = 50,
    min_shared: int = 2,
    diag_span_max: int = 20,
    grouping: Grouping = "components",
    min_seed_reads: int = 1,
    representative: str = "longest-above-quality",
    max_cluster_read_frac: float = DEFAULT_MAX_CLUSTER_READ_FRAC,
    min_chain_cluster_reads: int = DEFAULT_MIN_CHAIN_CLUSTER_READS,
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> KmerSeedResult:
    """Cluster the reads by kmer similarity; elect one template per cluster.

    ``reads`` carries ``(read_id, sequence, sample_id)`` and, where the demux
    dir is new enough, ``dorado_quality`` — the trimmed transcript windows,
    the same table :func:`~.seed.extract_seed_orfs` takes.

    Raises :class:`ChainedClusterError` when one cluster holds more than
    ``max_cluster_read_frac`` of the reads *and* at least
    ``min_chain_cluster_reads`` of them (0 disables either half); anything
    above 1% of a corpus that large is warned about regardless.
    """
    log = progress or (lambda _m: None)
    # Resolve the policy first: an unknown or ORF-dependent name should fail in
    # a millisecond rather than after the minimizer pass.
    policy = resolve_policy(representative, allow_orf_dependent=False)

    n_reads = reads.num_rows
    stats: dict[str, Any] = {
        "seeding": "kmer",
        "n_reads": int(n_reads),
        "identity": float(identity),
        "max_5p_overhang": int(max_5p_overhang),
        "max_3p_overhang": int(max_3p_overhang),
        "grouping": str(grouping),
    }
    if n_reads == 0:
        return KmerSeedResult(
            TEMPLATE_TABLE.empty_table(),
            READ_CLUSTER_MAP_SCHEMA.empty_table(),
            stats,
        )

    t0 = time.time()
    log(f"dereplicating {n_reads:,} reads…")
    uniq, read_map = dereplicate(reads)
    n_uniq = uniq.num_rows
    stats["n_uniq"] = int(n_uniq)
    stats["dereplicate_s"] = round(time.time() - t0, 1)
    if n_uniq == 0:
        return KmerSeedResult(
            TEMPLATE_TABLE.empty_table(),
            READ_CLUSTER_MAP_SCHEMA.empty_table(),
            stats,
        )

    abundance = uniq.column("abundance").to_numpy(zero_copy_only=False).astype(np.int64)
    seq_len = uniq.column("seq_len").to_numpy(zero_copy_only=False).astype(np.int64)
    uniq_quality, uniq_best_row = best_quality_per_uniq(reads, read_map, n_uniq)

    t0 = time.time()
    log(f"{n_uniq:,} unique reads — extracting minimizers (k={kmer}, w={window})…")
    index = extract_minimizers(
        uniq.column("sequence"), k=kmer, w=window, max_per_seq=minimizers_per_seq
    )
    stats["n_minimizers"] = int(index.mini_hash.shape[0])
    stats["minimizers_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    log(f"{stats['n_minimizers']:,} minimizers — generating candidate pairs…")
    candidates = generate_candidates(
        index, abundance, min_shared=min_shared, diag_span_max=diag_span_max
    )
    # The minimizer index is tens of GB at PromethION scale; free it before
    # verify rather than holding both live, exactly as `assemble_clusters` does.
    del index
    stats["n_candidates"] = int(candidates.num_rows)
    stats["candidates_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    log(
        f"{candidates.num_rows:,} candidate pairs — verifying "
        f"(edlib, identity≥{identity}, ends {_ov(max_5p_overhang)}:"
        f"{_ov(max_3p_overhang)})…"
    )
    seqs = uniq.column("sequence").to_pylist()
    accepted = verify_candidates(
        candidates,
        seqs,
        identity=identity,
        max_5p=max_5p_overhang,
        max_3p=max_3p_overhang,
        threads=threads,
    )
    del candidates
    stats["n_edges"] = int(accepted.num_rows)
    stats["verify_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    group = connected_components if grouping == "components" else greedy_set_cover
    log(f"{accepted.num_rows:,} accepted edges — grouping ({grouping})…")
    comp = group(
        n_uniq,
        abundance,
        seq_len,
        accepted.column("uniq_short").to_numpy(zero_copy_only=False),
        accepted.column("uniq_long").to_numpy(zero_copy_only=False),
    )
    del accepted
    cluster_of = comp.cluster_of
    # `comp.centroid_uniq` is deliberately unused: the grouping order elects a
    # centroid, but the TEMPLATE is elected by the --seed-representative
    # policy over the cluster's members. See the module docstring.
    n_clusters_raw = int(comp.centroid_uniq.shape[0])
    stats["grouping_s"] = round(time.time() - t0, 1)

    elected_uniq, cluster_reads = elect_representatives(
        cluster_of,
        n_clusters_raw,
        template_length=seq_len,
        abundance=abundance,
        quality=uniq_quality,
        orf_start=None,
        policy=policy,
    )

    # ── the chaining guard (counts READS, not uniques) ────────────────
    largest = int(cluster_reads.max()) if cluster_reads.size else 0
    largest_frac = largest / float(n_reads)
    stats["n_clusters_raw"] = n_clusters_raw
    stats["largest_cluster_reads"] = largest
    stats["largest_cluster_read_frac"] = round(largest_frac, 6)
    big_enough_to_judge = largest >= min_chain_cluster_reads
    if (
        max_cluster_read_frac > 0
        and big_enough_to_judge
        and largest_frac > max_cluster_read_frac
    ):
        raise ChainedClusterError(
            f"one cluster holds {largest:,} reads ({largest_frac:.1%} of the "
            f"corpus), above --max-seed-cluster-frac "
            f"{max_cluster_read_frac:.0%}. Connected components chain through "
            f"bridging reads, and at depth only a 3' cap prevents it: at 9.4M "
            f"reads, unbounded ends put 76% of the corpus in ONE component at "
            f"purity 0.154. Lower --max-3p-overhang (100 is the measured "
            f"operating point), raise --identity, or use "
            f"--seed-grouping greedy, which cannot chain by construction."
        )
    if big_enough_to_judge and largest_frac > _CHAIN_WARN_FRAC:
        log(
            f"  WARNING: the largest cluster holds {largest:,} reads "
            f"({largest_frac:.2%} of the corpus) — far past any real "
            f"transcript's depth, so this is very likely connected-components "
            f"chaining. Check --max-3p-overhang / --identity."
        )

    # ── size filter + dense remap ─────────────────────────────────────
    floor = max(int(min_seed_reads), 1)
    survive = (cluster_reads >= floor) & (elected_uniq >= 0)
    new_id = np.full(n_clusters_raw, -1, dtype=np.int64)
    n_templates = int(survive.sum())
    new_id[survive] = np.arange(n_templates, dtype=np.int64)
    elected_uniq = elected_uniq[survive]
    cluster_reads = cluster_reads[survive]
    stats["n_clusters"] = n_templates
    stats["n_clusters_ge2"] = int((cluster_reads >= 2).sum())
    stats["n_dropped_below_min_seed_reads"] = int(n_clusters_raw - n_templates)
    # Built before the early exit, not after it: the contract is one row per
    # INPUT read, and "every cluster was filtered out" is exactly the case
    # where knowing which reads were dropped matters most.
    read_cluster = _read_cluster_map(read_map, cluster_of, new_id)
    if n_templates == 0:
        log("no cluster cleared --min-seed-reads")
        return KmerSeedResult(TEMPLATE_TABLE.empty_table(), read_cluster, stats)

    del seqs
    take = pa.array(elected_uniq)
    seed_rows = uniq_best_row[elected_uniq]
    quality = (
        pa.array(
            np.where(
                seed_rows >= 0,
                uniq_quality[elected_uniq],
                np.nan,
            ).astype(np.float32),
            pa.float32(),
        )
        if np.any(seed_rows >= 0)
        else pa.nulls(n_templates, pa.float32())
    )
    templates = pa.table(
        {
            "template_id": pa.array(np.arange(n_templates, dtype=np.int64)),
            "sequence": pc.take(uniq.column("sequence"), take).cast(pa.large_string()),
            # No ORF, deliberately — see "the ORF is annotation" above.
            # (0, 0) is not a placeholder: it is what `gated_orf` reads as
            # "nothing here is certified by construction", which is the true
            # statement about a template this seeder elected.
            "orf_start": pa.array(np.zeros(n_templates, dtype=np.int32)),
            "orf_end": pa.array(np.zeros(n_templates, dtype=np.int32)),
            "orf_aa_length": pa.array(np.zeros(n_templates, dtype=np.int32)),
            # Both start as the cluster's read count: the first is what later
            # rounds overwrite with assigned mass, the second is round 1's
            # ranking key (`scheduler.rank_round1`) and is never overwritten.
            # Under this seeder that key is *cluster support* rather than ORF
            # replication — a strictly better round-1 prior, under a column
            # name kept for cross-round resume compatibility.
            "node_weight": pa.array(cluster_reads.astype(np.float64)),
            "orf_replication": pa.array(cluster_reads),
            "seed_read_quality": quality,
            "seed_read_row": pa.array(seed_rows.astype(np.int32)),
            "declared_variants": pa.array([[]] * n_templates, pa.list_(pa.int64())),
        },
        schema=TEMPLATE_TABLE,
    )

    stats["n_reads_in_templates"] = int(cluster_reads.sum())
    stats["singleton_frac"] = round(
        float((cluster_reads == 1).mean()) if n_templates else 0.0, 6
    )
    stats["template_mb"] = round(float(seq_len[elected_uniq].sum() / 1e6), 3)
    stats["reads_per_template"] = round(float(cluster_reads.sum() / n_templates), 3)
    if seed_rows.size and np.any(seed_rows >= 0):
        q = uniq_quality[elected_uniq]
        ok = q >= 0
        if ok.any():
            stats["seed_quality_median"] = round(float(np.median(q[ok])), 3)
            stats["seed_quality_ge_floor_frac"] = round(
                float((q[ok] >= 22.0).mean()), 4
            )
    log(
        f"seeded {n_templates:,} templates "
        f"({stats['template_mb']:.0f} Mb) over {n_reads:,} reads"
    )
    return KmerSeedResult(templates, read_cluster, stats)


def _read_cluster_map(
    read_map: pa.Table, cluster_of: np.ndarray, new_id: np.ndarray
) -> pa.Table:
    """One row per input read; ``cluster_id`` is -1 where the cluster was cut."""
    uid = read_map.column("uniq_id").to_numpy(zero_copy_only=False).astype(np.int64)
    return pa.table(
        {
            "read_id": read_map.column("read_id"),
            "cluster_id": pa.array(new_id[cluster_of[uid]]),
            "sample_id": read_map.column("sample_id"),
        },
        schema=READ_CLUSTER_MAP_SCHEMA,
    )


def _ov(v: int) -> str:
    """Render an overhang bound the way the bench writes them (`inf:100`)."""
    return "inf" if v >= UNBOUNDED_OVERHANG else str(int(v))


__all__ = [
    "DEFAULT_MAX_CLUSTER_READ_FRAC",
    "DEFAULT_MIN_CHAIN_CLUSTER_READS",
    "DEFAULT_SEED_IDENTITY",
    "DEFAULT_SEED_MAX_3P",
    "DEFAULT_SEED_MAX_5P",
    "READ_CLUSTER_MAP_SCHEMA",
    "ChainedClusterError",
    "KmerSeedResult",
    "seed_by_kmer_clustering",
]
