"""Stage 1 — seeding: one template per distinct sense ORF.

Every read contributes its longest sense ORF (≥30 aa). Those ORF nucleotide
strings are dereplicated exactly, and each distinct ORF elects one read whose
**full cDNA** becomes the template — the frame the M-step's PWM is built on
and the sequence the E-step aligns against.

Two things about this that are easy to get backwards:

* **Replication is a certificate of error-free sequence, not of abundance.**
  The exact-copy count of an ORF scales as exp(−εL), so for a long protein the
  most-replicated exact ORF is usually a *fragment*. Abundance decides which
  ORF anchors a group; it must not decide between proteoforms of different
  length (that is the fold stage's ±3-codon rule).
* **No replication filter.** Akap4's 849-aa proteoform was error-free in 4
  reads and never twice identical, so any "seen ≥2×" gate erases it. Every
  read's ORF is a seed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import register_schema
from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
    _hash_sequences,
    dereplicate,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.elect import (
    REPRESENTATIVE_POLICIES,
    SEED_QUALITY_FLOOR,
    RepCandidates,
    RepresentativePolicy,
    best_quality_per_uniq,
    elect_representatives,
    resolve_policy,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import (
    predict_orfs_parallel,
)


# One row per distinct ORF nucleotide sequence.
SEED_ORF_TABLE: pa.Schema = pa.schema(
    [
        pa.field("orf_id", pa.int64(), nullable=False),
        pa.field("orf_nucleotide", pa.large_string(), nullable=False),
        pa.field("orf_aa_length", pa.int32(), nullable=False),
        # Reads whose longest sense ORF is byte-identical to this one.
        pa.field("n_reads", pa.int64(), nullable=False),
        # The elected representative and its full cDNA — the template.
        pa.field("representative_read_id", pa.string(), nullable=False),
        pa.field("template_sequence", pa.large_string(), nullable=False),
        pa.field("orf_start_in_template", pa.int32(), nullable=False),
        pa.field("orf_end_in_template", pa.int32(), nullable=False),
        pa.field("template_length", pa.int32(), nullable=False),
        # Distinct cDNAs carrying this exact ORF — a diagnostic on how much
        # UTR variation the ORF key is collapsing.
        pa.field("n_distinct_templates", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"SeedOrfTable"},
)

READ_ORF_MAP_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("orf_id", pa.int64(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=True),
    ]
)

register_schema("SeedOrfTable", SEED_ORF_TABLE)


def extract_seed_orfs(
    reads: pa.Table,
    *,
    min_aa_length: int = 30,
    representative: str | RepresentativePolicy = "median-length",
    threads: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Seed one template per distinct sense ORF.

    ``reads`` carries ``(read_id, sequence, sample_id)`` — the trimmed
    transcript windows. Returns ``(SEED_ORF_TABLE, READ_ORF_MAP_SCHEMA)``;
    reads with no qualifying ORF are simply absent from the map (they are not
    lost — the E-step still assigns them to whichever template wins).
    """
    log = progress or (lambda _m: None)
    # Resolve the policy BEFORE ORF prediction: an unknown name should fail in
    # a millisecond, not after an hour of `best_sense_orf` over 9M cDNAs.
    policy: RepresentativePolicy = resolve_policy(representative)

    # Predict once per distinct cDNA, not once per read: at ~1% error most
    # reads are distinct, but the collapse is free and never wrong.
    log(f"dereplicating {reads.num_rows:,} reads for ORF seeding…")
    uniq, read_map = dereplicate(reads)
    n_uniq = uniq.num_rows
    if n_uniq == 0:
        return SEED_ORF_TABLE.empty_table(), READ_ORF_MAP_SCHEMA.empty_table()
    uniq_quality, uniq_best_read = best_quality_per_uniq(reads, read_map, n_uniq)

    seqs = uniq.column("sequence").to_pylist()
    log(f"predicting sense ORFs (≥{min_aa_length} aa) over {n_uniq:,} unique cDNAs…")
    hits = predict_orfs_parallel(
        seqs, min_aa_length=min_aa_length, threads=threads
    )
    if not hits:
        log("no read carries a qualifying ORF")
        return SEED_ORF_TABLE.empty_table(), READ_ORF_MAP_SCHEMA.empty_table()

    row = np.array([h[0] for h in hits], dtype=np.int64)
    orf_start = np.array([h[1] for h in hits], dtype=np.int64)
    orf_end = np.array([h[2] for h in hits], dtype=np.int64)
    abundance = uniq.column("abundance").to_numpy(zero_copy_only=False)[row]
    tmpl_len = (
        uniq.column("seq_len").to_numpy(zero_copy_only=False)[row].astype(np.int64)
    )
    orf_nt = pa.array(
        [seqs[int(i)][int(a) : int(b)] for i, a, b in zip(row, orf_start, orf_end)],
        type=pa.large_string(),
    )

    # Group on a fixed-width hash rather than the ORF string itself, for the
    # same reason dereplicate does: Acero's variable-length distinct-key
    # storage overflows its int32 offset once the keys exceed ~2 GB.
    orf_hash = _hash_sequences(orf_nt).to_numpy(zero_copy_only=False)
    log(f"{len(hits):,} unique cDNAs carry an ORF; grouping by exact ORF…")

    # Densify the hash into an orf_id. `np.unique` sorts, so the ids are in
    # hash order — the same order the previous inline lexsort assigned them
    # in, which is what keeps `orf_id` stable across this refactor.
    _uniq_hash, orf_id = np.unique(orf_hash, return_inverse=True)
    orf_id = orf_id.astype(np.int64)
    n_groups = int(_uniq_hash.shape[0])
    n_templates = np.bincount(orf_id, minlength=n_groups).astype(np.int32)

    # The election itself is shared with the kmer seeder — same policy, same
    # tie-breaks, different group key (ORF hash here, read cluster there).
    rep_row, n_reads = elect_representatives(
        orf_id,
        n_groups,
        template_length=tmpl_len,
        abundance=abundance,
        quality=uniq_quality[row],
        orf_start=orf_start,
        policy=policy,
    )

    rep_uniq = row[rep_row]
    # Name the best-quality read of the elected cDNA rather than whichever
    # row dereplication happened to keep. The sequence is identical either
    # way — reads share an exact cDNA — so this costs the template nothing
    # and gives the round-1 ranker an accurate seed quality to break ties on.
    rep_read_id = pc.take(uniq.column("representative_read_id"), pa.array(rep_uniq))
    best_ids = uniq_best_read[rep_uniq]
    if best_ids is not None:
        rep_read_id = pc.if_else(
            pa.array(best_ids >= 0),
            pc.take(reads.column("read_id"), pa.array(np.maximum(best_ids, 0))),
            rep_read_id,
        )
    n_orfs = n_groups
    seed = pa.table(
        {
            "orf_id": pa.array(np.arange(n_orfs, dtype=np.int64)),
            "orf_nucleotide": pc.take(orf_nt, pa.array(rep_row)),
            "orf_aa_length": pa.array(
                ((orf_end[rep_row] - orf_start[rep_row]) // 3 - 1).astype(np.int32)
            ),
            "n_reads": pa.array(n_reads),
            "representative_read_id": rep_read_id,
            "template_sequence": pc.take(
                uniq.column("sequence"), pa.array(rep_uniq)
            ).cast(pa.large_string()),
            "orf_start_in_template": pa.array(orf_start[rep_row].astype(np.int32)),
            "orf_end_in_template": pa.array(orf_end[rep_row].astype(np.int32)),
            "template_length": pa.array(tmpl_len[rep_row].astype(np.int32)),
            "n_distinct_templates": pa.array(n_templates),
        },
        schema=SEED_ORF_TABLE,
    )

    # read → orf_id, via the uniq_id each read dereplicated to.
    uniq_to_orf = pa.table(
        {
            "uniq_id": pa.array(row, type=pa.int64()),
            "orf_id": pa.array(orf_id),
        }
    )
    mapped = read_map.join(uniq_to_orf, keys="uniq_id", join_type="inner")
    read_orf = mapped.select(["read_id", "orf_id", "sample_id"]).cast(
        READ_ORF_MAP_SCHEMA
    )
    log(f"seeded {n_orfs:,} distinct ORFs over {read_orf.num_rows:,} reads")
    return seed, read_orf


__all__ = [
    "SEED_ORF_TABLE",
    "SEED_QUALITY_FLOOR",
    "READ_ORF_MAP_SCHEMA",
    "RepCandidates",
    "REPRESENTATIVE_POLICIES",
    "extract_seed_orfs",
]
