"""Tests for ``constellation.sequencing.transcriptome.cluster_genome``.

Validate the Phase 2 fingerprint-keyed genome-guided clustering:

    1. Reads with the same fingerprint collapse to one cluster.
    2. Reads with different fingerprints get separate clusters.
    3. Drift filter drops reads whose 5' or 3' span deviates from the
       cluster median by more than the threshold.
    4. Drift filter is strand-aware: on '-', 5' is span_end.
    5. Singletons skip the drift filter (no median to compare).
    6. Layer-0 derep picks the most-abundant unique sequence as the
       representative; ties on abundance break on dorado_quality.
    7. Layer-0 duplicates get role='duplicate'; non-duplicates get
       role='member'.
    8. min_cluster_size drops whole clusters below the threshold,
       including their drift_filtered tails.
    9. cluster_id is sequential int64 starting from cluster_id_seed.
   10. match_rate / indel_rate are populated from cs:long-aware blocks
       and null when only CIGAR-derived blocks are available.
   11. drop_drift_filtered=True elides the drift_filtered rows entirely.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    READ_FINGERPRINT_TABLE,
)
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)
from constellation.sequencing.transcriptome.cluster_genome import (
    cluster_by_fingerprint,
)


def _alignment(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "read_id": "r0",
        "acquisition_id": 1,
        "ref_name": "chr1",
        "ref_start": 0,
        "ref_end": 100,
        "strand": "+",
        "mapq": 60,
        "flag": 0,
        "cigar_string": "100M",
        "nm_tag": None,
        "as_tag": None,
        "read_group": None,
        "is_secondary": False,
        "is_supplementary": False,
    }
    base.update(overrides)
    return base


def _fp(**overrides) -> dict:
    base = {
        "read_id": "r0",
        "contig_id": 1,
        "strand": "+",
        "n_blocks": 1,
        "span_start": 0,
        "span_end": 100,
        "fingerprint_hash": 100,
        "junction_signature": "chr1:.",
    }
    base.update(overrides)
    return base


def _block(**overrides) -> dict:
    base = {
        "alignment_id": 0,
        "block_index": 0,
        "ref_start": 0,
        "ref_end": 100,
        "query_start": 0,
        "query_end": 100,
        "n_match": 100,
        "n_mismatch": 0,
        "n_insert": 0,
        "n_delete": 0,
    }
    base.update(overrides)
    return base


def _alignments(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE)


def _fingerprints(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=READ_FINGERPRINT_TABLE)


def _blocks(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE)


def _reads(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "read_id": pa.array([r["read_id"] for r in rows]),
            "sequence": pa.array([r["sequence"] for r in rows]),
            "dorado_quality": pa.array(
                [r.get("dorado_quality") for r in rows], type=pa.float32()
            ),
        }
    )


# ──────────────────────────────────────────────────────────────────────


def test_same_fingerprint_collapses_to_one_cluster() -> None:
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100),
            _fp(read_id="rB", fingerprint_hash=100),
            _fp(read_id="rC", fingerprint_hash=100),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "AAAA", "dorado_quality": 20.0},
        ]
    )
    clusters, membership = cluster_by_fingerprint(
        fps, rd, alignments=al
    )
    assert clusters.num_rows == 1
    assert clusters.column("n_reads").to_pylist() == [3]
    assert membership.num_rows == 3


def test_different_fingerprints_split_into_separate_clusters() -> None:
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100),
            _fp(read_id="rB", fingerprint_hash=200),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "TTTT", "dorado_quality": 20.0},
        ]
    )
    clusters, _ = cluster_by_fingerprint(fps, rd, alignments=al)
    assert clusters.num_rows == 2


def test_drift_filter_drops_5p_outlier() -> None:
    """A read whose 5' span deviates from cluster median by > max_5p_drift
    is retained with role='drift_filtered' (not dropped from output)."""
    fps = _fingerprints(
        [
            _fp(read_id="rA", span_start=0, span_end=1000),
            _fp(read_id="rB", span_start=5, span_end=1000),
            _fp(read_id="rC", span_start=100, span_end=1000),  # 5p drift
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "AAAA", "dorado_quality": 20.0},
        ]
    )
    clusters, membership = cluster_by_fingerprint(
        fps, rd, alignments=al, max_5p_drift=25
    )
    # n_reads counts kept members only (rC is drift_filtered)
    assert clusters.column("n_reads").to_pylist() == [2]
    # Membership has all 3 rows, with rC tagged
    roles = {
        r["read_id"]: r["role"] for r in membership.to_pylist()
    }
    assert roles["rC"] == "drift_filtered"


def test_drift_filter_strand_aware() -> None:
    """On '-' strand, 5' is span_end and 3' is span_start. A read whose
    span_end deviates large gets flagged as 5'-drift, not 3'-drift."""
    fps = _fingerprints(
        [
            _fp(read_id="rA", strand="-", span_start=0, span_end=1000),
            _fp(read_id="rB", strand="-", span_start=0, span_end=1005),
            _fp(read_id="rC", strand="-", span_start=0, span_end=1100),
            # rC's 5' (=span_end) drifts by ~100 bp from median.
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA", strand="-"),
            _alignment(alignment_id=2, read_id="rB", strand="-"),
            _alignment(alignment_id=3, read_id="rC", strand="-"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "AAAA", "dorado_quality": 20.0},
        ]
    )
    clusters, membership = cluster_by_fingerprint(
        fps, rd, alignments=al, max_5p_drift=25, max_3p_drift=75
    )
    roles = {r["read_id"]: r["role"] for r in membership.to_pylist()}
    assert roles["rC"] == "drift_filtered"
    # rC drift_5p_bp should be ~95, NOT drift_3p_bp.
    rc_row = next(
        r for r in membership.to_pylist()
        if r["read_id"] == "rC"
    )
    assert rc_row["drift_5p_bp"] is not None
    assert abs(rc_row["drift_5p_bp"]) > 25


def test_singleton_skips_drift_filter() -> None:
    fps = _fingerprints([_fp(read_id="rA", span_start=0, span_end=1000)])
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    rd = _reads(
        [{"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0}]
    )
    clusters, membership = cluster_by_fingerprint(fps, rd, alignments=al)
    assert clusters.num_rows == 1
    assert clusters.column("n_reads").to_pylist() == [1]
    assert membership.num_rows == 1
    assert membership.column("role").to_pylist() == ["representative"]


def test_layer0_derep_picks_most_abundant_unique_sequence() -> None:
    """rA + rB share sequence; rC has a different one. Representative
    should be rA (or rB, but the chosen exemplar is the first encountered
    with the winning hash)."""
    fps = _fingerprints(
        [
            _fp(read_id="rA"),
            _fp(read_id="rB"),
            _fp(read_id="rC"),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 22.0},
            {"read_id": "rC", "sequence": "TTTT", "dorado_quality": 25.0},
        ]
    )
    clusters, membership = cluster_by_fingerprint(fps, rd, alignments=al)
    rep = clusters.column("representative_read_id").to_pylist()[0]
    assert rep in {"rA", "rB"}
    n_unique = clusters.column("n_unique_sequences").to_pylist()[0]
    assert n_unique == 2
    roles = {r["read_id"]: r["role"] for r in membership.to_pylist()}
    # Whoever wasn't picked as rep but shared the sequence becomes a duplicate.
    other = "rB" if rep == "rA" else "rA"
    assert roles[other] == "duplicate"
    assert roles["rC"] == "member"
    assert roles[rep] == "representative"


def test_layer0_tiebreak_on_quality() -> None:
    """Two reads with two unique sequences, each with abundance 1. Tie
    breaks on mean dorado_quality."""
    fps = _fingerprints(
        [_fp(read_id="rA"), _fp(read_id="rB")]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 15.0},
            {"read_id": "rB", "sequence": "TTTT", "dorado_quality": 25.0},
        ]
    )
    clusters, _ = cluster_by_fingerprint(fps, rd, alignments=al)
    rep = clusters.column("representative_read_id").to_pylist()[0]
    assert rep == "rB"


def test_min_cluster_size_drops_small_clusters() -> None:
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100),
            _fp(read_id="rB", fingerprint_hash=100),
            _fp(read_id="rC", fingerprint_hash=200),  # singleton
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "TTTT", "dorado_quality": 20.0},
        ]
    )
    clusters, membership = cluster_by_fingerprint(
        fps, rd, alignments=al, min_cluster_size=2
    )
    # rC's singleton cluster drops; only the rA/rB cluster remains.
    assert clusters.num_rows == 1
    assert clusters.column("n_reads").to_pylist() == [2]
    member_reads = set(membership.column("read_id").to_pylist())
    assert "rC" not in member_reads


def test_cluster_id_is_sequential_from_seed() -> None:
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100),
            _fp(read_id="rB", fingerprint_hash=200),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "TTTT", "dorado_quality": 20.0},
        ]
    )
    clusters, _ = cluster_by_fingerprint(
        fps, rd, alignments=al, cluster_id_seed=1000
    )
    cids = sorted(clusters.column("cluster_id").to_pylist())
    assert cids == [1000, 1001]


def test_match_rate_populated_from_cs_aware_blocks() -> None:
    fps = _fingerprints([_fp(read_id="rA")])
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(
                alignment_id=1,
                n_match=95, n_mismatch=5, n_insert=0, n_delete=0
            )
        ]
    )
    rd = _reads(
        [{"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0}]
    )
    _, membership = cluster_by_fingerprint(
        fps, rd, alignments=al, alignment_blocks=blocks
    )
    row = membership.to_pylist()[0]
    assert row["match_rate"] is not None
    assert abs(row["match_rate"] - 0.95) < 1e-6
    assert row["indel_rate"] == 0.0
    assert row["n_aligned_bp"] == 100


def test_match_rate_null_when_cs_unavailable() -> None:
    fps = _fingerprints([_fp(read_id="rA")])
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    blocks = _blocks(
        [
            _block(
                alignment_id=1,
                n_match=None, n_mismatch=None, n_insert=0, n_delete=0
            )
        ]
    )
    rd = _reads(
        [{"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0}]
    )
    _, membership = cluster_by_fingerprint(
        fps, rd, alignments=al, alignment_blocks=blocks
    )
    row = membership.to_pylist()[0]
    assert row["match_rate"] is None
    assert row["indel_rate"] is None


def test_drop_drift_filtered_elides_them() -> None:
    fps = _fingerprints(
        [
            _fp(read_id="rA", span_start=0, span_end=1000),
            _fp(read_id="rB", span_start=5, span_end=1000),
            _fp(read_id="rC", span_start=200, span_end=1000),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "AAAA", "dorado_quality": 20.0},
        ]
    )
    _, membership = cluster_by_fingerprint(
        fps, rd, alignments=al, max_5p_drift=25, drop_drift_filtered=True
    )
    member_reads = set(membership.column("read_id").to_pylist())
    assert "rC" not in member_reads
    # Two surviving members of the cluster.
    assert membership.num_rows == 2


def test_empty_inputs_return_empty_schema_shaped_tables() -> None:
    fps = READ_FINGERPRINT_TABLE.empty_table()
    al = ALIGNMENT_TABLE.empty_table()
    rd = pa.table(
        {
            "read_id": pa.array([], type=pa.string()),
            "sequence": pa.array([], type=pa.string()),
            "dorado_quality": pa.array([], type=pa.float32()),
        }
    )
    clusters, membership = cluster_by_fingerprint(fps, rd, alignments=al)
    assert clusters.num_rows == 0
    assert membership.num_rows == 0
    assert clusters.schema.equals(TRANSCRIPT_CLUSTER_TABLE)
    assert membership.schema.equals(CLUSTER_MEMBERSHIP_TABLE)


def test_mode_field_is_genome_guided() -> None:
    fps = _fingerprints([_fp(read_id="rA")])
    al = _alignments([_alignment(alignment_id=1, read_id="rA")])
    rd = _reads(
        [{"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0}]
    )
    clusters, _ = cluster_by_fingerprint(fps, rd, alignments=al)
    assert clusters.column("mode").to_pylist() == ["genome-guided"]


def test_index_then_take_handles_reordered_sources() -> None:
    """The narrow-sort refactor resolves per-fingerprint row indices via
    ``pc.index_in`` so source tables (reads, primary_lookup, metrics) can
    arrive in any row order — the per-cluster ``pc.take`` must pull the
    right row from each source regardless of how it was sorted upstream.

    Build inputs where reads is sorted by sequence content and the
    alignments table is reverse-sorted relative to fingerprints; the
    cluster outputs must still match what we'd get with everything in
    the same order.
    """
    # 6 reads across 2 fingerprints (3 each).
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100,
                contig_id=1, span_start=1000, span_end=2000),
            _fp(read_id="rB", fingerprint_hash=100,
                contig_id=1, span_start=1000, span_end=2000),
            _fp(read_id="rC", fingerprint_hash=100,
                contig_id=1, span_start=1000, span_end=2000),
            _fp(read_id="rD", fingerprint_hash=200,
                contig_id=2, span_start=500, span_end=1500),
            _fp(read_id="rE", fingerprint_hash=200,
                contig_id=2, span_start=500, span_end=1500),
            _fp(read_id="rF", fingerprint_hash=200,
                contig_id=2, span_start=500, span_end=1500),
        ]
    )
    # alignments REVERSE-sorted relative to fingerprints to ensure the
    # index_in path resolves the right alignment per fingerprint.
    al = _alignments(
        [
            _alignment(alignment_id=6, read_id="rF", ref_name="chr2"),
            _alignment(alignment_id=5, read_id="rE", ref_name="chr2"),
            _alignment(alignment_id=4, read_id="rD", ref_name="chr2"),
            _alignment(alignment_id=3, read_id="rC"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=1, read_id="rA"),
        ]
    )
    # reads sorted alphabetically by SEQUENCE content (forces a different
    # ordering than fingerprints' read_id order).
    rd = _reads(
        [
            {"read_id": "rE", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAC", "dorado_quality": 20.0},
            {"read_id": "rF", "sequence": "CCCC", "dorado_quality": 20.0},
            {"read_id": "rA", "sequence": "GGGA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "GGGT", "dorado_quality": 20.0},
            {"read_id": "rD", "sequence": "TTTT", "dorado_quality": 20.0},
        ]
    )
    clusters, membership = cluster_by_fingerprint(fps, rd, alignments=al)
    assert clusters.num_rows == 2
    # Each cluster has 3 members regardless of source ordering.
    n_reads_by_cluster = sorted(clusters.column("n_reads").to_pylist())
    assert n_reads_by_cluster == [3, 3]
    # Membership rows mention all 6 reads exactly once.
    member_read_ids = sorted(
        r["read_id"] for r in membership.to_pylist()
    )
    assert member_read_ids == ["rA", "rB", "rC", "rD", "rE", "rF"]
    # contig_id per cluster matches: hash=100 → chr1 (contig_id 1);
    # hash=200 → chr2 (contig_id 2). The hash→contig mapping must
    # survive the narrow-sort path even though sources are scrambled.
    by_hash = {
        r["fingerprint_hash"]: r["contig_id"] for r in clusters.to_pylist()
    }
    assert by_hash[100] == 1
    assert by_hash[200] == 2


def test_alignment_cs_accepts_table_or_dataset() -> None:
    """``alignment_cs`` accepts either ``pa.Table`` (eager — what tests
    and small Jupyter use pass) or ``pa.dataset.Dataset`` (streaming —
    what the CLI passes at PromethION scale, to avoid materialising
    the ~50 GB cs:long table). Both code paths must produce the same
    consensus sequence for the same inputs.
    """
    import tempfile
    from pathlib import Path

    import pyarrow.dataset as pa_dataset
    import pyarrow.parquet as pq

    from constellation.sequencing.reference.reference import GenomeReference
    from constellation.sequencing.schemas.alignment import ALIGNMENT_CS_TABLE
    from constellation.sequencing.schemas.reference import CONTIG_TABLE

    # Genome: chr1 with the reference window all 'A' so any cs:long ":N"
    # (identity) reproduces 'A' × N.
    ref_seq = "A" * 200
    contigs = pa.Table.from_pylist(
        [{"contig_id": 1, "name": "chr1", "length": 200,
          "topology": None, "circular": None}],
        schema=CONTIG_TABLE,
    )
    sequences = pa.Table.from_pylist(
        [{"contig_id": 1, "sequence": ref_seq}]
    )
    genome = GenomeReference(contigs=contigs, sequences=sequences)

    # Three members of a cluster, all aligning identity to the reference.
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100,
                contig_id=1, span_start=10, span_end=30),
            _fp(read_id="rB", fingerprint_hash=100,
                contig_id=1, span_start=10, span_end=30),
            _fp(read_id="rC", fingerprint_hash=100,
                contig_id=1, span_start=10, span_end=30),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA",
                       ref_start=10, ref_end=30),
            _alignment(alignment_id=2, read_id="rB",
                       ref_start=10, ref_end=30),
            _alignment(alignment_id=3, read_id="rC",
                       ref_start=10, ref_end=30),
        ]
    )
    blocks = _blocks(
        [
            _block(alignment_id=1, ref_start=10, ref_end=30,
                   n_match=20, n_mismatch=0, n_insert=0, n_delete=0),
            _block(alignment_id=2, ref_start=10, ref_end=30,
                   n_match=20, n_mismatch=0, n_insert=0, n_delete=0),
            _block(alignment_id=3, ref_start=10, ref_end=30,
                   n_match=20, n_mismatch=0, n_insert=0, n_delete=0),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "A" * 20, "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "A" * 20, "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "A" * 20, "dorado_quality": 20.0},
        ]
    )
    cs_table = pa.Table.from_pylist(
        [
            {"alignment_id": 1, "cs_string": ":20"},
            {"alignment_id": 2, "cs_string": ":20"},
            {"alignment_id": 3, "cs_string": ":20"},
        ],
        schema=ALIGNMENT_CS_TABLE,
    )

    # Path 1: alignment_cs as pa.Table.
    clusters_a, _ = cluster_by_fingerprint(
        fps, rd,
        alignments=al,
        alignment_blocks=blocks,
        alignment_cs=cs_table,
        genome=genome,
        build_consensus_seq=True,
    )

    # Path 2: alignment_cs as pa.dataset.Dataset (write to temp parquet,
    # reopen as dataset, pass through).
    with tempfile.TemporaryDirectory() as td:
        cs_dir = Path(td) / "alignment_cs"
        cs_dir.mkdir()
        pq.write_table(cs_table, cs_dir / "part-00000.parquet")
        cs_dataset = pa_dataset.dataset(cs_dir)
        clusters_b, _ = cluster_by_fingerprint(
            fps, rd,
            alignments=al,
            alignment_blocks=blocks,
            alignment_cs=cs_dataset,
            genome=genome,
            build_consensus_seq=True,
        )

    consensus_a = clusters_a.column("consensus_sequence").to_pylist()
    consensus_b = clusters_b.column("consensus_sequence").to_pylist()
    assert consensus_a == consensus_b
    # Non-trivial: consensus actually populated, not just both None.
    assert consensus_a[0] is not None
    assert len(consensus_a[0]) > 0


def test_per_sample_clusters_partition_by_sample_id() -> None:
    """``per_sample_clusters=True`` partitions clusters by sample_id;
    the read_to_sample table is resolved via the pre-indexed lookup.
    """
    fps = _fingerprints(
        [
            _fp(read_id="rA", fingerprint_hash=100),
            _fp(read_id="rB", fingerprint_hash=100),
            _fp(read_id="rC", fingerprint_hash=100),
        ]
    )
    al = _alignments(
        [
            _alignment(alignment_id=1, read_id="rA"),
            _alignment(alignment_id=2, read_id="rB"),
            _alignment(alignment_id=3, read_id="rC"),
        ]
    )
    rd = _reads(
        [
            {"read_id": "rA", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rB", "sequence": "AAAA", "dorado_quality": 20.0},
            {"read_id": "rC", "sequence": "AAAA", "dorado_quality": 20.0},
        ]
    )
    # rA + rB share sample 7, rC is in sample 13.
    read_to_sample = pa.table(
        {
            "read_id": pa.array(["rA", "rB", "rC"]),
            "sample_id": pa.array([7, 7, 13], type=pa.int64()),
        }
    )
    clusters, _ = cluster_by_fingerprint(
        fps, rd, alignments=al,
        read_to_sample=read_to_sample,
        per_sample_clusters=True,
    )
    # Same fingerprint but split across two samples → 2 clusters.
    assert clusters.num_rows == 2
    sids = sorted(clusters.column("sample_id").to_pylist())
    assert sids == [7, 13]
    sizes_by_sample = {
        r["sample_id"]: r["n_reads"] for r in clusters.to_pylist()
    }
    assert sizes_by_sample[7] == 2
    assert sizes_by_sample[13] == 1
