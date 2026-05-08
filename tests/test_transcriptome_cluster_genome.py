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
