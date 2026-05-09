"""Tests for the 5 kernels added on top of the first vertical slice.

Covers gene_annotation, reference_sequence, splice_junctions,
read_pileup (vector + hybrid), cluster_pileup (vector). The hybrid path
imports datashader, which ships in the ``[viz]`` extras — these tests
gate on the import.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_BLOCK_TABLE,
    ALIGNMENT_TABLE,
    INTRON_TABLE,
)
from constellation.sequencing.schemas.reference import (
    CONTIG_TABLE,
    FEATURE_TABLE,
    SEQUENCE_TABLE,
)
from constellation.sequencing.schemas.transcriptome import (
    CLUSTER_MEMBERSHIP_TABLE,
    TRANSCRIPT_CLUSTER_TABLE,
)
from constellation.viz.server.session import Session
from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackQuery,
    get_kernel,
)
from constellation.viz.tracks.cluster_pileup import CLUSTER_PILEUP_VECTOR_SCHEMA
from constellation.viz.tracks.gene_annotation import (
    GENE_ANNOTATION_VECTOR_SCHEMA,
)
from constellation.viz.tracks.read_pileup import READ_PILEUP_VECTOR_SCHEMA
from constellation.viz.tracks.reference_sequence import (
    REFERENCE_SEQUENCE_VECTOR_SCHEMA,
)
from constellation.viz.tracks.splice_junctions import (
    SPLICE_JUNCTIONS_VECTOR_SCHEMA,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _write_genome(genome_dir: Path) -> None:
    genome_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contig_id": 1,
                    "name": "chr1",
                    "length": 1_000_000,
                    "topology": None,
                    "circular": None,
                }
            ],
            schema=CONTIG_TABLE,
        ),
        genome_dir / "contigs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"contig_id": 1, "sequence": "ACGTACGTACGTACGT" * 10}],
            schema=SEQUENCE_TABLE,
        ),
        genome_dir / "sequences.parquet",
    )


def _write_annotation(annotation_dir: Path, features: list[dict]) -> None:
    annotation_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(features, schema=FEATURE_TABLE),
        annotation_dir / "features.parquet",
    )


# ----------------------------------------------------------------------
# Reference sequence
# ----------------------------------------------------------------------


def test_reference_sequence_vector_returns_per_base(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    session = Session.from_root(root)
    kernel = get_kernel("reference_sequence")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=10, viewport_px=400)
    mode = kernel.threshold(binding, query)
    assert mode is ThresholdDecision.VECTOR
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, mode)),
        schema=REFERENCE_SEQUENCE_VECTOR_SCHEMA,
    )
    assert table.num_rows == 10
    assert table.column("step").to_pylist() == [1] * 10
    bases = table.column("base").to_pylist()
    assert bases == list("ACGTACGTAC")


def test_reference_sequence_decimates_when_window_exceeds_cap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    genome = root / "genome"
    genome.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contig_id": 1,
                    "name": "chr1",
                    "length": 100_000,
                    "topology": None,
                    "circular": None,
                }
            ],
            schema=CONTIG_TABLE,
        ),
        genome / "contigs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"contig_id": 1, "sequence": "A" * 50_000}],
            schema=SEQUENCE_TABLE,
        ),
        genome / "sequences.parquet",
    )
    session = Session.from_root(root)
    kernel = get_kernel("reference_sequence")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=50_000, viewport_px=1200)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=REFERENCE_SEQUENCE_VECTOR_SCHEMA,
    )
    # window 50000 / cap 5000 → step 10
    assert table.num_rows <= kernel.vector_glyph_limit
    steps = table.column("step").to_pylist()
    assert max(steps) > 1


# ----------------------------------------------------------------------
# Gene annotation
# ----------------------------------------------------------------------


def _feature(**kwargs) -> dict:
    base = {
        "feature_id": 1,
        "contig_id": 1,
        "start": 0,
        "end": 100,
        "strand": "+",
        "type": "gene",
        "name": "geneA",
        "parent_id": None,
        "source": "RefSeq",
        "score": None,
        "phase": None,
        "attributes_json": None,
    }
    base.update(kwargs)
    return base


def test_gene_annotation_discover_uses_reference_when_present(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_annotation(root / "annotation", [_feature()])
    session = Session.from_root(root)
    bindings = get_kernel("gene_annotation").discover(session)
    assert len(bindings) == 1
    assert bindings[0].binding_id == "reference"


def test_gene_annotation_discover_returns_both_when_both_present(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_annotation(root / "annotation", [_feature()])
    _write_annotation(
        root / "S2_align" / "derived_annotation",
        [_feature(feature_id=10, source="constellation_derived")],
    )
    session = Session.from_root(root)
    bindings = get_kernel("gene_annotation").discover(session)
    assert {b.binding_id for b in bindings} == {"reference", "derived"}


def test_gene_annotation_fetch_filters_by_window(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_annotation(
        root / "annotation",
        [
            _feature(feature_id=1, start=0, end=100, name="A"),
            _feature(feature_id=2, start=200, end=300, name="B"),
            _feature(feature_id=3, start=500, end=600, name="C"),
        ],
    )
    session = Session.from_root(root)
    kernel = get_kernel("gene_annotation")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=50, end=400)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=GENE_ANNOTATION_VECTOR_SCHEMA,
    )
    names = sorted(table.column("name").to_pylist())
    assert names == ["A", "B"]


def test_gene_annotation_truncates_above_feature_limit(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    # 10 short features + 5 long features in the same window. With a
    # cap of 8, the kernel should keep the longest 8.
    features = []
    for i in range(10):
        features.append(_feature(feature_id=i, start=10 + i, end=20 + i, name=f"short{i}"))
    for i in range(5):
        features.append(
            _feature(
                feature_id=100 + i,
                start=10 + i,
                end=10_000 + i,
                name=f"long{i}",
            )
        )
    _write_annotation(root / "annotation", features)
    session = Session.from_root(root)
    kernel = get_kernel("gene_annotation")
    kernel.feature_limit = 8  # tune for this test only
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=20_000)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=GENE_ANNOTATION_VECTOR_SCHEMA,
    )
    names = set(table.column("name").to_pylist())
    # All 5 longs should be kept (longest 8 of 15 = 5 longs + 3 shorts)
    assert {f"long{i}" for i in range(5)}.issubset(names)
    kernel.feature_limit = 2_000  # restore class default


# ----------------------------------------------------------------------
# Splice junctions
# ----------------------------------------------------------------------


def test_splice_junctions_emits_one_arc_per_cluster(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    align_dir = root / "S2_align"
    align_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                # Cluster 1: seed at (1000, 2000) + a near neighbor
                {
                    "intron_id": 1,
                    "contig_id": 1,
                    "strand": "+",
                    "donor_pos": 1000,
                    "acceptor_pos": 2000,
                    "read_count": 50,
                    "motif": "GT-AG",
                    "is_intron_seed": True,
                    "annotated": True,
                },
                {
                    "intron_id": 1,
                    "contig_id": 1,
                    "strand": "+",
                    "donor_pos": 1002,
                    "acceptor_pos": 2003,
                    "read_count": 5,
                    "motif": "GT-AG",
                    "is_intron_seed": False,
                    "annotated": False,
                },
                # Cluster 2: seed at (5000, 6000)
                {
                    "intron_id": 2,
                    "contig_id": 1,
                    "strand": "+",
                    "donor_pos": 5000,
                    "acceptor_pos": 6000,
                    "read_count": 30,
                    "motif": "GC-AG",
                    "is_intron_seed": True,
                    "annotated": False,
                },
            ],
            schema=INTRON_TABLE,
        ),
        align_dir / "introns.parquet",
    )
    session = Session.from_root(root)
    kernel = get_kernel("splice_junctions")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=10_000)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=SPLICE_JUNCTIONS_VECTOR_SCHEMA,
    )
    assert table.num_rows == 2
    rows_by_id = {r["intron_id"]: r for r in table.to_pylist()}
    # Cluster 1 support sums member counts (50 + 5 = 55)
    assert rows_by_id[1]["support"] == 55
    assert rows_by_id[1]["donor_pos"] == 1000  # seed coordinates
    assert rows_by_id[2]["support"] == 30


# ----------------------------------------------------------------------
# Read pileup — vector + hybrid
# ----------------------------------------------------------------------


def _write_alignments(align_dir: Path, rows: list[dict]) -> None:
    align_dir.mkdir(parents=True, exist_ok=True)
    aln_partitioned = align_dir / "alignments"
    aln_partitioned.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=ALIGNMENT_TABLE),
        aln_partitioned / "part-00000.parquet",
    )


def _alignment(**kwargs) -> dict:
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
    base.update(kwargs)
    return base


def test_read_pileup_vector_returns_packed_rows(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_alignments(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r2", ref_start=50, ref_end=150),
            _alignment(alignment_id=3, read_id="r3", ref_start=200, ref_end=300),
        ],
    )
    session = Session.from_root(root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
    mode = kernel.threshold(binding, query)
    assert mode is ThresholdDecision.VECTOR
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, mode)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    assert table.num_rows == 3
    rows = {row["alignment_id"]: row["row"] for row in table.to_pylist()}
    # r1 occupies row 0; r2 overlaps r1 → row 1; r3 doesn't overlap → row 0 reused
    assert rows[1] == 0
    assert rows[2] == 1
    assert rows[3] == 0


def test_read_pileup_hybrid_emits_png(tmp_path: Path) -> None:
    pytest.importorskip("datashader")
    pytest.importorskip("PIL")
    from PIL import Image

    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_alignments(
        root / "S2_align",
        [_alignment(alignment_id=i, read_id=f"r{i}", ref_start=0, ref_end=100) for i in range(50)],
    )
    session = Session.from_root(root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    # Force hybrid via the explicit param so we don't depend on
    # threshold tuning.
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=200,
        viewport_px=1200,
        force=ThresholdDecision.HYBRID,
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.HYBRID)),
        schema=HYBRID_SCHEMA,
    )
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["mode"] == "hybrid"
    assert row["n_items"] == 50
    img = Image.open(io.BytesIO(row["png_bytes"]))
    assert img.size == (1200, row["height_px"])


def test_read_pileup_threshold_force_override(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_alignments(root / "S2_align", [_alignment()])
    session = Session.from_root(root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    q = TrackQuery(
        contig="chr1",
        start=0,
        end=200,
        viewport_px=1200,
        force=ThresholdDecision.HYBRID,
    )
    assert kernel.threshold(binding, q) is ThresholdDecision.HYBRID


# ----------------------------------------------------------------------
# Cluster pileup
# ----------------------------------------------------------------------


def _cluster(**kwargs) -> dict:
    base = {
        "cluster_id": 1,
        "representative_read_id": "r0",
        "n_reads": 5,
        "identity_threshold": None,
        "consensus_sequence": None,
        "predicted_protein": None,
        "orf_start": None,
        "orf_end": None,
        "orf_strand": None,
        "codon_table": None,
        "mode": "genome-guided",
        "contig_id": 1,
        "strand": "+",
        "span_start": 0,
        "span_end": 100,
        "fingerprint_hash": 0,
        "n_unique_sequences": 1,
        "sample_id": -1,
    }
    base.update(kwargs)
    return base


def test_cluster_pileup_vector_emits_packed_rows(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    cluster_dir = root / "S2_cluster"
    cluster_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                _cluster(cluster_id=1, span_start=0, span_end=100),
                _cluster(cluster_id=2, span_start=50, span_end=150),
                _cluster(cluster_id=3, span_start=200, span_end=300, n_reads=2),
            ],
            schema=TRANSCRIPT_CLUSTER_TABLE,
        ),
        cluster_dir / "clusters.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [],
            schema=CLUSTER_MEMBERSHIP_TABLE,
        ),
        cluster_dir / "cluster_membership.parquet",
    )
    session = Session.from_root(root)
    kernel = get_kernel("cluster_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
    mode = kernel.threshold(binding, query)
    assert mode is ThresholdDecision.VECTOR
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, mode)),
        schema=CLUSTER_PILEUP_VECTOR_SCHEMA,
    )
    assert table.num_rows == 3
    rows = {row["cluster_id"]: row for row in table.to_pylist()}
    assert rows[1]["row"] == 0
    assert rows[2]["row"] == 1
    assert rows[3]["row"] == 0


# ----------------------------------------------------------------------
# datashader helper
# ----------------------------------------------------------------------


def test_greedy_row_assign_basic() -> None:
    from constellation.viz.raster.datashader_png import greedy_row_assign

    rows = greedy_row_assign(
        starts=[0, 50, 200, 60],
        ends=[100, 150, 300, 70],
    )
    # Reads sorted by start: 0->0(end100), 50->row1(50<100), 60->row2(60<100,60<150 -> row2), 200->row0(reuses)
    # Row layout sorts internally; just check that overlapping reads get distinct rows
    assert len(set(rows)) >= 3


def test_rasterize_segments_returns_png_bytes() -> None:
    pytest.importorskip("datashader")
    from constellation.viz.raster.datashader_png import rasterize_segments

    png = rasterize_segments(
        starts=[0, 50, 100],
        ends=[40, 90, 200],
        rows=[0, 1, 0],
        x_range=(0, 200),
        n_rows=2,
        width_px=200,
        height_px=40,
    )
    assert png.startswith(b"\x89PNG")


def test_rasterize_segments_empty_input_returns_blank_png() -> None:
    pytest.importorskip("datashader")
    from constellation.viz.raster.datashader_png import rasterize_segments

    png = rasterize_segments(
        starts=[],
        ends=[],
        rows=[],
        x_range=(0, 100),
        n_rows=1,
        width_px=100,
        height_px=20,
    )
    assert png.startswith(b"\x89PNG")
