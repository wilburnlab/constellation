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
    ALIGNMENT_CS_TABLE,
    ALIGNMENT_TABLE,
    INTRON_TABLE,
    READ_SAMPLE_TABLE,
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
from constellation.sequencing.transcriptome.manifest import (
    write_align_manifest,
    write_cluster_manifest,
)
from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackQuery,
    get_kernel,
)
from _viz_fixtures import DEFAULT_ASSEMBLY, DEFAULT_HANDLE, install_fake_reference
from constellation.viz.tracks.cluster_pileup import (
    CLUSTER_MEMBER_VECTOR_SCHEMA,
    CLUSTER_PILEUP_VECTOR_SCHEMA,
)
from constellation.viz.tracks.gene_annotation import (
    GENE_ANNOTATION_VECTOR_SCHEMA,
)
from constellation.viz.tracks.read_pileup import READ_PILEUP_VECTOR_SCHEMA
from constellation.viz.tracks import reference_sequence as ref_seq_mod
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
    """No-op wrapper retained for test-body compatibility. The actual
    reference is installed into the cache via `_install_ref` per-test."""
    return None


def _write_annotation(annotation_dir: Path, features: list[dict]) -> None:
    """Stash the features list on the parent dir so `_make_session` can
    later install them into the reference cache (or, when the dir name
    is `derived_annotation`, into the align source)."""
    annotation_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(features, schema=FEATURE_TABLE),
        annotation_dir / "features.parquet",
    )


def _install_ref(monkeypatch, tmp_path: Path, *, with_annotation_features=None) -> Path:
    """Install the standard 1M-bp chr1 reference into a fake cache and
    point ``CONSTELLATION_REFERENCES_HOME`` at it. Returns the release
    dir for callers that want to wire up annotation paths manually."""
    cache_root = tmp_path / "refs"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return install_fake_reference(
        cache_root,
        handle=DEFAULT_HANDLE,
        assembly_accession=DEFAULT_ASSEMBLY,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 1_000_000,
                "topology": None,
                "circular": None,
            }
        ],
        sequences=[{"contig_id": 1, "sequence": "ACGTACGTACGTACGT" * 10}],
        features=with_annotation_features,
    )


def _make_session(
    monkeypatch,
    tmp_path: Path,
    root: Path,
    *,
    use_root_annotation: bool = False,
) -> Session:
    """Compose a Session from a test-written ``<root>/{genome,annotation,
    S2_align,S2_cluster}/...`` layout — the legacy fixture shape — by
    rerouting genome/annotation into the reference cache and packaging
    S2_align/S2_cluster as schema-v2 source dirs.

    ``use_root_annotation`` reads ``<root>/annotation/features.parquet``
    (if present) and installs the rows into the reference cache's
    annotation slot. The on-disk layout under ``<root>`` is preserved —
    each source dir gets a fresh ``manifest.json`` written alongside the
    existing artifact files.
    """
    features = None
    if use_root_annotation:
        ann_path = root / "annotation" / "features.parquet"
        if ann_path.exists():
            features = pq.read_table(ann_path).to_pylist()
    release_dir = _install_ref(
        monkeypatch, tmp_path, with_annotation_features=features
    )

    sources: list[dict[str, object]] = []
    align_dir = root / "S2_align"
    if align_dir.is_dir():
        outputs: dict[str, str] = {}
        for slot, rel in (
            ("alignments", "alignments"),
            ("alignment_blocks", "alignment_blocks"),
            ("alignment_cs", "alignment_cs"),
            ("read_samples", "read_samples.parquet"),
            ("coverage", "coverage.parquet"),
            ("introns", "introns.parquet"),
            ("derived_annotation", "derived_annotation"),
        ):
            candidate = align_dir / rel
            if candidate.exists():
                outputs[slot] = str(candidate)
        write_align_manifest(
            align_dir / "manifest.json",
            reference_handle=DEFAULT_HANDLE,
            reference_path=str(release_dir),
            assembly_accession=DEFAULT_ASSEMBLY,
            demux_dir="",
            input_files=[],
            parameters={},
            stages={},
            outputs=outputs,
        )
        sources.append(
            {"path": str(align_dir), "kind": "align", "label": "align"}
        )
    cluster_dir = root / "S2_cluster"
    if cluster_dir.is_dir():
        outputs = {}
        for slot, rel in (
            ("clusters", "clusters.parquet"),
            ("cluster_membership", "cluster_membership.parquet"),
        ):
            candidate = cluster_dir / rel
            if candidate.exists():
                outputs[slot] = str(candidate)
        # Point align_dir at the sibling S2_align dir when it exists —
        # the cluster source resolves its alignment-related slots
        # (alignments / alignment_blocks / alignment_cs / read_samples)
        # from there at Session load time, which is what the
        # cluster_pileup `members` view needs.
        align_dir_for_cluster = (
            str(align_dir) if align_dir.is_dir() else ""
        )
        write_cluster_manifest(
            cluster_dir / "manifest.json",
            reference_handle=DEFAULT_HANDLE,
            reference_path=str(release_dir),
            assembly_accession=DEFAULT_ASSEMBLY,
            align_dir=align_dir_for_cluster,
            demux_dir="",
            parameters={},
            stages={},
            outputs=outputs,
        )
        sources.append(
            {"path": str(cluster_dir), "kind": "cluster", "label": "cluster"}
        )
    return Session.open(
        reference_handle=DEFAULT_HANDLE,
        sources=sources,
    )


# ----------------------------------------------------------------------
# Reference sequence
# ----------------------------------------------------------------------


def test_reference_sequence_vector_returns_per_base(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    session = _make_session(monkeypatch, tmp_path, root)
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


def test_reference_sequence_cache_reuses_string_across_calls(
    tmp_path: Path, monkeypatch
) -> None:
    """A second `fetch` for the same contig must not re-read the
    on-disk sequence — verified by intercepting the loader and counting
    invocations. Without the cache, fast pan/zoom blew the server RAM
    on real genomes (multi-MB sequences re-decoded per request)."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("reference_sequence")
    [binding] = kernel.discover(session)

    ref_seq_mod._clear_sequence_cache()
    call_count = 0
    original_loader = ref_seq_mod._load_contig_sequence

    def _counting_loader(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_loader(*args, **kwargs)

    ref_seq_mod._load_contig_sequence = _counting_loader
    try:
        first = pa.Table.from_batches(
            list(
                kernel.fetch(
                    binding,
                    TrackQuery(contig="chr1", start=0, end=10, viewport_px=400),
                    ThresholdDecision.VECTOR,
                )
            ),
            schema=REFERENCE_SEQUENCE_VECTOR_SCHEMA,
        )
        second = pa.Table.from_batches(
            list(
                kernel.fetch(
                    binding,
                    TrackQuery(contig="chr1", start=20, end=30, viewport_px=400),
                    ThresholdDecision.VECTOR,
                )
            ),
            schema=REFERENCE_SEQUENCE_VECTOR_SCHEMA,
        )
    finally:
        ref_seq_mod._load_contig_sequence = original_loader
        ref_seq_mod._clear_sequence_cache()

    assert call_count == 1
    assert first.num_rows == 10
    assert second.num_rows == 10
    # Each slice is correct independent of which order they ran.
    assert first.column("base").to_pylist() == list("ACGTACGTAC")


def test_reference_sequence_loader_reads_only_matching_contig(
    tmp_path: Path,
) -> None:
    """Cache miss path should pull just the requested contig's row,
    not the whole `sequences.parquet`. This keeps the worst-case
    allocation bounded by single-chromosome size rather than the full
    genome."""
    root = tmp_path / "run"
    genome = root / "genome"
    genome.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "contig_id": 1,
                    "name": "chr1",
                    "length": 100,
                    "topology": None,
                    "circular": None,
                },
                {
                    "contig_id": 2,
                    "name": "chr2",
                    "length": 100,
                    "topology": None,
                    "circular": None,
                },
            ],
            schema=CONTIG_TABLE,
        ),
        genome / "contigs.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"contig_id": 1, "sequence": "A" * 100},
                {"contig_id": 2, "sequence": "T" * 100},
            ],
            schema=SEQUENCE_TABLE,
        ),
        genome / "sequences.parquet",
    )
    ref_seq_mod._clear_sequence_cache()
    try:
        loaded = ref_seq_mod._load_contig_sequence(
            genome / "sequences.parquet", 1
        )
    finally:
        ref_seq_mod._clear_sequence_cache()
    assert loaded == "A" * 100


def test_reference_sequence_decimates_when_window_exceeds_cap(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "refs"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    install_fake_reference(
        cache_root,
        handle=DEFAULT_HANDLE,
        contigs=[
            {
                "contig_id": 1,
                "name": "chr1",
                "length": 100_000,
                "topology": None,
                "circular": None,
            }
        ],
        sequences=[{"contig_id": 1, "sequence": "A" * 50_000}],
    )
    session = Session.open(reference_handle=DEFAULT_HANDLE, sources=[])
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
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_annotation(root / "annotation", [_feature()])
    session = _make_session(monkeypatch, tmp_path, root, use_root_annotation=True)
    bindings = get_kernel("gene_annotation").discover(session)
    assert len(bindings) == 1
    assert bindings[0].binding_id == "reference"


def test_gene_annotation_discover_returns_both_when_both_present(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_annotation(root / "annotation", [_feature()])
    _write_annotation(
        root / "S2_align" / "derived_annotation",
        [_feature(feature_id=10, source="constellation_derived")],
    )
    session = _make_session(monkeypatch, tmp_path, root, use_root_annotation=True)
    bindings = get_kernel("gene_annotation").discover(session)
    assert {b.binding_id for b in bindings} == {"reference", "derived-0"}


def test_gene_annotation_fetch_filters_by_window(
    tmp_path: Path, monkeypatch
) -> None:
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
    session = _make_session(monkeypatch, tmp_path, root, use_root_annotation=True)
    kernel = get_kernel("gene_annotation")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=50, end=400)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=GENE_ANNOTATION_VECTOR_SCHEMA,
    )
    names = sorted(table.column("name").to_pylist())
    assert names == ["A", "B"]


def test_gene_annotation_truncates_above_feature_limit(
    tmp_path: Path, monkeypatch
) -> None:
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
    session = _make_session(monkeypatch, tmp_path, root, use_root_annotation=True)
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


def test_splice_junctions_emits_one_arc_per_cluster(
    tmp_path: Path, monkeypatch
) -> None:
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
    session = _make_session(monkeypatch, tmp_path, root)
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


def _write_alignment_blocks(align_dir: Path, rows: list[dict]) -> None:
    """Companion to ``_write_alignments`` — writes the per-CIGAR block
    rows the read_pileup kernel hard-requires after PR 2."""
    blocks_dir = align_dir / "alignment_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=ALIGNMENT_BLOCK_TABLE),
        blocks_dir / "part-00000.parquet",
    )


def _write_alignment_cs(align_dir: Path, rows: list[dict]) -> None:
    """Companion to ``_write_alignments`` — writes the cs:long sidecar
    the read_pileup kernel hard-requires after PR 2."""
    cs_dir = align_dir / "alignment_cs"
    cs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=ALIGNMENT_CS_TABLE),
        cs_dir / "part-00000.parquet",
    )


def _write_read_samples(align_dir: Path, rows: list[dict]) -> None:
    """Companion to ``_write_alignments`` — writes the per-read sample
    map the read_pileup kernel hard-requires after PR 3."""
    pq.write_table(
        pa.Table.from_pylist(rows, schema=READ_SAMPLE_TABLE),
        align_dir / "read_samples.parquet",
    )


def _write_read_pileup_inputs(
    align_dir: Path,
    alignments: list[dict],
    *,
    blocks: list[dict] | None = None,
    cs_rows: list[dict] | None = None,
    read_samples: list[dict] | None = None,
) -> None:
    """Write the four parquet artifacts the read_pileup kernel needs.

    When ``blocks`` is None, one synthetic block per alignment is
    written (one block spanning the alignment's full ref range). When
    ``cs_rows`` is None, one row per alignment with an empty cs string
    is written. When ``read_samples`` is None, one row per alignment
    is written under a single ``sample_a`` (sample_id=1). These
    defaults keep the post-PR-3 hard-require gate happy for tests that
    don't care about block / cs / sample detail.
    """
    _write_alignments(align_dir, alignments)
    if blocks is None:
        blocks = [
            _block(
                alignment_id=row["alignment_id"],
                block_index=0,
                ref_start=row["ref_start"],
                ref_end=row["ref_end"],
            )
            for row in alignments
        ]
    _write_alignment_blocks(align_dir, blocks)
    if cs_rows is None:
        cs_rows = [
            {"alignment_id": row["alignment_id"], "cs_string": ""}
            for row in alignments
        ]
    _write_alignment_cs(align_dir, cs_rows)
    if read_samples is None:
        read_samples = [
            {
                "read_id": row["read_id"],
                "sample_id": 1,
                "sample_name": "sample_a",
            }
            for row in alignments
        ]
    _write_read_samples(align_dir, read_samples)


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


def _block(**kwargs) -> dict:
    base = {
        "alignment_id": 0,
        "block_index": 0,
        "ref_start": 0,
        "ref_end": 100,
        "query_start": 0,
        "query_end": 100,
        "n_match": None,
        "n_mismatch": None,
        "n_insert": 0,
        "n_delete": 0,
    }
    base.update(kwargs)
    return base


def test_read_pileup_vector_returns_packed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r2", ref_start=50, ref_end=150),
            _alignment(alignment_id=3, read_id="r3", ref_start=200, ref_end=300),
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
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


def test_read_pileup_vector_emits_blocks_column(
    tmp_path: Path, monkeypatch
) -> None:
    """The vector wire payload carries one nested-list `blocks` cell
    per alignment with the per-CIGAR M/=/X spans from alignment_blocks/.
    """
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [_alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=300)],
        blocks=[
            _block(alignment_id=1, block_index=0, ref_start=0, ref_end=100),
            _block(alignment_id=1, block_index=1, ref_start=200, ref_end=300),
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    [row] = table.to_pylist()
    blocks = row["blocks"]
    assert len(blocks) == 2
    assert (blocks[0]["ref_start"], blocks[0]["ref_end"]) == (0, 100)
    assert (blocks[1]["ref_start"], blocks[1]["ref_end"]) == (200, 300)


def test_read_pileup_vector_emits_mismatch_positions(
    tmp_path: Path, monkeypatch
) -> None:
    """When alignment_cs/ ships a cs:long string with a substitution,
    the renderer-facing `mismatch_positions` list carries the absolute
    ref position of each `*ab` token."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    # cs:long: 10 matches, A→C substitution at ref_start+10, then 4
    # more matches. The single mismatch lands at ref position 10.
    _write_read_pileup_inputs(
        root / "S2_align",
        [_alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=15)],
        cs_rows=[{"alignment_id": 1, "cs_string": ":10*ac:4"}],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    # viewport_px chosen so bp/pixel < mismatch_glyph_bp_per_pixel_limit
    # (default 5.0); cs parse runs.
    query = TrackQuery(contig="chr1", start=0, end=15, viewport_px=1200)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    [row] = table.to_pylist()
    assert row["mismatch_positions"] == [10]


def test_read_pileup_skips_mismatch_parse_at_coarse_zoom(
    tmp_path: Path, monkeypatch
) -> None:
    """Above mismatch_glyph_bp_per_pixel_limit the cs parse is skipped
    entirely — mismatch_positions emits as empty lists. Saves the
    cs:long walk at zoom levels where the glyphs wouldn't resolve."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [_alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=15)],
        cs_rows=[{"alignment_id": 1, "cs_string": ":10*ac:4"}],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    # 100_000 bp window at 1200 px → ~83 bp/pixel, well above the 5.0
    # threshold → kernel skips the cs:long parse.
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=100_000,
        viewport_px=1200,
        force=ThresholdDecision.VECTOR,
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    [row] = table.to_pylist()
    assert row["mismatch_positions"] == []


def test_read_pileup_vector_emits_sample_id_and_name(
    tmp_path: Path, monkeypatch
) -> None:
    """The vector wire payload carries the alignment's sample_id +
    sample_name resolved via the read_id join against
    read_samples.parquet."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r2", ref_start=200, ref_end=300),
        ],
        read_samples=[
            {"read_id": "r1", "sample_id": 1, "sample_name": "sample_a"},
            {"read_id": "r2", "sample_id": 2, "sample_name": "sample_b"},
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    by_id = {row["alignment_id"]: row for row in table.to_pylist()}
    assert by_id[1]["sample_id"] == 1
    assert by_id[1]["sample_name"] == "sample_a"
    assert by_id[2]["sample_id"] == 2
    assert by_id[2]["sample_name"] == "sample_b"


def test_read_pileup_vector_handles_missing_sample_row(
    tmp_path: Path, monkeypatch
) -> None:
    """An alignment whose read_id has no row in read_samples.parquet
    emits null sample_id / sample_name — the renderer falls back to
    the strand-based palette default rather than dropping the read."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r_known", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r_orphan", ref_start=200, ref_end=300),
        ],
        read_samples=[
            {"read_id": "r_known", "sample_id": 1, "sample_name": "sample_a"},
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    by_id = {row["alignment_id"]: row for row in table.to_pylist()}
    assert by_id[2]["sample_id"] is None
    assert by_id[2]["sample_name"] is None


def test_read_pileup_metadata_surfaces_samples_in_data(
    tmp_path: Path, monkeypatch
) -> None:
    """metadata() pre-scans read_samples.parquet so the per-sample
    palette UI can render before the first data fetch arrives. Output
    matches coverage_histogram's `samples_in_data` shape (sorted int
    list) plus a parallel `sample_names` array."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r2", ref_start=0, ref_end=100),
        ],
        read_samples=[
            {"read_id": "r1", "sample_id": 7, "sample_name": "sample_g"},
            {"read_id": "r2", "sample_id": 3, "sample_name": "sample_c"},
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    meta = kernel.metadata(binding)
    assert meta["samples_in_data"] == [3, 7]
    assert meta["sample_names"] == ["sample_c", "sample_g"]


def test_read_pileup_filter_by_sample_name(
    tmp_path: Path, monkeypatch
) -> None:
    """query.samples filtering accepts sample_name strings; only
    matching rows are admitted to row-packing."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r2", ref_start=200, ref_end=300),
        ],
        read_samples=[
            {"read_id": "r1", "sample_id": 1, "sample_name": "sample_a"},
            {"read_id": "r2", "sample_id": 2, "sample_name": "sample_b"},
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=400,
        viewport_px=1200,
        samples=("sample_a",),
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    [row] = table.to_pylist()
    assert row["alignment_id"] == 1


def test_read_pileup_filter_by_sample_id(
    tmp_path: Path, monkeypatch
) -> None:
    """query.samples filtering also accepts stringified numeric ids
    — matches coverage_histogram's contract so the same param feeds
    both kernels."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100),
            _alignment(alignment_id=2, read_id="r2", ref_start=200, ref_end=300),
        ],
        read_samples=[
            {"read_id": "r1", "sample_id": 1, "sample_name": "sample_a"},
            {"read_id": "r2", "sample_id": 2, "sample_name": "sample_b"},
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=400,
        viewport_px=1200,
        samples=("2",),
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    [row] = table.to_pylist()
    assert row["alignment_id"] == 2


def test_read_pileup_min_mapq_drops_low_quality_alignments(
    tmp_path: Path, monkeypatch
) -> None:
    """query.min_mapq pushes down to the parquet scan — alignments
    below the threshold disappear from the row count, not just the
    rendered output. Verifies the predicate is applied at the scanner
    level (not post-scan)."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100, mapq=60),
            _alignment(alignment_id=2, read_id="r2", ref_start=50, ref_end=150, mapq=10),
            _alignment(alignment_id=3, read_id="r3", ref_start=200, ref_end=300, mapq=30),
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=400,
        viewport_px=1200,
        min_mapq=20,
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    kept_ids = sorted(row["alignment_id"] for row in table.to_pylist())
    assert kept_ids == [1, 3]


def test_read_pileup_min_mapq_zero_admits_all(
    tmp_path: Path, monkeypatch
) -> None:
    """Default min_mapq=0 admits every primary alignment — matches
    pre-PR-4 behavior, so existing callers don't see a regression."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100, mapq=60),
            _alignment(alignment_id=2, read_id="r2", ref_start=50, ref_end=150, mapq=0),
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=READ_PILEUP_VECTOR_SCHEMA,
    )
    assert table.num_rows == 2


def test_read_pileup_min_mapq_affects_threshold_count(
    tmp_path: Path, monkeypatch
) -> None:
    """The threshold's `_count_in_window` reflects the MAPQ filter —
    so a min_mapq high enough to drop the population below
    vector_glyph_limit keeps the renderer in vector mode even when the
    unfiltered count would tip to hybrid. Verifies the predicate is
    threaded through both _scan_window AND _count_in_window."""
    from constellation.viz.tracks.read_pileup import ReadPileupKernel

    # Fixture: more rows than the default vector_glyph_limit when
    # unfiltered; few rows above the MAPQ threshold.
    rows = []
    for i in range(20):
        rows.append(
            _alignment(
                alignment_id=i + 1,
                read_id=f"r{i}",
                ref_start=i * 5,
                ref_end=i * 5 + 100,
                mapq=60 if i < 3 else 5,
            )
        )
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(root / "S2_align", rows)
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    [binding] = kernel.discover(session)

    # Stress the count path with a tiny vector_glyph_limit so the
    # threshold would otherwise pick hybrid; the MAPQ filter trims
    # the population down to 3 < 5.
    monkeypatch_limit = 5
    original_limit = ReadPileupKernel.vector_glyph_limit
    ReadPileupKernel.vector_glyph_limit = monkeypatch_limit
    try:
        q_unfiltered = TrackQuery(contig="chr1", start=0, end=400, viewport_px=1200)
        q_filtered = TrackQuery(
            contig="chr1", start=0, end=400, viewport_px=1200, min_mapq=20
        )
        assert kernel.threshold(binding, q_unfiltered) is ThresholdDecision.HYBRID
        assert kernel.threshold(binding, q_filtered) is ThresholdDecision.VECTOR
    finally:
        ReadPileupKernel.vector_glyph_limit = original_limit


def test_read_pileup_discover_skips_source_without_read_samples(
    tmp_path: Path, monkeypatch
) -> None:
    """Hard-require: a source that's missing read_samples.parquet
    produces no binding. Forces users on pre-PR-1 align outputs to
    re-run align before read_pileup viz becomes available."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_alignments(
        root / "S2_align",
        [_alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100)],
    )
    _write_alignment_blocks(
        root / "S2_align",
        [_block(alignment_id=1, ref_start=0, ref_end=100)],
    )
    _write_alignment_cs(
        root / "S2_align", [{"alignment_id": 1, "cs_string": ""}]
    )
    # Deliberately omit read_samples.parquet.
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    assert kernel.discover(session) == []


def test_read_pileup_discover_skips_source_without_alignment_cs(
    tmp_path: Path, monkeypatch
) -> None:
    """Hard-require: a source that's missing alignment_cs/ produces no
    binding. Surfaces in the UI as an absent read_pileup track; users
    re-run align without --no-emit-cs-tags to restore visualization."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_alignments(
        root / "S2_align",
        [_alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100)],
    )
    _write_alignment_blocks(
        root / "S2_align",
        [_block(alignment_id=1, ref_start=0, ref_end=100)],
    )
    _write_read_samples(
        root / "S2_align",
        [{"read_id": "r1", "sample_id": 1, "sample_name": "sample_a"}],
    )
    # Deliberately omit alignment_cs/.
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    assert kernel.discover(session) == []


def test_read_pileup_discover_skips_source_without_alignment_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    """Hard-require: a source that's missing alignment_blocks/ also
    produces no binding (the schema's `blocks` column is non-null)."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_alignments(
        root / "S2_align",
        [_alignment(alignment_id=1, read_id="r1", ref_start=0, ref_end=100)],
    )
    _write_alignment_cs(
        root / "S2_align", [{"alignment_id": 1, "cs_string": ""}]
    )
    _write_read_samples(
        root / "S2_align",
        [{"read_id": "r1", "sample_id": 1, "sample_name": "sample_a"}],
    )
    # Deliberately omit alignment_blocks/.
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("read_pileup")
    assert kernel.discover(session) == []


def test_read_pileup_hybrid_emits_png(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("datashader")
    pytest.importorskip("PIL")
    from PIL import Image

    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [_alignment(alignment_id=i, read_id=f"r{i}", ref_start=0, ref_end=100) for i in range(50)],
    )
    session = _make_session(monkeypatch, tmp_path, root)
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


def test_read_pileup_threshold_force_override(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(root / "S2_align", [_alignment()])
    session = _make_session(monkeypatch, tmp_path, root)
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


def test_cluster_pileup_vector_emits_packed_rows(
    tmp_path: Path, monkeypatch
) -> None:
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
    session = _make_session(monkeypatch, tmp_path, root)
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
# Cluster pileup — members view (PR 5)
# ----------------------------------------------------------------------


def _membership(**kwargs) -> dict:
    """Row builder for CLUSTER_MEMBERSHIP_TABLE."""
    base = {
        "cluster_id": 1,
        "read_id": "r0",
        "role": "representative",
        "drift_5p_bp": None,
        "drift_3p_bp": None,
        "match_rate": None,
        "indel_rate": None,
        "n_aligned_bp": 100,
    }
    base.update(kwargs)
    return base


def _write_cluster_membership(cluster_dir: Path, rows: list[dict]) -> None:
    cluster_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CLUSTER_MEMBERSHIP_TABLE),
        cluster_dir / "cluster_membership.parquet",
    )


def _write_cluster_table(cluster_dir: Path, rows: list[dict]) -> None:
    cluster_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=TRANSCRIPT_CLUSTER_TABLE),
        cluster_dir / "clusters.parquet",
    )


def _setup_cluster_members_session(
    tmp_path: Path, monkeypatch
):
    """Build an S2_align + S2_cluster pair where the cluster source's
    align_dir points at S2_align, enabling the members-view code path."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    # Two clusters spanning the same window, with three member reads
    # each — six alignments total.
    align_alignments = [
        _alignment(alignment_id=1, read_id="rA", ref_start=0, ref_end=100),
        _alignment(alignment_id=2, read_id="rB", ref_start=20, ref_end=120),
        _alignment(alignment_id=3, read_id="rC", ref_start=40, ref_end=140),
        _alignment(alignment_id=4, read_id="rD", ref_start=200, ref_end=300),
        _alignment(alignment_id=5, read_id="rE", ref_start=220, ref_end=320),
        _alignment(alignment_id=6, read_id="rF", ref_start=240, ref_end=340),
    ]
    _write_read_pileup_inputs(root / "S2_align", align_alignments)

    _write_cluster_table(
        root / "S2_cluster",
        [
            _cluster(cluster_id=1, span_start=0, span_end=140, n_reads=3),
            _cluster(cluster_id=2, span_start=200, span_end=340, n_reads=3),
        ],
    )
    _write_cluster_membership(
        root / "S2_cluster",
        [
            _membership(cluster_id=1, read_id="rA", role="representative"),
            _membership(cluster_id=1, read_id="rB", role="member"),
            _membership(cluster_id=1, read_id="rC", role="member"),
            _membership(cluster_id=2, read_id="rD", role="representative"),
            _membership(cluster_id=2, read_id="rE", role="member"),
            _membership(cluster_id=2, read_id="rF", role="drift_filtered"),
        ],
    )
    return _make_session(monkeypatch, tmp_path, root)


def test_cluster_pileup_members_emits_per_alignment_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """`cluster_view=members` expands each cluster into its member
    alignments, joined via cluster_membership. One row per member;
    cluster_id rides with each row so the renderer can color by it."""
    session = _setup_cluster_members_session(tmp_path, monkeypatch)
    kernel = get_kernel("cluster_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=400,
        viewport_px=1200,
        mode_extra={"cluster_view": "members"},
    )
    mode = kernel.threshold(binding, query)
    assert mode is ThresholdDecision.VECTOR
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, mode)),
        schema=CLUSTER_MEMBER_VECTOR_SCHEMA,
    )
    rows = table.to_pylist()
    # 3 alignments per cluster, 2 clusters → 6 member rows.
    assert table.num_rows == 6
    by_alignment = {row["alignment_id"]: row for row in rows}
    # Member-to-cluster mapping is intact.
    assert by_alignment[1]["cluster_id"] == 1
    assert by_alignment[4]["cluster_id"] == 2
    # Role column travels through unchanged.
    assert by_alignment[1]["role"] == "representative"
    assert by_alignment[6]["role"] == "drift_filtered"
    # Blocks list is populated (synthetic single-block fallback per
    # PR 2's defaults — every alignment has at least one block row).
    assert len(by_alignment[1]["blocks"]) >= 1


def test_cluster_pileup_members_uses_member_schema(
    tmp_path: Path, monkeypatch
) -> None:
    """The kernel's `schema_for` returns CLUSTER_MEMBER_VECTOR_SCHEMA
    when cluster_view is 'members' — the endpoint passes this to the
    IPC writer so the wire shape matches the emitted batches."""
    session = _setup_cluster_members_session(tmp_path, monkeypatch)
    kernel = get_kernel("cluster_pileup")
    [binding] = kernel.discover(session)
    members_query = TrackQuery(
        contig="chr1",
        start=0,
        end=400,
        viewport_px=1200,
        mode_extra={"cluster_view": "members"},
    )
    clusters_query = TrackQuery(
        contig="chr1", start=0, end=400, viewport_px=1200
    )
    assert (
        kernel.schema_for(members_query, ThresholdDecision.VECTOR)
        == CLUSTER_MEMBER_VECTOR_SCHEMA
    )
    assert (
        kernel.schema_for(clusters_query, ThresholdDecision.VECTOR)
        == CLUSTER_PILEUP_VECTOR_SCHEMA
    )


def test_cluster_pileup_metadata_reports_member_view_supported(
    tmp_path: Path, monkeypatch
) -> None:
    """When the cluster source resolves alignment slots from its
    align_dir, metadata's `cluster_view_supported` is True. The
    dashboard uses this flag to decide whether to show the toggle."""
    session = _setup_cluster_members_session(tmp_path, monkeypatch)
    kernel = get_kernel("cluster_pileup")
    [binding] = kernel.discover(session)
    meta = kernel.metadata(binding)
    assert meta["cluster_view_supported"] is True


def test_cluster_pileup_members_view_unsupported_without_align_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """When the cluster source has no align_dir back-pointer (or one
    that doesn't resolve), the alignment slots stay None;
    cluster_view_supported is False and a members-view fetch returns
    no rows (defense in depth — the dashboard would already have hid
    the toggle)."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    # No S2_align — only a cluster source.
    _write_cluster_table(
        root / "S2_cluster",
        [_cluster(cluster_id=1, span_start=0, span_end=100, n_reads=1)],
    )
    _write_cluster_membership(
        root / "S2_cluster",
        [_membership(cluster_id=1, read_id="rA", role="representative")],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("cluster_pileup")
    [binding] = kernel.discover(session)
    meta = kernel.metadata(binding)
    assert meta["cluster_view_supported"] is False
    # A members-view fetch defends against an unsupported call.
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=400,
        viewport_px=1200,
        mode_extra={"cluster_view": "members"},
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=CLUSTER_MEMBER_VECTOR_SCHEMA,
    )
    assert table.num_rows == 0


def test_cluster_pileup_members_min_mapq_pushdown(
    tmp_path: Path, monkeypatch
) -> None:
    """The MAPQ pushdown from PR 4 applies in members view too — same
    contract as read_pileup, so users can dial out low-quality member
    alignments when assessing cluster quality."""
    root = tmp_path / "run"
    _write_genome(root / "genome")
    _write_read_pileup_inputs(
        root / "S2_align",
        [
            _alignment(alignment_id=1, read_id="rA", ref_start=0, ref_end=100, mapq=60),
            _alignment(alignment_id=2, read_id="rB", ref_start=10, ref_end=110, mapq=5),
        ],
    )
    _write_cluster_table(
        root / "S2_cluster",
        [_cluster(cluster_id=1, span_start=0, span_end=110, n_reads=2)],
    )
    _write_cluster_membership(
        root / "S2_cluster",
        [
            _membership(cluster_id=1, read_id="rA", role="representative"),
            _membership(cluster_id=1, read_id="rB", role="member"),
        ],
    )
    session = _make_session(monkeypatch, tmp_path, root)
    kernel = get_kernel("cluster_pileup")
    [binding] = kernel.discover(session)
    query = TrackQuery(
        contig="chr1",
        start=0,
        end=200,
        viewport_px=1200,
        min_mapq=20,
        mode_extra={"cluster_view": "members"},
    )
    table = pa.Table.from_batches(
        list(kernel.fetch(binding, query, ThresholdDecision.VECTOR)),
        schema=CLUSTER_MEMBER_VECTOR_SCHEMA,
    )
    assert table.num_rows == 1
    [row] = table.to_pylist()
    assert row["alignment_id"] == 1


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
