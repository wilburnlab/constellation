"""Arrow fixtures for the frontend renderer tests, produced by the real kernels.

The TypeScript renderers are tested (vitest) against the wire format the
Python kernels actually emit — Int64 columns, nested ``list<struct>``
blocks, nullable strings — rather than against tables hand-built in JS,
which would infer different types. This module builds one small session,
runs every kernel against it, and returns the encoded payloads.

Two consumers:

- ``scripts/build-viz-frontend-fixtures.py`` writes the payloads under
  ``constellation/viz/frontend/src/__fixtures__/genome/``.
- ``tests/test_viz_frontend_fixtures.py`` rebuilds them and fails when the
  committed files no longer match, so a kernel change that alters the wire
  shape surfaces as a fixture diff (and then a renderer snapshot diff).

Underscore-prefixed so pytest doesn't collect it.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from _viz_fixtures import (
    DEFAULT_HANDLE,
    install_fake_reference,
    write_align_source,
    write_cluster_source,
)
from constellation.viz.server.session import Session
from constellation.viz.tracks.base import (
    HYBRID_SCHEMA,
    ThresholdDecision,
    TrackQuery,
    get_kernel,
)


FIXTURE_DIR: Path = (
    Path(__file__).resolve().parents[1]
    / "constellation"
    / "viz"
    / "frontend"
    / "src"
    / "__fixtures__"
    / "genome"
)

_CONTIG = "chr1"
_CONTIG_LENGTH = 12_000

# 1x1 transparent PNG. The hybrid fixture carries a fixed payload instead of
# a datashader render so its bytes don't move with the datashader / PIL
# version; the frontend only needs the frame's shape.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAC"
    "hwGA60e6kgAAAABJRU5ErkJggg=="
)


# ----------------------------------------------------------------------
# Source rows
# ----------------------------------------------------------------------


def _sequence() -> str:
    seq = list("ACGTTGCA" * (_CONTIG_LENGTH // 8))
    seq[16:19] = "NNN"
    return "".join(seq)


def _feature(feature_id: int, type_: str, start: int, end: int, **kw: Any) -> dict:
    row = {
        "feature_id": feature_id,
        "contig_id": 1,
        "start": start,
        "end": end,
        "strand": "+",
        "type": type_,
        "name": None,
        "parent_id": None,
        "source": "RefSeq",
        "score": None,
        "phase": None,
        "attributes_json": None,
    }
    row.update(kw)
    return row


_FEATURES: list[dict] = [
    _feature(1, "gene", 100, 900, name="geneA"),
    _feature(2, "mRNA", 100, 900, name="geneA-201", parent_id=1),
    _feature(3, "exon", 100, 300, parent_id=2),
    _feature(4, "exon", 500, 900, parent_id=2),
    _feature(5, "CDS", 150, 300, parent_id=2),
    _feature(6, "CDS", 500, 800, parent_id=2),
    _feature(7, "five_prime_UTR", 100, 150, parent_id=2),
    _feature(8, "three_prime_UTR", 800, 900, parent_id=2),
    # Overlaps geneA on the opposite strand: forces a second row.
    _feature(9, "gene", 700, 1500, name="geneB", strand="-"),
    _feature(10, "repeat_region", 1600, 1700, strand="."),
    # A type with no palette entry exercises the fallback colour.
    _feature(11, "pseudogene", 2000, 2400, name="psi1"),
]


def _coverage(sample_id: int, start: int, end: int, depth: int) -> dict:
    return {
        "contig_id": 1,
        "sample_id": sample_id,
        "start": start,
        "end": end,
        "depth": depth,
    }


_COVERAGE: list[dict] = [
    _coverage(0, 0, 200, 5),
    _coverage(0, 200, 400, 12),
    _coverage(0, 400, 600, 3),
    _coverage(1, 0, 300, 8),
    _coverage(1, 300, 600, 1),
]


def _alignment(alignment_id: int, read_id: str, start: int, end: int, **kw: Any) -> dict:
    row = {
        "alignment_id": alignment_id,
        "read_id": read_id,
        "acquisition_id": 1,
        "ref_name": _CONTIG,
        "ref_start": start,
        "ref_end": end,
        "strand": "+",
        "mapq": 60,
        "flag": 0,
        "cigar_string": f"{end - start}M",
        "nm_tag": None,
        "as_tag": None,
        "read_group": None,
        "is_secondary": False,
        "is_supplementary": False,
    }
    row.update(kw)
    return row


def _block(alignment_id: int, block_index: int, start: int, end: int) -> dict:
    return {
        "alignment_id": alignment_id,
        "block_index": block_index,
        "ref_start": start,
        "ref_end": end,
        "query_start": 0,
        "query_end": end - start,
        "n_match": None,
        "n_mismatch": None,
        "n_insert": 0,
        "n_delete": 0,
    }


_ALIGNMENTS: list[dict] = [
    # Spliced read with two substitutions (one per exon block).
    _alignment(1, "r1", 0, 300),
    _alignment(2, "r2", 50, 250, strand="-", mapq=20),
    _alignment(3, "r3", 400, 700),
    # No READ_SAMPLE row: the renderer's null-sample fallback.
    _alignment(4, "r4", 420, 650, mapq=5),
]

_BLOCKS: list[dict] = [
    _block(1, 0, 0, 100),
    _block(1, 1, 200, 300),
    _block(2, 0, 50, 250),
    _block(3, 0, 400, 700),
    _block(4, 0, 420, 650),
]

_CS: list[dict] = [
    {"alignment_id": 1, "cs_string": ":10*ac:89~gt98ag:50*tg:49"},
    {"alignment_id": 2, "cs_string": ":200"},
    {"alignment_id": 3, "cs_string": ":120*ga:179"},
    {"alignment_id": 4, "cs_string": ""},
]

_READ_SAMPLES: list[dict] = [
    {"read_id": "r1", "sample_id": 1, "sample_name": "alpha"},
    {"read_id": "r2", "sample_id": 2, "sample_name": "beta"},
    {"read_id": "r3", "sample_id": 1, "sample_name": "alpha"},
]


def _intron(intron_id: int, donor: int, acceptor: int, read_count: int, **kw: Any) -> dict:
    row = {
        "intron_id": intron_id,
        "contig_id": 1,
        "strand": "+",
        "donor_pos": donor,
        "acceptor_pos": acceptor,
        "read_count": read_count,
        "motif": "GT-AG",
        "is_intron_seed": True,
        "annotated": True,
    }
    row.update(kw)
    return row


_INTRONS: list[dict] = [
    _intron(1, 100, 200, 12),
    _intron(1, 102, 203, 3, is_intron_seed=False, annotated=False),
    _intron(2, 1000, 2000, 40, motif="GC-AG", annotated=False),
    _intron(3, 2500, 2600, 2, motif="AT-AC", annotated=False),
    _intron(4, 2700, 2900, 1, motif=None, annotated=None),
]


def _cluster(cluster_id: int, start: int, end: int, **kw: Any) -> dict:
    row = {
        "cluster_id": cluster_id,
        "representative_read_id": "r1",
        "n_reads": 1,
        "identity_threshold": None,
        "consensus_sequence": None,
        "predicted_protein": None,
        "orf_start": None,
        "orf_end": None,
        "orf_strand": None,
        "codon_table": None,
        "mode": "genome",
        "contig_id": 1,
        "strand": "+",
        "span_start": start,
        "span_end": end,
        "fingerprint_hash": 0,
        "n_unique_sequences": 1,
        "sample_id": -1,
    }
    row.update(kw)
    return row


_CLUSTERS: list[dict] = [
    _cluster(1, 0, 300, n_reads=2),
    _cluster(2, 50, 250, mode="kmer", strand="-", representative_read_id="r2"),
    _cluster(3, 400, 700, mode="em", n_reads=12, representative_read_id="r3"),
    _cluster(4, 350, 500, n_reads=2, representative_read_id="r4"),
]


def _member(cluster_id: int, read_id: str, role: str = "member") -> dict:
    return {
        "cluster_id": cluster_id,
        "read_id": read_id,
        "role": role,
        "drift_5p_bp": None,
        "drift_3p_bp": None,
        "match_rate": None,
        "indel_rate": None,
        "n_aligned_bp": 100,
    }


_MEMBERSHIP: list[dict] = [
    _member(1, "r1", "representative"),
    _member(1, "r2"),
    _member(3, "r3", "representative"),
    _member(3, "r4"),
]


# ----------------------------------------------------------------------
# Session + payloads
# ----------------------------------------------------------------------


def _build_session(tmp_path: Path, monkeypatch) -> Session:
    cache_root = tmp_path / "refs"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", str(cache_root))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    release_dir = install_fake_reference(
        cache_root,
        contigs=[
            {
                "contig_id": 1,
                "name": _CONTIG,
                "length": _CONTIG_LENGTH,
                "topology": None,
                "circular": None,
            }
        ],
        sequences=[{"contig_id": 1, "sequence": _sequence()}],
        features=_FEATURES,
    )
    align_dir = write_align_source(
        tmp_path / "align",
        reference_path=str(release_dir),
        alignments=_ALIGNMENTS,
        alignment_blocks=_BLOCKS,
        alignment_cs=_CS,
        read_samples=_READ_SAMPLES,
        coverage=_COVERAGE,
        introns=_INTRONS,
    )
    cluster_dir = write_cluster_source(
        tmp_path / "cluster",
        reference_path=str(release_dir),
        clusters=_CLUSTERS,
        cluster_membership=_MEMBERSHIP,
        align_dir=str(align_dir),
    )
    return Session.open(
        reference_handle=DEFAULT_HANDLE,
        sources=[
            {"path": str(align_dir), "kind": "align", "label": "run-a"},
            {"path": str(cluster_dir), "kind": "cluster", "label": "run-a clusters"},
        ],
    )


def _encode(table: pa.Table) -> bytes:
    """Arrow IPC stream bytes — the format the server puts on the wire."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def decode(data: bytes) -> pa.Table:
    return pa.ipc.open_stream(io.BytesIO(data)).read_all()


# (file stem, kernel kind, binding_id, query, optional row cap)
_VECTOR_CASES: list[tuple[str, str, str, TrackQuery, int | None]] = [
    (
        "reference_sequence.letters",
        "reference_sequence",
        "reference_sequence",
        TrackQuery(contig=_CONTIG, start=0, end=48),
        None,
    ),
    (
        # A window past the kernel's glyph cap decimates (step > 1). The
        # renderer only reads the first row's step, so keep a short prefix.
        "reference_sequence.decimated",
        "reference_sequence",
        "reference_sequence",
        TrackQuery(contig=_CONTIG, start=0, end=_CONTIG_LENGTH),
        12,
    ),
    (
        "gene_annotation",
        "gene_annotation",
        "reference",
        TrackQuery(contig=_CONTIG, start=0, end=3000),
        None,
    ),
    (
        "coverage_histogram",
        "coverage_histogram",
        "coverage-0",
        TrackQuery(contig=_CONTIG, start=0, end=3000),
        None,
    ),
    (
        "read_pileup",
        "read_pileup",
        "read_pileup-0",
        TrackQuery(contig=_CONTIG, start=0, end=1000, viewport_px=1000),
        None,
    ),
    (
        "cluster_pileup.clusters",
        "cluster_pileup",
        "cluster_pileup-1",
        TrackQuery(contig=_CONTIG, start=0, end=1000, viewport_px=1000),
        None,
    ),
    (
        "cluster_pileup.members",
        "cluster_pileup",
        "cluster_pileup-1",
        TrackQuery(
            contig=_CONTIG,
            start=0,
            end=1000,
            viewport_px=1000,
            mode_extra={"cluster_view": "members"},
        ),
        None,
    ),
    (
        "splice_junctions",
        "splice_junctions",
        "splice_junctions-0",
        TrackQuery(contig=_CONTIG, start=0, end=3000),
        None,
    ),
]


def build_fixture_payloads(tmp_path: Path, monkeypatch) -> dict[str, bytes]:
    """Return ``{filename: bytes}`` for every frontend fixture."""
    session = _build_session(tmp_path, monkeypatch)
    payloads: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for stem, kind, binding_id, query, row_cap in _VECTOR_CASES:
        kernel = get_kernel(kind)
        bindings = {b.binding_id: b for b in kernel.discover(session)}
        if binding_id not in bindings:
            raise KeyError(
                f"{kind}: no binding {binding_id!r} (have {sorted(bindings)})"
            )
        binding = bindings[binding_id]
        mode = kernel.threshold(binding, query)
        if mode is not ThresholdDecision.VECTOR:
            raise AssertionError(f"{stem}: expected vector mode, got {mode}")
        table = pa.Table.from_batches(
            list(kernel.fetch(binding, query, mode)),
            schema=kernel.schema_for(query, mode),
        )
        if table.num_rows == 0:
            raise AssertionError(f"{stem}: kernel returned no rows")
        if row_cap is not None:
            table = table.slice(0, row_cap).combine_chunks()
        payloads[f"{stem}.arrow"] = _encode(table)
        metadata[f"{kind}/{binding_id}"] = kernel.metadata(binding)

    hybrid = pa.Table.from_pylist(
        [
            {
                "png_bytes": _PNG_1X1,
                "extent_start": 0,
                "extent_end": 1000,
                "extent_y0": 0.0,
                "extent_y1": 4.0,
                "width_px": 1000,
                "height_px": 40,
                # Under 1,000: labels format counts with toLocaleString().
                "n_items": 432,
                "mode": "hybrid",
            }
        ],
        schema=HYBRID_SCHEMA,
    )
    payloads["hybrid_frame.arrow"] = _encode(hybrid)

    payloads["metadata.json"] = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return payloads
