"""Gene-annotation track — GFF3-shaped features as vector glyphs.

Reads `FEATURE_TABLE` from either the reference's `annotation/` ParquetDir
(`<root>/annotation/features.parquet` — preferred) or the
`derived_annotation/` ParquetDir produced by S2 (inferred exons; fallback
when no reference annotation is attached).

Always vector. The renderer composes three layers per feature row:

- A rectangle from `start` to `end` colored by `type` (gene / mRNA /
  CDS / exon / repeat_region / ...).
- An arrow / strand glyph (small chevrons) from the renderer's CSS,
  driven by `strand` (`+`, `-`, `.`).
- A label using `name` (or `feature_id` when `name` is null).

The frontend uses `parent_id` to stack child features (CDS / exon)
under their parent (mRNA / gene); the kernel ships that column raw and
the renderer resolves the parent chain client-side.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

from constellation.viz.server.session import Session
from constellation.viz.tracks.base import (
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    register_track,
)


GENE_ANNOTATION_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("feature_id", pa.int64(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=True),
        pa.field("parent_id", pa.int64(), nullable=True),
        pa.field("source", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"VizGeneAnnotation"},
)


@register_track
class GeneAnnotationKernel(TrackKernel):
    """GFF3-shaped feature-annotation track."""

    kind = "gene_annotation"
    schema = GENE_ANNOTATION_VECTOR_SCHEMA

    # Soft cap: when a window contains more features than this, the
    # kernel returns the longest N (rank by `end - start` desc). The
    # frontend surfaces a "showing N of M" indicator and lets users
    # zoom in for the rest. Empirical tuning will adjust this number;
    # the constant lives here so tracks/base.py threshold logic can
    # stay generic.
    feature_limit = 2_000

    def discover(self, session: Session) -> list[TrackBinding]:
        out: list[TrackBinding] = []
        if session.reference_genome is None:
            return out

        # Prefer the curated reference annotation; fall back to the
        # constellation-derived annotation produced by S2 when no
        # curated reference is attached. We surface BOTH when both
        # exist so users can compare reference annotations against the
        # data-derived call set.
        if session.reference_annotation is not None:
            features = _features_path(session.reference_annotation)
            if features is not None:
                out.append(
                    TrackBinding(
                        session_id=session.session_id,
                        kind=self.kind,
                        binding_id="reference",
                        label="Annotation (reference)",
                        paths={
                            "features": features,
                            "genome": session.reference_genome,
                        },
                        config={"source": "reference"},
                    )
                )
        if session.derived_annotation is not None:
            features = _features_path(session.derived_annotation)
            if features is not None:
                out.append(
                    TrackBinding(
                        session_id=session.session_id,
                        kind=self.kind,
                        binding_id="derived",
                        label="Annotation (derived)",
                        paths={
                            "features": features,
                            "genome": session.reference_genome,
                        },
                        config={"source": "derived"},
                    )
                )
        return out

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        # No on-disk scan here: the renderer assigns colors from a
        # hardcoded SO-term palette, and enumerating types_in_data
        # would force a full read of the type column (tens of millions
        # of rows on a mammalian annotation) at first-mount time.
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "source": binding.config.get("source", "unknown"),
            "feature_limit": self.feature_limit,
            "default_height_px": 60,
        }

    def threshold(
        self, binding: TrackBinding, query: TrackQuery
    ) -> ThresholdDecision:
        # Annotations are always vector. A future hybrid-density mode
        # for whole-chromosome views would land here; for v1 we cap at
        # `feature_limit` and rely on zoom-out semantics.
        if query.force is not None:
            return query.force
        return ThresholdDecision.VECTOR

    def fetch(
        self,
        binding: TrackBinding,
        query: TrackQuery,
        mode: ThresholdDecision,
    ) -> Iterator[pa.RecordBatch]:
        if mode is not ThresholdDecision.VECTOR:
            return iter(())

        contig_id = _resolve_contig_id(binding.paths["genome"], query.contig)
        if contig_id is None:
            return iter(())

        dataset = pa_ds.dataset(str(binding.paths["features"]), format="parquet")
        contig_field = pc.field("contig_id")
        start_field = pc.field("start")
        end_field = pc.field("end")
        predicate = (
            (contig_field == pa.scalar(contig_id, pa.int64()))
            & (end_field > pa.scalar(int(query.start), pa.int64()))
            & (start_field < pa.scalar(int(query.end), pa.int64()))
        )

        scanner = dataset.scanner(
            columns=[
                "feature_id",
                "start",
                "end",
                "strand",
                "type",
                "name",
                "parent_id",
                "source",
            ],
            filter=predicate,
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return iter(())

        # Soft truncation: when the window has more features than the
        # cap, keep the longest. The renderer surfaces the "showing N
        # of M" indicator via a (future) header field; for v1 we just
        # clip silently — the kernel's cost-estimate (below) tells the
        # client when truncation will happen.
        if table.num_rows > self.feature_limit:
            lengths = pc.subtract(table.column("end"), table.column("start"))
            order = pc.sort_indices(lengths, sort_keys=[("", "descending")])
            top = pc.take(table, order.slice(0, self.feature_limit))
            table = top.cast(GENE_ANNOTATION_VECTOR_SCHEMA)
        else:
            table = table.cast(GENE_ANNOTATION_VECTOR_SCHEMA)
        return iter(table.to_batches())

    def estimate_vector_cost(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int | None:
        contig_id = _resolve_contig_id(binding.paths["genome"], query.contig)
        if contig_id is None:
            return 0
        dataset = pa_ds.dataset(str(binding.paths["features"]), format="parquet")
        contig_field = pc.field("contig_id")
        start_field = pc.field("start")
        end_field = pc.field("end")
        predicate = (
            (contig_field == pa.scalar(contig_id, pa.int64()))
            & (end_field > pa.scalar(int(query.start), pa.int64()))
            & (start_field < pa.scalar(int(query.end), pa.int64()))
        )
        return int(dataset.count_rows(filter=predicate))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _features_path(annotation_dir: Any) -> Any | None:
    """Annotation containers (both reference and derived) write
    `features.parquet` inside a ParquetDir bundle. Return the path when
    it exists; `None` otherwise — kernels skip bindings whose target
    parquet is missing."""
    if annotation_dir is None:
        return None
    candidate = annotation_dir / "features.parquet"
    return candidate if candidate.exists() else None


def _resolve_contig_id(genome_dir: Any, contig_name: str) -> int | None:
    """Look up `contig_id` for `contig_name` in `<genome_dir>/contigs.parquet`.

    Mirrors `coverage_histogram._resolve_contig_id`. Both kernels do the
    same lookup; we duplicate per-kernel rather than introduce a shared
    helper to keep kernels self-contained — small price for module
    independence at this scale.
    """
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return None
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    matches = table.filter(pc.field("name") == pa.scalar(contig_name, pa.string()))
    if matches.num_rows == 0:
        return None
    return int(matches.column("contig_id")[0].as_py())
