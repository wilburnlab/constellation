"""Splice-junction track — intron arcs from clustered INTRON_TABLE.

Reads `<root>/S2_align/introns.parquet`. Each row is one observed
`(contig, donor, acceptor, strand)` tuple, augmented with cluster IDs
(`intron_id`) and `read_count` evidence. The renderer draws one arc
per emitted row with stroke-width scaled by `read_count` and color by
`motif` (canonical GT-AG vs cryptic).

We aggregate by `intron_id` before emit so the renderer sees one arc
per cluster (not one per stray cryptic neighbor that absorbed into the
same cluster). The seed row's donor/acceptor positions identify the
arc; the cluster's total `read_count` becomes the `support` field.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.viz.server.session import Session
from constellation.viz.tracks.base import (
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    register_track,
)


SPLICE_JUNCTIONS_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("intron_id", pa.int64(), nullable=False),
        pa.field("donor_pos", pa.int64(), nullable=False),
        pa.field("acceptor_pos", pa.int64(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("support", pa.int64(), nullable=False),
        pa.field("motif", pa.string(), nullable=True),
        pa.field("annotated", pa.bool_(), nullable=True),
    ],
    metadata={b"schema_name": b"VizSpliceJunctions"},
)


@register_track
class SpliceJunctionsKernel(TrackKernel):
    """Splice-junction arcs."""

    kind = "splice_junctions"
    schema = SPLICE_JUNCTIONS_VECTOR_SCHEMA

    # Soft cap on visible junctions. When exceeded, the kernel keeps
    # the highest-`support` rows. A future hybrid-density mode would
    # rasterize an arc-density heatmap when the cap is hit; v1 punts.
    vector_glyph_limit = 1_500

    def discover(self, session: Session) -> list[TrackBinding]:
        if session.introns is None or session.reference_genome is None:
            return []
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id="splice_junctions",
                label="Splice junctions",
                paths={
                    "introns": session.introns,
                    "genome": session.reference_genome,
                },
                config={},
            )
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        # Read the motif column to populate the renderer's palette.
        path = binding.paths["introns"]
        motifs: list[str] = []
        max_support = 0
        if path.exists():
            table = pq.read_table(path, columns=["motif", "read_count"])
            motifs = sorted(
                {m for m in table.column("motif").to_pylist() if isinstance(m, str)}
            )
            if table.num_rows > 0:
                max_support = int(pc.max(table.column("read_count")).as_py() or 0)
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "motifs_in_data": motifs,
            "max_support": max_support,
            "vector_glyph_limit": self.vector_glyph_limit,
            "default_height_px": 80,
        }

    def threshold(
        self, binding: TrackBinding, query: TrackQuery
    ) -> ThresholdDecision:
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

        # Read seed rows only (one per cluster) and aggregate cluster
        # support across the cluster's members. The seed carries the
        # representative donor/acceptor coordinates; non-seed rows
        # contribute to the support count.
        introns = pq.read_table(
            binding.paths["introns"],
            columns=[
                "intron_id",
                "contig_id",
                "strand",
                "donor_pos",
                "acceptor_pos",
                "read_count",
                "motif",
                "is_intron_seed",
                "annotated",
            ],
        )
        # Filter to the contig + viewport (any junction that overlaps
        # the visible window — i.e. donor or acceptor falls inside, or
        # the arc spans across).
        contig_match = pc.field("contig_id") == pa.scalar(contig_id, pa.int64())
        donor_in = (pc.field("donor_pos") >= pa.scalar(int(query.start), pa.int64())) & (
            pc.field("donor_pos") < pa.scalar(int(query.end), pa.int64())
        )
        acc_in = (
            pc.field("acceptor_pos") >= pa.scalar(int(query.start), pa.int64())
        ) & (pc.field("acceptor_pos") < pa.scalar(int(query.end), pa.int64()))
        spans = (
            pc.field("donor_pos") < pa.scalar(int(query.start), pa.int64())
        ) & (pc.field("acceptor_pos") >= pa.scalar(int(query.end), pa.int64()))
        scanned = introns.filter(contig_match & (donor_in | acc_in | spans))
        if scanned.num_rows == 0:
            return iter(())

        # Aggregate cluster support, then keep one row per cluster
        # (the seed). support = sum(read_count) across the cluster.
        support_table = scanned.group_by("intron_id").aggregate(
            [("read_count", "sum")]
        )
        support_table = support_table.rename_columns(["intron_id", "support"])
        seeds = scanned.filter(pc.field("is_intron_seed"))
        # Some clusters may have their seed outside the viewport while
        # a non-seed member falls inside — fall back to the highest-
        # support member as the representative for those.
        cluster_ids_with_seed = set(seeds.column("intron_id").to_pylist())
        if cluster_ids_with_seed != set(support_table.column("intron_id").to_pylist()):
            missing_ids = set(support_table.column("intron_id").to_pylist()) - cluster_ids_with_seed
            if missing_ids:
                non_seeds = scanned.filter(
                    pc.is_in(pc.field("intron_id"), pa.array(list(missing_ids), pa.int64()))
                )
                # Sort by read_count desc within each cluster, take first per cluster
                fallback_seeds = (
                    non_seeds.sort_by([("read_count", "descending")])
                    .group_by("intron_id")
                    .aggregate([])
                )
                fallback_rows = non_seeds.join(
                    fallback_seeds.select(["intron_id"]), keys="intron_id"
                )
                # Just take the first occurrence per intron_id
                seen: set[int] = set()
                keep_idx = []
                for i, iid in enumerate(fallback_rows.column("intron_id").to_pylist()):
                    if iid not in seen:
                        seen.add(iid)
                        keep_idx.append(i)
                if keep_idx:
                    seeds = pa.concat_tables([seeds, fallback_rows.take(pa.array(keep_idx))])

        joined = seeds.select(
            [
                "intron_id",
                "donor_pos",
                "acceptor_pos",
                "strand",
                "motif",
                "annotated",
            ]
        ).join(support_table, keys="intron_id")

        if joined.num_rows > self.vector_glyph_limit:
            order = pc.sort_indices(
                joined.column("support"), sort_keys=[("", "descending")]
            )
            joined = pc.take(joined, order.slice(0, self.vector_glyph_limit))

        # Cast to wire schema (column order + dtypes)
        out = pa.Table.from_arrays(
            [
                joined.column("intron_id"),
                joined.column("donor_pos"),
                joined.column("acceptor_pos"),
                joined.column("strand"),
                joined.column("support"),
                joined.column("motif"),
                joined.column("annotated"),
            ],
            schema=SPLICE_JUNCTIONS_VECTOR_SCHEMA,
        )
        return iter(out.to_batches())


def _resolve_contig_id(genome_dir: Any, contig_name: str) -> int | None:
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return None
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    matches = table.filter(pc.field("name") == pa.scalar(contig_name, pa.string()))
    if matches.num_rows == 0:
        return None
    return int(matches.column("contig_id")[0].as_py())
