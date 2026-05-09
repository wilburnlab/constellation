"""Reference-sequence track — per-base nucleotides at deep zoom.

Reads `<root>/genome/sequences.parquet` (`SEQUENCE_TABLE`: one row per
contig with the full nucleotide string). The kernel slices the sequence
for the visible window and ships per-base records — the renderer draws
letters at deep zoom, dashes / colored stripes at moderate zoom, and a
single horizontal line when the window is wide enough that individual
bases would be sub-pixel.

Always vector. Threshold logic is `vector_glyph_limit`-aware: when the
window width exceeds the cap, the kernel decimates the sequence into a
sub-sample whose count fits — the renderer interprets non-1bp `step`
as "draw decimated dashes" rather than letters.
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


REFERENCE_SEQUENCE_VECTOR_SCHEMA: pa.Schema = pa.schema(
    [
        # Genomic position of the residue (or block start when decimated).
        pa.field("position", pa.int64(), nullable=False),
        # Single nucleotide character ('A','C','G','T','N','U', case
        # preserved for soft-mask awareness). When `step > 1` the
        # `base` is the consensus letter for the bin (most-frequent of
        # the bin's residues; renderer falls back to a dash glyph).
        pa.field("base", pa.string(), nullable=False),
        # Number of original residues this row covers. 1 in normal
        # mode; >1 when the kernel decimates to fit the glyph cap.
        pa.field("step", pa.int32(), nullable=False),
    ],
    metadata={b"schema_name": b"VizReferenceSequence"},
)


@register_track
class ReferenceSequenceKernel(TrackKernel):
    """Per-base reference-sequence track."""

    kind = "reference_sequence"
    schema = REFERENCE_SEQUENCE_VECTOR_SCHEMA

    # ~5kb of letters before decimation kicks in. The renderer uses
    # `step > 1` as a signal to draw dashes / blocks instead of letters.
    vector_glyph_limit = 5_000

    def discover(self, session: Session) -> list[TrackBinding]:
        if session.reference_genome is None:
            return []
        sequences = session.reference_genome / "sequences.parquet"
        if not sequences.exists():
            return []
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id="reference_sequence",
                label="Reference sequence",
                paths={
                    "sequences": sequences,
                    "genome": session.reference_genome,
                },
                config={},
            )
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "binding_id": binding.binding_id,
            "label": binding.label,
            "vector_glyph_limit": self.vector_glyph_limit,
            "default_height_px": 24,
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

        seq = _slice_sequence(
            binding.paths["sequences"],
            contig_id,
            query.start,
            query.end,
        )
        if seq is None:
            return iter(())

        window_len = query.end - query.start
        # Decimation: if the window is too wide for per-base rendering,
        # bucket residues into bins of size `step` and surface one
        # representative letter per bin.
        if window_len > self.vector_glyph_limit:
            step = max(1, (window_len + self.vector_glyph_limit - 1) // self.vector_glyph_limit)
        else:
            step = 1

        positions: list[int] = []
        bases: list[str] = []
        steps: list[int] = []
        for i in range(0, len(seq), step):
            chunk = seq[i : i + step]
            if not chunk:
                continue
            positions.append(query.start + i)
            bases.append(chunk[0] if step == 1 else _consensus(chunk))
            steps.append(min(step, len(chunk)))

        if not positions:
            return iter(())

        table = pa.Table.from_arrays(
            [
                pa.array(positions, pa.int64()),
                pa.array(bases, pa.string()),
                pa.array(steps, pa.int32()),
            ],
            schema=REFERENCE_SEQUENCE_VECTOR_SCHEMA,
        )
        return iter(table.to_batches())

    def estimate_vector_cost(
        self, binding: TrackBinding, query: TrackQuery
    ) -> int | None:
        window_len = query.end - query.start
        # The cap caps the row count regardless of zoom.
        return min(window_len, self.vector_glyph_limit)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_contig_id(genome_dir: Any, contig_name: str) -> int | None:
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return None
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    matches = table.filter(pc.field("name") == pa.scalar(contig_name, pa.string()))
    if matches.num_rows == 0:
        return None
    return int(matches.column("contig_id")[0].as_py())


def _slice_sequence(
    sequences_path: Any, contig_id: int, start: int, end: int
) -> str | None:
    """Return the slice `seq[start:end]` for the named contig, or
    `None` when the contig is absent. The whole-contig string is read
    once per call; sequence rows are large (a chromosome can be 10s of
    MB) so production usage should cache, but the kernel keeps this
    simple — caching lives in the server's per-session state if/when
    profiling shows it matters."""
    table = pq.read_table(
        sequences_path,
        columns=["contig_id", "sequence"],
    )
    matches = table.filter(pc.field("contig_id") == pa.scalar(contig_id, pa.int64()))
    if matches.num_rows == 0:
        return None
    sequence = matches.column("sequence")[0].as_py()
    if not isinstance(sequence, str):
        return None
    s = max(0, int(start))
    e = min(len(sequence), int(end))
    if s >= e:
        return ""
    return sequence[s:e]


def _consensus(chunk: str) -> str:
    """Most-frequent character in `chunk`. Tiebreak by alphabetical
    order (deterministic)."""
    if not chunk:
        return "-"
    counts: dict[str, int] = {}
    for ch in chunk:
        counts[ch] = counts.get(ch, 0) + 1
    # Sort by (-count, char) — most frequent first, alphabetical tiebreak.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
