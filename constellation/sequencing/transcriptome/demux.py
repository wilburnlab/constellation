"""Segmented edlib-based barcode demultiplex.

The lab's three-step algorithm (lifted from NanoporeAnalysis as a
clean rewrite):

    1. Anchor on polyA — error-tolerant 'AAAA' edit-distance scan
       (locate.locate_substring with edge_distance=1) finds the polyA
       window first; without this, the rest is unmoored.
    2. 3' segment — locate the PCR adapter on the polyA-side of the
       read; the gap between adapter end and polyA start is the
       barcode region. Score that region against the BarcodeSlot
       panel; pick the best match above min_score.
    3. 5' segment — locate the 5' adapter on the opposite side of the
       polyA. The gap between 5' adapter end and polyA start is the
       transcript window — emit it as a TranscriptSlot match.

Beats Dorado's full-primer Smith-Waterman (which scores the whole
construct including the oligo-dT) because high homopolymer error in
nanopore A-runs (lab's measurements: mean A=26 against expected A=20)
distorts SW scores enough to misassign barcodes. Segmenting on polyA
first and scoring only the discriminating barcode region sidesteps
the homopolymer-distortion problem.

Output is ``READ_SEGMENT_TABLE`` — one row per located segment.
Chimeras / concatemers (very common in nanopore) emerge as multiple
runs of segments per read; ``resolve_demux`` collapses to per-
transcript-window rows in ``READ_DEMUX_TABLE``.

Status: STUB. Pending Phase 5.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.sequencing.transcriptome.adapters import LibraryConstruct


_PHASE = "Phase 5 (transcriptome/{adapters, demux})"


def locate_segments(
    reads: pa.Table,                  # READ_TABLE
    construct: LibraryConstruct,
    *,
    polyA_max_edge_distance: int | None = None,    # override construct default
    adapter_max_distance: int | None = None,
    barcode_min_score: float | None = None,
    threads: int = 4,
) -> pa.Table:                       # READ_SEGMENT_TABLE
    """Apply the construct's segment layout to each read; emit one
    row per located segment.

    For chimeric / concatemer reads, multiple segment-runs appear in
    the output — segment_index orders them within a read. Reads where
    no polyA is found yield zero segments (they're effectively
    rejected; downstream code can still QC them via the absence).

    Override args take precedence over the values declared on the
    construct's slots; this lets demux runs sweep edit-distance
    parameters without redefining the construct.
    """
    raise NotImplementedError(f"locate_segments pending {_PHASE}")


def resolve_demux(
    segments: pa.Table,               # READ_SEGMENT_TABLE
    *,
    chimera_handling: str = "split",  # 'split' | 'reject' | 'first_only'
) -> pa.Table:                        # READ_DEMUX_TABLE
    """Collapse READ_SEGMENT_TABLE to one row per resolved transcript.

    ``chimera_handling`` controls multi-transcript reads:

        split        emit one READ_DEMUX_TABLE row per transcript
                     window (preserves total throughput, recommended)
        reject       drop chimeras entirely (conservative)
        first_only   emit only the 5'-most transcript (legacy)

    Returns rows keyed by ``(read_id, transcript_segment_index)``;
    ``is_chimera`` is True for any read with ≥2 transcript segments
    regardless of which strategy was used.
    """
    raise NotImplementedError(f"resolve_demux pending {_PHASE}")


__all__ = [
    "locate_segments",
    "resolve_demux",
]
