"""Segmented edlib-based barcode demultiplex — central S1 algorithm.

Per-read pipeline (mirrors NanoporeAnalysis ``align.process_read``):

    1. Annotate the read in BOTH forward and reverse-complement
       orientations via :func:`_annotate_strand`.
    2. Pick a winning orientation per the NanoporeAnalysis rule
       (``Complete`` > anything > forward as fallback; ``Complex`` when
       both have non-Unknown annotations but neither is ``Complete``).
    3. Emit ``READ_SEGMENT_TABLE`` rows (one per located segment) +
       ``READ_DEMUX_TABLE`` rows (one per read; chimera handling is
       deferred until S3 — every read produces exactly one demux row
       in S1).

Within ``_annotate_strand`` the algorithm is:

    sequence
    │
    ├─ find polyA  →  (start, end)  →  pre_polyA, post_polyA
    │
    ├─ find SSP in pre_polyA  →  trim_idx  →  transcript_window
    │
    ├─ find 3' primer in post_polyA  →  putative_umi
    │
    └─ score barcode panel against putative_umi  →  best match

The orchestrator extracts slot parameters from the
:class:`LibraryConstruct` and hands them to a :class:`Scorer`; the
scorer is the only seam where hard-threshold and probabilistic modes
differ.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from constellation.sequencing.schemas.transcriptome import (
    READ_DEMUX_TABLE,
    READ_SEGMENT_TABLE,
)
from constellation.sequencing.transcriptome.adapters import (
    AdapterSlot,
    BarcodeSlot,
    LibraryConstruct,
    PolyASlot,
    TranscriptSlot,
)
from constellation.sequencing.transcriptome.classify import (
    Classification,
    ReadStatus,
    classify_read,
)
from constellation.sequencing.transcriptome.scoring import (
    AdapterVerdict,
    BarcodeVerdict,
    HardThresholdScorer,
    PolyAVerdict,
    Scorer,
)


# ──────────────────────────────────────────────────────────────────────
# Internal — one orientation's annotation result
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StrandAnnotation:
    """All the per-orientation outputs for one read.

    Coordinates are in the ORIENTATION-LOCAL sequence — i.e. if
    ``orientation == '-'``, all offsets are relative to ``RC(read_seq)``,
    not the original. The orchestrator emits this orientation into
    READ_SEGMENT_TABLE as the ``orientation`` column so consumers can
    reconstruct the chosen-orientation sequence from the original read.
    """

    orientation: str  # '+' or '-'
    polyA: PolyAVerdict
    ssp: AdapterVerdict
    primer3: AdapterVerdict
    barcode: BarcodeVerdict

    transcript_start: int | None
    transcript_end: int | None  # half-open
    transcript_length: int | None

    classification: Classification


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _slot_of_kind(layout: tuple, *kinds: str):
    """Return slot instances of any of the requested ``kind`` values
    from a construct layout, in 5'→3' order."""
    return [s for s in layout if getattr(s, "kind", None) in kinds]


def _extract_construct_params(construct: LibraryConstruct) -> dict[str, Any]:
    """Pull the slot params S1's hard-mode demux needs.

    Single SSP / single primer3 / single barcode panel — matches the
    cdna_wilburn_v1 layout. Multi-adapter slots (OR-of-targets) and
    multi-panel constructs are deferred; raise clearly if encountered.
    """
    adapter_slots = _slot_of_kind(construct.layout, "adapter")
    if len(adapter_slots) != 2:
        raise ValueError(
            f"S1 demux expects exactly 2 adapter slots (5' SSP + 3' primer); "
            f"construct {construct.name!r} has {len(adapter_slots)}"
        )
    ssp_slot, primer3_slot = adapter_slots
    if not isinstance(ssp_slot, AdapterSlot) or not isinstance(primer3_slot, AdapterSlot):
        raise TypeError("adapter slots must be AdapterSlot instances")
    if len(ssp_slot.adapters) != 1 or len(primer3_slot.adapters) != 1:
        raise ValueError(
            "S1 demux supports single-adapter slots only (multi-target "
            "OR-of-adapters lands when a real chemistry needs it)"
        )

    polyA_slots = _slot_of_kind(construct.layout, "polyA")
    if len(polyA_slots) != 1:
        raise ValueError(
            f"construct {construct.name!r} must have exactly one PolyASlot"
        )
    polyA_slot = polyA_slots[0]
    if not isinstance(polyA_slot, PolyASlot):
        raise TypeError("polyA slot must be PolyASlot")

    barcode_slots = _slot_of_kind(construct.layout, "barcode")
    if len(barcode_slots) != 1:
        raise ValueError(
            f"construct {construct.name!r} must have exactly one BarcodeSlot"
        )
    barcode_slot = barcode_slots[0]
    if not isinstance(barcode_slot, BarcodeSlot):
        raise TypeError("barcode slot must be BarcodeSlot")

    transcript_slots = _slot_of_kind(construct.layout, "transcript")
    if len(transcript_slots) != 1:
        raise ValueError(
            f"construct {construct.name!r} must have exactly one TranscriptSlot"
        )
    transcript_slot = transcript_slots[0]
    if not isinstance(transcript_slot, TranscriptSlot):
        raise TypeError("transcript slot must be TranscriptSlot")

    return {
        "ssp_seq": ssp_slot.adapters[0].sequence,
        "ssp_max_distance": ssp_slot.max_distance,
        "primer3_seq": primer3_slot.adapters[0].sequence,
        "primer3_max_distance": primer3_slot.max_distance,
        "polyA_min_length": polyA_slot.min_length,
        "polyA_max_length": polyA_slot.max_length,
        "polyA_edge_distance": polyA_slot.edge_distance,
        "barcodes": barcode_slot.barcodes,
        # Hard mode uses an integer max-distance; derive it from the
        # slot's normalized min_score.
        "barcode_max_distance": _max_distance_from_min_score(
            barcode_slot.min_score, barcode_length=len(barcode_slot.barcodes[0].sequence)
            if barcode_slot.barcodes else 16,
        ),
        "transcript_min_length": transcript_slot.min_length,
    }


def _max_distance_from_min_score(min_score: float, *, barcode_length: int) -> int:
    """Convert a normalized ``min_score`` (1 - dist/length) to a hard
    integer edit-distance budget. ``floor((1 - min_score) * length)``
    is the largest distance that still satisfies the score threshold.
    """
    return int((1.0 - min_score) * barcode_length)


# ──────────────────────────────────────────────────────────────────────
# Per-orientation annotation
# ──────────────────────────────────────────────────────────────────────


def _annotate_strand(
    sequence: str,
    orientation: str,
    params: dict[str, Any],
    scorer: Scorer,
) -> _StrandAnnotation:
    """Run the demux pipeline against one orientation of a read."""

    # 1. PolyA
    polyA = scorer.find_polyA(
        sequence,
        min_length=params["polyA_min_length"],
        max_length=params["polyA_max_length"],
        edge_distance=params["polyA_edge_distance"],
    )

    # 2. Slice into pre_polyA / post_polyA
    if polyA.found and polyA.start is not None and polyA.end_inclusive is not None:
        pre_polyA: str | None = sequence[: polyA.start]
        post_polyA: str | None = sequence[polyA.end_inclusive + 1 :]
    else:
        pre_polyA = sequence
        post_polyA = None

    # 3. SSP in pre_polyA → transcript window starts at SSP boundary
    ssp = scorer.find_5p_adapter(
        pre_polyA, params["ssp_seq"], max_distance=params["ssp_max_distance"]
    )

    # 4. 3' primer in post_polyA → putative_umi = post_polyA[:boundary]
    primer3 = scorer.find_3p_adapter(
        post_polyA, params["primer3_seq"], max_distance=params["primer3_max_distance"]
    )
    if primer3.found and primer3.boundary is not None and post_polyA is not None:
        putative_umi: str | None = post_polyA[: primer3.boundary]
    else:
        putative_umi = None

    # 5. Score barcodes against putative_umi
    barcode = scorer.score_barcode(
        putative_umi,
        params["barcodes"],
        max_distance=params["barcode_max_distance"],
    )

    # 6. Compute transcript window in the chosen-orientation coords.
    # NanoporeAnalysis: transcript_seq = pre_polyA[trim_idx:] when SSP
    # is found, else transcript_seq = pre_polyA (full pre-polyA region).
    # ORF prediction runs on this transcript_seq for both 'Complete'
    # and "3' Only" reads — so 3'-only reads need transcript_start=0
    # (start at the beginning of pre_polyA) rather than None.
    if ssp.found and ssp.boundary is not None:
        transcript_start: int = ssp.boundary
    else:
        transcript_start = 0
    if polyA.found and polyA.start is not None:
        transcript_end: int = polyA.start
    else:
        transcript_end = len(sequence)
    transcript_length: int | None = transcript_end - transcript_start

    # 7. Classify
    classification = classify_read(
        ssp=ssp,
        primer3=primer3,
        barcode=barcode,
        transcript_length=transcript_length,
        min_transcript_length=params["transcript_min_length"],
    )

    return _StrandAnnotation(
        orientation=orientation,
        polyA=polyA,
        ssp=ssp,
        primer3=primer3,
        barcode=barcode,
        transcript_start=transcript_start,
        transcript_end=transcript_end,
        transcript_length=transcript_length,
        classification=classification,
    )


# ──────────────────────────────────────────────────────────────────────
# Strand-resolution rule (NanoporeAnalysis process_read)
# ──────────────────────────────────────────────────────────────────────


def _is_unknown(c: Classification) -> bool:
    """Match NanoporeAnalysis's ``cDNA == 'Unknown' | cDNA == 'Unknown Fragment'``
    string check. Unknown with ANY fragment flag still counts as Unknown
    for the strand-resolution rule."""
    return c.status == ReadStatus.UNKNOWN


def _is_strict_complete(c: Classification) -> bool:
    """Match NanoporeAnalysis's ``cDNA == 'Complete'`` (exact string).
    'Complete Fragment' does NOT qualify — NanoporeAnalysis treats it
    as a non-Complete annotation that triggers the Complex tie-break.
    """
    return c.status == ReadStatus.COMPLETE and not c.is_fragment


def _pick_orientation(
    forward: _StrandAnnotation,
    reverse: _StrandAnnotation,
) -> _StrandAnnotation:
    """Apply the strand-resolution rule when both orientations have
    been annotated.

    Decision tree:

        both Unknown*           →  forward (no signal either way; pick
                                   the canonical orientation)
        exactly one Unknown*    →  pick the OTHER orientation (it has
                                   the real annotation)
        both annotated AND
            either is exactly
            'Complete' (no
            Fragment suffix)    →  the Complete one
        otherwise (both
            annotated, neither
            strictly Complete)  →  forward, status overridden to
                                   'Complex' (Fragment suffix dropped)

    Departure from NanoporeAnalysis (``align.process_read``).
    NanoporeAnalysis writes the "exactly one Unknown" branch as
    ``analysis = f_analysis if cDNA[1] == 'Unknown' else r_analysis``
    — a literal-string equality check that only matches the bare
    ``'Unknown'`` value. When the reverse orientation is the literal
    string ``'Unknown Fragment'`` (i.e. Unknown classification with a
    short transcript), the check fails and it picks reverse, even
    though forward has the real annotation. That's a bug — it loses
    the forward annotation for ~10 reads in our test data, downgrading
    real ``3' Only Fragment`` / ``5' Only Fragment`` classifications
    to ``Unknown Fragment``. We accept the small parity divergence in
    exchange for keeping those annotations.
    """
    f_unk = _is_unknown(forward.classification)
    r_unk = _is_unknown(reverse.classification)

    if f_unk and r_unk:
        return forward
    if f_unk:
        return reverse
    if r_unk:
        return forward
    # Both have non-Unknown annotations. NanoporeAnalysis checks for
    # the literal string 'Complete' here — Complete-with-Fragment does
    # NOT qualify and falls through to the Complex tie-break.
    if _is_strict_complete(forward.classification):
        return forward
    if _is_strict_complete(reverse.classification):
        return reverse
    # Tie-breaker: forward orientation, status overridden to Complex.
    # NanoporeAnalysis sets `analysis["cDNA status"] = "Complex"` here
    # — a literal string replacement that drops any " Fragment" suffix
    # the underlying forward orientation may have carried. Mirror that
    # by clearing is_fragment too.
    return _StrandAnnotation(
        orientation=forward.orientation,
        polyA=forward.polyA,
        ssp=forward.ssp,
        primer3=forward.primer3,
        barcode=forward.barcode,
        transcript_start=forward.transcript_start,
        transcript_end=forward.transcript_end,
        transcript_length=forward.transcript_length,
        classification=Classification(
            status=ReadStatus.COMPLEX,
            is_fragment=False,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Per-read demux entry-point (single read → annotation)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReadDemuxResult:
    """Bundled output for one read — used by the orchestrator and
    available to callers (e.g. the ORF predictor) without a second
    parquet round-trip."""

    read_id: str
    annotation: _StrandAnnotation
    chosen_sequence: str  # the sequence in the chosen orientation
    chosen_quality: str | None


def demux_one_read(
    read_id: str,
    sequence: str,
    quality: str | None,
    construct: LibraryConstruct,
    scorer: Scorer,
    *,
    _params: dict[str, Any] | None = None,
) -> ReadDemuxResult:
    """Demux a single read against ``construct``.

    Public so unit tests / callers that already have a single read can
    skip the table-shaped path. The batch entry point
    :func:`locate_segments` precomputes ``_params`` once for the full
    table.
    """
    params = _params or _extract_construct_params(construct)

    forward = _annotate_strand(sequence, "+", params, scorer)
    if construct.allow_reverse_complement:
        rc_seq = _reverse_complement(sequence)
        rc_qual = quality[::-1] if quality is not None else None
        reverse = _annotate_strand(rc_seq, "-", params, scorer)
        chosen = _pick_orientation(forward, reverse)
    else:
        rc_seq = sequence
        rc_qual = quality
        chosen = forward

    chosen_sequence = sequence if chosen.orientation == "+" else rc_seq
    chosen_quality = quality if chosen.orientation == "+" else rc_qual

    return ReadDemuxResult(
        read_id=read_id,
        annotation=chosen,
        chosen_sequence=chosen_sequence,
        chosen_quality=chosen_quality,
    )


# ──────────────────────────────────────────────────────────────────────
# Batch entry point — produces READ_SEGMENT_TABLE + READ_DEMUX_TABLE
# ──────────────────────────────────────────────────────────────────────


def _emit_segment_rows(
    read_id: str, anno: _StrandAnnotation
) -> Iterator[dict[str, Any]]:
    """Yield READ_SEGMENT_TABLE rows for one read's chosen-orientation
    annotation. Order is 5'→3' on the chosen orientation.
    """
    seg_idx = 0
    o = anno.orientation

    # 5' adapter (SSP)
    if anno.ssp.found and anno.ssp.boundary is not None:
        yield {
            "read_id": read_id,
            "segment_index": seg_idx,
            "segment_kind": "adapter_5p",
            "start": 0,
            "end": anno.ssp.boundary,
            "score": anno.ssp.edit_distance,
            "barcode_id": None,
            "orientation": o,
        }
        seg_idx += 1

    # Transcript
    if anno.transcript_start is not None and anno.transcript_end is not None:
        yield {
            "read_id": read_id,
            "segment_index": seg_idx,
            "segment_kind": "transcript",
            "start": anno.transcript_start,
            "end": anno.transcript_end,
            "score": -1,  # transcripts aren't aligned against a target
            "barcode_id": None,
            "orientation": o,
        }
        seg_idx += 1

    # PolyA
    if anno.polyA.found and anno.polyA.start is not None and anno.polyA.end_inclusive is not None:
        yield {
            "read_id": read_id,
            "segment_index": seg_idx,
            "segment_kind": "polyA",
            "start": anno.polyA.start,
            "end": anno.polyA.end_inclusive + 1,  # half-open
            "score": anno.polyA.length,
            "barcode_id": None,
            "orientation": o,
        }
        seg_idx += 1

    # Barcode (the putative_umi region between polyA and primer3)
    if (
        anno.barcode.found
        and anno.polyA.end_inclusive is not None
        and anno.primer3.found
        and anno.primer3.boundary is not None
    ):
        bc_start = anno.polyA.end_inclusive + 1
        bc_end = bc_start + anno.primer3.boundary
        yield {
            "read_id": read_id,
            "segment_index": seg_idx,
            "segment_kind": "barcode",
            "start": bc_start,
            "end": bc_end,
            "score": anno.barcode.edit_distance,
            "barcode_id": anno.barcode.index,
            "orientation": o,
        }
        seg_idx += 1

    # 3' adapter (primer3)
    if (
        anno.primer3.found
        and anno.polyA.end_inclusive is not None
        and anno.primer3.boundary is not None
    ):
        p3_start = anno.polyA.end_inclusive + 1 + anno.primer3.boundary
        # End is left as start + len(primer3-target); we don't have the
        # exact match-end in chosen-orientation coords from edlib's
        # boundary alone, but the *primer3 segment* is conventionally
        # taken from match start through the end of post_polyA. For S1
        # we record the start position correctly and emit end = start
        # + len(primer_target) as a reasonable approximation.
        yield {
            "read_id": read_id,
            "segment_index": seg_idx,
            "segment_kind": "adapter_3p",
            "start": p3_start,
            "end": p3_start,  # len-zero placeholder; refined post-S1
            "score": anno.primer3.edit_distance,
            "barcode_id": None,
            "orientation": o,
        }
        seg_idx += 1


def locate_segments(
    reads: pa.Table,
    construct: LibraryConstruct,
    *,
    scorer: Scorer | None = None,
    progress_cb=None,
) -> tuple[pa.Table, pa.Table, list[ReadDemuxResult]]:
    """Apply the construct's segment layout to each read.

    Returns ``(segments_table, demux_table, results)`` where:

    - ``segments_table`` conforms to ``READ_SEGMENT_TABLE`` and holds
      one row per located segment per read.
    - ``demux_table`` conforms to ``READ_DEMUX_TABLE`` (without the
      ``sample_id`` populated yet — that's :func:`resolve_demux`'s
      job). One row per read in S1; chimera handling is deferred.
    - ``results`` is the per-read :class:`ReadDemuxResult` list,
      preserved so downstream stages (ORF prediction) can consume the
      chosen-orientation sequence directly without a second pass.

    ``progress_cb`` is reserved for the future ProgressCallback Protocol;
    pass-through for now.
    """
    if scorer is None:
        scorer = HardThresholdScorer()
    params = _extract_construct_params(construct)

    read_ids = reads.column("read_id").to_pylist()
    sequences = reads.column("sequence").to_pylist()
    qualities = (
        reads.column("quality").to_pylist()
        if "quality" in reads.column_names
        else [None] * len(read_ids)
    )

    seg_rows: list[dict[str, Any]] = []
    demux_rows: list[dict[str, Any]] = []
    results: list[ReadDemuxResult] = []

    for read_id, seq, qual in zip(read_ids, sequences, qualities):
        result = demux_one_read(
            read_id, seq, qual, construct, scorer, _params=params
        )
        results.append(result)
        anno = result.annotation

        seg_rows.extend(_emit_segment_rows(read_id, anno))

        # Build the READ_DEMUX_TABLE row. sample_id is left None here
        # — resolve_demux fills it from the barcode_id + Samples.
        demux_rows.append(
            {
                "read_id": read_id,
                "transcript_segment_index": 0,
                "sample_id": None,
                "transcript_start": (
                    anno.transcript_start if anno.transcript_start is not None else -1
                ),
                "transcript_end": (
                    anno.transcript_end if anno.transcript_end is not None else -1
                ),
                "score": (
                    float(anno.barcode.edit_distance)
                    if anno.barcode.found and anno.barcode.edit_distance is not None
                    else None
                ),
                "is_chimera": False,  # S1 collapses everything to single transcript
                "status": anno.classification.status.value,
                "is_fragment": anno.classification.is_fragment,
                "artifact": "none",
            }
        )

    segments_table = (
        pa.Table.from_pylist(seg_rows, schema=READ_SEGMENT_TABLE)
        if seg_rows
        else READ_SEGMENT_TABLE.empty_table()
    )
    demux_table = (
        pa.Table.from_pylist(demux_rows, schema=READ_DEMUX_TABLE)
        if demux_rows
        else READ_DEMUX_TABLE.empty_table()
    )
    return segments_table, demux_table, results


# ──────────────────────────────────────────────────────────────────────
# Sample assignment
# ──────────────────────────────────────────────────────────────────────


def resolve_demux(
    demux_table: pa.Table,
    segments_table: pa.Table,
    *,
    samples,  # Samples
    acquisition_id: int,
) -> pa.Table:
    """Populate ``sample_id`` on a READ_DEMUX_TABLE.

    For each demux row, looks up the read's barcode_id from the
    segments table (the row where ``segment_kind == 'barcode'``), then
    asks the supplied :class:`Samples` container for the sample_id at
    ``(acquisition_id, barcode_id)``. Reads with no barcode get
    ``sample_id = None``; reads whose barcode isn't registered in the
    Samples panel get ``sample_id = None`` too (callers can audit
    these via the segments table).
    """
    # Build read_id → barcode_id map from segments
    bc_by_read: dict[str, int | None] = {}
    if segments_table.num_rows > 0:
        seg_kinds = segments_table.column("segment_kind").to_pylist()
        seg_reads = segments_table.column("read_id").to_pylist()
        seg_bcs = segments_table.column("barcode_id").to_pylist()
        for kind, rid, bc in zip(seg_kinds, seg_reads, seg_bcs):
            if kind == "barcode":
                bc_by_read[rid] = bc

    # Fill sample_id per row
    new_rows: list[dict[str, Any]] = []
    for row in demux_table.to_pylist():
        bc_id = bc_by_read.get(row["read_id"])
        if bc_id is None:
            sample_id: int | None = None
        else:
            candidates = samples.samples_for(acquisition_id, bc_id)
            sample_id = candidates[0] if candidates else None
        new_row = dict(row)
        new_row["sample_id"] = sample_id
        new_rows.append(new_row)

    if not new_rows:
        return demux_table
    return pa.Table.from_pylist(new_rows, schema=READ_DEMUX_TABLE)


__all__ = [
    "ReadDemuxResult",
    "demux_one_read",
    "locate_segments",
    "resolve_demux",
]
