"""Per-segment scoring — pluggable Scorer Protocol + concrete scorers.

The demux pipeline orchestrator (``transcriptome.demux``) calls into a
``Scorer`` to evaluate each structural segment in a read: polyA tail,
5' adapter, 3' adapter, barcode panel. The Protocol seam lets us swap
between hard-threshold mode (NanoporeAnalysis-parity edit-distance
checks; this module's :class:`HardThresholdScorer`) and a probabilistic
mode (Session 2 — :class:`ProbabilisticScorer` ships then) without
touching the demux algorithm itself.

Scorer outputs are flat dataclasses (``PolyAVerdict``, ``AdapterVerdict``,
``BarcodeVerdict``). Each carries enough boundary information for the
orchestrator to slice the read into the next segment's region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from constellation.sequencing.align.locate import locate_substring
from constellation.sequencing.transcriptome.adapters import Barcode


# ──────────────────────────────────────────────────────────────────────
# Verdict types — flat dataclasses returned by Scorer methods
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PolyAVerdict:
    """Outcome of a polyA-tail location attempt.

    ``start`` / ``end_inclusive`` are 0-based offsets into the read
    sequence; ``end_inclusive`` is the position of the last A in the
    tail (NanoporeAnalysis convention — `post_polyA = seq[end+1:]`).
    ``length`` = ``end_inclusive - start + 1`` for convenience.
    """

    found: bool
    start: int | None = None
    end_inclusive: int | None = None
    length: int | None = None


@dataclass(frozen=True)
class AdapterVerdict:
    """Outcome of an adapter (5' SSP / 3' primer) location attempt.

    For a 5' adapter (located in pre_polyA), ``boundary`` is the 0-based
    position of the *first* base after the matched adapter end — i.e.
    the start of the transcript window. For a 3' adapter (located in
    post_polyA), ``boundary`` is the 0-based start of the matched
    adapter — i.e. the end of the barcode region (putative_umi =
    post_polyA[:boundary]).
    """

    found: bool
    edit_distance: int | None = None
    boundary: int | None = None


@dataclass(frozen=True)
class BarcodeVerdict:
    """Outcome of barcode-panel scoring.

    ``index`` is the 0-based position within the supplied barcode tuple
    of the winning barcode. ``delta_edit_distance`` is the difference
    between the best and second-best edit distances (None when only
    one barcode survived the threshold filter); a small delta flags
    ambiguous matches that downstream code may want to treat as
    suspect.
    """

    found: bool
    index: int | None = None
    name: str | None = None
    edit_distance: int | None = None
    delta_edit_distance: int | None = None


# ──────────────────────────────────────────────────────────────────────
# Scorer Protocol — what demux.py calls into
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class Scorer(Protocol):
    """Pluggable scoring backend for the demux pipeline.

    Hard-mode and probabilistic-mode implementations differ in how they
    rank candidates and what edit-distance / log-odds budgets they
    accept; the orchestrator never needs to know which is in use.
    """

    def find_polyA(
        self,
        sequence: str,
        *,
        min_length: int,
        max_length: int | None,
        edge_distance: int,
    ) -> PolyAVerdict: ...

    def find_5p_adapter(
        self,
        region: str | None,
        target: str,
        *,
        max_distance: int,
    ) -> AdapterVerdict: ...

    def find_3p_adapter(
        self,
        region: str | None,
        target: str,
        *,
        max_distance: int,
    ) -> AdapterVerdict: ...

    def score_barcode(
        self,
        putative_umi: str | None,
        barcodes: tuple[Barcode, ...],
        *,
        max_distance: int,
    ) -> BarcodeVerdict: ...


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def _merge_runs(
    pairs: list[tuple[int, int]],
    *,
    edge_distance: int,
) -> list[tuple[int, int]]:
    """Merge overlapping / adjacent ``(start, end_inclusive)`` runs.

    Two runs are merged when the second's end is within
    ``current_end + 1 + edge_distance`` of the running end — i.e. they
    overlap or are separated by at most ``edge_distance`` mismatches.
    Mirrors NanoporeAnalysis's ``merge_overlapped_indices`` byte-for-
    byte (same iteration order, same boundary check) so polyA detection
    parity is preserved.

    Input is expected sorted ascending by start (caller responsibility).
    """
    if not pairs:
        return []
    out: list[tuple[int, int]] = []
    cur_start, cur_end = pairs[0]
    for start, end in pairs[1:]:
        if end <= cur_end + 1 + edge_distance:
            # overlap or near-adjacency → merge
            cur_end = max(cur_end, end)
        else:
            out.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    out.append((cur_start, cur_end))
    return out


# ──────────────────────────────────────────────────────────────────────
# HardThresholdScorer — NanoporeAnalysis-parity hard cutoffs
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardThresholdScorer:
    """Hard-threshold scorer matching NanoporeAnalysis ``align.py``
    semantics byte-for-byte (with the ``_fixed1`` correction that
    filters edlib's ``-1`` no-match sentinels before picking the
    winning barcode).

    PolyA: exact 'AAAA' anchor + adjacent-run merge with edge-distance
    tolerance + length filter + last-run wins. SSP / primer3 / barcode:
    edlib HW with hard ``max_distance``; barcode picks the smallest
    surviving edit distance over the panel (filtering ``-1`` no-match
    sentinels first).
    """

    polyA_query: str = "AAAA"
    """Anchor query for polyA detection; NanoporeAnalysis uses 'AAAA'."""

    def find_polyA(
        self,
        sequence: str,
        *,
        min_length: int,
        max_length: int | None,
        edge_distance: int,
    ) -> PolyAVerdict:
        if not sequence:
            return PolyAVerdict(found=False)
        all_matches = locate_substring(
            self.polyA_query, sequence, max_distance=0, return_all=True
        )
        if not all_matches:
            return PolyAVerdict(found=False)
        # Convert half-open `end` back to inclusive end for parity with
        # NanoporeAnalysis's merge bookkeeping.
        pairs = sorted(
            ((m.start, m.end - 1) for m in all_matches),
            key=lambda p: (p[0], p[1]),
        )
        merged = _merge_runs(pairs, edge_distance=edge_distance)
        cap = max_length if max_length is not None else (1 << 30)
        valid = [
            (s, e) for s, e in merged if min_length <= (e - s + 1) <= cap
        ]
        if not valid:
            return PolyAVerdict(found=False)
        # NanoporeAnalysis takes the LAST (largest-start) run.
        valid.sort(key=lambda p: p[0], reverse=True)
        s, e = valid[0]
        return PolyAVerdict(
            found=True,
            start=s,
            end_inclusive=e,
            length=e - s + 1,
        )

    def find_5p_adapter(
        self,
        region: str | None,
        target: str,
        *,
        max_distance: int,
    ) -> AdapterVerdict:
        """Locate a 5' adapter (SSP) in pre_polyA. Boundary is the
        post-match offset (start of the transcript window).

        NanoporeAnalysis's ``parse_adapter5_seq`` takes
        ``trim_idx = max([loc[1] for loc in locations]) + 1`` (the
        last inclusive-end + 1). Our half-open convention makes this
        ``max(m.end for m in matches)``.
        """
        if region is None or not region:
            return AdapterVerdict(found=False)
        matches = locate_substring(
            target, region, max_distance=max_distance, return_all=True
        )
        if not matches:
            return AdapterVerdict(found=False)
        boundary = max(m.end for m in matches)
        return AdapterVerdict(
            found=True,
            edit_distance=matches[0].edit_distance,
            boundary=boundary,
        )

    def find_3p_adapter(
        self,
        region: str | None,
        target: str,
        *,
        max_distance: int,
    ) -> AdapterVerdict:
        """Locate a 3' adapter (primer3) in post_polyA. Boundary is the
        match-start offset (end of the barcode / putative_umi window).

        NanoporeAnalysis's ``parse_adapter3_seq`` takes
        ``post_start = min([loc[0] for loc in primer_locations])``.
        """
        if region is None or not region:
            return AdapterVerdict(found=False)
        matches = locate_substring(
            target, region, max_distance=max_distance, return_all=True
        )
        if not matches:
            return AdapterVerdict(found=False)
        boundary = min(m.start for m in matches)
        return AdapterVerdict(
            found=True,
            edit_distance=matches[0].edit_distance,
            boundary=boundary,
        )

    def score_barcode(
        self,
        putative_umi: str | None,
        barcodes: tuple[Barcode, ...],
        *,
        max_distance: int,
    ) -> BarcodeVerdict:
        """Score each barcode's RC against the putative_umi region;
        pick the smallest surviving edit distance.

        Replicates NanoporeAnalysis's ``score_umi`` semantics with the
        ``_fixed1`` correction: edlib returns ``editDistance=-1`` as
        the no-match sentinel, NOT None, so the post-edlib filter must
        explicitly drop those before sorting; otherwise no-match reads
        get assigned to whichever barcode happens to be first in panel
        order (the original-parquet bug). Concretely we use
        ``locate_substring(..., max_distance)`` which already performs
        the filter at edlib's level — a None return means "edlib found
        nothing within budget", which we treat as no match.
        """
        if putative_umi is None or not putative_umi or not barcodes:
            return BarcodeVerdict(found=False)
        results: list[tuple[int, Barcode, int]] = []
        for i, bc in enumerate(barcodes):
            rc = _reverse_complement(bc.sequence)
            m = locate_substring(rc, putative_umi, max_distance=max_distance)
            if m is None:
                continue
            results.append((i, bc, m.edit_distance))
        if not results:
            return BarcodeVerdict(found=False)
        # Smallest edit distance wins; ties broken by panel order
        # (stable sort).
        results.sort(key=lambda r: r[2])
        best_i, best_bc, best_ed = results[0]
        delta: int | None = None
        if len(results) >= 2:
            delta = results[1][2] - best_ed
        return BarcodeVerdict(
            found=True,
            index=best_i,
            name=best_bc.name,
            edit_distance=best_ed,
            delta_edit_distance=delta,
        )


# ──────────────────────────────────────────────────────────────────────
# ProbabilisticScorer — Session 2 placeholder
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbabilisticScorer:
    """Probabilistic scorer with calibratable per-segment models.

    Pending Session 2: distribution selection is empirical (gated on
    larger-dataset exploration), and the boundary-uncertainty machinery
    that recovers information from the polyA-barcode boundary
    ambiguity (`SegmentMatch.next_start_candidates` etc.) lands here.
    """

    def find_polyA(
        self,
        sequence: str,  # noqa: ARG002 - stub
        *,
        min_length: int,  # noqa: ARG002
        max_length: int | None,  # noqa: ARG002
        edge_distance: int,  # noqa: ARG002
    ) -> PolyAVerdict:
        raise NotImplementedError(
            "ProbabilisticScorer pending Session 2 — see plan file"
        )

    def find_5p_adapter(
        self,
        region: str | None,  # noqa: ARG002
        target: str,  # noqa: ARG002
        *,
        max_distance: int,  # noqa: ARG002
    ) -> AdapterVerdict:
        raise NotImplementedError(
            "ProbabilisticScorer pending Session 2 — see plan file"
        )

    def find_3p_adapter(
        self,
        region: str | None,  # noqa: ARG002
        target: str,  # noqa: ARG002
        *,
        max_distance: int,  # noqa: ARG002
    ) -> AdapterVerdict:
        raise NotImplementedError(
            "ProbabilisticScorer pending Session 2 — see plan file"
        )

    def score_barcode(
        self,
        putative_umi: str | None,  # noqa: ARG002
        barcodes: tuple[Barcode, ...],  # noqa: ARG002
        *,
        max_distance: int,  # noqa: ARG002
    ) -> BarcodeVerdict:
        raise NotImplementedError(
            "ProbabilisticScorer pending Session 2 — see plan file"
        )


__all__ = [
    "AdapterVerdict",
    "BarcodeVerdict",
    "HardThresholdScorer",
    "PolyAVerdict",
    "ProbabilisticScorer",
    "Scorer",
]
