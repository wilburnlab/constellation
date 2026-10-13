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
from constellation.sequencing.transcriptome.demux.adapters import Barcode


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
        residual_max_gap: int = 0,
        residual_min_run: int = 1,
        residual_max_walk: int = 0,
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
    """Merge ``(start, end_inclusive)`` runs separated by at most
    ``edge_distance`` non-anchor bases.

    Two runs are merged when the second's **start** lies within
    ``current_end + 1 + edge_distance`` — i.e. they overlap, abut, or
    are separated by a gap of ≤ ``edge_distance`` bases.

    **Deliberate divergence from NanoporeAnalysis.** Upstream's
    ``merge_overlapped_indices`` compares the next run's *end* against
    that bound. With 4-mer ``AAAA`` anchors a run's end is its start
    + 3, so the effective gap tolerance is ``edge_distance − 3``: at
    the shipped ``edge_distance = 1`` a tail interrupted by a single
    miscall (``AAAAAAAA G AAAA…``) was never bridged, only the last
    fragment was called, and the 5' fragment plus the interrupting
    base stayed in the transcript window — a random-length terminal
    A-run that the de novo M-step then split templates on.

    Input is expected sorted ascending by start (caller responsibility).
    """
    return [(s, e) for s, e, _ in _merge_runs_with_fragments(
        pairs, edge_distance=edge_distance
    )]


def _merge_runs_with_fragments(
    pairs: list[tuple[int, int]],
    *,
    edge_distance: int,
) -> list[tuple[int, int, int]]:
    """:func:`_merge_runs`, also reporting each merged run's **3'-most
    uninterrupted fragment** length (a stretch covered by overlapping /
    abutting anchors).

    ``PolyASlot.max_length`` is judged on that fragment. It is the one
    the upstream merge would have called in isolation, so a tail made
    of a single fragment is capped exactly as before, while a long tail
    bridged across a miscall (43 A + TG + 30 A, seen on the parity
    fixture) is not rejected merely because it is now called whole.
    """
    if not pairs:
        return []
    # Pass 1: contiguous fragments (anchors that overlap or abut).
    frags: list[tuple[int, int]] = []
    fs, fe = pairs[0]
    for start, end in pairs[1:]:
        if start <= fe + 1:
            fe = max(fe, end)
        else:
            frags.append((fs, fe))
            fs, fe = start, end
    frags.append((fs, fe))
    # Pass 2: bridge gaps of ≤ edge_distance bases between fragments.
    out: list[tuple[int, int, int]] = []
    cs, ce = frags[0]
    last = ce - cs + 1
    for start, end in frags[1:]:
        if start <= ce + 1 + edge_distance:
            ce = max(ce, end)
            last = end - start + 1
        else:
            out.append((cs, ce, last))
            cs, ce = start, end
            last = end - start + 1
    out.append((cs, ce, last))
    return out


def _walk_residual_polyA(
    sequence: str,
    start: int,
    *,
    max_gap: int,
    min_run: int,
    max_walk: int,
) -> int:
    """Move a poly-A start 5'-ward through an A-rich remnant.

    Repeatedly: absorb consecutive ``A``; then absorb up to ``max_gap``
    non-A bases **only if** at least ``min_run`` consecutive ``A`` lie
    immediately 5' of them. Never moves more than ``max_walk`` bases.
    Returns the new start.

    This catches tails fragmented by more miscalls than the merge
    bridges. It is a canonicalisation rather than a measurement: a
    genomic A-run abutting the cleavage site cannot be told apart from
    a tail remnant, and every read of the transcript is trimmed the
    same way — which is the property clustering depends on.
    """
    lo = max(0, start - max_walk)
    pos = start
    while True:
        i = pos - 1
        while i >= lo and sequence[i] == "A":
            i -= 1
        pos = i + 1
        # Try to bridge a short non-A gap into a further A-run.
        j = i
        gap = 0
        while j >= lo and sequence[j] != "A" and gap < max_gap:
            j -= 1
            gap += 1
        if gap == 0 or j < lo or sequence[j] != "A":
            return pos
        run = 0
        k = j
        while k >= lo and sequence[k] == "A":
            run += 1
            k -= 1
        if run < min_run:
            return pos
        pos = j + 1  # the loop head absorbs the run


# ──────────────────────────────────────────────────────────────────────
# HardThresholdScorer — NanoporeAnalysis-parity hard cutoffs
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardThresholdScorer:
    """Hard-threshold scorer matching NanoporeAnalysis ``align.py``
    semantics byte-for-byte (with the ``_fixed1`` correction that
    filters edlib's ``-1`` no-match sentinels before picking the
    winning barcode).

    PolyA: exact 'AAAA' anchor + gap-bridging run merge (a deliberate
    divergence from upstream — see :func:`_merge_runs`) + length filter
    + last-run wins + optional 5' residual-remnant walk. SSP / primer3 / barcode:
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
        residual_max_gap: int = 0,
        residual_min_run: int = 1,
        residual_max_walk: int = 0,
    ) -> PolyAVerdict:
        """Locate the poly-A tail.

        Anchors are exact ``AAAA`` hits; runs separated by ≤
        ``edge_distance`` non-A bases merge (see :func:`_merge_runs`
        for the deliberate divergence from NanoporeAnalysis). A merged
        run is valid when its full span is ≥ ``min_length`` and its
        3'-most *uninterrupted* fragment is ≤ ``max_length`` (the
        fragment upstream would have judged; see
        :func:`_merge_runs_with_fragments`). The last valid run wins. When
        ``residual_max_walk > 0`` its start is then walked 5'-ward
        through any A-rich remnant (:func:`_walk_residual_polyA`), so
        the transcript window cut at ``start`` carries no tail.
        """
        if not sequence:
            return PolyAVerdict(found=False)
        all_matches = locate_substring(
            self.polyA_query, sequence, max_distance=0, return_all=True
        )
        if not all_matches:
            return PolyAVerdict(found=False)
        # Convert half-open `end` back to inclusive end for the merge
        # bookkeeping.
        pairs = sorted(
            ((m.start, m.end - 1) for m in all_matches),
            key=lambda p: (p[0], p[1]),
        )
        merged = _merge_runs_with_fragments(pairs, edge_distance=edge_distance)
        cap = max_length if max_length is not None else (1 << 30)
        valid = [
            (s, e)
            for s, e, last_frag in merged
            if (e - s + 1) >= min_length and last_frag <= cap
        ]
        if not valid:
            return PolyAVerdict(found=False)
        # NanoporeAnalysis takes the LAST (largest-start) run.
        valid.sort(key=lambda p: p[0], reverse=True)
        s, e = valid[0]
        if residual_max_walk > 0:
            s = _walk_residual_polyA(
                sequence,
                s,
                max_gap=residual_max_gap,
                min_run=residual_min_run,
                max_walk=residual_max_walk,
            )
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
        order (the original-parquet bug).

        ``delta_edit_distance`` is computed across the FULL panel — the
        gap between the best match and the *true* second-best,
        regardless of whether the second-best fits within
        ``max_distance``. This matches the intuition the future
        ProbabilisticScorer needs: a 0-vs-1 gap is much weaker
        evidence than a 0-vs-6 gap, even if both look like a "Complete"
        match under the hard threshold. Each barcode is scored with a
        generous ``barcode_length``-budget so edlib returns the true
        edit distance instead of a no-match sentinel.
        """
        if putative_umi is None or not putative_umi or not barcodes:
            return BarcodeVerdict(found=False)
        # Generous budget so we get the true edit distance for every
        # barcode (not just those within `max_distance`). For 16-nt
        # barcodes against a 16-17 nt region, max budget = 16 covers
        # the worst case.
        budget = max(len(bc.sequence) for bc in barcodes)
        all_scores: list[tuple[int, Barcode, int]] = []
        for i, bc in enumerate(barcodes):
            rc = _reverse_complement(bc.sequence)
            m = locate_substring(rc, putative_umi, max_distance=budget)
            if m is None:
                continue
            all_scores.append((i, bc, m.edit_distance))
        if not all_scores:
            return BarcodeVerdict(found=False)
        all_scores.sort(key=lambda r: r[2])
        best_i, best_bc, best_ed = all_scores[0]
        delta: int | None = (
            all_scores[1][2] - best_ed if len(all_scores) >= 2 else None
        )
        # Apply the user's threshold to the FOUND decision only.
        if best_ed > max_distance:
            return BarcodeVerdict(found=False)
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
        residual_max_gap: int = 0,  # noqa: ARG002
        residual_min_run: int = 1,  # noqa: ARG002
        residual_max_walk: int = 0,  # noqa: ARG002
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
