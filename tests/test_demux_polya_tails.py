"""Poly-A tails fragmented by basecall miscalls must be trimmed whole.

Pins the split-tail defect from the em-kmer handoff (2026-09-22): the
upstream run merge compared the next run's *end* rather than its start,
so a tail interrupted by one miscall was never bridged, only its 3'
fragment was called, and the 5' fragment plus the interrupting base
stayed in the transcript window. That random-length terminal A-run
forms a perfect covariance clique downstream and splits templates on
library chemistry.

The fixtures are built from the worked read in the handoff (the last
31 nt of 3' UTR are real sequence; the tail variants follow the
observed fragmentation buckets). Assertions are on the **transcript
window** — the property clustering depends on — not on the verdict.
"""

from __future__ import annotations

import re

import pytest

from constellation.sequencing.transcriptome.demux.demux import (
    demux_one_read,
    polyA_provenance,
)
from constellation.sequencing.transcriptome.demux.designs import CDNA_WILBURN_V1
from constellation.sequencing.transcriptome.demux.scoring import (
    HardThresholdScorer,
    _merge_runs,
    _walk_residual_polyA,
)

_RC = str.maketrans("ACGT", "TGCA")


def _rc(s: str) -> str:
    return s.translate(_RC)[::-1]


_SSP = CDNA_WILBURN_V1.layout[0].adapters[0].sequence
_PRIMER3 = CDNA_WILBURN_V1.layout[4].adapters[0].sequence
_BC01 = CDNA_WILBURN_V1.layout[3].barcodes[0]

# 250 nt of A-poor coding-like body so the window clears the design's
# 200-nt transcript minimum and holds no accidental AAAA anchor.
_BODY = ("ATGGCTTCCGGTCTGCAGTCCGTGGACGTCTTCCTGGCTGGCGTCAGCCTGCG" * 5)[:250]
# The real 3' UTR end of handoff read d0446171-4ad3-4c85…
_UTR = "ACAAAGTTCAATAAAATTTTGGAAACCTGTT"

# (name, tail, expected window suffix). The expected suffix is the UTR
# end the window must finish on — anything A-rich after it is tail.
CASES = [
    ("clean_tail", "A" * 30, _UTR),
    ("split_1", "A" * 8 + "G" + "A" * 27, _UTR),
    ("split_2", "A" * 8 + "GC" + "A" * 27, _UTR),
    ("split_many", "A" * 6 + "G" + "A" * 5 + "T" + "A" * 4 + "C" + "A" * 20, _UTR),
    ("both_short", "A" * 12 + "G" + "A" * 15, _UTR),
    # A long tail bridged across a miscall exceeds the design's 40-A
    # cap as a span but not as a homopolymer; it must still be called.
    ("long_split", "A" * 20 + "G" + "A" * 25, _UTR),
    # Parity-fixture read 06be6862: a 43-A run the cap rejects on its
    # own, bridged by TG to a 30-A run. Upstream called only the 30.
    ("very_long_split", "A" * 43 + "TG" + "A" * 30, _UTR),
]


def _read(utr: str, tail: str, *, body: str = _BODY) -> str:
    return _SSP + body + utr + tail + _rc(_BC01.sequence) + _PRIMER3


def _window(read: str):
    res = demux_one_read("r", read, None, CDNA_WILBURN_V1, HardThresholdScorer())
    a = res.annotation
    return res, a, res.chosen_sequence[a.transcript_start : a.transcript_end]


def _trailing_a(s: str) -> int:
    return len(s) - len(s.rstrip("A"))


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("name,tail,suffix", CASES, ids=[c[0] for c in CASES])
def test_window_carries_no_tail(name, tail, suffix, strand):
    read = _read(_UTR, tail)
    if strand == "-":
        read = _rc(read)
    res, anno, window = _window(read)
    assert anno.orientation == strand
    assert anno.polyA.found, name
    assert anno.barcode.found and anno.barcode.name == _BC01.name
    assert window.endswith(suffix), window[-40:]
    assert _trailing_a(window) <= 2
    assert window == _BODY + _UTR


def test_worked_example_calls_whole_tail():
    """The handoff's read: 8 A + G + 27 A. Upstream called [start of the
    27-run]; the whole 36-base tail must be called."""
    tail = "A" * 8 + "G" + "A" * 27
    read = _read(_UTR, tail)
    _, anno, _ = _window(read)
    tail_start = len(_SSP) + len(_BODY) + len(_UTR)
    assert anno.polyA.start == tail_start
    assert anno.polyA.length == 36


def test_tail_abutting_primer_off_by_one():
    """No barcode gap-sequence: the tail runs straight into the barcode
    RC. The tail end must stop at the last A, and the barcode still
    resolves."""
    read = _read(_UTR, "A" * 26)
    _, anno, window = _window(read)
    assert anno.polyA.end_inclusive == len(_SSP) + len(_BODY) + len(_UTR) + 25
    assert window == _BODY + _UTR


def test_remnant_abutting_ssp():
    """A window that is *entirely* A-rich next to the SSP: the walk is
    bounded, and never eats into the SSP (which ends in G)."""
    body = "A" * 5 + "G" + _BODY
    read = _SSP + body + "A" * 30 + _rc(_BC01.sequence) + _PRIMER3
    _, anno, window = _window(read)
    assert anno.ssp.found
    assert window.startswith("AAAAAG")
    assert window == body


def test_genomic_pas_is_not_eaten():
    """A polyadenylation signal ~20 nt upstream of the cleavage site is
    UTR, not tail."""
    utr = _UTR[:-20] + "AATAAA" + "GCTTCCGTCTGGCTTC" + "CTGTT"
    read = _read(utr, "A" * 8 + "G" + "A" * 27)
    _, _, window = _window(read)
    assert window.endswith(utr)


def test_barcode_panel_cannot_be_bridged_into():
    """With edge_distance ≤ 2 a tail bridges into the barcode only if a
    barcode RC carries AAAA within its first 3 + 4 bases."""
    ed = CDNA_WILBURN_V1.layout[2].edge_distance
    for bc in CDNA_WILBURN_V1.layout[3].barcodes:
        assert "AAAA" not in _rc(bc.sequence)[: ed + 1 + 4], bc.name


def test_merge_runs_bridges_gap_of_edge_distance():
    # two contiguous fragments [0,7] and [9+g-1 ...]
    for ed in (0, 1, 2, 3):
        for gap in range(0, 5):
            second = 8 + gap
            pairs = [(i, i + 3) for i in range(0, 5)] + [
                (second + i, second + i + 3) for i in range(0, 10)
            ]
            merged = _merge_runs(pairs, edge_distance=ed)
            assert (len(merged) == 1) == (gap <= ed), (ed, gap, merged)


def test_long_homopolymer_still_capped():
    """The design's 40-A cap still rejects an uninterrupted homopolymer."""
    s = HardThresholdScorer()
    v = s.find_polyA("CG" + "A" * 45 + "CG", min_length=15, max_length=40, edge_distance=2)
    assert not v.found


@pytest.mark.parametrize(
    "seq,start,expect",
    [
        ("CCTGTT" + "AAAAAA" + "G" + "A" * 20, 13, 6),  # remnant absorbed
        ("CCTGTT" + "AA" + "G" + "A" * 20, 9, 9),  # 2-run < min_run: stop
        ("CCTGTT" + "AAA" + "GCT" + "AAAA" + "A" * 20, 12, 12),  # gap 3 > max
        ("CCTG" + "AAAA" + "C" + "AAAA" + "T" + "A" * 20, 14, 4),  # chains
    ],
)
def test_walk_rule(seq, start, expect):
    assert seq[start] == "A" and seq[start - 1] != "A"
    got = _walk_residual_polyA(seq, start, max_gap=2, min_run=3, max_walk=60)
    assert got == expect, (seq[:got], seq[got:])


def test_walk_is_bounded():
    seq = ("AAAAC" * 30) + "A" * 20
    start = 150
    got = _walk_residual_polyA(seq, start, max_gap=2, min_run=3, max_walk=60)
    assert start - got <= 60


def test_provenance_marks_corrected_merge():
    p = polyA_provenance(CDNA_WILBURN_V1)
    assert p["polyA_merge"] == "gap"
    assert p["polyA_edge_distance"] == 2
    assert p["polyA_residual_max_walk"] > 0
    assert not re.search("None", str(p["polyA_residual_min_run"]))
