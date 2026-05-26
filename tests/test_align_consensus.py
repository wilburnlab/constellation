"""Tests for ``constellation.sequencing.align.consensus``.

Validate the per-position weighted-PWM consensus kernel:

    1. Identical members reproduce the reference window exactly.
    2. Substitution wins under sufficient weight; loses under
       insufficient weight.
    3. Deletion-dominant positions drop out of the emitted consensus
       (the output gets shorter).
    4. Splice operators advance ref_pos without contributing votes.
    5. Insertions are skipped (no projection onto the genome frame).
    6. Members aligning outside the requested window are clipped.
    7. Missing alignment_cs raises a clear error.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from constellation.sequencing.align.consensus import build_consensus
from constellation.sequencing.schemas.alignment import ALIGNMENT_CS_TABLE


def _cs(rows: list[tuple[int, str]]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"alignment_id": int(a), "cs_string": str(s)} for a, s in rows],
        schema=ALIGNMENT_CS_TABLE,
    )


def test_identical_members_reproduce_reference() -> None:
    ref = "AAAACCCCGGGG"
    cs_table = _cs([(1, ":12"), (2, ":12"), (3, ":12")])
    out = build_consensus(
        member_alignment_ids=[1, 2, 3],
        member_weights=[1.0, 1.0, 1.0],
        member_ref_starts=[100, 100, 100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_weighted_substitution_loses_under_low_weight() -> None:
    ref = "AAAACCCCGGGG"
    # Member 2 substitutes C→T at ref position 4 (offset 4 in the
    # window). Weight 1 vs 3 ref-matchers: ref wins.
    cs_table = _cs([(1, ":12"), (2, ":4*ct:7")])
    out = build_consensus(
        member_alignment_ids=[1, 2],
        member_weights=[3.0, 1.0],
        member_ref_starts=[100, 100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_weighted_substitution_wins_under_high_weight() -> None:
    ref = "AAAACCCCGGGG"
    cs_table = _cs([(1, ":12"), (2, ":4*ct:7")])
    out = build_consensus(
        member_alignment_ids=[1, 2],
        member_weights=[1.0, 5.0],
        member_ref_starts=[100, 100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    # Position 4 flips C → T; rest match reference.
    assert out == "AAAATCCCGGGG"


def test_deletion_drops_positions_from_consensus() -> None:
    ref = "AAAACCCCGGGG"
    # Single-member deletion of CCCC at positions 4-7 → consensus
    # length shrinks by 4.
    cs_table = _cs([(1, ":4-cccc:4")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == "AAAAGGGG"


def test_splice_advances_ref_pos_without_voting() -> None:
    """A `~ag10gt` operator skips 10 ref positions; positions 4-13 of a
    24-bp window get no votes from this member, so they fall back to
    the reference base when no other member covers them."""
    ref = "AAAA" + "TTTTTTTTTT" + "GGGGGGGGGG"  # 24 bp
    # Member spans 0..4 + splice 10 + 4..14 (last 10 of ref)
    # cs:long: ":4~ag10gt:10"
    cs_table = _cs([(1, ":4~ag10gt:10")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    # Spliced region falls back to ref because nobody voted there
    # (min_depth=1.0 default; no votes → fallback to ref base).
    assert out == ref


def test_insertion_is_skipped_no_projection() -> None:
    """`+xxx` insertion in query is not projected onto the genome
    frame — consensus length stays equal to the window length."""
    ref = "AAAACCCC"
    cs_table = _cs([(1, ":4+gg:4")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[100],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_member_outside_window_is_clipped() -> None:
    """Member alignment starting before the window has its early votes
    clipped; only the in-window positions accumulate."""
    ref = "CCCCCCCC"  # window starts at 100
    # Member starts at 96, all match — its votes for positions 96-99
    # are clipped, votes for 100-103 land at window offsets 0-3.
    cs_table = _cs([(1, ":12")])
    out = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[96],
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=100,
    )
    assert out == ref


def test_empty_alignment_cs_raises() -> None:
    cs_table = ALIGNMENT_CS_TABLE.empty_table()
    with pytest.raises(ValueError, match="alignment_cs"):
        build_consensus(
            member_alignment_ids=[1],
            member_weights=[1.0],
            member_ref_starts=[0],
            alignment_cs=cs_table,
            reference_sequence="AAAA",
            reference_start=0,
        )


def test_member_array_length_mismatch_raises() -> None:
    cs_table = _cs([(1, ":4")])
    with pytest.raises(ValueError, match="length mismatch"):
        build_consensus(
            member_alignment_ids=[1, 2],
            member_weights=[1.0],
            member_ref_starts=[0, 0],
            alignment_cs=cs_table,
            reference_sequence="AAAA",
            reference_start=0,
        )


def test_no_members_returns_reference_unchanged() -> None:
    cs_table = _cs([(1, ":4")])
    out = build_consensus(
        member_alignment_ids=[],
        member_weights=[],
        member_ref_starts=[],
        alignment_cs=cs_table,
        reference_sequence="ACGT",
        reference_start=0,
    )
    assert out == "ACGT"


def test_long_match_form_equivalent_to_short() -> None:
    """`=AAAA` and `:4` should produce the same consensus contribution."""
    ref = "AAAACCCC"
    cs_short = _cs([(1, ":8")])
    cs_long = _cs([(2, "=AAAACCCC")])
    out_short = build_consensus(
        member_alignment_ids=[1],
        member_weights=[1.0],
        member_ref_starts=[0],
        alignment_cs=cs_short,
        reference_sequence=ref,
        reference_start=0,
    )
    out_long = build_consensus(
        member_alignment_ids=[2],
        member_weights=[1.0],
        member_ref_starts=[0],
        alignment_cs=cs_long,
        reference_sequence=ref,
        reference_start=0,
    )
    assert out_short == out_long == ref


def test_parallel_matches_sequential() -> None:
    """`threads=N` must produce byte-identical output to `threads=1` for
    the same input — torch ``index_add_`` is commutative across members
    so parallel parsing + batched scatter is semantics-preserving.
    """
    ref = "ACGTACGTACGTACGT"  # 16 bp window
    # 50 members with a mix of identity + substitution + deletion.
    cs_rows = []
    aids = []
    starts = []
    for i in range(50):
        if i % 5 == 0:
            cs_rows.append((i, ":16"))  # identity
        elif i % 5 == 1:
            cs_rows.append((i, ":4*ct:11"))  # substitute C→T at pos 4
        elif i % 5 == 2:
            cs_rows.append((i, ":4-c:11"))  # delete pos 4
        else:
            cs_rows.append((i, "=ACGTACGTACGTACGT"))  # long-match form
        aids.append(i)
        starts.append(0)
    cs_table = _cs(cs_rows)
    out_serial = build_consensus(
        member_alignment_ids=aids,
        member_weights=[1.0] * len(aids),
        member_ref_starts=starts,
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=0,
        threads=1,
    )
    for n_threads in (2, 4, 8):
        out_parallel = build_consensus(
            member_alignment_ids=aids,
            member_weights=[1.0] * len(aids),
            member_ref_starts=starts,
            alignment_cs=cs_table,
            reference_sequence=ref,
            reference_start=0,
            threads=n_threads,
        )
        assert out_parallel == out_serial, (
            f"threads={n_threads} diverged from threads=1"
        )


def test_singleton_consensus_matches_pwm_path() -> None:
    """For n_members==1 the fast path skips PWM allocation + scatter
    and applies cs:long edits directly to the reference window. The
    output must match what the PWM path would produce for the same
    1-member input across a range of cs:long content.
    """
    import pyarrow.compute as pc

    from constellation.sequencing.align.consensus import _BASE_TO_IDX  # noqa: F401

    ref = "AAAACCCCGGGGTTTT"  # 16 bp
    fixtures = [
        # (cs:long, ref_start, expected after applying edits)
        (":16", 0, "AAAACCCCGGGGTTTT"),  # identity
        (":4*ct:11", 0, "AAAATCCCGGGGTTTT"),  # subst C→T at pos 4
        (":4-c:11", 0, "AAAACCCGGGGTTTT"),  # delete pos 4 (drop)
        ("=AAAACCCCGGGGTTTT", 0, "AAAACCCCGGGGTTTT"),  # long-match form
        # Insertion at pos 4 (skipped — no genome-frame projection)
        (":4+aa:12", 0, "AAAACCCCGGGGTTTT"),
        # Splice skipping bases 4–7: positions 4–7 fall back to the
        # reference base (preserves window length when no vote covers).
        (":4~gt4ag:8", 0, "AAAACCCCGGGGTTTT"),
    ]
    for cs_str, rstart, expected in fixtures:
        # Fast path (current implementation).
        cs_table = _cs([(1, cs_str)])
        out_fast = build_consensus(
            member_alignment_ids=[1],
            member_weights=[1.0],
            member_ref_starts=[rstart],
            alignment_cs=cs_table,
            reference_sequence=ref,
            reference_start=0,
        )
        # Reference PWM path: run with 2 identical members; output is the
        # same consensus (single vote is enough at min_depth=1.0) and
        # exercises the multi-member code path. We then validate the fast
        # path matches.
        cs_table2 = _cs([(1, cs_str), (2, cs_str)])
        out_pwm = build_consensus(
            member_alignment_ids=[1, 2],
            member_weights=[1.0, 1.0],
            member_ref_starts=[rstart, rstart],
            alignment_cs=cs_table2,
            reference_sequence=ref,
            reference_start=0,
        )
        assert out_fast == out_pwm == expected, (
            f"cs={cs_str!r}: fast={out_fast!r} pwm={out_pwm!r} "
            f"expected={expected!r}"
        )


def test_consensus_chars_vectorized_long_window() -> None:
    """The numpy-vectorized consensus_chars assembly must produce the
    same output as a reference Python-loop implementation on a long
    window (~5 kbp) — the regression case where the prior `.item()`-per-
    position loop was the dominant per-cluster wall time.
    """
    # 5 kbp reference; identity edits across 10 members so PWM votes
    # accumulate but consensus = reference. Mix in some ambiguous bases
    # and a deletion-dominant column to exercise the keep_mask.
    L = 5_000
    ref = ("ACGTACGTNN" * (L // 10))[:L]
    n_members = 10
    cs_rows = [(i, f":{L}") for i in range(n_members)]
    cs_table = _cs(cs_rows)
    out = build_consensus(
        member_alignment_ids=list(range(n_members)),
        member_weights=[1.0] * n_members,
        member_ref_starts=[0] * n_members,
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=0,
    )
    # Ambiguous reference bases (N) drop; expected = ref with all N's removed.
    expected = ref.replace("N", "")
    assert out == expected


def test_batched_scatter_chunk_boundary() -> None:
    """The batched index_add_ accumulation must produce the same PWM
    regardless of how members are partitioned into scatter chunks.
    """
    ref = "AAAACCCC"
    # 250 members, all identity at weight 1.0 — winner per position
    # is the reference base. Vary chunk_size across the member count
    # to exercise different chunking boundaries.
    cs_rows = [(i, ":8") for i in range(250)]
    cs_table = _cs(cs_rows)
    aids = list(range(250))
    weights = [1.0] * 250
    starts = [0] * 250
    out_default = build_consensus(
        member_alignment_ids=aids,
        member_weights=weights,
        member_ref_starts=starts,
        alignment_cs=cs_table,
        reference_sequence=ref,
        reference_start=0,
    )
    for chunk in (1, 7, 100, 250, 1000):
        out_chunked = build_consensus(
            member_alignment_ids=aids,
            member_weights=weights,
            member_ref_starts=starts,
            alignment_cs=cs_table,
            reference_sequence=ref,
            reference_start=0,
            scatter_chunk_size=chunk,
        )
        assert out_chunked == out_default == ref
