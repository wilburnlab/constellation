"""Insertion folding + end extension in the de novo consensus kernel.

Each test plants exactly one property. The headline one is
``test_centroid_independence_both_directions``: before insertion folding, a
deletion error *in the frame* was unrepairable no matter how many members
carried the base, so the consensus depended on which read became the frame.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

edlib = pytest.importorskip("edlib")

from constellation.sequencing.transcriptome.cluster.denovo import (  # noqa: E402
    consensus as consensus_mod,
)
from constellation.sequencing.transcriptome.cluster.denovo._cigar import (  # noqa: E402
    parse_cigar,
)
from constellation.sequencing.transcriptome.cluster.denovo.consensus import (  # noqa: E402
    COL_EXT_3P,
    COL_EXT_5P,
    COL_INSERTED,
    COL_TEMPLATE,
    MemberSpec,
    centroid_consensus,
    consensus_of_frame,
    frame_consensus,
    frame_of_consensus,
)


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _spec(frame, member, weight=1.0, member_id=-1):
    """Build a MemberSpec from a live edlib HW align, shorter-as-query —
    the same orientation convention verify.py uses."""
    short, long = (frame, member) if len(frame) <= len(member) else (member, frame)
    a = edlib.align(short, long, mode="HW", task="path")
    assert a["editDistance"] >= 0, "fixture members must align"
    return MemberSpec(
        member_seq=member,
        weight=weight,
        cigar=a["cigar"],
        centroid_is_query=(frame == short),
        ref_start=a["locations"][0][0],
        member_id=member_id,
    )


def _specs(frame, members, weight=1.0):
    return [_spec(frame, m, weight, i) for i, m in enumerate(members)]


# ── insertion folding ─────────────────────────────────────────────────


def test_majority_insertion_is_folded():
    """A base missing from the frame that most members carry is restored."""
    rng = np.random.default_rng(17)
    truth = _rand(rng, 700)
    frame = truth[:350] + truth[351:]  # frame is missing base 350
    members = [truth] * 21
    res = frame_consensus(frame, _specs(frame, members))
    assert res.consensus == truth
    assert res.n_inserted_columns == 1


def test_centroid_independence_both_directions():
    """The consensus is the same whichever sequence became the frame.

    This is the property the whole kernel change exists for. Direction (a)
    fails without insertion folding: members carrying a base cannot put it
    back into a frame that lacks it.
    """
    rng = np.random.default_rng(23)
    truth = _rand(rng, 700)
    deleted = truth[:350] + truth[351:]

    # (a) frame lacks the base, members carry it
    a = frame_consensus(deleted, _specs(deleted, [truth] * 21))
    # (b) frame carries the base, members lack it
    b = frame_consensus(truth, _specs(truth, [deleted] * 21))

    assert a.consensus == truth
    assert b.consensus == deleted
    # Each direction follows its own members — that is the point. Flip the
    # members and the two agree.
    c = frame_consensus(truth, _specs(truth, [truth] * 21))
    assert c.consensus == a.consensus == truth


def test_minority_insertion_is_not_folded():
    rng = np.random.default_rng(29)
    truth = _rand(rng, 600)
    frame = truth[:300] + truth[301:]
    members = [truth] * 5 + [frame] * 16  # 5 of 21 carry the base
    res = frame_consensus(frame, _specs(frame, members))
    assert res.consensus == frame
    assert res.n_inserted_columns == 0


def test_homopolymer_insertion_resolves_to_modal_not_maximal_length():
    """A homopolymer settles on the modal inserted length, not the longest
    observation — otherwise every run grows to its worst-case read."""
    rng = np.random.default_rng(3)
    body, tail = _rand(rng, 200), _rand(rng, 200)
    frame = body + "AAA" + tail
    # 10 members insert 1 base, 3 insert 3. Majority insert (13 of 20), but
    # the modal inserted length is 1.
    members = [body + "AAAA" + tail] * 10 + [body + "AAAAAA" + tail] * 3
    members += [frame] * 7
    res = frame_consensus(frame, _specs(frame, members))
    assert res.consensus == body + "AAAA" + tail
    assert res.n_inserted_columns == 1


def test_insertion_denominator_is_covering_members_only():
    """A member whose alignment never reaches a column is not evidence
    against an insertion there."""
    rng = np.random.default_rng(31)
    truth = _rand(rng, 800)
    frame = truth[:400] + truth[401:]
    # 6 members span the column and all carry the base; 14 are truncated well
    # 3' of it and cannot speak. Counting all 20 as the denominator would
    # give 6/20 and refuse to splice.
    members = [truth] * 6 + [truth[600:]] * 14
    res = frame_consensus(frame, _specs(frame, members))
    assert res.n_inserted_columns == 1
    assert res.consensus[:401] == truth[:401]


# ── end extension ─────────────────────────────────────────────────────


def test_planted_5p_extension():
    rng = np.random.default_rng(41)
    truth = _rand(rng, 600)
    frame = truth[40:]
    res = frame_consensus(frame, _specs(frame, [truth] * 20))
    assert res.consensus == truth
    assert res.n_extended_5p == 40
    assert res.column_kind is not None
    assert np.all(res.column_kind[:40] == COL_EXT_5P)
    assert np.all(res.template_of_frame[:40] == -1)


def test_planted_3p_extension():
    rng = np.random.default_rng(43)
    truth = _rand(rng, 600)
    frame = truth[:-35]
    res = frame_consensus(frame, _specs(frame, [truth] * 20))
    assert res.consensus == truth
    assert res.n_extended_3p == 35
    assert np.all(res.column_kind[-35:] == COL_EXT_3P)


def test_minority_5p_extension_does_not_extend():
    """One long member must not donate a whole flank."""
    rng = np.random.default_rng(47)
    truth = _rand(rng, 600)
    frame = truth[40:]
    members = [truth] * 3 + [frame] * 17  # only 3 of 20 reach past the frame
    res = frame_consensus(frame, _specs(frame, members))
    assert res.n_extended_5p == 0
    assert res.consensus == frame


def test_extension_is_graded_by_depth():
    """Extension stops where support stops, rather than following the
    single furthest-reaching member."""
    rng = np.random.default_rng(53)
    truth = _rand(rng, 600)
    frame = truth[100:]
    # 18 members reach 30 bases upstream; 2 reach the full 100.
    members = [truth[70:]] * 18 + [truth] * 2
    res = frame_consensus(frame, _specs(frame, members))
    assert res.n_extended_5p == 30
    assert res.consensus == truth[70:]


# ── coordinate maps ───────────────────────────────────────────────────


def test_frame_maps_round_trip():
    rng = np.random.default_rng(59)
    truth = _rand(rng, 700)
    frame = truth[40:350] + truth[351:-25]  # 5' short, 3' short, one deletion
    res = frame_consensus(frame, _specs(frame, [truth] * 20))

    f = res.pwm.shape[0]
    assert res.winner.shape[0] == f == len(res.frame)
    assert len(res.consensus) <= f
    # c → f → c is the identity.
    assert np.array_equal(
        res.cons_of_frame[res.frame_of_cons], np.arange(len(res.consensus))
    )
    # Template columns keep their provenance, in order, and cover the whole
    # original template exactly once.
    tmpl = res.template_of_frame[res.column_kind == COL_TEMPLATE]
    assert np.array_equal(tmpl, np.arange(len(frame)))
    # Every non-template column is marked as such and has no template origin.
    assert np.all(res.template_of_frame[res.column_kind != COL_TEMPLATE] == -1)
    assert set(np.unique(res.column_kind)) <= {
        COL_TEMPLATE,
        COL_INSERTED,
        COL_EXT_5P,
        COL_EXT_3P,
    }


def test_unresolved_n_column_maps_through_frame_of_cons():
    """An N the frame carries and no member covers stays a consensus column.

    ``np.flatnonzero(winner < 4)`` drops it while the consensus keeps it, so
    open-coding that map shifts every variant downstream of the first N.
    ``frame_of_consensus`` is the shared definition that does not.
    """
    rng = np.random.default_rng(61)
    truth = _rand(rng, 400)
    frame = truth[:10] + "N" + truth[11:]
    # Every member starts 3' of the N, so column 10 has no coverage at all.
    members = [truth[20:]] * 12
    res = frame_consensus(frame, _specs(frame, members), extend_ends=False)

    assert "N" in res.consensus
    assert len(res.frame_of_cons) == len(res.consensus)
    assert np.array_equal(res.frame_of_cons, frame_of_consensus(res.winner))
    # The naive map is short by exactly the N columns — the bug this pins.
    naive = np.flatnonzero(res.winner < 4)
    assert len(naive) == len(res.consensus) - res.consensus.count("N")
    # Round-trip still holds through the correct map.
    assert np.array_equal(
        consensus_of_frame(res.winner)[res.frame_of_cons],
        np.arange(len(res.consensus)),
    )


# ── termination ───────────────────────────────────────────────────────


def test_fixpoint_terminates_when_nothing_reaches_a_majority():
    rng = np.random.default_rng(67)
    truth = _rand(rng, 900)
    # 40 members each insert one base at a different column: no junction ever
    # reaches a majority of its covering members.
    members = []
    for i in range(40):
        p = 100 + i * 15
        members.append(truth[:p] + "A" + truth[p:])
    res = frame_consensus(truth, _specs(truth, members))
    assert res.n_passes == 1
    assert res.n_inserted_columns == 0
    assert res.consensus == truth


def test_max_passes_bounds_the_work():
    rng = np.random.default_rng(71)
    truth = _rand(rng, 500)
    frame = truth[:250] + truth[251:]
    specs = _specs(frame, [truth] * 20)
    for cap in (1, 2, 3):
        res = frame_consensus(frame, specs, max_passes=cap)
        assert res.n_passes <= cap
    # One pass cannot splice at all — it only builds the PWM.
    assert frame_consensus(frame, specs, max_passes=1).n_inserted_columns == 0


def test_large_shared_insertion_terminates():
    """A 300-base insertion carried by every member is a real exon, not a
    runaway — it splices once and settles."""
    rng = np.random.default_rng(73)
    body, tail, exon = _rand(rng, 300), _rand(rng, 300), _rand(rng, 300)
    frame = body + tail
    res = frame_consensus(frame, _specs(frame, [body + exon + tail] * 20))
    assert res.consensus == body + exon + tail
    assert res.n_passes <= 3


# ── returned alignments describe the FINAL frame ──────────────────────


def _assert_alignments_walk(res):
    """Every FrameAlignment must describe the frame that came back, in the
    member-as-query convention it advertises (``I`` consumes the member,
    ``D`` consumes the frame). A swapped orientation or a drifted offset
    shows up as a walk that overruns its interval or matches nothing."""
    fseq = res.frame
    assert res.alignments, "expected alignments to inspect"
    for aln in res.alignments:
        fpos, mpos = aln.frame_start, aln.member_start
        matched = total = 0
        for length, op in parse_cigar(aln.cigar):
            if op in ("=", "X", "M"):
                for k in range(length):
                    total += 1
                    matched += fseq[fpos + k] == aln.member_seq[mpos + k]
                fpos += length
                mpos += length
            elif op == "D":  # consumes the frame only
                fpos += length
            elif op == "I":  # consumes the member only
                mpos += length
        assert fpos == aln.frame_end, "CIGAR does not consume the frame span"
        assert mpos == aln.member_end, "CIGAR does not consume the member span"
        assert 0 <= aln.frame_start and aln.frame_end <= len(fseq)
        assert 0 <= aln.member_start and aln.member_end <= len(aln.member_seq)
        assert matched / max(total, 1) > 0.95, "alignment coordinates drifted"


def test_alignments_consistent_after_splice_and_realign():
    """The splice / shift / re-align path is where a coordinate can silently
    drift: a member on the shift fast path keeps its old CIGAR, a re-aligned
    one gets a fresh CIGAR against a longer frame."""
    rng = np.random.default_rng(89)
    truth = _rand(rng, 900)
    # Short at both ends AND missing an interior base, so folding, both
    # extensions and re-alignment all fire at once.
    frame = truth[60:400] + truth[401:-45]
    members = [truth] * 12 + [truth[100:]] * 8 + [truth[:-200]] * 6
    res = frame_consensus(frame, _specs(frame, members))
    assert res.n_inserted_columns > 0
    assert res.n_extended_5p > 0 and res.n_extended_3p > 0
    _assert_alignments_walk(res)


def test_alignments_normalised_when_the_frame_was_the_edlib_query():
    """A member that is never re-aligned keeps its *cached* CIGAR, and when
    the frame was the shorter edlib query that CIGAR is frame-as-query — I and
    D mean the opposite of what FrameAlignment promises. It has to be
    transposed on the way out, or every allele read off it is wrong."""
    rng = np.random.default_rng(97)
    truth = _rand(rng, 600)
    # Each member drops base 300 and carries a 3' tail, so it is LONGER than
    # the frame (edlib makes the frame the query) and the cached CIGAR really
    # contains an indel op — without one, transposing is a no-op and the test
    # would pass whether or not the bug is present.
    tail = _rand(rng, 40)
    members = [truth[:300] + truth[301:] + tail for _ in range(6)]
    specs = _specs(truth, members)
    assert all(s.centroid_is_query for s in specs), "fixture must be frame-as-query"
    assert "I" in specs[0].cigar, "fixture must exercise an indel op"
    # No folding and no extension ⇒ nobody is re-aligned, so the cached
    # orientation is what reaches the caller.
    res = frame_consensus(truth, specs, fold_insertions=False, extend_ends=False)
    assert res.n_passes == 1
    # The frame's own I became the member's D on the way out.
    assert "D" in res.alignments[0].cigar and "I" not in res.alignments[0].cigar
    _assert_alignments_walk(res)


# ── the legacy shim ───────────────────────────────────────────────────


def test_centroid_consensus_shim_never_grows_the_frame():
    """``centroid_consensus`` is the pre-folding kernel: one pass, no splice,
    frame-length output, and the centroid self-votes."""
    rng = np.random.default_rng(79)
    truth = _rand(rng, 500)
    frame = truth[:250] + truth[251:]
    res = centroid_consensus(frame, 1.0, _specs(frame, [truth] * 20))
    assert res.n_passes == 1
    assert res.n_inserted_columns == res.n_extended_5p == res.n_extended_3p == 0
    assert len(res.frame) == len(frame)
    assert res.consensus == frame  # the deletion is NOT repaired — v1 behaviour


def test_shim_centroid_self_vote_outweighs_members():
    rng = np.random.default_rng(83)
    c = _rand(rng, 300)
    m = c[:50] + ("A" if c[50] != "A" else "C") + c[51:]
    assert centroid_consensus(c, 5.0, _specs(c, [m])).consensus == c
    assert centroid_consensus(c, 1.0, _specs(c, [m], weight=5.0)).consensus == m


# ── fork safety ───────────────────────────────────────────────────────


def test_kernel_does_not_use_torch():
    """The per-cluster pool is fork-based, and a torch op after fork()
    deadlocks on OpenMP because the parent already spawned torch's thread
    pool during minimizer extraction. numpy + edlib only in here."""
    src = inspect.getsource(consensus_mod)
    assert "import torch" not in src
    assert "torch." not in src
    assert not any(
        getattr(v, "__module__", "").startswith("torch")
        for v in vars(consensus_mod).values()
    )
