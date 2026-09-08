"""Stage 3 of the ORF-anchored EM: consensus, haplotypes, ORF support gate.

The headline cases are the two the design exists for: a homopolymer deletion
that produces a readthrough ORF must surface as its own node with its own read
count, and an ORF must not extend through template flanks that only one read
supports.
"""

from __future__ import annotations

import numpy as np
import pytest

edlib = pytest.importorskip("edlib")

from constellation.sequencing.transcriptome.cluster.denovo.consensus import (  # noqa: E402
    MemberSpec,
    frame_consensus,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import (  # noqa: E402
    best_sense_orf,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.mstep import (  # noqa: E402
    certified_columns,
    gated_orf,
    refine_template,
    specs_from_assignments,
)


_CODONS = ["GCT", "TGT", "GAT", "GAA", "TTT", "GGT", "CAT", "ATC", "AAA", "CTG"]
_STOPS = ("TAA", "TAG", "TGA")


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _flank(rng, n):
    return _rand(rng, n).replace("ATG", "ATC")


def _spec(frame, member, weight=1.0, member_id=-1):
    short, long = (frame, member) if len(frame) <= len(member) else (member, frame)
    a = edlib.align(short, long, mode="HW", task="path")
    assert a["editDistance"] >= 0
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


# ── the support gate ──────────────────────────────────────────────────


def test_certified_columns_tracks_base_coverage_not_total():
    """At a ragged end many members vote a deletion, so total coverage stays
    flat while base coverage ramps down. The gate must follow the latter."""
    rng = np.random.default_rng(3)
    truth = _rand(rng, 400)
    # 12 members cover only the middle; the flanks are the frame's own bases.
    members = _specs(truth, [truth[100:300]] * 12)
    cres = frame_consensus(truth, members, extend_ends=False)
    cert = certified_columns(cres, min_depth=3.0, min_agreement=0.6)
    assert cert.shape[0] == len(cres.consensus)
    assert cert[150:250].all(), "well-covered core must certify"
    assert not cert[:50].any(), "uncovered 5' flank must not certify"
    assert not cert[-50:].any(), "uncovered 3' flank must not certify"


def test_orf_is_truncated_at_the_last_certified_column():
    """An ORF running past the seed ORF into single-read flank is reported as
    ending at the last certified column, and flagged."""
    consensus = "ATG" + "GCT" * 100 + "TAA"
    certified = np.zeros(len(consensus), dtype=bool)
    certified[:180] = True  # support runs out mid-ORF
    got = gated_orf(consensus, certified, seed_orf_end=90, min_aa_length=30)
    assert got is not None
    prot, st, en, cert_end, truncated = got
    assert truncated is True
    assert st == 0
    assert en <= 180 and en % 3 == 0
    assert cert_end == 180
    assert len(prot) == en // 3
    assert "*" not in prot


def test_fully_certified_orf_is_not_truncated():
    consensus = "ATG" + "GCT" * 100 + "TAA"
    certified = np.ones(len(consensus), dtype=bool)
    prot, st, en, cert_end, truncated = gated_orf(
        consensus, certified, seed_orf_end=90, min_aa_length=30
    )
    assert truncated is False
    assert en == len(consensus)
    assert cert_end == en


def test_orf_within_the_seed_boundary_is_never_gated():
    """Everything at or before the previous round's ORF end is certified by
    construction — the gate only judges *new* ground."""
    consensus = "ATG" + "GCT" * 40 + "TAA"
    certified = np.zeros(len(consensus), dtype=bool)
    _prot, _st, en, _ce, truncated = gated_orf(
        consensus, certified, seed_orf_end=len(consensus), min_aa_length=30
    )
    assert truncated is False
    assert en == len(consensus)


def test_support_gate_end_to_end_on_a_one_read_flank():
    """The real shape: a template whose 3' flank comes from one read, and an
    ORF that would run into it. Adding reads that cover the flank should let
    the ORF extend."""
    rng = np.random.default_rng(11)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120))
    tail = "".join(rng.choice(_CODONS) for _ in range(40))  # no stop in here
    frame = _flank(rng, 30) + body + tail + "TAA" + _flank(rng, 30)
    short = frame[:33 + len(body)]  # members stop before the tail

    starved = refine_template(frame, _specs(frame, [short] * 25), min_aa_length=60)
    assert len(starved) == 1
    assert starved[0].orf_is_truncated_by_support is True

    covered = refine_template(
        frame, _specs(frame, [short] * 25 + [frame] * 25), min_aa_length=60
    )
    assert covered[0].orf_is_truncated_by_support is False
    assert len(covered[0].protein) > len(starved[0].protein)


# ── haplotypes ────────────────────────────────────────────────────────


def test_readthrough_haplotype_becomes_its_own_node():
    """The H3f3b shape, synthetically: a homopolymer deletion three codons
    before the stop bypasses it and reads through into the 3' UTR, giving a
    longer protein. Both nodes must survive with their own read counts."""
    rng = np.random.default_rng(17)
    head = "ATG" + "".join(rng.choice(_CODONS) for _ in range(99))  # 100 codons
    run, stop = "GGGGGG", "TAA"  # in-frame homopolymer, then the real stop
    # GCT has no stop in any of its three reading frames, so a -1 frameshift
    # reads straight through it to an explicit terminator — the 3' UTR
    # readthrough the deletion causes.
    downstream = "GCT" * 60 + "TAATTAATTAA"
    frame = _flank(rng, 40) + head + run + stop + downstream + _flank(rng, 40)
    deleted = frame.replace(run + stop, "GGGGG" + stop, 1)

    # The fixture must actually encode a readthrough, or the test proves
    # nothing about haplotype nodes getting their own ORFs.
    assert len(best_sense_orf(frame, min_aa_length=60)[0]) == 102
    assert len(best_sense_orf(deleted, min_aa_length=60)[0]) == 164

    nodes = refine_template(
        frame,
        _specs(frame, [frame] * 100 + [deleted] * 25),
        min_aa_length=60,
        min_haplotype_reads=3,
    )
    assert len(nodes) >= 2, "the deletion haplotype was absorbed"
    counts = sorted(n.n_reads for n in nodes)
    assert counts[-1] >= 90 and counts[-2] >= 20
    prots = {len(n.protein) for n in nodes if n.protein}
    assert len(prots) >= 2, "the two haplotypes must give different proteins"
    # The readthrough node is the longer protein on the smaller read count —
    # exactly the form a length-only heuristic would have thrown away.
    by_reads = sorted(nodes, key=lambda n: -n.n_reads)
    assert len(by_reads[0].protein) < len(by_reads[1].protein)


def test_minor_haplotype_below_the_read_floor_is_absorbed():
    rng = np.random.default_rng(19)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    alt = frame[:80] + ("A" if frame[80] != "A" else "G") + frame[81:]

    many = refine_template(
        frame, _specs(frame, [frame] * 60 + [alt] * 20), min_aa_length=60,
        min_haplotype_reads=3,
    )
    few = refine_template(
        frame, _specs(frame, [frame] * 60 + [alt] * 2), min_aa_length=60,
        min_haplotype_reads=3,
    )
    assert len(many) >= 2
    assert len(few) == 1, "a 2-read minor haplotype must fold into the major"
    assert few[0].n_reads == 62, "absorbed reads keep their weight"


def test_fraction_floor_defaults_off_so_a_deep_template_keeps_its_minority():
    """1% of a 20,000-read template is 200 — a floor that would discard
    exactly the ~5-read minority proteoform the design exists to preserve."""
    rng = np.random.default_rng(23)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    alt = frame[:80] + ("A" if frame[80] != "A" else "G") + frame[81:]
    # 20 minority reads against 500 — comfortably FDR-real, so this isolates
    # the fraction floor rather than conflating it with the variant caller.
    members = _specs(frame, [frame] * 500 + [alt] * 20)

    kept = refine_template(frame, members, min_aa_length=60, min_haplotype_reads=3)
    dropped = refine_template(
        frame, members, min_aa_length=60, min_haplotype_reads=3,
        min_haplotype_frac=0.10,  # 10% of 520 = 52 > 20
    )
    assert len(kept) >= 2, "default must keep the minority"
    assert len(dropped) == 1, "an explicit fraction floor discards it"
    assert dropped[0].n_reads == 520, "absorbed reads keep their weight"


def test_haplotype_read_counts_sum_to_the_template_total():
    rng = np.random.default_rng(29)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    alt = frame[:80] + ("A" if frame[80] != "A" else "G") + frame[81:]
    members = _specs(frame, [frame] * 40 + [alt] * 15)
    nodes = refine_template(frame, members, min_aa_length=60)
    assert sum(n.n_reads for n in nodes) == 55


# ── minimap2 CIGARs feed the PWM directly ─────────────────────────────


def test_specs_from_minimap2_alignments_project_correctly():
    """Round 1 needs no alignment work: minimap2's cg:Z is already a
    read→template alignment in the convention project_member wants."""
    rng = np.random.default_rng(31)
    truth = _rand(rng, 600)
    frame = truth[:300] + truth[301:]  # template missing base 300
    # A read that is the full truth, soft-clipped 20 at each end by minimap2.
    read = truth
    rows = [
        {
            "read_id": f"r{i}",
            # query=read, target=frame: the read has one base the frame lacks
            "cigar": "300=1I299=",
            "t_start": 0,
            "q_start": 0,
            "weight": 1.0,
        }
        for i in range(21)
    ]
    specs = specs_from_assignments(rows, {f"r{i}": read for i in range(21)})
    assert len(specs) == 21
    assert all(s.centroid_is_query is False for s in specs)
    res = frame_consensus(frame, specs)
    assert res.consensus == truth, "the majority insertion was not folded"


def test_specs_skips_rows_without_a_cigar():
    rows = [
        {"read_id": "a", "cigar": None, "t_start": 0, "q_start": 0},
        {"read_id": "b", "cigar": "100=", "t_start": 0, "q_start": 0},
    ]
    specs = specs_from_assignments(rows, {"a": "A" * 100, "b": "C" * 100})
    assert [s.member_seq[0] for s in specs] == ["C"]


def test_refined_orf_is_recallable_from_its_own_consensus():
    """Translation-late: the ORF the node reports must be the ORF you get by
    predicting on its consensus, not something read off haplotype columns."""
    rng = np.random.default_rng(37)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    nodes = refine_template(frame, _specs(frame, [frame] * 30), min_aa_length=60)
    node = nodes[0]
    again = best_sense_orf(node.consensus, min_aa_length=60)
    assert again is not None
    assert again[0] == node.protein
