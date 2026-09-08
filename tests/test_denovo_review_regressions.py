"""Regressions for defects found in external review of the ORF-EM stages.

Each test fails on the pre-fix code. They are collected here rather than
scattered across the per-stage files because they share one theme: a value
computed in one coordinate space, or over one population, being consumed as
if it belonged to another.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

edlib = pytest.importorskip("edlib")

from constellation.sequencing.readers.paf import (  # noqa: E402
    iter_paf_batches,
    paf_to_alignment_table,
    read_paf,
)
from constellation.sequencing.transcriptome.cluster.denovo.consensus import (  # noqa: E402
    COL_INSERTED,
    MemberSpec,
    frame_consensus,
    member_alleles,
)
from constellation.sequencing.transcriptome.cluster.denovo.variants import (  # noqa: E402
    ErrorModel,
    call_variants,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.estep import (  # noqa: E402
    TemplateSet,
    assign_reads,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.mstep import (  # noqa: E402
    gated_orf,
    refine_template,
)

_CODONS = ["GCT", "TGT", "GAT", "GAA", "TTT", "GGT", "CAT", "ATC", "AAA", "CTG"]


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _flank(rng, n):
    return _rand(rng, n).replace("ATG", "ATC")


def _spec(frame, member, i, weight=1.0):
    short, long = (frame, member) if len(frame) <= len(member) else (member, frame)
    a = edlib.align(short, long, mode="HW", task="path")
    return MemberSpec(
        member_seq=member,
        weight=weight,
        cigar=a["cigar"],
        centroid_is_query=(frame == short),
        ref_start=a["locations"][0][0],
        member_id=i,
    )


def _split_fixture(rng, at=200, n_maj=75, n_min=25):
    """A template plus a 75/25 minority, expressible either direction."""
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120)) + "TAA"
    full = _flank(rng, 40) + body + _flank(rng, 40)
    short = full[:at] + full[at + 1 :]
    return full, short, body


# ── #1 the E-step must consume its own alignment stream ───────────────


def test_paf_reader_accepts_an_iterable_of_byte_chunks():
    """minimap2_stream yields byte chunks off a live pipe — it has no
    `.read`, so a reader that only handles file objects cannot consume the
    stream the E-step actually produces."""
    raw = (
        b"r1\t900\t0\t900\t+\tt0\t1600\t0\t900\t880\t900\t60\tAS:i:1700\tcg:Z:900=\n"
        b"r2\t900\t0\t900\t+\tt1\t1600\t0\t900\t870\t900\t0\tAS:i:1690\tcg:Z:900=\n"
    )

    def chunks(size):
        for i in range(0, len(raw), size):
            yield raw[i : i + size]

    for size in (1, 7, 4096):  # splits land mid-field and mid-record
        batches = list(iter_paf_batches(chunks(size)))
        names = [n for b in batches for n in b.column("q_name").to_pylist()]
        assert names == ["r1", "r2"], f"chunk size {size}"


# ── #2 minority insertions must survive a null that explains the indel ─


def test_minority_insertion_resolves_under_an_explaining_error_model():
    """The haplotype gate's indel clause has to key off EITHER allele being a
    gap. On a minority insertion the MAJOR allele is the gap, so checking the
    minor alone reinstates the insertion/deletion asymmetry."""
    rng = np.random.default_rng(5)
    full, short, _ = _split_fixture(rng)
    # An error model that explains a 25% indel, so `call == 'real'` cannot
    # carry the column on its own.
    lax = ErrorModel(
        eps_sub=0.30, eps_indel=0.30, eps_hp0=0.30, hp_min=0.30, hp_max=0.40
    )
    counts = {}
    for label, frame, majority, minority in (
        ("insertion", short, short, full),
        ("deletion", full, full, short),
    ):
        members = [_spec(frame, majority, i) for i in range(75)]
        members += [_spec(frame, minority, 75 + i) for i in range(25)]
        nodes = refine_template(
            frame, members, min_aa_length=60, min_haplotype_reads=3, error_model=lax
        )
        counts[label] = sorted(n.n_reads for n in nodes)
    assert counts["insertion"] == counts["deletion"] == [25, 75]


# ── #3 declared variants must address each child's own consensus ──────


def test_declared_variants_are_mapped_into_each_child_consensus():
    """A minority insertion exists in the node that carries it and nowhere
    else. Copying the parent's positions into every child makes the majority
    node declare a column it does not have, and the next E-step then reads a
    different base than the node claims."""
    rng = np.random.default_rng(5)
    full, short, body = _split_fixture(rng)
    members = [_spec(short, short, i) for i in range(75)]
    members += [_spec(short, full, 75 + i) for i in range(25)]
    nodes = refine_template(
        short, members, min_aa_length=60, min_haplotype_reads=3,
        seed_orf=(40, 40 + len(body)),
    )
    assert len(nodes) == 2
    by_reads = {n.n_reads: n for n in nodes}
    major, minor = by_reads[75], by_reads[25]

    # The majority lacks the base, so it declares no column for it.
    assert major.declared_variants.size == 0
    # The minority declares it, and reading its own consensus there returns
    # the allele it claims.
    assert minor.declared_variants.size == 1
    pos = int(minor.declared_variants[0])
    assert 0 <= pos < len(minor.consensus)
    assert minor.consensus[pos] == minor.allele_string


# ── #4 / #5 the support gate has two ends, in the child's coordinates ─


def test_upstream_start_through_unsupported_sequence_is_gated():
    """An upstream ATG can extend a protein through uncovered flank while
    sharing the seed's stop — the 3'-only gate never sees it."""
    # Two in-frame starts sharing one stop: an upstream one sitting in
    # uncovered flank, and a supported one 30 nt in.
    consensus = "ATG" + "GCT" * 9 + "ATG" + "GCT" * 90 + "TAA"
    certified = np.ones(len(consensus), dtype=bool)
    certified[:30] = False  # the first 10 codons are one read's sequence
    prot, st, en, _cert, truncated = gated_orf(
        consensus, certified, seed_orf_start=30, seed_orf_end=len(consensus),
        min_aa_length=30,
    )
    assert truncated is True, "an uncertified upstream start must be refused"
    assert st == 30, "the ORF must restart at the supported ATG"
    assert consensus[st : st + 3] == "ATG"
    assert en == len(consensus), "it still runs to the shared stop"


def test_each_child_reports_its_orf_in_its_own_coordinates():
    """Two haplotypes with different length shifts must each express their ORF
    in their own consensus.

    Unlike the others here this pins the post-fix invariant rather than
    reproducing the pre-fix failure: children are now built on the *pooled*
    column plan and the seed boundary is mapped per child, so the drift this
    guards against is no longer expressible through the public API (the old
    signature took a bare consensus position, which a caller could only get
    right for one child).
    """
    rng = np.random.default_rng(31)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120)) + "TAA"
    frame = _flank(rng, 60) + body + _flank(rng, 40)
    # The majority deletes 30 nt upstream of the ORF, so its consensus is 30
    # shorter than the minority's and every downstream coordinate shifts.
    trimmed = frame[:10] + frame[40:]
    members = [_spec(frame, trimmed, i) for i in range(70)]
    members += [_spec(frame, frame, 70 + i) for i in range(30)]
    nodes = refine_template(
        frame, members, min_aa_length=60, min_haplotype_reads=3,
        seed_orf=(60, 60 + len(body)),
    )
    assert len(nodes) >= 2, "the two length classes must separate"
    lengths = {len(n.consensus) for n in nodes}
    assert len(lengths) > 1, "the children must actually differ in length"
    for n in nodes:
        if n.protein is None:
            continue
        assert 0 <= n.orf_start < n.orf_end <= len(n.consensus)
        assert n.consensus[n.orf_start : n.orf_start + 3] == "ATG"
        for pos in n.declared_variants.tolist():
            assert 0 <= pos < len(n.consensus), "a declared column must exist here"


# ── #6 reads ending at a junction have no evidence about it ───────────


def test_reads_ending_at_a_junction_do_not_vote_there():
    """Ten reads that stop exactly at the junction must not outvote six that
    span it and carry the insertion."""
    rng = np.random.default_rng(3)
    truth = _rand(rng, 400)
    at = 200
    frame = truth[:at] + truth[at + 1 :]
    members = [_spec(frame, truth, i) for i in range(6)]
    members += [_spec(frame, frame[:at], 100 + i) for i in range(10)]
    res = frame_consensus(frame, members)

    ins = np.flatnonzero(res.column_kind == COL_INSERTED)
    assert ins.size == 1
    col = member_alleles(res, ins)[:, 0]
    assert int(((col >= 0) & (col < 4)).sum()) == 6, "spanning reads vote a base"
    assert int((col == 4).sum()) == 0, "reads that stop short must not vote gap"
    assert int((col == -1).sum()) == 10, "they are uncovered, not evidence"
    assert res.consensus == truth


# ── #8 terminal extension columns are length variation, not alleles ───


def test_terminal_extension_columns_are_not_in_core():
    """A dropped terminal column anchors its consensus_pos to the last kept
    base, so a purely positional core test calls it core and a 75/25 length
    mixture starts defining haplotypes."""
    rng = np.random.default_rng(37)
    truth = _rand(rng, 500)
    frame = truth[:-40]
    members = [_spec(frame, frame, i) for i in range(75)]
    members += [_spec(frame, truth, 75 + i) for i in range(25)]
    res = frame_consensus(frame, members)
    assert res.n_columns_planned > len(frame), "fixture must plan a 3' block"

    terminal_cols = set(
        np.flatnonzero(np.isin(res.column_kind, (2, 3))).tolist()
    )
    for v in call_variants(res):
        if v[12] in terminal_cols:
            assert v[11] is False, "a terminal extension column must not be in_core"


# ── #9 fractional mass is per template, not per placement ─────────────


def test_fractional_weight_is_split_across_templates_not_placements():
    """A read placing twice on one template must not thereby claim two thirds
    of its own mass for it."""
    lines = [
        # two placements on t0, one on t1, all equal score
        "r1\t900\t0\t400\t+\tt0\t1600\t0\t400\t395\t400\t60\tAS:i:1700\tcg:Z:400=",
        "r1\t900\t500\t900\t+\tt0\t1600\t900\t1300\t395\t400\t0\tAS:i:1700\tcg:Z:400=",
        "r1\t900\t0\t400\t+\tt1\t1600\t0\t400\t395\t400\t0\tAS:i:1700\tcg:Z:400=",
    ]
    raw = ("\n".join(lines) + "\n").encode()
    n = 2
    templates = TemplateSet(
        name=[f"t{i}" for i in range(n)],
        template_id=np.arange(n, dtype=np.int64),
        sequence=["ACGT" * 400] * n,
        orf_start=np.full(n, 100, dtype=np.int64),
        orf_end=np.full(n, 400, dtype=np.int64),
        orf_aa_length=np.full(n, 100, dtype=np.int64),
        node_weight=np.ones(n, dtype=np.float64),  # equal weight
        declared_variants=[np.empty(0, dtype=np.int64) for _ in range(n)],
    )
    aln, cov = assign_reads(
        list(iter_paf_batches(io.BytesIO(raw))), templates,
        tie_resolution="fractional", emit_coverage=True,
    )
    by_t = dict(
        zip(aln.column("template_id").to_pylist(), aln.column("weight").to_pylist())
    )
    assert set(by_t) == {0, 1}
    assert by_t[0] == pytest.approx(0.5) and by_t[1] == pytest.approx(0.5)
    # And the support table stays one row per (read, template).
    pairs = list(
        zip(cov.column("read_id").to_pylist(), cov.column("template_id").to_pylist())
    )
    assert len(pairs) == len(set(pairs))


# ── #10 a clip and an insertion at one junction are one event ─────────


def test_terminal_clip_and_insertion_at_one_junction_are_not_collapsed():
    """`q_start=2` with a leading `2I` describes four member bases before the
    template starts; keeping only one of the two events loses half of them."""
    rng = np.random.default_rng(41)
    truth = _rand(rng, 400)
    prefix = "GGGG"
    members = [
        MemberSpec(
            member_seq=prefix + truth,
            weight=1.0,
            cigar="2I400=",
            centroid_is_query=False,
            ref_start=0,
            member_start=2,
            member_id=i,
        )
        for i in range(20)
    ]
    res = frame_consensus(truth, members)
    assert res.n_extended_5p == 4, "all four unanimous 5' bases must be kept"
    assert res.consensus == prefix + truth


# ── #11 reverse-strand clipping is alignment-oriented ─────────────────


def test_reverse_strand_clipping_matches_the_cigar_orientation():
    """PAF reports q_start/q_end on the forward query, but a SAM CIGAR is
    written in alignment orientation — so on '-' the clips swap."""
    raw = (
        b"r1\t100\t10\t70\t-\tt0\t1600\t0\t60\t58\t60\t60\tAS:i:100\tcg:Z:60=\n"
        b"r2\t100\t10\t70\t+\tt0\t1600\t0\t60\t58\t60\t60\tAS:i:100\tcg:Z:60=\n"
    )
    aln = paf_to_alignment_table(read_paf(io.BytesIO(raw)))
    cigars = aln.column("cigar_string").to_pylist()
    assert cigars[0] == "30S60=10S", "reverse clips must be swapped"
    assert cigars[1] == "10S60=30S", "forward clips are unchanged"


# ── #12 ORF identity is measured over the ORF ─────────────────────────


def test_orf_identity_ignores_errors_outside_the_orf():
    """A read matching the CDS perfectly but disagreeing across both UTRs is
    perfect ORF evidence; whole-alignment identity calls it half-wrong."""
    n = 1
    templates = TemplateSet(
        name=["t0"],
        template_id=np.zeros(1, dtype=np.int64),
        sequence=["ACGT" * 250],
        orf_start=np.array([300], dtype=np.int64),
        orf_end=np.array([700], dtype=np.int64),
        orf_aa_length=np.array([133], dtype=np.int64),
        node_weight=np.ones(n, dtype=np.float64),
        declared_variants=[np.empty(0, dtype=np.int64)],
    )
    # 300 mismatches of UTR, 400 perfect ORF, 300 mismatches of UTR.
    cg = "300X400=300X"
    raw = (
        f"r1\t1000\t0\t1000\t+\tt0\t1000\t0\t1000\t400\t1000\t60\tAS:i:800\tcg:Z:{cg}\n"
    ).encode()
    _aln, cov = assign_reads(
        list(iter_paf_batches(io.BytesIO(raw))), templates, emit_coverage=True
    )
    assert cov.num_rows == 1
    assert cov.column("orf_identity").to_pylist()[0] == pytest.approx(1.0)
    assert cov.column("orf_cov_frac").to_pylist()[0] == pytest.approx(1.0)
