"""Stage 3 of the EM: consensus, haplotypes, and the ORF as an annotation.

The headline case is the one the design exists for: a homopolymer deletion
that produces a readthrough ORF must surface as its own node with its own read
count. The ORF itself is `best_sense_orf` on each node's consensus, under the
run's floor, and nothing more — the support gate that used to stand between
them is gone (ledger #52).
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
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (  # noqa: E402
    pooled_node,
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


# ── the ORF is an annotation of the consensus ─────────────────────────


def _one_read_flank(rng):
    """A template whose 3' flank is covered by exactly ONE read, with an ORF
    that runs into it: the shape the support gate was built for."""
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120))
    tail = "".join(rng.choice(_CODONS) for _ in range(40))  # no stop in here
    frame = _flank(rng, 30) + body + tail + "TAA" + _flank(rng, 30)
    short = frame[: 33 + len(body)]  # members stop before the tail
    return frame, short


def test_the_orf_is_the_consensus_s_longest_sense_orf_whatever_the_depth():
    """No support gate. Measured on 9.4M reads the gate cost 9.7% of
    full-length RefSeq proteins: a per-node certification evaluated on a
    3-read node was noise, and under kmer seeding there was never a seed
    interval to anchor it. The protein is what the consensus encodes."""
    rng = np.random.default_rng(11)
    frame, short = _one_read_flank(rng)

    starved = refine_template(frame, _specs(frame, [short] * 25 + [frame]))
    assert len(starved) == 1, "one read is not a node"
    assert starved[0].n_reads == 26, "…and it is not discarded either"
    node = starved[0]
    expect = best_sense_orf(node.consensus, min_aa_length=30)
    assert expect is not None
    assert (node.protein, node.orf_start, node.orf_end) == expect
    assert not hasattr(node, "orf_is_truncated_by_support")
    assert not hasattr(node, "orf_certified_end")

    covered = refine_template(frame, _specs(frame, [short] * 25 + [frame] * 25))
    assert len(covered) == 2, "25 reads of extra 3' extent is a clique"
    # The short form holds no stop codon, so it has no ORF to report; the
    # long one reports the full protein, uncertified flank or not.
    orfs = {
        (n.protein, n.orf_start, n.orf_end) == (
            best_sense_orf(n.consensus, min_aa_length=30) or (None, -1, -1)
        )
        for n in covered
    }
    assert orfs == {True}
    assert sorted(len(n.protein or "") for n in covered) == [0, 161]


def test_the_floor_is_applied_and_is_the_only_thing_that_is():
    """Prm1 is 51 aa, so the floor is 30, not 60. And it IS applied: a floor
    of 1 shipped 14,813 sub-30-aa proteins in one round."""
    rng = np.random.default_rng(12)
    tiny = "ATG" + "GCT" * 9 + "TAA"  # 10 aa
    frame = _flank(rng, 40) + tiny + _flank(rng, 40)
    members = _specs(frame, [frame] * 8)
    assert refine_template(frame, members)[0].protein is None
    assert refine_template(frame, members, min_aa_length=10)[0].protein == "MAAAAAAAAA"
    assert refine_template(frame, members, min_aa_length=11)[0].protein is None

    prm1 = "ATG" + "".join(rng.choice(_CODONS) for _ in range(50)) + "TAA"  # 51 aa
    frame = _flank(rng, 40) + prm1 + _flank(rng, 40)
    members = _specs(frame, [frame] * 8)
    assert len(refine_template(frame, members)[0].protein) == 51
    assert refine_template(frame, members, min_aa_length=60)[0].protein is None


def test_nothing_in_the_mstep_reads_a_seed_orf_or_a_support_threshold():
    import inspect

    from constellation.sequencing.transcriptome.cluster.denovo.em import mstep
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    params = inspect.signature(refine_template).parameters
    assert "min_aa_length" in params and "coverage_route" in params
    for gone in ("seed_orf", "support_min_depth", "support_min_agreement"):
        assert gone not in params
    for gone in ("gated_orf", "certified_columns", "_NO_LENGTH_FLOOR"):
        assert not hasattr(mstep, gone)
    assert MStepParams().min_aa_length == 30
    assert "min_aa_length" in MStepParams().kernel_kwargs()
    assert not hasattr(MStepParams(), "support_min_depth")


# ── the pooled node: the consensus step without the split ─────────────


def test_a_pooled_node_is_the_consensus_of_every_member_with_its_orf():
    """What the final merge rebuilds a survivor from. The members of a
    merged group, aligned to the survivor, give one consensus — trimmed to
    what they cover, with the deepest form's bases — and its ORF."""
    rng = np.random.default_rng(14)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120)) + "TAA"
    frame = _flank(rng, 60) + body + _flank(rng, 60)
    # 30 members carry a substitution the frame lacks; 5 agree with it.
    at = 60 + 3 * 50 + 1
    other = "C" if frame[at] != "C" else "G"
    variant = frame[:at] + other + frame[at + 1 :]
    members = _specs(frame, [variant] * 30 + [frame] * 5)
    node = pooled_node(frame, members, template_id=7, haplotype_id=2)
    assert node is not None
    assert node.consensus == variant
    assert (node.parent_template_id, node.haplotype_id) == (7, 2)
    assert node.n_reads == 35 and node.node_weight == 35.0
    assert node.member_ids.tolist() == list(range(35))
    assert (node.protein, node.orf_start, node.orf_end) == best_sense_orf(
        variant, min_aa_length=30
    )
    assert node.protein != best_sense_orf(frame, min_aa_length=30)[0]

    # Trimmed to the covered span, like any node.
    short = frame[40:-40]
    trimmed = pooled_node(frame, _specs(frame, [short] * 10))
    assert trimmed is not None
    assert abs(len(trimmed.consensus) - len(short)) <= 5
    assert trimmed.n_trimmed_5p + trimmed.n_trimmed_3p >= 70

    assert pooled_node(frame, []) is None
    assert pooled_node(frame, members, min_aa_length=200).protein is None


# ── the coverage route is a parameter ─────────────────────────────────


def test_the_coverage_route_can_be_switched_off():
    """Every bench result since 2026-09-23 was measured with it off. Off,
    the alternative start below is one node; the allelic split still fires."""
    rng = np.random.default_rng(31)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(150)) + "TAA"
    frame = _flank(rng, 200) + body + _flank(rng, 40)
    short = frame[200:]
    members = _specs(frame, [frame] * 40 + [short] * 40)
    by_extent = refine_template(frame, members, coverage_route=True)
    assert len(by_extent) == 2
    assert by_extent[0].stats["n_coverage"] > 0

    alone = refine_template(frame, members, coverage_route=False)
    assert len(alone) == 1
    assert alone[0].n_reads == 80
    assert alone[0].stats["n_coverage"] == 0

    # Two linked substitutions: a column earns a node only by co-varying
    # with another, so one alone never splits.
    variant = list(frame)
    for at in (200 + 3 * 40 + 1, 200 + 3 * 90 + 1):
        variant[at] = "C" if frame[at] != "C" else "G"
    variant = "".join(variant)
    allelic = refine_template(
        frame, _specs(frame, [frame] * 40 + [variant] * 40), coverage_route=False
    )
    assert len(allelic) == 2
    assert sorted(n.n_reads for n in allelic) == [40, 40]


def test_a_node_is_trimmed_to_the_sequence_its_own_reads_support():
    """Zero coverage is not thin coverage. A child built on the pooled column
    plan has a column for every position of the parent, and the kernel falls
    back to the template base where nobody voted — so without trimming, a node
    emits parent sequence no read of it ever supported. That is the mechanism
    behind references inflating 1,558 → 2,346 nt across six rounds."""
    rng = np.random.default_rng(13)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(120)) + "TAA"
    tail = _flank(rng, 150)
    frame = _flank(rng, 30) + body + tail
    short = frame[: 33 + len(body)]
    nodes = refine_template(frame, _specs(frame, [short] * 25))
    assert len(nodes) == 1
    # Within a few nt: edlib's HW placement can carry the alignment a base or
    # two past the planted cut, which is coverage, not fabrication.
    assert abs(nodes[0].n_trimmed_3p - len(tail)) <= 5
    assert abs(len(nodes[0].consensus) - len(short)) <= 5
    assert not nodes[0].consensus.endswith(tail[-20:])


# ── haplotypes ────────────────────────────────────────────────────────


def test_a_lone_readthrough_deletion_does_not_become_its_own_node():
    """**Inverted deliberately.** Under pure covariance a column earns a node
    only by co-varying with another, and a lone systematic indel co-varies
    with nothing. So the H3f3b-style readthrough — a 1-nt deletion in a 5-G
    run three codons before the stop, bypassing it and reading ~60 residues
    out of the 3' UTR — is **not** created.

    That reverses ledger #22 ("kept with its 97 reads and flagged, not deleted
    or believed") and it is the stated cost of the decision, not an accident:
    the same rule is what removes the 53,239 single-indel nodes encoding an
    identical protein and the 22%-homopolymer artifacts that a fitted null
    expects anyway. The sibling test below shows the node IS created as soon
    as the deletion has a covarying partner, so the rule is pinned in both
    directions rather than only where it happens to be convenient.
    """
    rng = np.random.default_rng(17)
    head = "ATG" + "".join(rng.choice(_CODONS) for _ in range(99))  # 100 codons
    run, stop = "GGGGGG", "TAA"  # in-frame homopolymer, then the real stop
    # GCT has no stop in any of its three reading frames, so a -1 frameshift
    # reads straight through it to an explicit terminator.
    downstream = "GCT" * 60 + "TAATTAATTAA"
    frame = _flank(rng, 40) + head + run + stop + downstream + _flank(rng, 40)
    deleted = frame.replace(run + stop, "GGGGG" + stop, 1)

    # The fixture really does encode a readthrough — otherwise the test proves
    # nothing about what was given up.
    assert len(best_sense_orf(frame)[0]) == 102
    assert len(best_sense_orf(deleted)[0]) == 164

    nodes = refine_template(
        frame, _specs(frame, [frame] * 100 + [deleted] * 25)
    )
    assert len(nodes) == 1, "a lone indel has no covarying partner"
    assert nodes[0].n_reads == 125, "…and its reads are not lost, only unsplit"
    assert len(nodes[0].protein) == 102, "the majority protein"


def test_the_same_deletion_with_a_covarying_partner_does_become_a_node():
    """The sibling. Identical readthrough deletion, plus one substitution the
    same 25 reads carry. One edge is the whole difference."""
    rng = np.random.default_rng(17)
    head = "ATG" + "".join(rng.choice(_CODONS) for _ in range(99))
    run, stop = "GGGGGG", "TAA"
    downstream = "GCT" * 60 + "TAATTAATTAA"
    frame = _flank(rng, 40) + head + run + stop + downstream + _flank(rng, 40)
    deleted = frame.replace(run + stop, "GGGGG" + stop, 1)
    # A linked substitution 60 nt upstream, inside the CDS and synonymous to
    # the argument: what matters is that the same reads carry both.
    at = 100
    linked = deleted[:at] + ("A" if deleted[at] != "A" else "C") + deleted[at + 1 :]

    nodes = refine_template(
        frame, _specs(frame, [frame] * 100 + [linked] * 25)
    )
    assert len(nodes) == 2
    assert sorted(n.n_reads for n in nodes) == [25, 100]
    # The readthrough node is the longer protein on the smaller read count —
    # exactly the form a length-only heuristic would have thrown away.
    by_reads = sorted(nodes, key=lambda n: -n.n_reads)
    assert len(by_reads[0].protein) < len(by_reads[1].protein)


def test_a_minority_with_no_edge_is_not_split_off():
    """Replaces the read-floor test. The mechanism is now "no edge, not
    retained" — there is no ``min_haplotype_reads`` to fall below."""
    rng = np.random.default_rng(19)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)

    def _alt(n_pos):
        out = list(frame)
        for p in (80, 140)[:n_pos]:
            out[p] = "A" if frame[p] != "A" else "G"
        return "".join(out)

    linked = refine_template(
        frame, _specs(frame, [frame] * 60 + [_alt(2)] * 20)
    )
    lone = refine_template(
        frame, _specs(frame, [frame] * 60 + [_alt(1)] * 20)
    )
    tiny = refine_template(
        frame, _specs(frame, [frame] * 60 + [_alt(2)] * 2)
    )
    assert len(linked) == 2, "two linked columns split"
    assert len(lone) == 1, "one column alone does not"
    assert len(tiny) == 1, "and two reads give an expected cell below 5"
    assert tiny[0].n_reads == 62, "unsplit reads keep their weight"


def test_a_minority_is_kept_by_evidence_not_by_fraction_of_the_template():
    """Replaces ``min_haplotype_frac``, whose 1% default on a 20,000-read
    template was 200 reads. There is no fraction floor now: 20 linked reads
    against 500 survive, and ``f_min`` is the only fraction in the path."""
    rng = np.random.default_rng(23)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    alt = list(frame)
    for p in (80, 140):
        alt[p] = "A" if frame[p] != "A" else "G"
    members = _specs(frame, [frame] * 500 + ["".join(alt)] * 20)

    kept = refine_template(frame, members)
    assert len(kept) == 2, "20 of 520 is 3.8% — above f_min, and linked"
    assert sorted(n.n_reads for n in kept) == [20, 500]
    # Raising f_min above the effect size is what discards it, explicitly.
    coarse = refine_template(frame, members, f_min=0.10)
    assert len(coarse) == 1
    assert coarse[0].n_reads == 520, "unsplit reads keep their weight"


def test_an_alternative_start_gives_two_nodes_of_different_length():
    """Consensus length is an OUTPUT of the signature, not a constraint — which
    is how extent differences produce different-length references and why no
    separate split operation is needed. A block of coverage columns with
    identical uncovered read-sets is a clique; a ramp is not."""
    rng = np.random.default_rng(31)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(150)) + "TAA"
    frame = _flank(rng, 200) + body + _flank(rng, 40)
    short = frame[200:]  # an alternative start exactly 200 nt in
    nodes = refine_template(frame, _specs(frame, [frame] * 40 + [short] * 40))
    assert len(nodes) == 2
    assert sorted(n.n_reads for n in nodes) == [40, 40]
    lengths = sorted(len(n.consensus) for n in nodes)
    assert lengths[1] - lengths[0] > 150, "the two references differ in length"


def test_a_degradation_ramp_gives_one_node():
    """The same total extent variation, spread continuously. Every read ends
    somewhere different, so no boundary is a mode and no node is created."""
    rng = np.random.default_rng(37)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(150)) + "TAA"
    frame = _flank(rng, 200) + body + _flank(rng, 40)
    members = _specs(frame, [frame[5 * i :] for i in range(40)])
    nodes = refine_template(frame, members)
    assert len(nodes) == 1
    assert nodes[0].n_reads == 40


def test_the_major_node_carries_the_refit_substrate_and_the_counters():
    """Unit 8: ``disagreement_stats`` is components-only today, which is why
    the indel bypass existed. Exposing it here lets the round loop merge round
    r's disagreements and pass the fitted model in as round r+1's prior —
    which sidesteps the components path's hardest part entirely. There, a
    refit is retro-applied via ``reclassify_variants`` and then needs
    ``_reconcile_haplotypes_after_reclassification`` to undo columns the refit
    demoted, and that reconciliation explicitly cannot handle *promotions*.
    Consuming the fit in the next round means no second consensus pass, no
    reclassify, no reconciliation, and no promotion problem.
    """
    rng = np.random.default_rng(41)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    nodes = refine_template(frame, _specs(frame, [frame] * 30))
    st = nodes[0].stats
    assert "disagreements" in st
    assert st["n_reads"] == 30
    # "The method returned one template in the tail" has to be visible rather
    # than inferred from a quiet result.
    for key in ("n_candidates", "n_allelic", "n_coverage", "n_signatures", "n_nodes"):
        assert key in st, key
    assert all(n.stats == {} for n in nodes[1:]), "diagnostics on the major only"


def test_haplotype_read_counts_sum_to_the_template_total():
    """A hard invariant, and the reason nodes are emitted by state-tuple: every
    read lands in exactly one, so quant adds up. Two independent signatures
    would otherwise give two marginals that each count every read once."""
    rng = np.random.default_rng(29)
    body = "ATG" + "".join(rng.choice(_CODONS) for _ in range(100)) + "TAA"
    frame = _flank(rng, 40) + body + _flank(rng, 40)
    alt = list(frame)
    for p in (80, 140):
        alt[p] = "A" if frame[p] != "A" else "G"
    short = frame[60:]
    members = _specs(frame, [frame] * 40 + ["".join(alt)] * 15 + [short] * 20)
    nodes = refine_template(frame, members)
    assert len(nodes) >= 2, "the fixture must actually split"
    assert sum(n.n_reads for n in nodes) == 75


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
    nodes = refine_template(frame, _specs(frame, [frame] * 30))
    node = nodes[0]
    again = best_sense_orf(node.consensus)
    assert again is not None
    assert again[0] == node.protein


def test_removing_the_floor_is_monotone():
    """It can only ADD short ORFs, never change a long one.

    `best_sense_orf` keeps the longest ATG→stop per (frame, stop) and takes
    the global max, and the longest for a given stop is its earliest in-frame
    ATG — which does not depend on the floor. That is why every M-step test
    above passes unchanged.
    """
    from constellation.sequencing.transcriptome.cluster.denovo.orf import (
        best_sense_orf,
    )

    rng = np.random.default_rng(31)
    codons = ["GCT", "TGC", "GAT", "GAA", "TTT", "GGT", "CAT", "ATT", "AAA", "CTT"]
    long_orf = "ATG" + "".join(rng.choice(codons) for _ in range(99)) + "TAA"
    seq = "GGCACT" + long_orf + "CCTAGG"
    assert best_sense_orf(seq, min_aa_length=1) == best_sense_orf(
        seq, min_aa_length=60
    )
    # ...and where the floor found nothing, no floor finds the short one.
    short = "GGCACT" + "ATG" + "GCT" * 4 + "TAA" + "CCTAGG"
    assert best_sense_orf(short, min_aa_length=60) is None
    assert best_sense_orf(short, min_aa_length=1) is not None
