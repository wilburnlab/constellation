"""Stage 1 of the ORF-anchored EM: seeding + pair-classified folding.

The load-bearing tests are the two *negatives* — a suffix ORF from a
downstream ATG must never fold into its parent, and two proteoforms differing
by an alternative first exon must survive as separate templates. Both are
cases where a similarity gate alone silently erases real biology.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("edlib")

from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (  # noqa: E402
    connected_components,
    greedy_set_cover,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import (  # noqa: E402
    best_sense_orf,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.fold import (  # noqa: E402
    FoldRule,
    classify_pair,
    fold_orfs,
)
from constellation.sequencing.transcriptome.cluster.denovo.orfem.seed import (  # noqa: E402
    REPRESENTATIVE_POLICIES,
    extract_seed_orfs,
)


_CODONS = ["GCT", "TGT", "GAT", "GAA", "TTT", "GGT", "CAT", "ATC", "AAA", "CTG"]


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _orf(rng, n_aa):
    """An ORF with no internal stop: ATG + n_aa-1 sense codons + TAA."""
    body = "".join(rng.choice(_CODONS) for _ in range(n_aa - 1))
    return "ATG" + body + "TAA"


_STOPS = frozenset({"TAA", "TAG", "TGA"})


def _flank(rng, n):
    """Random flank with no ATG, so it can never extend the planted ORF.

    A random 5' flank containing an in-frame ATG with no intervening stop
    makes ``best_sense_orf`` report a *longer* ORF, which then falls outside
    the ±3-codon fold bound for a reason the test is not about.
    """
    s = _rand(rng, n)
    return s.replace("ATG", "ATC")


def _point_mutate(orf, positions):
    """Exactly ``len(positions)`` single-base substitutions, each chosen so
    its codon stays a sense codon (a new in-frame stop would truncate the
    predicted ORF and change its length)."""
    out = list(orf)
    for p in positions:
        c0 = 3 * (p // 3)
        for alt in "ACGT":
            if alt == out[p]:
                continue
            trial = out[:]
            trial[p] = alt
            if "".join(trial[c0 : c0 + 3]) not in _STOPS:
                out = trial
                break
    return "".join(out)


def _reads(rows):
    return pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": pa.array([r[1] for r in rows], pa.large_string()),
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


# ── the pair classifier ───────────────────────────────────────────────


def test_rule_1_folds_in_frame_length_variants():
    # ±3 codons with a frame-preserving overhang → same proteoform.
    assert (
        classify_pair(
            len_short=300,
            len_long=306,
            n_mismatch=2,
            n_insert=0,
            n_delete=0,
            cigar="300=",
        )
        is FoldRule.ERROR_VARIANT
    )


def test_rule_1_refuses_beyond_the_length_bound():
    assert (
        classify_pair(
            len_short=300,
            len_long=330,
            n_mismatch=0,
            n_insert=0,
            n_delete=0,
            cigar="300=",
        )
        is FoldRule.SEPARATE
    )


def test_contained_without_indel_is_never_folded():
    """A suffix ORF from a downstream ATG. Contained, zero indels, far
    shorter — at the ORF level a 5' truncation and an alternative start are
    indistinguishable, so this has to stay separate for the E-step to decide."""
    assert (
        classify_pair(
            len_short=180,
            len_long=420,
            n_mismatch=0,
            n_insert=0,
            n_delete=0,
            cigar="180=",
        )
        is FoldRule.SEPARATE
    )


def test_rule_2_requires_n_insert_zero_and_one_indel_run():
    kw = dict(len_short=300, len_long=409, n_mismatch=1, allow_frameshift=True)
    # One deletion run, net frameshift, shorter adds nothing → frameshift pair.
    assert (
        classify_pair(n_insert=0, n_delete=1, cigar="150=1D149=", **kw)
        is FoldRule.FRAMESHIFT
    )
    # Two indel runs → not a single event.
    assert (
        classify_pair(n_insert=0, n_delete=2, cigar="100=1D50=1D148=", **kw)
        is FoldRule.SEPARATE
    )
    # The shorter contributes bases the longer lacks → not containment.
    assert (
        classify_pair(n_insert=1, n_delete=2, cigar="100=1I50=2D147=", **kw)
        is FoldRule.SEPARATE
    )


def test_rule_2_is_off_unless_asked_for():
    kw = dict(
        len_short=300, len_long=409, n_mismatch=1, n_insert=0, n_delete=1,
        cigar="150=1D149=",
    )
    assert classify_pair(**kw) is FoldRule.SEPARATE
    assert classify_pair(allow_frameshift=True, **kw) is FoldRule.FRAMESHIFT


# ── greedy grouping ───────────────────────────────────────────────────


def test_greedy_radius_1_does_not_chain_where_components_do():
    """A–B and B–C verified, A–C not. Components merge all three; radius-1
    greedy claims only the seed's direct neighbours."""
    abundance = np.array([10, 5, 1], dtype=np.int64)
    seq_len = np.array([300, 300, 300], dtype=np.int64)
    ea, eb = np.array([0, 1]), np.array([1, 2])

    greedy = greedy_set_cover(3, abundance, seq_len, ea, eb)
    comp = connected_components(3, abundance, seq_len, ea, eb)

    assert len(set(comp.cluster_of.tolist())) == 1  # one chained component
    assert len(set(greedy.cluster_of.tolist())) == 2
    # The most abundant unique seeds the first group and claims B.
    assert greedy.cluster_of[0] == greedy.cluster_of[1] != greedy.cluster_of[2]
    assert greedy.centroid_uniq[0] == 0


def test_greedy_seeds_by_abundance_then_length():
    abundance = np.array([1, 9, 9], dtype=np.int64)
    seq_len = np.array([900, 300, 600], dtype=np.int64)
    res = greedy_set_cover(3, abundance, seq_len, np.array([]), np.array([]))
    # No edges ⇒ three singleton groups, seeded in priority order.
    assert res.centroid_uniq.tolist() == [2, 1, 0]


# ── end to end over reads ─────────────────────────────────────────────


def test_error_variants_of_one_orf_fold_together():
    rng = np.random.default_rng(11)
    orf = _orf(rng, 120)
    rows = [("r0", _flank(rng, 40) + orf + _flank(rng, 60), 0)]
    # Nine reads carrying the same-length protein, one substitution each.
    for i in range(1, 10):
        mutated = _point_mutate(orf, [30 + 7 * i])
        rows.append((f"r{i}", _flank(rng, 40) + mutated + _flank(rng, 60), 0))
    seed, _ = extract_seed_orfs(_reads(rows), min_aa_length=60)
    assert set(seed.column("orf_aa_length").to_pylist()) == {120}
    assert seed.num_rows > 1, "fixture must produce distinct ORFs to fold"
    res = fold_orfs(seed, identity=0.9)
    assert res.group_rep_orf.shape[0] == 1
    assert int(res.group_n_reads[0]) == 10


def test_alternative_first_exon_survives_as_two_templates():
    """The Akap4 property. Two proteoforms share a long body but differ by an
    unrelated 5' exon and a short N-terminal extension. Their *cDNAs* are
    >0.97 identical, so a template-level fold would merge them — which is
    exactly why there is no template-level fold."""
    import edlib

    rng = np.random.default_rng(13)
    # Body sized so the 130-nt differing exon costs the same ~2-3% of the cDNA
    # it costs on the real gene (a 2.9 kb window); on a short cDNA it costs
    # ~5% and the fixture stops being the case under test.
    body = _orf(rng, 850)[:-3]  # shared body, stop added below
    n_ext = "ATG" + "".join(rng.choice(_CODONS) for _ in range(8))  # 9 extra aa
    utr3 = _rand(rng, 120)

    long_cdna = _rand(rng, 130) + n_ext + body + "TAA" + utr3  # 859-aa form
    short_cdna = _rand(rng, 130) + body + "TAA" + utr3  # 850-aa form
    rows = [(f"L{i}", long_cdna, 0) for i in range(4)]
    rows += [(f"S{i}", short_cdna, 0) for i in range(40)]

    ident = 1.0 - edlib.align(short_cdna, long_cdna, mode="HW")["editDistance"] / len(
        short_cdna
    )
    assert ident > 0.97, "fixture must be inside a naive template-level gate"

    seed, _ = extract_seed_orfs(_reads(rows), min_aa_length=60)
    res = fold_orfs(seed, identity=0.97, max_len_delta=9)
    aa = seed.column("orf_aa_length").to_pylist()
    assert sorted(aa) == [850, 859]
    # 27 nt apart — outside the ±9 nt bound, so they stay separate seeds.
    assert res.group_rep_orf.shape[0] == 2, "the minority proteoform was erased"


def test_seed_representative_policies_pick_different_reads():
    rng = np.random.default_rng(17)
    orf = _orf(rng, 100)
    rows = [
        (f"r{i}", _rand(rng, 20 + 15 * i) + orf + _rand(rng, 40), 0) for i in range(7)
    ]
    picks = {}
    for name in REPRESENTATIVE_POLICIES:
        seed, _ = extract_seed_orfs(_reads(rows), min_aa_length=60, representative=name)
        deepest = max(seed.to_pylist(), key=lambda r: r["n_reads"])
        picks[name] = (deepest["representative_read_id"], deepest["template_length"])
    assert picks["longest-template"][1] > picks["median-length"][1]
    assert picks["most-5p-flank"] == picks["longest-template"]  # 5' flank grows here


def test_every_read_with_an_orf_is_mapped():
    rng = np.random.default_rng(19)
    rows = [
        (f"r{i}", _rand(rng, 30) + _orf(rng, 90) + _rand(rng, 30), i % 3)
        for i in range(12)
    ]
    seed, read_map = extract_seed_orfs(_reads(rows), min_aa_length=60)
    assert read_map.num_rows == 12
    assert set(read_map.column("read_id").to_pylist()) == {f"r{i}" for i in range(12)}
    # n_reads over all seeds accounts for every mapped read exactly once.
    assert sum(seed.column("n_reads").to_pylist()) == 12


def test_reads_without_an_orf_are_dropped_not_crashed():
    rng = np.random.default_rng(23)
    rows = [("noorf", "A" * 400, 0), ("short", _rand(rng, 50), 0)]
    rows.append(("real", _rand(rng, 30) + _orf(rng, 90) + _rand(rng, 30), 0))
    seed, read_map = extract_seed_orfs(_reads(rows), min_aa_length=60)
    assert read_map.column("read_id").to_pylist() == ["real"]
    assert seed.num_rows == 1


def test_min_edit_budget_rescues_short_orf_pairs():
    """At identity 0.97 a 30-aa ORF gets int(0.03 * 93) = 2 edits, but two
    reads each at ~1% error differ by ~1.9 bases in expectation — so true
    pairs fall out of the fold and fragment into singletons."""
    rng = np.random.default_rng(29)
    orf = _orf(rng, 30)  # 93 nt ⇒ int(0.03 * 93) == 2 edits of budget
    # Each variant is 3 substitutions from the base — inside a budget of 3,
    # outside a budget of 2.
    variants = [orf] + [
        _point_mutate(orf, [a, b, c])
        for a, b, c in ((10, 25, 40), (14, 31, 55), (20, 37, 61))
    ]
    rows = [
        (f"r{i}", _flank(rng, 25) + v + _flank(rng, 25), 0)
        for i, v in enumerate(variants)
    ]
    seed, _ = extract_seed_orfs(_reads(rows), min_aa_length=30)
    assert seed.num_rows == 4

    tight = fold_orfs(seed, identity=0.97, min_edit_budget=0, kmer=9, window=4)
    loose = fold_orfs(seed, identity=0.97, min_edit_budget=3, kmer=9, window=4)
    assert loose.group_rep_orf.shape[0] < tight.group_rep_orf.shape[0]


def test_fold_is_deterministic():
    rng = np.random.default_rng(31)
    rows = [
        (f"r{i}", _rand(rng, 30) + _orf(rng, 80) + _rand(rng, 30), 0) for i in range(20)
    ]
    tbl = _reads(rows)
    a_seed, a_map = extract_seed_orfs(tbl, min_aa_length=60)
    b_seed, b_map = extract_seed_orfs(tbl, min_aa_length=60)
    assert a_seed.equals(b_seed) and a_map.equals(b_map)
    ra = fold_orfs(a_seed, identity=0.97)
    rb = fold_orfs(b_seed, identity=0.97)
    assert np.array_equal(ra.group_of_orf, rb.group_of_orf)
    assert np.array_equal(ra.group_rep_orf, rb.group_rep_orf)


def test_best_sense_orf_interval_is_codon_aligned():
    """Every downstream length rule assumes ATG→stop inclusive, so the ORF
    interval length is always a multiple of 3."""
    rng = np.random.default_rng(37)
    for n_aa in (30, 77, 150):
        seq = _rand(rng, 40) + _orf(rng, n_aa) + _rand(rng, 40)
        hit = best_sense_orf(seq, min_aa_length=30)
        assert hit is not None
        _prot, st, en = hit
        assert (en - st) % 3 == 0
        assert seq[st : st + 3] == "ATG"
        assert seq[en - 3 : en] in ("TAA", "TAG", "TGA")
