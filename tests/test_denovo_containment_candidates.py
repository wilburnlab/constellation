"""The probe join: candidate pairs for the template graph.

The anchor-star answers "who is a near-duplicate of this bucket's most
abundant member". The template graph asks "what does this sequence sit
inside", and what is pinned here is the difference: a fragment reaches
*every* container, a family is paired *completely*, and the caps that keep
the join bounded say when they bound it.

Two behaviours are pinned because the obvious implementation gets them
wrong: a k-mer repeated far away must not veto a pair (a global diagonal span
does), and an antisense pair must not become a candidate (a diagonal window
alone lets it, because the minimizer hash is canonical).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pytest

import constellation.sequencing.transcriptome.cluster.denovo.candidates as candidates
from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
    CANDIDATE_SCHEMA,
    CONTAINMENT_CANDIDATE_SCHEMA,
    ContainmentCandidates,
    generate_candidates,
    generate_containment_candidates,
)
from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
    extract_minimizers,
)


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _rand(rng, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _substitute(rng, seq: str, n: int, *, keep: tuple[range, ...] = ()) -> str:
    """``n`` substitutions, none inside the ``keep`` ranges."""
    out = list(seq)
    done = 0
    while done < n:
        at = int(rng.integers(len(out)))
        if any(at in span for span in keep):
            continue
        out[at] = rng.choice([b for b in "ACGT" if b != out[at]])
        done += 1
    return "".join(out)


def _index(seqs: list[str]):
    return extract_minimizers(
        pa.array(seqs, type=pa.large_string()), k=19, w=19, max_per_seq=None
    )


def _lengths(seqs: list[str]) -> np.ndarray:
    return np.array([len(s) for s in seqs], dtype=np.int64)


def _run(seqs: list[str], support=None, **kwargs) -> ContainmentCandidates:
    if support is None:
        support = np.ones(len(seqs), dtype=np.int64)
    return generate_containment_candidates(
        _index(seqs), _lengths(seqs), support, **kwargs
    )


def _rows(res: ContainmentCandidates) -> list[tuple]:
    t = res.table
    return list(zip(*(t.column(name).to_pylist() for name in t.column_names)))


def _pairs(res: ContainmentCandidates) -> set[tuple[int, int]]:
    return {(r[0], r[1]) for r in _rows(res)}


def _windows(rng, parent: str, n: int, lo: int, hi: int):
    """``n`` random windows of ``parent``, each with 0-2 substitutions."""
    seqs, spans = [], []
    for _ in range(n):
        length = int(rng.integers(lo, hi + 1))
        a = int(rng.integers(0, len(parent) - length + 1))
        seqs.append(_substitute(rng, parent[a : a + length], int(rng.integers(0, 3))))
        spans.append((a, a + length))
    return seqs, spans


@pytest.fixture(scope="module")
def family():
    """120 windows (300-1200 nt) of one 2 kb parent, 0-2 substitutions each.

    Windows rather than end-trims, so the family holds nested pairs,
    staggered pairs and pairs that do not overlap at all.
    """
    rng = np.random.default_rng(17)
    seqs, spans = _windows(rng, _rand(rng, 2000), 120, 300, 1200)
    support = rng.permutation(len(seqs)).astype(np.int64) + 1  # all distinct
    return seqs, spans, support


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return min(a[1], b[1]) - max(a[0], b[0])


# ── a row-by-row statement of the rule ────────────────────────────────


def _reference(
    index,
    seq_len,
    support,
    *,
    k=19,
    probes_per_seq=16,
    bucket_cap=20_480,
    overflow_anchors=32,
    max_candidates=20_480,
    min_shared=2,
    diag_band=64,
):
    """The same rule as dicts and loops — slow, and hard to get wrong."""
    length = seq_len.tolist()
    sup = support.tolist()
    buckets = defaultdict(list)
    for h, u, p in zip(
        index.mini_hash.tolist(), index.uniq_id.tolist(), index.pos.tolist()
    ):
        buckets[h].append((u, p))

    eligible = defaultdict(list)
    for h, members in buckets.items():
        if len({u for u, _ in members}) < 2:
            continue
        over = len(members) > bucket_cap
        for u, p in members:
            stratum = min(
                p * probes_per_seq // max(length[u] - k + 1, 1), probes_per_seq - 1
            )
            eligible[(u, stratum)].append((over, h, p))
    probes = {key: min(options) for key, options in eligible.items()}
    owned = Counter(owner for owner, _ in probes)

    hits = defaultdict(set)
    overflow_rows = set()
    for (q, _), (over, h, p) in probes.items():
        members = buckets[h]
        if over:
            overflow_rows.add(q)
            members = sorted(
                members, key=lambda e: (-sup[e[0]], -length[e[0]], e[0], e[1])
            )[:overflow_anchors]
        for x, x_pos in members:
            if (length[q], q) < (length[x], x):
                hits[(q, x)].add((p, x_pos - p, over))

    by_src = defaultdict(list)
    n_antisense = 0
    for (q, x), found in hits.items():
        diags = sorted(d for _, d, _ in found)
        best, lead = 0, 0
        for i, d in enumerate(diags):
            inside = sum(1 for e in diags[i:] if e <= d + diag_band)
            if inside > best:
                best, lead = inside, i
        if best < min(min_shared, owned[q]):
            continue
        window = [(p, d) for p, d, _ in found if 0 <= d - diags[lead] <= diag_band]
        sense = [d for _, d in window]
        anti = [d + 2 * p for p, d in window]
        if max(anti) - min(anti) < max(sense) - min(sense):
            n_antisense += 1
            continue
        median = diags[lead + (best - 1) // 2]
        by_src[q].append((q, x, best, median, all(o for _, _, o in found)))

    rows, truncated = [], set()
    for q, found in by_src.items():
        if len(found) > max_candidates:
            truncated.add(q)
            found = sorted(found, key=lambda r: (-r[2], -sup[r[1]], r[1]))
            found = found[:max_candidates]
        rows.extend(found)
    return sorted(rows), sorted(overflow_rows), sorted(truncated), n_antisense


@pytest.fixture(scope="module")
def mixed():
    """Three families, their length ties, antisense copies and strangers."""
    rng = np.random.default_rng(29)
    seqs: list[str] = []
    for parent_len, n in ((900, 14), (1600, 18), (2400, 10)):
        parent = _rand(rng, parent_len)
        for _ in range(n):
            a = int(rng.integers(0, parent_len // 3))
            b = parent_len - int(rng.integers(0, parent_len // 3))
            seqs.append(_substitute(rng, parent[a:b], int(rng.integers(0, 3))))
    seqs.append(seqs[0])  # a byte-identical twin: the tie every key must break
    seqs += [_revcomp(s) for s in seqs[:6]]
    seqs += [_rand(rng, 500) for _ in range(4)]
    support = rng.integers(1, 40, len(seqs)).astype(np.int64)
    return seqs, support


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bucket_cap": 8, "overflow_anchors": 3},
        {"max_candidates": 4},
        {"probes_per_seq": 5, "diag_band": 0},
        {"min_shared": 3},
    ],
    ids=["defaults", "bucket-cap", "candidate-cap", "few-probes", "three-hits"],
)
@pytest.mark.parametrize("packed", [True, False], ids=["one-key", "lexsort"])
def test_the_join_agrees_with_a_row_by_row_reference(
    mixed, kwargs, packed, monkeypatch
):
    """Both sort paths: a wrong bit width is silent in whichever is untested."""
    seqs, support = mixed
    if not packed:
        monkeypatch.setattr(candidates, "_PACKED_ROW_BITS", 0)
    index, lengths = _index(seqs), _lengths(seqs)
    want_rows, want_overflow, want_truncated, want_antisense = _reference(
        index, lengths, support, **kwargs
    )
    got = generate_containment_candidates(
        index, lengths, support, chunk_rows=3_000, **kwargs
    )

    assert _rows(got) == want_rows
    assert got.overflow_rows.tolist() == want_overflow
    assert got.truncated_rows.tolist() == want_truncated
    assert got.stats["n_antisense"] == want_antisense


# ── what the anchor-star could not do ─────────────────────────────────


def test_a_fragment_pairs_with_every_container():
    """Two isoforms differing by a 130-nt alternative 5' exon share a body."""
    rng = np.random.default_rng(41)
    body = _rand(rng, 2870)
    isoform_a = _rand(rng, 130) + body
    isoform_b = _rand(rng, 130) + body
    fragment = body[1000:1600]
    res = _run([isoform_a, fragment, isoform_b])

    assert res.table.schema.equals(CONTAINMENT_CANDIDATE_SCHEMA)
    by_pair = {(r[0], r[1]): r for r in _rows(res)}
    assert {(1, 0), (1, 2)} <= set(by_pair)
    # The diagonal is where the fragment starts on each container.
    assert by_pair[(1, 0)][3] == 1130
    assert by_pair[(1, 2)][3] == 1130
    assert not by_pair[(1, 0)][4] and not by_pair[(1, 2)][4]


def test_a_family_is_paired_completely_not_as_a_star(family):
    """Every pair overlapping by 200 nt, which here is two probe strata.

    A probe is one per sixteenth of the SHORTER sequence, so for a staggered
    pair "complete" holds while the overlap spans two of those; a nested pair
    needs no such condition.
    """
    seqs, spans, support = family
    res = _run(seqs, support)
    got = _pairs(res)
    got |= {(b, a) for a, b in got}

    n = len(seqs)
    want = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _overlap(spans[i], spans[j]) >= 200
    }
    apart = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _overlap(spans[i], spans[j]) <= 0
    }
    assert len(want) > 20 * n, "the fixture must be far from a star"
    assert apart, "the fixture must hold pairs that share nothing"
    assert want <= got
    assert not (apart & got)
    # No indels in this family, so the diagonal is exact: where src starts
    # on dst, in the parent's coordinates.
    for src, dst, _, diag, _ in _rows(res):
        assert diag == spans[src][0] - spans[dst][0]


def test_each_pair_appears_once_shorter_first_and_ties_go_to_the_lower_row():
    rng = np.random.default_rng(43)
    parent = _rand(rng, 1200)
    spans = [(0, 1200), (100, 1100), (0, 1200), (50, 1000), (100, 1100), (250, 1200)]
    seqs = [_substitute(rng, parent[a:b], 1) for a, b in spans]
    seqs.append(seqs[3])  # byte-identical: a tie in length AND content
    lengths = _lengths(seqs)
    rows = _rows(_run(seqs))

    unordered = [frozenset((r[0], r[1])) for r in rows]
    assert len(set(unordered)) == len(unordered) == 7 * 6 // 2
    for src, dst, *_ in rows:
        assert (lengths[src], src) < (lengths[dst], dst)
    pairs = {(r[0], r[1]) for r in rows}
    assert {(0, 2), (1, 4), (3, 6)} <= pairs  # equal lengths: lower row is src


def test_output_does_not_depend_on_chunk_rows_or_threads(family):
    seqs, _, support = family
    seqs = seqs + [_revcomp(s) for s in seqs[:10]]
    support = np.concatenate([support, support[:10]])
    index, lengths = _index(seqs), _lengths(seqs)

    runs = [
        generate_containment_candidates(
            index,
            lengths,
            support,
            bucket_cap=50,
            overflow_anchors=20,
            max_candidates=30,
            chunk_rows=chunk_rows,
            threads=threads,
        )
        for chunk_rows in (2_000, 50_000, 8_000_000)
        for threads in (1, 3)
    ]
    first = runs[0]
    # The fixture has to exercise every reduction, or equality proves little.
    assert first.table.num_rows > 1_000
    assert first.overflow_rows.size and first.truncated_rows.size
    assert first.stats["n_antisense"] > 0
    for other in runs[1:]:
        assert other.table.equals(first.table)
        assert other.stats == first.stats
        assert np.array_equal(other.overflow_rows, first.overflow_rows)
        assert np.array_equal(other.truncated_rows, first.truncated_rows)

    src = first.table.column("src_row").to_numpy()
    dst = first.table.column("dst_row").to_numpy()
    order = np.lexsort((dst, src))
    assert np.array_equal(order, np.arange(src.shape[0])), "sorted by (src, dst)"


# ── the caps say when they bind ───────────────────────────────────────


def test_a_bucket_over_the_cap_falls_back_to_anchors_and_says_so(family):
    seqs, _, support = family
    index, lengths = _index(seqs), _lengths(seqs)
    whole = generate_containment_candidates(index, lengths, support)
    capped = generate_containment_candidates(
        index, lengths, support, bucket_cap=40, overflow_anchors=8
    )

    assert whole.overflow_rows.size == 0
    assert not any(whole.table.column("overflow").to_pylist())
    assert whole.stats["n_overflow_probes"] == 0

    assert capped.stats["bucket_cap"] == 40
    assert capped.stats["bucket_size_max"] > 40
    assert capped.stats["n_overflow_probes"] > 0
    assert capped.overflow_rows.size > 0
    assert capped.stats["n_overflow_templates"] == capped.overflow_rows.size
    assert np.array_equal(capped.overflow_rows, np.unique(capped.overflow_rows))
    flagged = [r for r in _rows(capped) if r[4]]
    assert flagged
    assert {r[0] for r in flagged} <= set(capped.overflow_rows.tolist())
    # The fallback reaches the best-supported members only, so the list is
    # incomplete — which is exactly what the flag is for.
    assert capped.table.num_rows < whole.table.num_rows
    want_rows, want_overflow, _, _ = _reference(
        index, lengths, support, bucket_cap=40, overflow_anchors=8
    )
    assert _rows(capped) == want_rows
    assert capped.overflow_rows.tolist() == want_overflow


def test_a_candidate_list_over_the_cap_is_cut_by_rank_and_says_so(family):
    seqs, _, support = family
    index, lengths = _index(seqs), _lengths(seqs)
    whole = generate_containment_candidates(index, lengths, support)
    cut = generate_containment_candidates(index, lengths, support, max_candidates=10)

    assert whole.truncated_rows.size == 0
    by_src = defaultdict(list)
    for row in _rows(whole):
        by_src[row[0]].append(row)
    crowded = sorted(src for src, rows in by_src.items() if len(rows) > 10)
    assert crowded
    assert cut.truncated_rows.tolist() == crowded
    assert cut.stats["n_truncated_templates"] == len(crowded)

    want = []
    for src, rows in by_src.items():
        ranked = sorted(rows, key=lambda r: (-r[2], -int(support[r[1]]), r[1]))
        want.extend(ranked[:10])
    assert _rows(cut) == sorted(want)


def test_a_row_budget_demotes_the_largest_buckets_and_records_where_it_stopped(
    family,
):
    seqs, _, support = family
    index, lengths = _index(seqs), _lengths(seqs)
    whole = generate_containment_candidates(index, lengths, support)
    assert whole.stats["smallest_demoted_bucket"] == 0
    assert whole.stats["n_rows_projected"] > 100_000

    budget = whole.stats["n_rows_projected"] * 3 // 4
    held = generate_containment_candidates(index, lengths, support, max_rows=budget)
    threshold = held.stats["smallest_demoted_bucket"]
    # Above the smallest bucket demotion could touch: it stopped once it fit.
    assert 33 < threshold <= held.stats["bucket_size_max"]
    assert held.stats["n_rows_projected"] <= budget
    assert held.stats["bucket_cap"] == 20_480, "the cap itself never moves"
    assert held.overflow_rows.size > 0
    assert any(held.table.column("overflow").to_pylist())

    # Largest first: a tighter budget reaches further down, never less far.
    tighter = generate_containment_candidates(
        index, lengths, support, max_rows=budget * 4 // 5
    )
    assert 33 < tighter.stats["smallest_demoted_bucket"] < threshold
    assert tighter.stats["n_rows_projected"] <= budget * 4 // 5
    # A budget nothing can meet demotes all it can — every bucket the 32
    # anchors are fewer than — and reports the overrun.
    floor = generate_containment_candidates(index, lengths, support, max_rows=1)
    assert floor.stats["smallest_demoted_bucket"] == 33
    assert floor.stats["n_rows_projected"] > 1


# ── the two the window has to get right ───────────────────────────────


def test_one_repeated_kmer_does_not_veto_a_true_pair():
    """A 40-nt segment duplicated 1.5 kb downstream, in every member.

    Its minimizer puts a second diagonal, 1,500 away, on every pair it takes
    part in. A global ``dmax - dmin`` span reads that as inconsistency and
    drops the pair; the window counts the diagonal that has the hits.
    """
    rng = np.random.default_rng(47)
    parent = _rand(rng, 3000)
    parent = parent[:2100] + parent[600:640] + parent[2140:]
    repeats = (range(600, 640), range(2100, 2140))
    seqs, starts = [], []
    for _ in range(14):
        a = int(rng.integers(0, 301))
        b = 3000 - int(rng.integers(0, 301))
        keep = tuple(range(r.start - a, r.stop - a) for r in repeats)
        seqs.append(_substitute(rng, parent[a:b], int(rng.integers(0, 3)), keep=keep))
        starts.append(a)

    index = _index(seqs)
    twice = Counter(zip(index.mini_hash.tolist(), index.uniq_id.tolist()))
    carriers = {row for (_, row), n in twice.items() if n == 2}
    assert carriers == set(range(14)), "every member must carry the repeat"

    lengths, support = _lengths(seqs), np.ones(14, dtype=np.int64)
    # At the default, and with a stratum per position — where every shared
    # minimizer probes, so the repeated one certainly does.
    for probes_per_seq in (16, 4096):
        res = generate_containment_candidates(
            index, lengths, support, probes_per_seq=probes_per_seq
        )
        rows = _rows(res)
        assert len(rows) == 14 * 13 // 2
        for src, dst, _, diag, _ in rows:
            assert diag == starts[src] - starts[dst]
        assert res.stats["n_antisense"] == 0


@pytest.mark.parametrize(
    ("length", "probes_per_seq"),
    [(300, 16), (600, 16), (1500, 4096), (3000, 4096)],
)
def test_a_sequence_and_its_reverse_complement_yield_no_candidate(
    length, probes_per_seq
):
    """The window alone does not do this: the hash is canonical.

    An antisense pair shares every minimizer along an anti-diagonal, so two
    probes within 32 nt of each other share a 64-nt window — measured, 200 of
    200 pairs at 600 nt became candidates before the rule that asks which
    line the hits agree on.

    Every case here is one where it is the RULE that removes the pair. With
    16 probes that is up to 600 nt; beyond it two probes seldom share a
    window and the pair has no candidate with or without the rule (5 of 5
    trials at 3 kb), so the long ones probe at every position.
    """
    rng = np.random.default_rng(length)
    for _ in range(5):
        seq = _rand(rng, length)
        res = _run([seq, _revcomp(seq)], probes_per_seq=probes_per_seq)
        # They do share their minimizers, and the join did see the pair.
        assert res.stats["n_with_probes"] == 2 and res.stats["n_rows"] >= 8
        assert res.table.num_rows == 0
        assert res.stats["n_antisense"] == 1


def test_a_fragment_of_a_tandem_repeat_is_not_called_antisense():
    """Each probe of the fragment hits its own copy and the copies one unit
    either side, so the best window spreads further along the diagonal axis
    than the probes are apart on the sequence — and "fits the anti-diagonal
    better" was read off exactly that. Two probes that agree on a diagonal
    are a candidate whatever else the window holds."""
    rng = np.random.default_rng(71)
    n_spread = 0
    for _ in range(400):
        unit = int(rng.choice([20, 25, 30]))
        length = int(rng.choice([40, 50, 60]))
        container = _rand(rng, 400) + _rand(rng, unit) * 3 + _rand(rng, 400)
        start = 400 + int(rng.integers(0, 2 * unit))
        seqs = [container[start : start + length], container]
        index = _index(seqs)
        mine = index.mini_hash[index.uniq_id == 0].tolist()
        theirs = index.mini_hash[index.uniq_id == 1].tolist()
        n_spread += any(theirs.count(h) > 1 for h in mine)

        res = _run(seqs)
        assert res.stats["n_antisense"] == 0
        assert _pairs(res) == {(0, 1)}
    # Otherwise no probe ever hit two copies and this says nothing.
    assert n_spread >= 200


def test_probes_an_indel_apart_are_a_sense_pair():
    """The other half of the rule. An indel between two probes puts them on
    two diagonals, so no two agree on one — and the pair is still sense: the
    diagonal moves by the length of the indel, the anti-diagonal by twice
    the distance between the probes."""
    rng = np.random.default_rng(72)
    n_judged = 0
    for _ in range(1200):
        fragment = _rand(rng, 90)
        cuts = sorted(int(c) for c in rng.choice(np.arange(20, 70), 2, replace=False))
        inside = (
            fragment[: cuts[0]]
            + _rand(rng, int(rng.integers(1, 4)))
            + fragment[cuts[0] : cuts[1]]
            + fragment[cuts[1] + int(rng.integers(1, 4)) :]
        )
        seqs = [fragment, _rand(rng, 300) + inside + _rand(rng, 300)]
        index = _index(seqs)
        own = index.uniq_id == 0
        mine = dict(zip(index.mini_hash[own].tolist(), index.pos[own].tolist()))
        diags = [
            int(x) - mine[int(h)]
            for h, x in zip(index.mini_hash[~own].tolist(), index.pos[~own].tolist())
            if int(h) in mine
        ]
        if len(diags) < 2 or len(set(diags)) != len(diags):
            continue
        n_judged += 1
        res = _run(seqs)
        assert res.stats["n_antisense"] == 0
        assert _pairs(res) == {(0, 1)}
    assert n_judged >= 150, n_judged


_RAN = mp.get_context("fork").Value("i", 0)
_REAL_REDUCE = candidates._reduce_block


def _fails_on_the_first_block(p0: int, p1: int) -> tuple:
    if p0 == 0:
        raise RuntimeError("planted in a worker")
    with _RAN.get_lock():
        _RAN.value += 1
    time.sleep(0.05)
    return _REAL_REDUCE(p0, p1)


def test_a_failure_in_a_worker_stops_the_join(family, monkeypatch):
    """The first block fails at once. Leaving the pool waits for whatever is
    still queued, so unless the rest is cancelled the whole join runs — for
    a result nobody will read — before the caller hears of the failure."""
    seqs, _, support = family
    index, lengths = _index(seqs), _lengths(seqs)
    cut = []
    real_cut = candidates._cut_blocks

    def spy(*args):
        cut.append(real_cut(*args))
        return cut[-1]

    monkeypatch.setattr(candidates, "_cut_blocks", spy)
    monkeypatch.setattr(candidates, "_reduce_block", _fails_on_the_first_block)
    _RAN.value = 0
    with pytest.raises(RuntimeError, match="planted in a worker"):
        generate_containment_candidates(
            index, lengths, support, chunk_rows=200, threads=2
        )
    n_blocks = len(cut[0])
    assert n_blocks >= 40
    assert _RAN.value < n_blocks // 2, (_RAN.value, n_blocks)
    assert candidates._JOIN_STATE is None


def test_antisense_copies_cost_a_family_none_of_its_pairs(family):
    seqs, _, support = family
    seqs, support = seqs[:40], support[:40]
    alone = _pairs(_run(seqs, support))
    both = _run(seqs + [_revcomp(s) for s in seqs], np.tile(support, 2))

    sense = {(a, b) for a, b in _pairs(both) if a < 40 and b < 40}
    crossed = {(a, b) for a, b in _pairs(both) if (a < 40) != (b < 40)}
    assert sense == alone
    assert not crossed
    assert both.stats["n_antisense"] > 0, "the rule removed them, not luck"


def test_a_sequence_with_one_probe_is_held_to_one_hit():
    """37 nt is one window, so one minimizer: it cannot show two hits."""
    rng = np.random.default_rng(53)
    container = _rand(rng, 1000)
    res = _run([container, container[400:437], _rand(rng, 1000)])
    assert _rows(res) == [(1, 0, 1, 400, False)]
    assert res.stats["n_with_probes"] == 2


# ── the edges of the contract ─────────────────────────────────────────


def test_sequences_that_share_nothing_cost_no_rows():
    rng = np.random.default_rng(59)
    res = _run([_rand(rng, 800) for _ in range(6)])
    assert res.table.num_rows == 0
    assert res.table.schema.equals(CONTAINMENT_CANDIDATE_SCHEMA)
    assert res.stats["n_probes"] == 0 and res.stats["n_rows_projected"] == 0
    assert res.stats["n_sequences"] == 6

    nothing = _run(["ACGT", "AC"])  # shorter than k: no minimizer at all
    assert nothing.table.num_rows == 0 and nothing.stats["n_buckets"] == 0


def test_stats_are_plain_numbers_with_the_documented_keys(family):
    seqs, _, support = family
    stats = _run(seqs, support).stats
    assert all(type(v) in (int, float) for v in stats.values())
    assert json.loads(json.dumps(stats)) == stats
    assert {
        "n_sequences",
        "n_with_probes",
        "n_probes",
        "n_overflow_probes",
        "n_rows_projected",
        "n_rows",
        "n_candidates",
        "n_overflow_templates",
        "n_truncated_templates",
        "bucket_cap",
        "smallest_demoted_bucket",
        "bucket_size_max",
        "bucket_size_mean",
        "bucket_size_biased_mean",
        "n_buckets",
    } <= set(stats)
    assert stats["n_candidates"] > 0
    assert 0 < stats["n_rows"] <= stats["n_rows_projected"]
    # Size-biased: what a random shared entry sees, never below the plain mean.
    assert stats["bucket_size_biased_mean"] >= stats["bucket_size_mean"] > 1.0


def test_inputs_that_do_not_describe_the_index_are_refused():
    rng = np.random.default_rng(61)
    parent = _rand(rng, 600)
    seqs = [parent, parent[50:500]]
    index, lengths = _index(seqs), _lengths(seqs)
    ones = np.ones(2, dtype=np.int64)

    with pytest.raises(ValueError, match="same length"):
        generate_containment_candidates(index, lengths, np.ones(3))
    with pytest.raises(ValueError, match="seq_len has 1 rows"):
        generate_containment_candidates(index, lengths[:1], ones[:1])
    with pytest.raises(ValueError, match="len - k"):
        generate_containment_candidates(index, lengths // 4, ones)
    with pytest.raises(ValueError, match="bucket_cap"):
        generate_containment_candidates(index, lengths, ones, bucket_cap=1)
    with pytest.raises(ValueError, match="NaN"):
        generate_containment_candidates(index, lengths, np.array([1.0, np.nan]))


# ── the shipped generator, unchanged ──────────────────────────────────


class _Lcg:
    """Knuth's MMIX generator, written out so the golden fixture cannot move
    with numpy's or the standard library's."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def below(self, n: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (
            1 << 64
        )
        return (self.state >> 33) % n

    def bases(self, n: int) -> str:
        return "".join("ACGT"[self.below(4)] for _ in range(n))


def _golden_fixture() -> tuple[list[str], np.ndarray]:
    """Three families of ten: end-trimmed, 0-2 substitutions, abundance 1-9."""
    gen = _Lcg(20260927)
    seqs, abundance = [], []
    for _ in range(3):
        parent = gen.bases(900)
        for _ in range(10):
            a = gen.below(120)
            b = 900 - gen.below(120)
            seq = list(parent[a:b])
            for _ in range(gen.below(3)):
                at = gen.below(len(seq))
                seq[at] = "ACGT"[("ACGT".index(seq[at]) + 1 + gen.below(3)) % 4]
            seqs.append("".join(seq))
            abundance.append(1 + gen.below(9))
    return seqs, np.array(abundance, dtype=np.int64)


#: What `generate_candidates` returned on `_golden_fixture()` at e112119, the
#: commit the probe join was added on top of — observed, not recomputed.
_GOLDEN_ROWS = [
    (0, 1, 41),
    (0, 2, 4),
    (0, 3, 4),
    (0, 5, 8),
    (0, 6, 2),
    (0, 7, 3),
    (0, 8, 2),
    (0, 9, 7),
    (1, 2, 45),
    (1, 3, 46),
    (1, 4, 49),
    (1, 5, 39),
    (1, 6, 46),
    (1, 7, 46),
    (1, 8, 47),
    (1, 9, 42),
    (10, 11, 8),
    (10, 12, 8),
    (10, 13, 38),
    (10, 14, 8),
    (10, 16, 2),
    (10, 17, 7),
    (10, 18, 8),
    (10, 19, 6),
    (11, 12, 3),
    (11, 13, 39),
    (11, 14, 2),
    (11, 18, 2),
    (12, 13, 39),
    (13, 14, 40),
    (13, 15, 49),
    (13, 16, 47),
    (13, 17, 43),
    (13, 18, 39),
    (13, 19, 40),
    (20, 25, 45),
    (20, 28, 2),
    (20, 29, 3),
    (21, 25, 46),
    (21, 29, 3),
    (22, 25, 42),
    (22, 29, 5),
    (23, 25, 42),
    (23, 29, 5),
    (24, 25, 45),
    (25, 26, 42),
    (25, 27, 43),
    (25, 28, 43),
    (25, 29, 45),
    (26, 29, 3),
    (27, 29, 4),
]


def test_the_anchor_star_returns_exactly_what_it_did_before():
    """Golden: the probe join was appended beside it, not into it."""
    seqs, abundance = _golden_fixture()
    # If this fails the FIXTURE moved, and the rows below say nothing.
    digest = hashlib.sha256("".join(seqs).encode()).hexdigest()
    assert digest == (
        "ad9739f492398e6f380e0b8212c09fc39234334da94235b1ff902cc60a37a14a"
    )
    assert "".join(map(str, abundance)) == "372311213286591973447761596438"

    # The shipped sketch: k15 / w10, bottom-50.
    index = extract_minimizers(pa.array(seqs, type=pa.large_string()))
    table = generate_candidates(index, abundance)

    assert table.schema.equals(CANDIDATE_SCHEMA)
    got = list(
        zip(
            table.column("uniq_a").to_pylist(),
            table.column("uniq_b").to_pylist(),
            table.column("n_shared").to_pylist(),
        )
    )
    assert got == _GOLDEN_ROWS
