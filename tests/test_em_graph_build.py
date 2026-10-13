"""The template graph's builder, merge predicate and cluster edges.

``test_em_graph.py`` pins what ONE pair is. This pins what happens around the
kernel: which pairs reach it, how byte-identical rows are measured once and
written for each of them, that the edges are the same however the work was
cut up, and what is left on disk when the work is done — or is not.

The rule the builder is held to is that it adds nothing to the kernel's
answer: over a family small enough to measure every pair, the edges are
exactly the ones a double loop over ``relate_pair`` produces.

None of this needs minimap2.
"""

from __future__ import annotations

import ast
import json
import multiprocessing as mp
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("edlib")
pytest.importorskip("xxhash")

import constellation.sequencing.transcriptome.cluster.denovo.em.graph as graph
from constellation.sequencing.transcriptome.cluster.denovo.em.graph import (
    CLUSTER_EDGE_TABLE,
    EDGE_FEATURE_FIELDS,
    GRAPH_KERNEL_VERSION,
    PRODUCED_RELATIONS,
    RELATION_NAMES,
    TEMPLATE_EDGE_TABLE,
    TRUNCATION_NAMES,
    GraphParams,
    GraphResult,
    MergePredicate,
    Relation,
    build_graph,
    cluster_edges,
    graph_stamp,
    input_digest,
    is_mergeable,
    mergeable_pairs,
    relate_pair,
    split_origins,
)

P = GraphParams()
_DICT = pa.dictionary(pa.int8(), pa.string())
_OTHER_BASE = str.maketrans("ACGT", "CGTA")


def rnd(n: int, rng: random.Random) -> str:
    return "".join(rng.choices("ACGT", k=n))


def substituted(seq: str, *positions: int) -> str:
    """``seq`` with a different base at each of ``positions``."""
    out = list(seq)
    for p in positions:
        out[p] = seq[p].translate(_OTHER_BASE)
    return "".join(out)


def without(seq: str, at: int, n: int) -> str:
    """``seq`` with the ``n`` bases from ``at`` removed."""
    return seq[:at] + seq[at + n :]


def column(seqs: list[str]) -> pa.Array:
    return pa.array(seqs, type=pa.large_string())


def ids_of(n: int) -> np.ndarray:
    """Template ids that cannot be mistaken for rows."""
    return (3 << 40) | np.arange(n, dtype=np.int64)


def build(seqs: list[str], **kwargs) -> GraphResult:
    n = len(seqs)
    kwargs.setdefault("ids", ids_of(n))
    kwargs.setdefault("n_reads", np.arange(n, dtype=np.int64) + 1)
    kwargs.setdefault("node_round", 2)
    return build_graph(column(seqs), **kwargs)


def rows(table: pa.Table) -> list[tuple]:
    """Every edge as a tuple, sorted: all columns but ``identity`` exactly,
    and ``identity`` — whose last digits are path-dependent — rounded."""
    names = [n for n in table.column_names if n != "identity"]
    exact = zip(*(table.column(n).to_pylist() for n in names))
    identity = [round(x, 9) for x in table.column("identity").to_pylist()]
    return sorted((*row, x) for row, x in zip(exact, identity))


def edge(table: pa.Table, src: int, dst: int) -> dict:
    """The one edge from row ``src`` to row ``dst``."""
    found = [r for r in table.to_pylist() if (r["src_row"], r["dst_row"]) == (src, dst)]
    assert len(found) == 1, (src, dst, len(found))
    return found[0]


def pairs(table: pa.Table) -> set[tuple[int, int, str]]:
    return set(
        zip(
            table.column("src_row").to_pylist(),
            table.column("dst_row").to_pylist(),
            table.column("relation").to_pylist(),
        )
    )


@pytest.fixture
def small_tasks(monkeypatch):
    """Cut the kernel's work into many tasks, so a pool has something to
    share out on a fixture small enough to check."""
    monkeypatch.setattr(graph, "_TASK_MAX_PAIRS", 37)


# ──────────────────────────────────────────────────────────────────────
# A family, measured both ways
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def family():
    """60 rows cut from one 2.5 kb parent: windows of 300-1500 nt carrying
    0-2 substitutions, equal-length variants of some of them, and
    byte-identical copies of others — shuffled, so copies and variants do not
    sit beside what they were made from.

    Substitutions keep 20 nt clear of a window's ends: a difference within
    4 nt of a junction can be read as overhang or as alignment at equal cost,
    and what is compared here is the builder, not that ambiguity.
    """
    rng = random.Random(41)
    parent = rnd(2500, rng)
    seqs: list[str] = []
    spans: list[tuple[int, int]] = []
    for _ in range(44):
        length = rng.randint(300, 1500)
        a = rng.randint(0, len(parent) - length)
        at = rng.sample(range(20, length - 20), rng.randint(0, 2))
        seqs.append(substituted(parent[a : a + length], *at))
        spans.append((a, a + length))
    for i in rng.sample(range(44), 8):  # equal-length variants
        a, b = spans[i]
        seqs.append(substituted(parent[a:b], rng.randrange(20, b - a - 20)))
        spans.append((a, b))
    for i in rng.sample(range(52), 8):  # byte-identical copies
        seqs.append(seqs[i])
        spans.append(spans[i])
    order = list(range(len(seqs)))
    rng.shuffle(order)
    seqs = [seqs[i] for i in order]
    spans = [spans[i] for i in order]
    n_reads = np.array([rng.randint(1, 50) for _ in seqs], dtype=np.int64)
    return seqs, spans, n_reads


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return min(a[1], b[1]) - max(a[0], b[0])


def test_a_family_yields_exactly_the_edges_of_a_double_loop(family):
    """The builder adds nothing to the kernel and loses nothing from it.

    Both sides are restricted to pairs overlapping by at least 200 nt on the
    parent. Below that the probe join's recall falls with the overlap, and
    whether a candidate was FOUND is the join's own test's question; this one
    asks what becomes of a pair once it has been.
    """
    seqs, spans, n_reads = family
    raw = [s.encode() for s in seqs]
    want = set()
    n_tied = 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if _overlap(spans[i], spans[j]) < 200:
                continue
            src, dst = (i, j) if len(raw[i]) <= len(raw[j]) else (j, i)
            m = relate_pair(raw[src], raw[dst], spans[src][0] - spans[dst][0], P)
            if m.relation in PRODUCED_RELATIONS:
                want.add((src, dst, RELATION_NAMES[m.relation]))
                n_tied += len(raw[src]) == len(raw[dst])

    res = build(seqs, n_reads=n_reads, threads=1)
    got = {
        (s, d, name)
        for s, d, name in pairs(res.edges)
        if _overlap(spans[s], spans[d]) >= 200
    }
    assert got == want
    # Otherwise this compared two empty sets, or never met a tie or a copy.
    assert len(want) >= 100
    assert {name for _, _, name in want} == {"equivalent", "contained"}
    assert n_tied >= 16
    assert res.stats["n_identical_classes"] >= 6
    assert res.stats["n_edges"] == res.edges.num_rows == len(pairs(res.edges))


def test_the_edges_do_not_depend_on_threads_or_on_streaming(
    family, small_tasks, tmp_path
):
    seqs, _, n_reads = family
    origin = np.arange(len(seqs), dtype=np.int64) % 7 - 1
    kwargs = dict(n_reads=n_reads, split_origin=origin)

    one = build(seqs, threads=1, **kwargs)
    three = build(seqs, threads=3, **kwargs)
    path = tmp_path / "edges.parquet"
    streamed = build(seqs, threads=3, output_path=path, **kwargs)

    assert one.stats["n_pairs_aligned"] > 10 * 37  # several tasks, so a pool
    assert rows(three.edges) == rows(one.edges)
    assert rows(pq.read_table(path)) == rows(one.edges)
    # Not only the same set: the same rows in the same order.
    assert three.edges.drop(["identity"]).equals(one.edges.drop(["identity"]))
    assert pq.read_table(path).drop(["identity"]).equals(one.edges.drop(["identity"]))
    for other in (three, streamed):
        counted = {k: v for k, v in other.stats.items() if k != "seconds"}
        assert counted == {k: v for k, v in one.stats.items() if k != "seconds"}


def test_the_edges_do_not_depend_on_how_the_tasks_were_cut(family, monkeypatch):
    seqs, _, n_reads = family
    whole = build(seqs, n_reads=n_reads, threads=1)
    monkeypatch.setattr(graph, "_TASK_MAX_PAIRS", 5)
    monkeypatch.setattr(graph, "_BATCH_ROWS", 11)
    cut = build(seqs, n_reads=n_reads, threads=2)
    assert cut.edges.drop(["identity"]).equals(whole.edges.drop(["identity"]))


def test_the_stats_count_the_rows_that_were_written(family):
    seqs, _, n_reads = family
    origin = np.arange(len(seqs), dtype=np.int64) % 2  # kin: same parity
    pred = MergePredicate(max_edits=1)
    res = build(seqs, n_reads=n_reads, split_origin=origin, predicate=pred)
    t, s = res.edges.to_pydict(), res.stats

    eq = [i for i, r in enumerate(t["relation"]) if r == "equivalent"]
    con = [i for i, r in enumerate(t["relation"]) if r == "contained"]
    assert s["n_edges"] == len(t["relation"]) == len(eq) + len(con)
    assert s["n_equivalent"] == len(eq)
    assert s["n_contained"] == len(con)
    assert s["n_equivalent_exact"] == sum(t["n_edits"][i] == 0 for i in eq)
    assert s["n_exact_nested"] == sum(t["n_edits"][i] == 0 for i in con)
    assert s["contained"] == {
        name: sum(t["truncation"][i] == name for i in con) for name in TRUNCATION_NAMES
    }
    assert sum(s["contained"].values()) == s["n_contained"]
    edits = [t["n_edits"][i] for i in eq]
    assert s["n_edits_hist"] == {
        "0": edits.count(0),
        "1": edits.count(1),
        "2": edits.count(2),
        "3-5": sum(3 <= e <= 5 for e in edits),
        "6+": sum(e >= 6 for e in edits),
    }
    assert s["n_mergeable"] == sum(t["mergeable"]) > 0
    assert s["n_same_split_origin"] == sum(t["same_split_origin"][i] for i in eq) > 0
    assert s["n_pairs_aligned"] == s["n_edges_measured"] + sum(s["dropped"].values())
    assert set(s["dropped"]) == {
        RELATION_NAMES[int(r)] for r in Relation if r not in PRODUCED_RELATIONS
    }
    assert s["n_pairs_aligned"] == s["candidates"]["n_candidates"]
    assert s["n_sequences"] == len(seqs)
    assert s["n_unique"] == len(set(seqs)) == s["candidates"]["n_sequences"]
    assert set(s["seconds"]) == {"sketch", "candidates", "kernel", "total"}


def test_the_stats_are_json(family):
    seqs, _, n_reads = family
    res = build(seqs, n_reads=n_reads, params=GraphParams(max_candidates=3))
    assert res.stats["n_truncated_templates"] > 0
    assert json.loads(json.dumps(res.stats)) == res.stats


# ──────────────────────────────────────────────────────────────────────
# What the join has to find
# ──────────────────────────────────────────────────────────────────────


def test_a_fragment_is_contained_in_both_isoforms():
    """An anchor-star pairs a fragment with one container, the best
    supported; the fragment lies just as much inside the other."""
    rng = random.Random(42)
    body = rnd(2870, rng)
    one = rnd(130, rng) + body
    two = rnd(130, rng) + body
    fragment = body[1000:1600]
    res = build([one, fragment, two], n_reads=np.array([900, 1, 2]))

    assert pairs(res.edges) == {(1, 0, "contained"), (1, 2, "contained")}
    for dst in (0, 2):
        e = edge(res.edges, 1, dst)
        assert e["truncation"] == "both"
        assert (e["dst_overhang_5p"], e["dst_overhang_3p"]) == (1130, 1270)
        assert (e["delta_5p"], e["delta_3p"]) == (1130, 1270)
        assert (e["src_len"], e["dst_len"], e["n_edits"]) == (600, 3000, 0)
        assert (e["src_n_reads"], e["dst_n_reads"]) == (1, 900 if dst == 0 else 2)
        assert not e["mergeable"]
    # The isoforms met, and are not an edge: they differ by a first exon.
    assert res.stats["n_pairs_aligned"] == 3
    assert sum(res.stats["dropped"].values()) == 1


def test_an_edge_carries_its_rows_ids_and_round():
    rng = random.Random(43)
    a = rnd(1500, rng)
    res = build(
        [a[200:], a],
        ids=np.array([77, 99]),
        n_reads=np.array([4, 6]),
        node_round=5,
    )
    e = edge(res.edges, 0, 1)
    assert (e["node_round"], e["src_template_id"], e["dst_template_id"]) == (5, 77, 99)
    assert (e["relation"], e["truncation"]) == ("contained", "5p")
    assert e["n_shared_probes"] >= 2 and not e["candidate_overflow"]
    assert e["identity"] == 1.0 and e["aligned_len"] == 1300


def test_the_dictionary_index_is_the_enum_value(tmp_path):
    rng = random.Random(44)
    a = rnd(1200, rng)
    seqs = [a, substituted(a, 600), a[100:], a[:1000], a[150:1050]]
    path = tmp_path / "edges.parquet"
    in_memory = build(seqs).edges
    build(seqs, output_path=path)
    for table in (in_memory, pq.read_table(path)):
        assert set(table["relation"].to_pylist()) == {"equivalent", "contained"}
        assert set(table["truncation"].to_pylist()) == {None, "5p", "3p", "both"}
        for chunk in table["relation"].chunks:
            assert chunk.dictionary.to_pylist() == [RELATION_NAMES[i] for i in range(8)]
            assert [Relation(i).name.lower() for i in chunk.indices.to_pylist()] == (
                chunk.to_pylist()
            )
        for chunk in table["truncation"].chunks:
            assert chunk.dictionary.to_pylist() == list(TRUNCATION_NAMES)


# ──────────────────────────────────────────────────────────────────────
# Byte-identical rows
# ──────────────────────────────────────────────────────────────────────

_TWIN = {
    "relation": "equivalent",
    "truncation": None,
    "src_overhang_5p": 0,
    "src_overhang_3p": 0,
    "dst_overhang_5p": 0,
    "dst_overhang_3p": 0,
    "delta_5p": 0,
    "delta_3p": 0,
    "n_edits": 0,
    "core_len_delta": 0,
    "n_mismatch": 0,
    "n_insert": 0,
    "n_delete": 0,
    "identity": 1.0,
    "edit_distance_placed": 0,
    "div_5p": 0,
    "div_3p": 0,
    "n_shared_probes": 0,
    "candidate_overflow": False,
}

_OF_THE_PAIR = [
    n
    for n in TEMPLATE_EDGE_TABLE.names
    if n
    not in (
        "src_template_id",
        "src_row",
        "src_n_reads",
        "dst_template_id",
        "dst_row",
        "dst_n_reads",
        "same_split_origin",
        "mergeable",
    )
]


def test_byte_identical_rows_are_joined_by_twin_edges():
    rng = random.Random(45)
    a = rnd(1400, rng)
    twin = a[300:1100]
    seqs = [twin, a, twin, rnd(900, rng), twin]
    res = build(seqs, n_reads=np.array([5, 50, 7, 1, 9]))

    for src, dst in ((0, 2), (0, 4), (2, 4)):
        e = edge(res.edges, src, dst)
        assert {k: e[k] for k in _TWIN} == _TWIN
        assert e["src_len"] == e["dst_len"] == e["aligned_len"] == 800
        assert (e["src_template_id"], e["dst_template_id"]) == (
            ids_of(5)[src],
            ids_of(5)[dst],
        )
        # Each member's own reads, not the class's 21.
        assert (e["src_n_reads"], e["dst_n_reads"]) == (
            [5, 50, 7, 1, 9][src],
            [5, 50, 7, 1, 9][dst],
        )
        assert e["mergeable"]
    s = res.stats
    assert (s["n_unique"], s["n_identical_classes"]) == (3, 1)
    assert (s["largest_identical_class"], s["n_exact_twins"]) == (3, 3)


def test_every_edge_of_a_representative_is_written_for_each_twin():
    rng = random.Random(45)
    a = rnd(1400, rng)
    twin = a[300:1100]
    seqs = [twin, a, twin, rnd(900, rng), twin]
    res = build(seqs)

    assert pairs(res.edges) == {
        (0, 2, "equivalent"),
        (0, 4, "equivalent"),
        (2, 4, "equivalent"),
        (0, 1, "contained"),
        (2, 1, "contained"),
        (4, 1, "contained"),
    }
    first = edge(res.edges, 0, 1)
    assert (first["dst_overhang_5p"], first["dst_overhang_3p"]) == (300, 300)
    assert first["n_shared_probes"] > 0
    for src in (2, 4):
        e = edge(res.edges, src, 1)
        assert {k: e[k] for k in _OF_THE_PAIR} == {k: first[k] for k in _OF_THE_PAIR}
        assert e["src_template_id"] == ids_of(5)[src]
        assert e["src_n_reads"] == src + 1
    # One alignment, three edges.
    assert res.stats["n_pairs_aligned"] == res.stats["n_edges_measured"] == 1
    assert res.stats["n_contained"] == 3


def test_two_classes_are_joined_member_by_member():
    """|U| x |V| edges from the one measurement, split kin included."""
    rng = random.Random(46)
    a = rnd(1000, rng)
    short = a[:700]
    seqs = [a, short, a, short, a]
    origin = np.array([9, 9, -1, 4, 4])
    res = build(seqs, split_origin=origin)

    contained = {(s, d) for s, d, name in pairs(res.edges) if name == "contained"}
    assert contained == {(s, d) for s in (1, 3) for d in (0, 2, 4)}
    assert res.stats["n_pairs_aligned"] == 1
    assert res.stats["n_edges"] == 6 + 3 + 1
    kin = {
        (r["src_row"], r["dst_row"])
        for r in res.edges.to_pylist()
        if r["same_split_origin"]
    }
    assert kin == {(1, 0), (3, 4)}


def test_a_twin_of_a_capped_row_is_named_too():
    """Byte-identical rows share one candidate list. The 400-mer lies in four
    longer rows and may keep three; the 700-mer lies in three."""
    rng = random.Random(47)
    a = rnd(1500, rng)
    seqs = [a[:400], a, a[:900], a[:400], a[:1200], a[:700]]
    res = build(seqs, params=GraphParams(max_candidates=3))
    assert res.truncated_rows.tolist() == [0, 3]
    assert res.truncated_rows.dtype == np.int64
    assert res.overflow_rows.tolist() == []
    assert res.stats["n_truncated_templates"] == 2
    assert res.stats["candidates"]["n_truncated_templates"] == 1
    assert res.stats["n_overflow_templates"] == 0
    # An edge of a row whose list was cut says so; a twin edge came through
    # no list at all.
    through_a_cap = {
        (r["src_row"], r["dst_row"])
        for r in res.edges.to_pylist()
        if r["candidate_overflow"]
    }
    kept = {(s, d) for s, d, _ in pairs(res.edges) if s in (0, 3) and d not in (0, 3)}
    assert through_a_cap == kept and len(kept) == 2 * 3


def test_an_edge_found_through_the_anchor_fallback_says_so():
    """A bucket over the cap is joined against its best-supported members
    only. What the join says of a pair — how it was found, and on how many
    probes — is on the edge."""
    from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
        generate_containment_candidates,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
        extract_minimizers,
    )

    rng = random.Random(55)
    a = rnd(1200, rng)
    seqs = [a, a[:900], a[100:], a[50:1000], a[:1100]]
    n_reads = np.array([9, 1, 7, 3, 5])
    tight = GraphParams(bucket_cap=2, overflow_anchors=2)
    res = build(seqs, n_reads=n_reads, params=tight)

    found = generate_containment_candidates(
        extract_minimizers(column(seqs), k=19, w=19, max_per_seq=None),
        np.array([len(s) for s in seqs]),
        n_reads,
        k=19,
        bucket_cap=2,
        overflow_anchors=2,
    )
    told = found.table.to_pydict()
    said = {
        pair: what
        for pair, what in zip(
            zip(told["src_row"], told["dst_row"]),
            zip(told["n_shared"], told["overflow"]),
        )
    }
    on_the_edge = {
        (r["src_row"], r["dst_row"]): (r["n_shared_probes"], r["candidate_overflow"])
        for r in res.edges.to_pylist()
    }
    assert on_the_edge == {pair: said[pair] for pair in on_the_edge}
    assert {over for _, over in on_the_edge.values()} == {True, False}
    assert res.overflow_rows.tolist() == found.overflow_rows.tolist() != []
    assert res.stats["n_overflow_templates"] == len(res.overflow_rows)
    assert res.stats["n_pairs_aligned"] == found.table.num_rows


# ──────────────────────────────────────────────────────────────────────
# Equal lengths
# ──────────────────────────────────────────────────────────────────────


def _tied_pair() -> tuple[str, str]:
    """Two sequences of one length that differ at both ends and inside:
    ``a`` starts 11 nt early and lacks one base, ``b`` runs 10 nt on."""
    rng = random.Random(25)
    core = rnd(1500, rng)
    a = rnd(11, rng) + without(substituted(core, 400), 900, 1)
    b = core + rnd(10, rng)
    assert len(a) == len(b)
    return a, b


def test_an_equivalent_edge_between_equal_lengths_runs_from_the_lower_row(family):
    seqs, _, n_reads = family
    t = build(seqs, n_reads=n_reads).edges.to_pydict()
    tied = [
        i
        for i in range(len(t["relation"]))
        if t["relation"][i] == "equivalent" and t["src_len"][i] == t["dst_len"][i]
    ]
    assert len(tied) >= 16
    assert all(t["src_row"][i] < t["dst_row"][i] for i in tied)
    assert all(t["src_len"][i] <= t["dst_len"][i] for i in range(len(t["relation"])))
    assert all(t["src_row"][i] != t["dst_row"][i] for i in range(len(t["relation"])))


def test_an_edge_turned_round_is_the_measurement_mirrored():
    """Rows 0 and 1 are measured; rows 2 and 3 are their copies, standing the
    other way round. The edge between a copy and an original is what the
    kernel says of that pair in that order."""
    a, b = _tied_pair()
    n_reads = np.array([10, 20, 30, 40])
    res = build([a, b, b, a], n_reads=n_reads)
    assert res.stats["n_pairs_aligned"] == 1

    measured = ("n_edits", "aligned_len", "core_len_delta", "n_mismatch")
    measured += ("n_insert", "n_delete", "div_5p", "div_3p")
    ends = ("src_overhang_5p", "src_overhang_3p", "dst_overhang_5p", "dst_overhang_3p")
    for src, dst, first, second, diag in (
        (0, 1, a, b, -11),
        (0, 2, a, b, -11),
        (1, 3, b, a, 11),
        (2, 3, b, a, 11),
    ):
        e = edge(res.edges, src, dst)
        m = relate_pair(first.encode(), second.encode(), diag, P)
        assert e["relation"] == "equivalent"
        assert [e[k] for k in ends] == [getattr(m, k) for k in ends]
        assert [e[k] for k in measured] == [getattr(m, k) for k in measured]
        assert e["delta_5p"] == m.dst_overhang_5p - m.src_overhang_5p
        assert e["delta_3p"] == m.dst_overhang_3p - m.src_overhang_3p
        assert (e["src_n_reads"], e["dst_n_reads"]) == (n_reads[src], n_reads[dst])
        assert e["src_template_id"] == ids_of(4)[src]
    assert edge(res.edges, 0, 1)["delta_5p"] == -11
    assert edge(res.edges, 1, 3)["delta_5p"] == 11
    assert edge(res.edges, 1, 3)["n_insert"] == 1
    # The one column that is not mirrored: b was never placed inside a.
    as_measured = relate_pair(a.encode(), b.encode(), -11, P).edit_distance_placed
    for src, dst in ((0, 1), (0, 2), (1, 3), (2, 3)):
        assert edge(res.edges, src, dst)["edit_distance_placed"] == as_measured
    assert {(s, d) for s, d, _ in pairs(res.edges)} == {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (0, 3),
        (1, 2),
    }


_MIRRORED = ("relation", "truncation", "src_len", "dst_len", "n_edits", "aligned_len")
_MIRRORED += ("src_overhang_5p", "src_overhang_3p", "dst_overhang_5p")
_MIRRORED += ("dst_overhang_3p", "delta_5p", "delta_3p", "div_5p", "div_3p")
_MIRRORED += ("core_len_delta", "n_mismatch", "n_insert", "n_delete", "mergeable")


def _staggered_near_an_end() -> tuple[str, str]:
    """Equal lengths, 28 nt apart at both ends, and one substitution 5 nt
    into the shared span — so the 5' end is 34 nt from agreeing, past the
    tolerance, and the 3' end 28, inside it. ``a`` is the one cut short."""
    body = rnd(1256, random.Random(61))
    return substituted(body[28:], 5), body[:-28]


def test_the_edge_of_an_equal_length_pair_does_not_depend_on_row_order():
    """Between equal lengths the kernel's ``src`` is the lower row, and the
    extent rule once asked only whether ``src`` reached past ``dst``: rows
    ``[a, b]`` wrote one edge and rows ``[b, a]`` wrote none."""
    a, b = _staggered_near_an_end()
    one = build([a, b]).edges.to_pylist()
    other = build([b, a]).edges.to_pylist()
    assert len(one) == len(other) == 1
    assert (one[0]["src_row"], one[0]["dst_row"]) == (0, 1)
    assert (other[0]["src_row"], other[0]["dst_row"]) == (1, 0)
    assert [one[0][k] for k in _MIRRORED] == [other[0][k] for k in _MIRRORED]
    e = one[0]
    assert (e["relation"], e["truncation"]) == ("contained", "5p")
    assert (e["src_overhang_3p"], e["dst_overhang_5p"], e["div_5p"]) == (28, 28, 6)
    assert (e["delta_5p"], e["delta_3p"], e["n_edits"]) == (28, -28, 1)


def test_a_contained_edge_runs_from_the_contained_member_even_if_longer():
    """The shorter reaches 25 nt past the longer at the 5' end, with a
    difference 10 nt inside the junction; the longer reaches 30 nt past at
    the 3' end, inside the tolerance. It is the longer that is cut short, so
    the edge runs from it, and ``src_len > dst_len``."""
    rng = random.Random(62)
    core = rnd(2000, rng)
    short = rnd(25, rng) + substituted(core, 10)
    long = core + rnd(30, rng)
    for seqs, src, dst in (([short, long], 1, 0), ([long, short], 0, 1)):
        edges = build(seqs).edges.to_pylist()
        assert len(edges) == 1
        e = edges[0]
        assert (e["src_row"], e["dst_row"]) == (src, dst)
        assert (e["relation"], e["truncation"]) == ("contained", "5p")
        assert (e["src_len"], e["dst_len"]) == (2030, 2025)
        assert (e["src_overhang_5p"], e["src_overhang_3p"]) == (0, 30)
        assert (e["dst_overhang_5p"], e["dst_overhang_3p"]) == (25, 0)
        assert (e["delta_5p"], e["delta_3p"], e["div_5p"]) == (25, -30, 11)
        assert e["src_template_id"] == ids_of(2)[src]
        assert not e["mergeable"]


def test_a_turned_edge_is_turned_for_every_copy_of_either_sequence():
    a, b = _staggered_near_an_end()
    res = build([b, a, b, a])
    assert res.stats["n_pairs_aligned"] == 1
    contained = {(s, d) for s, d, rel in pairs(res.edges) if rel == "contained"}
    assert contained == {(1, 0), (1, 2), (3, 0), (3, 2)}
    assert "src_is_container" not in res.edges.column_names


# ──────────────────────────────────────────────────────────────────────
# Rows that cannot be paired
# ──────────────────────────────────────────────────────────────────────


def test_rows_too_short_to_sketch_are_counted_and_make_no_edge():
    """37 = k + w - 1 is the shortest sequence holding one whole window."""
    rng = random.Random(48)
    a = rnd(800, rng)
    seqs = [a, a[100:136], a[100:136], "", a[200:237], "ACGT", a[:500]]
    assert [len(s) for s in seqs] == [800, 36, 36, 0, 37, 4, 500]
    res = build(seqs)

    assert res.stats["n_unpairable_short"] == 4
    assert res.stats["n_unique"] == 3
    assert res.stats["n_identical_classes"] == 0  # the two 36-mers are not a class
    touched = {r for s, d, _ in pairs(res.edges) for r in (s, d)}
    assert touched.isdisjoint({1, 2, 3, 5})
    assert (6, 0, "contained") in pairs(res.edges)
    assert (4, 0, "contained") in pairs(res.edges)  # 37 nt is long enough


def test_rows_that_are_all_too_short_make_an_empty_graph():
    res = build(["ACGT", "", "ACGT"])
    assert res.edges.num_rows == 0
    assert res.edges.schema.equals(TEMPLATE_EDGE_TABLE, check_metadata=True)
    assert res.stats["n_unpairable_short"] == 3
    assert res.stats["n_unique"] == res.stats["n_edges"] == 0
    assert res.stats["candidates"]["n_candidates"] == 0


def test_one_row_makes_no_edge():
    res = build([rnd(500, random.Random(49))])
    assert res.edges.num_rows == 0
    assert res.stats["n_unique"] == 1


# ──────────────────────────────────────────────────────────────────────
# How the sequences are held
# ──────────────────────────────────────────────────────────────────────


def _held_otherwise(seqs: list[str]) -> dict[str, pa.Array | pa.ChunkedArray]:
    n = len(seqs)
    padded = pa.array(["TTTTGGGG", *seqs, "CCCC"], type=pa.large_string())
    return {
        "string": pa.array(seqs, type=pa.string()),
        "chunked": pa.chunked_array([column(seqs[:3]), column(seqs[3:])]),
        "chunked string": pa.chunked_array(
            [pa.array(seqs[:1], pa.string()), pa.array(seqs[1:], pa.string())]
        ),
        "one chunk": pa.chunked_array([column(seqs)]),
        "empty chunks": pa.chunked_array([column([]), column(seqs), column([])]),
        "sliced": padded.slice(1, n),
        "sliced chunk": pa.chunked_array([padded]).slice(1, n),
        "sliced chunks": pa.chunked_array([padded.slice(1, 2), padded.slice(3, n - 2)]),
    }


@pytest.fixture(scope="module")
def handful():
    rng = random.Random(50)
    a = rnd(1300, rng)
    seqs = [a, a[150:], substituted(a, 640), a[:1000], "ACGT", a[150:], a[300:900]]
    return seqs, ids_of(len(seqs)), np.arange(len(seqs), dtype=np.int64) + 1


@pytest.mark.parametrize(
    "held",
    ["string", "chunked", "chunked string", "one chunk", "empty chunks"]
    + ["sliced", "sliced chunk", "sliced chunks"],
)
def test_the_edges_do_not_depend_on_how_the_sequences_are_held(handful, held):
    """A sliced array keeps its parent's buffers and says where it starts;
    read from the buffer's own start, row 0 is some other row's sequence."""
    seqs, ids, n_reads = handful
    want = build_graph(column(seqs), ids=ids, n_reads=n_reads, node_round=2)
    assert want.edges.num_rows >= 8
    other = _held_otherwise(seqs)[held]
    assert other.to_pylist() == seqs
    got = build_graph(other, ids=ids, n_reads=n_reads, node_round=2)
    assert got.edges.equals(want.edges)


@pytest.mark.parametrize(
    "held",
    ["string", "chunked", "chunked string", "one chunk", "empty chunks"]
    + ["sliced", "sliced chunk", "sliced chunks"],
)
def test_the_digest_is_of_the_rows_not_of_how_they_are_held(handful, held):
    seqs, ids, n_reads = handful
    origin = np.array([-1, 5, 5, -1, -1, 8, 8])
    want = input_digest(column(seqs), ids, n_reads, origin)
    assert input_digest(_held_otherwise(seqs)[held], ids, n_reads, origin) == want


def test_the_digest_changes_with_anything_the_edges_depend_on(handful):
    seqs, ids, n_reads = handful
    origin = np.array([-1, 5, 5, -1, -1, 8, 8])
    want = input_digest(column(seqs), ids, n_reads, origin)
    assert len(want) == 32 and int(want, 16) >= 0
    assert input_digest(column(seqs), ids.copy(), n_reads.copy(), origin.copy()) == want

    def bumped(values: np.ndarray, at: int) -> np.ndarray:
        out = values.copy()
        out[at] += 1
        return out

    one_base = [*seqs[:3], substituted(seqs[3], 500), *seqs[4:]]
    changed = {
        "n_reads": input_digest(column(seqs), ids, bumped(n_reads, 6), origin),
        "origin": input_digest(column(seqs), ids, n_reads, bumped(origin, 0)),
        "id": input_digest(column(seqs), bumped(ids, 2), n_reads, origin),
        "base": input_digest(column(one_base), ids, n_reads, origin),
        "order": input_digest(column(seqs[::-1]), ids, n_reads, origin),
        "no origin": input_digest(column(seqs), ids, n_reads, None),
    }
    assert want not in changed.values()
    assert len(set(changed.values())) == len(changed)


def test_the_digest_knows_where_one_sequence_ends():
    ids = n_reads = np.array([1, 2])
    a = input_digest(column(["ACGT", "TTT"]), ids, n_reads, None)
    b = input_digest(column(["ACG", "TTTT"]), ids, n_reads, None)
    assert a != b


def test_no_split_origins_and_none_known_are_the_same_input():
    ids = n_reads = np.array([1, 2])
    seqs = column(["ACGT", "TTT"])
    unknown = np.array([-1, -1])
    assert input_digest(seqs, ids, n_reads, None) == input_digest(
        seqs, ids, n_reads, unknown
    )


def test_the_stamp_is_the_kernel_the_parameters_and_the_input():
    stamp = graph_stamp(GraphParams(tol_5p=12, chunk_rows=5), "abc")
    assert stamp == {
        "kernel_version": GRAPH_KERNEL_VERSION,
        "graph": GraphParams(tol_5p=12).semantic(),
        "input": "abc",
    }
    assert stamp["graph"]["tol_5p"] == 12 and "chunk_rows" not in stamp["graph"]
    assert json.loads(json.dumps(stamp)) == stamp


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"ids": np.arange(3)}, ValueError),
        ({"n_reads": np.arange(5)}, ValueError),
        ({"split_origin": np.arange(2)}, ValueError),
        ({"n_reads": np.ones(4)}, TypeError),
        ({"ids": np.arange(8).reshape(4, 2)}, ValueError),
    ],
)
def test_a_column_that_is_not_one_integer_per_row_is_refused(kwargs, error):
    rng = random.Random(51)
    with pytest.raises(error):
        build([rnd(100, rng) for _ in range(4)], **kwargs)


def test_a_null_sequence_is_refused():
    with pytest.raises(ValueError, match="null"):
        build_graph(
            pa.array(["ACGT", None], pa.large_string()),
            ids=np.arange(2),
            n_reads=np.arange(2),
            node_round=1,
        )
    with pytest.raises(TypeError, match="string"):
        build_graph(
            pa.array([1, 2]), ids=np.arange(2), n_reads=np.arange(2), node_round=1
        )


# ──────────────────────────────────────────────────────────────────────
# On disk
# ──────────────────────────────────────────────────────────────────────


def test_streamed_edges_read_back_as_the_template_edge_table(handful, tmp_path):
    seqs, ids, n_reads = handful
    path = tmp_path / "r02" / "graph" / "edges.parquet"  # nothing of it exists
    res = build_graph(
        column(seqs), ids=ids, n_reads=n_reads, node_round=2, output_path=path
    )
    assert res.edges is None and res.path == path

    back = pq.read_table(path)
    assert back.schema.equals(TEMPLATE_EDGE_TABLE, check_metadata=True)
    assert back.schema.field("relation").type == _DICT
    assert back.schema.field("truncation").type == _DICT
    assert back.num_rows == res.stats["n_edges"] >= 8
    assert sorted(p.name for p in path.parent.iterdir()) == ["edges.parquet"]
    # The caller's filter, which is how merging reads the file.
    some = pq.read_table(path, filters=[("mergeable", "==", True)])
    assert some.num_rows == res.stats["n_mergeable"] > 0


def test_a_path_given_as_a_string_is_a_path(handful, tmp_path):
    seqs, ids, n_reads = handful
    res = build_graph(
        column(seqs),
        ids=ids,
        n_reads=n_reads,
        node_round=2,
        output_path=str(tmp_path / "edges.parquet"),
    )
    assert res.path == tmp_path / "edges.parquet" and res.path.exists()


@pytest.mark.parametrize("seqs", [[], ["ACGT"], ["ACGT" * 30, "TTGCA" * 30]])
def test_no_edges_are_written_as_a_valid_empty_file(seqs, tmp_path):
    path = tmp_path / "edges.parquet"
    res = build(seqs, output_path=path)
    back = pq.read_table(path)
    assert back.num_rows == res.stats["n_edges"] == 0
    assert back.schema.equals(TEMPLATE_EDGE_TABLE, check_metadata=True)
    assert [p.name for p in tmp_path.iterdir()] == ["edges.parquet"]
    assert build(seqs).edges.schema.equals(TEMPLATE_EDGE_TABLE, check_metadata=True)


@pytest.mark.parametrize("threads", [1, 3])
def test_a_failed_build_leaves_the_file_that_was_there(
    family, small_tasks, tmp_path, monkeypatch, threads
):
    """The edges are written beside their name and renamed onto it, so a
    reader never opens half a graph — and a failure renames nothing."""
    seqs, _, n_reads = family
    path = tmp_path / "edges.parquet"
    build(seqs, n_reads=n_reads, output_path=path)
    before = path.read_bytes()

    real, calls = graph._annotate, []

    def fails_late(cols, nodes):
        calls.append(1)
        if len(calls) == 6:  # one batch of twins and four tasks are in
            raise RuntimeError("disk full")
        return real(cols, nodes)

    monkeypatch.setattr(graph, "_annotate", fails_late)
    monkeypatch.setattr(graph, "_BATCH_ROWS", 20)  # so some of it was written
    with pytest.raises(RuntimeError, match="disk full"):
        build(seqs, n_reads=n_reads, output_path=path, threads=threads)
    assert len(calls) == 6
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["edges.parquet"]
    assert graph._KERNEL_STATE is None


def test_a_failed_build_leaves_no_file_where_there_was_none(
    family, tmp_path, monkeypatch
):
    seqs, _, n_reads = family

    def fails(cols, nodes):
        raise RuntimeError("disk full")

    monkeypatch.setattr(graph, "_annotate", fails)
    with pytest.raises(RuntimeError, match="disk full"):
        build(seqs, n_reads=n_reads, output_path=tmp_path / "edges.parquet")
    assert list(tmp_path.iterdir()) == []


def _pools(monkeypatch) -> list[int]:
    """The worker count of every pool the builder's kernel starts."""
    started: list[int] = []
    real = graph.ProcessPoolExecutor

    def spy(*args, **kwargs):
        started.append(kwargs["max_workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(graph, "ProcessPoolExecutor", spy)
    return started


def test_the_sequences_are_not_left_in_the_module(family, small_tasks, monkeypatch):
    """After a pool, that is — one task is measured in the parent, and a
    fixture that cuts one task says nothing about what a pool leaves."""
    seqs, _, n_reads = family
    started = _pools(monkeypatch)
    res = build(seqs, n_reads=n_reads, threads=3)
    assert started == [3]
    assert res.edges.num_rows > 0
    assert graph._KERNEL_STATE is None


_CALLS = mp.get_context("fork").Value("i", 0)
_REAL_RELATE = relate_pair


def _fails_on_the_fortieth_pair(src, dst, diag, params):
    with _CALLS.get_lock():
        _CALLS.value += 1
        if _CALLS.value == 40:
            raise RuntimeError("planted in a worker")
    return _REAL_RELATE(src, dst, diag, params)


def test_a_failure_inside_a_worker_is_the_build_s_failure(
    family, small_tasks, tmp_path, monkeypatch
):
    """Raised below the fork, in a process that is not this one: it has to
    come back as the exception it was, leave the file that was there, and
    leave nothing of its own."""
    seqs, _, n_reads = family
    path = tmp_path / "edges.parquet"
    build(seqs, n_reads=n_reads, output_path=path)
    before = path.read_bytes()

    started = _pools(monkeypatch)
    _CALLS.value = 0
    monkeypatch.setattr(graph, "relate_pair", _fails_on_the_fortieth_pair)
    with pytest.raises(RuntimeError, match="planted in a worker"):
        build(seqs, n_reads=n_reads, output_path=path, threads=3)
    assert started == [3], "the failure was in a worker, not in the parent"
    assert _CALLS.value >= 40
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["edges.parquet"]
    assert graph._KERNEL_STATE is None


def test_progress_is_told_at_each_stage(handful):
    seqs, ids, n_reads = handful
    said: list[str] = []
    build_graph(
        column(seqs), ids=ids, n_reads=n_reads, node_round=2, progress=said.append
    )
    assert len(said) == 3
    assert all(isinstance(line, str) and "\n" not in line for line in said)
    assert "7 sequences" in said[0] and "candidate pairs" in said[1]


# ──────────────────────────────────────────────────────────────────────
# The pool
# ──────────────────────────────────────────────────────────────────────


def _tasks_of(n: int, length: int) -> tuple[list[tuple], np.ndarray]:
    """The tasks cut from ``n`` candidate pairs among 1,000 sequences of
    ``length`` nt, and the pairs' src rows."""
    rng = np.random.default_rng(52)
    src = np.sort(rng.integers(0, 1000, n))
    tasks = graph._cut_tasks(
        src,
        rng.integers(0, 1000, n),
        rng.integers(-500, 500, n).astype(np.int32),
        rng.integers(2, 17, n).astype(np.int32),
        rng.integers(0, 2, n).astype(bool),
        np.full(1000, length, dtype=np.int64),
    )
    return list(tasks), src


def test_a_task_of_5000_pairs_pickles_small(monkeypatch):
    """A task is cut out of arrays a hundred times its size. numpy pickles a
    view as its own elements; an Arrow slice would carry its parent."""
    monkeypatch.setattr(graph, "_TASK_MAX_PAIRS", 5000)
    tasks, src = _tasks_of(500_000, 900)
    assert len(tasks) == 100
    assert all(len(part) == 5000 for task in tasks for part in task)
    assert max(len(pickle.dumps(task)) for task in tasks) < 150_000
    assert np.concatenate([task[0] for task in tasks]).tolist() == src.tolist()


def test_the_largest_task_there_can_be_pickles_small():
    tasks, _ = _tasks_of(100_000, 200)
    assert max(len(task[0]) for task in tasks) == graph._TASK_MAX_PAIRS
    assert max(len(pickle.dumps(task)) for task in tasks) < 500_000


def test_tasks_are_cut_by_cost_not_by_count():
    """A pair of 15 kb templates is ~100x a pair of 1.4 kb ones."""
    lengths = np.array([1400, 15_000], dtype=np.int64)
    n = 40_000
    src = np.repeat(np.array([0, 1]), n // 2)
    zeros = np.zeros(n, dtype=np.int32)
    tasks = list(graph._cut_tasks(src, src, zeros, zeros, zeros.astype(bool), lengths))
    sizes = [len(task[0]) for task in tasks]
    assert sum(sizes) == n
    short = [s for s, task in zip(sizes, tasks) if task[0][0] == 0 and task[0][-1] == 0]
    long = [s for s, task in zip(sizes, tasks) if task[0][0] == 1]
    assert max(long) <= graph._TASK_COST // (15_000 * 15_000) + 1 == 23
    assert min(short[:-1]) == 2551  # _TASK_COST // 1400^2
    assert max(sizes) <= graph._TASK_MAX_PAIRS


def test_a_worker_returns_only_the_edges_it_produced(monkeypatch):
    """Nothing at candidate cardinality goes back to the parent: a pair that
    is not an edge comes back as one more in a count of eight."""
    rng = random.Random(53)
    a = rnd(1300, rng)
    seqs = [a, a[150:], rnd(700, rng), a[:1000], a[300:900], rnd(700, rng)]
    raw = [s.encode() for s in seqs]
    buffer = np.frombuffer(b"".join(raw), dtype=np.uint8)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in raw])])
    monkeypatch.setattr(graph, "_KERNEL_STATE", (buffer, offsets, P))
    out, seen = graph._relate_task(
        np.array([1, 2, 3, 4, 5]),
        np.array([0, 0, 0, 0, 0]),
        np.array([150, 0, 0, 300, 40], dtype=np.int32),
        np.array([16, 2, 15, 14, 3], dtype=np.int32),
        np.array([False, False, True, False, True]),
    )
    assert seen.tolist() == [0, 3, 2, 0, 0, 0, 0, 0] and seen.dtype == np.int64
    assert out["src_row"].tolist() == [1, 3, 4]
    assert out["dst_row"].tolist() == [0, 0, 0]
    assert out["relation"].tolist() == [Relation.CONTAINED] * 3
    assert [TRUNCATION_NAMES[i] for i in out["truncation"]] == ["5p", "3p", "both"]
    assert out["dst_overhang_5p"].tolist() == [150, 0, 300]
    assert out["n_shared_probes"].tolist() == [16, 15, 14]
    assert out["candidate_overflow"].tolist() == [False, True, False]
    assert all(isinstance(v, np.ndarray) and v.shape == (3,) for v in out.values())
    assert len(pickle.dumps((out, seen))) < 5000


@pytest.mark.xfail(
    strict=True,
    reason="constellation.core.io.schemas imports torch at module level, and "
    "every import below constellation.sequencing passes through it",
)
def test_the_module_does_not_import_torch():
    """What the builder was asked for, and cannot deliver from this module
    alone. Strict: when the import above stops pulling torch in, this starts
    passing, and the marker comes off."""
    code = (
        "import sys\n"
        "import constellation.sequencing.transcriptome.cluster.denovo.em.graph\n"
        "assert 'torch' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_module_does_not_import_the_sketch():
    """torch is the sketch's, and the sketch is imported where it is called.

    This is the module's own part of "importing it does not import torch";
    the test above is the whole of it.
    """
    name = "constellation.sequencing.transcriptome.cluster.denovo"
    code = (
        "import sys\n"
        f"import {name}.em.graph\n"
        f"loaded = [m for m in ('minimizers', 'encode', 'candidates')"
        f" if '{name}.' + m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    tree = ast.parse(Path(graph.__file__).read_text())
    at_import = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            at_import.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            at_import.add(node.module)
    assert not {m for m in at_import if m.split(".")[0] == "torch"}
    assert not {m for m in at_import if m.endswith((".minimizers", ".candidates"))}


# ──────────────────────────────────────────────────────────────────────
# Merge predicate
# ──────────────────────────────────────────────────────────────────────

_BASE = {
    "node_round": 2,
    "src_template_id": 100,
    "dst_template_id": 200,
    "src_row": 0,
    "dst_row": 1,
    "relation": "equivalent",
    "truncation": None,
    "src_len": 1000,
    "dst_len": 1010,
    "src_overhang_5p": 0,
    "src_overhang_3p": 0,
    "dst_overhang_5p": 4,
    "dst_overhang_3p": 6,
    "delta_5p": 4,
    "delta_3p": 6,
    "aligned_len": 1000,
    "n_edits": 0,
    "core_len_delta": 0,
    "n_mismatch": 0,
    "n_insert": 0,
    "n_delete": 0,
    "identity": 1.0,
    "edit_distance_placed": 0,
    "div_5p": 0,
    "div_3p": 0,
    "n_shared_probes": 16,
    "candidate_overflow": False,
    "same_split_origin": False,
    "src_n_reads": 10,
    "dst_n_reads": 20,
    "mergeable": False,
}


def edges_of(*changes: dict, schema: pa.Schema = TEMPLATE_EDGE_TABLE) -> pa.Table:
    """One edge per dict: the mergeable base edge with ``changes`` applied."""
    names = set(schema.names)
    return pa.Table.from_pylist(
        [{k: v for k, v in (_BASE | c).items() if k in names} for c in changes],
        schema=schema,
    )


def verdict(change: dict, predicate: MergePredicate = MergePredicate(), **kw) -> bool:
    out = is_mergeable(edges_of(change), predicate, **kw)
    assert out.dtype == bool and out.shape == (1,)
    return bool(out[0])


def test_the_base_edge_is_mergeable():
    assert verdict({})
    assert verdict({"identity": 0.5})  # never read
    assert verdict({"mergeable": False}) and verdict({"mergeable": True})


def test_only_an_equivalent_edge_merges():
    assert not verdict({"relation": "contained", "truncation": "5p"})
    assert not verdict({"relation": "staggered"})


def test_an_edit_too_many_does_not_merge():
    assert not verdict({"n_edits": 1})
    assert verdict({"n_edits": 1}, MergePredicate(max_edits=1))
    assert not verdict({"n_edits": 2}, MergePredicate(max_edits=1))


@pytest.mark.parametrize("who", ["src", "dst"])
@pytest.mark.parametrize("end", ["5p", "3p"])
def test_each_overhang_is_held_to_its_own_ends_tolerance(who, end):
    other = "3p" if end == "5p" else "5p"
    over = f"{who}_overhang_{end}"
    tight = MergePredicate(**{f"tol_{end}": 10})
    assert verdict({over: 30})
    assert not verdict({over: 31})
    assert verdict({over: 10}, tight)
    assert not verdict({over: 11}, tight)
    # The other end's tolerance is not this end's.
    assert not verdict({over: 11}, MergePredicate(**{f"tol_{end}": 10}))
    assert verdict({over: 11}, MergePredicate(**{f"tol_{other}": 10}))


@pytest.mark.parametrize("who", ["src", "dst"])
@pytest.mark.parametrize("end", ["5p", "3p"])
def test_terminal_divergence_counts_toward_the_tolerance(who, end):
    over = f"{who}_overhang_{end}"
    assert verdict({over: 20, f"div_{end}": 10})
    assert not verdict({over: 20, f"div_{end}": 11})
    assert not verdict({over: 0, f"div_{end}": 31})
    other = "3p" if end == "5p" else "5p"
    assert verdict({over: 20, f"div_{other}": 11})


def test_the_less_supported_end_decides_the_depth_gate():
    assert verdict({}, MergePredicate(min_reads=10))
    assert not verdict({}, MergePredicate(min_reads=11))
    swapped = {"src_n_reads": 20, "dst_n_reads": 10}
    assert verdict(swapped, MergePredicate(min_reads=10))
    assert not verdict(swapped, MergePredicate(min_reads=11))


def test_split_kin_do_not_merge_unless_asked():
    assert not verdict({"same_split_origin": True})
    assert verdict({"same_split_origin": True}, MergePredicate(merge_siblings=True))


_IDENTICAL = {
    "same_split_origin": True,
    "dst_len": 1000,
    "dst_overhang_5p": 0,
    "dst_overhang_3p": 0,
}


def test_byte_identical_kin_merge_anyway():
    assert verdict(_IDENTICAL)


@pytest.mark.parametrize(
    "change",
    [
        {"n_edits": 1},
        {"dst_len": 1001},
        {"src_overhang_5p": 1},
        {"src_overhang_3p": 1},
        {"dst_overhang_5p": 1},
        {"dst_overhang_3p": 1},
    ],
)
def test_kin_that_differ_at_all_are_not_byte_identical(change):
    loose = MergePredicate(max_edits=5)
    assert not verdict(_IDENTICAL | change, loose)
    assert verdict(_IDENTICAL | change | {"same_split_origin": False}, loose)


def test_exact_only_overrides_the_edit_allowance():
    loose = MergePredicate(max_edits=2)
    assert verdict({"n_edits": 1}, loose)
    assert not verdict({"n_edits": 1}, loose, exact_only=True)
    assert verdict({"n_edits": 0}, loose, exact_only=True)


def test_every_edge_is_judged_alone():
    """So a file can be passed through one record batch at a time."""
    changes = [
        {},
        {"n_edits": 1},
        {"relation": "contained", "truncation": "both"},
        {"src_overhang_3p": 31},
        _IDENTICAL,
        {"same_split_origin": True},
    ]
    table = edges_of(*changes)
    want = [True, False, False, False, True, False]
    assert is_mergeable(table, MergePredicate()).tolist() == want
    batches = table.to_batches(max_chunksize=2)
    assert [b.num_rows for b in batches] == [2, 2, 2]
    for i, batch in enumerate(batches):
        one = pa.Table.from_batches([batch])
        assert is_mergeable(one, MergePredicate()).tolist() == want[2 * i : 2 * i + 2]
        assert is_mergeable(batch, MergePredicate()).tolist() == want[2 * i : 2 * i + 2]
    chunked = pa.concat_tables([table.slice(4), table.slice(0, 4)])
    assert chunked["relation"].num_chunks == 2
    assert is_mergeable(chunked, MergePredicate()).tolist() == want[4:] + want[:4]
    assert is_mergeable(table.slice(0, 0), MergePredicate()).tolist() == []


def test_the_predicate_reads_either_edge_table():
    table = edges_of({}, {"n_edits": 3}, schema=CLUSTER_EDGE_TABLE)
    assert is_mergeable(table, MergePredicate()).tolist() == [True, False]


def test_the_predicate_does_not_trust_the_dictionary_index():
    """A table that was re-encoded on its way back need not have kept the
    enum value in the index."""
    table = edges_of({"relation": "contained", "truncation": "3p"}, {})
    other = pa.DictionaryArray.from_arrays(
        pa.array([0, 1], pa.int8()), pa.array(["contained", "equivalent"])
    )
    table = table.set_column(
        table.schema.get_field_index("relation"),
        pa.field("relation", _DICT, nullable=False),
        other,
    )
    assert table["relation"].to_pylist() == ["contained", "equivalent"]
    assert is_mergeable(table, MergePredicate()).tolist() == [False, True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_edits": -1},
        {"max_edits": 1.0},
        {"max_edits": True},
        {"tol_5p": -1},
        {"tol_3p": float("inf")},
        {"tol_3p": None},
        {"min_reads": -5},
        {"min_reads": False},
        {"merge_siblings": 1},
        {"merge_siblings": None},
    ],
)
def test_a_predicate_that_cannot_mean_anything_is_refused(kwargs):
    with pytest.raises(ValueError):
        MergePredicate(**kwargs)


def test_the_default_predicate_is_exact_and_guarded():
    p = MergePredicate()
    assert (p.max_edits, p.tol_5p, p.tol_3p) == (0, 30, 30)
    assert (p.min_reads, p.merge_siblings) == (0, False)


def test_a_tolerance_of_nothing_is_a_tolerance():
    none = MergePredicate(tol_5p=0, tol_3p=0)
    flush = {"dst_overhang_5p": 0, "dst_overhang_3p": 0}
    assert verdict(flush, none)
    assert not verdict(flush | {"dst_overhang_5p": 1}, none)
    assert not verdict(flush | {"div_3p": 1}, none)


def test_the_stored_verdict_is_the_builds_predicate(family):
    seqs, _, n_reads = family
    origin = np.arange(len(seqs), dtype=np.int64) % 2  # kin: same parity
    for pred in (
        MergePredicate(),
        MergePredicate(max_edits=2, tol_5p=12, min_reads=20),
        MergePredicate(max_edits=1, merge_siblings=True),
    ):
        t = build(seqs, n_reads=n_reads, split_origin=origin, predicate=pred).edges
        stored = t.column("mergeable").to_numpy(zero_copy_only=False)
        assert stored.tolist() == is_mergeable(t, pred).tolist()
        assert 0 < stored.sum() < t.num_rows


def test_mergeable_pairs_are_the_mergeable_edges(family):
    seqs, _, n_reads = family
    pred = MergePredicate(max_edits=1)
    t = build(seqs, n_reads=n_reads, predicate=pred).edges
    keep = is_mergeable(t, pred)
    got = mergeable_pairs(t, pred)

    assert list(got) == [
        "src_template_id",
        "dst_template_id",
        "src_row",
        "dst_row",
        "off_5p",
        "off_3p",
        "identical",
    ]
    assert all(v.shape == (int(keep.sum()),) for v in got.values())
    assert all(got[k].dtype == np.int64 for k in list(got)[:6])
    assert got["identical"].dtype == bool
    kept = t.filter(pa.array(keep)).to_pydict()
    assert got["src_row"].tolist() == kept["src_row"]
    assert got["dst_row"].tolist() == kept["dst_row"]
    assert got["src_template_id"].tolist() == kept["src_template_id"]
    assert got["dst_template_id"].tolist() == kept["dst_template_id"]
    assert got["off_5p"].tolist() == kept["delta_5p"]
    assert got["off_3p"].tolist() == kept["delta_3p"]
    assert set(kept["relation"]) == {"equivalent"}
    assert any(got["off_5p"] < 0) and any(got["off_5p"] > 0)
    twins = [
        e == 0 and a == b and d5 == 0 and d3 == 0
        for e, a, b, d5, d3 in zip(
            kept["n_edits"],
            kept["src_len"],
            kept["dst_len"],
            kept["delta_5p"],
            kept["delta_3p"],
        )
    ]
    assert got["identical"].tolist() == twins
    assert 0 < sum(twins) < len(twins)

    exact = mergeable_pairs(t, pred, exact_only=True)
    assert 0 < exact["src_row"].shape[0] < got["src_row"].shape[0]
    assert exact["src_row"].shape[0] == int(is_mergeable(t, MergePredicate()).sum())
    none = mergeable_pairs(t.slice(0, 0), pred)
    assert [v.shape for v in none.values()] == [(0,)] * 7


# ──────────────────────────────────────────────────────────────────────
# Split origin
# ──────────────────────────────────────────────────────────────────────


def tid(round_index: int, row: int) -> int:
    return (round_index << 40) | row


def write_lineage(rounds_dir: Path, r: int, lines: list[tuple[int, int, str]]):
    """``rounds_dir/rNN/lineage.parquet`` from ``(child, parent, rule)``.
    Round ``r``'s lineage is stamped ``r + 1``: its children are that
    round's templates."""
    table = pa.table(
        {
            "round": pa.array([r + 1] * len(lines), pa.int32()),
            "child_template_id": pa.array([c for c, _, _ in lines], pa.int64()),
            "parent_template_id": pa.array([p for _, p, _ in lines], pa.int64()),
            "rule": pa.array([rule for _, _, rule in lines], pa.string()),
            "n_reads": pa.array([1.0] * len(lines), pa.float64()),
        }
    )
    path = rounds_dir / f"r{r:02d}" / "lineage.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


X, Y = tid(1, 0), tid(1, 1)
A, B, C = tid(2, 0), tid(2, 1), tid(2, 2)


def test_nodes_of_one_parent_are_split_siblings(tmp_path):
    parents = np.array([tid(3, 5), tid(3, 9), tid(3, 5), tid(3, 5)])
    assert split_origins(tmp_path, 3, parents).tolist() == [
        tid(3, 5),
        -1,
        tid(3, 5),
        tid(3, 5),
    ]


def test_a_split_is_remembered_through_a_carry(tmp_path):
    """Guarding this round's siblings alone turns the period-1 split/merge
    cycle into a period-2 one."""
    write_lineage(tmp_path, 1, [(A, X, "split"), (B, X, "split"), (C, Y, "carry")])
    a, b, c = tid(3, 0), tid(3, 1), tid(3, 2)
    write_lineage(tmp_path, 2, [(a, A, "carry"), (b, B, "carry"), (c, C, "carry")])

    origin = split_origins(tmp_path, 3, np.array([b, c, a]))
    assert origin.tolist() == [X, -1, X]
    assert origin.dtype == np.int64
    assert origin[0] == origin[2] >= 0  # kin


def test_a_line_that_split_again_takes_the_newer_origin(tmp_path):
    write_lineage(tmp_path, 1, [(A, X, "split"), (B, X, "split")])
    a1, a2, b1 = tid(3, 0), tid(3, 1), tid(3, 2)
    write_lineage(tmp_path, 2, [(a1, A, "split"), (a2, A, "split"), (b1, B, "carry")])
    assert split_origins(tmp_path, 3, np.array([a1, a2, b1])).tolist() == [A, A, X]


def test_a_split_this_round_is_newer_than_any_before_it(tmp_path):
    write_lineage(tmp_path, 1, [(A, X, "split"), (B, X, "split")])
    assert split_origins(tmp_path, 2, np.array([A, A, B])).tolist() == [A, A, X]


def test_a_seed_has_no_origin(tmp_path):
    write_lineage(tmp_path, 1, [(A, X, "carry")])
    write_lineage(tmp_path, 2, [(tid(3, 0), A, "carry")])
    # Carried all the way back to a round-1 seed, and one in no lineage.
    assert split_origins(tmp_path, 3, np.array([tid(3, 0), tid(3, 7)])).tolist() == [
        -1,
        -1,
    ]
    assert split_origins(tmp_path, 1, np.array([X, Y])).tolist() == [-1, -1]
    assert split_origins(tmp_path, 3, np.array([], dtype=np.int64)).tolist() == []


def test_a_missing_lineage_file_ends_the_walk(tmp_path):
    a, b = tid(3, 0), tid(3, 1)
    write_lineage(tmp_path, 2, [(a, A, "carry"), (b, B, "split")])
    assert not (tmp_path / "r01").exists()
    assert split_origins(tmp_path, 3, np.array([a, b])).tolist() == [-1, B]
    assert split_origins(tmp_path / "nowhere", 3, np.array([a, b])).tolist() == [-1, -1]


def test_a_lineage_file_that_does_not_open_ends_the_walk(tmp_path):
    a = tid(3, 0)
    write_lineage(tmp_path, 1, [(A, X, "split")])
    write_lineage(tmp_path, 2, [(a, A, "carry")])
    assert split_origins(tmp_path, 3, np.array([a])).tolist() == [X]
    (tmp_path / "r02" / "lineage.parquet").write_bytes(b"PAR1 half a footer")
    assert split_origins(tmp_path, 3, np.array([a])).tolist() == [-1]


def test_a_survivor_inherits_the_split_of_what_it_absorbed(tmp_path):
    """The hub, one round on. ``h`` came of its own parent and absorbed ``a``,
    one of two children split from X; the other child ``b`` was refused.
    Walking only the survivor's own line forgets ``a``, and the next round
    joins ``b`` to it: the split undone through the hub, in two rounds."""
    own = tid(1, 8)
    h, b = A, B
    write_lineage(
        tmp_path,
        1,
        [(h, own, "carry"), (h, X, "merge"), (b, X, "split")],
    )
    origin = split_origins(tmp_path, 2, np.array([h, b]))
    assert origin.tolist() == [X, X]

    # And through a carry, a round later still.
    h3, b3 = tid(3, 0), tid(3, 1)
    write_lineage(tmp_path, 2, [(h3, h, "carry"), (b3, b, "carry")])
    assert split_origins(tmp_path, 3, np.array([b3, h3])).tolist() == [X, X]


def test_an_absorbed_only_child_passes_on_its_ancestry(tmp_path):
    """A merge row keeps the absorbed node's parent and loses its rule. A
    parent with one row emitted one node: that row was a carry, and the line
    goes on through it to the split before."""
    p, q = tid(2, 0), tid(2, 1)
    write_lineage(tmp_path, 1, [(p, X, "split"), (q, X, "split")])
    h, own, c = tid(3, 0), tid(2, 5), tid(3, 1)
    # h absorbed p's only node; q carried.
    write_lineage(tmp_path, 2, [(h, own, "carry"), (h, p, "merge"), (c, q, "carry")])
    assert split_origins(tmp_path, 3, np.array([h, c])).tolist() == [X, X]


def test_a_parent_whose_nodes_were_all_absorbed_still_split(tmp_path):
    """Both children of X went to different survivors. Neither row says
    ``split`` any more; that X has two rows does."""
    own_a, own_b = tid(1, 8), tid(1, 9)
    write_lineage(
        tmp_path,
        1,
        [
            (A, own_a, "carry"),
            (B, own_b, "carry"),
            (A, X, "merge"),
            (B, X, "merge"),
            (-1, tid(1, 7), "unrecruited"),
            (C, Y, "carry"),
        ],
    )
    assert split_origins(tmp_path, 2, np.array([A, B, C])).tolist() == [X, X, -1]


def test_origins_that_meet_in_one_survivor_are_one_class(tmp_path):
    """``h`` is a child of Y's split and absorbed a child of X's. One id per
    node is what the grouping takes, so the two origins become one class,
    named by the smaller — and the sibling on each side is kin of it."""
    h, g, b = A, B, C
    write_lineage(
        tmp_path,
        1,
        [(h, Y, "split"), (g, Y, "split"), (h, X, "merge"), (b, X, "split")],
    )
    lone = tid(2, 9)
    write_lineage(tmp_path, 1, [*_lines(tmp_path, 1), (lone, tid(1, 5), "carry")])
    origin = split_origins(tmp_path, 2, np.array([h, g, b, lone]))
    assert origin.tolist() == [min(X, Y)] * 3 + [-1]


def test_without_a_merge_the_class_is_the_origin_itself(tmp_path):
    write_lineage(
        tmp_path,
        1,
        [(A, X, "split"), (B, X, "split"), (C, Y, "split"), (tid(2, 3), Y, "split")],
    )
    parents = np.array([A, B, C, tid(2, 3)])
    assert split_origins(tmp_path, 2, parents).tolist() == [X, X, Y, Y]


def _lines(rounds_dir: Path, r: int) -> list[tuple[int, int, str]]:
    t = pq.read_table(rounds_dir / f"r{r:02d}" / "lineage.parquet")
    return list(
        zip(
            t.column("child_template_id").to_pylist(),
            t.column("parent_template_id").to_pylist(),
            t.column("rule").to_pylist(),
        )
    )


def _origins_line_by_line(
    node_round: int, parents: list[int], files: dict
) -> list[int]:
    """`split_origins`, restated one line at a time in plain Python: every
    line of every node, then the classes by union-find."""
    from collections import Counter

    shared = Counter(parents)
    found: list[set[int]] = []
    for p in parents:
        if shared[p] > 1:
            found.append({p})
            continue
        mine: set[int] = set()
        frontier = {p}
        for r in range(node_round - 1, 0, -1):
            rows = [x for x in files.get(r, ()) if x[2] in ("carry", "split", "merge")]
            if not rows or not frontier:
                break
            emitted = Counter(parent for _, parent, _ in rows)
            reached: set[int] = set()
            for child, parent, rule in rows:
                if child not in frontier:
                    continue
                if rule == "split" or (rule == "merge" and emitted[parent] > 1):
                    mine.add(parent)
                else:
                    reached.add(parent)
            frontier = reached
        found.append(mine)

    leader: dict[int, int] = {}

    def find(x: int) -> int:
        while leader.setdefault(x, x) != x:
            x = leader[x]
        return x

    for mine in found:
        first, *rest = sorted(mine) or [None]
        for other in rest:
            leader[find(other)] = find(first)
    members: dict[int, list[int]] = {}
    for origin in {o for mine in found for o in mine}:
        members.setdefault(find(origin), []).append(origin)
    return [min(members[find(min(mine))]) if mine else -1 for mine in found]


def test_the_walk_agrees_with_a_line_by_line_reference(tmp_path):
    """Random lineages of 2-6 rounds with splits, carries, merges of one or
    several nodes into a survivor, and rounds whose file is missing."""
    rng = np.random.default_rng(55)
    n_merged = n_classes = n_kin = 0
    for trial in range(400):
        rounds_dir = tmp_path / f"t{trial}"
        n_rounds = int(rng.integers(2, 7))
        width = int(rng.integers(3, 12))
        files = {}
        for r in range(1, n_rounds):
            came_from = rng.integers(0, width, width)
            emitted = np.bincount(came_from, minlength=width)
            rows = [
                (
                    tid(r + 1, c),
                    tid(r, int(came_from[c])),
                    "split" if emitted[came_from[c]] > 1 else "carry",
                )
                for c in range(width)
            ]
            if rng.random() < 0.6:
                for _ in range(int(rng.integers(1, 4))):
                    i, j = (int(x) for x in rng.integers(0, width, 2))
                    # j survives: it must still stand on a row of its own.
                    if rows[i][0] != rows[j][0] and rows[j][2] != "merge":
                        rows[i] = (rows[j][0], rows[i][1], "merge")
            rng.shuffle(rows)
            if rng.random() < 0.9:
                files[r] = [tuple(row) for row in rows]
                write_lineage(rounds_dir, r, files[r])
        parents = [
            tid(n_rounds, int(x))
            for x in rng.integers(0, width, int(rng.integers(1, 10)))
        ]
        got = split_origins(rounds_dir, n_rounds, np.array(parents)).tolist()
        want = _origins_line_by_line(n_rounds, parents, files)
        assert got == want, trial
        n_merged += any(row[2] == "merge" for rows in files.values() for row in rows)
        old = _origins_without_merges(n_rounds, parents, files)
        n_classes += got != old
        n_kin += len(set(got) - {-1}) < len([g for g in got if g >= 0])
    assert n_merged >= 150
    assert n_classes >= 40, "the merge rows changed no answer: nothing was tested"
    assert n_kin >= 150


def _origins_without_merges(node_round, parents, files) -> list[int]:
    """What the walk answered when it followed a node's own line alone."""
    own = {r: [row for row in rows if row[2] != "merge"] for r, rows in files.items()}
    return _origins_line_by_line(node_round, parents, own)


def test_each_line_is_walked_in_one_pass_per_round(tmp_path, monkeypatch):
    rng = random.Random(54)
    n = 3000
    first = [(tid(2, i), tid(1, i // 3), "split") for i in range(n)]
    second = [(tid(3, i), tid(2, i), "carry") for i in range(n)]
    rng.shuffle(first)
    rng.shuffle(second)
    write_lineage(tmp_path, 1, first)
    write_lineage(tmp_path, 2, second)
    opened = []
    real = graph._read_lineage
    monkeypatch.setattr(
        graph, "_read_lineage", lambda path: opened.append(path) or real(path)
    )
    parents = np.array([tid(3, i) for i in range(n)])
    origin = split_origins(tmp_path, 3, parents)
    assert origin.tolist() == [tid(1, i // 3) for i in range(n)]
    assert [p.parent.name for p in opened] == ["r02", "r01"]


# ──────────────────────────────────────────────────────────────────────
# Cluster edges
# ──────────────────────────────────────────────────────────────────────


def test_cluster_edges_are_the_edges_between_survivors():
    table = edges_of(
        {"src_row": 0, "dst_row": 1},
        {"src_row": 2, "dst_row": 5, "relation": "contained", "truncation": "3p"},
        {"src_row": 5, "dst_row": 7, "n_edits": 2, "identity": 0.998},
        {"src_row": 3, "dst_row": 2},  # 3 was absorbed
        {"src_row": 7, "dst_row": 4},  # 4 was absorbed
        {"src_row": 3, "dst_row": 4},  # both were
        {"src_row": 9, "dst_row": 2, "same_split_origin": True, "mergeable": True},
    )
    keep = np.array([0, 1, 2, 5, 7, 9])
    lengths = np.array([100, 110, 120, 130, 140, 150])
    reads = np.array([1, 2, 3, 4, 5, 6])
    out = cluster_edges(table, keep, cluster_len=lengths, cluster_n_reads=reads)

    assert out.schema.equals(CLUSTER_EDGE_TABLE, check_metadata=True)
    got = out.to_pylist()
    assert [(r["src_cluster_id"], r["dst_cluster_id"]) for r in got] == [
        (0, 1),
        (2, 3),
        (3, 4),
        (5, 2),
    ]
    assert [(r["src_len"], r["dst_len"]) for r in got] == [
        (100, 110),
        (120, 130),
        (130, 140),
        (150, 120),
    ]
    assert [(r["src_n_reads"], r["dst_n_reads"]) for r in got] == [
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 3),
    ]
    assert [r["relation"] for r in got] == ["equivalent", "contained"] + [
        "equivalent"
    ] * 2
    assert [r["truncation"] for r in got] == [None, "3p", None, None]
    assert [r["n_edits"] for r in got] == [0, 0, 2, 0]
    assert [r["identity"] for r in got] == [1.0, 1.0, 0.998, 1.0]
    assert [r["same_split_origin"] for r in got] == [False, False, False, True]
    assert [r["mergeable"] for r in got] == [False, False, False, True]
    assert [r["dst_overhang_3p"] for r in got] == [6, 6, 6, 6]
    for chunk in out["relation"].chunks:
        assert chunk.dictionary.to_pylist() == [RELATION_NAMES[i] for i in range(8)]


def test_cluster_edges_of_a_built_graph(family):
    """Ids below the cluster count, no cluster against itself, no pair
    twice — and exactly the edges whose two rows both survived."""
    seqs, _, n_reads = family
    t = build(seqs, n_reads=n_reads).edges
    n = len(seqs)
    keep = np.array([i for i in range(n) if i % 4 != 1])
    lengths = np.array([len(seqs[i]) for i in keep])
    out = cluster_edges(t, keep, cluster_len=lengths, cluster_n_reads=n_reads[keep])

    src = np.array(t["src_row"].to_pylist())
    dst = np.array(t["dst_row"].to_pylist())
    survives = np.isin(src, keep) & np.isin(dst, keep)
    assert 0 < out.num_rows == int(survives.sum()) < t.num_rows
    a, b = out["src_cluster_id"].to_pylist(), out["dst_cluster_id"].to_pylist()
    assert max(a + b) < len(keep) and min(a + b) >= 0
    assert all(x != y for x, y in zip(a, b))
    assert len(set(zip(a, b))) == out.num_rows
    assert [int(keep[x]) for x in a] == src[survives].tolist()
    assert [int(keep[x]) for x in b] == dst[survives].tolist()
    # The lengths and read counts handed in are the rows' own, so every
    # feature column is the template edge's, unchanged.
    kept = t.filter(pa.array(survives))
    for name in (f.name for f in EDGE_FEATURE_FIELDS):
        assert out[name].to_pylist() == kept[name].to_pylist(), name


def test_cluster_edges_can_be_made_a_batch_at_a_time(family):
    seqs, _, n_reads = family
    t = build(seqs, n_reads=n_reads).edges
    keep = np.arange(0, len(seqs), 2)
    about = dict(
        cluster_len=np.array([len(seqs[i]) for i in keep]),
        cluster_n_reads=n_reads[keep],
    )
    whole = cluster_edges(t, keep, **about)
    parts = [
        cluster_edges(pa.Table.from_batches([batch]), keep, **about)
        for batch in t.to_batches(max_chunksize=17)
    ]
    assert len(parts) > 5
    assert pa.concat_tables(parts).combine_chunks().equals(whole)


def test_cluster_edges_of_nothing_are_an_empty_table():
    table = edges_of({}, {"src_row": 3, "dst_row": 4})
    none = np.array([], dtype=np.int64)
    for out in (
        cluster_edges(table, none, cluster_len=none, cluster_n_reads=none),
        cluster_edges(
            table.slice(0, 0),
            np.array([0, 1]),
            cluster_len=np.array([5, 5]),
            cluster_n_reads=np.array([1, 1]),
        ),
        cluster_edges(
            table,
            np.array([8, 9]),
            cluster_len=np.array([5, 5]),
            cluster_n_reads=np.array([1, 1]),
        ),
    ):
        assert out.num_rows == 0
        assert out.schema.equals(CLUSTER_EDGE_TABLE, check_metadata=True)


@pytest.mark.parametrize(
    "keep, lengths, reads",
    [
        ([1, 0], [5, 5], [1, 1]),
        ([0, 0], [5, 5], [1, 1]),
        ([0, 1], [5], [1, 1]),
        ([0, 1], [5, 5], [1, 1, 1]),
    ],
)
def test_survivors_that_are_not_a_sorted_set_are_refused(keep, lengths, reads):
    with pytest.raises(ValueError):
        cluster_edges(
            edges_of({}),
            np.array(keep),
            cluster_len=np.array(lengths),
            cluster_n_reads=np.array(reads),
        )
