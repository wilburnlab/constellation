"""The native E-step: minimizer-join candidates, no minimap2 anywhere.

The containment join's behaviours are pinned again HERE, bipartite: an
antisense pair yields no candidate but a tandem-repeat fragment does, the
caps say when they bind, and the result is independent of how the work is
cut — because this join is what replaces minimap2's candidate generation,
whose minimizer masking is the loss this path exists to avoid.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

edlib = pytest.importorskip("edlib")

from constellation.sequencing.transcriptome.cluster.denovo._io import (  # noqa: E402
    _READS_SCHEMA,
)
from constellation.sequencing.transcriptome.cluster.denovo.em import (  # noqa: E402
    native as nv,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (  # noqa: E402
    EM_ASSIGNMENT_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (  # noqa: E402
    ReadStore,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (  # noqa: E402
    TEMPLATE_TABLE,
    TemplateStore,
    write_templates,
)

_COMP = str.maketrans("ACGT", "TGCA")


def _rnd(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutated(rng, s, rate=0.01):
    out = []
    for ch in s:
        r = rng.random()
        if r < rate * 0.6:
            out.append(rng.choice([b for b in "ACGT" if b != ch]))
        elif r < rate * 0.8:
            continue
        elif r < rate:
            out.extend([ch, rng.choice("ACGT")])
        else:
            out.append(ch)
    return "".join(out)


def _write_corpus(tmp_path, rows, *, batches=1):
    """rows = [(read_id, sequence, dorado_quality | None)]"""
    tbl = pa.table(
        {
            "read_id": pa.array([r[0] for r in rows], pa.string()),
            "sequence": pa.array([r[1] for r in rows], pa.large_string()),
            "sample_id": pa.array(np.zeros(len(rows), np.int64)),
            "dorado_quality": pa.array([r[2] for r in rows], pa.float32()),
        },
        schema=_READS_SCHEMA,
    )
    path = tmp_path / "reads.arrow"
    step = max(1, len(rows) // batches)
    with pa.OSFile(str(path), "wb") as s, pa.ipc.new_file(s, _READS_SCHEMA) as w:
        for lo in range(0, len(rows), step):
            w.write_table(tbl.slice(lo, step))
    return path


def _write_templates(tmp_path, seqs, *, weight=None):
    n = len(seqs)
    weight = weight if weight is not None else [10.0] * n
    write_templates(
        pa.table(
            {
                "template_id": pa.array(np.arange(n, dtype=np.int64)),
                "sequence": pa.array(seqs, pa.large_string()),
                "orf_start": pa.array(np.zeros(n, np.int32)),
                "orf_end": pa.array(np.zeros(n, np.int32)),
                "orf_aa_length": pa.array(np.zeros(n, np.int32)),
                "node_weight": pa.array([float(x) for x in weight]),
                "orf_replication": pa.array([int(x) for x in weight], pa.int64()),
                "seed_read_quality": pa.array(np.full(n, 25.0, np.float32)),
                "seed_read_row": pa.array(np.zeros(n, np.int32)),
                "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            },
            schema=TEMPLATE_TABLE,
        ),
        tmp_path / "tpl",
    )
    return tmp_path / "tpl" / "templates.arrow"


def _index(store, params=None):
    return nv.TemplateMinimizerIndex.build(
        store, kmer=15, window=10, params=params or nv.NativeParams()
    )


def _join(tmp_path, corpus, t_path, params=None):
    minis = nv.write_read_minimizers(corpus, tmp_path / "minis", kmer=15, window=10)
    ms = nv.ReadMinimizerStore.open(minis)
    reads = ReadStore.open(corpus)
    try:
        h, p, offs = ms.block(0, ms.n_reads)
        store = TemplateStore.open(t_path)
        try:
            return nv.candidates_block(
                h,
                p,
                offs,
                reads.lengths().astype(np.int64),
                _index(store, params),
                params or nv.NativeParams(),
                kmer=15,
            )
        finally:
            store.close()
    finally:
        reads.close()
        ms.close()


def _pairs(found):
    return list(
        zip(
            found.local.tolist(),
            found.template_row.tolist(),
            found.n_shared.tolist(),
        )
    )


# ── the read minimizer store ──────────────────────────────────────────


def test_the_store_is_built_once_and_rebuilt_when_its_key_changes(tmp_path):
    rng = random.Random(1)
    corpus = _write_corpus(
        tmp_path, [(f"r{i}", _rnd(rng, 300), 30.0) for i in range(20)], batches=3
    )
    out = nv.write_read_minimizers(corpus, tmp_path / "m", kmer=15, window=10)
    stamp = json.loads((out / nv.MINIS_META).read_text())
    assert stamp["kmer"] == 15 and stamp["n_reads"] == 20
    before = (out / nv.MINIS_ARROW).stat().st_mtime_ns
    again = nv.write_read_minimizers(corpus, tmp_path / "m", kmer=15, window=10)
    assert (again / nv.MINIS_ARROW).stat().st_mtime_ns == before, "reused"
    rebuilt = nv.write_read_minimizers(corpus, tmp_path / "m", kmer=17, window=10)
    assert json.loads((rebuilt / nv.MINIS_META).read_text())["kmer"] == 17
    assert (rebuilt / nv.MINIS_ARROW).stat().st_mtime_ns != before


def test_a_block_slice_is_the_rows_of_the_whole(tmp_path):
    rng = random.Random(2)
    corpus = _write_corpus(
        tmp_path,
        [(f"r{i}", _rnd(rng, 100 + 40 * (i % 5)), 30.0) for i in range(23)],
        batches=4,
    )
    out = nv.write_read_minimizers(corpus, tmp_path / "m", kmer=15, window=10)
    store = nv.ReadMinimizerStore.open(out)
    try:
        h, p, offs = store.block(0, store.n_reads)
        assert offs[-1] == h.shape[0] == p.shape[0]
        for lo, hi in ((0, 5), (5, 23), (13, 14), (7, 7)):
            h1, p1, o1 = store.block(lo, hi)
            assert (h1 == h[offs[lo] : offs[hi]]).all()
            assert (p1 == p[offs[lo] : offs[hi]]).all()
            assert (o1 == offs[lo : hi + 1] - offs[lo]).all()
        # Within a read, positions ascend: the stratum walk depends on it.
        for r in range(store.n_reads):
            seg = p[offs[r] : offs[r + 1]]
            assert (np.diff(seg) > 0).all()
    finally:
        store.close()


# ── the join ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("panel")
    rng = random.Random(3)
    truths = [_rnd(rng, 600) for _ in range(3)]
    rows = []
    for t_i, t in enumerate(truths):
        rows += [(f"g{t_i}_r{i}", _mutated(rng, t), 30.0) for i in range(12)]
    rows.append(("junk", _rnd(rng, 600), 30.0))
    rows.append(("anti", truths[0].translate(_COMP)[::-1], 30.0))
    corpus = _write_corpus(tmp_path, rows, batches=4)
    t_path = _write_templates(tmp_path, truths, weight=[30, 20, 10])
    return tmp_path, corpus, t_path, truths


def test_every_read_finds_its_template_and_only_sense_pairs_count(panel):
    tmp_path, corpus, t_path, truths = panel
    found = _join(tmp_path, corpus, t_path)
    best: dict[int, tuple[int, int]] = {}
    for loc, row, ns in _pairs(found):
        if ns > best.get(loc, (-1, -1))[1]:
            best[loc] = (row, ns)
    for g in range(3):
        for i in range(12):
            assert best[g * 12 + i][0] == g, (g, i)
    assert 36 not in best, "random sequence has no candidate"
    assert 37 not in best, "the reverse complement is rejected as antisense"
    assert found.n_antisense >= 1
    assert not found.cap_hit.any() and not found.overflow.any()


def test_the_join_is_invariant_to_blocks_and_subchunks(panel, monkeypatch):
    tmp_path, corpus, t_path, _ = panel
    whole = _pairs(_join(tmp_path, corpus, t_path))
    monkeypatch.setattr(nv, "_SUBCHUNK_ROWS", 5)
    tiny = _join(tmp_path, corpus, t_path)
    assert _pairs(tiny) == whole
    monkeypatch.undo()

    minis = nv.ReadMinimizerStore.open(tmp_path / "minis")
    reads = ReadStore.open(corpus)
    store = TemplateStore.open(t_path)
    try:
        rl = reads.lengths().astype(np.int64)
        idx = _index(store)
        parts = []
        for lo, hi in ((0, 17), (17, 38)):
            h, p, offs = minis.block(lo, hi)
            f = nv.candidates_block(
                h, p, offs, rl[lo:hi], idx, nv.NativeParams(), kmer=15
            )
            parts += [(loc + lo, row, ns) for loc, row, ns in _pairs(f)]
        assert parts == whole
    finally:
        minis.close()
        reads.close()
        store.close()


def test_a_fragment_of_a_tandem_repeat_is_not_called_antisense(tmp_path):
    """The two-clause rule, bipartite: probes agreeing on ONE diagonal are a
    candidate however far the window spreads along the diagonal axis."""
    rng = random.Random(7)
    hits = 0
    for trial in range(12):
        unit = _rnd(rng, 25)
        template = _rnd(rng, 300) + unit * 3 + _rnd(rng, 300)
        start = 300 + rng.randrange(0, 50)
        read = template[start : start + 160]
        d = tmp_path / f"t{trial}"
        d.mkdir()
        corpus = _write_corpus(d, [("frag", read, 30.0)])
        t_path = _write_templates(d, [template])
        found = _join(d, corpus, t_path)
        hits += _pairs(found) == [(0, 0, found.n_shared[0])] or bool(
            found.local.shape[0]
        )
        assert found.local.shape[0] == 1, trial
    assert hits == 12


def test_the_row_budget_keeps_two_probes_and_flags_the_read(tmp_path):
    """A family bigger than the budget still yields candidates — min_shared
    needs two probes, so two always survive — and the read is flagged."""
    rng = random.Random(8)
    truth = _rnd(rng, 400)
    family = [_mutated(rng, truth, 0.004) for _ in range(40)]
    d = tmp_path
    corpus = _write_corpus(d, [("deep", _mutated(rng, truth), 30.0)])
    t_path = _write_templates(d, family)
    tight = nv.NativeParams(max_rows_per_read=90)
    found = _join(d, corpus, t_path, tight)
    assert found.cap_hit[0], "the budget bound"
    assert found.local.shape[0] >= 1, "…and candidates still exist"
    loose = _join(d, corpus, t_path)
    assert not loose.cap_hit[0]
    assert loose.local.shape[0] > found.local.shape[0]


def test_the_candidate_cap_keeps_the_most_shared_and_flags_the_read(tmp_path):
    rng = random.Random(9)
    truth = _rnd(rng, 400)
    family = [truth] + [_mutated(rng, truth, 0.02) for _ in range(9)]
    corpus = _write_corpus(tmp_path, [("r", _mutated(rng, truth, 0.002), 30.0)])
    t_path = _write_templates(tmp_path, family)
    capped = _join(tmp_path, corpus, t_path, nv.NativeParams(max_candidates=3))
    assert capped.local.shape[0] == 3 and capped.cap_hit[0]
    assert 0 in capped.template_row.tolist(), "the exact template survives the cut"
    full = _join(tmp_path, corpus, t_path)
    assert full.local.shape[0] == 10 and not full.cap_hit[0]


def test_an_oversized_bucket_joins_through_its_best_supported_anchors(tmp_path):
    rng = random.Random(10)
    truth = _rnd(rng, 400)
    family = [_mutated(rng, truth, 0.003) for _ in range(8)]
    weight = [1.0] * 8
    weight[5] = 50.0
    corpus = _write_corpus(tmp_path, [("r", _mutated(rng, truth, 0.002), 30.0)])
    t_path = _write_templates(tmp_path, family, weight=weight)
    small = nv.NativeParams(bucket_cap=4, overflow_anchors=2)
    found = _join(tmp_path, corpus, t_path, small)
    assert found.overflow[0], "every probe joined through anchors"
    assert 5 in found.template_row.tolist(), "the best-supported anchor is there"
    assert found.local.shape[0] <= 2 * small.overflow_anchors


# ── the driver ────────────────────────────────────────────────────────


def _run(panel, out, *, round_index=2, workers=1, params=None, **kw):
    tmp_path, corpus, t_path, _ = panel
    nv.write_read_minimizers(corpus, tmp_path / "minis", kmer=15, window=10)
    reads = ReadStore.open(corpus)
    store = TemplateStore.open(t_path)
    try:
        stats = nv.run_native_estep(
            store=store,
            reads=reads,
            output_dir=out,
            round_index=round_index,
            minis_dir=tmp_path / "minis",
            kmer=15,
            window=10,
            params=params,
            align_workers=workers,
            corpus_path=corpus,
            templates_path=t_path,
            p_floor=kw.pop("p_floor", 0.97),
            delta_logl=5.0,
            support_ratio=20.0,
            shortlist_k=kw.pop("shortlist_k", 16),
            shortlist_frac=0.8,
            **kw,
        )
    finally:
        reads.close()
        store.close()
    table = ds.dataset(
        sorted(out.glob("part-*.parquet")), schema=EM_ASSIGNMENT_TABLE
    ).to_table()
    return stats, table.sort_by("read_row")


def test_the_driver_assigns_every_real_read_and_accounts_for_the_rest(panel, tmp_path):
    stats, table = _run(panel, tmp_path / "out")
    tid = table.column("template_id").to_pylist()
    assert tid[:36] == [0] * 12 + [1] * 12 + [2] * 12
    assert tid[36:] == [-1, -1]
    assert table.column("unassigned_reason").to_pylist()[36:] == [
        "no_candidate",
        "no_candidate",
    ]
    ident = table.column("identity").to_pylist()
    alen = table.column("aligned_len").to_pylist()
    assert all(v is not None and v >= 0.97 for v in ident[:36])
    assert all(v is not None and v > 500 for v in alen[:36])
    assert ident[36] is None and alen[36] is None
    assert table.column("chain_score").to_pylist()[0] >= 2, "shared probes"
    assert stats["aligner"] == "native"
    assert stats["n_reads_seen"] == 38 and stats["n_assigned"] == 36
    assert stats["n_no_candidate"] == 2 and stats["n_unassigned"] == 2
    assert stats["n_join_rows"] > 0 and stats["n_dropped_strand"] >= 1


def test_the_worker_count_does_not_change_a_single_byte(panel, tmp_path):
    small = nv.NativeParams(block_reads=10)
    _, _ = _run(panel, tmp_path / "w1", workers=1, params=small)
    _, _ = _run(panel, tmp_path / "w3", workers=3, params=small)
    ones = sorted((tmp_path / "w1").glob("part-*.parquet"))
    threes = sorted((tmp_path / "w3").glob("part-*.parquet"))
    assert len(ones) == len(threes) == 4
    for a, b in zip(ones, threes):
        assert (
            hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
        )


def test_round_one_walks_in_replication_order_and_aligns_lazily(panel, tmp_path):
    """Two admissible templates — round 1 takes the better-replicated one
    and never aligns the rest of the list."""
    stats, table = _run(panel, tmp_path / "r1", round_index=1)
    assert (
        table.column("template_id").to_pylist()[:36] == [0] * 12 + [1] * 12 + [2] * 12
    )
    assert stats["n_aligned"] <= 40, "about one alignment per read"


def test_a_failed_alignment_is_its_own_reason(panel, tmp_path):
    stats, table = _run(panel, tmp_path / "none", align_fn=lambda *a, **k: None)
    reasons = table.column("unassigned_reason").to_pylist()
    assert reasons[:36] == ["no_alignment"] * 36
    assert stats["n_no_alignment"] == 36 and stats["n_assigned"] == 0


def test_the_flat_floor_drops_a_low_quality_read_and_the_eased_floor_keeps_it(
    tmp_path,
):
    rng = random.Random(11)
    truth = _rnd(rng, 800)
    rows = [(f"c{i}", _mutated(rng, truth, 0.01), 30.0) for i in range(6)]
    noisy = "".join(
        rng.choice([b for b in "ACGT" if b != ch]) if rng.random() < 0.045 else ch
        for ch in truth
    )
    rows.append(("lowq", noisy, 13.0))
    rows.append(("noq", truth, None))
    corpus = _write_corpus(tmp_path, rows)
    t_path = _write_templates(tmp_path, [truth])
    fake_panel = (tmp_path, corpus, t_path, [truth])

    _, flat = _run(fake_panel, tmp_path / "flat")
    by = dict(
        zip(flat.column("read_id").to_pylist(), flat.column("template_id").to_pylist())
    )
    reason = dict(
        zip(
            flat.column("read_id").to_pylist(),
            flat.column("unassigned_reason").to_pylist(),
        )
    )
    ident = dict(
        zip(flat.column("read_id").to_pylist(), flat.column("identity").to_pylist())
    )
    assert by["lowq"] == -1 and reason["lowq"] == "below_floor"
    assert 0.90 < ident["lowq"] < 0.97, "the best identity is on the record"
    assert by["noq"] == 0, "no quality still clears the flat floor when clean"

    _, eased = _run(fake_panel, tmp_path / "eased", p_floor_quality_scale=1.5)
    by = dict(
        zip(
            eased.column("read_id").to_pylist(),
            eased.column("template_id").to_pylist(),
        )
    )
    assert by["lowq"] == 0, "Q13 may miss by 1.5x its expected error"
    assert all(v == 0 for k, v in by.items() if k != "lowq")


# ── parent-only torch ─────────────────────────────────────────────────


def test_nothing_below_the_fork_imports_torch():
    """The sketch is torch; a torch op in a forked child deadlocks on
    OpenMP. The module's top level — what a worker imports — must not
    touch it, and only the two parent-side builders may."""
    source = Path(nv.__file__).read_text()
    tree = ast.parse(source)
    top = {
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in getattr(n, "names", [None]) or [None]
        if n is not None
    }
    banned = {
        "torch",
        "constellation.sequencing.transcriptome.cluster.denovo.minimizers",
    }
    assert not {t for t in top if t and any(b in str(t) for b in banned)}
    lazy = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module and "minimizers" in n.module
    ]
    assert len(lazy) == 2, "exactly the two parent-side builders"


def test_params_that_cannot_mean_anything_are_refused():
    for kw in (
        {"probes_per_read": 0},
        {"bucket_cap": 1},
        {"min_shared": 0},
        {"diag_band": -1},
        {"max_candidates": 0},
        {"max_rows_per_read": 0},
        {"probes_per_read": 2.5},
    ):
        with pytest.raises(ValueError):
            nv.NativeParams(**kw)
    assert "block_reads" not in nv.NativeParams().semantic()
