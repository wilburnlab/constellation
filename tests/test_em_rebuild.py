"""The final merge's consensus rebuild, on its own: one survivor at a time."""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pytest

edlib = pytest.importorskip("edlib")

from constellation.sequencing.transcriptome.cluster.denovo._io import (  # noqa: E402
    _READS_SCHEMA,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (  # noqa: E402
    MStepParams,
    REFINED_NODE_TABLE,
    NODE_MEMBERSHIP_TABLE,
)
from constellation.sequencing.transcriptome.cluster.denovo.em.rebuild import (  # noqa: E402
    REBUILT_COLUMNS,
    align_to_survivor,
    rebuild_one,
    rebuild_survivors,
)


def _rnd(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _corpus(tmp_path, seqs, quality=None):
    path = tmp_path / "reads.arrow"
    quality = quality if quality is not None else [30.0] * len(seqs)
    table = pa.table(
        {
            "read_id": pa.array([f"r{i}" for i in range(len(seqs))], pa.string()),
            "sequence": pa.array(seqs, pa.large_string()),
            "sample_id": pa.array(np.zeros(len(seqs), np.int64)),
            "dorado_quality": pa.array(quality, pa.float32()),
        },
        schema=_READS_SCHEMA,
    )
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, _READS_SCHEMA) as w:
        w.write_table(table)
    return str(path)


def test_a_read_is_placed_on_the_survivor_with_its_flanks_clipped():
    rng = random.Random(1)
    body = _rnd(rng, 600)
    read = _rnd(rng, 20) + body + _rnd(rng, 7)
    cigar, t_start, q_start = align_to_survivor(read, body, identity_floor=0.97)
    assert (t_start, q_start) == (0, 20)
    assert cigar == "600="
    inside = body[50:400]
    cigar, t_start, q_start = align_to_survivor(inside, body, identity_floor=0.97)
    assert (t_start, q_start, cigar) == (50, 0, "350=")


def test_a_read_below_the_floor_or_off_the_survivor_is_not_placed():
    rng = random.Random(2)
    body = _rnd(rng, 600)
    assert align_to_survivor(_rnd(rng, 600), body, identity_floor=0.97) is None
    noisy = list(body)
    for at in range(0, 600, 12):
        noisy[at] = "C" if body[at] != "C" else "G"
    assert align_to_survivor("".join(noisy), body, identity_floor=0.97) is None
    assert align_to_survivor("".join(noisy), body, identity_floor=0.5) is not None


def test_one_survivor_is_rebuilt_from_its_reads_and_the_cap_samples(tmp_path):
    rng = random.Random(3)
    body = _rnd(rng, 700)
    at = 350
    other = "C" if body[at] != "C" else "G"
    variant = body[:at] + other + body[at + 1 :]
    seqs = [variant] * 30 + [body] * 10 + [_rnd(rng, 700)] * 2
    corpus = _corpus(tmp_path, seqs)
    rows = np.arange(len(seqs))
    row, rebuilt, n_skipped, fraction = rebuild_one(
        4,
        body,
        rows,
        template_id=99,
        corpus_path=corpus,
        params=MStepParams(),
        identity_floor=0.97,
    )
    assert row == 4 and n_skipped == 2 and fraction == 1.0
    assert rebuilt["consensus"] == variant, "30 of 40 placed reads carry it"
    assert rebuilt["n_members_used"] == 40
    assert set(rebuilt) == set(REBUILT_COLUMNS)

    _, capped, _, fraction = rebuild_one(
        4,
        body,
        rows,
        template_id=99,
        corpus_path=corpus,
        params=MStepParams(max_members_per_template=12),
        identity_floor=0.97,
    )
    assert fraction == pytest.approx(12 / 42)
    assert capped["n_members_used"] <= 12 and capped["subsample_fraction"] == fraction
    # Seeded by the template: the same again.
    _, again, _, _ = rebuild_one(
        4,
        body,
        rows,
        template_id=99,
        corpus_path=corpus,
        params=MStepParams(max_members_per_template=12),
        identity_floor=0.97,
    )
    assert again == capped

    _, none, n_skipped, _ = rebuild_one(
        4,
        body,
        np.array([40, 41]),
        template_id=99,
        corpus_path=corpus,
        params=MStepParams(),
        identity_floor=0.97,
    )
    assert none is None and n_skipped == 2


def _substituted(rng, seq, n):
    out = list(seq)
    for at in rng.sample(range(len(seq)), n):
        out[at] = rng.choice([b for b in "ACGT" if b != out[at]])
    return "".join(out)


def test_a_read_is_held_to_the_floor_it_was_admitted_under(tmp_path):
    """The rebuild pools the reads the E-step ASSIGNED. Under
    ``--p-floor-quality-scale`` a Q13 read at 0.95 identity was admitted
    (its floor is ~0.925); held to the flat 0.97 here it was dropped from
    the pool, and a survivor made of such reads was not rebuilt at all."""
    rng = random.Random(5)
    body = _rnd(rng, 800)
    noisy = [_substituted(rng, body, 40) for _ in range(6)]  # 0.95 identity
    seqs = [body] * 10 + noisy
    corpus = _corpus(tmp_path, seqs, quality=[30.0] * 10 + [13.0] * 6)
    kwargs = {
        "template_id": 7,
        "corpus_path": corpus,
        "params": MStepParams(),
        "identity_floor": 0.97,
    }
    _, flat, n_skipped, _ = rebuild_one(0, body, np.arange(16), **kwargs)
    assert n_skipped == 6 and flat["n_members_used"] == 10
    _, eased, n_skipped, _ = rebuild_one(
        0, body, np.arange(16), quality_scale=1.5, **kwargs
    )
    assert n_skipped == 0 and eased["n_members_used"] == 16
    assert eased["consensus"] == body

    # The floor only comes down for the reads whose quality earns it: the
    # same noisy reads called Q30 are still refused under the scale.
    (tmp_path / "hi").mkdir()
    hi = _corpus(tmp_path / "hi", seqs)
    _, _, n_skipped, _ = rebuild_one(
        0, body, np.arange(16), quality_scale=1.5, **{**kwargs, "corpus_path": hi}
    )
    assert n_skipped == 6

    # A survivor made ONLY of such reads: nothing placed, nothing rebuilt.
    _, none, n_skipped, _ = rebuild_one(0, body, np.arange(10, 16), **kwargs)
    assert none is None and n_skipped == 6
    _, some, n_skipped, _ = rebuild_one(
        0, body, np.arange(10, 16), quality_scale=1.5, **kwargs
    )
    assert some is not None and n_skipped == 0 and some["consensus"] == body


def _nodes(rows):
    n = len(rows)
    return pa.table(
        {
            "round": pa.array([1] * n, pa.int32()),
            "parent_template_id": pa.array([r[0] for r in rows], pa.int64()),
            "parent_template_row": pa.array([0] * n, pa.int32()),
            "haplotype_id": pa.array([r[1] for r in rows], pa.int32()),
            "consensus": pa.array([r[2] for r in rows], pa.large_string()),
            "n_reads": pa.array([r[3] for r in rows], pa.int64()),
            "node_weight": pa.array([float(r[3]) for r in rows], pa.float64()),
            "protein": pa.array([None] * n, pa.large_string()),
            "orf_start": pa.array([-1] * n, pa.int32()),
            "orf_end": pa.array([-1] * n, pa.int32()),
            "allele_string": pa.array([None] * n, pa.string()),
            "declared_variants": pa.array([[]] * n, pa.list_(pa.int64())),
            "n_inserted_columns": pa.array([0] * n, pa.int32()),
            "n_extended_5p": pa.array([0] * n, pa.int32()),
            "n_extended_3p": pa.array([0] * n, pa.int32()),
            "n_trimmed_5p": pa.array([0] * n, pa.int32()),
            "n_trimmed_3p": pa.array([0] * n, pa.int32()),
            "n_members_used": pa.array([r[3] for r in rows], pa.int32()),
            "subsample_fraction": pa.array([1.0] * n, pa.float32()),
        },
        schema=REFINED_NODE_TABLE,
    )


def _membership(entries):
    """entries = [(parent_template_id, haplotype_id, read_row)]"""
    return pa.table(
        {
            "round": pa.array([1] * len(entries), pa.int32()),
            "parent_template_id": pa.array([e[0] for e in entries], pa.int64()),
            "haplotype_id": pa.array([e[1] for e in entries], pa.int32()),
            "read_row": pa.array([e[2] for e in entries], pa.int32()),
            "weight": pa.array([1.0] * len(entries), pa.float32()),
        },
        schema=NODE_MEMBERSHIP_TABLE,
    )


def test_only_the_rows_asked_for_are_rewritten_and_a_failure_is_counted(tmp_path):
    rng = random.Random(4)
    a, b = _rnd(rng, 650), _rnd(rng, 720)
    seqs = [a] * 6 + [b[5:]] * 4 + [_rnd(rng, 300)] * 3
    corpus = _corpus(tmp_path, seqs)
    nodes = _nodes([(10, 0, a[20:], 6), (11, 0, b, 4), (12, 0, _rnd(rng, 300), 3)])
    membership = _membership(
        [(10, 0, i) for i in range(6)]
        + [(11, 0, 6 + i) for i in range(4)]
        + [(12, 0, 10 + i) for i in range(3)]
    )
    said = []
    out, counts = rebuild_survivors(
        nodes,
        membership,
        np.array([0, 2]),
        corpus,
        params=MStepParams(),
        identity_floor=0.97,
        threads=1,
        log=said.append,
    )
    assert counts == {
        "n_rebuilt": 1,
        "n_rebuild_failed": 1,
        "n_rebuild_reads_skipped": 3,
    }
    assert out.schema.equals(nodes.schema)
    assert out.column("consensus").to_pylist()[1] == b, "not asked for: untouched"
    assert (
        out.column("consensus").to_pylist()[2] == nodes.column("consensus")[2].as_py()
    )
    assert out.column("consensus").to_pylist()[0] == a, "six reads reach 20 nt past"
    assert out.column("n_extended_5p").to_pylist()[0] == 20
    assert out.column("n_members_used").to_pylist() == [6, 4, 3]
    assert out.column("n_reads").to_pylist() == [6, 4, 3], "the merge's counts stand"
    assert said and "rebuilt 1" in said[0] and "1 kept their own" in said[0]

    same, none = rebuild_survivors(
        nodes,
        membership,
        np.array([], dtype=np.int64),
        corpus,
        params=MStepParams(),
        identity_floor=0.97,
    )
    assert same is nodes and none["n_rebuilt"] == 0
