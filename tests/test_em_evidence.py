"""The EM evidence contract, on a run that needs no minimap2.

Two runs of :func:`tests._em_fixtures.three_gene_panel` under ORF seeding and
the native E-step — one round (whose final merge rebuilds a survivor) and
three rounds (which merge between rounds) — are shared by every test here.

The first tests pin the fixture itself: if it ever stops producing a
covariance split, a merge between rounds or a rebuilt survivor, the tests that
rely on those would pass vacuously, so that is a failure of its own. The
golden test pins the run's outputs to what the code produced before the
evidence work began: everything this contract adds is supposed to be
additive, and that is the check that it is.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (pa.compute)
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _em_fixtures import three_gene_panel  # noqa: E402

from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (  # noqa: E402
    EmParams,
    run_em,
)


def evidence_params(**kw) -> EmParams:
    base = dict(
        threads=1,
        mstep_workers=1,
        max_window_length=None,
        min_aa_length=30,
        estep_aligner="native",
    )
    base.update(kw)
    return EmParams(**base)


@pytest.fixture(scope="module")
def panel(tmp_path_factory) -> tuple[Path, Path]:
    """``(base, demux_dir)`` the shared runs are made from."""
    base = tmp_path_factory.mktemp("em_evidence")
    return base, three_gene_panel(base)


@pytest.fixture(scope="module")
def runs(panel):
    """``{1: out_dir, 3: out_dir}`` — the one-round and three-round runs."""
    base, demux = panel
    out = {}
    for rounds in (1, 3):
        d = base / f"em{rounds}"
        run_em(demux, d, params=evidence_params(rounds=rounds), report=False)
        out[rounds] = d
    return out


def _rounds(out: Path) -> list[Path]:
    return sorted(p.parent for p in (out / "rounds").glob("r*/_SUCCESS"))


def _nodes(rd: Path) -> pa.Table:
    return pa_ds.dataset(sorted((rd / "mstep" / "nodes").glob("part-*.parquet"))).to_table()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


# ── the fixture exercises what the contract has to record ─────────────


def test_the_fixture_splits_gene_c_on_its_three_utr_sites(runs):
    nodes = _nodes(_rounds(runs[1])[0])
    per_parent = Counter(nodes.column("parent_template_id").to_pylist())
    split = [p for p, n in per_parent.items() if n >= 2]
    assert split, "no template split: the covariance path is not exercised"
    alleles = [a for a in nodes.column("allele_string").to_pylist() if a]
    assert any(len(a) == 3 and a != "===" for a in alleles)
    clusters = pq.read_table(runs[1] / "clusters.parquet")
    assert sorted(clusters.column("n_reads").to_pylist()) == [12, 24, 34, 40]


def test_the_fixture_merges_between_rounds_and_rebuilds_a_final_survivor(runs):
    final_1 = _json(_rounds(runs[1])[-1] / "final.json")
    assert final_1["n_rebuilt"] >= 1, "no rebuilt survivor: rebuild is not exercised"
    merged = [
        _json(rd / "refine.json").get("n_merged", 0)
        for rd in _rounds(runs[3])
        if (rd / "refine.json").exists()
    ]
    assert max(merged, default=0) >= 1, "no merge between rounds"


# ── golden: the contract only ever adds ───────────────────────────────


def _digest(table: pa.Table, sort_by: list[str]) -> str:
    """Order-independent digest of a table's rows (all columns)."""
    if table.num_rows:
        table = table.sort_by([(c, "ascending") for c in sort_by])
    rows = table.to_pylist()
    blob = json.dumps(rows, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_digests(out: Path) -> dict[str, str]:
    """Digests of what a run wrote that the evidence work must not change:
    every round's M-step nodes and membership, and the final clusters and
    their membership. New columns or tables elsewhere do not enter."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        NODE_MEMBERSHIP_TABLE,
        REFINED_NODE_TABLE,
    )

    d: dict[str, str] = {}
    for rd in _rounds(out):
        nodes = _nodes(rd).select(REFINED_NODE_TABLE.names)
        d[f"{rd.name}/nodes"] = _digest(nodes, ["parent_template_id", "haplotype_id"])
        mem = pa_ds.dataset(
            sorted((rd / "mstep" / "node_membership").glob("part-*.parquet"))
        ).to_table().select(NODE_MEMBERSHIP_TABLE.names)
        d[f"{rd.name}/membership"] = _digest(
            mem, ["parent_template_id", "haplotype_id", "read_row"]
        )
    d["clusters"] = _digest(pq.read_table(out / "clusters.parquet"), ["cluster_id"])
    d["cluster_membership"] = _digest(
        pq.read_table(out / "cluster_membership.parquet"), ["cluster_id", "read_id"]
    )
    return d


#: Captured on the code before the evidence work (main at a633080). A phase
#: that changes one of these on purpose says so in its commit and updates it
#: here; any other change is a regression.
GOLDEN: dict[int, dict[str, str]] = {
    1: {
        "r01/nodes": "171aec4c670df56e",
        "r01/membership": "9cb916ce1a4357de",
        "clusters": "a18370fb16b84068",
        "cluster_membership": "64181ffcf3bfbe4b",
    },
    3: {
        "r01/nodes": "171aec4c670df56e",
        "r01/membership": "9cb916ce1a4357de",
        "r02/nodes": "9eaa77a036af7ff0",
        "r02/membership": "ae7dc4ce553be2ca",
        "clusters": "8658cbffad4535c2",
        "cluster_membership": "146463bbeb9d13c4",
    },
}


def test_the_runs_match_the_golden_digests(runs):
    for rounds, out in runs.items():
        assert run_digests(out) == GOLDEN[rounds], f"{rounds}-round run changed"


# ── P1.1: what a round runs under is on disk before it runs ───────────


def test_every_round_records_its_parameters(runs):
    from dataclasses import asdict

    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        corpus_digest,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (
        ROUND_PARAMS_VERSION,
    )

    out = runs[3]
    want = json.loads(json.dumps(asdict(evidence_params(rounds=3)), default=str))
    for rd in _rounds(out):
        rec = _json(rd / "params.json")
        assert rec["version"] == ROUND_PARAMS_VERSION
        assert rec["params"] == want
        # The kernel's own former defaults are recorded with the rest.
        for knob in (
            "local_window",
            "boundary_tolerance",
            "max_cooccurrence_budget",
            "min_insertion_support",
        ):
            assert knob in rec["params"]["mstep"]
        assert rec["error_model"]["refit"] is False
        assert rec["libraries"]["numpy"]
        assert rec["corpus_digest"] == corpus_digest(out / "corpus" / "reads.arrow")


def test_a_round_that_dies_still_says_what_it_ran_under(tmp_path, monkeypatch):
    """The point of writing it first: manifest.json only appears at the end."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import rounds

    def boom(*a, **k):
        raise RuntimeError("killed in the M-step")

    monkeypatch.setattr(rounds, "_run_mstep", boom)
    out = tmp_path / "em"
    with pytest.raises(RuntimeError, match="killed"):
        run_em(three_gene_panel(tmp_path), out, params=evidence_params(rounds=1))
    rd = out / "rounds" / "r01"
    assert (rd / "params.json").exists()
    assert not (rd / "_SUCCESS").exists()
    assert not (out / "manifest.json").exists()


def test_the_kernel_knobs_reach_refine_template():
    import inspect

    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep import (
        refine_template,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    kwargs = MStepParams().kernel_kwargs()
    accepted = set(inspect.signature(refine_template).parameters)
    assert set(kwargs) <= accepted
    # Same values the kernel used to default to: recording them changed nothing.
    defaults = {
        k: v.default for k, v in inspect.signature(refine_template).parameters.items()
    }
    for knob in (
        "local_window",
        "boundary_tolerance",
        "max_cooccurrence_budget",
        "min_insertion_support",
    ):
        assert kwargs[knob] == defaults[knob]


# ── P1.2: a round's M-step shards are the ones it recorded (F12) ──────


def _templates_table(out: Path, r: int) -> pa.Table:
    path = out / "rounds" / f"r{r:02d}" / "templates" / "templates.arrow"
    with pa.memory_map(str(path), "r") as mm:
        return pa.ipc.open_file(mm).read_all()


def _plant_stale_shards(rd: Path, index: int = 7) -> None:
    """What an earlier attempt with more units leaves behind: shards past the
    ones the round wrote, holding nodes the round does not have."""
    for key in ("nodes", "node_membership"):
        d = rd / "mstep" / key
        shutil.copy(d / "part-00000.parquet", d / f"part-{index:05d}.parquet")


#: Without the merge: a stale shard is a copy of nodes the round has, and the
#: merge would collapse the copies as byte-identical twins, hiding exactly
#: the duplication these tests look for. Under --no-merge a duplicate node
#: is a duplicate template.
_NO_MERGE = dict(merge=False)


@pytest.fixture(scope="module")
def uninterrupted_two_rounds(tmp_path_factory):
    base = tmp_path_factory.mktemp("em_two")
    out = base / "em"
    run_em(
        three_gene_panel(base),
        out,
        params=evidence_params(rounds=2, **_NO_MERGE),
        report=False,
    )
    return out


def test_an_extended_run_ignores_shards_an_earlier_attempt_left(
    tmp_path, uninterrupted_two_rounds
):
    """Extending a finished run rebuilds the last round's successor from its
    node shards — it always does, a final round never writes them — and that
    used to glob the directory, turning stale shards into duplicate
    templates."""
    out = tmp_path / "em"
    demux = three_gene_panel(tmp_path)
    params = evidence_params(rounds=1, **_NO_MERGE)
    run_em(demux, out, params=params, report=False)
    _plant_stale_shards(out / "rounds" / "r01")
    run_em(demux, out, params=params, resume=True, report=False)

    got = _templates_table(out, 2)
    want = _templates_table(uninterrupted_two_rounds, 2)
    assert got.column("sequence").to_pylist() == want.column("sequence").to_pylist()


def test_a_legacy_round_without_the_record_reads_the_prefix_it_wrote(
    tmp_path, uninterrupted_two_rounds
):
    """A round from before the record has only n_nodes: its shards are the
    prefix whose unique nodes add up to it."""
    out = tmp_path / "em"
    demux = three_gene_panel(tmp_path)
    params = evidence_params(rounds=1, **_NO_MERGE)
    run_em(demux, out, params=params, report=False)
    rd = out / "rounds" / "r01"
    record = _json(rd / "round.json")
    record.pop("mstep", None)
    (rd / "round.json").write_text(json.dumps(record))
    _plant_stale_shards(rd)
    run_em(demux, out, params=params, resume=True, report=False)

    got = _templates_table(out, 2)
    want = _templates_table(uninterrupted_two_rounds, 2)
    assert got.column("sequence").to_pylist() == want.column("sequence").to_pylist()


def test_shards_no_prefix_explains_are_refused_not_guessed(tmp_path):
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        StaleShardsError,
        read_round_nodes,
    )

    out = tmp_path / "em"
    run_em(three_gene_panel(tmp_path), out, params=evidence_params(rounds=1), report=False)
    rd = out / "rounds" / "r01"
    assert read_round_nodes(rd).num_rows == _json(rd / "round.json")["n_nodes"]

    record = _json(rd / "round.json")
    record.pop("mstep")
    record["n_nodes"] = int(record["n_nodes"]) + 1  # matches no prefix
    (rd / "round.json").write_text(json.dumps(record))
    with pytest.raises(StaleShardsError, match="no prefix"):
        read_round_nodes(rd)


def test_a_rerun_round_clears_what_its_dead_attempt_wrote(tmp_path):
    """The M-step clears mstep/ before writing, so a round re-run after a
    crash never sits beside the dead attempt's extra shards."""
    out = tmp_path / "em"
    demux = three_gene_panel(tmp_path)
    run_em(demux, out, params=evidence_params(rounds=1), report=False)
    rd = out / "rounds" / "r01"
    _plant_stale_shards(rd, index=9)
    (rd / "_SUCCESS").unlink()  # the round died after its M-step
    run_em(demux, out, params=evidence_params(rounds=1), resume=True, report=False)
    assert (rd / "_SUCCESS").exists()
    for key in ("nodes", "node_membership"):
        assert not (rd / "mstep" / key / "part-00009.parquet").exists()
    n_units = _json(rd / "round.json")["mstep"]["n_units"]
    assert len(list((rd / "mstep" / "nodes").glob("part-*.parquet"))) == n_units


# ── P1.3: every template the M-step touched is recorded (F2, F5) ──────


def _assignments_of(rd: Path) -> pa.Table:
    from constellation.sequencing.transcriptome.cluster.denovo.em.assign import (
        EM_ASSIGNMENT_TABLE,
    )

    files = sorted((rd / "assignments").glob("part-*.parquet"))
    return pa_ds.dataset(files, schema=EM_ASSIGNMENT_TABLE).to_table()


def _templates_record(rd: Path) -> pa.Table:
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_templates,
    )

    t = read_round_templates(rd)
    assert t is not None, f"{rd.name} has no mstep/templates"
    return t


def _split_template(rd: Path) -> dict:
    t = _templates_record(rd).to_pylist()
    split = [row for row in t if row["n_nodes_emitted"] >= 2]
    assert split, "no template split"
    return split[0]


def test_every_template_with_reads_is_recorded_and_no_read_is_lost(runs):
    for rd in _rounds(runs[3]):
        t = _templates_record(rd)
        a = _assignments_of(rd)
        live = a.filter(pa.compute.greater_equal(a.column("template_row"), 0))
        assert t.num_rows == len(set(live.column("template_row").to_pylist()))
        assert sum(t.column("n_assigned").to_pylist()) == live.num_rows
        assert set(t.column("status").to_pylist()) == {"ok"}
        block = _json(rd / "round.json")["mstep"]
        assert block["n_reads_assigned"] == block["n_reads_in_nodes"] == live.num_rows
        assert block["n_failed"] == 0 and block["n_templates"] == t.num_rows


def test_the_split_is_visible_in_the_record(runs):
    """The counters refine_template built "so that the method returned one
    template is visible rather than inferred" — now actually on disk."""
    rd = _rounds(runs[1])[0]
    split = _split_template(rd)
    assert split["n_nodes"] == split["n_nodes_emitted"] == 2
    assert split["n_significant_edges"] >= 1 and split["n_signatures"] >= 1
    assert split["n_candidates"] >= 3, "the three UTR sites are candidates"
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_disagreements,
    )

    d = read_round_disagreements(rd)
    assert d is not None and d.num_rows > 0
    assert set(d.column("class_code").to_pylist()) <= {0, 1, 2}


def _replay_inputs(out: Path, rd: Path, template_id: int):
    """The template's slice, corpus, store and parameters, as the run had
    them — the inputs replay will use."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.corpus import (
        ReadStore,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
        sort_assignments_by_template,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
        TemplateStore,
    )

    store = TemplateStore.open(rd / "templates" / "templates.arrow")
    row = int(np.flatnonzero(np.asarray(store.template_id) == template_id)[0])
    srt, _ = sort_assignments_by_template(_assignments_of(rd))
    mine = srt.filter(pa.compute.equal(srt.column("template_row"), row))
    reads = ReadStore.open(out / "corpus" / "reads.arrow")
    params = MStepParams(**_json(rd / "params.json")["params"]["mstep"])
    return mine, reads, store, row, params


def test_the_record_is_what_the_shared_code_path_produces(runs):
    """refine_one_template is the worker body; called directly on the
    round's own inputs it reproduces the round's record and nodes."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        refine_one_template,
    )

    out = runs[1]
    rd = _rounds(out)[0]
    split = _split_template(rd)
    mine, reads, store, row, params = _replay_inputs(out, rd, split["template_id"])
    try:
        res = refine_one_template(
            mine, reads, store, row, params, round_index=1, on_error="raise"
        )
    finally:
        reads.close()
        store.close()
    for name in ("n_assigned", "n_members_used", "status", "n_nodes_emitted"):
        assert getattr(res.record, name) == split[name]
    for name, value in res.record.counters.items():
        assert split[name] == value, name
    nodes = _nodes(rd).filter(
        pa.compute.equal(_nodes(rd).column("parent_template_id"), split["template_id"])
    )
    assert sorted(r[4] for r in res.node_rows) == sorted(
        nodes.column("consensus").to_pylist()
    )


@pytest.fixture(scope="module")
def split_template_id(runs) -> int:
    return int(_split_template(_rounds(runs[1])[0])["template_id"])


def _fail_on(monkeypatch, template_id: int, *, pooled_too: bool = False):
    from constellation.sequencing.transcriptome.cluster.denovo.em import mstep

    real = mstep.refine_template_with_internals

    def flaky(frame, members, *, template_id: int = 0, **kw):
        if template_id == target:
            raise RuntimeError("injected kernel failure")
        return real(frame, members, template_id=template_id, **kw)

    target = template_id
    monkeypatch.setattr(mstep, "refine_template_with_internals", flaky)
    if pooled_too:
        def no_pool(*a, **k):
            raise RuntimeError("injected pooled failure")

        monkeypatch.setattr(mstep, "pooled_node", no_pool)


def test_a_template_whose_kernel_raises_keeps_its_reads(
    tmp_path, monkeypatch, split_template_id
):
    """It used to be skipped: no node, no membership, so its reads were in
    the round's assignments and in no cluster, and nothing counted them."""
    _fail_on(monkeypatch, split_template_id)
    out = tmp_path / "em"
    run_em(three_gene_panel(tmp_path), out, params=evidence_params(rounds=1))
    rd = out / "rounds" / "r01"
    rec = {r["template_id"]: r for r in _templates_record(rd).to_pylist()}[
        split_template_id
    ]
    assert rec["status"] == "fallback_pooled"
    assert rec["error_type"] == "RuntimeError" and "injected" in rec["error_message"]
    assert rec["n_nodes_emitted"] == 1
    block = _json(rd / "round.json")["mstep"]
    assert block["n_failed"] == 1
    assert block["n_reads_assigned"] == block["n_reads_in_nodes"] == 110
    clusters = pq.read_table(out / "clusters.parquet")
    assert sum(clusters.column("n_reads").to_pylist()) == 110
    report = (out / "diagnostics" / "report.md").read_text()
    assert "the M-step raised on 1 template" in report


def test_when_the_pooled_consensus_fails_too_the_template_carries_forward(
    tmp_path, monkeypatch, split_template_id
):
    _fail_on(monkeypatch, split_template_id, pooled_too=True)
    out = tmp_path / "em"
    run_em(three_gene_panel(tmp_path), out, params=evidence_params(rounds=1))
    rd = out / "rounds" / "r01"
    rec = {r["template_id"]: r for r in _templates_record(rd).to_pylist()}[
        split_template_id
    ]
    assert rec["status"] == "fallback_template"
    nodes = _nodes(rd)
    mine = nodes.filter(pa.compute.equal(nodes.column("parent_template_id"), split_template_id))
    from constellation.sequencing.transcriptome.cluster.denovo.em.templates import (
        TemplateStore,
    )

    store = TemplateStore.open(rd / "templates" / "templates.arrow")
    try:
        row = int(np.flatnonzero(np.asarray(store.template_id) == split_template_id)[0])
        assert mine.column("consensus").to_pylist() == [store.sequence(row)]
    finally:
        store.close()
    assert sum(pq.read_table(out / "clusters.parquet").column("n_reads").to_pylist()) == 110


class _Store:
    def __init__(self, seq: str, template_id: int):
        self.template_id = np.array([template_id], dtype=np.int64)
        self._seq = seq

    def sequence(self, row: int) -> str:
        return self._seq


class _Reads:
    def __init__(self, seqs):
        self._seqs = list(seqs)

    def take_sequences(self, rows):
        return [self._seqs[int(r)] for r in rows]


def test_a_template_none_of_whose_reads_aligned_keeps_them():
    """Every CIGAR null: there is nothing to build a consensus from, and the
    template used to be skipped with its reads."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
        refine_one_template,
    )

    frame = "ATG" + "GCT" * 40 + "TAA"
    rows = pa.table(
        {
            "template_row": pa.array([0, 0, 0], pa.int32()),
            "read_row": pa.array([4, 7, 9], pa.int32()),
            "cigar": pa.array([None, "", None], pa.large_string()),
            "t_start": pa.array([0, 0, 0], pa.int32()),
            "q_start": pa.array([0, 0, 0], pa.int32()),
            "weight": pa.array([1.0, 1.0, 1.0], pa.float32()),
        }
    )
    res = refine_one_template(
        rows,
        _Reads([frame] * 10),
        _Store(frame, 42),
        0,
        MStepParams(),
        round_index=1,
        on_error="raise",
    )
    assert res.record.status == "no_members"
    assert len(res.nodes) == 1 and res.nodes[0].consensus == frame
    assert res.read_row.tolist() == [4, 7, 9]
    assert res.haplotype_of_read.tolist() == [0, 0, 0]
    assert [r[5] for r in res.node_rows] == [3]  # n_reads


def test_a_rebuild_that_raises_is_counted_and_the_outputs_still_written(
    tmp_path, monkeypatch
):
    """A surviving fallback node can fail again in the final rebuild; that
    used to take the whole final stage down after every round finished."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import rebuild

    def boom(*a, **k):
        raise RuntimeError("injected rebuild failure")

    monkeypatch.setattr(rebuild, "rebuild_one", boom)
    out = tmp_path / "em"
    run_em(three_gene_panel(tmp_path), out, params=evidence_params(rounds=1))
    final = _json(_rounds(out)[-1] / "final.json")
    assert final["n_rebuild_errors"] >= 1
    assert final["n_rebuild_failed"] >= final["n_rebuild_errors"]
    assert final["n_rebuilt"] == 0
    assert sum(pq.read_table(out / "clusters.parquet").column("n_reads").to_pylist()) == 110


# ── P1.4: the compact evidence (F1, F7) ───────────────────────────────


def _for_template(table: pa.Table, template_id: int) -> pa.Table:
    return table.filter(pa.compute.equal(table.column("template_id"), template_id))


def test_the_split_s_columns_edges_and_reads_are_on_disk(runs, split_template_id):
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_columns,
        read_round_covariance,
        read_round_reads,
    )

    rd = _rounds(runs[1])[0]
    cols = _for_template(read_round_columns(rd), split_template_id)
    kept = cols.filter(pa.compute.field("kept"))
    # The three UTR sites, at the 12/36 minor fraction the fixture planted.
    assert kept.num_rows == 3
    assert all(abs(f - 12 / 36) < 0.05 for f in kept.column("minor_frac").to_pylist())
    assert len(set(kept.column("signature").to_pylist())) == 1
    # Substitutions: template columns, not insertion columns (-1).
    assert min(kept.column("template_col").to_pylist()) >= 0

    cov = _for_template(read_round_covariance(rd), split_template_id)
    pairs = cov.filter(pa.compute.invert(pa.compute.field("self_edge")))
    assert pairs.num_rows == 3, "every pair of the three sites"
    assert min(pairs.column("phi").to_pylist()) > 0.9, "perfectly linked"
    assert all(abs(n - 12) < 1.5 for n in pairs.column("n11").to_pylist())

    reads = _for_template(read_round_reads(rd), split_template_id)
    assert reads.num_rows == 36
    assert sorted(Counter(reads.column("haplotype_id").to_pylist()).values()) == [12, 24]
    assert all(reads.column("sampled").to_pylist())
    margins = reads.column("margin").to_pylist()
    assert min(margins) > 0, "every read clearly on its side of a 3-site split"
    assert _json(rd / "round.json")["mstep"]["evidence_bytes"] > 0


def test_the_evidence_is_what_the_shared_code_path_computes(runs, split_template_id):
    """The stored rows equal the in-memory evidence, and the stored 2x2s are
    the read states' own co-observation counts."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import covariance as cv
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        refine_one_template,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        read_round_columns,
        read_round_covariance,
    )

    out = runs[1]
    rd = _rounds(out)[0]
    mine, reads, store, row, params = _replay_inputs(out, rd, split_template_id)
    try:
        res = refine_one_template(
            mine, reads, store, row, params, round_index=1, evidence="full",
            on_error="raise",
        )
    finally:
        reads.close()
        store.close()
    assert res.evidence["columns"].equals(
        _for_template(read_round_columns(rd), split_template_id)
    )
    stored = _for_template(read_round_covariance(rd), split_template_id)
    assert res.evidence["covariance"].sort_by(
        [("u_frame_col", "ascending"), ("v_frame_col", "ascending")]
    ).equals(stored)

    internals = res.internals
    x = cv.dense_states(internals.states, np.arange(internals.states.n_candidates))
    w = np.asarray(internals.states.weight)
    for e in stored.filter(pa.compute.invert(pa.compute.field("self_edge"))).to_pylist():
        u, v = e["u"], e["v"]
        both = (x[:, u] != cv.UNOBSERVED) & (x[:, v] != cv.UNOBSERVED)
        nm_u = both & (x[:, u] != cv.MAJOR)
        nm_v = both & (x[:, v] != cv.MAJOR)
        assert e["n"] == pytest.approx(w[both].sum())
        assert e["n1x"] == pytest.approx(w[nm_u].sum())
        assert e["nx1"] == pytest.approx(w[nm_v].sum())
        assert e["n11"] == pytest.approx(w[nm_u & nm_v].sum())


def _evidence_tables(out: Path) -> dict[str, pa.Table]:
    from constellation.sequencing.transcriptome.cluster.denovo.em import round_io

    tabs = {}
    for rd in _rounds(out):
        for name in ("templates", "disagreements", "columns", "covariance", "reads"):
            t = getattr(round_io, f"read_round_{name}")(rd)
            tabs[f"{rd.name}/{name}"] = t
    return tabs


def test_the_evidence_does_not_depend_on_the_worker_count(runs, tmp_path):
    out = tmp_path / "em"
    run_em(
        three_gene_panel(tmp_path),
        out,
        params=evidence_params(rounds=1, threads=2, mstep_workers=2),
        report=False,
    )
    one, two = _evidence_tables(runs[1]), _evidence_tables(out)
    assert one.keys() == two.keys()
    for key in one:
        assert one[key].equals(two[key]), key


def test_no_compact_evidence_keeps_the_accounting_and_changes_no_node(tmp_path):
    from constellation.sequencing.transcriptome.cluster.denovo.em import round_io

    out = tmp_path / "em"
    run_em(
        three_gene_panel(tmp_path),
        out,
        params=evidence_params(rounds=1, mstep_evidence="none"),
        report=False,
    )
    rd = _rounds(out)[0]
    assert round_io.read_round_templates(rd).num_rows > 0
    assert round_io.read_round_disagreements(rd).num_rows > 0
    for name in ("columns", "covariance", "reads"):
        assert getattr(round_io, f"read_round_{name}")(rd) is None
    assert set(round_io.read_round_templates(rd).column("evidence_status").to_pylist()) == {
        "none"
    }
    assert run_digests(out) == GOLDEN[1]


def test_evidence_that_raises_never_changes_a_node(tmp_path, monkeypatch):
    from constellation.sequencing.transcriptome.cluster.denovo.em import evidence

    def boom(**k):
        raise RuntimeError("injected evidence failure")

    monkeypatch.setattr(evidence, "compact_evidence", boom)
    out = tmp_path / "em"
    run_em(three_gene_panel(tmp_path), out, params=evidence_params(rounds=1), report=False)
    t = _templates_record(_rounds(out)[0])
    assert set(t.column("status").to_pylist()) == {"ok"}
    assert set(t.column("evidence_status").to_pylist()) == {"failed"}
    assert run_digests(out) == GOLDEN[1]


# ── P1.5: who came second (F6) ────────────────────────────────────────


def test_the_split_haplotypes_are_each_other_s_runner_up(runs):
    """Round 2 under the native E-step: gene C's two templates are its two
    haplotypes, so each read's runner-up is the other, three substitutions
    of likelihood behind."""
    rd = _rounds(runs[3])[1]
    a = _assignments_of(rd)
    gene_c = a.filter(pa.compute.starts_with(a.column("read_id"), "gC_"))
    winners = set(gene_c.column("template_id").to_pylist())
    assert len(winners) == 2, "gene C sits on its two haplotype templates"
    for row in gene_c.to_pylist():
        assert row["runner_up_template_id"] in winners - {row["template_id"]}
        assert row["runner_up_logl_gap"] is not None
    gaps = [r["runner_up_logl_gap"] for r in gene_c.to_pylist()]
    assert sum(g > 0 for g in gaps) / len(gaps) > 0.9


# ── P1.6: a final cluster joined to its history; content ids (F3, F8) ──


def test_every_cluster_names_the_round_node_it_is(runs):
    from constellation.core.ids import content_id
    from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (
        template_id_for,
    )

    out = runs[1]
    rd = _rounds(out)[-1]
    clusters = pq.read_table(out / "clusters.parquet")
    nodes_map = pq.read_table(out / "cluster_nodes.parquet")
    assert nodes_map.column("cluster_id").to_pylist() == clusters.column(
        "cluster_id"
    ).to_pylist()

    round_nodes = _nodes(rd)
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        in_node_order,
    )

    round_nodes = in_node_order(round_nodes)
    key_row = {
        (p, h): i
        for i, (p, h) in enumerate(
            zip(
                round_nodes.column("parent_template_id").to_pylist(),
                round_nodes.column("haplotype_id").to_pylist(),
            )
        )
    }
    final = _json(rd / "final.json")
    merged = pq.read_table(rd / "merged_final.parquet")
    absorbed = set(merged.column("absorbed_template_id").to_pylist())
    seen_nodes = set()
    for row, cluster in zip(nodes_map.to_pylist(), clusters.to_pylist()):
        node_row = key_row[(row["parent_template_id"], row["haplotype_id"])]
        assert row["node_template_id"] == template_id_for(row["round"] + 1, node_row)
        assert row["node_template_id"] not in absorbed
        seen_nodes.add(row["node_template_id"])
        assert row["consensus_content_id"] == content_id(cluster["consensus_sequence"])
        if cluster["predicted_protein"]:
            assert row["protein_content_id"] == content_id(cluster["predicted_protein"])
        if not row["rebuilt"]:
            # Not rebuilt: the cluster IS the node — its consensus is the node's.
            node_cons = round_nodes.column("consensus")[node_row].as_py()
            assert cluster["consensus_sequence"] == node_cons
    # Every node the merge kept is exactly one cluster; the absorbed are none.
    assert len(seen_nodes) == clusters.num_rows
    assert sum(nodes_map.column("rebuilt").to_pylist()) == final["n_rebuilt"]
    total_absorbed = sum(nodes_map.column("n_absorbed").to_pylist())
    assert total_absorbed == merged.num_rows == final["n_merged"]


def test_without_a_final_merge_the_clusters_are_the_nodes_in_order(tmp_path):
    out = tmp_path / "em"
    run_em(
        three_gene_panel(tmp_path),
        out,
        params=evidence_params(rounds=1, merge=False),
        report=False,
    )
    m = pq.read_table(out / "cluster_nodes.parquet")
    assert not any(m.column("rebuilt").to_pylist())
    assert set(m.column("n_absorbed").to_pylist()) == {0}
    from constellation.sequencing.transcriptome.cluster.denovo.em.refine import (
        template_id_for,
    )

    assert m.column("node_template_id").to_pylist() == [
        template_id_for(2, i) for i in range(m.num_rows)
    ]


# ── P1.7: replay, the support tables and the read API (F1 dense, F4) ──


def test_every_template_of_every_round_replays_as_the_run(runs):
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    for out in runs.values():
        for r in R.finished_rounds(out):
            loc = R.assignment_locator(out / "rounds" / f"r{r:02d}")
            ids = R.mstep_record(out, r, "templates").column("template_id").to_pylist()
            assert ids
            for tid in ids:
                rep = R.replay_template(out, r, tid, locator=loc)
                assert rep.verified, (out.name, r, tid, rep.mismatch)
                assert rep.params_source == "round"


def test_every_final_cluster_replays_rebuilt_or_not(runs):
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    out = runs[1]
    kinds = set()
    for cid in R.cluster_nodes(out).column("cluster_id").to_pylist():
        rep = R.replay_cluster(out, cid)
        assert rep.verified, (cid, rep.mismatch)
        kinds.add(rep.kind)
    assert kinds == {"node", "rebuilt"}


def test_a_replay_that_is_not_the_run_says_so(runs, split_template_id):
    """Run the kernel under parameters that cannot split, and the check
    must notice — a replay is only worth showing if it is the run."""
    from dataclasses import replace

    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
        ReplayError,
        round_params,
    )

    out = runs[1]
    params, _em, _src = round_params(out, 1)
    rep = R.replay_template(
        out, 1, split_template_id, params=replace(params, min_node_reads=1000.0)
    )
    assert not rep.verified and rep.params_source == "explicit"
    with pytest.raises(ReplayError, match="does not match"):
        R.replay_template(
            out,
            1,
            split_template_id,
            params=replace(params, min_node_reads=1000.0),
            strict=True,
        )


def test_a_run_from_before_the_records_still_replays(runs, tmp_path):
    """No params.json, no cluster_nodes.parquet, no M-step record: the
    manifest's parameters and the merge's own files are enough. Without the
    manifest too, explicit parameters are."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
        ReplayError,
    )

    old = tmp_path / "old"
    shutil.copytree(runs[1], old)
    (old / "cluster_nodes.parquet").unlink()
    for rd in _rounds(old):
        (rd / "params.json").unlink()
        (rd / "final_params.json").unlink(missing_ok=True)
        for key in ("templates", "disagreements", "columns", "covariance", "reads"):
            shutil.rmtree(rd / "mstep" / key, ignore_errors=True)
        record = _json(rd / "round.json")
        record.pop("mstep", None)
        (rd / "round.json").write_text(json.dumps(record))
    for cid in R.cluster_nodes(old).column("cluster_id").to_pylist():
        rep = R.replay_cluster(old, cid)
        assert rep.verified, (cid, rep.mismatch)
    rep = R.replay_template(old, 1, int(R.nodes(old, 1).column("parent_template_id")[0].as_py()))
    assert rep.verified and rep.params_source == "manifest"

    (old / "manifest.json").unlink()
    tid = int(R.nodes(old, 1).column("parent_template_id")[0].as_py())
    with pytest.raises(ReplayError, match="params"):
        R.replay_template(old, 1, tid)
    assert R.replay_template(old, 1, tid, params=MStepParams()).verified


def test_the_locator_meets_rows_in_the_order_the_m_step_did(runs, tmp_path):
    """Across several shards: the sorted glob concatenated, then a stable
    sort on template_row — sort_assignments_by_template's order."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        sort_assignments_by_template,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
        assignment_locator,
    )

    rd = tmp_path / "r01"
    (rd / "assignments").mkdir(parents=True)
    table = _assignments_of(_rounds(runs[1])[0])
    cuts = [0, 37, 71, table.num_rows]
    for i in range(3):
        pq.write_table(
            table.slice(cuts[i], cuts[i + 1] - cuts[i]),
            rd / "assignments" / f"part-{i:05d}.parquet",
        )
    srt, _ = sort_assignments_by_template(table)
    loc = assignment_locator(rd)
    by_locator = table.take(pa.array(loc.order)).column("read_row").to_pylist()
    assert by_locator == srt.column("read_row").to_pylist()


def test_departures_rebuild_every_sampled_read_exactly(runs, split_template_id):
    """The support view's encoding is lossless against the M-step's own
    projection: the reference across each read's span, with its departures,
    is the read's allele at every PWM column."""
    from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
        member_alleles,
        member_spans,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    rep = R.replay_template(runs[1], 1, split_template_id)
    st = R.support_tables(rep, space="frame")
    pooled = rep.result.internals.pooled
    want = member_alleles(pooled, np.arange(pooled.plan.n_columns))
    # Inside a read's span an ambiguous base (-1 to member_alleles, "no
    # evidence") is carried as 5, N — never left to read as the reference.
    lo, hi = member_spans(pooled)
    cols = np.arange(want.shape[1])
    inside = (cols[None, :] >= lo[:, None]) & (cols[None, :] < hi[:, None])
    want = np.where((want == -1) & inside, 5, want)
    ref = np.where(np.asarray(pooled.winner) == 5, -2, np.asarray(pooled.winner))
    reads = st.reads.to_pylist()
    deps = st.departures.to_pylist()
    member_of_idx = {}
    for k, pos in enumerate(rep.result.member_pos.tolist()):
        member_of_idx[pos] = k
    got = np.full(want.shape, -1, dtype=np.int64)
    for r in reads:
        if not r["sampled"]:
            continue
        k = member_of_idx[r["idx"]]
        got[k, r["span_lo"] : r["span_hi"]] = ref[r["span_lo"] : r["span_hi"]]
    for d in deps:
        got[member_of_idx[d["read_idx"]], d["col"]] = d["allele"]
    assert np.array_equal(got, want.astype(np.int64))
    # The split is visible: the minor haplotype's reads depart at the sites.
    minor = {r["idx"] for r in reads if r["node"] == 1}
    at = {}
    for d in deps:
        if d["read_idx"] in minor:
            at[d["col"]] = at.get(d["col"], 0) + 1
    assert sum(1 for n in at.values() if n >= 10) >= 3


def test_the_consensus_space_carries_the_node_and_its_arcs(runs, split_template_id):
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    rep = R.replay_template(runs[1], 1, split_template_id)
    node1 = [n for n in rep.result.nodes if n.haplotype_id == 1][0]
    st = R.support_tables(rep, node=1)
    assert st.consensus == node1.consensus and st.n_columns == len(node1.consensus)
    assert st.reads.num_rows == 12 and set(st.reads.column("node").to_pylist()) == {1}
    assert st.columns.num_rows == len(node1.consensus)
    assert st.arcs.num_rows >= 3
    assert all(0 <= c < st.n_columns for c in st.arcs.column("u_col").to_pylist())
    # Against its own consensus a node's reads depart only by error.
    per_read = st.reads.column("n_departures").to_numpy()
    assert np.median(per_read) < 0.03 * len(node1.consensus)


def test_reads_beyond_the_member_cap_are_listed_unsampled(tmp_path):
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    out = tmp_path / "em"
    run_em(
        three_gene_panel(tmp_path),
        out,
        params=evidence_params(rounds=1, mstep=MStepParams(max_members_per_template=10)),
        report=False,
    )
    t = R.mstep_record(out, 1, "templates")
    big = t.filter(pa.compute.greater(t.column("n_assigned"), 10)).to_pylist()[0]
    rep = R.replay_template(out, 1, big["template_id"])
    assert rep.verified
    st = R.support_tables(rep)
    sampled = st.reads.column("sampled").to_pylist()
    assert sum(sampled) == 10 and len(sampled) == big["n_assigned"]
    unsampled = st.reads.filter(pa.compute.invert(st.reads.column("sampled")))
    assert unsampled.column("span_lo").null_count == unsampled.num_rows


def test_the_read_api_answers_the_bench_s_first_questions(runs):
    """Where did gene C's reads go, round by round — the question every
    investigation started with."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    out = runs[3]
    info = R.run_info(out)
    assert info["finished"] and info["has_compact_evidence"] and info["n_reads"] == 110
    assert R.round_summaries(out).num_rows == len(info["rounds"])
    clusters = R.clusters(out)
    assert "node_template_id" in clusters.column_names
    ids = [f"gC_minor_r{i:02d}" for i in range(12)]
    rows = R.read_rows_for_ids(out, ids)
    assert sorted(rows.column("read_id").to_pylist()) == ids
    hist = R.read_history(out, rows.column("read_row").to_numpy())
    assert set(hist.column("round").to_pylist()) == set(info["rounds"])
    found = R.find_clusters(out, ids[0])
    assert found.num_rows == 1
    assert R.find_clusters(out, str(found.column("cluster_id")[0].as_py())).num_rows >= 1
    assert R.lineage(out).num_rows > 0


# ── ledger #78 (a)–(c): what the Nov2026 run found in replay ──────────


def test_a_rebuilt_survivor_that_was_not_node_0_has_support_tables(runs):
    """(a) The rebuilt consensus is the survivor's and carries its haplotype
    id; a survivor need not have been its parent's node 0."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    out = runs[1]
    nodes = R.cluster_nodes(out)
    cid = nodes.filter(pa.compute.field("rebuilt")).column("cluster_id")[0].as_py()
    rep = R.replay_cluster(out, cid)
    assert rep.kind == "rebuilt" and rep.verified
    rep.haplotype_id = 3  # as if the survivor had been its parent's node 3
    st = R.support_tables(rep)
    assert st.reference == "3"
    assert set(st.reads.column("node").to_pylist()) == {3}
    assert st.reads.num_rows == rep.pool_read_row.size


def test_verification_covers_the_compact_evidence(runs, tmp_path, split_template_id):
    """(b) A replay certifies what the support view draws: the stored
    columns, covariance and margins, not only the nodes they led to."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
        index_round,
    )

    out = tmp_path / "copy"
    shutil.copytree(runs[1], out)
    rd = out / "rounds" / "r01"
    index = index_round(rd)
    unit = int(index.node_unit[index.node_rows_of(split_template_id)[0]])
    path = index.shards["columns"][unit]
    table = pq.read_table(path)
    mine = pa.compute.equal(table.column("template_id"), split_template_id)
    frac = table.column("minor_frac").to_numpy().copy()
    frac[np.flatnonzero(mine.to_numpy(zero_copy_only=False))[0]] += 0.25
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("minor_frac"), "minor_frac", pa.array(frac)
        ),
        path,
    )
    rep = R.replay_template(out, 1, split_template_id)
    assert not rep.verified
    assert any(m.startswith("columns: minor_frac") for m in rep.mismatch)


def test_round_index_returns_nodes_in_the_order_asked(runs):
    """A rebuilt survivor's group is [survivor, absorbed...] and its head is
    read as the survivor; returning the rows sorted put an absorbed node of
    a lower row there (every rebuilt cluster of the Nov2026 run whose
    survivor out-rowed what it absorbed failed to verify)."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
        index_round,
    )

    index = index_round(runs[1] / "rounds" / "r01")
    n = index.node_parent.size
    assert n >= 3
    asked = np.array([n - 1, 0, n // 2], dtype=np.int64)
    got = index.nodes_at(asked)
    assert got.column("parent_template_id").to_pylist() == index.node_parent[
        asked
    ].tolist()
    assert got.column("haplotype_id").to_pylist() == index.node_hap[asked].tolist()


def test_verification_accepts_floats_from_another_machine(
    runs, tmp_path, split_template_id
):
    """A p-value recomputed on another machine or library build can differ in
    its last bits with no decision differing (OSC against the workstation on
    the Nov2026 run: 4.6e-13 relative). That verifies, and says so."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.replay import (
        index_round,
    )

    out = tmp_path / "copy"
    shutil.copytree(runs[1], out)
    rd = out / "rounds" / "r01"
    index = index_round(rd)
    unit = int(index.node_unit[index.node_rows_of(split_template_id)[0]])
    path = index.shards["columns"][unit]
    table = pq.read_table(path)
    mine = pa.compute.equal(table.column("template_id"), split_template_id)
    p = table.column("p_value").to_numpy(zero_copy_only=False).copy()
    rows = np.flatnonzero(mine.to_numpy(zero_copy_only=False) & (p > 0))
    assert rows.size
    p[rows] *= 1 + 5e-13
    pq.write_table(
        table.set_column(table.schema.get_field_index("p_value"), "p_value", pa.array(p)),
        path,
    )
    rep = R.replay_template(out, 1, split_template_id)
    assert rep.verified, rep.mismatch
    assert any("not bit for bit" in w for w in rep.warnings)


def test_replay_reads_one_unit_not_the_round(runs, monkeypatch):
    """(c) A replay reads the shards of the unit that refined its template,
    never the round's whole node or membership table, and indexes a round
    once."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import replay
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em import round_io

    def whole_round(*a, **k):
        raise AssertionError("replay read a whole round table")

    monkeypatch.setattr(round_io, "read_round_nodes", whole_round)
    monkeypatch.setattr(round_io, "read_round_membership", whole_round)
    built = []
    real = replay.assignment_locator
    monkeypatch.setattr(
        replay, "assignment_locator", lambda rd: built.append(rd) or real(rd)
    )
    replay._INDEX_CACHE.clear()
    out = runs[1]
    cids = R.cluster_nodes(out).column("cluster_id").to_pylist()
    for cid in cids:
        assert R.replay_cluster(out, cid).verified
    assert len(built) == 1, "the round was indexed once and reused"


# ── review of the evidence branch (Codex, 2026-11-19) ─────────────────

_FINAL_OUTPUTS = (
    "clusters.parquet",
    "cluster_membership.parquet",
    "cluster_nodes.parquet",
    "feature_quant.parquet",
    "cluster.fa",
    "proteins.fasta",
    "cluster_edges.parquet",
    "manifest.json",
    "_SUCCESS",
)


def _without_final_stage(src: Path, dst: Path) -> Path:
    """A copy of a finished run with its final stage gone, as
    ``--resume --rounds 0`` finds one."""
    shutil.copytree(src, dst)
    for name in _FINAL_OUTPUTS:
        (dst / name).unlink(missing_ok=True)
    last = _rounds(dst)[-1]
    for name in ("merged_final.parquet", "final.json", "final_params.json"):
        (last / name).unlink(missing_ok=True)
    shutil.rmtree(last / "graph_final", ignore_errors=True)
    return dst


def test_a_rebuilt_survivor_is_replayed_under_the_final_stage_s_parameters(
    runs, panel, tmp_path
):
    """1. ``--resume --rounds 0`` finalized with a member cap of 10, and the
    replay — under the last round's parameters — pooled all of a survivor's
    reads and still verified."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.mstep_pool import (
        MStepParams,
    )

    _base, demux = panel
    out = _without_final_stage(runs[1], tmp_path / "em")
    run_em(
        demux,
        out,
        params=evidence_params(rounds=0, mstep=MStepParams(max_members_per_template=10)),
        resume=True,
        report=False,
    )
    assert (out / "rounds" / "r01" / "final_params.json").exists()
    nodes = R.cluster_nodes(out).filter(pa.compute.field("rebuilt"))
    cids = nodes.column("cluster_id").to_pylist()
    assert cids
    seen_capped = False
    for cid in cids:
        rep = R.replay_cluster(out, cid)
        assert rep.kind == "rebuilt" and rep.verified, (cid, rep.mismatch)
        assert rep.member_read_row.size <= 10
        seen_capped |= rep.pool_read_row.size > 10
    assert seen_capped, "no rebuilt survivor pools more than the cap"


def test_a_survivor_whose_rebuild_failed_keeps_every_member(
    runs, panel, tmp_path, monkeypatch
):
    """2. The final merge gave a survivor other nodes' reads and its rebuild
    failed: its consensus stands, its membership is still the group's, and
    the support view showed only the survivor's own reads."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import rebuild
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    def fails(*_a, **_k):
        raise RuntimeError("injected rebuild failure")

    _base, demux = panel
    out = _without_final_stage(runs[1], tmp_path / "em")
    monkeypatch.setattr(rebuild, "rebuild_one", fails)
    run_em(demux, out, params=evidence_params(rounds=0), resume=True, report=False)
    monkeypatch.undo()

    nodes = R.cluster_nodes(out)
    merged = nodes.filter(
        pa.compute.greater(nodes.column("n_absorbed"), 0)
    ).to_pylist()
    assert merged and not any(r["rebuilt"] for r in merged)
    members = pq.read_table(out / "cluster_membership.parquet")
    for row in merged:
        want = members.filter(
            pa.compute.equal(members.column("cluster_id"), row["cluster_id"])
        ).column("read_id")
        rep = R.replay_cluster(out, row["cluster_id"])
        assert rep.kind == "merged" and rep.verified, rep.mismatch
        st = R.support_tables(rep)
        assert sorted(st.reads.column("read_id").to_pylist()) == sorted(want.to_pylist())
        absorbed = st.reads.filter(pa.compute.is_valid(st.reads.column("absorbed_from")))
        assert absorbed.num_rows == rep.absorbed_read_row.size > 0
        assert set(absorbed.column("absorbed_from").to_pylist()) == set(
            row["absorbed_node_template_ids"]
        )
        # Never aligned to this node's frame: listed, with no span.
        assert absorbed.column("span_lo").null_count == absorbed.num_rows


def test_support_tables_keep_the_cluster_s_verdict(runs):
    """3. A cluster's check (its consensus against the cluster's) is the
    replay's; the support tables reported the template's instead."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    nodes = R.cluster_nodes(runs[1])
    plain = nodes.filter(
        pa.compute.and_(
            pa.compute.invert(nodes.column("rebuilt")),
            pa.compute.equal(nodes.column("n_absorbed"), 0),
        )
    )
    rep = R.replay_cluster(runs[1], plain.column("cluster_id")[0].as_py())
    assert rep.kind == "node" and rep.verified
    rep.mismatch = ["the node's consensus is not the cluster's"]
    rep.verified = False
    st = R.support_tables(rep)
    assert not st.verified
    assert st.mismatch == rep.mismatch


def _plant_n(out: Path, read_row: int, position: int) -> None:
    """Replace one base of one corpus read with N, in place."""
    import os

    path = out / "corpus" / "reads.arrow"
    with pa.OSFile(str(path)) as src:
        table = pa.ipc.open_file(src).read_all()
    seqs = table.column("sequence").to_pylist()
    s = seqs[read_row]
    seqs[read_row] = s[:position] + "N" + s[position + 1 :]
    i = table.schema.get_field_index("sequence")
    table = table.set_column(
        i, table.schema.field(i), pa.array(seqs, table.schema.field(i).type)
    )
    tmp = path.with_suffix(".tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, table.schema) as w:
        w.write_table(table)
    os.replace(tmp, path)


def test_an_ambiguous_base_inside_a_read_is_not_the_reference_s(
    runs, tmp_path, split_template_id
):
    """4. An interior N is no evidence — not agreement: span + departures
    rebuilt it as the reference's base, in both coordinate spaces."""
    import re

    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R

    out = tmp_path / "copy"
    shutil.copytree(runs[1], out)
    rep = R.replay_template(out, 1, split_template_id)
    k = int(rep.result.member_pos[0])
    read_row = int(rep.result.read_row[k])
    i = rep.rows.column("read_row").to_pylist().index(read_row)
    q = int(rep.rows.column("q_start")[i].as_py())
    position = None
    for n, op in re.findall(r"(\d+)([=XIDM])", rep.rows.column("cigar")[i].as_py()):
        n = int(n)
        if op in "=M" and n >= 9:
            position = q + n // 2
            break
        if op in "=XMI":
            q += n
    assert position is not None
    _plant_n(out, read_row, position)

    rep = R.replay_template(out, 1, split_template_id)
    k = rep.result.read_row.tolist().index(read_row)
    m = rep.result.member_pos.tolist().index(k)
    proj = rep.result.internals.pooled.projections[m][0]
    n_cols = proj.match_fpos[proj.match_base >= 4]
    assert n_cols.size == 1
    for space in ("frame", "consensus"):
        st = R.support_tables(rep, space=space)
        idx = st.reads.column("read_row").to_pylist().index(read_row)
        deps = st.departures.filter(pa.compute.equal(st.departures.column("read_idx"), idx))
        assert deps.filter(pa.compute.equal(deps.column("allele"), 5)).num_rows == 1, space
        if space == "frame":
            at = deps.filter(pa.compute.equal(deps.column("allele"), 5))
            assert at.column("col").to_pylist() == [int(n_cols[0])]


def test_a_narrow_node_query_reads_no_sequence(runs, monkeypatch):
    """5. ``columns=`` was applied after the whole table — every consensus —
    had been read and sorted."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import round_io

    seen = []
    real = round_io.in_node_order

    def spy(table):
        seen.append(table.column_names)
        return real(table)

    monkeypatch.setattr(round_io, "in_node_order", spy)
    rd = _rounds(runs[1])[0]
    table = round_io.read_round_nodes(rd, columns=["parent_template_id", "n_reads"])
    assert table.column_names == ["parent_template_id", "n_reads"]
    assert seen and all("consensus" not in names for names in seen)
    full = round_io.read_round_nodes(rd)
    assert table.equals(full.select(["parent_template_id", "n_reads"]))


def test_a_round_that_never_measured_a_runner_up_says_unknown(runs, tmp_path):
    """6. Round 1's lazy walk (ORF seeding, native) stops at its first
    admitted template; the report read its null runner-ups as 'no rival'
    and called the round 0% contested."""
    from constellation.sequencing.transcriptome.cluster.denovo.em.diagnostics import (
        section_assignment_rule,
    )
    from constellation.sequencing.transcriptome.cluster.denovo.em.round_io import (
        round_runner_up_measured,
    )

    rd = _rounds(runs[1])[0]
    assert _json(rd / "round.json")["estep"]["runner_up_measured"] is False
    body = section_assignment_rule(runs[1]).body
    assert "unknown" in body and "uncontested (no runner-up template)" not in body
    # A round from before the record is judged by the run's E-step stamp.
    old = tmp_path / "old"
    shutil.copytree(runs[1], old)
    old_rd = old / "rounds" / rd.name
    record = _json(old_rd / "round.json")
    record["estep"].pop("runner_up_measured")
    (old_rd / "round.json").write_text(json.dumps(record))
    assert round_runner_up_measured(old_rd) is False
    # Later rounds do measure it.
    assert all(round_runner_up_measured(d) for d in _rounds(runs[3])[1:])


def test_a_cluster_lists_the_reads_the_closing_fold_gave_it(runs, tmp_path):
    """The final stage's closing fold joins a cluster to one it had become the
    same as, keeping the survivor's consensus (2026-11-24). Replay re-creates
    that consensus from its own pool, unchanged, and lists the folded
    cluster's reads beside it: members, from their own node, never aligned
    to the survivor's frame. Declared here on a copy of the run, since the
    panel's rebuild leaves nothing the same."""
    from constellation.sequencing.transcriptome.cluster.denovo.em import results as R
    from constellation.sequencing.transcriptome.cluster.denovo.em.outputs import (
        EM_CLUSTER_NODE_TABLE,
    )

    out = tmp_path / "copy"
    shutil.copytree(runs[1], out)
    path = out / "cluster_nodes.parquet"
    rows = pq.read_table(path).to_pylist()
    assert all(r["n_folded"] == 0 for r in rows), "the run folded nothing"
    rebuilt = next(r for r in rows if r["rebuilt"])
    plain = next(r for r in rows if not r["rebuilt"] and not r["n_absorbed"])
    rebuilt["n_folded"] = 1
    rebuilt["folded_node_template_ids"] = [plain["node_template_id"]]
    pq.write_table(pa.Table.from_pylist(rows, schema=EM_CLUSTER_NODE_TABLE), path)

    members = pq.read_table(out / "cluster_membership.parquet")

    def reads_of(cid):
        return set(
            members.filter(pa.compute.equal(members.column("cluster_id"), cid))
            .column("read_id")
            .to_pylist()
        )

    rep = R.replay_cluster(out, rebuilt["cluster_id"])
    assert rep.kind == "rebuilt" and rep.verified, rep.mismatch
    assert set(rep.folded_read_ids.to_pylist()) == reads_of(plain["cluster_id"])
    assert set(rep.folded_node_template_id.tolist()) == {plain["node_template_id"]}
    st = R.support_tables(rep)
    got = st.reads.column("read_id").to_pylist()
    assert len(got) == len(set(got))
    assert set(got) == reads_of(rebuilt["cluster_id"]) | reads_of(plain["cluster_id"])
    folded = st.reads.filter(
        pa.compute.equal(st.reads.column("absorbed_from"), plain["node_template_id"])
    )
    assert set(folded.column("read_id").to_pylist()) == reads_of(plain["cluster_id"])
    assert any("folded into it" in w for w in st.warnings)
    assert R.replay_summary(rep)["n_folded_reads"] == len(reads_of(plain["cluster_id"]))
