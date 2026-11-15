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
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
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
def runs(tmp_path_factory):
    """``{1: out_dir, 3: out_dir}`` — the one-round and three-round runs."""
    base = tmp_path_factory.mktemp("em_evidence")
    demux = three_gene_panel(base)
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
