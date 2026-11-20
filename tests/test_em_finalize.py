"""The EM run's final stage on its own, and the cluster table past 2 GiB of sequence.

Two defects of the first 9.4M-read run with the evidence contract (2026-11-18): the cluster table cast the final
nodes' ``large_string`` consensus to ``string`` in one piece, which Arrow refuses past 2 GiB of characters (and the
final rebuild combines that column into one chunk); and a run whose rounds had all finished but whose final stage
had not could not be finished at all — ``--resume --rounds 0`` wrote nothing, ``--rounds 1`` ran an extra round.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _em_fixtures import three_gene_panel  # noqa: E402

from constellation.sequencing.transcriptome.cluster.denovo.em import outputs  # noqa: E402
from constellation.sequencing.transcriptome.cluster.denovo.em.rounds import (  # noqa: E402
    EmParams,
    run_em,
)


def _params(**kw) -> EmParams:
    base = dict(
        threads=1,
        mstep_workers=1,
        max_window_length=None,
        min_aa_length=30,
        estep_aligner="native",
    )
    base.update(kw)
    return EmParams(**base)


# ── _as_string ──────────────────────────────────────────────────────────


def test_as_string_cuts_a_large_string_chunk_under_the_offset_limit(monkeypatch):
    monkeypatch.setattr(outputs, "_STRING_CHUNK_BYTES", 10)
    values = ["ACGT", None, "", "ACGTACGTAC", "A" * 3, "CC", None, "GGGGGGG", "T"]
    col = pa.chunked_array([pa.array(values[:4], pa.large_string()),
                            pa.array(values[4:], pa.large_string())])
    out = outputs._as_string(col)
    assert out.type == pa.string()
    assert out.to_pylist() == values
    assert out.null_count == 2
    # No chunk holds more than the limit of characters, and the input was cut.
    for chunk in out.chunks:
        assert sum(len(v or "") for v in chunk.to_pylist()) <= 10
    assert out.num_chunks > col.num_chunks


def test_as_string_respects_a_sliced_array_and_an_oversized_string(monkeypatch):
    monkeypatch.setattr(outputs, "_STRING_CHUNK_BYTES", 5)
    base = pa.array(["XX", "ACG", "TTTT", "G", "CCCCCC", "A"], pa.large_string())
    sliced = base.slice(1, 4)  # ACG, TTTT, G, CCCCCC — offsets do not start at 0
    out = outputs._as_string(sliced)
    assert out.to_pylist() == ["ACG", "TTTT", "G", "CCCCCC"]
    # A single string longer than the limit is its own chunk (Arrow decides
    # whether that is castable), never an endless loop.
    assert ["CCCCCC"] in [c.to_pylist() for c in out.chunks]


def test_as_string_passes_empty_and_plain_strings_through():
    assert outputs._as_string(pa.chunked_array([], pa.large_string())).to_pylist() == []
    out = outputs._as_string(pa.array(["A", None], pa.string()))
    assert out.type == pa.string() and out.to_pylist() == ["A", None]


# ── --resume --rounds 0 ─────────────────────────────────────────────────

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


@pytest.fixture(scope="module")
def finished_runs(tmp_path_factory):
    """``{1: dir, 3: dir}``: one round (its final merge rebuilds a survivor, the
    path that combines the consensus column) and three (with history)."""
    base = tmp_path_factory.mktemp("em_finalize")
    demux = three_gene_panel(base)
    out = {}
    for rounds in (1, 3):
        out[rounds] = base / f"em{rounds}"
        run_em(demux, out[rounds], params=_params(rounds=rounds), report=False)
    return demux, out


def _snapshot(out: Path) -> dict[str, pa.Table]:
    return {
        name: pq.read_table(out / name)
        for name in ("clusters.parquet", "cluster_membership.parquet",
                     "cluster_nodes.parquet", "feature_quant.parquet")
        if (out / name).exists()
    }


def test_the_one_round_fixture_rebuilds_a_final_survivor(finished_runs):
    _demux, runs = finished_runs
    rebuilt = pq.read_table(runs[1] / "cluster_nodes.parquet").column("rebuilt").to_pylist()
    assert rebuilt.count(True) >= 1, "no rebuilt survivor: the combined-column path is not exercised"


@pytest.mark.parametrize("rounds", [1, 3])
def test_finalize_only_rebuilds_the_final_stage_exactly(finished_runs, tmp_path, rounds):
    demux, runs = finished_runs
    done = runs[rounds]
    want = _snapshot(done)
    assert "clusters.parquet" in want and want["clusters.parquet"].num_rows

    # The run as the 9.4M one was left: every round finished, the final stage
    # gone — its outputs and its records in the last round.
    work = tmp_path / "em"
    shutil.copytree(done, work)
    for name in _FINAL_OUTPUTS:
        (work / name).unlink(missing_ok=True)
    last = sorted((work / "rounds").glob("r*/_SUCCESS"))[-1].parent
    for name in ("merged_final.parquet", "final.json"):
        (last / name).unlink(missing_ok=True)
    shutil.rmtree(last / "graph_final", ignore_errors=True)
    rounds_before = sorted(p.name for p in (work / "rounds").iterdir() if p.is_dir())
    lineage_before = (last / "lineage.parquet").exists()

    results = run_em(demux, work, params=_params(rounds=0), resume=True, report=False)

    # The last FINISHED round (the three-round fixture converges at two).
    last_round = int(last.name[1:])
    assert [r.round_index for r in results] == [last_round]
    got = _snapshot(work)
    assert set(got) == set(want)
    for name, table in want.items():
        assert got[name].equals(table), name
    assert (work / "cluster.fa").read_text() == (done / "cluster.fa").read_text()
    # No extra round, and the last round was not refined into a successor.
    assert sorted(p.name for p in (work / "rounds").iterdir() if p.is_dir()) == rounds_before
    assert (last / "lineage.parquet").exists() == lineage_before
    import json

    assert json.loads((work / "manifest.json").read_text())["stages"]["n_rounds"] == last_round


def test_finalize_only_refuses_a_run_with_no_finished_round(tmp_path):
    demux = three_gene_panel(tmp_path)
    with pytest.raises(ValueError, match="rounds 0"):
        run_em(demux, tmp_path / "empty", params=_params(rounds=0), resume=True, report=False)
