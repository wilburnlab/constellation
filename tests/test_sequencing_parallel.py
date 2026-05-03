"""Tests for ``constellation.sequencing.parallel.run_batched`` and the
transcriptome stage orchestrator.

Synthetic-fixture invariants (no lab data needed):

    - row-count preservation across n_workers ∈ {1, 2, 4}
    - identical-output invariant: 1-worker and N-worker runs concat to
      identical tables (up to row order, which is preserved by the
      shard-index sort in ``parallel.py``)
    - stage-level _SUCCESS marker short-circuit
    - shard-level partial recovery: delete one shard, rerun, only
      that shard regenerates

Plus an integration test against ``test_simplex.sam`` (lab fixture)
that asserts ``--threads 4`` produces output equivalent to ``--threads
1``; marked ``slow`` so CI without the fixture skips it.
"""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from constellation.sequencing.parallel import (
    StageOutput,
    read_dataset,
    run_batched,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic worker — must be at module top level so multiprocessing
# can pickle it.
# ──────────────────────────────────────────────────────────────────────


def _double_table_worker(batch: pa.Table) -> dict[str, pa.Table]:
    """Worker that doubles every value in column 'x' and emits the
    result under output key 'doubled'."""
    doubled = pa.compute.multiply(batch.column("x"), 2)
    out = pa.table({"x": doubled})
    return {"doubled": out}


def _multi_output_worker(batch: pa.Table) -> dict[str, pa.Table]:
    """Worker emitting two output tables — exercises the multi-key
    shard layout."""
    return {
        "doubled": pa.table({"x": pa.compute.multiply(batch.column("x"), 2)}),
        "tripled": pa.table({"x": pa.compute.multiply(batch.column("x"), 3)}),
    }


def _make_batches(n_batches: int, rows_per_batch: int) -> list[pa.Table]:
    """Synthesize ``n_batches`` Arrow tables, each with ``rows_per_batch``
    rows of ``x`` = sequential integers."""
    batches = []
    val = 0
    for _ in range(n_batches):
        col = list(range(val, val + rows_per_batch))
        batches.append(pa.table({"x": pa.array(col, type=pa.int64())}))
        val += rows_per_batch
    return batches


# ──────────────────────────────────────────────────────────────────────
# Single-process correctness
# ──────────────────────────────────────────────────────────────────────


def test_single_worker_basic(tmp_path: pathlib.Path) -> None:
    batches = _make_batches(4, 10)
    result = run_batched(
        worker_fn=_double_table_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled",),
        n_workers=1,
    )
    assert "doubled" in result
    out = result["doubled"]
    assert isinstance(out, StageOutput)
    assert len(out.shard_paths) == 4
    # Every shard exists and contains the expected doubled values.
    for shard in out.shard_paths:
        assert shard.is_file() and shard.stat().st_size > 0
    # Concatenated dataset has expected row count + values.
    concat = read_dataset(out)
    assert concat.num_rows == 40
    assert concat.column("x").to_pylist() == [2 * i for i in range(40)]


def test_success_marker_written(tmp_path: pathlib.Path) -> None:
    batches = _make_batches(2, 5)
    run_batched(
        worker_fn=_double_table_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled",),
        n_workers=1,
    )
    success = tmp_path / "doubled" / "_SUCCESS"
    assert success.is_file()


def test_multi_output_keys(tmp_path: pathlib.Path) -> None:
    batches = _make_batches(3, 5)
    result = run_batched(
        worker_fn=_multi_output_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled", "tripled"),
        n_workers=1,
    )
    assert set(result.keys()) == {"doubled", "tripled"}
    d = read_dataset(result["doubled"])
    t = read_dataset(result["tripled"])
    assert d.column("x").to_pylist() == [2 * i for i in range(15)]
    assert t.column("x").to_pylist() == [3 * i for i in range(15)]


def test_invalid_n_workers_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="n_workers must be"):
        run_batched(
            worker_fn=_double_table_worker,
            batches=[],
            output_dir=tmp_path,
            output_keys=("doubled",),
            n_workers=0,
        )


# ──────────────────────────────────────────────────────────────────────
# Multi-process correctness (4 workers vs 1 worker should be identical)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_workers", [2, 4])
def test_multi_worker_output_matches_single(
    tmp_path: pathlib.Path, n_workers: int
) -> None:
    """N-worker concat should equal single-worker concat row-for-row."""
    batches_a = _make_batches(6, 7)
    batches_b = _make_batches(6, 7)
    out_dir_a = tmp_path / "single"
    out_dir_b = tmp_path / "parallel"

    res_a = run_batched(
        worker_fn=_double_table_worker,
        batches=batches_a,
        output_dir=out_dir_a,
        output_keys=("doubled",),
        n_workers=1,
    )
    res_b = run_batched(
        worker_fn=_double_table_worker,
        batches=batches_b,
        output_dir=out_dir_b,
        output_keys=("doubled",),
        n_workers=n_workers,
    )
    a = read_dataset(res_a["doubled"]).to_pylist()
    b = read_dataset(res_b["doubled"]).to_pylist()
    assert a == b


# ──────────────────────────────────────────────────────────────────────
# Resume / partial recovery
# ──────────────────────────────────────────────────────────────────────


def test_resume_short_circuits_completed_stage(tmp_path: pathlib.Path) -> None:
    """Second call with resume=True should not re-invoke worker_fn —
    we sentinel that by having the second-call worker raise."""
    batches = _make_batches(3, 5)
    run_batched(
        worker_fn=_double_table_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled",),
        n_workers=1,
    )

    def _bad_worker(_b: pa.Table) -> dict[str, pa.Table]:
        raise AssertionError("worker should not have been called on resume")

    # Should NOT raise; resume short-circuits before dispatch.
    result = run_batched(
        worker_fn=_bad_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled",),
        n_workers=1,
        resume=True,
    )
    out = read_dataset(result["doubled"])
    assert out.num_rows == 15


def test_partial_recovery_reruns_only_missing_shards(
    tmp_path: pathlib.Path,
) -> None:
    """Delete the _SUCCESS marker AND one shard; rerun should only
    regenerate that one shard."""
    batches = _make_batches(4, 5)
    run_batched(
        worker_fn=_double_table_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled",),
        n_workers=1,
    )

    # Wipe the _SUCCESS marker (simulating an interrupted run) and
    # one shard.
    stage_dir = tmp_path / "doubled"
    (stage_dir / "_SUCCESS").unlink()
    (stage_dir / "part-00002.parquet").unlink()

    # Track which batch indices the worker is invoked on.
    invocations: list[int] = []

    def _tracking_worker(batch: pa.Table) -> dict[str, pa.Table]:
        # The first row's value tells us which batch this is —
        # batches were generated sequentially (val starts at idx*5).
        first_x = batch.column("x")[0].as_py()
        invocations.append(first_x // 5)
        return _double_table_worker(batch)

    run_batched(
        worker_fn=_tracking_worker,
        batches=batches,
        output_dir=tmp_path,
        output_keys=("doubled",),
        n_workers=1,
        resume=True,
    )

    # Only batch index 2 should have been re-run.
    assert invocations == [2]


# ──────────────────────────────────────────────────────────────────────
# Integration: full demux pipeline at --threads 4 vs --threads 1
# ──────────────────────────────────────────────────────────────────────


_FIXTURE_SAM = pathlib.Path(
    "/home/dbwilburn/WilburnLab/Coding/Constellation/test_nanopore_data/"
    "pod5_cDNA/Constellation_test_data/test_simplex.sam"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not _FIXTURE_SAM.exists(),
    reason="lab test_simplex.sam fixture not present",
)
def test_demux_pipeline_threads_4_matches_threads_1(tmp_path: pathlib.Path) -> None:
    """End-to-end demux at threads=4 should produce identical output
    to threads=1 — both for the count matrix and the per-read demux
    table."""
    import pyarrow.parquet as pq

    from constellation.sequencing.samples import Samples
    from constellation.sequencing.transcriptome.stages import (
        run_demux_pipeline,
    )

    samples = Samples.from_records(
        samples=[
            {
                "sample_id": i + 1,
                "sample_name": f"test_simplex_BC{i + 1:02d}",
                "description": None,
            }
            for i in range(12)
        ],
        edges=[
            {"sample_id": i + 1, "acquisition_id": 1, "barcode_id": i}
            for i in range(12)
        ],
    )

    art_a = run_demux_pipeline(
        _FIXTURE_SAM,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=tmp_path / "single",
        batch_size=1000,
        n_workers=1,
    )
    art_b = run_demux_pipeline(
        _FIXTURE_SAM,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=tmp_path / "parallel",
        batch_size=1000,
        n_workers=4,
    )

    # Demux table row count + content match.
    import pyarrow.dataset as ds
    a_demux = ds.dataset(tmp_path / "single" / "read_demux", format="parquet").to_table().sort_by(
        "read_id"
    )
    b_demux = ds.dataset(tmp_path / "parallel" / "read_demux", format="parquet").to_table().sort_by(
        "read_id"
    )
    assert a_demux.num_rows == b_demux.num_rows == 4009
    assert a_demux.to_pylist() == b_demux.to_pylist()

    # TSV (count matrix) byte-equivalent.
    a_tsv = (tmp_path / "single" / "protein_counts.tsv").read_text()
    b_tsv = (tmp_path / "parallel" / "protein_counts.tsv").read_text()
    assert a_tsv == b_tsv

    # FASTA byte-equivalent.
    a_fasta = (tmp_path / "single" / "proteins.fasta").read_text()
    b_fasta = (tmp_path / "parallel" / "proteins.fasta").read_text()
    assert a_fasta == b_fasta
