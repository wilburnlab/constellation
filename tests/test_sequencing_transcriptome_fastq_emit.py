"""Tests for ``constellation.sequencing.transcriptome.fastq``.

Three groups:
  * Helper unit tests (sanitiser collision detection) — fast, no I/O.
  * End-to-end synthetic pipeline test — runs the full
    ``run_demux_pipeline`` against a tiny simulator-built SAM with
    ``emit_fastq=True`` and verifies the per-sample ``.fq.gz`` files
    against the ``read_demux`` parquet ground truth.
  * Resume + bolt-on tests — re-run with the ``_SUCCESS`` marker
    present, and the "user adds --emit-fastq later" workflow.
"""

from __future__ import annotations

import gzip
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest

from constellation.sequencing.samples import Samples
from constellation.sequencing.transcriptome.designs import CDNA_WILBURN_V1
from constellation.sequencing.transcriptome.fastq import (
    _build_sample_filename_map,
    _sanitize_name,
    emit_per_sample_fastq,
)
from constellation.sequencing.transcriptome.simulator import (
    generate_stress_test_specs,
    is_deterministic_clean,
    simulate_panel,
)


# ──────────────────────────────────────────────────────────────────────
# Helper unit tests
# ──────────────────────────────────────────────────────────────────────


def test_sanitize_name_replaces_unsafe_chars() -> None:
    assert _sanitize_name("BC01") == "BC01"
    assert _sanitize_name("sample_1") == "sample_1"
    assert _sanitize_name("sample.A-2") == "sample.A-2"
    assert _sanitize_name("path/like") == "path_like"
    assert _sanitize_name("with spaces") == "with_spaces"
    assert _sanitize_name("weird*name?") == "weird_name_"


def test_build_sample_filename_map_rejects_collisions() -> None:
    samples = Samples.from_records(
        samples=[
            {"sample_id": 1, "sample_name": "a/b", "description": None},
            {"sample_id": 2, "sample_name": "a_b", "description": None},
        ],
        edges=[
            {"sample_id": 1, "acquisition_id": 1, "barcode_id": 0},
            {"sample_id": 2, "acquisition_id": 1, "barcode_id": 1},
        ],
    )
    with pytest.raises(ValueError, match="colliding filenames"):
        _build_sample_filename_map(samples)


def test_build_sample_filename_map_allows_unique() -> None:
    samples = Samples.from_records(
        samples=[
            {"sample_id": 1, "sample_name": "sample_one", "description": None},
            {"sample_id": 2, "sample_name": "sample-two", "description": None},
        ],
        edges=[
            {"sample_id": 1, "acquisition_id": 1, "barcode_id": 0},
            {"sample_id": 2, "acquisition_id": 1, "barcode_id": 1},
        ],
    )
    mapping = _build_sample_filename_map(samples)
    assert mapping == {1: "sample_one", 2: "sample-two"}


# ──────────────────────────────────────────────────────────────────────
# End-to-end pipeline test
# ──────────────────────────────────────────────────────────────────────


def _build_pipeline_inputs(tmp_path: Path) -> tuple[Path, Samples]:
    """Stand up a tiny simulator-built SAM + matching Samples object."""
    all_specs = generate_stress_test_specs(
        CDNA_WILBURN_V1, n_per_category=2, seed=42
    )
    clean = [s for s in all_specs if is_deterministic_clean(s)]
    assert len(clean) >= 10, f"clean subset too small ({len(clean)})"

    sim_dir = tmp_path / "simulator"
    sam_path = sim_dir / "synthetic.sam"
    gt_path = sim_dir / "ground_truth.parquet"
    simulate_panel(
        clean, CDNA_WILBURN_V1, sam_path=sam_path, ground_truth_path=gt_path
    )

    n_barcodes = len(CDNA_WILBURN_V1.layout[3].barcodes)
    samples = Samples.from_records(
        samples=[
            {
                "sample_id": i + 1,
                "sample_name": f"BC{i + 1:02d}",
                "description": None,
            }
            for i in range(n_barcodes)
        ],
        edges=[
            {"sample_id": i + 1, "acquisition_id": 1, "barcode_id": i}
            for i in range(n_barcodes)
        ],
    )
    return sam_path, samples


def _expected_per_sample_counts(demux_dir: Path) -> dict[int, int]:
    """Per-sample Complete-read counts from the on-disk read_demux dataset."""
    filt = (
        (pc.field("status") == "Complete")
        & pc.is_valid(pc.field("sample_id"))
        & pc.invert(pc.field("is_fragment"))
        & pc.invert(pc.field("is_chimera"))
        & (pc.field("transcript_start") >= 0)
        & (pc.field("transcript_end") > pc.field("transcript_start"))
    )
    table = ds.dataset(demux_dir / "read_demux", format="parquet").to_table(
        columns=["sample_id"], filter=filt
    )
    counts: dict[int, int] = {}
    for sid in table.column("sample_id").to_pylist():
        counts[sid] = counts.get(sid, 0) + 1
    return counts


def _decode_fastq(path: Path) -> list[tuple[str, str, str]]:
    """Decode a .fq.gz file to a list of (read_id, sequence, quality)."""
    out: list[tuple[str, str, str]] = []
    with gzip.open(path, "rt", encoding="ascii") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().rstrip("\n")
            plus = fh.readline()
            qual = fh.readline().rstrip("\n")
            assert header.startswith("@"), f"bad FASTQ header: {header!r}"
            assert plus.startswith("+"), f"bad FASTQ plus line: {plus!r}"
            assert len(seq) == len(qual), (
                f"seq/qual length mismatch in {path}: {len(seq)} vs {len(qual)}"
            )
            out.append((header[1:].rstrip("\n"), seq, qual))
    return out


def test_emit_per_sample_fastq_synthetic(tmp_path: Path) -> None:
    """End-to-end: run pipeline with emit_fastq=True, validate every file.

    Checks (1) ``fastq/_SUCCESS`` exists, (2) one ``.fq.gz`` per sample
    that received any Complete read, (3) total read count across files
    equals the filtered ``read_demux`` count, (4) per-file read count
    matches the per-sample filtered count, (5) each emitted sequence
    matches ``READ_TABLE.sequence[transcript_start:transcript_end]``
    for that read_id.
    """
    from constellation.sequencing.transcriptome.stages import run_demux_pipeline

    sam_path, samples = _build_pipeline_inputs(tmp_path)
    out_dir = tmp_path / "demux"
    artefacts = run_demux_pipeline(
        sam_path,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=out_dir,
        batch_size=1000,
        n_workers=1,
        emit_fastq=True,
    )

    assert artefacts["fastq_dir"] == out_dir / "fastq"
    fastq_dir = out_dir / "fastq"
    assert (fastq_dir / "_SUCCESS").is_file()

    expected = _expected_per_sample_counts(out_dir)
    # Pull the join of read_demux + reads from disk so we can verify
    # each emitted (read_id, sequence, quality) triple end-to-end.
    reads_table = ds.dataset(out_dir / "reads", format="parquet").to_table(
        columns=["read_id", "sequence", "quality"]
    )
    seq_by_read = dict(
        zip(
            reads_table.column("read_id").to_pylist(),
            reads_table.column("sequence").to_pylist(),
        )
    )
    qual_by_read = dict(
        zip(
            reads_table.column("read_id").to_pylist(),
            reads_table.column("quality").to_pylist(),
        )
    )
    demux_table = ds.dataset(out_dir / "read_demux", format="parquet").to_table(
        columns=[
            "read_id",
            "sample_id",
            "transcript_start",
            "transcript_end",
            "status",
            "is_fragment",
            "is_chimera",
        ]
    )
    demux_by_read = {r["read_id"]: r for r in demux_table.to_pylist()}

    id_to_name = dict(
        zip(
            samples.samples.column("sample_id").to_pylist(),
            samples.samples.column("sample_name").to_pylist(),
        )
    )

    total_emitted = 0
    for sid, n_expected in expected.items():
        path = fastq_dir / f"{id_to_name[sid]}.fq.gz"
        assert path.is_file(), f"missing emitted FASTQ for sample {sid}: {path}"
        records = _decode_fastq(path)
        assert len(records) == n_expected, (
            f"sample {sid}: emitted {len(records)} reads, expected {n_expected}"
        )
        total_emitted += len(records)
        for read_id, seq, qual in records:
            d = demux_by_read[read_id]
            ts, te = d["transcript_start"], d["transcript_end"]
            assert d["sample_id"] == sid
            assert d["status"] == "Complete"
            assert not d["is_fragment"]
            assert not d["is_chimera"]
            assert seq == seq_by_read[read_id][ts:te], (
                f"sequence mismatch on {read_id}"
            )
            assert qual == qual_by_read[read_id][ts:te], (
                f"quality mismatch on {read_id}"
            )

    assert total_emitted == sum(expected.values())

    # Samples with zero Complete reads should NOT produce a file.
    sample_ids_with_reads = set(expected)
    for sid, name in id_to_name.items():
        if sid not in sample_ids_with_reads:
            assert not (fastq_dir / f"{name}.fq.gz").exists(), (
                f"unexpected empty file for sample {sid}"
            )


def test_emit_fastq_resume_skips(tmp_path: Path) -> None:
    """Re-running with resume=True + an existing _SUCCESS skips work.

    After a clean emission, the per-sample file mtimes must NOT change
    on the second invocation.
    """
    from constellation.sequencing.transcriptome.stages import run_demux_pipeline

    sam_path, samples = _build_pipeline_inputs(tmp_path)
    out_dir = tmp_path / "demux"
    run_demux_pipeline(
        sam_path,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=out_dir,
        batch_size=1000,
        n_workers=1,
        emit_fastq=True,
    )

    fastq_dir = out_dir / "fastq"
    before = {
        p.name: p.stat().st_mtime_ns for p in fastq_dir.glob("*.fq.gz")
    }
    success_mtime = (fastq_dir / "_SUCCESS").stat().st_mtime_ns
    assert before, "first run should have produced at least one fastq"

    # Sleep a hair so any rewrite would actually move the mtime.
    time.sleep(0.05)

    emit_per_sample_fastq(out_dir, samples=samples, resume=True)

    after = {p.name: p.stat().st_mtime_ns for p in fastq_dir.glob("*.fq.gz")}
    assert after == before
    assert (fastq_dir / "_SUCCESS").stat().st_mtime_ns == success_mtime


def test_emit_fastq_bolt_on_via_resume(tmp_path: Path) -> None:
    """The headline UX: first run without --emit-fastq, second run with
    --emit-fastq + --resume only executes the new emission stage.

    Verifies: the second invocation does not rewrite the demux shards
    (their mtimes are unchanged), but the ``fastq/`` directory now
    exists with the expected per-sample files.
    """
    from constellation.sequencing.transcriptome.stages import run_demux_pipeline

    sam_path, samples = _build_pipeline_inputs(tmp_path)
    out_dir = tmp_path / "demux"

    # First run — no --emit-fastq.
    run_demux_pipeline(
        sam_path,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=out_dir,
        batch_size=1000,
        n_workers=1,
    )
    assert not (out_dir / "fastq").exists()
    demux_mtimes_before = {
        p.name: p.stat().st_mtime_ns
        for p in (out_dir / "read_demux").glob("*.parquet")
    }
    reads_mtimes_before = {
        p.name: p.stat().st_mtime_ns
        for p in (out_dir / "reads").glob("*.parquet")
    }
    assert demux_mtimes_before and reads_mtimes_before

    time.sleep(0.05)

    # Second run — same command + --emit-fastq + --resume.
    run_demux_pipeline(
        sam_path,
        library_design="cdna_wilburn_v1",
        samples=samples,
        acquisition_id=1,
        output_dir=out_dir,
        batch_size=1000,
        n_workers=1,
        resume=True,
        emit_fastq=True,
    )

    # Demux shards untouched.
    demux_mtimes_after = {
        p.name: p.stat().st_mtime_ns
        for p in (out_dir / "read_demux").glob("*.parquet")
    }
    reads_mtimes_after = {
        p.name: p.stat().st_mtime_ns
        for p in (out_dir / "reads").glob("*.parquet")
    }
    assert demux_mtimes_after == demux_mtimes_before
    assert reads_mtimes_after == reads_mtimes_before

    # Fastq emission ran.
    fastq_dir = out_dir / "fastq"
    assert (fastq_dir / "_SUCCESS").is_file()
    assert list(fastq_dir.glob("*.fq.gz")), "no fastq files emitted"
