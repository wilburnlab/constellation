"""Unit tests for the Stage-3 assembly knobs' pure helpers.

``_resolve_threads`` (CLI thread auto-detect), ``_resolve_scratch_root`` (where
the throwaway reads FASTQ lives), and ``_rel_or_abs`` (manifest path that must
not raise when the FASTQ is off-tree on node-local scratch).
"""

from __future__ import annotations

from pathlib import Path

from constellation.cli.__main__ import _resolve_threads
from constellation.sequencing.assembly.pipeline import (
    _rel_or_abs,
    _resolve_scratch_root,
)


# ── _resolve_threads ────────────────────────────────────────────────────────


def test_resolve_threads_explicit_wins(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert _resolve_threads(4) == 4


def test_resolve_threads_clamps_to_one(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert _resolve_threads(0) == 1


def test_resolve_threads_reads_slurm_cpus_per_task(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    assert _resolve_threads(None) == 12


def test_resolve_threads_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    assert _resolve_threads(None) >= 1  # os.cpu_count() or 1


# ── _resolve_scratch_root ───────────────────────────────────────────────────


def test_scratch_root_default_is_output_dir(tmp_path):
    out = tmp_path / "out"
    assert _resolve_scratch_root(None, out) == out


def test_scratch_root_explicit_path_gets_job_subdir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    out = tmp_path / "out"
    scratch = tmp_path / "node_local"
    assert _resolve_scratch_root(scratch, out) == scratch / "constellation_asm_777"


def test_scratch_root_auto_uses_tmpdir(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    out = tmp_path / "out"
    assert _resolve_scratch_root("auto", out) == tmp_path / "tmp" / "constellation_asm_9"


def test_scratch_root_auto_without_env_stays_in_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    out = tmp_path / "out"
    assert _resolve_scratch_root("auto", out) == out


def test_scratch_root_equal_to_output_dir_has_no_subdir(tmp_path):
    out = tmp_path / "out"
    assert _resolve_scratch_root(out, out) == out


# ── _rel_or_abs ─────────────────────────────────────────────────────────────


def test_rel_or_abs_in_tree_is_relative(tmp_path):
    base = tmp_path / "out"
    assert _rel_or_abs(base / "reads" / "reads.fastq.gz", base) == "reads/reads.fastq.gz"


def test_rel_or_abs_off_tree_is_absolute(tmp_path):
    base = tmp_path / "out"
    off = tmp_path / "scratch" / "reads.fastq"
    assert _rel_or_abs(off, base) == str(off)
