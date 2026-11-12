"""DoradoRunner + RunHandle — via a mock ``dorado`` binary.

A bash stub resolved through ``$CONSTELLATION_DORADO_HOME`` stands in for
Dorado: it writes mock output to stdout, prints a progress line to
stderr, and records its argv. Exercises argv construction, the
foreground + detached RunHandle lifecycle, attach, and tail_progress —
without the real (multi-day, GPU) basecaller.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from _dorado_stub import install, parse_dorado_argv

from constellation.sequencing.basecall.dorado import DoradoRunner, RunHandle, RunStatus
from constellation.sequencing.basecall.models import DoradoModel


@pytest.fixture
def mock_dorado(tmp_path: Path, monkeypatch):
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available for the mock binary")
    return install(tmp_path, monkeypatch)


def test_basecaller_foreground(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    handle = DoradoRunner(device="cpu").basecaller(model, [tmp_path / "r.pod5"], out)
    assert handle.wait() == 0
    assert handle.poll() == RunStatus.COMPLETED
    assert out.read_bytes().startswith(b"MOCKBAM")
    assert handle.stderr_log.exists()
    verb, positionals, options = parse_dorado_argv(mock_dorado.argv())
    assert verb == "basecaller"
    assert options["--device"] == ["cpu"]
    assert positionals == [model.model_name(), str(tmp_path / "r.pod5")]


def test_basecaller_modified_bases_from_model(tmp_path: Path, mock_dorado):
    # `--modified-bases` reads a list up to the next option. Before the
    # positionals, with no `--emit-moves` after it (the library default),
    # it took the model and the POD5 path as modification codes and Dorado
    # was left with no model at all.
    model = DoradoModel.parse("hac@v4.3+5mC,5hmC")
    out = tmp_path / "calls.bam"
    pod5 = [tmp_path / "a.pod5", tmp_path / "b.pod5"]
    DoradoRunner(device="cpu").basecaller(model, pod5, out).wait()
    _, positionals, options = parse_dorado_argv(mock_dorado.argv())
    assert positionals == [model.model_name(), *(str(p) for p in pod5)]
    assert options["--modified-bases"] == ["5mC", "5hmC"]


def test_basecaller_modified_bases_after_extra_args(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    DoradoRunner(
        device="cpu", extra_args=("--models-directory", str(tmp_path / "m"))
    ).basecaller(
        model, [tmp_path / "r.pod5"], out,
        modified_bases=("5mCG_5hmCG",), emit_moves=True,
    ).wait()
    _, positionals, options = parse_dorado_argv(mock_dorado.argv())
    assert positionals == [model.model_name(), str(tmp_path / "r.pod5")]
    assert options["--modified-bases"] == ["5mCG_5hmCG"]
    assert "--emit-moves" in options
    assert options["--models-directory"] == [str(tmp_path / "m")]


def test_basecaller_emit_moves_flag(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    DoradoRunner(device="cpu").basecaller(
        model, [tmp_path / "r.pod5"], out, emit_moves=True
    ).wait()
    _, _, options = parse_dorado_argv(mock_dorado.argv())
    assert "--emit-moves" in options


def test_basecaller_detach_and_attach(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    handle = DoradoRunner(device="cpu").basecaller(
        model, [tmp_path / "r.pod5"], out, detach=True
    )
    assert handle.pid_file.exists()
    assert handle.wait(timeout=15) == 0
    assert out.read_bytes().startswith(b"MOCKBAM")
    # a fresh session can rebind from the PID file
    rebound = RunHandle.attach(handle.pid_file)
    assert rebound.poll() == RunStatus.COMPLETED


def test_tail_progress_yields_raw_lines(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    handle = DoradoRunner(device="cpu").basecaller(model, [tmp_path / "r.pod5"], out)
    handle.wait()
    events = list(handle.tail_progress())
    assert any("basecalling" in e.raw for e in events)


def test_aligner_and_summary(tmp_path: Path, mock_dorado):
    runner = DoradoRunner(device="cpu")
    aligned = tmp_path / "aligned.bam"
    runner.aligner(tmp_path / "ref.fa", tmp_path / "reads.bam", aligned).wait()
    assert aligned.read_bytes().startswith(b"MOCKALN")

    summary = tmp_path / "summary.tsv"
    runner.summary(tmp_path / "reads.bam", summary).wait()
    assert summary.read_text().startswith("read_id")


def test_fetch_model_returns_dir(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    models_dir = tmp_path / "models"
    resolved = DoradoRunner().fetch_model(model, models_dir=models_dir)
    assert resolved == models_dir / model.model_name()


def test_basecaller_resume_moves_partial(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    out.write_bytes(b"PARTIAL")  # simulate a killed prior run
    DoradoRunner(device="cpu").basecaller(
        model, [tmp_path / "r.pod5"], out, resume=True
    ).wait()
    _, _, options = parse_dorado_argv(mock_dorado.argv())
    assert options["--resume-from"] == [str(tmp_path / "calls.bam.partial")]
    assert (tmp_path / "calls.bam.partial").exists()


def test_resolve_dorado_rejects_pre_2_0(tmp_path: Path, monkeypatch):
    # A dorado that reports a pre-2.0.0 version must be refused — the
    # pipeline carries no backwards compatibility for older Dorado.
    from constellation.sequencing.basecall.dorado_run import (
        DoradoVersionError,
        resolve_dorado,
    )

    home = tmp_path / "old_dorado"
    (home / "bin").mkdir(parents=True)
    stub = home / "bin" / "dorado"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "${1:-}" in --version) echo "dorado 1.4.0+old"; exit 0 ;; *) exit 0 ;; esac\n'
    )
    stub.chmod(0o755)
    monkeypatch.setenv("CONSTELLATION_DORADO_HOME", str(home))
    with pytest.raises(DoradoVersionError, match="too old"):
        resolve_dorado()


def test_resolve_dorado_accepts_2_0(tmp_path: Path, mock_dorado):
    # The mock reports 2.0.0+mock → resolves without raising.
    from constellation.sequencing.basecall.dorado_run import resolve_dorado

    assert resolve_dorado().name == "dorado"
