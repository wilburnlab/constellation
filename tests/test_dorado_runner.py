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

from constellation.sequencing.basecall.dorado import DoradoRunner, RunHandle, RunStatus
from constellation.sequencing.basecall.models import DoradoModel

_STUB = r"""#!/usr/bin/env bash
verb="${1:-}"
case "$verb" in
  --version) echo "dorado 2.0.0+mock"; exit 0 ;;
  download) exit 0 ;;
esac
if [[ -n "${MOCK_ARGV_FILE:-}" ]]; then
  printf '%s\n' "$*" > "${MOCK_ARGV_FILE}"
fi
case "$verb" in
  basecaller|duplex) printf 'MOCKBAM'; echo "[info] basecalling 50% done" >&2; exit 0 ;;
  aligner) printf 'MOCKALN'; exit 0 ;;
  summary) printf 'read_id\tn\nr1\t1\n'; exit 0 ;;
  demux) exit 0 ;;
  *) printf 'MOCKOUT'; exit 0 ;;
esac
"""


@pytest.fixture
def mock_dorado(tmp_path: Path, monkeypatch):
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available for the mock binary")
    home = tmp_path / "dorado_home"
    (home / "bin").mkdir(parents=True)
    stub = home / "bin" / "dorado"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("CONSTELLATION_DORADO_HOME", str(home))
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("MOCK_ARGV_FILE", str(argv_file))
    return argv_file


def test_basecaller_foreground(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    handle = DoradoRunner(device="cpu").basecaller(model, [tmp_path / "r.pod5"], out)
    assert handle.wait() == 0
    assert handle.poll() == RunStatus.COMPLETED
    assert out.read_bytes().startswith(b"MOCKBAM")
    assert handle.stderr_log.exists()
    argv = mock_dorado.read_text()
    assert "basecaller" in argv
    assert "--device cpu" in argv
    assert model.model_name() in argv


def test_basecaller_modified_bases_from_model(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("hac@v4.3+5mC,5hmC")
    out = tmp_path / "calls.bam"
    DoradoRunner(device="cpu").basecaller(model, [tmp_path / "r.pod5"], out).wait()
    argv = mock_dorado.read_text()
    assert "--modified-bases 5mC 5hmC" in argv


def test_basecaller_emit_moves_flag(tmp_path: Path, mock_dorado):
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    DoradoRunner(device="cpu").basecaller(
        model, [tmp_path / "r.pod5"], out, emit_moves=True
    ).wait()
    assert "--emit-moves" in mock_dorado.read_text()


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
    argv = mock_dorado.read_text()
    assert "--resume-from" in argv
    assert (tmp_path / "calls.bam.partial").exists()
