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


def test_resume_that_dies_while_copying_keeps_the_partial(
    tmp_path: Path, mock_dorado, monkeypatch
):
    # A resume killed while Dorado was still copying the partial's reads
    # left a few bytes in the output; the next resume moved those bytes
    # over the partial and days of reads were gone.
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    partial = tmp_path / "calls.bam.partial"
    out.write_bytes(b"R" * 19_000)  # the first run, killed
    runner = DoradoRunner(device="cpu")

    monkeypatch.setenv("MOCK_FAIL_ON", "--resume-from")
    monkeypatch.setenv("MOCK_PARTIAL_BYTES", "3")
    assert runner.basecaller(model, [tmp_path / "r.pod5"], out, resume=True).wait() == 1
    assert partial.read_bytes() == b"R" * 19_000
    assert out.stat().st_size == 3

    monkeypatch.delenv("MOCK_FAIL_ON")
    assert runner.basecaller(model, [tmp_path / "r.pod5"], out, resume=True).wait() == 0
    _, _, options = parse_dorado_argv(mock_dorado.argv())
    assert options["--resume-from"] == [str(partial)]
    assert partial.read_bytes() == b"R" * 19_000
    # the stub writes the resume file's reads first, as Dorado does
    assert out.read_bytes() == b"R" * 19_000 + b"MOCKBAM"


def test_resume_takes_an_output_larger_than_the_partial(tmp_path: Path, mock_dorado):
    # An interrupted resume that had finished copying and gone on to new
    # reads holds more than the partial it started from.
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    partial = tmp_path / "calls.bam.partial"
    partial.write_bytes(b"P" * 100)
    out.write_bytes(b"P" * 100 + b"N" * 400)
    DoradoRunner(device="cpu").basecaller(
        model, [tmp_path / "r.pod5"], out, resume=True
    ).wait()
    assert partial.read_bytes() == b"P" * 100 + b"N" * 400
    assert out.read_bytes().startswith(b"P" * 100 + b"N" * 400)


@pytest.mark.parametrize("output_state", ["missing", "empty"])
def test_resume_from_a_partial_with_no_output(
    tmp_path: Path, mock_dorado, output_state: str
):
    # A partial with no output beside it was ignored, and the run started
    # again from zero.
    model = DoradoModel.parse("sup@v5.0.0")
    out = tmp_path / "calls.bam"
    partial = tmp_path / "calls.bam.partial"
    partial.write_bytes(b"P" * 50)
    if output_state == "empty":
        out.write_bytes(b"")
    DoradoRunner(device="cpu").basecaller(
        model, [tmp_path / "r.pod5"], out, resume=True
    ).wait()
    _, _, options = parse_dorado_argv(mock_dorado.argv())
    assert options["--resume-from"] == [str(partial)]
    assert out.read_bytes() == b"P" * 50 + b"MOCKBAM"


def test_unusable_dorado_leaves_the_output_where_it_was(tmp_path: Path, monkeypatch):
    # The output used to be moved to the partial before Dorado was
    # resolved; a too-old (or missing) Dorado then left nothing at the
    # output, and the next --resume started from zero.
    from constellation.sequencing.basecall.dorado_run import DoradoVersionError

    install(tmp_path, monkeypatch)
    monkeypatch.setenv("MOCK_DORADO_VERSION", "1.4.0")
    out = tmp_path / "calls.bam"
    out.write_bytes(b"R" * 19_000)
    with pytest.raises(DoradoVersionError):
        DoradoRunner(device="cpu").basecaller(
            DoradoModel.parse("sup@v5.0.0"), [tmp_path / "r.pod5"], out, resume=True
        )
    assert out.read_bytes() == b"R" * 19_000
    assert not (tmp_path / "calls.bam.partial").exists()


def test_fresh_run_refuses_to_orphan_a_partial(tmp_path: Path, mock_dorado):
    partial = tmp_path / "calls.bam.partial"
    partial.write_bytes(b"P" * 50)
    with pytest.raises(FileExistsError, match="--resume"):
        DoradoRunner(device="cpu").basecaller(
            DoradoModel.parse("sup@v5.0.0"), [tmp_path / "r.pod5"],
            tmp_path / "calls.bam", resume=False,
        )
    assert mock_dorado.calls() == []
    assert partial.read_bytes() == b"P" * 50


@pytest.mark.parametrize("version", ["1.4.0", "2.0.0", "2.1.1"])
def test_resolve_dorado_rejects_below_the_floor(tmp_path: Path, monkeypatch, version: str):
    # Below 2.1.2 the SUP v5+ and HAC v6 models do not give the same reads
    # twice, so an older Dorado is refused rather than run.
    from constellation.sequencing.basecall.dorado_run import (
        DoradoVersionError,
        resolve_dorado,
    )

    install(tmp_path, monkeypatch)
    monkeypatch.setenv("MOCK_DORADO_VERSION", version)
    with pytest.raises(DoradoVersionError, match="too old"):
        resolve_dorado()


def test_resolve_dorado_accepts_the_floor(tmp_path: Path, mock_dorado):
    # The mock reports 2.1.2+mock → resolves without raising.
    from constellation.sequencing.basecall.dorado_run import resolve_dorado

    assert resolve_dorado().name == "dorado"


def test_doctor_and_resolver_share_one_floor():
    from constellation.sequencing.basecall.dorado_run import DORADO_MIN_VERSION
    from constellation.thirdparty.registry import registered

    spec = next(s for s in registered() if s.name == "dorado")
    assert spec.min_version == DORADO_MIN_VERSION == "2.1.2"
