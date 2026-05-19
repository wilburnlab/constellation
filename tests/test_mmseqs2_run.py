"""Tier A tests for :mod:`constellation.thirdparty.mmseqs2_run`.

Strategy:
  * Pure-function tests for ``build_easy_search_args`` (argv composition
    across the supported parameter knobs).
  * Bash-script "fake mmseqs" for end-to-end ``run_mmseqs_search``
    coverage: success path, exit-code propagation, stdout/stderr
    separation + log file capture, ``MmseqsRunError`` shape on
    non-zero exit.

The fake-mmseqs approach mirrors :file:`tests/test_thirdparty_jvm.py`'s
Hello.java compilation idiom: register a real (synthetic) binary in
the thirdparty registry and exercise the runner against it. No
subprocess mocking — the runner's behaviour around Popen / threads /
log files is what we're testing.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from constellation.thirdparty.mmseqs2_run import (
    DEFAULT_FORMAT_OUTPUT,
    MmseqsResult,
    MmseqsRunError,
    build_easy_search_args,
    run_mmseqs_search,
)
from constellation.thirdparty.registry import ToolSpec, _REGISTRY, register


# ── build_easy_search_args (pure) ───────────────────────────────────────


def test_build_easy_search_args_default_layout() -> None:
    args = build_easy_search_args(
        query_fasta=Path("/q.fa"),
        target_fasta=Path("/t.fa"),
        output_tab=Path("/out.tab"),
        scratch_dir=Path("/scratch"),
    )
    # Positional shape: easy-search <query> <target> <output> <tmp>
    assert args[:5] == [
        "easy-search", "/q.fa", "/t.fa", "/out.tab", "/scratch",
    ]
    # Canonical flags follow in declared order.
    assert "-e" in args and args[args.index("-e") + 1] == "1e-20"
    assert "--threads" in args and args[args.index("--threads") + 1] == "4"
    assert "-s" in args and args[args.index("-s") + 1] == "5.7"
    fmt_idx = args.index("--format-output")
    assert args[fmt_idx + 1] == DEFAULT_FORMAT_OUTPUT


def test_build_easy_search_args_evalue_format() -> None:
    """E-value renders via ``format(..., 'g')`` so 1e-20 stays compact."""
    args = build_easy_search_args(
        query_fasta=Path("q"), target_fasta=Path("t"),
        output_tab=Path("o"), scratch_dir=Path("s"),
        evalue=1e-20,
    )
    assert args[args.index("-e") + 1] == "1e-20"
    args2 = build_easy_search_args(
        query_fasta=Path("q"), target_fasta=Path("t"),
        output_tab=Path("o"), scratch_dir=Path("s"),
        evalue=0.001,
    )
    assert args2[args2.index("-e") + 1] == "0.001"


def test_build_easy_search_args_custom_threads_sensitivity() -> None:
    args = build_easy_search_args(
        query_fasta=Path("q"), target_fasta=Path("t"),
        output_tab=Path("o"), scratch_dir=Path("s"),
        threads=16,
        sensitivity=7.5,
    )
    assert args[args.index("--threads") + 1] == "16"
    assert args[args.index("-s") + 1] == "7.5"


def test_build_easy_search_args_custom_format_output() -> None:
    args = build_easy_search_args(
        query_fasta=Path("q"), target_fasta=Path("t"),
        output_tab=Path("o"), scratch_dir=Path("s"),
        format_output="query,target,pident,evalue",
    )
    assert args[args.index("--format-output") + 1] == "query,target,pident,evalue"


def test_build_easy_search_args_extra_args_appended() -> None:
    """``extra_args`` rides after the canonical flags."""
    args = build_easy_search_args(
        query_fasta=Path("q"), target_fasta=Path("t"),
        output_tab=Path("o"), scratch_dir=Path("s"),
        extra_args=["--min-seq-id", "0.3", "-c", "0.8"],
    )
    # The extras should be the suffix of args.
    assert args[-4:] == ["--min-seq-id", "0.3", "-c", "0.8"]


# ── fake-mmseqs end-to-end ──────────────────────────────────────────────


_FAKE_MMSEQS = r"""#!/usr/bin/env bash
# A minimal fake `mmseqs` binary for testing the runner.
#
# Behaviour:
#   mmseqs version           → prints "fake-v1" on stdout
#   mmseqs easy-search Q T O TMP [...]   → writes "STDOUT:ok" / "STDERR:ok",
#                                          touches O, exits 0
#   mmseqs easy-search ... --fail-arg fail → writes failure noise to stderr, exits 7
set -eu
case "${1:-}" in
    version)
        echo "fake-v1"
        ;;
    easy-search)
        echo "STDOUT:easy-search-running"
        echo "STDERR:easy-search-running" >&2
        # Output path is the 4th positional arg (Q T O TMP).
        out_path="$4"
        # Detect a fail trigger anywhere in argv.
        for arg in "$@"; do
            if [[ "${arg}" == "FAIL_TRIGGER" ]]; then
                echo "STDERR:forced failure line 1" >&2
                echo "STDERR:forced failure line 2" >&2
                exit 7
            fi
        done
        : > "${out_path}"
        echo "STDOUT:easy-search-done"
        ;;
    *)
        echo "fake mmseqs: unknown subcommand $1" >&2
        exit 2
        ;;
esac
"""


@pytest.fixture
def fake_mmseqs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a session-scoped bash fake-mmseqs binary."""
    install = tmp_path_factory.mktemp("mmseqs_install")
    bin_dir = install / "bin"
    bin_dir.mkdir()
    mmseqs = bin_dir / "mmseqs"
    mmseqs.write_text(_FAKE_MMSEQS)
    mmseqs.chmod(mmseqs.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return mmseqs


@pytest.fixture
def register_fake_mmseqs(fake_mmseqs: Path):
    """Register ``mmseqs2`` in the thirdparty registry pointing at the
    fake binary's install dir."""
    name = "mmseqs2"
    env_var = "CONSTELLATION_MMSEQS2_HOME"

    saved_spec = _REGISTRY.get(name)
    saved_env = os.environ.get(env_var)

    # The real adapter declares artifact="bin/mmseqs"; keep that contract
    # so registry resolution finds <install>/bin/mmseqs.
    spec = ToolSpec(
        name=name,
        env_var=env_var,
        artifact="bin/mmseqs",
        path_bin=None,                          # skip $PATH fallback in tests
        install_script="scripts/install-mmseqs2.sh",
        version_probe=lambda _p: "fake-v1",
    )
    register(spec)
    os.environ[env_var] = str(fake_mmseqs.parent.parent)
    try:
        yield
    finally:
        if saved_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = saved_env
        if saved_spec is None:
            _REGISTRY.pop(name, None)
        else:
            _REGISTRY[name] = saved_spec


def _make_input_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (query, target, output_tab) under tmp_path."""
    q = tmp_path / "query.fa"
    q.write_text(">q1\nMKLIGHTPEPTR\n")
    t = tmp_path / "target.fa"
    t.write_text(">t1\nMKLIGHTPEPTR\n")
    o = tmp_path / "out.tab"
    return q, t, o


def test_run_mmseqs_search_success(
    register_fake_mmseqs, tmp_path: Path
) -> None:
    q, t, o = _make_input_files(tmp_path)
    log_dir = tmp_path / "logs"
    result = run_mmseqs_search(
        query_fasta=q,
        target_fasta=t,
        output_tab=o,
        log_dir=log_dir,
        stream_to_stderr=False,
    )
    assert isinstance(result, MmseqsResult)
    assert result.returncode == 0
    assert result.elapsed_seconds >= 0
    assert log_dir.exists()
    stdout_text = (log_dir / "stdout.log").read_text()
    stderr_text = (log_dir / "stderr.log").read_text()
    assert "STDOUT:easy-search-running" in stdout_text
    assert "STDOUT:easy-search-done" in stdout_text
    assert "STDERR:easy-search-running" in stderr_text
    assert (log_dir / "mmseqs-version.txt").exists()
    # Output file was touched by the fake binary.
    assert o.is_file()
    # mmseqs_sha256 is 64 hex chars (cached).
    assert len(result.mmseqs_sha256) == 64
    assert all(c in "0123456789abcdef" for c in result.mmseqs_sha256)
    # Version came from the spec's probe.
    assert result.mmseqs_version == "fake-v1"


def test_run_mmseqs_search_argv_composition(
    register_fake_mmseqs, tmp_path: Path
) -> None:
    """``mmseqs`` binary is argv[0]; easy-search is argv[1]; positional
    quartet follows; canonical flags after that; ``extra_args`` last."""
    q, t, o = _make_input_files(tmp_path)
    result = run_mmseqs_search(
        query_fasta=q,
        target_fasta=t,
        output_tab=o,
        log_dir=tmp_path / "logs",
        evalue=1e-10,
        threads=8,
        sensitivity=6.5,
        extra_args=["--min-seq-id", "0.5"],
        scratch_dir=tmp_path / "scratch",
        stream_to_stderr=False,
    )
    argv = result.argv
    assert Path(argv[0]).name == "mmseqs"
    assert argv[1] == "easy-search"
    assert argv[2] == str(q)
    assert argv[3] == str(t)
    assert argv[4] == str(o)
    assert argv[5] == str(tmp_path / "scratch")
    assert "-e" in argv and argv[argv.index("-e") + 1] == "1e-10"
    assert "--threads" in argv and argv[argv.index("--threads") + 1] == "8"
    assert "-s" in argv and argv[argv.index("-s") + 1] == "6.5"
    assert argv[-2:] == ["--min-seq-id", "0.5"]


def test_run_mmseqs_search_failure_raises(
    register_fake_mmseqs, tmp_path: Path
) -> None:
    """Non-zero exit + ``check=True`` (default) raises with tail+log."""
    q, t, o = _make_input_files(tmp_path)
    with pytest.raises(MmseqsRunError) as exc_info:
        run_mmseqs_search(
            query_fasta=q,
            target_fasta=t,
            output_tab=o,
            log_dir=tmp_path / "logs",
            extra_args=["FAIL_TRIGGER"],
            stream_to_stderr=False,
        )
    err = exc_info.value
    assert err.returncode == 7
    assert "STDERR:forced failure line 2" in err.stderr_tail
    assert err.stderr_log == tmp_path / "logs" / "stderr.log"
    assert "forced failure" in err.stderr_log.read_text()


def test_run_mmseqs_search_failure_no_check(
    register_fake_mmseqs, tmp_path: Path
) -> None:
    """``check=False`` returns the result with non-zero ``returncode``
    instead of raising — for callers that want to inspect both
    outcomes."""
    q, t, o = _make_input_files(tmp_path)
    result = run_mmseqs_search(
        query_fasta=q, target_fasta=t, output_tab=o,
        log_dir=tmp_path / "logs",
        extra_args=["FAIL_TRIGGER"],
        check=False,
        stream_to_stderr=False,
    )
    assert result.returncode == 7
    assert "forced failure" in (tmp_path / "logs" / "stderr.log").read_text()


def test_run_mmseqs_search_scratch_dir_autocleanup(
    register_fake_mmseqs, tmp_path: Path
) -> None:
    """When ``scratch_dir`` is None, the runner uses a TemporaryDirectory
    that's cleaned up on return."""
    q, t, o = _make_input_files(tmp_path)
    result = run_mmseqs_search(
        query_fasta=q, target_fasta=t, output_tab=o,
        log_dir=tmp_path / "logs",
        scratch_dir=None,
        stream_to_stderr=False,
    )
    # Extract the scratch dir from argv (5th positional after binary).
    scratch_path = Path(result.argv[5])
    assert not scratch_path.exists()    # auto-cleaned by TemporaryDirectory


def test_run_mmseqs_search_explicit_scratch_kept(
    register_fake_mmseqs, tmp_path: Path
) -> None:
    """Caller-supplied ``scratch_dir`` is NOT cleaned up by the runner —
    the orchestrator wants this for ``<output-dir>/03_alignment/.scratch/``
    so subsequent ``--resume`` runs can pick up where they left off."""
    q, t, o = _make_input_files(tmp_path)
    scratch = tmp_path / "kept_scratch"
    run_mmseqs_search(
        query_fasta=q, target_fasta=t, output_tab=o,
        log_dir=tmp_path / "logs",
        scratch_dir=scratch,
        stream_to_stderr=False,
    )
    assert scratch.is_dir()
