"""Tests for the multithreaded compression helper + ``samtools_fastq`` codec wiring.

Dev-box safe: no real samtools/bgzip required. The external ``bgzip`` is mocked
with a bash stub that delegates to the always-present ``gzip`` CLI, so the
two-process-pipe wiring, dual return-code propagation, and the stdlib-gzip
fallback are all exercised without htslib installed. ``resolve_bgzip``'s
resolution order is controlled deterministically by patching the registry's
``try_find`` (so a real bgzip on a CI ``$PATH`` can't perturb the assertions).
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from constellation.sequencing.assembly._samtools import samtools_fastq
from constellation.thirdparty import compress
from constellation.thirdparty.compress import compress_stream_to_path, resolve_bgzip
from constellation.thirdparty.registry import ToolHandle, ToolSpec

_RECORD = b"@r1\nACGTACGT\n+\nIIIIIIII\n"

# bgzip stub: --version probe, otherwise compress stdin->stdout via real gzip
# (BGZF and plain gzip are both valid gzip streams for our consumers).
_BGZIP_OK = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "bgzip (htslib) 1.21"; exit 0; fi
exec gzip -c
"""
# bgzip stub that consumes stdin then fails (so the producer exits 0 cleanly
# and the failure is unambiguously the compressor's).
_BGZIP_FAIL = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "bgzip (htslib) 1.21"; exit 0; fi
cat > /dev/null
exit 7
"""
# samtools stub: emits one fastq record; honors `-o FILE` (plain write) else stdout.
_SAMTOOLS = """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "samtools 1.21"; exit 0; fi
out=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  [[ "${args[$i]}" == "-o" ]] && out="${args[$((i+1))]}"
done
if [[ -n "$out" ]]; then printf '@r1\\nACGTACGT\\n+\\nIIIIIIII\\n' > "$out";
else printf '@r1\\nACGTACGT\\n+\\nIIIIIIII\\n'; fi
"""


@pytest.fixture(autouse=True)
def _need_shell():
    if shutil.which("bash") is None or shutil.which("gzip") is None:  # pragma: no cover
        pytest.skip("bash/gzip unavailable")


def _stub(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _producer(data: bytes) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({data!r})"]


def _fail_producer(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


def _handle(path: Path, name: str, env: str) -> ToolHandle:
    spec = ToolSpec(name=name, env_var=env, artifact=f"bin/{name}", path_bin=name)
    return ToolHandle(spec, Path(path), "env", None)


def _fake_find(*, samtools: Path | None = None, bgzip: Path | None = None):
    def _f(name: str) -> ToolHandle | None:
        if name == "samtools" and samtools is not None:
            return _handle(samtools, "samtools", "CONSTELLATION_SAMTOOLS_HOME")
        if name == "bgzip" and bgzip is not None:
            return _handle(bgzip, "bgzip", "CONSTELLATION_BGZIP_HOME")
        return None

    return _f


# ── resolve_bgzip resolution order ──────────────────────────────────────────


def test_resolve_bgzip_prefers_samtools_sibling(tmp_path, monkeypatch):
    bindir = tmp_path / "st" / "bin"
    _stub(bindir / "samtools", _SAMTOOLS)
    _stub(bindir / "bgzip", _BGZIP_OK)
    # samtools resolves to the stub; bgzip registry lookup deliberately empty —
    # the sibling must win without consulting it.
    monkeypatch.setattr(compress, "try_find", _fake_find(samtools=bindir / "samtools"))
    assert resolve_bgzip() == bindir / "bgzip"


def test_resolve_bgzip_falls_back_to_registry(tmp_path, monkeypatch):
    # samtools has no sibling bgzip; a registered/PATH bgzip is used instead.
    reg = _stub(tmp_path / "b" / "bin" / "bgzip", _BGZIP_OK)
    monkeypatch.setattr(compress, "try_find", _fake_find(bgzip=reg))
    assert resolve_bgzip() == reg


def test_resolve_bgzip_none_when_absent(monkeypatch):
    monkeypatch.setattr(compress, "try_find", _fake_find())
    assert resolve_bgzip() is None


# ── compress_stream_to_path ─────────────────────────────────────────────────


def test_compress_uses_bgzip_and_roundtrips(tmp_path, monkeypatch):
    bgz = _stub(tmp_path / "bin" / "bgzip", _BGZIP_OK)
    monkeypatch.setattr(compress, "try_find", _fake_find(bgzip=bgz))
    out = tmp_path / "out.gz"
    compress_stream_to_path(_producer(b"hello\nworld\n"), out, threads=4)
    with gzip.open(out, "rb") as fh:
        assert fh.read() == b"hello\nworld\n"


def test_compress_falls_back_to_stdlib_gzip(tmp_path, monkeypatch):
    monkeypatch.setattr(compress, "try_find", _fake_find())  # no bgzip anywhere
    assert resolve_bgzip() is None
    out = tmp_path / "out.gz"
    compress_stream_to_path(_producer(b"abc\n"), out, threads=2)
    with gzip.open(out, "rb") as fh:
        assert fh.read() == b"abc\n"


def test_compress_allow_bgzip_false_skips_bgzip(tmp_path, monkeypatch):
    bgz = _stub(tmp_path / "bin" / "bgzip", _BGZIP_FAIL)  # would fail if used
    monkeypatch.setattr(compress, "try_find", _fake_find(bgzip=bgz))
    out = tmp_path / "out.gz"
    # allow_bgzip=False -> stdlib path, never touches the failing bgzip
    compress_stream_to_path(_producer(b"xyz\n"), out, threads=1, allow_bgzip=False)
    with gzip.open(out, "rb") as fh:
        assert fh.read() == b"xyz\n"


def test_compress_producer_failure_raises(tmp_path, monkeypatch):
    bgz = _stub(tmp_path / "bin" / "bgzip", _BGZIP_OK)
    monkeypatch.setattr(compress, "try_find", _fake_find(bgzip=bgz))
    with pytest.raises(subprocess.CalledProcessError):
        compress_stream_to_path(_fail_producer(3), tmp_path / "out.gz", threads=2)


def test_compress_compressor_failure_raises(tmp_path, monkeypatch):
    bgz = _stub(tmp_path / "bin" / "bgzip", _BGZIP_FAIL)
    monkeypatch.setattr(compress, "try_find", _fake_find(bgzip=bgz))
    with pytest.raises(subprocess.CalledProcessError):
        compress_stream_to_path(_producer(b"data\n"), tmp_path / "out.gz", threads=2)


# ── samtools_fastq codec wiring ─────────────────────────────────────────────


def _samtools_home(tmp_path: Path) -> Path:
    """samtools + sibling bgzip under one HOME (mirrors a real htslib bin/)."""
    home = tmp_path / "st"
    _stub(home / "bin" / "samtools", _SAMTOOLS)
    _stub(home / "bin" / "bgzip", _BGZIP_OK)
    return home


def test_samtools_fastq_bgzf_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_SAMTOOLS_HOME", str(_samtools_home(tmp_path)))
    out = tmp_path / "reads.fastq.gz"
    samtools_fastq(tmp_path / "in.bam", out, threads=4, codec="bgzf")
    with gzip.open(out, "rb") as fh:
        assert fh.read() == _RECORD


def test_samtools_fastq_none_writes_plain(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_SAMTOOLS_HOME", str(_samtools_home(tmp_path)))
    out = tmp_path / "reads.fastq"
    samtools_fastq(tmp_path / "in.bam", out, threads=4, codec="none")
    assert out.read_bytes() == _RECORD  # plain, not gzip-wrapped


def test_samtools_fastq_gzip_codec_forces_stdlib(tmp_path, monkeypatch):
    # codec='gzip' must produce a valid gzip even though it skips bgzip.
    monkeypatch.setenv("CONSTELLATION_SAMTOOLS_HOME", str(_samtools_home(tmp_path)))
    out = tmp_path / "reads.fastq.gz"
    samtools_fastq(tmp_path / "in.bam", out, threads=4, codec="gzip")
    with gzip.open(out, "rb") as fh:
        assert fh.read() == _RECORD
