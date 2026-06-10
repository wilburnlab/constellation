"""CLI dispatch for ``constellation genome`` + ``basecall`` via main()."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from constellation.cli.__main__ import main

pytest.importorskip("pysam")
pytest.importorskip("matplotlib")

_DORADO = r"""#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "dorado 2.0.0+mock"; exit 0 ;;
  basecaller|duplex)
    [[ -n "${MOCK_ARGV_FILE:-}" ]] && printf '%s\n' "$*" > "${MOCK_ARGV_FILE}"
    printf 'MOCKBAM'; echo "[info] 50%% done" >&2; exit 0 ;;
  aligner) printf 'MOCKALN'; exit 0 ;;
  polish) draft="${@: -1}"; cat "$draft"; exit 0 ;;
  *) exit 0 ;;
esac
"""

_HIFIASM = r"""#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "hifiasm 0.25.0-r700"; exit 0; fi
prefix=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do [[ "${args[$i]}" == "-o" ]] && prefix="${args[$((i+1))]}"; done
[[ -z "$prefix" ]] && exit 2
printf 'S\tptg1l\tACGTACGTACGT\tLN:i:12\trd:i:30\n' > "${prefix}.bp.p_ctg.gfa"
"""

_SAMTOOLS = r"""#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "samtools 1.21"; exit 0 ;;
  cat) shift; out=""; first=""
    while [[ $# -gt 0 ]]; do case "$1" in -o) out="$2"; shift 2 ;; *) [[ -z "$first" ]] && first="$1"; shift ;; esac; done
    cp "$first" "$out" ;;
  addreplacerg) shift; out=""; inp=""
    while [[ $# -gt 0 ]]; do case "$1" in -@|-m|-r) shift 2 ;; -o) out="$2"; shift 2 ;; *) inp="$1"; shift ;; esac; done
    cp "$inp" "$out" ;;
  view) printf '@HD\tVN:1.6\n@RG\tID:o\tDS:basecall_model=sup@v5.0.0\n' ;;
  reheader) cat "${@: -1}" ;;
  index) : > "${@: -1}.bai" ;;
  sort) shift; out=""; inp=""
    while [[ $# -gt 0 ]]; do case "$1" in -@) shift 2 ;; -o) out="$2"; shift 2 ;; *) inp="$1"; shift ;; esac; done
    cp "$inp" "$out" ;;
  fastq) printf '@r\nACGT\n+\nIIII\n' ;;
  *) exit 0 ;;
esac
"""


def _stub(home: Path, rel: str, body: str, monkeypatch, env: str) -> None:
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    p.chmod(0o755)
    monkeypatch.setenv(env, str(home))


def _bam(path: Path, rg: str) -> None:
    import pysam  # type: ignore[import-not-found]

    header = {"HD": {"VN": "1.6"}, "RG": [{"ID": rg, "DS": "basecall_model=sup@v5.0.0"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as fh:
        seg = pysam.AlignedSegment(fh.header)
        seg.query_name = "r0"
        seg.query_sequence = "ACGT"
        seg.flag = 4
        seg.query_qualities = pysam.qualitystring_to_array("IIII")
        seg.set_tag("RG", rg)
        fh.write(seg)


@pytest.fixture
def mocks(tmp_path: Path, monkeypatch):
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash unavailable")
    _stub(tmp_path / "d", "bin/dorado", _DORADO, monkeypatch, "CONSTELLATION_DORADO_HOME")
    _stub(tmp_path / "h", "hifiasm", _HIFIASM, monkeypatch, "CONSTELLATION_HIFIASM_HOME")
    _stub(tmp_path / "s", "bin/samtools", _SAMTOOLS, monkeypatch, "CONSTELLATION_SAMTOOLS_HOME")


def test_cli_genome_assemble_bam(tmp_path: Path, mocks):
    b1 = tmp_path / "fc1.bam"
    _bam(b1, "RGA")
    out = tmp_path / "out"
    rc = main([
        "genome", "assemble",
        "--reads", str(b1),
        "--output-dir", str(out),
        "--device", "cpu",
        "--threads", "1",
        "--no-report",
    ])
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert (out / "assembly" / "assembly").is_dir()


def test_cli_genome_assemble_pod5_requires_model(tmp_path: Path, mocks):
    rc = main([
        "genome", "assemble",
        "--pod5", str(tmp_path / "x.pod5"),
        "--output-dir", str(tmp_path / "o"),
    ])
    assert rc == 2


def test_cli_basecall(tmp_path: Path, mocks):
    rc = main([
        "basecall",
        "--pod5", str(tmp_path / "reads.pod5"),
        "--model", "sup@v5.0.0",
        "--output-dir", str(tmp_path / "bc"),
        "--device", "cpu",
    ])
    assert rc == 0
    assert (tmp_path / "bc" / "calls.bam").read_bytes().startswith(b"MOCKBAM")


def test_cli_genome_diagnose(tmp_path: Path, mocks):
    # First assemble to produce a bundle, then diagnose read-only.
    b1 = tmp_path / "fc1.bam"
    _bam(b1, "RGA")
    out = tmp_path / "out"
    main([
        "genome", "assemble", "--reads", str(b1), "--output-dir", str(out),
        "--device", "cpu", "--threads", "1", "--no-report",
    ])
    rc = main(["genome", "diagnose", "--assembly-dir", str(out)])
    assert rc == 0
    assert (out / "diagnostics" / "report.md").exists()


def test_cli_basecall_emit_moves_on_by_default(tmp_path: Path, mocks, monkeypatch):
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("MOCK_ARGV_FILE", str(argv_file))
    rc = main([
        "basecall", "--pod5", str(tmp_path / "r.pod5"), "--model", "sup@v5.0.0",
        "--output-dir", str(tmp_path / "bc"), "--device", "cpu",
    ])
    assert rc == 0
    assert "--emit-moves" in argv_file.read_text()


def test_cli_basecall_no_emit_moves_opts_out(tmp_path: Path, mocks, monkeypatch):
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("MOCK_ARGV_FILE", str(argv_file))
    rc = main([
        "basecall", "--pod5", str(tmp_path / "r.pod5"), "--model", "sup@v5.0.0",
        "--output-dir", str(tmp_path / "bc"), "--device", "cpu", "--no-emit-moves",
    ])
    assert rc == 0
    assert "--emit-moves" not in argv_file.read_text()
