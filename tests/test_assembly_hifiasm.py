"""HiFiAsmRunner — argv construction + GFA→Assembly, via a mock binary.

No real hifiasm needed: a tiny shell stub resolved through
``$CONSTELLATION_HIFIASM_HOME`` writes the GFA outputs and records its
argv. Exercises the orchestrator (flag threading, GFA parsing, circular
inference, provenance) without the assembler.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from constellation.sequencing.assembly.hifiasm import HiFiAsmRunner

# Mock hifiasm: parse -o <prefix>, write primary + hap1 + hap2 GFAs, and
# (only on a real assembly call, i.e. when -o is present) record argv so
# the `--version` probe calls don't clobber the recording.
_STUB = r"""#!/usr/bin/env bash
set -e
prefix=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "-o" ]]; then
    prefix="${args[$((i+1))]}"
  fi
done
if [[ -z "$prefix" ]]; then
  exit 2   # e.g. a `--version` probe — no output
fi
if [[ -n "${MOCK_ARGV_FILE:-}" ]]; then
  printf '%s\n' "$*" > "${MOCK_ARGV_FILE}"
fi
write_gfa() {
  {
    printf 'S\tptg000001l\tACGTACGTAC\tLN:i:10\trd:i:30\n'
    printf 'S\tptg000002c\tTTTTGGGGCCCCAAAA\tLN:i:16\trd:i:45\n'
  } > "$1"
}
write_gfa "${prefix}.bp.p_ctg.gfa"
write_gfa "${prefix}.bp.hap1.p_ctg.gfa"
write_gfa "${prefix}.bp.hap2.p_ctg.gfa"
"""


@pytest.fixture
def mock_hifiasm(tmp_path: Path, monkeypatch):
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available for the mock binary")
    home = tmp_path / "tools"
    home.mkdir()
    stub = home / "hifiasm"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("CONSTELLATION_HIFIASM_HOME", str(home))
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("MOCK_ARGV_FILE", str(argv_file))
    return argv_file


def test_run_builds_assembly_from_gfa(tmp_path: Path, mock_hifiasm):
    reads = tmp_path / "reads.fastq.gz"
    reads.write_bytes(b"")
    runner = HiFiAsmRunner(threads=4, mode="ont")
    asm = runner.run([reads], tmp_path / "asm" / "primary")

    assert asm.n_contigs == 2
    contigs = asm.contigs.to_pylist()
    assert contigs[0]["name"] == "ptg000001l"
    assert contigs[0]["length"] == 10
    assert contigs[0]["read_coverage"] == 30.0
    assert contigs[0]["circular"] is False  # 'l' suffix → linear
    assert contigs[1]["circular"] is True  # 'c' suffix → circular
    assert contigs[0]["haplotype"] == "primary"
    assert contigs[0]["polish_rounds"] is None
    # provenance + stats populated
    assert "hifiasm" in contigs[0]["provenance_json"]
    assert asm.stats.to_pylist()[0]["n_contigs"] == 2


def test_run_threads_and_ont_flag_in_argv(tmp_path: Path, mock_hifiasm):
    reads = tmp_path / "reads.fastq.gz"
    reads.write_bytes(b"")
    HiFiAsmRunner(threads=7, mode="ont").run([reads], tmp_path / "asm" / "p")
    argv = mock_hifiasm.read_text()
    assert "--ont" in argv
    assert "-t 7" in argv
    assert str(reads) in argv


def test_hifi_mode_omits_ont_flag(tmp_path: Path, mock_hifiasm):
    reads = tmp_path / "reads.fastq.gz"
    reads.write_bytes(b"")
    HiFiAsmRunner(mode="hifi").run([reads], tmp_path / "asm" / "p")
    assert "--ont" not in mock_hifiasm.read_text()


def test_extra_args_threaded(tmp_path: Path, mock_hifiasm):
    reads = tmp_path / "reads.fastq.gz"
    reads.write_bytes(b"")
    HiFiAsmRunner(mode="ont", extra_args=("-l", "0")).run([reads], tmp_path / "asm" / "p")
    argv = mock_hifiasm.read_text()
    assert "-l 0" in argv


def test_run_diploid_two_haplotypes(tmp_path: Path, mock_hifiasm):
    reads = tmp_path / "reads.fastq.gz"
    reads.write_bytes(b"")
    hap1, hap2 = HiFiAsmRunner(mode="ont").run_diploid([reads], tmp_path / "asm" / "p")
    assert hap1.n_contigs == 2
    assert hap2.n_contigs == 2
    assert hap1.contigs.to_pylist()[0]["haplotype"] == "hap1"
    assert hap2.contigs.to_pylist()[0]["haplotype"] == "hap2"


def test_missing_gfa_raises(tmp_path: Path, monkeypatch):
    # A stub that runs cleanly but writes no GFA → FileNotFoundError.
    home = tmp_path / "tools"
    home.mkdir()
    stub = home / "hifiasm"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("CONSTELLATION_HIFIASM_HOME", str(home))
    reads = tmp_path / "r.fastq.gz"
    reads.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no hifiasm contig GFA"):
        HiFiAsmRunner().run([reads], tmp_path / "asm" / "p")


def test_tool_not_found_hint(tmp_path: Path, monkeypatch):
    # Point at an empty home + scrub PATH so the registry can't resolve.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("CONSTELLATION_HIFIASM_HOME", str(empty))
    monkeypatch.setenv("PATH", str(empty))
    if shutil.which("hifiasm") is not None:  # pragma: no cover
        pytest.skip("hifiasm resolvable despite scrubbed PATH")
    reads = tmp_path / "r.fastq.gz"
    reads.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="hifiasm not found"):
        HiFiAsmRunner().run([reads], tmp_path / "asm" / "p")
