"""End-to-end ``run_assembly_pipeline`` (BAM mode) via mock binaries.

Wires together mock dorado / samtools / hifiasm so the whole spine —
input collection, @RG harmonization, FASTQ, assembly, polish, per-stage +
comparative reports, manifest — runs without the real (GPU, multi-hour)
tools. Real input BAMs (built with pysam) are needed so the read-group
model validation has genuine headers to read.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from constellation.sequencing.assembly.manifest import read_manifest_dir
from constellation.sequencing.assembly.pipeline import run_assembly_pipeline

pytest.importorskip("pysam")
pytest.importorskip("matplotlib")

_HIFIASM_STUB = r"""#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "hifiasm 0.25.0-r700"; exit 0; fi
prefix=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  [[ "${args[$i]}" == "-o" ]] && prefix="${args[$((i+1))]}"
done
[[ -z "$prefix" ]] && exit 2
{
  printf 'S\tptg000001l\tACGTACGTACGTACGT\tLN:i:16\trd:i:30\n'
  printf 'S\tptg000002l\tTTTTGGGGCCCCAAAA\tLN:i:16\trd:i:25\n'
} > "${prefix}.bp.p_ctg.gfa"
"""

_DORADO_STUB = r"""#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "dorado 2.0.0+mock"; exit 0 ;;
  aligner) printf 'MOCKALN'; exit 0 ;;
  polish) draft="${@: -1}"; cat "$draft"; exit 0 ;;
  *) exit 0 ;;
esac
"""

_SAMTOOLS_STUB = r"""#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "samtools 1.21"; exit 0 ;;
  cat)
    shift; out=""; first=""
    while [[ $# -gt 0 ]]; do
      case "$1" in -o) out="$2"; shift 2 ;; *) [[ -z "$first" ]] && first="$1"; shift ;; esac
    done
    cp "$first" "$out" ;;
  addreplacerg)
    shift; out=""; inp=""
    while [[ $# -gt 0 ]]; do
      case "$1" in -@|-m|-r) shift 2 ;; -o) out="$2"; shift 2 ;; *) inp="$1"; shift ;; esac
    done
    cp "$inp" "$out" ;;
  view) printf '@HD\tVN:1.6\n@RG\tID:old\tDS:basecall_model=sup@v5.0.0\n' ;;
  reheader) tagged="${@: -1}"; cat "$tagged" ;;
  index) bam="${@: -1}"; : > "${bam}.bai" ;;
  sort)
    shift; out=""; inp=""
    while [[ $# -gt 0 ]]; do
      case "$1" in -@) shift 2 ;; -o) out="$2"; shift 2 ;; *) inp="$1"; shift ;; esac
    done
    cp "$inp" "$out" ;;
  fastq) printf '@r1\nACGTACGT\n+\nIIIIIIII\n' ;;
  *) exit 0 ;;
esac
"""


def _stub(home: Path, rel: str, body: str, monkeypatch, env_var: str) -> None:
    path = home / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    monkeypatch.setenv(env_var, str(home))


def _write_bam(path: Path, rg_id: str, model: str) -> None:
    import pysam  # type: ignore[import-not-found]

    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "RG": [{"ID": rg_id, "DS": f"basecall_model={model}", "PL": "ONT"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as fh:
        seg = pysam.AlignedSegment(fh.header)
        seg.query_name = f"{rg_id}_r0"
        seg.query_sequence = "ACGTACGT"
        seg.flag = 4
        seg.query_qualities = pysam.qualitystring_to_array("IIIIIIII")
        seg.set_tag("RG", rg_id)
        fh.write(seg)


@pytest.fixture
def mocks(tmp_path: Path, monkeypatch):
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash unavailable")
    _stub(tmp_path / "hf", "hifiasm", _HIFIASM_STUB, monkeypatch, "CONSTELLATION_HIFIASM_HOME")
    _stub(tmp_path / "dr", "bin/dorado", _DORADO_STUB, monkeypatch, "CONSTELLATION_DORADO_HOME")
    _stub(tmp_path / "st", "bin/samtools", _SAMTOOLS_STUB, monkeypatch, "CONSTELLATION_SAMTOOLS_HOME")


def test_pipeline_bam_assemble_and_polish(tmp_path: Path, mocks):
    b1 = tmp_path / "fc1.bam"
    b2 = tmp_path / "fc2.bam"
    _write_bam(b1, "RGA", "sup@v5.0.0")
    _write_bam(b2, "RGB", "sup@v5.0.0")

    out = tmp_path / "asm_out"
    run_assembly_pipeline(
        output_dir=out,
        reads=[b1, b2],
        device="cpu",
        threads=1,
        polish_rounds=1,
        hifiasm_mode="ont",
    )

    # stage artifacts
    assert (out / "bam" / "harmonized.bam").exists()
    assert (out / "reads" / "reads.fastq.gz").exists()
    assert (out / "assembly" / "assembly").is_dir()
    assert (out / "polish" / "assembly").is_dir()
    assert (out / "assembly" / "diagnostics" / "report.md").exists()
    assert (out / "diagnostics" / "comparison.md").exists()

    # manifest
    m = read_manifest_dir(out)
    assert m.input_mode == "bam"
    assert m.polish_rounds == 1
    assert m.basecaller_model_ds == "sup@v5.0.0"
    assert m.unified_read_group == "constellation_unified"
    assert m.stages["assemble"]["n_contigs"] == 2
    assert "polish" in m.stages


def test_pipeline_resume_skips_completed_stages(tmp_path: Path, mocks):
    b1 = tmp_path / "fc1.bam"
    _write_bam(b1, "RGA", "sup@v5.0.0")
    out = tmp_path / "asm_out"
    run_assembly_pipeline(output_dir=out, reads=[b1], device="cpu", threads=1)
    assert (out / "assembly" / "_SUCCESS").exists()

    # remove the hifiasm stub; a resume must NOT re-run assembly
    import os

    os.environ["CONSTELLATION_HIFIASM_HOME"] = str(tmp_path / "nonexistent")
    run_assembly_pipeline(output_dir=out, reads=[b1], device="cpu", threads=1, resume=True)
    assert (out / "manifest.json").exists()


def test_pipeline_rejects_both_inputs(tmp_path: Path, mocks):
    b1 = tmp_path / "fc1.bam"
    _write_bam(b1, "RGA", "sup@v5.0.0")
    with pytest.raises(ValueError, match="exactly one"):
        run_assembly_pipeline(
            output_dir=tmp_path / "o", reads=[b1], pod5=[tmp_path / "x.pod5"]
        )
