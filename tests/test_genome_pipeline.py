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
  fastq)
    shift; out=""
    while [[ $# -gt 0 ]]; do
      case "$1" in -@) shift 2 ;; -o) out="$2"; shift 2 ;; *) shift ;; esac
    done
    if [[ -n "$out" ]]; then printf '@r1\nACGTACGT\n+\nIIIIIIII\n' > "$out";
    else printf '@r1\nACGTACGT\n+\nIIIIIIII\n'; fi ;;
  *) exit 0 ;;
esac
"""

# bgzip ships beside samtools in the same htslib bin/; the stub delegates to
# the always-present `gzip` CLI so the fastq stage exercises the bgzip-pipe
# path rather than the stdlib-gzip fallback.
_BGZIP_STUB = r"""#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "bgzip (htslib) 1.21"; exit 0; fi
exec gzip -c
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
    # bgzip as a samtools sibling so compress.resolve_bgzip() finds it.
    bgz = tmp_path / "st" / "bin" / "bgzip"
    bgz.write_text(_BGZIP_STUB)
    bgz.chmod(0o755)


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


def test_pipeline_reads_compression_none_writes_plain_fastq(tmp_path: Path, mocks):
    b1 = tmp_path / "fc1.bam"
    _write_bam(b1, "RGA", "sup@v5.0.0")
    out = tmp_path / "asm_out"
    run_assembly_pipeline(
        output_dir=out, reads=[b1], device="cpu", threads=1, reads_compression="none"
    )
    assert (out / "reads" / "reads.fastq").exists()  # plain, no .gz
    assert not (out / "reads" / "reads.fastq.gz").exists()
    assert (out / "assembly" / "assembly").is_dir()


def test_pipeline_scratch_dir_routes_fastq_off_tree(tmp_path: Path, mocks, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    b1 = tmp_path / "fc1.bam"
    _write_bam(b1, "RGA", "sup@v5.0.0")
    out = tmp_path / "asm_out"
    scratch = tmp_path / "node_local"
    run_assembly_pipeline(
        output_dir=out, reads=[b1], device="cpu", threads=1, scratch_dir=scratch
    )
    fastq = scratch / "constellation_asm_4242" / "reads" / "reads.fastq.gz"
    assert fastq.exists()
    assert not (out / "reads" / "reads.fastq.gz").exists()  # not in the output tree
    assert (out / "assembly" / "assembly").is_dir()
    # manifest records the off-tree FASTQ as an absolute path (no relative_to raise)
    m = read_manifest_dir(out)
    assert m.outputs["reads_fastq"] == str(fastq)


def test_pipeline_no_keep_intermediates_deletes_fastq(tmp_path: Path, mocks):
    b1 = tmp_path / "fc1.bam"
    _write_bam(b1, "RGA", "sup@v5.0.0")
    out = tmp_path / "asm_out"
    run_assembly_pipeline(
        output_dir=out, reads=[b1], device="cpu", threads=1, keep_intermediates=False
    )
    assert not (out / "reads" / "reads.fastq.gz").exists()  # cleaned up
    assert (out / "assembly" / "assembly").is_dir()  # but assembly survived


# ── review findings on #92 ─────────────────────────────────────────────


def test_default_shorthand_expands_to_a_canonical_model_name() -> None:
    """R10.4.1 identifiers carry the translocation speed.

    Without it the advertised `sup@v5.0.0` shorthand named a model that
    does not exist; the same header form is what this package's own @RG
    parser documents and tests against.
    """
    from constellation.sequencing.basecall.models import DoradoModel

    assert (
        DoradoModel.parse("sup@v5.0.0").model_name()
        == "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"
    )


def test_stage_key_tracks_parameters_and_chains() -> None:
    from constellation.sequencing.assembly.pipeline import _stage_key

    a = _stage_key(up="x", rounds=1)
    assert _stage_key(up="x", rounds=2) != a, "a changed parameter must differ"
    assert _stage_key(up="y", rounds=1) != a, "an upstream change must cascade"
    assert _stage_key(up="x", rounds=1) == a, "and be stable otherwise"


def test_done_requires_a_matching_key(tmp_path) -> None:
    """Reuse hinged on _SUCCESS alone, so a rerun with different inputs
    reused the old bundle while the manifest recorded the new ones."""
    from constellation.sequencing.assembly.pipeline import _done, _mark_done

    d = tmp_path / "stage"
    _mark_done(d, "key-1")
    assert _done(d, "key-1")
    assert not _done(d, "key-2")
    # A legacy marker with no key is a miss: re-run rather than trust
    # output of unknown provenance.
    (d / "_SUCCESS").write_text("")
    assert not _done(d, "key-1")


def test_latest_assembly_bundle_prefers_the_most_advanced_stage(tmp_path) -> None:
    """Standalone polish restarted from the unscaffolded draft."""
    from constellation.cli.__main__ import _latest_assembly_bundle

    root = tmp_path / "run"
    (root / "assembly" / "assembly").mkdir(parents=True)
    assert _latest_assembly_bundle(root).name == "assembly"
    assert _latest_assembly_bundle(root).parent.name == "assembly"

    (root / "scaffold" / "assembly").mkdir(parents=True)
    assert _latest_assembly_bundle(root).parent.name == "scaffold"

    (root / "polish" / "assembly").mkdir(parents=True)
    assert _latest_assembly_bundle(root).parent.name == "polish"

    # Scaffolding does NOT walk forward: the canonical order is
    # assemble -> scaffold -> polish, so inheriting a polish (or an
    # existing scaffold) bundle from directory contents would silently
    # re-order the pipeline.
    assert (
        _latest_assembly_bundle(root, prefer=("assembly",)).parent.name
        == "assembly"
    )

    assert _latest_assembly_bundle(tmp_path / "nothing") is None


def test_missing_model_metadata_is_rejected(tmp_path) -> None:
    """One BAM naming a model and another naming none is a mismatch."""
    import pytest

    from constellation.sequencing.basecall.readgroup import validate_single_model

    models = {Path("a.bam"): {"dna_r10.4.1_e8.2_400bps_sup@v5.0.0"}, Path("b.bam"): set()}
    with pytest.raises(ValueError, match="declare no basecaller model"):
        validate_single_model(models)
    # --allow-multi-model is the documented override.
    assert validate_single_model(models, allow_multi=True) is not None
    # And no models anywhere is still the documented None.
    assert validate_single_model({Path("a.bam"): set()}) is None


def test_allow_multi_model_does_not_fabricate_one_model() -> None:
    """--allow-multi-model must skip the guard, not invent homogeneity.

    Returning the lexicographically first model stamped it onto every
    read, so `dorado polish` would apply m1's model to m2's reads with
    nothing left to detect the mismatch.
    """
    from constellation.sequencing.basecall.readgroup import validate_single_model

    mixed = {Path("a.bam"): {"m1"}, Path("b.bam"): {"m2"}}
    assert validate_single_model(mixed, allow_multi=True) is None
    # A genuine single model is still reported.
    assert (
        validate_single_model({Path("a.bam"): {"m1"}}, allow_multi=True) == "m1"
    )


def test_negative_polish_rounds_rejected() -> None:
    """0 disables polishing; a negative count silently returned the input
    unchanged while the caller reported success."""
    import pytest

    from constellation.sequencing.assembly.polish import PolishRunner

    with pytest.raises(ValueError, match="must be >= 0"):
        PolishRunner(rounds=-1).run(None, [], Path("/tmp/x"), rounds=-1)


def test_ont_mode_requires_a_capable_hifiasm(monkeypatch) -> None:
    """An old system hifiasm used to fail hours in, on an unknown flag."""
    import pytest

    from constellation.sequencing.assembly import hifiasm as H

    monkeypatch.setattr(H, "_hifiasm_version", lambda: "0.19.5")
    with pytest.raises(RuntimeError, match=r"does not support --ont"):
        H._require_ont_capable_hifiasm("ont")

    # New enough, a non-ONT mode, and an unprobeable version all proceed.
    monkeypatch.setattr(H, "_hifiasm_version", lambda: "0.25.0")
    H._require_ont_capable_hifiasm("ont")
    monkeypatch.setattr(H, "_hifiasm_version", lambda: "0.19.5")
    H._require_ont_capable_hifiasm("hifi")
    monkeypatch.setattr(H, "_hifiasm_version", lambda: None)
    H._require_ont_capable_hifiasm("ont")
