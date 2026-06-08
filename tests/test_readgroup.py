"""Read-group harmonization — pure header logic + samtools integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.basecall.readgroup import (
    _models_from_header,
    extract_basecall_model,
    harmonize_read_group,
    read_basecaller_models,
    validate_single_model,
)
from constellation.thirdparty.registry import ToolNotFoundError, find


# ── pure: DS parsing ──────────────────────────────────────────────────


def test_extract_basecall_model_from_ds():
    ds = "basecall_model=dna_r10.4.1_e8.2_400bps_sup@v5.0.0 runid=abc123 flow_cell=FC1"
    assert extract_basecall_model(ds) == "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"


def test_extract_basecall_model_missing_token():
    assert extract_basecall_model("runid=abc flow_cell=FC1") is None
    assert extract_basecall_model("") is None
    assert extract_basecall_model(None) is None


def test_models_from_header():
    header = {
        "RG": [
            {"ID": "rg1", "DS": "basecall_model=modelA runid=x"},
            {"ID": "rg2", "DS": "basecall_model=modelA runid=y"},
        ]
    }
    assert _models_from_header(header) == {"modelA"}


# ── pure: validation ──────────────────────────────────────────────────


def test_validate_single_model_ok():
    models = {Path("a.bam"): {"m1"}, Path("b.bam"): {"m1"}}
    assert validate_single_model(models) == "m1"


def test_validate_no_model_returns_none():
    models = {Path("a.bam"): set(), Path("b.bam"): set()}
    assert validate_single_model(models) is None


def test_validate_multiple_models_raises():
    models = {Path("a.bam"): {"m1"}, Path("b.bam"): {"m2"}}
    with pytest.raises(ValueError, match="multiple basecaller models"):
        validate_single_model(models)


def test_validate_multiple_models_allow_override():
    models = {Path("a.bam"): {"m2"}, Path("b.bam"): {"m1"}}
    # deterministic pick (sorted) when overridden
    assert validate_single_model(models, allow_multi=True) == "m1"


# ── integration: harmonize (needs samtools + pysam) ──────────────────


def _have_samtools() -> bool:
    try:
        find("samtools")
        return True
    except ToolNotFoundError:
        return False


def _write_unaligned_bam(path: Path, rg_id: str, model: str, n: int = 2) -> None:
    import pysam  # type: ignore[import-not-found]

    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "RG": [{"ID": rg_id, "DS": f"basecall_model={model}", "PL": "ONT"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as fh:
        for i in range(n):
            seg = pysam.AlignedSegment(fh.header)
            seg.query_name = f"{rg_id}_read{i}"
            seg.query_sequence = "ACGTACGT"
            seg.flag = 4  # unmapped
            seg.query_qualities = pysam.qualitystring_to_array("IIIIIIII")
            seg.set_tag("RG", rg_id)
            fh.write(seg)


@pytest.mark.skipif(not _have_samtools(), reason="samtools not installed")
def test_harmonize_collapses_to_single_rg(tmp_path: Path):
    pytest.importorskip("pysam")
    b1 = tmp_path / "fc1.bam"
    b2 = tmp_path / "fc2.bam"
    _write_unaligned_bam(b1, "RGA", "sup@v5.0.0")
    _write_unaligned_bam(b2, "RGB", "sup@v5.0.0")

    models = read_basecaller_models([b1, b2])
    assert models[b1] == {"sup@v5.0.0"}
    model = validate_single_model(models)
    assert model == "sup@v5.0.0"

    out = tmp_path / "harmonized.bam"
    harmonize_read_group([b1, b2], out, unified_rg_id="uni", model_ds=model)

    import pysam  # type: ignore[import-not-found]

    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as fh:
        rgs = fh.header.to_dict().get("RG", [])
        assert len(rgs) == 1
        assert rgs[0]["ID"] == "uni"
        assert "basecall_model=sup@v5.0.0" in rgs[0]["DS"]
    with pysam.AlignmentFile(str(out), "rb", check_sq=False) as fh:
        assert all(rec.get_tag("RG") == "uni" for rec in fh)
    # intermediates cleaned up
    assert not (tmp_path / "harmonized.merged.bam").exists()
