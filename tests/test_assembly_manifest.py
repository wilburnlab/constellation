"""AssemblyManifest round-trip + guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.assembly.manifest import (
    read_manifest_dir,
    write_assembly_manifest,
)


def test_manifest_roundtrip(tmp_path: Path):
    write_assembly_manifest(
        tmp_path / "manifest.json",
        input_mode="bam",
        input_files=["fc1.bam", "fc2.bam"],
        unified_read_group="constellation_unified",
        polish_rounds=2,
        parameters={"threads": 16, "hifiasm_mode": "ont"},
        stages={"assemble": {"n_contigs": 42, "n50": 1234567}},
        outputs={"primary_fasta": "assembly/primary.fasta"},
        basecaller_model_ds="dna_r10.4.1_e8.2_sup@v5.0.0",
        scaffold_reference_handle="homo_sapiens@ensembl-111",
        scaffold_reference_path="/refs/homo_sapiens/ensembl-111",
        assembly_accession="GCF_000001405.40",
        tool_versions={"hifiasm": "0.25.0", "dorado": "0.8.3"},
        tool_args={"hifiasm": ["--ont"]},
        busco_lineage="eukaryota_odb10",
    )
    m = read_manifest_dir(tmp_path)
    assert m.kind == "assembly"
    assert m.input_mode == "bam"
    assert m.input_files == ["fc1.bam", "fc2.bam"]
    assert m.polish_rounds == 2
    assert m.scaffold_reference_handle == "homo_sapiens@ensembl-111"
    assert m.assembly_accession == "GCF_000001405.40"
    assert m.busco_lineage == "eukaryota_odb10"
    assert m.stages["assemble"]["n_contigs"] == 42
    assert m.tool_versions["hifiasm"] == "0.25.0"


def test_manifest_pod5_mode_basecall_provenance(tmp_path: Path):
    write_assembly_manifest(
        tmp_path / "manifest.json",
        input_mode="pod5",
        input_files=["flowcell1/"],
        unified_read_group="uni",
        polish_rounds=0,
        parameters={},
        stages={},
        outputs={},
        basecall_model="sup@v5.0.0",
        modified_bases=["5mC", "5hmC"],
        device="cuda:0",
    )
    m = read_manifest_dir(tmp_path)
    assert m.input_mode == "pod5"
    assert m.basecall_model == "sup@v5.0.0"
    assert m.modified_bases == ["5mC", "5hmC"]
    assert m.scaffold_reference_handle is None


def test_manifest_schema_version_guard(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('{"schema_version": 99, "kind": "assembly"}')
    with pytest.raises(ValueError, match="schema_version"):
        read_manifest_dir(tmp_path)


def test_manifest_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no manifest"):
        read_manifest_dir(tmp_path)
