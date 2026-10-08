"""``transcriptome demultiplex --resume``: what is reused must be what the
manifest says it is.

``--resume`` over a directory demultiplexed before the poly-A run-merge was
corrected reused every shard and then wrote a manifest stamped
``polyA_merge = "gap"`` — the one field that says a directory's transcript
windows are free of the split-tail remnant, asserted of windows that carry
it (review of 91e7c69). The settings a directory's shards are a function of are
now stamped before any work and checked on resume; a directory from before
the stamp keeps the provenance its own manifest gave it, or is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from constellation.cli.__main__ import main
from constellation.sequencing.samples import Samples
from constellation.sequencing.transcriptome.demux.designs import CDNA_WILBURN_V1
from constellation.sequencing.transcriptome.demux.simulator import (
    generate_stress_test_specs,
    is_deterministic_clean,
    simulate_panel,
)
from constellation.sequencing.transcriptome.stages import (
    DEMUX_SETTINGS,
    check_demux_resume,
    demux_settings,
    run_demux_pipeline,
)

_KEYS = ("reads", "read_segments", "read_demux", "orfs", "feature_quant")


@pytest.fixture(scope="module")
def sam(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("sim")
    specs = generate_stress_test_specs(CDNA_WILBURN_V1, n_per_category=2, seed=42)
    clean = [s for s in specs if is_deterministic_clean(s)]
    path = d / "synthetic.sam"
    simulate_panel(
        clean, CDNA_WILBURN_V1, sam_path=path, ground_truth_path=d / "truth.parquet"
    )
    return path


def _samples() -> Samples:
    n = len(CDNA_WILBURN_V1.layout[3].barcodes)
    return Samples.from_records(
        samples=[
            {"sample_id": i + 1, "sample_name": f"BC{i + 1:02d}", "description": None}
            for i in range(n)
        ],
        edges=[
            {"sample_id": i + 1, "acquisition_id": 1, "barcode_id": i} for i in range(n)
        ],
    )


def _samples_tsv(tmp_path: Path) -> Path:
    n = len(CDNA_WILBURN_V1.layout[3].barcodes)
    path = tmp_path / "samples.tsv"
    path.write_text(
        "sample_name\tbarcode_id\n" + "".join(f"BC{i + 1:02d}\t{i}\n" for i in range(n))
    )
    return path


def _cli(sam: Path, tmp_path: Path, out: Path, *extra: str) -> int:
    return main(
        [
            "transcriptome", "demultiplex",
            "--reads", str(sam),
            "--samples", str(_samples_tsv(tmp_path)),
            "--output-dir", str(out),
            *extra,
        ]
    )  # fmt: skip


def _run(sam: Path, out: Path, **kw):
    return run_demux_pipeline(
        sam,
        library_design="cdna_wilburn_v1",
        samples=_samples(),
        acquisition_id=1,
        output_dir=out,
        batch_size=1000,
        n_workers=1,
        **kw,
    )


def _shard_times(out: Path) -> dict[str, int]:
    return {
        str(p.relative_to(out)): p.stat().st_mtime_ns
        for key in _KEYS
        for p in (out / key).glob("part-*.parquet")
    }


def _make_legacy(out: Path) -> None:
    """Turn ``out`` into what the pre-fix code left: no stamp, and a
    manifest that says nothing of how poly-A was merged."""
    (out / DEMUX_SETTINGS).unlink()
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["parameters"] = {
        k: v for k, v in manifest["parameters"].items() if not k.startswith("polyA_")
    }
    path.write_text(json.dumps(manifest, indent=2))


def test_a_run_stamps_its_settings_before_it_works_and_resumes_as_itself(sam, tmp_path):
    out = tmp_path / "demux"
    first = _run(sam, out)
    stamp = json.loads((out / DEMUX_SETTINGS).read_text())
    assert stamp == first["settings"]
    assert (
        stamp["polyA_merge"] == "gap" and stamp["library_design"] == "cdna_wilburn_v1"
    )
    assert stamp["min_aa_length"] == 60 and stamp["min_protein_count"] == 2
    before = _shard_times(out)
    again = _run(sam, out, resume=True)
    assert again["settings"] == stamp and _shard_times(out) == before


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"min_aa_length": 30}, "min_aa_length"),
        ({"min_protein_count": 5}, "min_protein_count"),
    ],
)
def test_a_resume_under_other_settings_is_refused_before_anything_is_touched(
    sam, tmp_path, change, named
):
    """The ORF shards and the count matrix on disk were cut at the old
    thresholds; reused, the manifest would report the new ones."""
    out = tmp_path / "demux"
    _run(sam, out)
    before = _shard_times(out)
    stamp = (out / DEMUX_SETTINGS).read_text()
    with pytest.raises(ValueError, match=rf"--resume.*{named}: .* on disk, .* now"):
        _run(sam, out, resume=True, **change)
    assert _shard_times(out) == before and (out / DEMUX_SETTINGS).read_text() == stamp
    # Without --resume the directory is this run's to rewrite, and restamp.
    _run(sam, out, **change)
    assert json.loads((out / DEMUX_SETTINGS).read_text())[named] == change[named]


def test_a_changed_design_parameter_is_another_trimming(sam, tmp_path):
    out = tmp_path / "demux"
    _run(sam, out)
    stamp = json.loads((out / DEMUX_SETTINGS).read_text())
    stamp["polyA_edge_distance"] = 1  # what the shipped design used to say
    (out / DEMUX_SETTINGS).write_text(json.dumps(stamp))
    with pytest.raises(ValueError, match="polyA_edge_distance: 1 on disk, 2 now"):
        _run(sam, out, resume=True)


def test_a_legacy_directory_keeps_its_own_provenance_through_a_resume(
    sam, tmp_path, capsys
):
    """The case in the review: every shard reused, and the manifest written
    afterwards claimed the corrected trimming. Nothing is re-trimmed here,
    so nothing can be mixed and the reuse is allowed — as what it is."""
    out = tmp_path / "demux"
    assert _cli(sam, tmp_path, out) == 0
    assert (
        json.loads((out / "manifest.json").read_text())["parameters"]["polyA_merge"]
        == "gap"
    )
    _make_legacy(out)
    before = _shard_times(out)

    assert _cli(sam, tmp_path, out, "--resume", "--emit-fastq") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert not [k for k in manifest["parameters"] if k.startswith("polyA_")]
    assert manifest["library_design"] == "cdna_wilburn_v1"
    assert manifest["parameters"]["min_aa_length"] == 60
    assert manifest["parameters"]["resumed"] is True
    assert _shard_times(out) == before, "reused, not re-trimmed"
    assert (out / "fastq" / "_SUCCESS").exists(), "the bolt-on still works"
    assert not (out / DEMUX_SETTINGS).exists(), "still nobody's to vouch for"
    assert "before the poly-A run-merge was corrected" in capsys.readouterr().err

    # ...and it stays legacy through a second resume, rather than being
    # laundered by the manifest the first one wrote.
    assert _cli(sam, tmp_path, out, "--resume") == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert "polyA_merge" not in manifest["parameters"]


def test_a_legacy_directory_is_still_held_to_the_settings_it_recorded(sam, tmp_path):
    out = tmp_path / "demux"
    assert _cli(sam, tmp_path, out) == 0
    _make_legacy(out)
    with pytest.raises(ValueError, match="min_aa_length: 60 on disk, 30 now"):
        _run(sam, out, resume=True, min_aa_length=30)


def test_an_unfinished_directory_with_no_stamp_is_refused(sam, tmp_path, capsys):
    """New shards would sit beside ones whose trimming nothing records."""
    out = tmp_path / "demux"
    assert _cli(sam, tmp_path, out) == 0
    _make_legacy(out)
    for key in _KEYS:
        (out / key / "_SUCCESS").unlink()
    victim = next((out / "read_demux").glob("part-*.parquet"))
    victim.unlink()
    before = _shard_times(out)

    assert _cli(sam, tmp_path, out, "--resume") == 2
    err = capsys.readouterr().err
    assert "did not record its trimming settings" in err
    assert "demultiplexing did not finish" in err
    assert _shard_times(out) == before and not victim.exists()
    assert not (out / DEMUX_SETTINGS).exists()


def test_a_finished_directory_with_no_stamp_and_no_manifest_is_refused(sam, tmp_path):
    out = tmp_path / "demux"
    _run(sam, out)  # the library path writes no manifest
    (out / DEMUX_SETTINGS).unlink()
    with pytest.raises(ValueError, match="so is a readable manifest.json"):
        _run(sam, out, resume=True)


def test_a_stampless_directory_whose_manifest_is_current_is_adopted(sam, tmp_path):
    """Written after the fix and before the stamp: its manifest already
    says ``gap`` with these parameters, field for field."""
    out = tmp_path / "demux"
    assert _cli(sam, tmp_path, out) == 0
    (out / DEMUX_SETTINGS).unlink()
    assert _cli(sam, tmp_path, out, "--resume") == 0
    stamp = json.loads((out / DEMUX_SETTINGS).read_text())
    assert stamp["polyA_merge"] == "gap"
    assert (
        json.loads((out / "manifest.json").read_text())["parameters"]["polyA_merge"]
        == "gap"
    )

    # The same directory, had its design said something else at the time.
    (out / DEMUX_SETTINGS).unlink()
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["parameters"]["polyA_edge_distance"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="polyA_edge_distance: 1 on disk, 2 now"):
        _run(sam, out, resume=True)


def test_resume_on_an_empty_directory_is_a_fresh_run(tmp_path):
    settings = demux_settings("cdna_wilburn_v1", min_aa_length=60, min_protein_count=2)
    got = check_demux_resume(tmp_path / "new", settings, resume=True)
    assert got == settings
    assert json.loads((tmp_path / "new" / DEMUX_SETTINGS).read_text()) == settings
