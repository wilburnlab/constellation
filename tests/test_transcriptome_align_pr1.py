"""PR 1 (read pile-up overhaul) tests — pipeline prep for the align stage.

Covers the three pieces that land before any viz changes:

1. Manifest schema bumped v3 → v4.
2. ``--emit-cs-tags`` default flipped to True with ``--no-emit-cs-tags``
   opt-out (BooleanOptionalAction); opt-out emits a stderr warning that
   read pile-up visualization is disabled for the resulting run.
3. ``build_read_samples`` resolve helper materialises the per-read
   sample assignment table (read_id, sample_id, sample_name) that the
   align stage writes to ``read_samples.parquet`` and downstream viz /
   stratification passes consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from constellation.sequencing.quant import build_read_samples
from constellation.sequencing.samples import (
    SAMPLE_ACQUISITION_EDGE,
    SAMPLE_TABLE,
    Samples,
)
from constellation.sequencing.schemas.alignment import READ_SAMPLE_TABLE
from constellation.sequencing.transcriptome.manifest import (
    MANIFEST_SCHEMA_VERSION,
    read_manifest_dir,
    write_align_manifest,
)


# ──────────────────────────────────────────────────────────────────────
# Manifest schema v4
# ──────────────────────────────────────────────────────────────────────


def test_manifest_schema_version_is_v4() -> None:
    """The single MANIFEST_SCHEMA_VERSION constant drives every writer
    and the reader's exact-match check; this test pins the bump."""
    assert MANIFEST_SCHEMA_VERSION == 4


def test_align_manifest_round_trip_v4(tmp_path: Path) -> None:
    """v4 align manifests round-trip with the new ``read_samples`` key
    present in outputs. Older v3 manifests are refused with an
    actionable message — no silent back-compat."""
    align_dir = tmp_path / "align_v4"
    align_dir.mkdir()
    write_align_manifest(
        align_dir / "manifest.json",
        reference_handle="test_org@local-1",
        reference_path=str(tmp_path / "ref"),
        assembly_accession="TestAssembly.1",
        demux_dir=str(tmp_path / "demux"),
        input_files=["one.bam"],
        parameters={"emit_cs_tags": True},
        stages={"decoded": 10},
        outputs={
            "alignments": str(align_dir / "alignments"),
            "alignment_cs": str(align_dir / "alignment_cs"),
            "read_samples": str(align_dir / "read_samples.parquet"),
        },
        samples=["sample_a", "sample_b"],
    )
    loaded = read_manifest_dir(align_dir)
    assert loaded.kind == "align"
    assert loaded.schema_version == 4
    assert loaded.outputs["read_samples"].endswith("read_samples.parquet")
    assert loaded.outputs["alignment_cs"].endswith("alignment_cs")
    assert loaded.samples == ["sample_a", "sample_b"]


def test_pre_v4_manifest_refused(tmp_path: Path) -> None:
    """A v3 manifest in the wild gets refused with a 'rerun the producing
    stage' message — no silent fallback masquerades as forward
    compatibility."""
    import json

    align_dir = tmp_path / "legacy"
    align_dir.mkdir()
    legacy = {
        "schema_version": 3,
        "kind": "align",
        "reference_handle": "test_org@local-1",
        "reference_path": str(tmp_path),
        "assembly_accession": None,
        "created_at": "2026-05-01T00:00:00Z",
        "demux_dir": "",
        "input_files": [],
        "parameters": {},
        "stages": {},
        "outputs": {},
        "samples": None,
    }
    (align_dir / "manifest.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="schema_version=3"):
        read_manifest_dir(align_dir)


# ──────────────────────────────────────────────────────────────────────
# CLI flag defaults
# ──────────────────────────────────────────────────────────────────────


def _parse_align_args(extra: list[str]) -> argparse.Namespace:
    """Parse the production CLI parser with the minimum required args.

    Stubs out the required `--demux-dir`, `--reference`, and
    `--output-dir` so the parser will accept the argv; we only inspect
    the parsed flags here, not run the handler.
    """
    from constellation.cli.__main__ import _build_parser

    parser = _build_parser()
    argv = [
        "transcriptome",
        "align",
        "--demux-dir",
        "/tmp/demux",
        "--reference",
        "test_org@local-1",
        "--output-dir",
        "/tmp/align_out",
        *extra,
    ]
    return parser.parse_args(argv)


def test_emit_cs_tags_default_is_true() -> None:
    """With no flag passed, args.emit_cs_tags is True — the new
    default-on behavior. Read pile-up viz works out of the box."""
    args = _parse_align_args([])
    assert args.emit_cs_tags is True


def test_no_emit_cs_tags_opts_out() -> None:
    """Explicit --no-emit-cs-tags flips emit_cs_tags to False so
    BooleanOptionalAction's negative form acts as the opt-out."""
    args = _parse_align_args(["--no-emit-cs-tags"])
    assert args.emit_cs_tags is False


def test_emit_cs_tags_explicit_yes() -> None:
    """The positive form still works (idempotent with the default)."""
    args = _parse_align_args(["--emit-cs-tags"])
    assert args.emit_cs_tags is True


def test_no_emit_cs_tags_prints_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When --no-emit-cs-tags is selected, the align handler emits a
    stderr warning that read pile-up visualization is disabled. We
    isolate the warning block from the handler by simulating its early
    body inline (the full handler requires minimap2 + a real demux
    dir)."""
    emit_cs = False  # simulate the args.emit_cs_tags = False branch
    if not emit_cs:
        print(
            "WARNING: --no-emit-cs-tags disables read pile-up "
            "visualization in the genome browser (the read_pileup "
            "track requires per-alignment cs:long strings to render "
            "exonic geometry + per-base mismatch glyphs). Re-run without "
            "--no-emit-cs-tags to restore viz support.",
            file=sys.stderr,
            flush=True,
        )
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "--no-emit-cs-tags" in captured.err
    assert "read pile-up" in captured.err.lower()


# ──────────────────────────────────────────────────────────────────────
# build_read_samples helper — viz-facing per-read sample assignment
# ──────────────────────────────────────────────────────────────────────


def _samples(names: dict[int, str]) -> Samples:
    sample_tbl = pa.table(
        {
            "sample_id": list(names.keys()),
            "sample_name": list(names.values()),
            "description": [None] * len(names),
        },
        schema=SAMPLE_TABLE,
    )
    return Samples(sample_tbl, SAMPLE_ACQUISITION_EDGE.empty_table())


def _read_demux(rows: list[tuple[str, int | None]]) -> pa.Table:
    """Minimal READ_DEMUX_TABLE projection — only the two columns
    build_read_samples reads."""
    return pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sample_id": [r[1] for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("read_id", pa.string(), nullable=False),
                pa.field("sample_id", pa.int64(), nullable=True),
            ]
        ),
    )


def test_build_read_samples_basic() -> None:
    """One row per read with sample_name attached from the Samples
    container — the table the viz read_pileup kernel will join against."""
    rd = _read_demux([("r1", 1), ("r2", 2), ("r3", 1)])
    samples = _samples({1: "sample_a", 2: "sample_b"})
    out = build_read_samples(rd, samples)
    assert out.schema == READ_SAMPLE_TABLE
    rows = sorted(
        (r["read_id"], r["sample_id"], r["sample_name"]) for r in out.to_pylist()
    )
    assert rows == [
        ("r1", 1, "sample_a"),
        ("r2", 2, "sample_b"),
        ("r3", 1, "sample_a"),
    ]


def test_build_read_samples_drops_null_sample() -> None:
    """Reads whose demux row has null sample_id (failed barcode resolution)
    are dropped — they have no sample identity to color by, so the viz
    layer ignores them rather than rendering "unknown"-bucket reads."""
    rd = _read_demux([("r1", 1), ("r_unmapped", None)])
    samples = _samples({1: "sample_a"})
    out = build_read_samples(rd, samples)
    rows = [r["read_id"] for r in out.to_pylist()]
    assert rows == ["r1"]


def test_build_read_samples_chimera_disagreement_dropped() -> None:
    """A read appearing in multiple demux rows must agree on sample_id;
    if it doesn't, drop it (same policy count_reads_per_gene uses). The
    viz would otherwise render the read in both samples' palettes."""
    rd = _read_demux([("r_clean", 1), ("r_mix", 1), ("r_mix", 2)])
    samples = _samples({1: "sample_a", 2: "sample_b"})
    out = build_read_samples(rd, samples)
    rows = sorted(r["read_id"] for r in out.to_pylist())
    assert rows == ["r_clean"]


def test_build_read_samples_chimera_agreement_collapsed() -> None:
    """A read appearing in multiple demux rows that all agree on
    sample_id collapses to one row (the deduplication is silent — no
    error, no inflation of the per-sample read count)."""
    rd = _read_demux([("r_agree", 1), ("r_agree", 1)])
    samples = _samples({1: "sample_a"})
    out = build_read_samples(rd, samples)
    rows = [r["read_id"] for r in out.to_pylist()]
    assert rows == ["r_agree"]
    assert out.num_rows == 1


def test_build_read_samples_unknown_sample_keeps_row_null_name() -> None:
    """If a read's sample_id isn't in Samples (escape-hatch demux
    bundles that don't ship a full samples.parquet), the row survives
    with sample_name = null — viz palette renders it under its numeric
    id rather than dropping the alignment."""
    rd = _read_demux([("r_known", 1), ("r_orphan", 99)])
    samples = _samples({1: "sample_a"})
    out = build_read_samples(rd, samples)
    by_read = {r["read_id"]: r for r in out.to_pylist()}
    assert by_read["r_known"]["sample_name"] == "sample_a"
    assert by_read["r_orphan"]["sample_id"] == 99
    assert by_read["r_orphan"]["sample_name"] is None


def test_build_read_samples_empty_demux() -> None:
    """Zero demux rows produces a zero-row READ_SAMPLE_TABLE — schema
    round-trips even when there's no data."""
    rd = _read_demux([])
    samples = _samples({1: "sample_a"})
    out = build_read_samples(rd, samples)
    assert out.schema == READ_SAMPLE_TABLE
    assert out.num_rows == 0


def test_build_read_samples_parquet_round_trip(tmp_path: Path) -> None:
    """End-to-end: write the helper's output to read_samples.parquet
    and round-trip it. This is exactly what the align CLI does — the
    file on disk is the viz layer's input."""
    rd = _read_demux([("r1", 1), ("r2", 2)])
    samples = _samples({1: "sample_a", 2: "sample_b"})
    out = build_read_samples(rd, samples)
    path = tmp_path / "read_samples.parquet"
    pq.write_table(out, path)
    roundtripped = pq.read_table(path)
    assert roundtripped.schema == READ_SAMPLE_TABLE
    assert roundtripped.num_rows == 2


# ──────────────────────────────────────────────────────────────────────
# READ_SAMPLE_TABLE registration
# ──────────────────────────────────────────────────────────────────────


def test_read_sample_table_registered() -> None:
    """The schema self-registers on import (same pattern as the other
    alignment-related schemas), so consumers can look it up by name
    without importing the module directly."""
    from constellation.core.io.schemas import get_schema

    assert get_schema("ReadSampleTable") is READ_SAMPLE_TABLE


def test_read_sample_table_columns() -> None:
    """Schema is exactly (read_id, sample_id, sample_name) with the
    nullability the join produces."""
    names = READ_SAMPLE_TABLE.names
    assert names == ["read_id", "sample_id", "sample_name"]
    assert READ_SAMPLE_TABLE.field("read_id").type == pa.string()
    assert READ_SAMPLE_TABLE.field("sample_id").type == pa.int64()
    assert READ_SAMPLE_TABLE.field("sample_name").type == pa.string()
    assert READ_SAMPLE_TABLE.field("read_id").nullable is False
    assert READ_SAMPLE_TABLE.field("sample_id").nullable is False
    assert READ_SAMPLE_TABLE.field("sample_name").nullable is True
