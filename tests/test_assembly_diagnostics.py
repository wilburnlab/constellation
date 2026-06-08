"""Assembly diagnostics — section functions, comparative report, _safe."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.diagnostics import (
    _safe,
    generate_assembly_report,
    generate_comparative_report,
    section_pipeline_comparison,
)
from constellation.sequencing.schemas.assembly import ASSEMBLY_STATS

pytest.importorskip("matplotlib")


def _assembly_with_coverage() -> Assembly:
    contigs = pa.table(
        {
            "contig_id": pa.array([0, 1, 2], type=pa.int64()),
            "name": ["c0", "c1", "c2"],
            "length": pa.array([6000, 5000, 4000], type=pa.int64()),
            "read_coverage": pa.array([35.0, 40.0, 5.0], type=pa.float32()),
        }
    )
    sequences = pa.table(
        {
            "contig_id": pa.array([0, 1, 2], type=pa.int64()),
            "sequence": ["GC" * 3000, "AT" * 2500, "ACGT" * 1000],
        }
    )
    return Assembly.from_tables(contigs, sequences)


def _stats_row(*, n50, n_contigs, total=15000, largest=6000, gc=0.5, busco=None):
    d = {
        "assembly_id": pa.array([0], type=pa.int64()),
        "n_contigs": pa.array([n_contigs], type=pa.int32()),
        "total_length": pa.array([total], type=pa.int64()),
        "n50": pa.array([n50], type=pa.int64()),
        "l50": pa.array([1], type=pa.int32()),
        "largest_contig": pa.array([largest], type=pa.int64()),
        "gc_content": pa.array([gc], type=pa.float32()),
    }
    if busco is not None:
        d["busco_complete"] = pa.array([busco], type=pa.float32())
    return cast_to_schema(pa.table(d), ASSEMBLY_STATS)


def test_generate_assembly_report(tmp_path: Path):
    report = generate_assembly_report(
        _assembly_with_coverage(), output_dir=tmp_path / "diag", stage_label="draft"
    )
    assert report.exists()
    text = report.read_text()
    assert "Contiguity" in text
    assert "N50" in text
    assert "GC content" in text
    assert "Read coverage" in text
    # a figure was emitted
    assert (tmp_path / "diag" / "figures" / "contiguity_cumulative.svg").exists()


def test_comparative_report_flags_regressions(tmp_path: Path):
    stage_stats = {
        "draft": _stats_row(n50=1000, n_contigs=10, busco=0.95),
        "scaffold": _stats_row(n50=5000, n_contigs=3, busco=0.95),
        "polish": _stats_row(n50=4000, n_contigs=3, busco=0.90),  # both regress
    }
    report = generate_comparative_report(stage_stats, output_dir=tmp_path / "cmp")
    text = report.read_text()
    assert "Pipeline comparison" in text
    assert "N50 regressed scaffold→polish" in text
    assert "BUSCO completeness dropped scaffold→polish" in text


def test_comparison_section_orders_stages(tmp_path: Path):
    stage_stats = {
        "polish": _stats_row(n50=4000, n_contigs=3),
        "draft": _stats_row(n50=1000, n_contigs=10),
    }
    sec = section_pipeline_comparison(stage_stats, figures_dir=tmp_path / "f")
    # draft column comes before polish regardless of dict insertion order
    header = sec.body.splitlines()[0]
    assert header.index("draft") < header.index("polish")


def test_safe_swallows_broken_metric():
    def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    boom.__name__ = "section_boom"
    sec = _safe(boom)
    assert "failed" in sec.body
    assert "kaboom" in sec.body
