"""Tests for ``constellation.sequencing.quant.gene_matrix``.

Pure-Python — no external tool dependencies.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.quant.gene_matrix import (
    build_gene_matrix,
    render_gene_matrix_tsv,
)
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.samples import Samples


def _make_genome():
    contigs = pa.Table.from_pylist(
        [
            {"contig_id": 0, "name": "chr1", "length": 1000, "topology": None, "circular": None},
            {"contig_id": 1, "name": "chr2", "length": 800, "topology": None, "circular": None},
        ]
    )
    sequences = pa.Table.from_pylist(
        [
            {"contig_id": 0, "sequence": "A" * 1000},
            {"contig_id": 1, "sequence": "C" * 800},
        ]
    )
    return GenomeReference(contigs=contigs, sequences=sequences)


def _make_annotation(*, with_attrs: bool = False):
    rows = [
        {
            "feature_id": 100,
            "contig_id": 0,
            "start": 50,
            "end": 250,
            "strand": "+",
            "type": "gene",
            "name": "geneA",
            "parent_id": None,
            "source": "test",
            "score": None,
            "phase": None,
            "attributes_json": json.dumps({"gene_biotype": "protein_coding"}) if with_attrs else None,
        },
        {
            "feature_id": 200,
            "contig_id": 0,
            "start": 400,
            "end": 600,
            "strand": "-",
            "type": "gene",
            "name": None,  # exercise fallback to attributes_json.Name
            "parent_id": None,
            "source": "test",
            "score": None,
            "phase": None,
            "attributes_json": json.dumps({"Name": "geneB_from_attrs", "gene_biotype": "ncRNA"}) if with_attrs else None,
        },
        {
            "feature_id": 300,
            "contig_id": 1,
            "start": 100,
            "end": 200,
            "strand": "+",
            "type": "gene",
            "name": None,  # exercise fallback to gene_{id}
            "parent_id": None,
            "source": "test",
            "score": None,
            "phase": None,
            "attributes_json": None,
        },
    ]
    return Annotation(features=pa.Table.from_pylist(rows))


def _make_samples():
    return Samples.from_records(
        samples=[
            {"sample_id": 1, "sample_name": "alpha", "description": None},
            {"sample_id": 2, "sample_name": "beta", "description": None},
        ],
        edges=[
            {"sample_id": 1, "acquisition_id": 1, "barcode_id": None},
            {"sample_id": 2, "acquisition_id": 1, "barcode_id": None},
        ],
    )


def _make_feature_quant(rows):
    """Build a FEATURE_QUANT-shaped table from minimal row dicts."""
    full_rows = []
    for r in rows:
        full_rows.append(
            {
                "feature_id": int(r["feature_id"]),
                "sample_id": int(r["sample_id"]),
                "engine": r.get("engine", "constellation_overlap"),
                "feature_origin": r.get("feature_origin", "gene_id"),
                "count": float(r["count"]),
                "tpm": float(r["tpm"]) if r.get("tpm") is not None else None,
                "cpm": None,
                "coverage_mean": None,
                "coverage_median": None,
                "coverage_fraction": None,
                "multimap_fraction": None,
            }
        )
    from constellation.sequencing.schemas.quant import FEATURE_QUANT
    return pa.Table.from_pylist(full_rows, schema=FEATURE_QUANT)


# ──────────────────────────────────────────────────────────────────────
# build_gene_matrix
# ──────────────────────────────────────────────────────────────────────


def test_build_gene_matrix_wide_shape() -> None:
    """8 annotation columns + N sample columns; one row per annotated gene."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant(
        [
            {"feature_id": 100, "sample_id": 1, "count": 10, "tpm": 5e5},
            {"feature_id": 100, "sample_id": 2, "count": 20, "tpm": 4e5},
            {"feature_id": 200, "sample_id": 1, "count": 10, "tpm": 5e5},
        ]
    )
    matrix = build_gene_matrix(fq, annotation, genome, samples)

    # 8 annotation cols + 2 sample cols = 10 columns
    assert matrix.num_columns == 10
    assert matrix.column_names == [
        "feature_id", "gene_name", "gene_biotype", "contig",
        "start", "end", "strand", "length",
        "alpha", "beta",
    ]
    # 3 genes in annotation → 3 rows (gene 300 zero-filled)
    assert matrix.num_rows == 3


def test_build_gene_matrix_orders_rows_by_feature_id_ascending() -> None:
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant(
        [{"feature_id": 100, "sample_id": 1, "count": 5, "tpm": 1e6}]
    )
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    assert matrix.column("feature_id").to_pylist() == [100, 200, 300]


def test_build_gene_matrix_zero_fills_unobserved_genes() -> None:
    """Genes in annotation but absent from feature_quant get 0.0 cells."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant(
        [{"feature_id": 100, "sample_id": 1, "count": 5, "tpm": 1e6}]
    )
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    # gene 200 / 300 absent from fq → all sample cells are 0.0
    rows_by_fid = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows_by_fid[200]["alpha"] == 0.0
    assert rows_by_fid[200]["beta"] == 0.0
    assert rows_by_fid[300]["alpha"] == 0.0


def test_build_gene_matrix_min_count_filter() -> None:
    """min_count drops gene rows whose total count across samples is below threshold."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant(
        [
            {"feature_id": 100, "sample_id": 1, "count": 100, "tpm": 9e5},
            {"feature_id": 200, "sample_id": 1, "count": 1, "tpm": 1e4},
        ]
    )
    matrix = build_gene_matrix(fq, annotation, genome, samples, min_count=2)
    # gene 100 (count=100) kept; gene 200 (count=1) dropped; gene 300 (count=0) dropped
    assert matrix.column("feature_id").to_pylist() == [100]


def test_build_gene_matrix_resolves_gene_name_from_table_column() -> None:
    """FEATURE_TABLE.name is the first-choice gene name source."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    rows = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows[100]["gene_name"] == "geneA"


def test_build_gene_matrix_resolves_gene_name_from_attributes_fallback() -> None:
    """When FEATURE_TABLE.name is null, use attributes_json's Name= entry."""
    genome = _make_genome()
    annotation = _make_annotation(with_attrs=True)
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    rows = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows[200]["gene_name"] == "geneB_from_attrs"


def test_build_gene_matrix_resolves_gene_name_with_id_fallback() -> None:
    """When neither name nor attributes_json.Name is set, fall back to gene_{id}."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    rows = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows[300]["gene_name"] == "gene_300"


def test_build_gene_matrix_parses_gene_biotype() -> None:
    genome = _make_genome()
    annotation = _make_annotation(with_attrs=True)
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    rows = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows[100]["gene_biotype"] == "protein_coding"
    assert rows[200]["gene_biotype"] == "ncRNA"
    assert rows[300]["gene_biotype"] is None


def test_build_gene_matrix_value_count_vs_tpm() -> None:
    """value='count' fills cells with raw counts; value='tpm' with TPM."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant(
        [{"feature_id": 100, "sample_id": 1, "count": 42, "tpm": 999.5}]
    )
    counts = build_gene_matrix(fq, annotation, genome, samples, value="count")
    tpms = build_gene_matrix(fq, annotation, genome, samples, value="tpm")
    counts_rows = {r["feature_id"]: r for r in counts.to_pylist()}
    tpms_rows = {r["feature_id"]: r for r in tpms.to_pylist()}
    assert counts_rows[100]["alpha"] == 42
    assert tpms_rows[100]["alpha"] == 999.5


def test_build_gene_matrix_count_sample_columns_are_int64() -> None:
    """value='count' produces int64 sample columns (no spurious decimals)."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([{"feature_id": 100, "sample_id": 1, "count": 5, "tpm": 1e6}])
    matrix = build_gene_matrix(fq, annotation, genome, samples, value="count")
    assert matrix.schema.field("alpha").type == pa.int64()
    assert matrix.schema.field("beta").type == pa.int64()


def test_build_gene_matrix_tpm_sample_columns_are_float64() -> None:
    """value='tpm' keeps float64 sample columns (TPM is genuinely fractional)."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([{"feature_id": 100, "sample_id": 1, "count": 5, "tpm": 1e6}])
    matrix = build_gene_matrix(fq, annotation, genome, samples, value="tpm")
    assert matrix.schema.field("alpha").type == pa.float64()
    assert matrix.schema.field("beta").type == pa.float64()


def test_build_gene_matrix_rejects_unknown_value() -> None:
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([])
    with pytest.raises(ValueError, match="value must be"):
        build_gene_matrix(fq, annotation, genome, samples, value="cpm")  # type: ignore[arg-type]


def test_build_gene_matrix_resolves_contig_name() -> None:
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    rows = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows[100]["contig"] == "chr1"
    assert rows[300]["contig"] == "chr2"


def test_build_gene_matrix_length_is_genomic_span() -> None:
    """length = end - start (informational only; TPM doesn't use this)."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    rows = {r["feature_id"]: r for r in matrix.to_pylist()}
    assert rows[100]["length"] == 200  # 250 - 50
    assert rows[200]["length"] == 200  # 600 - 400
    assert rows[300]["length"] == 100  # 200 - 100


# ──────────────────────────────────────────────────────────────────────
# render_gene_matrix_tsv
# ──────────────────────────────────────────────────────────────────────


def test_render_gene_matrix_tsv_header_row() -> None:
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([{"feature_id": 100, "sample_id": 1, "count": 5, "tpm": 1e6}])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    tsv = render_gene_matrix_tsv(matrix)
    header = tsv.splitlines()[0]
    cols = header.split("\t")
    assert cols[:8] == [
        "feature_id", "gene_name", "gene_biotype", "contig",
        "start", "end", "strand", "length",
    ]
    assert cols[8:] == ["alpha", "beta"]


def test_render_gene_matrix_tsv_count_mode_renders_integers() -> None:
    """Count-mode sample columns are int64 → renderer emits no decimal points."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([{"feature_id": 100, "sample_id": 1, "count": 7, "tpm": 1e6}])
    matrix = build_gene_matrix(fq, annotation, genome, samples, value="count")
    tsv = render_gene_matrix_tsv(matrix, float_format="%.2f")
    line_for_100 = next(ln for ln in tsv.splitlines() if ln.startswith("100\t"))
    cells = line_for_100.split("\t")
    # cells[0..7] are annotation; cells[8] is alpha (count=7), cells[9] is beta (0)
    assert cells[8] == "7"
    assert cells[9] == "0"


def test_render_gene_matrix_tsv_tpm_mode_applies_float_format() -> None:
    """TPM-mode sample columns are float64 → renderer uses float_format."""
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant(
        [{"feature_id": 100, "sample_id": 1, "count": 7, "tpm": 1234.5678}]
    )
    matrix = build_gene_matrix(fq, annotation, genome, samples, value="tpm")
    tsv = render_gene_matrix_tsv(matrix, float_format="%.2f")
    line_for_100 = next(ln for ln in tsv.splitlines() if ln.startswith("100\t"))
    cells = line_for_100.split("\t")
    assert cells[8] == "1234.57"
    assert cells[9] == "0.00"


def test_render_gene_matrix_tsv_emits_trailing_newline() -> None:
    genome = _make_genome()
    annotation = _make_annotation()
    samples = _make_samples()
    fq = _make_feature_quant([])
    tsv = render_gene_matrix_tsv(build_gene_matrix(fq, annotation, genome, samples))
    assert tsv.endswith("\n")


def test_render_gene_matrix_tsv_handles_null_biotype_as_empty_string() -> None:
    genome = _make_genome()
    annotation = _make_annotation()  # no attributes_json → null biotype
    samples = _make_samples()
    fq = _make_feature_quant([])
    matrix = build_gene_matrix(fq, annotation, genome, samples)
    tsv = render_gene_matrix_tsv(matrix)
    line_for_100 = next(ln for ln in tsv.splitlines() if ln.startswith("100\t"))
    # Position 2 is gene_biotype (null) → empty string between tabs
    cells = line_for_100.split("\t")
    assert cells[2] == ""
