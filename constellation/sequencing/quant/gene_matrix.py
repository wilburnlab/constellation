"""Wide gene × sample matrix builder + TSV renderer.

Sibling to :mod:`genome_count` (which produces the long-form
:data:`FEATURE_QUANT` rows). Where ``count_reads_per_gene`` outputs one
row per ``(feature_id, sample_id)``, this module pivots that into the
familiar gene-rows / sample-columns shape that downstream stats tools
(DESeq2 / edgeR / limma) and human readers expect, with annotation
columns prepended to each gene row.

Two functions:

    build_gene_matrix(feature_quant, annotation, genome, samples, ...)
        Pivot + left-join. Returns a wide Arrow table with annotation
        columns followed by one column per sample.

    render_gene_matrix_tsv(matrix)
        Render the wide table to a TSV string.

TPM is *not* computed here — by the time a caller reaches this module,
:func:`count_reads_per_gene` has already populated the ``tpm`` column.
``value='count'`` and ``value='tpm'`` just pick which existing column
to project into the wide cells.

Future annotation-overlay variants (signal peptides, GO terms, domain
hits, ...) extend by left-joining additional annotation tables onto the
gene rows; the wide pivot stays unchanged.
"""

from __future__ import annotations

import json
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.samples import Samples


def _resolve_gene_name(name: str | None, attributes_json: str | None, feature_id: int) -> str:
    """Pick a display name: FEATURE_TABLE.name, then attributes_json's
    ``Name=`` (GFF3 standard), then ``f"gene_{feature_id}"`` fallback."""
    if name:
        return name
    if attributes_json:
        try:
            attrs = json.loads(attributes_json)
        except (ValueError, TypeError):
            attrs = None
        if isinstance(attrs, dict):
            n = attrs.get("Name") or attrs.get("name")
            if isinstance(n, str) and n:
                return n
    return f"gene_{feature_id}"


def _resolve_gene_biotype(attributes_json: str | None) -> str | None:
    """Parse ``gene_biotype`` (Ensembl) or ``biotype`` from attributes_json."""
    if not attributes_json:
        return None
    try:
        attrs = json.loads(attributes_json)
    except (ValueError, TypeError):
        return None
    if not isinstance(attrs, dict):
        return None
    v = attrs.get("gene_biotype") or attrs.get("biotype")
    return v if isinstance(v, str) and v else None


def build_gene_matrix(
    feature_quant: pa.Table,
    annotation: Annotation,
    genome: GenomeReference,
    samples: Samples,
    *,
    value: Literal["count", "tpm"] = "count",
    min_count: int = 0,
) -> pa.Table:
    """Pivot a long-form ``FEATURE_QUANT`` table into wide gene × sample.

    ``feature_quant`` must already have the requested ``value`` column
    populated (``count`` always; ``tpm`` after ``count_reads_per_gene``
    runs). Only rows with ``feature_origin='gene_id'`` participate;
    other origins are silently skipped — callers that need cross-origin
    matrices should partition first.

    Output schema:

        feature_id (int64)
        gene_name (string)        — FEATURE_TABLE.name → attrs.Name → gene_{id}
        gene_biotype (string)     — attrs.gene_biotype | attrs.biotype | null
        contig (string)           — GenomeReference.contigs.name
        start, end (int64)
        strand (string)
        length (int64)            — end - start (genomic span; informational)
        <sample_name_1> (float64) — value (count or TPM)
        <sample_name_2> (float64)
        ...

    Row order: ``feature_id`` ascending. Sample columns: ``sample_id``
    ascending. Genes present in ``annotation`` but absent from
    ``feature_quant`` emit zero-filled rows so the gene index is stable
    across runs. ``min_count`` drops gene rows whose summed ``count``
    across all samples is strictly less than the threshold (always
    measured against ``count``, even when ``value='tpm'``).
    """
    if value not in ("count", "tpm"):
        raise ValueError(f"value must be 'count' or 'tpm', got {value!r}")

    # ── Annotation: pull gene rows only and project to lookup table ──
    genes = annotation.features_of_type("gene")
    feature_ids = genes.column("feature_id").to_pylist()
    contig_ids = genes.column("contig_id").to_pylist()
    starts = genes.column("start").to_pylist()
    ends = genes.column("end").to_pylist()
    strands = genes.column("strand").to_pylist()
    names = genes.column("name").to_pylist()
    attrs = genes.column("attributes_json").to_pylist()

    contig_id_to_name = dict(
        zip(
            genome.contigs.column("contig_id").to_pylist(),
            genome.contigs.column("name").to_pylist(),
            strict=True,
        )
    )

    # Build the annotation skeleton, ordered by feature_id ascending so
    # the row index is deterministic across runs.
    skeleton: dict[int, dict[str, object]] = {}
    for fid, cid, s, e, strand, nm, aj in zip(
        feature_ids, contig_ids, starts, ends, strands, names, attrs, strict=True
    ):
        fid_int = int(fid)
        skeleton[fid_int] = {
            "feature_id": fid_int,
            "gene_name": _resolve_gene_name(nm, aj, fid_int),
            "gene_biotype": _resolve_gene_biotype(aj),
            "contig": contig_id_to_name.get(int(cid)),
            "start": int(s),
            "end": int(e),
            "strand": strand,
            "length": int(e) - int(s),
        }

    # ── Sample columns: sample_id ascending, name from Samples.samples
    samples_table = samples.samples
    sid_to_name = dict(
        zip(
            samples_table.column("sample_id").to_pylist(),
            samples_table.column("sample_name").to_pylist(),
            strict=True,
        )
    )
    sample_ids_sorted = sorted(sid_to_name.keys())
    sample_columns = [sid_to_name[sid] for sid in sample_ids_sorted]

    # ── Pivot: filter to gene_id rows + project (feature_id, sample_id, value)
    fq = feature_quant
    if "feature_origin" in fq.column_names:
        fq = fq.filter(pc.equal(fq.column("feature_origin"), "gene_id"))

    # value-column projection
    value_col = "count" if value == "count" else "tpm"
    if value_col not in fq.column_names:
        raise ValueError(
            f"feature_quant lacks '{value_col}' column; populate via "
            f"count_reads_per_gene before building a {value!r} matrix"
        )

    # Build (feature_id, sample_id) → value AND (feature_id) → total_count.
    # Counts are integer-valued by construction (group_by counts rows);
    # cast to int so the wide matrix carries the right dtype and the TSV
    # renderer doesn't print spurious decimal points.
    is_count_mode = value == "count"
    fq_rows = fq.select(["feature_id", "sample_id", "count", value_col]).to_pylist()
    cell_value: dict[tuple[int, int], int | float] = {}
    total_count: dict[int, float] = {}
    for row in fq_rows:
        fid_int = int(row["feature_id"])
        sid_int = int(row["sample_id"])
        v = row[value_col]
        if v is None:
            cell_value[(fid_int, sid_int)] = 0 if is_count_mode else 0.0
        elif is_count_mode:
            cell_value[(fid_int, sid_int)] = int(v)
        else:
            cell_value[(fid_int, sid_int)] = float(v)
        total_count[fid_int] = total_count.get(fid_int, 0.0) + (
            float(row["count"]) if row["count"] is not None else 0.0
        )

    # ── Compose wide rows in feature_id order, applying min_count filter
    zero_cell: int | float = 0 if is_count_mode else 0.0
    out_rows: list[dict[str, object]] = []
    for fid_int in sorted(skeleton):
        if total_count.get(fid_int, 0.0) < min_count:
            continue
        ann = skeleton[fid_int]
        row: dict[str, object] = dict(ann)
        for sid in sample_ids_sorted:
            row[sid_to_name[sid]] = cell_value.get((fid_int, sid), zero_cell)
        out_rows.append(row)

    # ── Materialise as Arrow with explicit schema (deterministic dtypes).
    # Sample columns are int64 in count mode, float64 in TPM mode — the
    # renderer dispatches on column dtype.
    fields: list[pa.Field] = [
        pa.field("feature_id", pa.int64(), nullable=False),
        pa.field("gene_name", pa.string(), nullable=False),
        pa.field("gene_biotype", pa.string(), nullable=True),
        pa.field("contig", pa.string(), nullable=True),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("length", pa.int64(), nullable=False),
    ]
    sample_dtype = pa.int64() if is_count_mode else pa.float64()
    for col in sample_columns:
        fields.append(pa.field(col, sample_dtype, nullable=False))
    schema = pa.schema(fields)

    if not out_rows:
        return schema.empty_table()
    return pa.Table.from_pylist(out_rows, schema=schema)


def render_gene_matrix_tsv(
    matrix: pa.Table,
    *,
    float_format: str = "%.4f",
) -> str:
    """Render a wide gene matrix (output of :func:`build_gene_matrix`) to TSV.

    Cell formatting dispatches on Arrow column dtype:

      * integer columns (annotation ints + count-mode sample columns) →
        ``str(int)`` — no spurious decimal points
      * floating columns (tpm-mode sample columns) → ``float_format``
      * string columns → unchanged
      * null cells → empty string

    Header is the column names verbatim. Includes a trailing newline.
    """
    columns = list(matrix.column_names)
    schema = matrix.schema
    is_float_col = [pa.types.is_floating(schema.field(c).type) for c in columns]

    lines = ["\t".join(columns)]
    if matrix.num_rows == 0:
        return lines[0] + "\n"

    column_data: list[list[object]] = [
        matrix.column(c).to_pylist() for c in columns
    ]
    for i in range(matrix.num_rows):
        cells: list[str] = []
        for is_float, col_values in zip(is_float_col, column_data, strict=True):
            v = col_values[i]
            if v is None:
                cells.append("")
            elif is_float:
                cells.append(float_format % float(v))
            else:
                cells.append(str(v))
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


__all__ = ["build_gene_matrix", "render_gene_matrix_tsv"]
