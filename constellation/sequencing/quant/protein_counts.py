"""Protein-counts I/O + TPM normalization for the transcriptome→proteomics
pipeline's Stage 1.

Two input shapes are supported, sniffed by extension/dir layout:

  * Wide TSV (cartographer-compat) — the ``protein_counts.tsv`` written
    by ``constellation transcriptome demultiplex`` (and historically by
    cartographer's NanoporeAnalysis port). Format::

        \\tProtein\\tqJS001\\tqJS002\\t...\\tSequence
        0\\tP0\\t5.0\\t3.0\\t...\\tMKLIGHTPEPT...
        1\\tP1\\t2.0\\t1.0\\t...\\tACDEFGHIK...

    Leading unnamed index column (pandas default); per-sample columns
    sit between ``Protein`` and ``Sequence``; counts are stored as
    floats but represent integers (NA / cartographer pandas convention).

  * Long parquet (Constellation-native) — the ``feature_quant.parquet``
    written by the demux pipeline. Conforms to ``PROTEIN_COUNT_TABLE``
    (``protein_label, protein_sequence, sample_id, sample_name, count``).

Both project into a uniform long-format Arrow table::

    pa.schema([
        pa.field("protein_id",   pa.string(),   nullable=False),
        pa.field("sequence",     pa.string(),   nullable=False),
        pa.field("sample_name",  pa.string(),   nullable=False),
        pa.field("count",        pa.int64(),    nullable=False),
    ])

``tpm_normalize`` adds a ``tpm`` (float64) column using the long-read
convention ``count × 1e6 / sum_in_sample(count)`` after an optional
``min_sequence_length`` filter (matches cartographer's pre-CPM filter
that excludes truncated ORFs from the denominator).

No pandas — pyarrow.csv + pyarrow.compute end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq


# Long-format output schema. Distinct from PROTEIN_COUNT_TABLE
# (which carries the sample_id int FK + the P0/P1 label) — here we
# carry the human-readable sample_name only, mirroring what the wide
# TSV gives us and what downstream stages need.
PROTEIN_COUNTS_LONG_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("protein_id", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("sample_name", pa.string(), nullable=False),
        pa.field("count", pa.int64(), nullable=False),
    ],
    metadata={b"schema_name": b"ProteinCountsLongTable"},
)


def read_protein_counts_tab(path: Path | str) -> pa.Table:
    """Read a protein-counts artefact into the long-format Arrow table.

    Sniffs the input shape by extension/dir layout:

      * ``.tsv`` / ``.tab`` / ``.txt`` — wide cartographer-shaped TSV.
      * ``.parquet`` — long-format ``PROTEIN_COUNT_TABLE`` shards.
      * Directory — looks for ``feature_quant.parquet`` first, then
        ``protein_counts.tsv`` (the dual demux-output convention).

    Returns a table conforming to :data:`PROTEIN_COUNTS_LONG_SCHEMA`.
    """
    p = Path(path)
    if p.is_dir():
        feature_quant = p / "feature_quant.parquet"
        if feature_quant.is_file():
            return _read_long_parquet(feature_quant)
        tsv = p / "protein_counts.tsv"
        if tsv.is_file():
            return _read_wide_tsv(tsv)
        raise FileNotFoundError(
            f"no protein-counts artefact found under {p}: looked for "
            f"feature_quant.parquet and protein_counts.tsv"
        )
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return _read_long_parquet(p)
    if suffix in {".tsv", ".tab", ".txt"}:
        return _read_wide_tsv(p)
    raise ValueError(
        f"unsupported protein-counts input {p!r}: expected .tsv, .tab, "
        f".txt, .parquet, or a directory containing feature_quant.parquet "
        f"or protein_counts.tsv"
    )


def _read_wide_tsv(path: Path) -> pa.Table:
    """Parse the wide cartographer-shaped TSV → long-format Arrow."""
    # Use pa.csv (no pandas). The leading unnamed index column has no
    # header; pa.csv handles this by treating the first tab as empty.
    table = pa_csv.read_csv(
        str(path),
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
    )
    # Drop the pandas-emitted index column. It has the literal name
    # "" in pa.csv's column_names. Some non-pandas writers omit it
    # entirely — tolerate that path.
    if "" in table.column_names:
        table = table.drop([""])
    cols = table.column_names
    if "Protein" not in cols or "Sequence" not in cols:
        raise ValueError(
            f"TSV {path!r} missing required columns 'Protein'/'Sequence'; "
            f"got {cols}"
        )
    sample_cols = [c for c in cols if c not in {"Protein", "Sequence"}]
    if not sample_cols:
        # Empty matrix — return correctly-shaped empty table.
        return PROTEIN_COUNTS_LONG_SCHEMA.empty_table()

    # Melt: stack each sample column as a long row. pa.compute doesn't
    # ship a native melt, but we can build it via repeated arrays.
    protein_arr = table.column("Protein")
    sequence_arr = table.column("Sequence")
    n_rows = table.num_rows

    protein_blocks: list[pa.Array] = []
    sequence_blocks: list[pa.Array] = []
    sample_blocks: list[pa.Array] = []
    count_blocks: list[pa.Array] = []
    for sample in sample_cols:
        col = table.column(sample)
        # Float-or-int → int64. cartographer / NA writes "5.0" style.
        count = pc.cast(col, pa.float64())
        count = pc.cast(count, pa.int64())
        protein_blocks.append(protein_arr)
        sequence_blocks.append(sequence_arr)
        sample_blocks.append(
            pa.array([sample] * n_rows, type=pa.string())
        )
        count_blocks.append(count)
    out = pa.table(
        {
            "protein_id": pa.concat_arrays(_to_chunks(protein_blocks)),
            "sequence": pa.concat_arrays(_to_chunks(sequence_blocks)),
            "sample_name": pa.concat_arrays(sample_blocks),
            "count": pa.concat_arrays(_to_chunks(count_blocks)),
        }
    )
    return out.cast(PROTEIN_COUNTS_LONG_SCHEMA)


def _to_chunks(arrays: list[pa.Array]) -> list[pa.Array]:
    """Concat-flatten any ChunkedArrays so pa.concat_arrays accepts them.

    ``pa.concat_arrays`` requires :class:`pa.Array`, not ``ChunkedArray``
    — but ``Table.column(...)`` returns ChunkedArray. Combine chunks
    first so the concat across sample columns works."""
    out: list[pa.Array] = []
    for a in arrays:
        if isinstance(a, pa.ChunkedArray):
            out.append(a.combine_chunks())
        else:
            out.append(a)
    return out


def _read_long_parquet(path: Path) -> pa.Table:
    """Read PROTEIN_COUNT_TABLE-shaped parquet → long-format projection."""
    table = pq.read_table(path)
    cols = table.column_names
    if "protein_label" in cols and "protein_sequence" in cols:
        # Native PROTEIN_COUNT_TABLE shape: rename onto the long schema.
        out = pa.table(
            {
                "protein_id": table.column("protein_label"),
                "sequence": table.column("protein_sequence"),
                "sample_name": table.column("sample_name"),
                "count": table.column("count"),
            }
        )
    else:
        # Already long? Verify required columns and project.
        required = {"protein_id", "sequence", "sample_name", "count"}
        missing = required - set(cols)
        if missing:
            raise ValueError(
                f"parquet {path!r} missing required columns {sorted(missing)}; "
                f"got {cols}"
            )
        out = table.select(["protein_id", "sequence", "sample_name", "count"])
    return out.cast(PROTEIN_COUNTS_LONG_SCHEMA)


def tpm_normalize(
    long_counts: pa.Table,
    *,
    min_sequence_length: int | None = 100,
) -> pa.Table:
    """Add a per-sample ``tpm`` column to a long-format counts table.

    Long-read TPM convention (matches the gene-counting code path in
    :mod:`constellation.sequencing.quant.genome_count`)::

        tpm[i,j] = count[i,j] * 1e6 / sum_in_sample(count[*,j])

    No length normalisation — one read = one transcript / protein in
    long-read data, so the classic length-corrected TPM doesn't apply.
    Cartographer historically called this "CPM"; Constellation
    standardises on "TPM" across all of ``sequencing.quant`` and
    downstream consumers (math unchanged; name normalised).

    Parameters
    ----------
    long_counts
        :data:`PROTEIN_COUNTS_LONG_SCHEMA`-shaped table.
    min_sequence_length
        When set (default 100), filter out rows whose ``sequence``
        length is below the threshold BEFORE computing per-sample totals.
        This matches cartographer's pre-CPM filter — short ORFs would
        otherwise inflate the denominator with low-confidence calls.
        Pass ``None`` to skip the filter (all rows enter the sum).

    Returns
    -------
    pa.Table
        Same schema as ``long_counts`` plus a ``tpm: float64`` column.
        Rows filtered by ``min_sequence_length`` are dropped from the
        output entirely.
    """
    if long_counts.num_rows == 0:
        return long_counts.append_column(
            "tpm", pa.array([], type=pa.float64())
        )

    table = long_counts
    if min_sequence_length is not None:
        seq_len = pc.utf8_length(table.column("sequence"))
        mask = pc.greater_equal(seq_len, pa.scalar(min_sequence_length))
        table = table.filter(mask)
        if table.num_rows == 0:
            return table.append_column(
                "tpm", pa.array([], type=pa.float64())
            )

    # Per-sample total. group_by(sample_name).sum(count).
    sums = (
        table.group_by(["sample_name"])
        .aggregate([("count", "sum")])
        .rename_columns(["sample_name", "_sample_total"])
    )
    # Join back to broadcast per-row.
    joined = table.join(sums, keys="sample_name")
    counts_f = pc.cast(joined.column("count"), pa.float64())
    totals_f = pc.cast(joined.column("_sample_total"), pa.float64())
    tpm = pc.divide(pc.multiply(counts_f, pa.scalar(1e6)), totals_f)
    # Drop the helper column, append tpm.
    return joined.drop(["_sample_total"]).append_column("tpm", tpm)


def build_tpm_matrix(long_tpm: pa.Table) -> pa.Table:
    """Pivot a long counts+tpm table into a wide per-protein summary.

    One row per ``(protein_id, sequence)`` with one integer count column
    per sample, plus ``avg_tpm`` — the mean TPM across the samples in
    which the protein appears (same denominator as the Stage-2 novelty
    filter ``group_by(protein_id).mean(tpm)``). ``protein_id`` is the
    transcript-derived ORF id in the long-read workflow.

    Human-facing terminal summary — single-file aggregation is
    intentional here (CLAUDE.md invariant #3 reserves it for exports),
    not a between-stage handoff.
    """
    empty_cols = {
        "protein_id": pa.array([], type=pa.string()),
        "sequence": pa.array([], type=pa.string()),
        "avg_tpm": pa.array([], type=pa.float64()),
    }
    if long_tpm.num_rows == 0:
        return pa.table(empty_cols)

    samples = sorted(set(long_tpm.column("sample_name").to_pylist()))
    avg_by_id = {
        r["protein_id"]: r["avg_tpm"]
        for r in (
            long_tpm.group_by(["protein_id"])
            .aggregate([("tpm", "mean")])
            .rename_columns(["protein_id", "avg_tpm"])
            .to_pylist()
        )
    }

    pid_c = long_tpm.column("protein_id").to_pylist()
    seq_c = long_tpm.column("sequence").to_pylist()
    smp_c = long_tpm.column("sample_name").to_pylist()
    cnt_c = long_tpm.column("count").to_pylist()

    per_sample: dict[str, dict[str, int]] = {}
    seq_by_id: dict[str, str] = {}
    order: list[str] = []
    for pid, seq, smp, cnt in zip(pid_c, seq_c, smp_c, cnt_c, strict=True):
        if pid not in per_sample:
            per_sample[pid] = {}
            seq_by_id[pid] = seq
            order.append(pid)
        per_sample[pid][smp] = cnt

    cols: dict[str, pa.Array] = {
        "protein_id": pa.array(order, type=pa.string()),
        "sequence": pa.array([seq_by_id[p] for p in order], type=pa.string()),
    }
    for s in samples:
        cols[s] = pa.array(
            [per_sample[p].get(s, 0) for p in order], type=pa.int64()
        )
    cols["avg_tpm"] = pa.array(
        [avg_by_id.get(p, 0.0) for p in order], type=pa.float64()
    )
    return pa.table(cols)


def render_tpm_matrix_tsv(matrix: pa.Table) -> str:
    """Render :func:`build_tpm_matrix` output as a TSV string.

    Floats (``avg_tpm``) print with 4 decimals; everything else via
    ``str``. Header row first, one protein per line.
    """
    names = matrix.column_names
    cols = [matrix.column(n).to_pylist() for n in names]
    lines = ["\t".join(names)]
    for i in range(matrix.num_rows):
        lines.append(
            "\t".join(
                f"{c[i]:.4f}" if isinstance(c[i], float) else str(c[i])
                for c in cols
            )
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "PROTEIN_COUNTS_LONG_SCHEMA",
    "build_tpm_matrix",
    "read_protein_counts_tab",
    "render_tpm_matrix_tsv",
    "tpm_normalize",
]
