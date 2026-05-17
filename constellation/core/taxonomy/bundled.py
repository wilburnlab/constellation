"""Loader for the bundled starter taxonomy.

The starter ships under ``constellation/data/taxonomy_starter.parquet``
as a single parquet file with three named tables encoded as columns:
``nodes_*``, ``names_*``, ``merged_*``. Single-file packaging keeps the
wheel layout flat (no nested ``constellation/data/taxonomy/`` dir) and
loads ~10× faster than zipping three files.

For the v1 cut, the starter is small (~100 hand-curated species across
the tree of life) — enough that the manual CLI tests + lab-priority
taxa resolve without a network round-trip. The build script at
``scripts/build-taxonomy-starter-parquet.py`` regenerates the full
~10K-taxon starter from a fresh NCBI taxdump using the curation rules
in [docs/plans/archive/.../starter-curation-rule.md]; that script is the
authoritative regen path.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.taxonomy.schemas import (
    TAXONOMY_MERGED_TABLE,
    TAXONOMY_NAMES_TABLE,
    TAXONOMY_NODES_TABLE,
)


_BUNDLED_FILENAME = "taxonomy_starter.parquet"


def load_bundled_taxonomy() -> tuple[pa.Table, pa.Table, pa.Table, dict[str, Any]]:
    """Read the bundled starter and split into ``(nodes, names, merged, meta)``.

    The bundled parquet stores three side-by-side tables, distinguished
    by a ``__table__`` column. We filter + project on load.
    """
    pkg = resources.files("constellation.data")
    with resources.as_file(pkg / _BUNDLED_FILENAME) as path:
        tbl = pq.read_table(path)
    meta_raw = tbl.schema.metadata or {}
    meta: dict[str, Any] = {}
    for k, v in meta_raw.items():
        ks = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        vs = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        meta[ks] = vs
    meta.setdefault("source", "bundled")
    meta.setdefault("schema_version", "1")

    nodes = _project_table(tbl, "nodes", TAXONOMY_NODES_TABLE)
    names = _project_table(tbl, "names", TAXONOMY_NAMES_TABLE)
    merged = _project_table(tbl, "merged", TAXONOMY_MERGED_TABLE)
    return nodes, names, merged, meta


def _project_table(combined: pa.Table, kind: str, schema: pa.Schema) -> pa.Table:
    """Pull rows tagged with ``__table__ == kind`` and project to ``schema``."""
    if "__table__" not in combined.column_names:
        raise ValueError(
            "bundled taxonomy parquet missing __table__ discriminator column"
        )
    import pyarrow.compute as pc

    mask = pc.equal(combined.column("__table__"), kind)
    sub = combined.filter(mask)
    cols = {}
    for field in schema:
        if field.name not in sub.column_names:
            raise ValueError(
                f"bundled taxonomy missing column {field.name!r} for table {kind!r}"
            )
        col = sub.column(field.name)
        if col.type != field.type:
            col = col.cast(field.type)
        cols[field.name] = col
    return pa.table(cols, schema=schema)


def write_bundled_starter(
    path: str,
    *,
    nodes: pa.Table,
    names: pa.Table,
    merged: pa.Table,
    meta: dict[str, Any] | None = None,
) -> None:
    """Combine ``nodes``/``names``/``merged`` into the single-file starter.

    Used by ``scripts/build-taxonomy-starter-parquet.py``. The combined
    table is wide (union of all columns across the three tables, plus a
    ``__table__`` discriminator); columns that don't apply to a given
    row are null.
    """
    combined = _combine_for_starter(nodes, names, merged)
    meta_bytes: dict[bytes, bytes] = {}
    for k, v in (meta or {}).items():
        meta_bytes[k.encode("utf-8")] = str(v).encode("utf-8")
    if meta_bytes:
        combined = combined.replace_schema_metadata(meta_bytes)
    pq.write_table(combined, path, compression="zstd", compression_level=15)


def _combine_for_starter(
    nodes: pa.Table, names: pa.Table, merged: pa.Table
) -> pa.Table:
    """Inner-join the three tables into one wide row format.

    Each row carries ``__table__ ∈ {"nodes", "names", "merged"}`` and
    null for columns belonging to the other two tables. Survives parquet
    write + read; loader projects back on demand.
    """
    n_nodes = nodes.num_rows
    n_names = names.num_rows
    n_merged = merged.num_rows
    total = n_nodes + n_names + n_merged

    discriminator = pa.array(
        ["nodes"] * n_nodes + ["names"] * n_names + ["merged"] * n_merged,
        type=pa.string(),
    )

    out_cols: dict[str, pa.Array] = {"__table__": discriminator}

    def _column(
        name: str, source: pa.Table, n: int, offset_before: int, offset_after: int
    ) -> pa.Array:
        if source.num_rows == 0:
            return pa.array(
                [None] * total, type=source.schema.field(name).type
            )
        if name in source.column_names:
            col = source.column(name).combine_chunks()
            dtype = source.schema.field(name).type
            head = pa.array([None] * offset_before, type=dtype)
            tail = pa.array([None] * offset_after, type=dtype)
            return pa.concat_arrays([head, col, tail])
        return pa.array([None] * total, type=pa.null())

    for f in TAXONOMY_NODES_TABLE:
        out_cols[f.name] = _column(
            f.name, nodes, n_nodes, offset_before=0, offset_after=n_names + n_merged
        )
    for f in TAXONOMY_NAMES_TABLE:
        if f.name in out_cols:
            # Column already populated for nodes rows — extend with names rows.
            existing = out_cols[f.name]
            names_col = names.column(f.name).combine_chunks() if names.num_rows else pa.array([], type=f.type)
            tail = pa.array([None] * n_merged, type=f.type)
            # Replace nulls for names rows with real values.
            head = existing.slice(0, n_nodes)
            out_cols[f.name] = pa.concat_arrays([head, names_col, tail])
        else:
            out_cols[f.name] = _column(
                f.name, names, n_names, offset_before=n_nodes, offset_after=n_merged
            )
    for f in TAXONOMY_MERGED_TABLE:
        if f.name in out_cols:
            existing = out_cols[f.name]
            merged_col = merged.column(f.name).combine_chunks() if merged.num_rows else pa.array([], type=f.type)
            head = existing.slice(0, n_nodes + n_names)
            out_cols[f.name] = pa.concat_arrays([head, merged_col])
        else:
            out_cols[f.name] = _column(
                f.name,
                merged,
                n_merged,
                offset_before=n_nodes + n_names,
                offset_after=0,
            )

    return pa.table(out_cols)


__all__ = ["load_bundled_taxonomy", "write_bundled_starter"]
