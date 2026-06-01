"""Adapters that normalize extraction sources into ``XIC_TARGET_TABLE``.

The chromatogram extractor consumes one shape — ``XIC_TARGET_TABLE`` (a
target list of what to extract). Every real source is adapted into it
here, so "general m/z+RT target" (a theoretical library) and
"search-results" (identified PSMs) are *sources*, not separate
operations:

  ``targets_from_library``        PRECURSOR_TABLE (+ PEPTIDE_TABLE) — the
                                  general theoretical target: precursor
                                  m/z + predicted RT, sample-agnostic.
  ``targets_from_search``         PSM_TABLE — identification-anchored;
                                  the only source carrying ``scan`` (so
                                  ``assigned_scans_only`` lights up).
  ``targets_from_precursor_quant``PRECURSOR_QUANT joined to a Library —
                                  re-extract already-quantified precursors
                                  at their observed RT.
  ``targets_from_table``          a bare hand-authored table validated
                                  against the contract (non-peptide m/z
                                  lists / ad-hoc targets).

``load_targets(path)`` is the convenience used by the CLI: it sniffs a
ParquetDir bundle (Library / Search) or a bare ``.parquet`` / ``.tsv``
file and dispatches to the right adapter.

The library/quant adapters map the ``-1.0`` "no prediction" sentinel
(``rt_predicted`` / ``rt_observed``) to null ``rt_center`` so absent RT
falls through to all-RT extraction rather than gating on a bogus 0-ish
window.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.massspec.quant.schemas import XIC_TARGET_TABLE

__all__ = [
    "targets_from_library",
    "targets_from_search",
    "targets_from_precursor_quant",
    "targets_from_table",
    "load_targets",
]


def _sentinel_to_null(
    arr: pa.Array | pa.ChunkedArray, sentinel: float = -1.0
) -> pa.Array:
    """Map a float sentinel (and any existing nulls) to null."""
    return pc.if_else(pc.equal(arr, sentinel), pa.scalar(None, pa.float64()), arr)


def _attach_modseq(precursors: pa.Table, peptides: pa.Table | None) -> pa.Table:
    """Left-join ``modified_sequence`` from a peptide table onto precursors."""
    if peptides is None or "modified_sequence" not in peptides.column_names:
        return precursors.append_column(
            "modified_sequence", pa.nulls(precursors.num_rows, pa.string())
        )
    pep = peptides.select(["peptide_id", "modified_sequence"])
    return precursors.join(pep, keys="peptide_id", join_type="left outer")


def targets_from_library(library: object) -> pa.Table:
    """Normalize a ``Library`` (its precursors + peptides) into targets.

    The general theoretical-target path: each precursor becomes a target
    keyed by ``precursor_id``, anchored on ``precursor_mz`` with predicted
    RT as the gating center.
    """
    precursors: pa.Table = library.precursors  # type: ignore[attr-defined]
    peptides: pa.Table = library.peptides  # type: ignore[attr-defined]
    joined = _attach_modseq(precursors, peptides)
    out = pa.table(
        {
            "target_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "precursor_charge": joined.column("charge"),
            "precursor_mz": joined.column("precursor_mz"),
            "rt_center": _sentinel_to_null(joined.column("rt_predicted")),
            "precursor_id": joined.column("precursor_id"),
            "peptide_id": joined.column("peptide_id"),
        }
    )
    return cast_to_schema(out, XIC_TARGET_TABLE)


def targets_from_search(search: object) -> pa.Table:
    """Normalize a ``Search``'s ``psms`` (PSM_TABLE) into targets.

    The identification-anchored path: each PSM becomes a target keyed by
    ``psm_id``, anchored on the measured precursor ``mz`` at its scan +
    retention time. ``scan`` is carried (the only source that does), so
    ``assigned_scans_only`` extraction is available. NOTE the PSM
    ``peptide_id`` is engine-internal — NOT a library FK — so it is
    deliberately dropped.
    """
    psms: pa.Table = search.psms  # type: ignore[attr-defined]
    out = pa.table(
        {
            "target_id": psms.column("psm_id"),
            "modified_sequence": psms.column("modified_sequence"),
            "precursor_charge": psms.column("charge"),
            "precursor_mz": psms.column("mz"),
            "rt_center": psms.column("retention_time_s"),
            "scan": psms.column("scan"),
        }
    )
    return cast_to_schema(out, XIC_TARGET_TABLE)


def targets_from_precursor_quant(quant: object, library: object) -> pa.Table:
    """Normalize ``PRECURSOR_QUANT`` (joined to a Library) into targets.

    Re-extract already-quantified precursors at their *observed* RT. The
    library supplies m/z + charge + modseq (quant rows carry only the
    ``precursor_id`` FK + observed RT).
    """
    pq_tbl: pa.Table = quant.precursor_quant  # type: ignore[attr-defined]
    precursors: pa.Table = library.precursors  # type: ignore[attr-defined]
    peptides: pa.Table = library.peptides  # type: ignore[attr-defined]
    prec = _attach_modseq(precursors, peptides).select(
        ["precursor_id", "peptide_id", "charge", "precursor_mz", "modified_sequence"]
    )
    joined = pq_tbl.select(["precursor_id", "rt_observed"]).join(
        prec, keys="precursor_id", join_type="left outer"
    )
    out = pa.table(
        {
            "target_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "precursor_charge": joined.column("charge"),
            "precursor_mz": joined.column("precursor_mz"),
            "rt_center": _sentinel_to_null(joined.column("rt_observed")),
            "precursor_id": joined.column("precursor_id"),
            "peptide_id": joined.column("peptide_id"),
        }
    )
    return cast_to_schema(out, XIC_TARGET_TABLE)


def targets_from_table(table: pa.Table | str | Path) -> pa.Table:
    """Validate a hand-authored target table against the contract.

    Accepts a ``pa.Table`` or a path to a ``.parquet`` / ``.tsv`` /
    ``.csv`` file. Raises if the required ``target_id`` column is absent;
    missing optional columns fill with null.
    """
    if not isinstance(table, pa.Table):
        path = Path(table)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path)
        elif path.suffix in (".tsv", ".csv", ".txt"):
            from pyarrow import csv

            delimiter = "," if path.suffix == ".csv" else "\t"
            table = csv.read_csv(
                path, parse_options=csv.ParseOptions(delimiter=delimiter)
            )
        else:
            raise ValueError(
                f"unsupported target file {path.suffix!r}; use .parquet/.tsv/.csv"
            )
    return cast_to_schema(table, XIC_TARGET_TABLE)


def load_targets(path: str | Path, *, library: object | None = None) -> pa.Table:
    """Load a target source from disk and normalize it.

    Auto-detects by what is on disk: a Library ParquetDir bundle
    (``precursors.parquet``), a Search bundle (``psms.parquet``), a
    ``PRECURSOR_QUANT`` bundle (``precursor_quant.parquet``, requires
    ``library=``), or a bare ``.parquet`` / ``.tsv`` target file.
    """
    p = Path(path)
    if p.is_dir():
        if (p / "precursors.parquet").exists():
            from constellation.massspec.library.io import load_library

            return targets_from_library(load_library(p, format="parquet_dir"))
        if (p / "psms.parquet").exists():
            from constellation.massspec.search.io import load_search

            return targets_from_search(load_search(p, format="parquet_dir"))
        if (p / "precursor_quant.parquet").exists():
            if library is None:
                raise ValueError(
                    "a PRECURSOR_QUANT source needs library= for m/z + charge"
                )
            from constellation.massspec.quant.io import load_quant

            return targets_from_precursor_quant(
                load_quant(p, format="parquet_dir"), library
            )
        raise ValueError(f"unrecognized target bundle at {p}")
    return targets_from_table(p)
