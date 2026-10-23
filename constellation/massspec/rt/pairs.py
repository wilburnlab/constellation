"""Pairs of retention times to align: the same peptide on two time axes.

Three sources, one shape (:data:`PAIR_SCHEMA`, seconds unless the
reference is a predicted scale):

``library_pairs``
    A library's retention time (``PRECURSOR_TABLE.rt_predicted``, its
    ``-1.0`` sentinel dropped) against where a run identified the
    precursor.
``run_pairs``
    Two runs' identifications joined on the peptide (or precursor).
``engine_pairs``
    The points a search engine fitted its own alignment to, from an
    experiment's ``rt_alignment`` table, in the engine's order and with
    its decoys: what reproduces the engine's fit.

``per="peptide"`` keeps one pair per peptide, its best-scoring
precursor's (EncyclopeDIA fits one point per peptide); ``"precursor"``
keeps charge states apart.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["PAIR_SCHEMA", "library_pairs", "run_pairs", "engine_pairs", "FINAL_PASSES"]

PAIR_SCHEMA = pa.schema(
    [
        pa.field("peptide_id", pa.int64()),
        pa.field("precursor_id", pa.int64()),
        pa.field("modified_sequence", pa.string()),
        pa.field("charge", pa.int32()),
        pa.field("reference_rt", pa.float64(), nullable=False),
        pa.field("observed_rt", pa.float64(), nullable=False),
        pa.field("qvalue", pa.float64()),
        pa.field("is_decoy", pa.bool_()),
    ],
    metadata={b"schema_name": b"RtPairs"},
)

#: The alignment pass each engine settles on (peptide-centric
#: EncyclopeDIA, then ScribeTwo), as ``results.rt_mapping`` reads it.
FINAL_PASSES = ("final", "main")

Per = Literal["peptide", "precursor"]


def _confident(
    ids: pa.Table, max_qvalue: float | None, include_decoys: bool
) -> pa.Table:
    keep = pc.is_valid(ids.column("rt"))
    keep = pc.and_(keep, pc.is_finite(pc.fill_null(ids.column("rt"), np.nan)))
    if max_qvalue is not None:
        keep = pc.and_(keep, pc.fill_null(pc.less_equal(ids.column("qvalue"), max_qvalue), False))
    if not include_decoys and "is_decoy" in ids.column_names:
        keep = pc.and_(keep, pc.invert(pc.fill_null(ids.column("is_decoy"), False)))
    return ids.filter(keep)


def _length_filter(ids: pa.Table, min_length: int | None, max_length: int | None) -> pa.Table:
    if min_length is None and max_length is None:
        return ids
    length = pc.utf8_length(pc.fill_null(ids.column("sequence"), ""))
    keep = pc.is_valid(ids.column("sequence"))
    if min_length is not None:
        keep = pc.and_(keep, pc.greater_equal(length, min_length))
    if max_length is not None:
        keep = pc.and_(keep, pc.less_equal(length, max_length))
    return ids.filter(keep)


def _best_per(table: pa.Table, key: str) -> pa.Table:
    """One row per ``key``: the lowest q-value, nulls last, ties to the first."""
    table = table.filter(pc.is_valid(table.column(key)))
    if table.num_rows == 0:
        return table
    table = table.append_column("_row", pa.array(np.arange(table.num_rows), pa.int64()))
    table = table.sort_by([(key, "ascending"), ("qvalue", "ascending"), ("_row", "ascending")])
    keys = table.column(key).to_numpy()
    first = np.concatenate([[True], keys[1:] != keys[:-1]])
    table = table.filter(pa.array(first)).sort_by("_row")
    return table.drop_columns(["_row"])


def _pairs(columns: dict) -> pa.Table:
    return pa.table({f.name: columns.get(f.name, pa.nulls(len(columns["reference_rt"]), f.type)) for f in PAIR_SCHEMA},
                    schema=PAIR_SCHEMA)


def library_pairs(
    identifications: pa.Table,
    precursors: pa.Table,
    *,
    max_qvalue: float | None = 0.01,
    include_decoys: bool = False,
    per: Per = "peptide",
    min_length: int | None = None,
    max_length: int | None = None,
) -> pa.Table:
    """A library's retention times against a run's identifications.

    ``identifications`` is ``results.identifications`` for one run;
    ``precursors`` a ``PRECURSOR_TABLE`` (``precursor_id`` and
    ``rt_predicted``) keyed by the same content-derived ids. Precursors
    the library has no time for are dropped. ``min_length`` /
    ``max_length`` bound the bare sequence (Chronologer-era pipelines
    aligned 7–30 residues only).
    """
    ids = _confident(identifications, max_qvalue, include_decoys)
    ids = _length_filter(ids, min_length, max_length)
    lib = precursors.select(["precursor_id", "rt_predicted"])
    lib = lib.filter(pc.and_(pc.not_equal(lib.column("rt_predicted"), -1.0),
                             pc.is_finite(lib.column("rt_predicted"))))
    joined = ids.join(lib, "precursor_id", join_type="inner")
    joined = _best_per(joined, "peptide_id" if per == "peptide" else "precursor_id")
    return _pairs(
        {
            "peptide_id": joined.column("peptide_id"),
            "precursor_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "charge": joined.column("charge"),
            "reference_rt": joined.column("rt_predicted"),
            "observed_rt": joined.column("rt"),
            "qvalue": joined.column("qvalue"),
            "is_decoy": joined.column("is_decoy"),
        }
    )


def run_pairs(
    reference: pa.Table,
    target: pa.Table,
    *,
    max_qvalue: float | None = 0.01,
    include_decoys: bool = False,
    per: Per = "peptide",
) -> pa.Table:
    """Two runs' identifications (``results.identifications``) joined.

    Each run keeps its best identification per peptide (or precursor);
    ``reference_rt`` is the reference run's time, ``observed_rt`` the
    target's, ``qvalue`` the larger of the two.
    """
    key = "peptide_id" if per == "peptide" else "precursor_id"
    ref = _best_per(_confident(reference, max_qvalue, include_decoys), key)
    tgt = _best_per(_confident(target, max_qvalue, include_decoys), key)
    ref = ref.select([key, "rt", "qvalue"]).rename_columns([key, "reference_rt", "reference_q"])
    joined = tgt.join(ref, key, join_type="inner")
    q = pc.max_element_wise(joined.column("qvalue"), joined.column("reference_q"))
    return _pairs(
        {
            "peptide_id": joined.column("peptide_id"),
            "precursor_id": joined.column("precursor_id"),
            "modified_sequence": joined.column("modified_sequence"),
            "charge": joined.column("charge"),
            "reference_rt": joined.column("reference_rt"),
            "observed_rt": joined.column("rt"),
            "qvalue": q,
            "is_decoy": joined.column("is_decoy"),
        }
    )


def engine_pairs(
    rt_alignment: pa.Table,
    *,
    pass_name: str | None = None,
    include_decoys: bool = True,
) -> pa.Table:
    """The points an engine fitted its alignment to (``RT_ALIGNMENT`` rows).

    One pass: ``pass_name``, or the engine's final one (``final``, then
    ``main``, then the last by name). Rows keep their order, which the
    engine's own fit depends on, and its decoys, which it fitted with.
    """
    if rt_alignment.num_rows == 0:
        return PAIR_SCHEMA.empty_table()
    passes = set(rt_alignment.column("pass").to_pylist())
    chosen = pass_name or next((p for p in FINAL_PASSES if p in passes), sorted(passes)[-1])
    rows = rt_alignment.filter(pc.equal(rt_alignment.column("pass"), chosen))
    keep = pc.and_(pc.is_finite(pc.fill_null(rows.column("library_rt"), np.nan)),
                   pc.is_finite(pc.fill_null(rows.column("observed_rt"), np.nan)))
    if not include_decoys:
        keep = pc.and_(keep, pc.invert(pc.fill_null(rows.column("is_decoy"), False)))
    rows = rows.filter(keep)
    return _pairs(
        {
            "peptide_id": rows.column("peptide_id"),
            "modified_sequence": rows.column("modified_sequence"),
            "charge": rows.column("charge"),
            "reference_rt": rows.column("library_rt"),
            "observed_rt": rows.column("observed_rt"),
            "is_decoy": rows.column("is_decoy"),
        }
    )
