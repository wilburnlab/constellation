"""Precursor grid → a Thermo-style instrument inclusion list.

An inclusion list is what a targeted acquisition method imports: one row
per precursor the instrument should watch for, minimally a name and an
m/z. It sits in ``library/`` rather than ``quant/`` because it is derived
purely from sequence + protease + charge — nothing here has seen an
acquisition.

The CSV is a **projection, never a store**: the canonical record is the
:class:`~constellation.massspec.library.digest.PrecursorSpec` grid (which
callers persist via ``specs_to_table`` → parquet, keeping full-precision
m/z, the bare sequence and every parent accession), and the vendor file
is regenerated from it on write. Rounding happens only here, at the
write boundary.

``Compound`` is the ProForma modseq with the charge appended —
``PEPTIDEK_2``, ``PEPTIDEC[UNIMOD:4]K_2`` — so modifications carry their
UNIMOD accession into the method editor and back out of the raw file.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import pyarrow as pa

from constellation.massspec.library.digest import PrecursorSpec

#: Column order and exact vendor spelling. The Arrow table carries these
#: names verbatim so there is no rename map between builder and writer to
#: drift — a deliberate departure from the package's snake_case schema
#: convention, since this is a vendor projection and not a schema
#: registered with ``core.io``.
INCLUSION_COLUMNS: Final[tuple[str, ...]] = ("Compound", "m/z")

#: Column → decimal places, applied at write time only. 1e-3 Da at m/z
#: 1000 is 1 ppm, orders of magnitude tighter than any isolation window
#: (0.4-4 Da). The deferred RT columns add their own entries here.
DEFAULT_DECIMALS: Final[dict[str, int]] = {"m/z": 3}

#: ProForma 2.0 encodes precursor charge as a ``/N`` suffix on the string
#: (``/[+Na+,-H+]`` when adducts are named).
_CHARGE_SUFFIX = re.compile(r"/(?:\d+|\[[^\]]*\])\Z")

_SortKey = Literal["mz", "compound", "input"]


def compound_name(spec: PrecursorSpec, *, separator: str = "_") -> str:
    """``PEPTIDEC[UNIMOD:4]K_2`` — ProForma modseq, separator, charge.

    ``PrecursorSpec.charge`` is the authority. ``enumerate_modforms``
    never sets ``Peptidoform.charge``, so the FASTA path's modseq carries
    no ``/N`` suffix — but specs read from a peptide list or an existing
    Library can, so strip it and let the charge appear exactly once, in
    the vendor ``_N`` spelling.
    """
    return f"{_CHARGE_SUFFIX.sub('', spec.modified_sequence)}{separator}{spec.charge}"


def dedupe_specs(
    specs: Sequence[PrecursorSpec], *, separator: str = "_"
) -> list[PrecursorSpec]:
    """Drop specs whose ``Compound`` name repeats, first-seen order kept.

    ``precursors_from_fasta`` already collapses peptides shared between
    proteins into one entry carrying every parent accession, so the FASTA
    path has nothing to drop. This exists because ``Compound`` is the
    instrument's row key — a repeat is a duplicate target regardless of
    which producer emitted it.
    """
    seen: set[str] = set()
    kept: list[PrecursorSpec] = []
    for spec in specs:
        name = compound_name(spec, separator=separator)
        if name in seen:
            continue
        seen.add(name)
        kept.append(spec)
    return kept


def build_inclusion_list(
    specs: Sequence[PrecursorSpec],
    *,
    sort: _SortKey = "mz",
    dedupe: bool = True,
    separator: str = "_",
) -> pa.Table:
    """The two-column table, shaped exactly like the emitted CSV.

    Sorted ascending by m/z by default — the vendor convention, how a
    human checks coverage against the scan range, and what lets
    :func:`find_mz_collisions` work as a single adjacent-pair pass. Ties
    break on ``Compound`` so the output is deterministic.

    m/z stays full-precision here; rounding belongs to
    :func:`write_inclusion_list`.
    """
    if sort not in ("mz", "compound", "input"):
        raise ValueError(f"unknown sort {sort!r}; expected mz/compound/input")

    rows = dedupe_specs(specs, separator=separator) if dedupe else list(specs)
    table = pa.table(
        {
            "Compound": pa.array(
                [compound_name(s, separator=separator) for s in rows], pa.string()
            ),
            "m/z": pa.array([float(s.precursor_mz) for s in rows], pa.float64()),
        }
    )
    if sort == "mz":
        return table.sort_by([("m/z", "ascending"), ("Compound", "ascending")])
    if sort == "compound":
        return table.sort_by([("Compound", "ascending")])
    return table


def write_inclusion_list(
    table: pa.Table,
    path: Path | str,
    *,
    decimals: Mapping[str, int] | None = None,
    line_terminator: str = "\r\n",
    encoding: str = "utf-8",
) -> Path:
    """Write `table` as a vendor CSV; returns the path written.

    Hand-rolled rather than ``pyarrow.csv.write_csv`` because that writer
    always quotes the header — ``"Compound","m/z"`` — and method-editor
    importers match header text literally. ``csv.writer`` with the
    default QUOTE_MINIMAL emits ``Compound,m/z`` and leaves
    ``PEPTIDEC[UNIMOD:4]K_2`` bare, while still quoting correctly if an
    exotic ProForma string ever does contain a comma (a global fixed-mod
    prefix, an adduct list).

    CRLF by default: the vendor tooling is Windows-native and its
    templates ship CRLF. Columns are taken from `table`, so adding the
    deferred RT columns is a builder-side change only.
    """
    dec = dict(DEFAULT_DECIMALS if decimals is None else decimals)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    columns = table.column_names
    data = {name: table.column(name).to_pylist() for name in columns}
    with out.open("w", newline="", encoding=encoding) as fh:
        writer = csv.writer(fh, lineterminator=line_terminator)
        writer.writerow(columns)
        for i in range(table.num_rows):
            writer.writerow(
                [
                    f"{data[name][i]:.{dec[name]}f}"
                    if name in dec and data[name][i] is not None
                    else data[name][i]
                    for name in columns
                ]
            )
    return out


def merge_isolation_groups(
    table: pa.Table,
    *,
    tolerance_da: float = 0.5,
    separator: str = ";",
) -> pa.Table:
    """Collapse entries that share an isolation window into one target.

    A low-resolution isolation window (an ion trap's is typically 0.4-2
    Da) cannot separate precursors a few mDa apart, so listing them as
    two rows asks the instrument to run the same isolation twice. Merged
    rows carry the mean m/z and a `separator`-joined Compound —
    ``ISTDDMK_2;AVDEGYR_2`` — so the ambiguity stays visible in the
    method and in the raw file rather than being silently dropped.

    Grouping is greedy over the m/z-sorted list and bounded by **span,
    not by neighbour distance**: an entry joins the open group only while
    ``mz - group_min <= tolerance_da``. Single-linkage chaining would let
    a group creep arbitrarily far — A-B and B-C each within tolerance
    while A-C is not — and the resulting "one target" would no longer fit
    in one window. Bounding the span guarantees every member is reachable
    from one isolation of that width, and that the mean sits within
    ``tolerance_da / 2`` of each member.

    ``tolerance_da <= 0`` disables merging and returns `table` unchanged.
    """
    if tolerance_da <= 0 or table.num_rows == 0:
        return table

    ordered = table.sort_by([("m/z", "ascending"), ("Compound", "ascending")])
    names = ordered.column("Compound").to_pylist()
    mzs = ordered.column("m/z").to_pylist()

    out_names: list[str] = []
    out_mzs: list[float] = []
    group_names: list[str] = [names[0]]
    group_mzs: list[float] = [mzs[0]]

    def _flush() -> None:
        out_names.append(separator.join(group_names))
        out_mzs.append(sum(group_mzs) / len(group_mzs))

    for name, mz in zip(names[1:], mzs[1:], strict=True):
        if mz - group_mzs[0] <= tolerance_da:
            group_names.append(name)
            group_mzs.append(mz)
            continue
        _flush()
        group_names, group_mzs = [name], [mz]
    _flush()

    return pa.table(
        {
            "Compound": pa.array(out_names, pa.string()),
            "m/z": pa.array(out_mzs, pa.float64()),
        }
    )


def find_mz_collisions(table: pa.Table, *, tolerance_ppm: float = 10.0) -> pa.Table:
    """Adjacent entries in the m/z-sorted list within `tolerance_ppm`.

    Reported, never filtered. Two distinct peptidoforms that fall inside
    one isolation window are a fact about the panel the user needs before
    spending instrument time; dropping one silently would leave them
    believing it was targeted.

    Columns: ``compound_a``, ``compound_b``, ``mz_a``, ``mz_b``,
    ``delta_ppm``.
    """
    ordered = table.sort_by([("m/z", "ascending"), ("Compound", "ascending")])
    names = ordered.column("Compound").to_pylist()
    mzs = ordered.column("m/z").to_pylist()

    a_names: list[str] = []
    b_names: list[str] = []
    a_mzs: list[float] = []
    b_mzs: list[float] = []
    deltas: list[float] = []
    for i in range(len(mzs) - 1):
        lo, hi = mzs[i], mzs[i + 1]
        mean = (lo + hi) / 2.0
        if mean <= 0:
            continue
        ppm = abs(hi - lo) / mean * 1e6
        if ppm > tolerance_ppm:
            continue
        a_names.append(names[i])
        b_names.append(names[i + 1])
        a_mzs.append(lo)
        b_mzs.append(hi)
        deltas.append(ppm)

    return pa.table(
        {
            "compound_a": pa.array(a_names, pa.string()),
            "compound_b": pa.array(b_names, pa.string()),
            "mz_a": pa.array(a_mzs, pa.float64()),
            "mz_b": pa.array(b_mzs, pa.float64()),
            "delta_ppm": pa.array(deltas, pa.float64()),
        }
    )


def count_levels(specs: Sequence[PrecursorSpec]) -> dict[str, int]:
    """Entry / peptide / peptidoform counts for a run summary.

    The three differ, and the difference is the point: one peptide
    contributes several entries once charges and variable modforms are
    swept, so "how many peptides am I targeting" is not the row count.
    """
    return {
        "entries": len(specs),
        "peptides": len({s.sequence for s in specs}),
        "peptidoforms": len({s.modified_sequence for s in specs}),
    }


__all__ = [
    "DEFAULT_DECIMALS",
    "INCLUSION_COLUMNS",
    "build_inclusion_list",
    "compound_name",
    "count_levels",
    "dedupe_specs",
    "find_mz_collisions",
    "merge_isolation_groups",
    "write_inclusion_list",
]
