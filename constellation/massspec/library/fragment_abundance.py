"""A library's fragment intensities as query channel abundances.

:class:`LibraryFragments` is a
:class:`~constellation.massspec.peptide.channels.FragmentAbundance`: given
the candidate fragment ions of each query (the ladder the channel settings
allow), it says how abundant each is from a library's
``LIBRARY_FRAGMENT_TABLE`` rows, looked up by precursor id (a query's
``precursor_id``). A predicted library (``predict-library``) and an
empirical one (a chromatogram library, an ``.msp``) both work.

* **The join** is exact on (precursor, ion type, bond position, fragment
  charge, neutral loss). ``loss_id`` is mapped to an index first, null to
  0, since a null never joins.
* **Abundance** is a row's intensity over the precursor's total (every
  row it has): its share of the whole spectrum, scale-free whatever the
  library's units. An ion the library's model covers and did not report
  was predicted below its floor: 0. An ion outside what it covers (an ion
  number above its last, a charge above its ceiling, a loss it does not
  predict), or of a precursor it did not predict: NaN. The coverage is the
  one ``predict-library`` records (``x.massspec.prediction.fragment_coverage``);
  a library without one (an empirical library) leaves absent ions NaN.
* **Model losses**: the losses the model covers get channels at every
  bond, whatever the loss rules say, since the model, not the rules, says
  where its losses are.
* **Checks**: a row's m/z must be the ladder's (within 1e-9 relative for
  a library with a coverage record, whose m/z come from the same ladder;
  within ``mz_tolerance_ppm`` otherwise). An intact ion, or one with a
  model loss, inside the channel settings that matches no channel is an
  error: the library and the query disagree about the molecule. Rows
  outside the settings (an ion type not asked, a charge above the
  ceiling, a loss neither asked nor covered) and rows the library could
  not name (null ion type) are counted in :attr:`LibraryFragments.report`.
* **Streaming**: from a directory (``library_pqdir``) only the rows of the
  precursors asked about are kept, read batch by batch, so a proteome's
  library (about 24 rows a precursor) is never loaded whole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.massspec.peptide.channels import CandidateIons
from constellation.massspec.peptide.ions import IonType

__all__ = ["ANNOTATION_TOLERANCE_KEY", "COVERAGE_KEY", "DEFAULT_ANNOTATION_TOLERANCE", "LibraryFragments"]

#: Where ``predict-library`` records what its fragment model covers.
COVERAGE_KEY = "x.massspec.prediction.fragment_coverage"
#: Where a library read from a file of observed spectra records the
#: tolerance its peaks were annotated at (``{"value": ..., "unit": "ppm"|"Da"}``):
#: an annotated peak's m/z is its stored one, so it differs from the ion's
#: by up to that much.
ANNOTATION_TOLERANCE_KEY = "x.massspec.library.annotation_tolerance"
#: What an EncyclopeDIA library's peaks are annotated at
#: (``io.encyclopedia``'s ``fragment_tolerance_ppm``), and so the check on a
#: library that records neither a coverage nor its own tolerance. Until
#: 2026-11-09 it was 5 ppm, which refused the lab's chromatogram library
#: (62 rows up to 19.6 ppm).
DEFAULT_ANNOTATION_TOLERANCE: tuple[float, str] = (20.0, "ppm")


def _unit(unit: Any) -> str:
    """``ppm`` or ``Da``, however a source spells it (EncyclopeDIA's results
    record ``da``; until 2026-11-09 anything but ``Da`` was taken as ppm,
    so a 0.35 Da annotation was held to 0.35 ppm; review by Codex)."""
    text = str(unit).strip().lower()
    if text == "ppm":
        return "ppm"
    if text in ("da", "th", "amu"):
        return "Da"
    raise ValueError(f"a tolerance unit is ppm or Da, not {unit!r}")


def _tolerance(value: Any) -> tuple[float, str] | None:
    if not value:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return float(value["value"]), _unit(value["unit"])
    v, unit = value
    return float(v), _unit(unit)


_COLUMNS = ["precursor_id", "ion_type", "position", "charge", "loss_id", "mz_theoretical", "intensity_predicted"]
_N_TERMINAL = (int(IonType.A), int(IonType.B), int(IonType.C))


@dataclass
class LibraryFragments:
    """Fragment abundances from a library's fragment table (module
    docstring). ``source`` is a ``LIBRARY_FRAGMENT_TABLE`` or the path of
    a Parquet file of one; ``coverage`` the model's coverage record (None
    for an empirical library); ``predicted`` the precursor ids the library
    speaks for (None: those with a fragment row); ``annotation_tolerance``
    how far an empirical library's annotated m/z may sit from the ion's
    (``(value, 'ppm'|'Da')``; None: what the library records, else
    :data:`DEFAULT_ANNOTATION_TOLERANCE`). A predicted library (one with a
    coverage) computed its m/z from the ions and is held to 1e-3 ppm."""

    source: pa.Table | Path
    coverage: dict[str, Any] | None = None
    predicted: np.ndarray | None = None
    annotation_tolerance: tuple[float, str] | None = None
    label: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    ranks: bool = field(default=True, init=False)
    _predicted_sorted: np.ndarray | None = field(default=None, init=False, repr=False)
    _cache: pa.Table | None = field(default=None, init=False, repr=False)
    _cache_keys: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_library(cls, library, *, annotation_tolerance: tuple[float, str] | None = None) -> LibraryFragments:
        """From a ``Library`` held in memory."""
        extras = library.metadata_extras or {}
        return cls(
            source=library.fragments,
            coverage=extras.get(COVERAGE_KEY),
            predicted=library.precursors.column("precursor_id").to_numpy(),
            annotation_tolerance=annotation_tolerance or _tolerance(extras.get(ANNOTATION_TOLERANCE_KEY)),
            label="library",
        )

    @classmethod
    def from_directory(
        cls, path: Path | str, *, annotation_tolerance: tuple[float, str] | None = None
    ) -> LibraryFragments:
        """From a library directory (``parquet_dir``, what ``predict-library``
        writes as ``library_pqdir``, or the prediction directory holding
        one), reading its fragments only for the precursors asked about."""
        root = Path(path)
        if not (root / "precursors.parquet").exists() and (root / "library_pqdir" / "precursors.parquet").exists():
            root = root / "library_pqdir"
        if not (root / "fragments.parquet").exists():
            raise ValueError(f"{root} holds no fragments.parquet: no fragment abundances to read")
        manifest = root / "manifest.json"
        metadata = json.loads(manifest.read_text()).get("metadata", {}) if manifest.exists() else {}
        precursors = pq.read_table(root / "precursors.parquet", columns=["precursor_id"]).column("precursor_id")
        if pq.read_metadata(root / "fragments.parquet").num_rows == 0:
            raise ValueError(f"{root}/fragments.parquet is empty: the library has no fragment intensities")
        return cls(source=root / "fragments.parquet", coverage=metadata.get(COVERAGE_KEY),
                   predicted=precursors.to_numpy(), label=str(root),
                   annotation_tolerance=annotation_tolerance or _tolerance(metadata.get(ANNOTATION_TOLERANCE_KEY)))  # fmt: skip

    # ── the FragmentAbundance contract ────────────────────────────────

    def model_losses(self) -> tuple[str, ...]:
        return tuple(self.coverage.get("losses") or ()) if self.coverage else ()

    def describe(self) -> dict[str, Any]:
        return {"source": "library", "library": self.label, "coverage": self.coverage, "report": dict(self.report)}

    #: Row groups decoded by :meth:`_rows` (what pruning saved is visible).
    row_groups_read: int = field(default=0, init=False, repr=False)

    def prepare(self, keys: np.ndarray) -> None:
        """Called once before a compile asks for ``keys`` chunk by chunk. A
        fragment file sorted by precursor (what ``predict-library`` writes)
        is read chunk by chunk, each chunk's row groups only; any other is
        read once here for every key asked, and chunks are served from that."""
        self._cache = self._cache_keys = None
        if isinstance(self.source, pa.Table) or _sorted_by_precursor(pq.ParquetFile(self.source)):
            return
        wanted = np.unique(np.asarray(keys, dtype=np.int64))
        rows = self._read(wanted[wanted >= 0])
        order = pc.sort_indices(rows, sort_keys=[("precursor_id", "ascending")])
        self._cache = rows.take(order)
        self._cache_keys = self._cache.column("precursor_id").to_numpy()

    def release(self) -> None:
        """Drop what :meth:`prepare` read."""
        self._cache = self._cache_keys = None

    def _rows(self, wanted: np.ndarray) -> pa.Table:
        if self._cache is not None:
            lo = np.searchsorted(self._cache_keys, wanted, "left")
            hi = np.searchsorted(self._cache_keys, wanted, "right")
            return self._cache.take(pa.array(_ragged(lo, hi - lo)))
        return self._read(wanted)

    def _read(self, wanted: np.ndarray) -> pa.Table:
        keys = pa.array(wanted, pa.int64())
        if isinstance(self.source, pa.Table):
            table = self.source.select(_COLUMNS)
            return table.filter(pc.is_in(table.column("precursor_id"), value_set=keys))
        f = pq.ParquetFile(self.source)
        # Only the row groups whose precursor range holds a wanted id: in a
        # file sorted by precursor, a chunk of ids touches few of them.
        groups = [g for g, (lo, hi) in enumerate(_precursor_ranges(f))
                  if lo is None or np.searchsorted(wanted, hi, "right") > np.searchsorted(wanted, lo, "left")]  # fmt: skip
        parts = []
        for g in groups:
            self.row_groups_read += 1
            group = f.read_row_group(g, columns=_COLUMNS)
            mask = pc.is_in(group.column("precursor_id"), value_set=keys)
            if pc.any(mask).as_py():
                parts.append(group.filter(mask))
        # The wanted precursors' rows only: one compile's, not a stage hand-off.
        return pa.concat_tables(parts) if parts else pa.schema(f.schema_arrow).empty_table().select(_COLUMNS)

    def abundance(self, ions: CandidateIons) -> np.ndarray:
        n_ions = ions.mz.size
        out = np.full(n_ions, np.nan)
        query = ions.query_of()
        keys = ions.key[query]
        wanted = np.unique(ions.key[ions.key >= 0])
        report: dict[str, Any] = {"rows": 0, "matched": 0, "unnamed": 0, "outside_settings": {}, "max_mz_error_ppm": 0.0}
        self.report = report
        if wanted.size == 0 or n_ions == 0:
            return out
        rows = self._rows(wanted)
        report["rows"] = rows.num_rows

        # Each precursor's total intensity, over every row it has.
        r_key = rows.column("precursor_id").to_numpy()
        r_int = rows.column("intensity_predicted").fill_null(-1.0).to_numpy(zero_copy_only=False).astype(np.float64)
        dense, r_dense = np.unique(r_key, return_inverse=True)
        total = np.bincount(r_dense, weights=np.where(r_int > 0, r_int, 0.0), minlength=dense.size)

        # 0 inside the model's coverage, NaN outside it.
        if self.predicted is not None and self._predicted_sorted is None:
            self._predicted_sorted = np.unique(self.predicted)
        predicted = self._predicted_sorted if self.predicted is not None else dense
        known = np.isin(keys, predicted)
        if self.coverage:
            out[known & self._covered(ions, query)] = 0.0

        # Rows that name their ion; the rest are counted.
        named = rows.column("ion_type").is_valid().to_numpy(zero_copy_only=False) & rows.column(
            "position").is_valid().to_numpy(zero_copy_only=False) & rows.column("charge").is_valid().to_numpy(
            zero_copy_only=False)  # fmt: skip
        report["unnamed"] = int((~named).sum())
        r_type = rows.column("ion_type").fill_null(-1).to_numpy().astype(np.int64)
        r_pos = rows.column("position").fill_null(-1).to_numpy().astype(np.int64)
        r_charge = rows.column("charge").fill_null(-1).to_numpy().astype(np.int64)
        loss_col = rows.column("loss_id")
        found = pc.index_in(loss_col, value_set=pa.array(list(ions.loss_ids), pa.string()))
        r_loss = np.where(loss_col.is_null().to_numpy(zero_copy_only=False), 0,
                          found.fill_null(-2).to_numpy(zero_copy_only=False).astype(np.int64) + 1)  # fmt: skip
        r_mz = rows.column("mz_theoretical").to_numpy().astype(np.float64)

        # Rows and candidates joined exactly on (precursor, ion type, position,
        # charge, loss): every candidate of a row's precursor (a precursor may
        # be several queries), rows in their order. Until 2026-11-09 the five
        # were packed into one integer, which overflowed past 2**23 precursors.
        usable = named & (r_loss >= 0) & (r_type >= 0)
        row_of, cand = _join(
            (r_key, r_type, r_pos, r_charge, r_loss), usable,
            (keys, ions.ion_type.astype(np.int64), ions.position.astype(np.int64),
             ions.fragment_charge.astype(np.int64), ions.loss.astype(np.int64)),
        )  # fmt: skip
        hit = np.zeros(r_key.size, dtype=bool)
        hit[row_of] = True
        theo = ions.mz[cand]
        err_da = r_mz[row_of] - theo
        err_ppm = err_da / theo * 1e6
        worst = float(np.abs(err_ppm).max()) if err_ppm.size else 0.0
        report["max_mz_error_ppm"] = worst
        if self.coverage:  # computed from the ions: 1e-9 relative is 1e-3 ppm
            excess, limit_text = np.abs(err_ppm) - 1e-3, "1e-3 ppm"
        else:
            value, unit = _tolerance(self.annotation_tolerance or DEFAULT_ANNOTATION_TOLERANCE)
            # The annotation's own test was against the stored m/z; allow the
            # float rounding between that and this ladder.
            excess = (np.abs(err_da) - value * (1 + 1e-9)) if unit == "Da" else (np.abs(err_ppm) - value - 1e-6)
            limit_text = f"{value:g} {unit}"
        report["annotation_tolerance"] = limit_text
        if excess.size and float(excess.max()) > 0:
            bad = int(np.argmax(excess))
            raise ValueError(
                f"library fragment m/z differs from the query's ladder by {abs(float(err_ppm[bad])):.3g} ppm, "
                f"beyond the library's {limit_text} (precursor "
                f"{int(r_key[row_of[bad]])}, ion type {IonType(int(r_type[row_of[bad]])).name.lower()}, position "
                f"{int(r_pos[row_of[bad]])}, charge {int(r_charge[row_of[bad]])}): the library and the query "
                f"describe different molecules (labels, modifications)"
            )
        whole = total[r_dense[row_of]]
        share = np.full(row_of.size, np.nan)
        has = whole > 0
        share[has] = np.maximum(r_int[row_of][has], 0.0) / whole[has]
        out[cand] = share
        report["matched"] = int(hit.sum())

        # Rows matching no channel: outside the settings (counted), or a
        # disagreement about the molecule (refused).
        miss = np.flatnonzero(usable & ~hit)
        if miss.size:
            self._classify_misses(ions, rows, miss, r_key, r_type, r_charge, r_loss, loss_col, report)
        unknown_loss = named & (r_loss < 0)
        if unknown_loss.any():
            report["outside_settings"]["loss_not_asked"] = int(unknown_loss.sum())
        return out

    def _covered(self, ions: CandidateIons, query: np.ndarray) -> np.ndarray:
        """Whether each candidate ion is one the model would have reported
        had it predicted it above its floor."""
        cov = self.coverage or {}
        types = {int(IonType[name.upper()]) for name in cov.get("ion_types", [])}
        ok = np.isin(ions.ion_type.astype(np.int64), list(types)) if types else np.zeros(ions.mz.size, bool)
        ceiling = int(cov.get("max_fragment_charge") or 0)
        z = ions.precursor_charge[query]
        top = np.minimum(ceiling, z) if cov.get("fragment_charge_at_most_precursor", True) else np.full(z.size, ceiling)
        ok &= ions.fragment_charge <= top
        losses = {name for name in cov.get("losses", [])}
        names = [None, *ions.loss_ids]
        covered_loss = np.array([k == 0 or names[k] in losses for k in range(len(names))], dtype=bool)
        ok &= covered_loss[ions.loss.astype(np.int64)]
        last = cov.get("max_ion_number")
        if last is not None:
            length = ions.length[query]
            number = np.where(np.isin(ions.ion_type.astype(np.int64), _N_TERMINAL), ions.position + 1,
                              length - 1 - ions.position)  # fmt: skip
            ok &= number <= int(last)
        return ok

    def _classify_misses(self, ions, rows, miss, r_key, r_type, r_charge, r_loss, loss_col, report) -> None:
        asked = list(ions.ion_types) if ions.ion_types else [int(t) for t in np.unique(ions.ion_type)]
        # A precursor's charge ceiling, from the first of its queries.
        keys = ions.key
        distinct, first_q = np.unique(keys, return_index=True)
        k = r_key[miss]
        at = np.minimum(np.searchsorted(distinct, k), max(distinct.size - 1, 0))
        known = distinct.size > 0
        known = (distinct[at] == k) if known else np.zeros(k.size, dtype=bool)
        miss, k, at = miss[known], k[known], at[known]
        q = first_q[at]
        ceiling = ions.charge_ceiling
        model = list(self.model_losses())
        is_model = pc.is_in(loss_col.take(pa.array(miss)), value_set=pa.array(model, pa.string())).fill_null(False)
        not_asked = ~np.isin(r_type[miss], asked)
        above = (~not_asked) & (r_charge[miss] > ceiling[q]) if ceiling is not None else np.zeros(miss.size, bool)
        rules = (~not_asked) & (~above) & (r_loss[miss] > 0) & ~is_model.to_numpy(zero_copy_only=False)
        outside = report["outside_settings"]
        for reason, mask in (("ion_type_not_asked", not_asked), ("charge_above_ceiling", above),
                             ("loss_not_placed_by_rules", rules)):  # fmt: skip
            if mask.any():
                outside[reason] = outside.get(reason, 0) + int(mask.sum())
        problems = miss[~(not_asked | above | rules)].tolist()
        names = loss_col
        if problems:
            i = problems[0]
            raise ValueError(
                f"{len(problems)} library fragment(s) inside the channel settings match no channel of their "
                f"query, e.g. precursor {int(r_key[i])}, ion type {IonType(int(r_type[i])).name.lower()}, position "
                f"{int(rows.column('position')[i].as_py())}, charge {int(r_charge[i])}, loss {names[i].as_py()}: the "
                f"library and the query describe different molecules"
            )


def _precursor_ranges(f: pq.ParquetFile) -> list[tuple[Any, Any]]:
    """Each row group's (min, max) precursor id; (None, None) where the
    footer does not say."""
    meta = f.metadata
    col = f.schema_arrow.get_field_index("precursor_id")
    out = []
    for g in range(meta.num_row_groups):
        stats = meta.row_group(g).column(col).statistics
        out.append((stats.min, stats.max) if stats is not None and stats.has_min_max else (None, None))
    return out


def _sorted_by_precursor(f: pq.ParquetFile) -> bool:
    """Whether a fragment file's row groups hold ascending, disjoint
    precursor ranges, so that a range of ids touches few of them."""
    ranges = _precursor_ranges(f)
    if any(lo is None for lo, _ in ranges):
        return False
    return all(ranges[g][0] >= ranges[g - 1][1] for g in range(1, len(ranges)))


def _join(
    left: tuple[np.ndarray, ...], usable: np.ndarray, right: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs ``(i, j)`` where row ``i`` of ``left`` (among ``usable``)
    equals row ``j`` of ``right`` in every column, sorted by ``(i, j)``.
    The first column is an identifier (any int64), the others small
    non-negative integers. They are combined into one integer in mixed
    radix over the ranges actually present, which is exact (each column
    gets exactly the room its values need); where even that would not fit
    in 62 bits, Arrow's hash join does it."""
    li = np.flatnonzero(usable)
    ids = np.unique(left[0][li])
    r_at = np.minimum(np.searchsorted(ids, right[0]), max(ids.size - 1, 0))
    r_ok = (ids[r_at] == right[0]) if ids.size else np.zeros(right[0].size, dtype=bool)
    rj = np.flatnonzero(r_ok)
    lcols = [np.searchsorted(ids, left[0][li])] + [np.asarray(c[li], dtype=np.int64) for c in left[1:]]
    rcols = [r_at[rj]] + [np.asarray(c[rj], dtype=np.int64) for c in right[1:]]
    if li.size == 0 or rj.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if min(int(c.min()) for c in lcols + rcols) < 0:
        raise ValueError("join columns must be non-negative")
    bases = [ids.size] + [max(int(a.max()), int(b.max())) + 1 for a, b in zip(lcols[1:], rcols[1:], strict=True)]
    if float(np.prod([float(b) for b in bases])) >= 2.0**62:
        return _arrow_join(left, li, right, rj)

    def combine(cols):
        key = cols[0].astype(np.int64)
        for c, base in zip(cols[1:], bases[1:], strict=True):
            key = key * base + c
        return key

    lk, rk = combine(lcols), combine(rcols)
    order = np.argsort(rk, kind="stable")
    sorted_rk = rk[order]
    first = np.searchsorted(sorted_rk, lk, "left")
    counts = np.searchsorted(sorted_rk, lk, "right") - first
    return np.repeat(li, counts), rj[order[_ragged(first, counts)]]


def _arrow_join(
    left: tuple[np.ndarray, ...], li: np.ndarray, right: tuple[np.ndarray, ...], rj: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    names = [f"c{k}" for k in range(len(left))]
    lt = pa.table({**{n: pa.array(c[li]) for n, c in zip(names, left, strict=True)}, "__i": pa.array(li)})
    rt = pa.table({**{n: pa.array(c[rj]) for n, c in zip(names, right, strict=True)}, "__j": pa.array(rj)})
    joined = lt.join(rt, keys=names, join_type="inner", use_threads=False)
    i = joined.column("__i").to_numpy()
    j = joined.column("__j").to_numpy()
    order = np.lexsort((j, i))
    return i[order], j[order]


def _ragged(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(counts)
    return np.arange(total, dtype=np.int64) + np.repeat(np.asarray(starts, dtype=np.int64) - (ends - counts), counts)
