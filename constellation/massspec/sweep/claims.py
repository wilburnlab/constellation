"""Peaks more than one query (or one query twice) was matched to.

Every query a scan was matched against is matched in the same pass, so
the map sees each peak's claimants together: the peaks a rule matched to
two or more (query, ion) cells are recorded with all their claimants.
That covers different peptides sharing a fragment and one peptide's two
ions on one peak (b3²⁺ on y1⁺, an M+2 at charge 4 on an M+1 at charge 2).
Claims are made by the rule's matches: the nearest peak for ``nearest``,
every peak in the window for ``all`` and ``sum``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from constellation.massspec.sweep.schemas import SWEEP_CLAIM_TABLE, SWEEP_PEAK_CLAIM_TABLE

__all__ = ["Claims", "find_claims", "claims_table"]

_FIELDS = ("level", "scan", "peak_index", "query", "ion", "n_queries", "n_pairs", "mz", "intensity")


@dataclass(frozen=True)
class Claims:
    level: np.ndarray  # int8
    scan: np.ndarray  # int64
    peak_index: np.ndarray  # int64, the peak's position in its scan's stored order
    query: np.ndarray  # int64
    ion: np.ndarray  # int64
    n_queries: np.ndarray  # int64
    n_pairs: np.ndarray  # int64
    mz: np.ndarray
    intensity: np.ndarray

    @property
    def n(self) -> int:
        return int(self.query.size)

    @classmethod
    def empty(cls) -> Claims:
        i, f = np.empty(0, np.int64), np.empty(0, np.float64)
        return cls(np.empty(0, np.int8), i, i, i, i, i, i, f, f)

    @classmethod
    def concat(cls, parts: list[Claims]) -> Claims:
        parts = [p for p in parts if p.n]
        if not parts:
            return cls.empty()
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in _FIELDS))

    def take(self, idx: np.ndarray) -> Claims:
        return Claims(*(getattr(self, f)[idx] for f in _FIELDS))

    def sorted(self) -> Claims:
        # Scan first: shards cover scan ranges in order, so theirs concatenate into this.
        return self.take(np.lexsort((self.ion, self.query, self.peak_index, self.scan)))

    def table(self) -> pa.Table:
        return pa.table(
            {
                "level": pa.array(self.level.astype(np.int8)),
                "scan": pa.array(self.scan.astype(np.int32)),
                "peak_index": pa.array(self.peak_index.astype(np.int32)),
                "query": pa.array(self.query.astype(np.int32)),
                "ion": pa.array(self.ion.astype(np.int32)),
                "n_queries": pa.array(self.n_queries.astype(np.int32)),
                "n_pairs": pa.array(self.n_pairs.astype(np.int32)),
                "mz_observed": pa.array(self.mz),
                "intensity": pa.array(self.intensity),
            },
            schema=SWEEP_CLAIM_TABLE,
        )

    @classmethod
    def from_table(cls, t: pa.Table) -> Claims:
        def col(name, dtype):
            return t.column(name).to_numpy().astype(dtype)

        return cls(col("level", np.int8), col("scan", np.int64), col("peak_index", np.int64),
                   col("query", np.int64), col("ion", np.int64), col("n_queries", np.int64),
                   col("n_pairs", np.int64), col("mz_observed", np.float64),
                   col("intensity", np.float64))  # fmt: skip


def find_claims(
    level: int,
    scan: np.ndarray,
    peak_index: np.ndarray,
    query: np.ndarray,
    ion: np.ndarray,
    mz: np.ndarray,
    intensity: np.ndarray,
) -> Claims:
    """Claims among one level's matches of a set of scans: every (scan,
    peak) matched by two or more cells, with each of them."""
    if scan.size < 2:
        return Claims.empty()
    by = np.lexsort((ion, query, peak_index, scan))
    scan, peak_index, query, ion, mz, intensity = (
        v[by] for v in (scan, peak_index, query, ion, mz, intensity)
    )
    new_peak = np.r_[True, (scan[1:] != scan[:-1]) | (peak_index[1:] != peak_index[:-1])]
    starts = np.flatnonzero(new_peak)
    n_pairs = np.diff(np.r_[starts, scan.size])
    new_query = new_peak | np.r_[True, query[1:] != query[:-1]]
    n_queries = np.add.reduceat(new_query.astype(np.int64), starts)
    per_row_pairs = np.repeat(n_pairs, n_pairs)
    per_row_queries = np.repeat(n_queries, n_pairs)
    keep = per_row_pairs >= 2
    return Claims(
        level=np.full(int(keep.sum()), level, np.int8),
        scan=scan[keep],
        peak_index=peak_index[keep],
        query=query[keep],
        ion=ion[keep],
        n_queries=per_row_queries[keep],
        n_pairs=per_row_pairs[keep],
        mz=mz[keep],
        intensity=intensity[keep],
    )


def claims_table(claims: Claims, plan, ledger, acquisition_id: int) -> pa.Table:
    """Claims with what each claimant is (``SWEEP_PEAK_CLAIM_TABLE``)."""
    c = claims.sorted()
    n = c.n
    if n == 0:
        return SWEEP_PEAK_CLAIM_TABLE.empty_table()
    iso = c.level == 1
    theo = np.empty(n)
    charge = np.empty(n, np.int8)
    isotope = np.zeros(n, np.int8)
    ion_type = np.zeros(n, np.int8)
    position = np.zeros(n, np.int32)
    frag_charge = np.zeros(n, np.int32)
    loss = np.full(n, None, dtype=object)
    for kind_iso, ions in ((True, plan.isotopes), (False, plan.fragments)):
        at = np.flatnonzero(iso == kind_iso)
        if at.size == 0 or ions is None:
            continue
        g = ions.offsets[c.query[at]] + c.ion[at]
        theo[at] = ions.mz[g]
        charge[at] = ions.precursor_charge[g]
        if kind_iso:
            isotope[at] = ions.isotope[g]
        else:
            ion_type[at] = ions.ion_type[g]
            position[at] = ions.position[g]
            frag_charge[at] = ions.fragment_charge[g]
            loss[at] = np.array([None, *ions.loss_ids], dtype=object)[ions.loss[g]]
    query = pa.array(c.query)
    return pa.table(
        {
            "acquisition_id": pa.array(np.full(n, acquisition_id, np.int64)),
            "level": pa.array(c.level),
            "scan": pa.array(c.scan.astype(np.int32)),
            "rt": pa.array(ledger.rt[ledger.rows_of(c.scan)]),
            "peak_index": pa.array(c.peak_index.astype(np.int32)),
            "mz_observed": pa.array(c.mz),
            "intensity": pa.array(c.intensity),
            "n_queries": pa.array(c.n_queries.astype(np.int32)),
            "n_pairs": pa.array(c.n_pairs.astype(np.int32)),
            "target_id": plan.targets.column("target_id").take(query),
            "query": pa.array(c.query.astype(np.int32)),
            "modified_sequence": plan.targets.column("modified_sequence").take(query),
            "precursor_charge": pa.array(charge),
            "isotope": pa.array(isotope, mask=~iso),
            "ion_type": pa.array(ion_type, mask=iso),
            "position": pa.array(position, mask=iso),
            "fragment_charge": pa.array(frag_charge, mask=iso),
            "loss_id": pa.array(loss, pa.string()),
            "mz_theoretical": pa.array(theo),
            "mz_error_ppm": pa.array((c.mz - theo) / theo * 1e6),
        },
        schema=SWEEP_PEAK_CLAIM_TABLE,
    )
