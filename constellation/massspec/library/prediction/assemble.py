"""Predicted fragments onto a library's own precursors.

A predictor answers for a request (an unlabelled peptide at a charge); a
library holds precursors, several of which may share one request (a heavy
and a light form, or the same peptide in two decoy sets). ``fragment_table``
writes each request's fragments under every source precursor that asked
it, as ``LIBRARY_FRAGMENT_TABLE`` rows:

* the bond position from the ion number (``peptide.ions.bond_position``):
  b_n breaks bond n-1, y_n bond L-1-n;
* the m/z from the **source** peptide's own ladder
  (``fragment_ladder_arrays_batch``), so a labelled precursor gets its
  labelled m/z beside the intensities predicted for its light form. The
  ladder's rows are in a fixed order (bond, then ion type, then charge),
  so each fragment's m/z is looked up by index, with no join;
* a fragment with a neutral loss (a Koina model's ``y3-H2O+1``) is a
  typed row with its ``loss_id``, its m/z the source peptide's ion less
  the loss (labels included). The loss rules of ``peptide.neutral_losses``
  are not applied: they say which losses to expect where nobody has said,
  and here the model has said; its m/z is physics either way;
* ``annotation`` left null on these typed rows (readers derive mzPAF);
  ``correlation`` and ``quantified`` null.

Everything is numpy over whole arrays. A backend that also returns m/z
(Koina) has them checked against the peptide that was sent
(``reported_mz_error_ppm``), never used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.massspec.library.prediction.base import FragmentArrays, Requests, sites_key
from constellation.massspec.library.schemas import LIBRARY_FRAGMENT_TABLE
from constellation.massspec.peptide.ions import IonType, LadderArrays, bond_position, fragment_ladder_arrays_batch

__all__ = ["Ladders", "fragment_table", "ladders_for", "reported_mz_error_ppm"]


class Ladders:
    """Intact-ion ladders of some peptides, for looking up fragment m/z.

    ``peptides`` are ``Peptidoform`` objects or ProForma strings. They are
    kept as given, so strings cost little to keep; a peptide is parsed again
    only if a fragment of it with a neutral loss is asked for, whose ladder
    is built then.
    """

    def __init__(
        self, peptides: Sequence[Peptidoform | str], ion_types: Sequence[IonType], max_fragment_charge: int
    ) -> None:
        self.ion_types = tuple(ion_types)
        self.max_fragment_charge = int(max_fragment_charge)
        self._peptides = list(peptides)
        parsed = [self._parsed(i) for i in range(len(self._peptides))]
        self.lengths = np.fromiter((len(p.sequence) for p in parsed), np.int64, len(parsed))
        self.arrays: LadderArrays = fragment_ladder_arrays_batch(
            parsed, ion_types=self.ion_types, max_fragment_charge=self.max_fragment_charge
        )
        per_bond = len(self.ion_types) * self.max_fragment_charge
        expected = np.maximum(self.lengths - 1, 0) * per_bond
        if not np.array_equal(np.diff(self.arrays.offsets), expected):
            raise ValueError("a ladder is missing intact ions; fragment m/z cannot be looked up by index")
        self._type_index = np.full(max(int(t) for t in IonType) + 1, -1, dtype=np.int64)
        for k, t in enumerate(self.ion_types):
            self._type_index[int(t)] = k

    def _parsed(self, i: int) -> Peptidoform:
        p = self._peptides[i]
        return parse_proforma(p) if isinstance(p, str) else p

    def mz(
        self,
        peptide: np.ndarray,
        ion_type: np.ndarray,
        position: np.ndarray,
        charge: np.ndarray,
        loss: np.ndarray | None = None,
        loss_ids: Sequence[str] = (),
    ) -> np.ndarray:
        """m/z of ions of ``peptide`` (indices into the peptides given);
        ``loss`` is 0 for an intact ion and ``k`` for ``loss_ids[k - 1]``."""
        peptide = np.asarray(peptide, dtype=np.int64)
        position = np.asarray(position, dtype=np.int64)
        charge = np.asarray(charge, dtype=np.int64)
        k = self._type_index[np.asarray(ion_type, dtype=np.int64)]
        if (k < 0).any() or (charge < 1).any() or (charge > self.max_fragment_charge).any():
            raise ValueError("an ion outside the ladder's ion types or charges")
        per_bond = len(self.ion_types) * self.max_fragment_charge
        row = self.arrays.offsets[peptide] + position * per_bond + k * self.max_fragment_charge + (charge - 1)
        if loss is None or not np.any(loss):
            return self.arrays.mz[row]
        loss = np.asarray(loss, dtype=np.int64)
        if (loss < 0).any() or (loss > len(loss_ids)).any():
            raise ValueError(f"a loss index outside the {len(loss_ids)} losses named")
        out = np.empty(len(peptide), dtype=np.float64)
        intact = loss == 0
        out[intact] = self.arrays.mz[row[intact]]
        lossy = ~intact
        out[lossy] = self._loss_mz(peptide[lossy], k[lossy], position[lossy], charge[lossy], loss[lossy], tuple(loss_ids))
        return out

    def _loss_mz(
        self,
        peptide: np.ndarray,
        k: np.ndarray,
        position: np.ndarray,
        charge: np.ndarray,
        loss: np.ndarray,
        loss_ids: tuple[str, ...],
    ) -> np.ndarray:
        """m/z of ions with a loss, from ladders of only the peptides that
        have one, built with every loss at every bond (``loss_rules=False``)
        so they too are looked up by index."""
        used, local = np.unique(peptide, return_inverse=True)
        arrays = fragment_ladder_arrays_batch(
            [self._parsed(int(i)) for i in used],
            ion_types=self.ion_types,
            max_fragment_charge=self.max_fragment_charge,
            neutral_losses=loss_ids,
            loss_rules=False,
        )
        slots = len(loss_ids) + 1
        per_bond = len(self.ion_types) * self.max_fragment_charge * slots
        if not np.array_equal(np.diff(arrays.offsets), np.maximum(self.lengths[used] - 1, 0) * per_bond):
            raise ValueError("a ladder with losses is missing ions; fragment m/z cannot be looked up by index")
        row = (
            arrays.offsets[local]
            + position * per_bond
            + k * self.max_fragment_charge * slots
            + (charge - 1) * slots
            + loss
        )
        return arrays.mz[row]


def ladders_for(sequences: Sequence[str], ion_types: Sequence[IonType], max_fragment_charge: int) -> Ladders:
    """Ladders of ProForma strings (labels and all)."""
    return Ladders(list(sequences), ion_types, max_fragment_charge)


def fragment_table(
    fragments: FragmentArrays,
    source_request: np.ndarray,
    source_precursor_id: np.ndarray,
    source_peptide: np.ndarray,
    ladders: Ladders,
) -> pa.Table:
    """Rows of ``LIBRARY_FRAGMENT_TABLE`` for source precursors.

    ``fragments.request`` indexes some requests 0..m-1; source precursor
    ``i`` asked request ``source_request[i]``, has id
    ``source_precursor_id[i]`` and its peptide is ``ladders``' peptide
    ``source_peptide[i]``. Rows come out by source precursor, then ion
    type, bond position, fragment charge and loss (intact first).
    """
    m = int(source_request.max()) + 1 if len(source_request) else 0
    order = np.argsort(fragments.request, kind="stable")
    counts = np.bincount(fragments.request, minlength=m)[:m] if m else np.zeros(0, np.int64)
    starts = np.zeros(m, dtype=np.int64)
    if m:
        np.cumsum(counts[:-1], out=starts[1:])
    per_source = counts[source_request] if m else np.zeros(len(source_request), np.int64)
    total = int(per_source.sum())
    owner = np.repeat(np.arange(len(source_request)), per_source)
    out_start = np.zeros(len(source_request), dtype=np.int64)
    if len(source_request):
        np.cumsum(per_source[:-1], out=out_start[1:])
    src = order[np.arange(total) - out_start[owner] + starts[source_request[owner]]] if total else np.zeros(0, np.int64)

    ion_type = fragments.ion_type[src].astype(np.int8)
    number = fragments.ion_number[src].astype(np.int64)
    charge = fragments.fragment_charge[src].astype(np.int32)
    loss = fragments.loss[src].astype(np.int64) if fragments.loss is not None else np.zeros(total, np.int64)
    peptide = source_peptide[owner]
    length = ladders.lengths[peptide]
    position = np.asarray(bond_position(ion_type, number, length), dtype=np.int32) if total else np.zeros(0, np.int32)
    mz = ladders.mz(peptide, ion_type, position, charge, loss, fragments.loss_ids) if total else np.zeros(0)

    sort = np.lexsort((loss, charge, position, ion_type, owner))
    n = total
    names = np.array([None, *fragments.loss_ids], dtype=object)
    table = pa.table(
        {
            "precursor_id": pa.array(source_precursor_id[owner][sort], type=pa.int64()),
            "ion_type": pa.array(ion_type[sort], type=pa.int8()),
            "position": pa.array(position[sort], type=pa.int32()),
            "charge": pa.array(charge[sort], type=pa.int32()),
            "loss_id": pa.array(names[loss[sort]], type=pa.string()),
            "mz_theoretical": pa.array(mz[sort], type=pa.float64()),
            "intensity_predicted": pa.array(fragments.intensity[src][sort].astype(np.float64), type=pa.float64()),
            "annotation": pa.nulls(n, type=pa.string()),
            "correlation": pa.nulls(n, type=pa.float64()),
            "quantified": pa.nulls(n, type=pa.bool_()),
        },
        schema=LIBRARY_FRAGMENT_TABLE,
    )
    return table


def reported_mz_error_ppm(fragments: FragmentArrays, requests: Requests, ion_types: Sequence[IonType], max_fragment_charge: int) -> float:
    """The worst disagreement (ppm) between the m/z a backend reported and
    the unlabelled peptide that was sent. Large means the peptide sent was
    not the molecule meant (a modification spelled wrongly)."""
    if fragments.mz_reported is None or not len(fragments):
        return 0.0
    used, local = np.unique(requests.peptide[fragments.request], return_inverse=True)
    ladders = Ladders([sites_key(requests.peptides[p]) for p in used], ion_types, max_fragment_charge)
    length = ladders.lengths[local]
    position = np.asarray(bond_position(fragments.ion_type, fragments.ion_number.astype(np.int64), length))
    ours = ladders.mz(local, fragments.ion_type, position, fragments.fragment_charge, fragments.loss, fragments.loss_ids)
    return float(np.max(np.abs(fragments.mz_reported - ours) / ours) * 1e6)
