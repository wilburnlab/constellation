"""Koina response arrays → a validated ``Library``.

Two decisions here are load-bearing:

**Fragment m/z comes from our own ladder, not from Koina's ``mz``
column.** Constellation's physics is the source of truth (the same rule
the MSP reader follows), and it means a Library built here is directly
comparable to one built anywhere else in the package. Koina's returned
m/z is then used as a *checksum*: if our modseq translation described a
different molecule than the one the server actually scored, the two
disagree. That makes an entire class of silent translation bugs loud,
for the cost of one array subtraction. The worst deviation is reported
in ``AssemblyStats`` and surfaced into the run manifest.

**Unparseable annotations are kept, not dropped.** They ride as
partial-ID rows with NULL ``ion_type``/``position``/``charge`` and the
raw string preserved, per ``LIBRARY_FRAGMENT_TABLE``'s nullability
contract, so a future grammar upgrade can re-parse them without another
round trip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.library.digest import PrecursorSpec
from constellation.massspec.library.koina._annotation import (
    decode_annotation,
    parse_koina_annotation,
)
from constellation.massspec.library.koina.models import Ms2Model
from constellation.massspec.library.library import Library, assign_ids
from constellation.massspec.peptide.ions import fragment_ladder_indices_batch

#: Prosit pads unused positions in its fixed-width output grid with -1.
#: These are not low-intensity peaks; they are absent ones.
_PAD_SENTINEL = 0.0


@dataclass(slots=True)
class AssemblyStats:
    """What happened during assembly — folded into the run manifest."""

    n_precursors: int = 0
    n_fragments: int = 0
    n_unparseable_annotations: int = 0
    n_ladder_misses: int = 0
    max_abs_ppm_deviation: float = 0.0
    worst_ppm_annotation: str | None = None
    #: Loss ids the response used that the model overlay does not
    #: declare. Non-empty means those peaks became partial-ID rows and
    #: skipped the m/z cross-check — add them to the overlay's
    #: ``neutral_losses``.
    undeclared_loss_ids: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_precursors": self.n_precursors,
            "n_fragments": self.n_fragments,
            "n_unparseable_annotations": self.n_unparseable_annotations,
            "n_ladder_misses": self.n_ladder_misses,
            "max_abs_ppm_deviation": round(self.max_abs_ppm_deviation, 4),
            "worst_ppm_annotation": self.worst_ppm_annotation,
            "undeclared_loss_ids": sorted(self.undeclared_loss_ids),
        }


@dataclass(slots=True)
class _Accum:
    proteins: list[dict[str, Any]] = field(default_factory=list)
    peptides: list[dict[str, Any]] = field(default_factory=list)
    precursors: list[dict[str, Any]] = field(default_factory=list)
    fragments: list[dict[str, Any]] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)


def assemble_library(
    specs: Sequence[PrecursorSpec],
    ms2_response: Mapping[str, np.ndarray],
    *,
    ms2_model: Ms2Model,
    rt_seconds: Mapping[str, float] | None = None,
    min_intensity: float = 1e-4,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Library, AssemblyStats]:
    """Build a ``Library`` from a Koina MS2 response.

    ``ms2_response`` rows must correspond 1:1 and in order with ``specs``
    — that is the contract ``api.predict_fragments`` maintains.
    ``rt_seconds`` maps modified sequence → predicted RT in **seconds**
    (already unit-converted); precursors without an entry get the
    ``-1.0`` "not set" sentinel per the package-wide convention.
    """
    intensities = np.asarray(ms2_response["intensities"], dtype=np.float64)
    mzs = np.asarray(ms2_response["mz"], dtype=np.float64)
    annotations = np.asarray(ms2_response["annotation"])

    if intensities.shape[0] != len(specs):
        raise ValueError(
            f"response has {intensities.shape[0]} rows but {len(specs)} "
            f"precursors were requested — the response is not aligned "
            f"with the request"
        )

    stats = AssemblyStats(n_precursors=len(specs))
    acc = _Accum()

    # One ladder per distinct peptidoform, not per precursor: a peptide
    # measured at 2+ and 3+ shares its (charge-naive) fragment masses.
    peptidoforms = {s.modified_sequence: parse_proforma(s.modified_sequence) for s in specs}
    order = list(peptidoforms)
    ladders = fragment_ladder_indices_batch(
        [peptidoforms[m] for m in order],
        ion_types=tuple(ms2_model.ion_types),
        max_fragment_charge=ms2_model.max_fragment_charge,
        # Without the model's loss channels every y3-H2O+1 misses the
        # ladder and becomes a partial-ID row. Empty for every model
        # registered today (all emit bare b/y); declared per-model so a
        # loss-emitting model works without changing this code.
        neutral_losses=list(ms2_model.neutral_losses) or None,
    )
    ladder_by_modseq = dict(zip(order, ladders, strict=True))

    seen_proteins: set[str] = set()
    seen_peptides: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    for i, spec in enumerate(specs):
        modseq = spec.modified_sequence
        if modseq not in seen_peptides:
            seen_peptides.add(modseq)
            acc.peptides.append(
                {"modified_sequence": modseq, "sequence": spec.sequence}
            )
        for accession in spec.proteins:
            if accession not in seen_proteins:
                seen_proteins.add(accession)
                acc.proteins.append({"accession": accession})
            if (accession, modseq) not in seen_edges:
                seen_edges.add((accession, modseq))
                acc.edges.append((accession, modseq))

        acc.precursors.append(
            {
                "modified_sequence": modseq,
                "charge": spec.charge,
                "precursor_mz": spec.precursor_mz,
                "rt_predicted": float((rt_seconds or {}).get(modseq, -1.0)),
            }
        )

        ladder = ladder_by_modseq[modseq]
        seq_len = len(spec.sequence)
        keep = np.flatnonzero(intensities[i] > max(min_intensity, _PAD_SENTINEL))
        for j in keep:
            raw = annotations[i, j]
            key = parse_koina_annotation(raw, seq_len)
            annotation = decode_annotation(raw)
            predicted_mz = float(mzs[i, j])

            if key is None:
                stats.n_unparseable_annotations += 1
                ion_type = position = charge = loss_id = None
                mz_theoretical = predicted_mz
            else:
                ion_type, position, charge, loss_id = key
                local = ladder.get(key)
                if local is None:
                    # In the model's grid but off our biochem-licensed
                    # ladder — keep as partial-ID rather than invent a mass.
                    stats.n_ladder_misses += 1
                    if loss_id is not None:
                        # A miss carrying a loss id means the overlay is
                        # missing that channel, which is fixable — name it
                        # rather than silently dropping the identity.
                        stats.undeclared_loss_ids.add(loss_id)
                    ion_type = position = charge = loss_id = None
                    mz_theoretical = predicted_mz
                else:
                    mz_theoretical = local
                    if predicted_mz > 0:
                        ppm = abs(local - predicted_mz) / predicted_mz * 1e6
                        if ppm > stats.max_abs_ppm_deviation:
                            stats.max_abs_ppm_deviation = ppm
                            stats.worst_ppm_annotation = f"{modseq}/{spec.charge} {annotation}"

            acc.fragments.append(
                {
                    "modified_sequence": modseq,
                    "precursor_charge": spec.charge,
                    "ion_type": ion_type,
                    "position": position,
                    "charge": charge,
                    "loss_id": loss_id,
                    "mz_theoretical": mz_theoretical,
                    "intensity_predicted": float(intensities[i, j]),
                    "annotation": annotation,
                }
            )

    stats.n_fragments = len(acc.fragments)
    library = assign_ids(
        proteins=acc.proteins,
        peptides=acc.peptides,
        precursors=acc.precursors,
        fragments=acc.fragments,
        protein_peptide=acc.edges,
        metadata=dict(metadata or {}),
    )
    return library, stats


__all__ = ["AssemblyStats", "assemble_library"]
