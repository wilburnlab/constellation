"""What predictors are asked about a library, and how the answers fan back out.

``build_requests(library, collision_energy)`` turns a library's
precursors into :class:`~constellation.massspec.library.prediction.base.Requests`:

* **Isotope labels are taken off.** A predictor is told about the
  chemistry, not the isotopes: a global label (``<13C><15N>``) and every
  isotopic substitution (``Modification.is_isotopic_substitution``: SILAC
  lysine and arginine) are removed; chemical modifications, TMT included,
  stay. Fragment m/z are later taken from the labelled peptide
  (``prediction.assemble``). Deuterium shifts retention time, which no
  model here knows; a deuterated peptide is predicted as its light form.
* **Each (unlabelled peptide, charge) is asked once**, in sorted order of
  the peptide's canonical string, then charge, so the requests (and the
  batches they are run in) do not depend on the library's row order. A
  heavy and a light precursor of one peptide share a request.
* **Each source precursor keeps its request**, and whether its peptide
  is a decoy. A target and the decoys made from it are not linked here:
  each precursor is kept or dropped on its own (PI, 2026-11-07; until
  then a target and its decoys at one charge were dropped together).

A peptide that has no reading as sites (a modification named by mass or
name, a localisation group) gets no request; its precursors are refused
for every predictor (``unrequestable``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pyarrow.compute as pc

from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import Peptidoform, parse_proforma
from constellation.massspec.library.library import Library
from constellation.massspec.library.prediction.base import Requests, sites_key
from constellation.massspec.library.prediction.collision_energy import CollisionEnergy
from constellation.massspec.tokenize.vocabulary import Sites, TokenizationRefused

__all__ = ["RequestPlan", "build_requests", "unlabelled_sites"]


@cache
def _is_label(accession: str) -> bool:
    try:
        return UNIMOD[accession].is_isotopic_substitution
    except KeyError:
        return False


def unlabelled_sites(modified_sequence: str) -> tuple[Sites, bool]:
    """The sites of a peptide with its isotope labels removed, and
    whether it had any. Raises ``TokenizationRefused`` for a peptide
    that has no reading as sites."""
    labelled = False
    sites = Sites._from_canonical(modified_sequence)
    if sites is None:
        peptide = parse_proforma(modified_sequence)
        if isinstance(peptide, Peptidoform) and peptide.global_isotopes:
            peptide = replace(peptide, global_isotopes=())
            labelled = True
        sites = Sites.from_peptidoform(peptide)

    def strip(accessions: tuple[str, ...]) -> tuple[str, ...]:
        nonlocal labelled
        if any(_is_label(a) for a in accessions):
            labelled = True
            return tuple(a for a in accessions if not _is_label(a))
        return accessions

    stripped = Sites(
        residues=sites.residues,
        n_term=strip(sites.n_term),
        residue_mods=tuple(strip(m) if m else m for m in sites.residue_mods),
        c_term=strip(sites.c_term),
    )
    return (stripped if labelled else sites), labelled


@dataclass(frozen=True)
class RequestPlan:
    """A library's precursors as requests, and the way back.

    Per source precursor (the library's precursor rows, in order):
    ``precursor_id``, ``peptide_id``, ``is_decoy`` and its ``request`` (−1
    if it has none). ``unrequestable`` names the precursor rows with no
    request and why.
    """

    requests: Requests
    precursor_id: np.ndarray
    peptide_id: np.ndarray
    is_decoy: np.ndarray
    request: np.ndarray
    unrequestable: dict[int, tuple[str, str]]
    labelled_peptides: int
    collision_energy: CollisionEnergy | None


def build_requests(library: Library, collision_energy: CollisionEnergy | None = None) -> RequestPlan:
    """The requests a library's precursors make (module docstring).

    ``collision_energy`` sets each request's energy from its charge
    (``CollisionEnergy.resolve``); without it the energies are NaN, for
    predictors that take none.
    """
    precursors = library.precursors
    peptides = library.peptides
    n = precursors.num_rows
    precursor_id = precursors.column("precursor_id").to_numpy()
    peptide_id = precursors.column("peptide_id").to_numpy()
    charge = precursors.column("charge").to_numpy().astype(np.int32)

    at = pc.index_in(precursors.column("peptide_id"), value_set=peptides.column("peptide_id").combine_chunks())
    at = at.to_numpy(zero_copy_only=False)
    sequences = peptides.column("modified_sequence").to_pylist()
    is_decoy_peptide = pc.fill_null(peptides.column("is_decoy"), False).to_numpy(zero_copy_only=False)

    # Unlabelled sites, once per peptide row.
    peptide_sites: list[Sites | None] = []
    peptide_refusal: dict[int, tuple[str, str]] = {}
    labelled_peptides = 0
    for row, text in enumerate(sequences):
        try:
            sites, labelled = unlabelled_sites(text)
        except TokenizationRefused as refusal:
            peptide_sites.append(None)
            peptide_refusal[row] = (refusal.reason, refusal.detail)
            continue
        peptide_sites.append(sites)
        labelled_peptides += labelled

    # Unique (sites, charge), sorted by (canonical string, charge).
    keys = {s: sites_key(s) for s in peptide_sites if s is not None}
    distinct = sorted(keys, key=keys.__getitem__)
    index_of = {s: i for i, s in enumerate(distinct)}
    request_peptide = np.full(n, -1, dtype=np.int64)
    unrequestable: dict[int, tuple[str, str]] = {}
    for row in range(n):
        sites = peptide_sites[at[row]]
        if sites is None:
            unrequestable[row] = peptide_refusal[at[row]]
        else:
            request_peptide[row] = index_of[sites]
    has = request_peptide >= 0
    pairs = np.unique(np.stack([request_peptide[has], charge[has]], axis=1), axis=0)  # sorted by (peptide, charge)
    request = np.full(n, -1, dtype=np.int64)
    if pairs.size:
        code = pairs[:, 0] * 1024 + pairs[:, 1]
        request[has] = np.searchsorted(code, request_peptide[has] * 1024 + charge[has])
    req_charge = pairs[:, 1].astype(np.int32) if pairs.size else np.zeros(0, np.int32)
    energies = (
        collision_energy.resolve(req_charge) if collision_energy is not None and req_charge.size
        else np.full(req_charge.size, np.nan, dtype=np.float32)
    )
    requests = Requests(
        peptides=tuple(distinct),
        peptide=pairs[:, 0].astype(np.int64) if pairs.size else np.zeros(0, np.int64),
        charge=req_charge,
        collision_energy=energies.astype(np.float32),
    )

    is_decoy = is_decoy_peptide[at].astype(bool)

    return RequestPlan(
        requests=requests,
        precursor_id=precursor_id,
        peptide_id=peptide_id,
        is_decoy=is_decoy,
        request=request,
        unrequestable=unrequestable,
        labelled_peptides=labelled_peptides,
        collision_energy=collision_energy,
    )

