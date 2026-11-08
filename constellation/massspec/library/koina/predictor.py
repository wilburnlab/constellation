"""Koina's models as predictors (``library.prediction.base.Predictor``).

The Koina server is one more predictor behind the contract JChronologer's
in-process models meet: its own input formatting and output parsing
around a network call.

* ``check`` is local and runs before any request: Koina's modified
  sequence is written straight from the request's unlabelled sites
  (residue ``[UNIMOD:n]`` brackets, ``[UNIMOD:n]-`` on the N-terminus for
  models that take one), and the overlay's limits (``models.Ms2Model`` /
  ``RtModel``: modifications, length, charges) refuse the rest with a
  reason, instead of a 5M-request job failing server-side.
* ``predict`` sends what was checked. Fragment annotations are parsed
  once per distinct string; Koina's padding (-1) and anything under
  ``min_intensity`` are left out; the m/z Koina returns ride along as
  ``mz_reported``, checked against the peptide sent and never used as the
  library's m/z (``prediction.assemble``).
* Retention time is sent the Koina-formatted sequence, the same dialect
  as the fragment model (``predict_rt`` sent canonical ProForma), with
  pyro-Glu on its residue for Chronologer_RT (``RtModel.
  pyro_glu_on_first_residue``), the only spelling for which Koina's
  Chronologer gives JChronologer's retention time.

Construction reads the model's declared inputs from the server (or from
a ``client`` given), so whether it takes a collision energy is known
before anything is predicted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from constellation.massspec.library.koina._annotation import _ANNOTATION_RE, _LETTER_TO_ION_TYPE, decode_annotation
from constellation.massspec.library.koina.client import PredictClient, make_client, resolve_server
from constellation.massspec.library.koina.models import ms2_model, rt_model
from constellation.massspec.library.prediction.base import (
    CHARGE_OUT_OF_RANGE,
    COLLISION_ENERGY_OUT_OF_RANGE,
    Capabilities,
    FragmentArrays,
    Refusals,
    Requests,
    ValueArray,
    merge_refusals,
)
from constellation.massspec.peptide.neutral_losses import LOSS_REGISTRY
from constellation.massspec.tokenize.vocabulary import (
    LENGTH_OUT_OF_RANGE,
    UNKNOWN_RESIDUE,
    UNSUPPORTED_MODIFICATION,
    Sites,
)

__all__ = ["KoinaFragments", "KoinaRetentionTime", "fragment_capabilities", "koina_sequence"]

_STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")


_PYRO = {"Q": "UNIMOD:28", "E": "UNIMOD:27"}


def koina_sequence(sites: Sites, *, pyro_glu_on_first_residue: bool = False) -> str:
    """Koina's modified sequence for unlabelled sites (no C-terminal
    modification: Koina's models take none). With
    ``pyro_glu_on_first_residue``, N-terminal pyro-Glu is written on its
    residue (``Q[UNIMOD:28]…``), as Koina's Chronologer_RT reads it."""
    n_term = list(sites.n_term)
    mods = list(sites.residue_mods)
    if pyro_glu_on_first_residue and sites.residues and _PYRO.get(sites.residues[0]) in n_term:
        n_term.remove(_PYRO[sites.residues[0]])
        mods[0] = (_PYRO[sites.residues[0]], *mods[0])
    head = "".join(f"[{a}]-" for a in n_term)
    return head + "".join(r + "".join(f"[{a}]" for a in m) for r, m in zip(sites.residues, mods))


def _peptide_refusal(
    sites: Sites, *, supported, allow_n_term: bool, max_length: int | None, model: str
) -> tuple[str, str] | None:
    if sites.c_term:
        return UNSUPPORTED_MODIFICATION, f"C-terminal {', '.join(sites.c_term)} ({model} takes none)"
    if sites.n_term and not allow_n_term:
        return UNSUPPORTED_MODIFICATION, f"N-terminal {', '.join(sites.n_term)} ({model} takes none)"
    bad = [r for r in sites.residues if r not in _STANDARD]
    if bad:
        return UNKNOWN_RESIDUE, f"{bad[0]} in {sites.residues}"
    if max_length is not None and len(sites.residues) > max_length:
        return LENGTH_OUT_OF_RANGE, f"{len(sites.residues)} residues ({model} takes up to {max_length})"
    if supported is not None:
        for i, mods in enumerate(sites.residue_mods):
            for a in mods:
                if a not in supported:
                    return UNSUPPORTED_MODIFICATION, f"{sites.describe(i)} (outside {model}'s {sorted(supported)})"
        for a in sites.n_term:
            if a not in supported:
                return UNSUPPORTED_MODIFICATION, f"N-terminal {a} (outside {model}'s {sorted(supported)})"
    return None


class _KoinaPredictor:
    def __init__(self, model: str, *, server: str | None = None, client: PredictClient | None = None) -> None:
        self.model = model
        self.server = resolve_server(server)
        self.client = client if client is not None else make_client(model, server=server)
        self.declared = set(self.client.model_inputs)
        self.name = f"koina:{model}"

    def _peptide_refusals(self, requests: Requests, **limits: Any) -> dict[int, tuple[str, str]]:
        peptides, inverse = requests.unique_peptides()
        per = [_peptide_refusal(requests.peptides[p], model=self.model, **limits) for p in peptides.tolist()]
        return {row: per[k] for row, k in enumerate(inverse.tolist()) if per[k] is not None}


def fragment_capabilities(
    overlay: Any,
    *,
    min_intensity: float,
    fragmentation: str | None = None,
    needs_collision_energy: bool = False,
) -> Capabilities:
    """What a Koina fragment model speaks for, from its overlay: its ion
    series, fragment charges and losses, and its floor (an ion is kept
    only above ``min_intensity``, ``assemble``'s rule). One record for
    both ways Koina predicts a library, through the prediction contract
    and ``predict_library``'s own specs path."""
    return Capabilities(
        property="fragments",
        needs_charge=True,
        needs_collision_energy=needs_collision_energy,
        ion_types=overlay.ion_types,
        max_fragment_charge=overlay.max_fragment_charge,
        losses=overlay.neutral_losses,
        fragmentation=fragmentation or overlay.fragmentation,
        min_intensity=float(min_intensity),
        intensity_floor="above",
    )


class KoinaFragments(_KoinaPredictor):
    """A Koina fragment-intensity model."""

    def __init__(
        self,
        model: str = "Prosit_2020_intensity_HCD",
        *,
        server: str | None = None,
        client: PredictClient | None = None,
        min_intensity: float = 1e-4,
        fragmentation: str | None = None,
        instrument: str = "LUMOS",
    ) -> None:
        super().__init__(model, server=server, client=client)
        self.overlay = ms2_model(model)
        self.min_intensity = float(min_intensity)
        self.fragmentation = fragmentation or self.overlay.fragmentation
        self.instrument = instrument
        self.capabilities = fragment_capabilities(
            self.overlay,
            min_intensity=self.min_intensity,
            fragmentation=self.fragmentation,
            needs_collision_energy="collision_energies" in self.declared,
        )
        self.unparsed = 0

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "koina",
            "model": self.model,
            "server": self.server,
            "min_intensity": self.min_intensity,
            "fragmentation": self.fragmentation,
            "instrument": self.instrument if "instrument_types" in self.declared else None,
            "takes_collision_energy": self.capabilities.needs_collision_energy,
        }

    def check(self, requests: Requests) -> Refusals:
        o = self.overlay
        charges = np.flatnonzero((requests.charge < o.min_charge) | (requests.charge > o.max_charge))
        energy = requests.collision_energy
        no_energy = (
            np.flatnonzero(~np.isfinite(energy)) if self.capabilities.needs_collision_energy else np.zeros(0, np.int64)
        )
        return merge_refusals(
            len(requests),
            self._peptide_refusals(
                requests, supported=o.supported_mods, allow_n_term=o.supports_n_term_mods, max_length=o.max_length
            ),
            {int(r): (CHARGE_OUT_OF_RANGE, f"charge {int(requests.charge[r])} (takes {o.min_charge}-{o.max_charge})") for r in charges},
            {int(r): (COLLISION_ENERGY_OUT_OF_RANGE, "no collision energy") for r in no_energy},
        )

    def predict(self, requests: Requests) -> FragmentArrays:
        n = len(requests)
        if not n:
            empty = np.zeros(0, np.int64)
            return FragmentArrays(empty, empty.astype(np.int8), empty.astype(np.int32), empty.astype(np.int32), empty.astype(float))
        sequences = [koina_sequence(requests.peptides[p]) for p in requests.peptide.tolist()]
        arrays: dict[str, np.ndarray] = {
            "peptide_sequences": np.array(sequences, dtype=object).reshape(-1, 1),
            "precursor_charges": requests.charge.astype(np.int32).reshape(-1, 1),
        }
        if self.capabilities.needs_collision_energy:
            arrays["collision_energies"] = requests.collision_energy.astype(np.float32).reshape(-1, 1)
        if "fragmentation_types" in self.declared:
            arrays["fragmentation_types"] = np.array([self.fragmentation] * n, dtype=object).reshape(-1, 1)
        if "instrument_types" in self.declared:
            arrays["instrument_types"] = np.array([self.instrument] * n, dtype=object).reshape(-1, 1)
        response = self.client.predict(arrays, min_intensity=self.min_intensity)
        return self._decode(response, requests)

    def _decode(self, response: dict, requests: Requests) -> FragmentArrays:
        intensity = np.asarray(response["intensities"], dtype=np.float64)
        mz = np.asarray(response["mz"], dtype=np.float64)
        annotation = np.asarray(response["annotation"])
        if intensity.shape[0] != len(requests):
            raise ValueError(f"{self.model} returned {intensity.shape[0]} rows for {len(requests)} requests")
        rows, cols = np.nonzero(intensity > self.min_intensity)
        labels = annotation[rows, cols]
        distinct, at = np.unique(np.array([decode_annotation(x) for x in labels.tolist()], dtype=object), return_inverse=True)
        losses = list(LOSS_REGISTRY.ids())
        parsed = np.full((len(distinct), 4), -1, dtype=np.int64)  # ion type, number, charge, loss (0 = none)
        for k, text in enumerate(distinct.tolist()):
            match = _ANNOTATION_RE.match(text)
            if match is None or (match["loss"] is not None and match["loss"] not in losses):
                continue
            loss = 0 if match["loss"] is None else losses.index(match["loss"]) + 1
            parsed[k] = (int(_LETTER_TO_ION_TYPE[match["letter"]]), int(match["ordinal"]), int(match["charge"]), loss)
        p = parsed[at]
        length = np.fromiter((len(requests.peptides[q].residues) for q in requests.peptide[rows].tolist()), np.int64, len(rows))
        ok = (p[:, 0] >= 0) & (p[:, 1] >= 1) & (p[:, 1] < length)
        self.unparsed += int((~ok).sum())
        rows, cols, p = rows[ok], cols[ok], p[ok]
        return FragmentArrays(
            request=rows.astype(np.int64),
            ion_type=p[:, 0].astype(np.int8),
            ion_number=p[:, 1].astype(np.int32),
            fragment_charge=p[:, 2].astype(np.int32),
            intensity=intensity[rows, cols],
            loss=p[:, 3].astype(np.int16) if (p[:, 3] > 0).any() else None,
            loss_ids=tuple(losses),
            mz_reported=mz[rows, cols],
        )


class KoinaRetentionTime(_KoinaPredictor):
    """A Koina retention-time model; seconds where the overlay knows the
    unit (Chronologer_RT's minutes), the model's own scale otherwise."""

    def __init__(self, model: str = "Chronologer_RT", *, server: str | None = None, client: PredictClient | None = None) -> None:
        super().__init__(model, server=server, client=client)
        self.overlay = rt_model(model)
        self.capabilities = Capabilities(property="retention_time", rt_scale=f"koina:{model}:{self.overlay.scale}")

    def identity(self) -> dict[str, Any]:
        return {"backend": "koina", "model": self.model, "server": self.server, "scale": self.overlay.scale}

    def check(self, requests: Requests) -> Refusals:
        o = self.overlay
        return merge_refusals(
            len(requests),
            self._peptide_refusals(requests, supported=o.supported_mods, allow_n_term=True, max_length=o.max_length),
        )

    def predict(self, requests: Requests) -> ValueArray:
        if not len(requests):
            return ValueArray(np.zeros(0), unit="s")
        peptides, inverse = requests.unique_peptides()
        on_residue = self.overlay.pyro_glu_on_first_residue
        sequences = [koina_sequence(requests.peptides[p], pyro_glu_on_first_residue=on_residue) for p in peptides.tolist()]
        response = self.client.predict({"peptide_sequences": np.array(sequences, dtype=object).reshape(-1, 1)})
        outputs = list(self.client.model_outputs)
        column = next((c for c in ("rt", "irt") if c in outputs), outputs[0] if len(outputs) == 1 else None)
        if column is None:
            raise ValueError(f"cannot identify the RT output of {self.model}; it declares {outputs}")
        values = np.asarray(response[column], dtype=np.float64).ravel() * self.overlay.to_seconds
        return ValueArray(values[inverse], unit="s" if self.overlay.scale == "minutes" else self.overlay.scale)
