"""JChronologer's four models as predictors (``prediction.base.Predictor``).

Inputs and decoding follow jchronologer 1.0.10
(``DefaultChronologerLibraryPredictor``, ``CartographerSpectrumDecoder``,
``PeptideMassCalculator.isFragmentPossible``), in the jar's float32
arithmetic:

=========================  ==================  ==========================================
predictor                  inputs              output
=========================  ==================  ==========================================
ChronologerRetentionTime   tokens[52]          ``float32(pred_hi * 60f)`` seconds, on
                                               Chronologer's scale (not a run's)
CartographerFragments      tokens[33], charge  174 values: for ion number n = 1..29 and
                           one-hot[6],         fragment charge c = 1..3, y at
                           NCE / 100           ``6(n-1) + c-1`` and b at ``6(n-1) + 3 + c-1``;
                                               kept when finite, >= 0, >= ``min_intensity``
                                               (0.01), n < length, c <= min(3, z)
ElectricianChargeState     tokens[33]          6 values; the finite positive ones over
                           (Cartographer's)    their double sum, as float: P(z = 1..6)
SculptorCcs                tokens[52], charge  ``ccs_norm * (float)std + (float)mean`` Å²
                           one-hot[6]
=========================  ==================  ==========================================

The energy is the jar's: a request's float32 collision energy, widened to
double, divided by 100 and narrowed to float32. Energies outside 10–60 and
charges outside 1–6 are refused, as the jar refuses them. Each predictor
tokenizes the distinct peptides of its requests once (``check`` and
``predict`` share it) and runs them through ``prediction.torch_runner``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from constellation.massspec.library.jchronologer.bundle import JChronologerBundle, locate_bundle
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
from constellation.massspec.library.prediction.torch_runner import DEFAULT_BATCH_SIZE, run_batched
from constellation.massspec.peptide.ions import IonType
from constellation.massspec.tokenize.vocabulary import TokenizedBatch

__all__ = [
    "MAX_NCE",
    "MIN_NCE",
    "CartographerFragments",
    "ChronologerRetentionTime",
    "ElectricianChargeState",
    "JChronologerPredictor",
    "SculptorCcs",
    "RT_SCALE",
]

#: ``DefaultChronologerLibraryPredictor.MIN_NCE`` / ``MAX_NCE``.
MIN_NCE, MAX_NCE = 10.0, 60.0
#: ``ChronologerLibraryOptions.DEFAULT_MINIMUM_REPORTED_INTENSITY``.
DEFAULT_MIN_INTENSITY = 0.01
#: Chronologer's retention times are seconds on its own hydrophobicity
#: index scale (pred_hi × 60), not any run's.
RT_SCALE = "chronologer_hi_seconds"
_MAX_ION_NUMBER = 29


class JChronologerPredictor:
    """What the four share: the bundle, the tokenizer, the model, the runtime."""

    role: str = ""
    capabilities: Capabilities

    def __init__(
        self,
        bundle: JChronologerBundle | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int = 1,
        device: str = "cpu",
    ) -> None:
        self.bundle = bundle if bundle is not None else locate_bundle()
        self.tokenizer = self.bundle.tokenizers[self.role]
        self.preprocessing = self.bundle.preprocessing(self.role)
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        self.device = device
        self.name = f"jchronologer:{self.role}"
        self._tokens: tuple[tuple, TokenizedBatch] | None = None

    def identity(self) -> dict[str, Any]:
        return {**self.bundle.identity(self.role), "batch_size": self.batch_size}

    # ── tokens, once per peptide tuple ──────────────────────────────
    def _tokenized(self, requests: Requests) -> TokenizedBatch:
        """Tokens of every peptide in ``requests.peptides`` (indexed like it)."""
        if self._tokens is None or self._tokens[0] is not requests.peptides:
            self._tokens = (requests.peptides, self.tokenizer.encode(requests.peptides))
        return self._tokens[1]

    def _peptide_refusals(self, requests: Requests) -> dict[int, tuple[str, str]]:
        batch = self._tokenized(requests)
        out: dict[int, tuple[str, str]] = {}
        for row, peptide in enumerate(requests.peptide.tolist()):
            reason = batch.reasons[peptide]
            if reason is not None:
                out[row] = (reason, batch.details[peptide])
        return out

    def _charge_refusals(self, requests: Requests, low: int, high: int) -> dict[int, tuple[str, str]]:
        bad = np.flatnonzero((requests.charge < low) | (requests.charge > high))
        return {int(r): (CHARGE_OUT_OF_RANGE, f"charge {int(requests.charge[r])} (takes {low}-{high})") for r in bad}

    def check(self, requests: Requests) -> Refusals:
        return merge_refusals(len(requests), self._peptide_refusals(requests))

    def _model(self):
        return self.bundle.load_model(self.role, self.device)

    def _run(self, *inputs: np.ndarray) -> np.ndarray:
        return run_batched(self._model(), inputs, batch_size=self.batch_size, threads=self.threads, device=self.device)


def _one_hot(charge: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((len(charge), width), dtype=np.float32)
    out[np.arange(len(charge)), charge.astype(np.int64) - 1] = 1.0
    return out


class ChronologerRetentionTime(JChronologerPredictor):
    """Retention time (Chronologer), one prediction per peptide."""

    role = "chronologer"
    capabilities = Capabilities(property="retention_time", rt_scale=RT_SCALE)

    def predict(self, requests: Requests) -> ValueArray:
        if not len(requests):
            return ValueArray(np.zeros(0), unit="s", extra={"scale": RT_SCALE})
        peptides, inverse = requests.unique_peptides()
        tokens = self._tokenized(requests).tokens[peptides]
        pred_hi = self._run(tokens).reshape(-1).astype(np.float32)
        seconds = (pred_hi * np.float32(60.0)).astype(np.float32)
        return ValueArray(seconds[inverse].astype(np.float64), unit="s", extra={"scale": RT_SCALE})


class CartographerFragments(JChronologerPredictor):
    """Fragment intensities (JChronologer's Cartographer model): b and y,
    fragment charges 1–3, no losses, HCD."""

    role = "cartographer"
    capabilities = Capabilities(
        property="fragments",
        needs_charge=True,
        needs_collision_energy=True,
        ion_types=(IonType.B, IonType.Y),
        max_fragment_charge=3,
        fragmentation="HCD",
    )

    def __init__(self, bundle: JChronologerBundle | None = None, *, min_intensity: float = DEFAULT_MIN_INTENSITY, **kw: Any) -> None:
        super().__init__(bundle, **kw)
        self.min_intensity = float(min_intensity)
        self.min_charge = int(self.preprocessing["min_precursor_charge"])
        self.max_charge = int(self.preprocessing["max_precursor_charge"])

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), "min_intensity": self.min_intensity}

    def check(self, requests: Requests) -> Refusals:
        energy = requests.collision_energy.astype(np.float64)
        bad = np.flatnonzero(~np.isfinite(energy) | (energy < MIN_NCE) | (energy > MAX_NCE))
        energies = {
            int(r): (
                COLLISION_ENERGY_OUT_OF_RANGE,
                "no collision energy" if not np.isfinite(energy[r]) else f"NCE {energy[r]:g} (takes {MIN_NCE:g}-{MAX_NCE:g})",
            )
            for r in bad
        }
        return merge_refusals(
            len(requests),
            self._peptide_refusals(requests),
            self._charge_refusals(requests, self.min_charge, self.max_charge),
            energies,
        )

    def predict(self, requests: Requests) -> FragmentArrays:
        if not len(requests):
            return _no_fragments()
        tokens = self._tokenized(requests).tokens[requests.peptide]
        charge = requests.charge.astype(np.int64)
        # The jar's arithmetic: a float energy, widened, over 100.0, narrowed.
        nce = (requests.collision_energy.astype(np.float64) / 100.0).astype(np.float32).reshape(-1, 1)
        out = self._run(tokens, _one_hot(charge - self.min_charge + 1, self.max_charge - self.min_charge + 1), nce)
        return decode_cartographer(out, self._lengths(requests), charge, self.min_intensity)

    def _lengths(self, requests: Requests) -> np.ndarray:
        lengths = np.fromiter((len(s.residues) for s in requests.peptides), np.int64, len(requests.peptides))
        return lengths[requests.peptide]


def _no_fragments() -> FragmentArrays:
    empty = np.zeros(0, dtype=np.int64)
    return FragmentArrays(empty, empty.astype(np.int8), empty.astype(np.int32), empty.astype(np.int32), empty.astype(np.float64))


# Column k of Cartographer's 174 outputs: ion number, fragment charge, series.
_COLUMN_NUMBER = np.repeat(np.arange(1, _MAX_ION_NUMBER + 1), 6)
_COLUMN_CHARGE = np.tile(np.array([1, 2, 3, 1, 2, 3]), _MAX_ION_NUMBER)
_COLUMN_TYPE = np.tile(np.array([IonType.Y] * 3 + [IonType.B] * 3, dtype=np.int8), _MAX_ION_NUMBER)


def decode_cartographer(
    out: np.ndarray, length: np.ndarray, precursor_charge: np.ndarray, min_intensity: float = DEFAULT_MIN_INTENSITY
) -> FragmentArrays:
    """``CartographerSpectrumDecoder.decode`` over a batch of 174-value rows."""
    values = np.asarray(out, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 * _MAX_ION_NUMBER:
        raise ValueError(f"expected Cartographer rows of {6 * _MAX_ION_NUMBER} values, got {values.shape}")
    keep = np.isfinite(values) & (values >= np.float32(0.0)) & (values >= np.float32(min_intensity))
    keep &= _COLUMN_NUMBER[None, :] < np.asarray(length)[:, None]
    keep &= _COLUMN_CHARGE[None, :] <= np.minimum(3, np.asarray(precursor_charge))[:, None]
    rows, cols = np.nonzero(keep)
    return FragmentArrays(
        request=rows.astype(np.int64),
        ion_type=_COLUMN_TYPE[cols],
        ion_number=_COLUMN_NUMBER[cols].astype(np.int32),
        fragment_charge=_COLUMN_CHARGE[cols].astype(np.int32),
        intensity=values[rows, cols].astype(np.float64),
    )


class ElectricianChargeState(JChronologerPredictor):
    """Charge-state distribution (Electrician), read from Cartographer's
    tokens; the value of a request is P(z = its charge)."""

    role = "electrician"
    capabilities = Capabilities(property="charge_state")

    def predict(self, requests: Requests) -> ValueArray:
        if not len(requests):
            return ValueArray(np.zeros(0), unit="probability")
        peptides, inverse = requests.unique_peptides()
        tokens = self._tokenized(requests).tokens[peptides]
        distribution = normalize_charge_distribution(self._run(tokens))
        z = requests.charge.astype(np.int64)
        values = np.full(len(requests), np.nan)
        ok = (z >= 1) & (z <= distribution.shape[1])
        values[ok] = distribution[inverse[ok], z[ok] - 1]
        return ValueArray(values, unit="probability")


def normalize_charge_distribution(out: np.ndarray) -> np.ndarray:
    """``DefaultChronologerLibraryPredictor.normalizeChargeDistribution``:
    finite positive values over their (double, in order) sum, as float32;
    all zero where none is positive."""
    values = np.asarray(out, dtype=np.float32)
    positive = np.where(np.isfinite(values) & (values > 0), values, np.float32(0.0))
    total = np.zeros(len(values), dtype=np.float64)
    for k in range(values.shape[1]):
        total += positive[:, k].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (positive.astype(np.float64) / total[:, None]).astype(np.float32)
    normalized[total <= 0.0] = 0.0
    return normalized


class SculptorCcs(JChronologerPredictor):
    """Collision cross section (Sculptor), in Å²."""

    role = "sculptor"
    capabilities = Capabilities(property="ccs", needs_charge=True)

    def __init__(self, bundle: JChronologerBundle | None = None, **kw: Any) -> None:
        super().__init__(bundle, **kw)
        self.charge_states = int(self.preprocessing["charge_dist_len"])
        self.mean = np.float32(self.preprocessing["train_ccs_mean"])
        self.std = np.float32(self.preprocessing["train_ccs_std"])

    def check(self, requests: Requests) -> Refusals:
        return merge_refusals(
            len(requests), self._peptide_refusals(requests), self._charge_refusals(requests, 1, self.charge_states)
        )

    def predict(self, requests: Requests) -> ValueArray:
        if not len(requests):
            return ValueArray(np.zeros(0), unit="A^2")
        tokens = self._tokenized(requests).tokens[requests.peptide]
        out = self._run(tokens, _one_hot(requests.charge, self.charge_states)).reshape(-1).astype(np.float32)
        ccs = (out * self.std + self.mean).astype(np.float32)
        return ValueArray(np.where(np.isfinite(ccs), ccs, np.nan).astype(np.float64), unit="A^2")
