"""Per-model knowledge that the Koina server does **not** report.

The server already tells us each model's declared inputs, its output
names and dtypes, and its batch size (see ``client.KoinaClient``), so
none of that is duplicated here — mirroring it locally would only create
a second source of truth that silently rots when a model is retrained.

What the server does *not* expose, and therefore lives here:

* ``to_seconds`` — an RT head emits a bare float. Chronologer's is in
  minutes; the Prosit/AlphaPept/DeepLC heads emit unitless iRT.
* ``ion_types`` / ``max_fragment_charge`` — needed to build the local
  fragment ladder we cross-check the returned m/z against.
* ``max_length`` / ``max_charge`` / ``supported_mods`` — pre-flight
  limits, so a 5M-precursor job fails in the first second with a useful
  message instead of 40 minutes in with an opaque one.

Unregistered models are usable: ``ms2_model`` / ``rt_model`` fall back to
permissive defaults with a warning rather than refusing, so a model added
to Koina next year works without a Constellation release.

Limits below were measured against koina.wilhelmlab.org on 2026-08-03;
see ``tests/test_koina_models.py`` for the live regression check.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from constellation.massspec.peptide.ions import IonType

#: UNIMOD accessions the Prosit 2020 series was trained on. Anything else
#: is rejected server-side with an opaque "at least one request failed",
#: so we pre-flight it into a message that names the offending mod.
_PROSIT_2020_MODS = frozenset({"UNIMOD:4", "UNIMOD:35"})


@dataclass(frozen=True, slots=True)
class Ms2Model:
    """Overlay for a fragment-intensity model."""

    name: str
    ion_types: tuple[IonType, ...] = (IonType.B, IonType.Y)
    max_fragment_charge: int = 3
    max_length: int = 30
    max_charge: int = 6
    min_charge: int = 1
    supported_mods: frozenset[str] | None = None
    supports_n_term_mods: bool = False
    #: Value sent for a declared ``fragmentation_types`` input. The PTM
    #: models require it; the 2020 series does not declare it at all.
    fragmentation: str = "HCD"
    registered: bool = True


@dataclass(frozen=True, slots=True)
class RtModel:
    """Overlay for a retention-time model.

    ``column`` is deliberately absent — it is read from the server's
    declared outputs at call time.
    """

    name: str
    to_seconds: float = 1.0
    scale: str = "irt"
    max_length: int | None = None
    supported_mods: frozenset[str] | None = None
    registered: bool = True


MS2_MODELS: dict[str, Ms2Model] = {
    m.name: m
    for m in (
        # Limits measured live: length 1-30 (31 rejected), charge 1-6
        # (7 rejected), Cam-C + Ox-M only, no N-terminal mod dialect
        # accepted in any bracket form.
        Ms2Model("Prosit_2020_intensity_HCD", supported_mods=_PROSIT_2020_MODS),
        Ms2Model("Prosit_2020_intensity_CID", supported_mods=_PROSIT_2020_MODS),
        Ms2Model("Prosit_2023_intensity_timsTOF", supported_mods=_PROSIT_2020_MODS),
        # PTM-aware series. Limits measured live 2026-08-04 on
        # Prosit_2025_intensity_22PTM: length 1-30 (31 rejected), charge
        # 1-6 (7 rejected), fragmentation_types accepts HCD/CID/ETD/ETHCD,
        # and it takes Phospho-S/Y, Acetyl-K and N-TERMINAL acetyl, which
        # the 2020 series rejects outright.
        #
        # supported_mods is left None (permissive) rather than guessed:
        # the model advertises 22 PTMs and enumerating them from probing
        # would be a partial list presented as complete.
        #
        # NOTE the length ceiling is still 30. Peptides longer than that
        # -- including the 34-46mers carrying the EphA3 activation-loop
        # tyrosines -- cannot be predicted by any of these models.
        Ms2Model("Prosit_2025_intensity_22PTM", max_charge=6,
                 supports_n_term_mods=True),
        Ms2Model("Prosit_2025_intensity_40PTM", max_charge=6,
                 supports_n_term_mods=True),
        Ms2Model("Prosit_2024_intensity_PTMs_gl", max_charge=6,
                 supports_n_term_mods=True),
    )
}

RT_MODELS: dict[str, RtModel] = {
    m.name: m
    for m in (
        # Chronologer emits minutes on its own gradient; everything else
        # emits unitless iRT. Both are calibrated to an observed gradient
        # downstream — the conversion here only fixes the unit.
        RtModel("Chronologer_RT", to_seconds=60.0, scale="minutes"),
        RtModel("Prosit_2019_irt"),
        RtModel("Prosit_2024_irt_cit"),
        RtModel("AlphaPept_rt_generic"),
        RtModel("Deeplc_hela_hf"),
    )
}


def ms2_model(name: str) -> Ms2Model:
    """Overlay for `name`, or a permissive default with a warning."""
    try:
        return MS2_MODELS[name]
    except KeyError:
        warnings.warn(
            f"no local overlay for MS2 model {name!r} — assuming b/y ions, "
            f"fragment charge <= 3, length <= 30, and no modification "
            f"restrictions. Pre-flight validation will be weaker than for a "
            f"registered model.",
            UserWarning,
            stacklevel=2,
        )
        return Ms2Model(name, registered=False)


def rt_model(name: str) -> RtModel:
    """Overlay for `name`, or a permissive default with a warning."""
    try:
        return RT_MODELS[name]
    except KeyError:
        warnings.warn(
            f"no local overlay for RT model {name!r} — assuming its output is "
            f"unitless iRT. If it emits minutes, rt_predicted will be wrong "
            f"by 60x; register it in massspec.library.koina.models.",
            UserWarning,
            stacklevel=2,
        )
        return RtModel(name, registered=False)


__all__ = [
    "MS2_MODELS",
    "RT_MODELS",
    "Ms2Model",
    "RtModel",
    "ms2_model",
    "rt_model",
]
