"""``BasecallerModel`` ABC — physics-grounded nanopore basecaller contract.

Direction the lab is pursuing: replace black-box CNN/transformer
basecallers (Dorado today) with models that explicitly parameterize
the *physics* of nanopore translocation:

    - per-k-mer mean current (electrostatic + steric contributions)
    - dwell-time distributions (Brownian motion in the pore + helicase
      ratchet kinetics)
    - modified-base current shifts (m6A, m5C, ψ direct-current effects)
    - chemistry-specific calibration (R10.4.1 vs RNA004 vs future pores)

These concerns map naturally to ``core.stats.Parametric`` — each
component is a fittable distribution / function, optimization goes
through ``core.optim``. The ``BasecallerModel`` ABC just collects them
into a basecalling-shaped contract.

Concrete models will train on in-house synthetic libraries — defined
oligonucleotide sequences with known position-resolved modifications
that the lab is generating. Synthesis-and-DC-analysis (`current.py`)
sit alongside this module.

Status: scaffold-only. The ABC shape is provisional and will be
refined as the lab's basecaller research matures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch

# Avoid importing torch.nn at module level — torch.nn.Module subclasses
# don't compose cleanly with @dataclass slots; concrete subclasses
# below will inherit nn.Module the conventional way.


class BasecallerModel(ABC):
    """Abstract physics-based nanopore basecaller.

    Implementations subclass ``torch.nn.Module`` *and* this ABC. The
    forward path from squiggle → bases is the contract:

        forward(signal_pa, lengths) -> (sequences, qualities)

    where signals are mean+MAD-normalized picoamperes (output of
    :mod:`signal.normalize`) and outputs are per-batch-row sequences
    + Phred Q-scores.

    Modified-base mode rides as a constructor argument; concrete models
    decide whether mods are emitted alongside canonical bases (single
    pass) or via a separate mod head (two-stage).
    """

    name: ClassVar[str]
    chemistry: ClassVar[str]  # e.g. 'dna_r10.4.1_e8.2', 'rna004_130bps'

    @abstractmethod
    def forward(
        self,
        signal_pa: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[list[str], list[str]]:
        """Decode normalized squiggles into ``(sequences, qualities)``.

        ``sequences`` are nucleotide strings; ``qualities`` are
        offset-33 Phred ASCII strings. Length lists are equal.
        """


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


BASECALLER_MODEL: dict[str, type[BasecallerModel]] = {}


def register_model(cls: type[BasecallerModel]) -> type[BasecallerModel]:
    """Class decorator: register a concrete BasecallerModel by name."""
    if not getattr(cls, "name", None):
        raise ValueError(
            f"{cls.__name__}: BasecallerModel subclass must declare class var 'name'"
        )
    if cls.name in BASECALLER_MODEL:
        raise ValueError(f"basecaller model already registered: {cls.name!r}")
    BASECALLER_MODEL[cls.name] = cls
    return cls


def find_model(name: str) -> type[BasecallerModel]:
    if name not in BASECALLER_MODEL:
        raise KeyError(
            f"basecaller model {name!r} not registered "
            f"(known: {sorted(BASECALLER_MODEL)})"
        )
    return BASECALLER_MODEL[name]


__all__ = [
    "BasecallerModel",
    "BASECALLER_MODEL",
    "register_model",
    "find_model",
]
