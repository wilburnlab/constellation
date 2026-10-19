"""The search settings Constellation gives a meaning to, and how they become jar flags.

Most EncyclopeDIA flags pass through under their own names
(:mod:`.runner` checks them against the captured help). This module
holds the few where Constellation adds something:

**Which engine.** EncyclopeDIA has two search engines that matter here,
and the choice between them is the user's *search mode*, not a program
name to remember:

==================  ==========================  ==========  =============================
``--search-mode``   jar program                 DIA data    DDA data
==================  ==========================  ==========  =============================
peptide-centric     the default program (the    yes, the    refused: there is no
                    original EncyclopeDIA       default     peptide-centric DDA search
                    algorithm); needs a
                    library
spectrum-centric    ``-scribetwo`` with         yes         yes, the default (``dda``,
                    ``-searchType``; the        (``dia``)   or ``dda_chimeric`` for
                    library is optional                     chimeric spectra)
==================  ==========================  ==========  =============================

:func:`resolve_search_mode` is that table. It takes the acquisition type
Constellation inferred from the data and refuses the combinations that
make no sense, before anything runs.

**Mass tolerances.** Each of the three is a value and a unit.
Constellation says ``ppm`` or ``Da``; the jar's enum spells the second
``AMU`` and hard-errors on ``Da``. ScribeTwo adjusts the precursor and
fragment tolerances it is given (a first pass on a sample of spectra at
twice them, then the 97.5th percentile of what it found) and records
the adjusted values, not the given ones.

**ScribeTwo without a library** predicts every peptide of the FASTA
first (JChronologer), with the modifications ``ptms`` names as
``-ptm<Name> off|var|fix``, into a cache shared between searches
(``prediction_cache``; the jar's default is ``~/.encyclopedia``). The
cache only saves time, so it is not part of what a search is.

Everything a search was run with is recorded twice in the unit's
manifest: as given (``SearchParameters.as_given``) and as the exact
command line.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from constellation.massspec.search.encyclopedia.runner import FlagValue

PEPTIDE_CENTRIC = "peptide-centric"
SPECTRUM_CENTRIC = "spectrum-centric"
SEARCH_MODES = (PEPTIDE_CENTRIC, SPECTRUM_CENTRIC)

#: ``--search-type`` spelling -> the jar's ``-searchType`` token.
SEARCH_TYPES = {"auto": "auto", "dia": "dia", "dda": "dda", "dda-chimeric": "dda_chimeric"}

_UNIT_TOKENS = {"ppm": "PPM", "da": "AMU"}
_READABLE_UNITS = {"ppm": "ppm", "amu": "Da", "da": "Da"}

#: Flags that say how a search is run, not what it is: the same search
#: on another thread count is the same search (its directory and its
#: digest do not change, and summaries do not count it as a difference).
EXECUTION_ONLY_FLAGS = frozenset({"-numberOfThreadsUsed", "-predictionCache"})

#: The modifications ScribeTwo's FASTA prediction takes as
#: ``-ptm<Name> off|var|fix``. Its help gives only the pattern
#: (``-ptm...``); these are the names a 6.9.23 ScribeTwo result records
#: (every one, with its state, in its ``metadata``).
CHRONOLOGER_PTMS = (
    "Acetyl", "Carbamidomethyl", "Deamidation", "Dimethyl", "GlyGly", "HexNAc", "Methyl",
    "Oxidation", "Phospho", "ProteinNTermAcetyl", "PyroGluQ", "Succinyl", "TMT", "Trimethyl",
)  # fmt: skip
PTM_STATES = ("off", "var", "fix")

#: Settings only the spectrum-centric engine has (``SearchParameters``
#: field -> what to call it in a message).
SPECTRUM_CENTRIC_SETTINGS = {"ptms": "--ptm", "prediction_cache": "--prediction-cache"}

#: The jar's flags for a search's files, which Constellation sets itself
#: and records, hashes and keys the search by: never passed through
#: (flag -> the option that gives it, or ``None``). Passed through, they
#: would replace the file the manifest and the key describe.
FILE_FLAGS: dict[str, str | None] = {"-i": "--input", "-l": "--library", "-f": "--fasta", "-o": None}


def refuse_file_flags(flags: Mapping[str, object]) -> None:
    """Raise ``ValueError`` for a pass-through flag that names one of a
    search's files (:data:`FILE_FLAGS`)."""
    for flag in flags:
        if flag not in FILE_FLAGS:
            continue
        option = FILE_FLAGS[flag]
        raise ValueError(
            f"{flag} is not passed through: "
            + (
                f"give the file with {option}, so that the search records, hashes and is "
                f"keyed by the file it searched"
                if option
                else "the jar's outputs belong beside its input, in the search's own "
                "directory, where the rest of the workflow looks for them"
            )
        )

#: The settings a summary leads with, in Constellation's spelling:
#: ``(label, value flag, unit flag or None)``.
HEADLINE_SETTINGS: tuple[tuple[str, str, str | None], ...] = (
    ("precursor tolerance", "-ptol", "-ptolunits"),
    ("fragment tolerance", "-ftol", "-ftolunits"),
    ("library fragment tolerance", "-lftol", "-lftolunits"),
    ("enzyme", "-enzyme", None),
    ("fragmentation", "-frag", None),
    ("acquisition type", "-acquisition", None),
    ("fixed modifications", "-fixed", None),
    ("Percolator", "-percolatorVersion", None),
    ("FDR threshold", "-percolatorThreshold", None),
)


class SearchModeError(ValueError):
    """A search mode, search type and acquisition type that do not go together."""


@dataclass(frozen=True)
class ResolvedSearch:
    """Which engine a search runs on."""

    mode: str
    #: Program name as in the captured command line (``encyclopedia``,
    #: ``scribetwo``).
    program: str
    #: Flags that follow from the mode (``-searchType`` for ScribeTwo).
    flags: dict[str, FlagValue] = field(default_factory=dict)
    #: Whether the program needs ``-l``.
    library_required: bool = True


def resolve_search_mode(
    mode: str | None,
    *,
    acquisition_type: str | None,
    search_type: str | None = None,
) -> ResolvedSearch:
    """Pick the engine for a search (the table in the module docstring).

    ``acquisition_type`` is ``"DIA"`` or ``"DDA"`` as inferred from the
    data, or ``None`` when it could not be told. ``mode=None`` takes the
    default for the data: peptide-centric for DIA (and when unknown),
    spectrum-centric for DDA. ``search_type`` belongs to the
    spectrum-centric engine and is an error under the other.
    """
    kind = acquisition_type.upper() if acquisition_type else None
    if kind not in (None, "DIA", "DDA"):
        raise SearchModeError(f"acquisition type must be DIA or DDA; got {acquisition_type!r}")
    if mode is None:
        mode = SPECTRUM_CENTRIC if kind == "DDA" else PEPTIDE_CENTRIC
    if mode not in SEARCH_MODES:
        raise SearchModeError(f"search mode must be one of {', '.join(SEARCH_MODES)}; got {mode!r}")

    if mode == PEPTIDE_CENTRIC:
        if search_type is not None:
            raise SearchModeError(
                "--search-type belongs to the spectrum-centric engine (ScribeTwo); the "
                "peptide-centric search has no such setting"
            )
        if kind == "DDA":
            raise SearchModeError(
                "this is a DDA acquisition and there is no peptide-centric search of DDA "
                "data: use --search-mode spectrum-centric"
            )
        return ResolvedSearch(mode=mode, program="encyclopedia")

    chosen = search_type or "auto"
    if chosen not in SEARCH_TYPES:
        raise SearchModeError(
            f"search type must be one of {', '.join(SEARCH_TYPES)}; got {search_type!r}"
        )
    if kind == "DDA" and chosen == "dia":
        raise SearchModeError("--search-type dia was given for a DDA acquisition")
    if kind == "DIA" and chosen in ("dda", "dda-chimeric"):
        raise SearchModeError(f"--search-type {chosen} was given for a DIA acquisition")
    return ResolvedSearch(
        mode=mode,
        program="scribetwo",
        flags={"-searchType": SEARCH_TYPES[chosen]},
        library_required=False,
    )


def unit_token(unit: str) -> str:
    """``ppm`` / ``Da`` (any case) as the jar's ``PPM`` / ``AMU``."""
    token = _UNIT_TOKENS.get(str(unit).strip().lower())
    if token is None:
        raise ValueError(f"tolerance unit must be ppm or Da; got {unit!r}")
    return token


@dataclass(frozen=True)
class Tolerance:
    value: float
    unit: str = "ppm"

    def __post_init__(self) -> None:
        if not self.value > 0:
            raise ValueError(f"a mass tolerance must be positive; got {self.value}")
        unit_token(self.unit)  # validates


@dataclass(frozen=True)
class SearchParameters:
    """What a search is run with, beyond its input files.

    ``None`` means "the jar's default", and nothing is put on the
    command line for it. ``extra`` is any other documented flag, by its
    own name, checked by the runner.
    """

    precursor_tolerance: Tolerance | None = None
    fragment_tolerance: Tolerance | None = None
    library_fragment_tolerance: Tolerance | None = None
    enzyme: str | None = None
    fragmentation: str | None = None
    percolator_version: str | None = None
    percolator_threshold: float | None = None
    percolator_protein_threshold: float | None = None
    threads: int | None = None
    #: Spectrum-centric, without a library: modification -> off / var /
    #: fix for the FASTA prediction (``CHRONOLOGER_PTMS``).
    ptms: dict[str, str] = field(default_factory=dict)
    #: Spectrum-centric: where the FASTA prediction cache lives (``None``:
    #: the jar's default).
    prediction_cache: Path | str | None = None
    extra: dict[str, FlagValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, state in self.ptms.items():
            if name not in CHRONOLOGER_PTMS:
                raise ValueError(
                    f"no modification {name!r} for ScribeTwo's prediction; it knows "
                    f"{', '.join(CHRONOLOGER_PTMS)}"
                )
            if state not in PTM_STATES:
                raise ValueError(
                    f"modification {name} must be one of {', '.join(PTM_STATES)}; got {state!r}"
                )
        refuse_file_flags(self.extra)
        if self.ptms and {"-fixed", "-variable"} & set(self.extra):
            raise ValueError(
                "named modifications (--ptm) and -fixed / -variable do not mix; the jar's "
                "help says to use one or the other"
            )

    def spectrum_centric_only(self) -> list[str]:
        """The settings given that only the spectrum-centric engine has."""
        return [
            label for name, label in SPECTRUM_CENTRIC_SETTINGS.items() if getattr(self, name)
        ]

    def flags(self) -> dict[str, FlagValue]:
        """The jar flags these settings become, in a fixed order."""
        out: dict[str, FlagValue] = {}
        for tolerance, value_flag, unit_flag in (
            (self.precursor_tolerance, "-ptol", "-ptolunits"),
            (self.fragment_tolerance, "-ftol", "-ftolunits"),
            (self.library_fragment_tolerance, "-lftol", "-lftolunits"),
        ):
            if tolerance is not None:
                out[value_flag] = float(tolerance.value)
                out[unit_flag] = unit_token(tolerance.unit)
        for value, flag in (
            (self.enzyme, "-enzyme"),
            (self.fragmentation, "-frag"),
            (self.percolator_version, "-percolatorVersion"),
            (self.percolator_threshold, "-percolatorThreshold"),
            (self.percolator_protein_threshold, "-percolatorProteinThreshold"),
            (self.threads, "-numberOfThreadsUsed"),
            (self.prediction_cache, "-predictionCache"),
        ):
            if value is not None:
                out[flag] = value
        for name, state in self.ptms.items():
            out[f"-ptm{name}"] = state
        for flag, value in self.extra.items():
            if flag in out:
                raise ValueError(
                    f"{flag} is set both as a named setting and as a pass-through flag"
                )
            out[flag] = value
        return out

    def as_given(self) -> dict[str, Any]:
        """The settings in Constellation's spelling, for a manifest: only
        those that were set, tolerances as ``{"value", "unit"}``."""
        out: dict[str, Any] = {}
        for name in ("precursor_tolerance", "fragment_tolerance", "library_fragment_tolerance"):
            tolerance = getattr(self, name)
            if tolerance is not None:
                out[name] = {"value": tolerance.value, "unit": tolerance.unit}
        for name in ("enzyme", "fragmentation", "percolator_version", "percolator_threshold",
                     "percolator_protein_threshold", "threads"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.ptms:
            out["ptms"] = dict(self.ptms)
        if self.prediction_cache is not None:
            out["prediction_cache"] = str(self.prediction_cache)
        if self.extra:
            out["extra"] = {k: str(v) for k, v in self.extra.items()}
        return out


def readable_settings(
    settings: Mapping[str, str], *, flags: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    """The jar's settings as ``(label, value)`` pairs in Constellation's
    spelling: a tolerance with its unit (``0.35 Da``), no fixed
    modification as ``none``, a path as its file name.

    ``settings`` is flag -> value, as a result records them or as they
    were passed. The headline settings come first, in a fixed order.
    With ``flags``, only the settings those flags belong to, any flag
    that is not a headline one under its own name, and a flag
    ``settings`` lacks as ``(not recorded)``.
    """
    wanted = None if flags is None else set(flags)
    out: list[tuple[str, str]] = []
    covered: set[str] = set()
    for label, value_flag, unit_flag in HEADLINE_SETTINGS:
        group = {value_flag} if unit_flag is None else {value_flag, unit_flag}
        covered |= group
        if wanted is not None and not group & wanted:
            continue
        if value_flag not in settings:
            if wanted is not None:
                out.append((label, "(not recorded)"))
            continue
        value = str(settings[value_flag])
        if unit_flag is not None:
            unit = str(settings.get(unit_flag, ""))
            value = f"{value} {_READABLE_UNITS.get(unit.lower(), unit)}".strip()
        elif value_flag == "-fixed" and value == "":
            value = "none"
        elif "/" in value:
            value = Path(value).name
        out.append((label, value))
    if wanted is not None:
        for flag in sorted(wanted - covered):
            out.append((flag, str(settings[flag]) if flag in settings else "(not recorded)"))
    return out


__all__ = [
    "CHRONOLOGER_PTMS",
    "EXECUTION_ONLY_FLAGS",
    "FILE_FLAGS",
    "HEADLINE_SETTINGS",
    "PEPTIDE_CENTRIC",
    "PTM_STATES",
    "SEARCH_MODES",
    "SEARCH_TYPES",
    "SPECTRUM_CENTRIC",
    "SPECTRUM_CENTRIC_SETTINGS",
    "ResolvedSearch",
    "SearchModeError",
    "SearchParameters",
    "Tolerance",
    "readable_settings",
    "refuse_file_flags",
    "resolve_search_mode",
    "unit_token",
]
