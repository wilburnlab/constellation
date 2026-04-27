"""Raw-format reader contract and registry.

Every vendor / instrument format Constellation ingests is a subclass of
``RawReader`` registered against one or more file suffixes. Domain
modules (``chromatography``, ``electrophoresis``, ``massspec``, ...)
import their reader subclass at package init, which triggers the
``@register_reader`` decorator and adds an entry to ``READER_REGISTRY``.

Resolution cascade for ``find_reader(path, modality=None)``:

    1. exact-suffix match against the registry (case-insensitive).
    2. if multiple readers claim that suffix, ``modality`` disambiguates.
    3. if still ambiguous (or no suffix match), raise
       ``ReaderNotFoundError`` with the registry contents enumerated.

Readers consume either a raw path or a ``Bundle`` (when the on-disk
representation is a directory or zip of related files — e.g. Agilent
``.dx`` OPC archives or Fragment-Analyzer ``.raw`` plus its companion
``*.txt`` / ``*.ANAI`` / ``*.current`` siblings). The ``ReadResult``
they emit is an attribute bag of ``pa.Table`` references (one
``primary``, plus any number of named ``companions``) and a free-form
``run_metadata`` dict that ends up stamped onto schema metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa  # noqa: F401  (referenced in type hints only)

    from constellation.core.io.bundle import Bundle


class ReaderNotFoundError(LookupError):
    """No reader is registered for the given suffix / modality pair."""


@dataclass
class ReadResult:
    """The output of ``RawReader.read()``.

    - ``primary`` — the most natural single-table view of the run
      (e.g. the default chromatogram channel for HPLC, the stacked
      capillary traces for CE, the peak list for MS). Always present.
    - ``companions`` — additional tables produced by the same read
      (DAD spectra, instrument telemetry, current/voltage logs, sample
      maps, ...). Keyed by short string names domain modules document.
    - ``run_metadata`` — flat dict of run-level provenance
      (``sample_name``, ``operator``, ``run_datetime``, instrument IDs,
      method paths). Producers stamp this into Arrow schema metadata
      via ``pack_metadata`` from ``core.io.schemas``.
    """

    primary: "pa.Table"
    companions: dict[str, "pa.Table"] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)


class RawReader(ABC):
    """Abstract reader for a single vendor / instrument format.

    Subclass contract:

    - declare ``suffixes`` (e.g. ``(".dx",)`` or ``(".raw",)``) — case is
      normalised to lower before matching;
    - declare ``modality`` as a short string tag
      (``"hplc-dad"``, ``"ce"``, ``"ms"``, ``"nanopore"``, ...);
    - implement ``read(source) -> ReadResult`` where ``source`` is a
      ``Path`` (single-file format) or a ``Bundle`` (multi-file).

    Register the subclass at package import time using the
    ``@register_reader`` decorator below.
    """

    suffixes: ClassVar[tuple[str, ...]] = ()
    modality: ClassVar[str | None] = None

    @abstractmethod
    def read(self, source: "Path | Bundle") -> ReadResult:
        """Decode the run at ``source``."""


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

# suffix (lowercased, including leading dot) → list of reader classes.
# We keep a list rather than a single entry so the same suffix can be
# claimed by readers in different domains (e.g. ``.raw`` is used by
# Fragment Analyzer CE and several MS vendors).
_REGISTRY: dict[str, list[type[RawReader]]] = {}


def register_reader(cls: type[RawReader]) -> type[RawReader]:
    """Class decorator: register ``cls`` against each declared suffix.

    The class must declare at least one suffix. ``modality`` is
    optional but strongly encouraged — without it, conflict resolution
    in ``find_reader`` falls back to "first registered wins".
    """
    if not cls.suffixes:
        raise ValueError(
            f"{cls.__name__}: RawReader subclass must declare at least one suffix "
            f"(e.g. suffixes = ('.dx',))"
        )
    for suffix in cls.suffixes:
        key = suffix.lower()
        if not key.startswith("."):
            raise ValueError(
                f"{cls.__name__}: suffix {suffix!r} must start with a dot"
            )
        _REGISTRY.setdefault(key, []).append(cls)
    return cls


def registered_readers() -> dict[str, list[type[RawReader]]]:
    """Snapshot of the suffix → readers map (for the doctor CLI / tests)."""
    return {k: list(v) for k, v in _REGISTRY.items()}


def find_reader(
    source: "Path | str | Bundle",
    modality: str | None = None,
) -> RawReader:
    """Resolve ``source`` to an instantiated reader.

    ``source`` may be:

    - a ``Path`` or ``str`` path — suffix dispatch on the basename;
    - a ``Bundle`` — suffix dispatch on the bundle's primary path.

    ``modality`` is consulted only when more than one reader claims the
    suffix; if exactly one reader is registered, ``modality`` (if
    given) is checked for consistency and the call raises if it
    disagrees.
    """
    suffix = _suffix_of(source)
    if suffix not in _REGISTRY:
        raise ReaderNotFoundError(
            f"no reader registered for suffix {suffix!r}; "
            f"registered suffixes: {sorted(_REGISTRY)}"
        )
    candidates = _REGISTRY[suffix]
    if modality is not None:
        filtered = [c for c in candidates if c.modality == modality]
        if not filtered:
            raise ReaderNotFoundError(
                f"no reader for suffix {suffix!r} with modality {modality!r}; "
                f"available: {[c.modality for c in candidates]}"
            )
        candidates = filtered
    if len(candidates) > 1:
        raise ReaderNotFoundError(
            f"suffix {suffix!r} is claimed by multiple readers "
            f"({[c.__name__ for c in candidates]}); pass modality= to disambiguate"
        )
    return candidates[0]()


def _suffix_of(source: "Path | str | Bundle") -> str:
    # Local import to keep this module a leaf — bundle.py imports nothing
    # but stdlib + Path, so this cycle is structural-only.
    from constellation.core.io.bundle import Bundle

    if isinstance(source, Bundle):
        return Path(source.path).suffix.lower()
    return Path(source).suffix.lower()
