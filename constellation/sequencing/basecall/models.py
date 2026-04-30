"""``DoradoModel`` — typed abstraction over Dorado model identifiers.

Dorado names models with the structure
``<chemistry>_<family>@<version>`` and the modified-base modes ride
as a separate ``--modified-bases`` argument. The lab shorthand
``"sup@v5.0+5mC,5hmC"`` parses to a model with chemistry inferred
from the surrounding context (DNA R10.4.1 by default; RNA004 for
direct-RNA experiments).

This module ships with a fully-implemented parser/formatter — small
enough that stubbing it would be silly when downstream code references
it.

Conventions:
    family       'sup' | 'hac' | 'fast'  (super-accurate / high-accuracy / fast)
    version      'v5.0.0' or 'v5.0' — patch component optional
    chemistry    'dna_r10.4.1_e8.2'  (genomic DNA R10.4.1 chemistry)
                 'rna004_130bps'    (direct-RNA RNA004 chemistry)
                 — others as ONT releases new pores
    mods         tuple of canonical ONT mod tokens: '5mC', '5hmC',
                 '6mA', 'm6A_DRACH', 'pseU', '5mC_5hmC' (combined),
                 etc. Matched against Dorado's modified-base catalog
                 at runtime; this layer doesn't validate the strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


_FAMILIES = frozenset({"sup", "hac", "fast"})

# Default chemistry by family — ``DoradoModel.parse`` uses this when the
# shorthand omits the chemistry prefix. Override by passing chemistry=
# explicitly to ``DoradoModel(...)``.
DEFAULT_DNA_CHEMISTRY = "dna_r10.4.1_e8.2"
DEFAULT_RNA_CHEMISTRY = "rna004_130bps"


# ``sup@v5.0.0+5mC,5hmC`` — the lab shorthand. Capture groups:
#   (chemistry?)(family)@(version)(+mods?)
_SHORTHAND_RE = re.compile(
    r"^(?:(?P<chemistry>[a-z][a-z0-9_.]+?)_)?"
    r"(?P<family>sup|hac|fast)"
    r"@(?P<version>v[0-9]+(?:\.[0-9]+){0,2})"
    r"(?:\+(?P<mods>[A-Za-z0-9_,]+))?$"
)


@dataclass(frozen=True, slots=True)
class DoradoModel:
    """Typed representation of a Dorado model identifier.

    ``model_name`` renders the canonical Dorado-on-disk name without
    the modifications (those ride separately on the ``--modified-bases``
    flag).
    """

    family: Literal["sup", "hac", "fast"]
    version: str  # 'v5.0.0' or 'v5.0' — kept verbatim
    chemistry: str = DEFAULT_DNA_CHEMISTRY
    mods: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(
                f"family must be one of {sorted(_FAMILIES)}, got {self.family!r}"
            )
        if not self.version.startswith("v"):
            raise ValueError(
                f"version must start with 'v' (e.g. 'v5.0.0'), got {self.version!r}"
            )

    @classmethod
    def parse(cls, shorthand: str) -> "DoradoModel":
        """Parse the lab shorthand ``'sup@v5.0+5mC,5hmC'`` (with optional
        chemistry prefix ``'rna004_130bps_sup@v5.0'``) into a
        :class:`DoradoModel`.

        Examples::

            >>> DoradoModel.parse("sup@v5.0.0")
            DoradoModel(family='sup', version='v5.0.0',
                        chemistry='dna_r10.4.1_e8.2', mods=())
            >>> DoradoModel.parse("hac@v4.3+5mC,5hmC")
            DoradoModel(family='hac', version='v4.3',
                        chemistry='dna_r10.4.1_e8.2',
                        mods=('5mC', '5hmC'))
            >>> DoradoModel.parse("rna004_130bps_sup@v5.0.0+m6A_DRACH")
            DoradoModel(family='sup', version='v5.0.0',
                        chemistry='rna004_130bps', mods=('m6A_DRACH',))
        """
        m = _SHORTHAND_RE.match(shorthand.strip())
        if m is None:
            raise ValueError(
                f"can't parse Dorado-model shorthand {shorthand!r}; "
                "expected '[<chemistry>_]<family>@<version>[+<mod>,<mod>,...]', "
                "e.g. 'sup@v5.0.0' or 'hac@v4.3+5mC,5hmC' or "
                "'rna004_130bps_sup@v5.0.0+m6A_DRACH'"
            )
        chemistry = m.group("chemistry") or DEFAULT_DNA_CHEMISTRY
        family = m.group("family")
        version = m.group("version")
        mods_str = m.group("mods")
        mods = tuple(s for s in mods_str.split(",") if s) if mods_str else ()
        return cls(
            family=family,
            version=version,
            chemistry=chemistry,
            mods=mods,
        )

    def model_name(self) -> str:
        """Canonical Dorado-on-disk model directory name (no mods).

        Example: ``'dna_r10.4.1_e8.2_sup@v5.0.0'`` for
        ``DoradoModel(family='sup', version='v5.0.0',
        chemistry='dna_r10.4.1_e8.2')``.
        """
        return f"{self.chemistry}_{self.family}@{self.version}"

    def modified_bases_arg(self) -> str | None:
        """Return the value to pass to ``dorado basecaller
        --modified-bases``, or None if no mods set."""
        if not self.mods:
            return None
        return ",".join(self.mods)


__all__ = [
    "DoradoModel",
    "DEFAULT_DNA_CHEMISTRY",
    "DEFAULT_RNA_CHEMISTRY",
]
