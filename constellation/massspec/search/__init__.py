"""Search-engine results — PSMs, scores, FDR / q-values.

The detection-side sibling to ``massspec.library`` (theoretical) and
``massspec.quant`` (abundance / transmission). A search engine consumes
a ``Library`` and an acquisition's spectra, produces a ``Search``
record (peptide-spectrum matches with confidence scores). Conventional
proteomics keeps detection and quantification organisationally
distinct; Constellation's Counter-philosophy unifies them at a future
higher composition layer that has yet to land.

Status: **scaffold only.** Schema design is held until the first
concrete consumer arrives — either a wrapper around the bundled
EncyclopeDIA jar (``thirdparty.encyclopedia``) or the Counter port —
because PSM-table column choices vary materially across search
engines and over-designing now would calcify the wrong contract.
The container raises ``NotImplementedError`` on construction so
import-only smoke tests remain green.

Planned modules (this scaffold reserves the namespace):

    schemas       PSM_TABLE, plus rollups (peptide-level, protein-level
                  inference) once the first wrapper tells us what columns
                  are universal vs. engine-specific
    search        Search container (parallel to Library / Quant)
    encyclopedia  jar subprocess via thirdparty registry
    msfragger     TSV reader
    counter       in-process Counter scoring (longer term)
"""

from __future__ import annotations

from typing import Any

_DEFERRED = (
    "massspec.search is scaffold only — schema design is deferred until "
    "the first concrete search-engine wrapper (EncyclopeDIA via "
    "thirdparty.encyclopedia, or a Counter port) lands. The namespace is "
    "reserved here so import-only smoke tests stay green; constructing "
    "Search before the schemas are settled is intentionally blocked."
)


class Search:
    """Container placeholder — see module docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_DEFERRED)


__all__ = ["Search"]
