"""Cross-tier file-format adapters for the massspec subpackage.

Sibling to ``library/``, ``quant/``, ``search/``, ``peptide/``. Houses
adapters whose on-disk format crosses tier-of-abstraction boundaries —
e.g. EncyclopeDIA ``.dlib``/``.elib`` carry library + quant + search
data in the same SQLite container, so forcing the reader into any one
of the tier subpackages would put the seam in the wrong place.

Each adapter self-registers with the appropriate Reader/Writer
registries on import. ``__init__`` triggers those imports so that the
suffix-based dispatch in ``library.io.load_library`` /
``quant.io.load_quant`` / ``search.io.load_search`` resolves the right
backend for ``.dlib`` / ``.elib`` paths.

Format-level raw-instrument readers (``RawReader`` subclasses for a
single vendor format, e.g. Thermo ``.raw``) live in the sibling
``massspec.readers`` package, mirroring the ``sequencing.readers`` /
``sequencing.io`` split.
"""

from __future__ import annotations

from constellation.massspec.io import encyclopedia as encyclopedia  # noqa: F401
from constellation.massspec.io import maxquant as maxquant  # noqa: F401
from constellation.massspec.io import msp as msp  # noqa: F401

__all__ = ["encyclopedia", "maxquant", "msp"]
