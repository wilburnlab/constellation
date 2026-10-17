"""EncyclopeDIA ``.dlib`` / ``.elib`` reader — produces ``(Library, Quant?, Search?)``.

DLIB and ELIB share one SQLite container schema; the ``Quant`` and
``Search`` halves get built only when the corresponding tables /
columns are populated. The reader is one piece of code that returns
all three projections; ``adapters.py`` slices out whichever one a
caller asked for via the Library / Quant / Search Reader Protocols.

The work is done in two layers underneath: :mod:`._mirror` turns the
file's tables into Arrow as they are, and :mod:`._project` says what
they mean. This module is the stable entry point over them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.massspec.acquisitions import Acquisitions
from constellation.massspec.io.encyclopedia._project import project_library_file
from constellation.massspec.library.library import Library
from constellation.massspec.quant.quant import Quant
from constellation.massspec.search.search import Search


@dataclass(frozen=True, slots=True)
class EncyclopediaReadResult:
    """All three projections of one ``.dlib`` / ``.elib`` file.

    ``library`` is always populated. ``quant`` is populated when the
    file carries any sample-specific data (chromatogram blob in
    ``entries`` rows, populated ``peptidequants`` / ``fragmentquants``
    / ``retentiontimes`` tables). ``search`` is populated when
    ``peptidescores`` or ``proteinscores`` carry any rows.

    ``caveats`` says, in sentences with counts, where the projection is
    not the whole file: quantity or score rows that could not be tied
    to one precursor, precursors with no quantity row, or a file that
    holds no quantities at all. Empty when none applies, which is the
    normal case. Callers that write the projection to disk should show
    them and record them.
    """

    library: Library
    quant: Quant | None
    search: Search | None
    caveats: tuple[str, ...] = ()


def read_encyclopedia(
    path: Path | str,
    *,
    annotate_fragments: bool = True,
    fragment_tolerance_ppm: float = 20.0,
    enabled_modifications: ModVocab = UNIMOD,
    include_decoys: bool = False,
    acquisitions: Acquisitions | None = None,
) -> EncyclopediaReadResult:
    """Read an EncyclopeDIA ``.dlib`` / ``.elib`` file.

    Parameters
    ----------
    path
        SQLite file on disk.
    annotate_fragments
        Whether to build the fragment table: one row per stored peak,
        annotated as a b/y ion or an intact-precursor ion where it is
        one. When ``False``, ``Library.fragments`` is empty, and the
        read is several times faster.
    fragment_tolerance_ppm
        Tolerance for that annotation. Stored fragment m/z are library
        values, so 20 ppm is generous whatever the instrument.
    enabled_modifications
        UNIMOD vocabulary used for parsing ``PeptideModSeq``. Use a
        subset to constrain modification interpretation.
    include_decoys
        When ``False`` (default), rows where ``peptidetoprotein.isDecoy``
        is true are dropped from the protein/peptide tables.
        ``Search.precursor_scores`` keeps decoy rows either way, flagged.
    acquisitions
        An experiment's acquisitions to bind the file's runs to, by
        source name. Without it the file's own source files are numbered
        in sorted order.

    What each output column holds, and why, is in :mod:`._project`.
    """
    projection = project_library_file(
        path,
        annotate_fragments=annotate_fragments,
        fragment_tolerance_ppm=fragment_tolerance_ppm,
        vocab=enabled_modifications,
        include_decoys=include_decoys,
        acquisitions=acquisitions,
    )
    return EncyclopediaReadResult(
        library=projection.library,
        quant=projection.quant,
        search=projection.search,
        caveats=projection.caveats,
    )


__all__ = ["EncyclopediaReadResult", "read_encyclopedia"]
