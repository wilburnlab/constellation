"""CLI parsers and handlers for ``constellation massspec <subcommand>``.

Mounted on the root ``constellation`` parser by
``constellation/cli/__main__.py``. One module per verb or verb group,
each holding the verb's parser and its handler:

    convert            Thermo .raw -> peak / scan-metadata parquet bundle
    search             EncyclopeDIA library search of one acquisition
    predict_library    FASTA -> predicted library (EncyclopeDIA or Koina)
    process_dia        jar-side conversion and merging (legacy)
    library_export     EncyclopeDIA's quantitative report
    novel_peptides     classify-novel-peptides
    collision_filter   collision-filter
    encyclopedia       export-spectra, import-spectra, compare-spectra,
                       compare-results
    chromatogram       build-index, extract
    counter            calibrate, estimate
    _common            argument groups several verbs share (output
                       directory, the three mass tolerances, JVM heap,
                       the pass-through escape hatch) and the note that
                       says what an ingest left out

Until 2026-10-11 this was a single 4,000-line module; the split moved
every function unchanged. Some private names are imported from this
package by code outside it (the transcriptome-to-proteome command reuses
the tolerance arguments, and tests call a few handlers directly); they
are re-exported below so those imports keep working. New code should
import a name from the module that defines it.
"""

from __future__ import annotations

import argparse

from constellation.massspec.cli.chromatogram import _build_chromatogram_parser
from constellation.massspec.cli.collision_filter import _build_collision_filter_parser
from constellation.massspec.cli.convert import _build_convert_parser
from constellation.massspec.cli.counter import _build_counter_parser
from constellation.massspec.cli.encyclopedia import _build_encyclopedia_parser
from constellation.massspec.cli.library_export import _build_library_export_parser
from constellation.massspec.cli.novel_peptides import (
    _build_classify_novel_peptides_parser,
)
from constellation.massspec.cli.predict_library import _build_predict_library_parser
from constellation.massspec.cli.process_dia import _build_process_dia_parser
from constellation.massspec.cli.search import _build_search_parser

# Re-exported for importers outside this package (see the docstring).
from constellation.massspec.cli._common import (
    _add_tolerance_args,
    _camel_to_kebab,
    _report_ingest_caveats,
    _resolve_tolerance_args,
)
from constellation.massspec.cli.convert import _cmd_massspec_convert, _collect_raw_files
from constellation.massspec.cli.counter import _counter_cofit_units, _resolve_collide_ppm
from constellation.massspec.cli.predict_library import (
    _cmd_massspec_predict_library,
    _parse_collision_energies,
)
from constellation.massspec.cli.search import _write_manifest_for_search
from constellation.massspec.search.encyclopedia.ptm_defaults import (
    PTM_NAMES as _PTM_NAMES,
    default_for as _ptm_default_for,
)


def build_parser(subs: argparse._SubParsersAction) -> None:
    """Mount the ``massspec`` subtree on the parent ``constellation`` parser."""
    p_ms = subs.add_parser(
        "massspec",
        help=(
            "Mass-spectrometry workflows (EncyclopeDIA library search, "
            "FASTA → DLIB prediction, gas-phase fraction merging, quant "
            "reports)"
        ),
    )
    ms_subs = p_ms.add_subparsers(dest="ms_subcommand", required=True)

    _build_convert_parser(ms_subs)
    _build_search_parser(ms_subs)
    _build_predict_library_parser(ms_subs)
    _build_process_dia_parser(ms_subs)
    _build_library_export_parser(ms_subs)
    _build_classify_novel_peptides_parser(ms_subs)
    _build_collision_filter_parser(ms_subs)
    _build_encyclopedia_parser(ms_subs)
    _build_chromatogram_parser(ms_subs)
    _build_counter_parser(ms_subs)


__all__ = ["build_parser"]
