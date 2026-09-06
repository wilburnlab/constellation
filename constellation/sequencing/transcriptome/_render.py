"""Back-compat shim — the shared render helpers moved to
:mod:`constellation.sequencing._render` so the genome-assembly stage can
share them. Existing transcriptome imports keep working.
"""

from __future__ import annotations

from constellation.sequencing._render import (
    ReportSection,
    _figure_ref,
    render_report,
    save_svg_figure,
)

__all__ = [
    "ReportSection",
    "render_report",
    "save_svg_figure",
    "_figure_ref",
]
