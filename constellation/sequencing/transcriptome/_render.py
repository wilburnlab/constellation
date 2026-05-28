"""Shared figure + markdown rendering helpers for transcriptome diagnostics.

The per-stage diagnostic generators
(:mod:`constellation.sequencing.transcriptome.align.diagnostics`,
:mod:`constellation.sequencing.transcriptome.cluster.diagnostics`)
each emit a ``report.md`` file with embedded SVG figures. This module
provides the cross-cutting plumbing:

* :func:`save_svg_figure` — Agg-backed matplotlib → SVG writer with the
  closes-after-save discipline so callers don't leak figures.
* :class:`ReportSection` — the section dataclass each metric function
  returns (title, markdown body, list of figure paths, list of
  short warning flags).
* :func:`render_report` — assembles a single ``report.md`` from a
  list of sections + a top-of-document FLAGS summary so readers
  skimming the report see red flags first.

The matplotlib import is **lazy** (inside :func:`save_svg_figure`) so
unrelated CLI commands don't pay the cold-start cost. The Agg backend
is selected the first time the helper is invoked, before any pyplot
import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import matplotlib.figure


# ──────────────────────────────────────────────────────────────────────
# Figure save helper
# ──────────────────────────────────────────────────────────────────────


def save_svg_figure(
    fig: "matplotlib.figure.Figure",
    path: Path,
    *,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure to ``path`` as SVG.

    Parameters
    ----------
    fig
        The matplotlib Figure to write.
    path
        Output path. Parent directory is created if missing.
    close
        Whether to close ``fig`` after saving (default True). Set to
        False only when the caller wants to inspect the figure further
        — leaking figures across a long-running pipeline is the typical
        OOM source.
    """
    # Lazy import so this module's load cost is ~0 when matplotlib isn't
    # actually needed (e.g. when the user runs `constellation reference
    # list` and unrelated CLI verbs).
    import matplotlib

    # The Agg backend is non-interactive + thread-safe; switching is a
    # no-op when something else already set it.
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (force pyplot init)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="svg", bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────
# Report data model
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One section of a diagnostic report.

    ``figures`` are paths to SVGs already written to disk; the report
    assembler emits markdown ``![](relative/path)`` refs to each one
    after ``body``. ``flags`` is a list of short single-line warning
    strings — these get hoisted into a top-of-document summary in
    addition to appearing inline below the section's body, so a reader
    skimming the report sees red flags first.

    ``body`` is markdown source. Conventional structure: a 1–2 sentence
    summary, then a markdown table or stat list. Don't include the
    section title in the body — :func:`render_report` emits the title
    as an h2.
    """

    title: str
    body: str
    figures: list[Path] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Report assembler
# ──────────────────────────────────────────────────────────────────────


def _figure_ref(figure_path: Path, report_path: Path) -> str:
    """Markdown image ref using a path relative to the report file."""
    try:
        rel = figure_path.resolve().relative_to(report_path.resolve().parent)
    except ValueError:
        # Not under the report dir — emit absolute path. Less portable
        # but always correct.
        rel = figure_path.resolve()
    alt = figure_path.stem.replace("_", " ")
    return f"![{alt}]({rel})"


def render_report(
    *,
    title: str,
    intro: str,
    sections: list[ReportSection],
    output_path: Path,
) -> Path:
    """Assemble a markdown report from a list of sections.

    Layout::

        # <title>
        Generated <iso-timestamp>

        <intro>

        ## Flags  (only emitted if any section has flags)
        - <flag 1>
        - <flag 2>
          ...

        ## <section 1 title>
        <body>
        ![](figure_1.svg)
        ![](figure_2.svg)
        - flag inline (if section has flags, repeated under it)

        ## <section 2 title>
        ...

    Returns the path to the written ``report.md`` (same as
    ``output_path``). Parent dir created as needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_flags = [flag for section in sections for flag in section.flags]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated {now}_")
    lines.append("")
    if intro:
        lines.append(intro)
        lines.append("")

    if all_flags:
        lines.append("## Flags")
        lines.append("")
        lines.append(
            "_Quick-scan summary; each flag is also repeated under its "
            "section below._"
        )
        lines.append("")
        for flag in all_flags:
            lines.append(f"- {flag}")
        lines.append("")

    for section in sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.body:
            lines.append(section.body)
            lines.append("")
        for figure in section.figures:
            lines.append(_figure_ref(figure, output_path))
            lines.append("")
        if section.flags:
            lines.append("**Flags raised by this section:**")
            lines.append("")
            for flag in section.flags:
                lines.append(f"- {flag}")
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "ReportSection",
    "render_report",
    "save_svg_figure",
]
