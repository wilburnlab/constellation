"""Shared figure + markdown rendering helpers for sequencing diagnostics.

Every per-stage diagnostic generator (transcriptome align / cluster,
genome assembly) emits a ``report.md`` with embedded SVG figures. This
module provides the cross-cutting plumbing, promoted out of
``transcriptome/`` so the assembly stage shares one implementation:

* :func:`save_svg_figure` — Agg-backed matplotlib → SVG writer with the
  closes-after-save discipline so callers don't leak figures.
* :class:`ReportSection` — the section dataclass each metric function
  returns (title, markdown body, list of figure paths, list of short
  warning flags).
* :func:`render_report` — assembles a single ``report.md`` from a list of
  sections + a top-of-document FLAGS summary so readers skimming the
  report see red flags first.

The matplotlib import is **lazy** (inside :func:`save_svg_figure`) so
unrelated CLI commands don't pay the cold-start cost.
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

    Lazy matplotlib import + Agg backend; closes ``fig`` after saving
    unless ``close=False`` (leaking figures across a long-running
    pipeline is the typical OOM source). Parent dir created if missing.
    """
    import matplotlib

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
    assembler emits markdown ``![](relative/path)`` refs after ``body``.
    ``flags`` are short single-line warnings, hoisted into a
    top-of-document summary so a reader skimming sees red flags first.
    Don't include the section title in ``body`` — :func:`render_report`
    emits it as an h2.
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

    Emits an h1 title + UTC timestamp + intro, a top-of-document ``##
    Flags`` quick-scan (only when any section has flags), then each
    section as an h2 with its body, figures, and inline flags. Returns
    ``output_path``.
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
