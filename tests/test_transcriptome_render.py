"""Tests for ``constellation.sequencing.transcriptome._render``."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.transcriptome._render import (
    ReportSection,
    render_report,
    save_svg_figure,
)


# ── save_svg_figure ───────────────────────────────────────────────────


def test_save_svg_figure_writes_valid_svg(tmp_path: Path):
    """A small matplotlib chart saves as a non-empty SVG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [3, 1, 4])
    out = save_svg_figure(fig, tmp_path / "test.svg")

    assert out.is_file()
    content = out.read_text()
    assert content.startswith("<?xml") or "<svg" in content[:200]
    assert "</svg>" in content


def test_save_svg_figure_creates_parent_dir(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    target = tmp_path / "deeply" / "nested" / "figure.svg"
    save_svg_figure(fig, target)
    assert target.is_file()


def test_save_svg_figure_closes_by_default(tmp_path: Path):
    """The figure should be closed after save (default behaviour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2])
    save_svg_figure(fig, tmp_path / "f.svg")

    # plt.fignum_exists is the canonical "is this figure still alive" probe.
    assert not plt.fignum_exists(fig.number)


def test_save_svg_figure_close_false_keeps_figure_alive(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2])
    try:
        save_svg_figure(fig, tmp_path / "f.svg", close=False)
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)


# ── ReportSection ─────────────────────────────────────────────────────


def test_report_section_defaults_to_empty_figures_and_flags():
    section = ReportSection(title="X", body="hello")
    assert section.figures == []
    assert section.flags == []


# ── render_report ─────────────────────────────────────────────────────


def test_render_report_emits_basic_structure(tmp_path: Path):
    sections = [
        ReportSection(title="First section", body="Some descriptive text."),
        ReportSection(
            title="Second section",
            body="More body.",
            flags=["second-section warning"],
        ),
    ]
    out = render_report(
        title="Test Report",
        intro="An intro paragraph.",
        sections=sections,
        output_path=tmp_path / "report.md",
    )
    content = out.read_text()

    assert content.startswith("# Test Report")
    assert "_Generated " in content
    assert "An intro paragraph." in content
    # Flags summary because at least one section has a flag
    assert "## Flags" in content
    assert "- second-section warning" in content
    # h2 per section
    assert "## First section" in content
    assert "## Second section" in content
    # Inline flag block under the section that raised it
    assert "**Flags raised by this section:**" in content


def test_render_report_skips_flags_section_when_no_flags(tmp_path: Path):
    sections = [ReportSection(title="Clean", body="all good")]
    out = render_report(
        title="Clean Report",
        intro="",
        sections=sections,
        output_path=tmp_path / "r.md",
    )
    content = out.read_text()
    assert "## Flags" not in content


def test_render_report_emits_figure_refs(tmp_path: Path):
    fig_path = tmp_path / "figures" / "intron_length.svg"
    fig_path.parent.mkdir(parents=True)
    fig_path.write_text("<svg/>")

    sections = [
        ReportSection(
            title="Introns",
            body="Distribution.",
            figures=[fig_path],
        ),
    ]
    out = render_report(
        title="Report",
        intro="",
        sections=sections,
        output_path=tmp_path / "report.md",
    )
    content = out.read_text()
    # Relative path used in the markdown ref
    assert "![intron length](figures/intron_length.svg)" in content


def test_render_report_handles_figure_outside_report_dir(tmp_path: Path):
    """An absolute figure path outside the report's tree should still
    render — falling back to an absolute path in the ref."""
    fig_path = tmp_path / "external" / "f.svg"
    fig_path.parent.mkdir(parents=True)
    fig_path.write_text("<svg/>")
    report_dir = tmp_path / "report_dir"
    report_dir.mkdir()

    sections = [
        ReportSection(title="X", body="", figures=[fig_path]),
    ]
    out = render_report(
        title="X",
        intro="",
        sections=sections,
        output_path=report_dir / "report.md",
    )
    content = out.read_text()
    # When the figure isn't under the report dir, _figure_ref falls back
    # to an absolute path (sometimes a relative-with-../ path is more
    # natural, but absolute always works).
    assert str(fig_path.resolve()) in content


def test_render_report_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "new" / "dir" / "report.md"
    sections = [ReportSection(title="X", body="")]
    render_report(
        title="X",
        intro="",
        sections=sections,
        output_path=target,
    )
    assert target.is_file()
