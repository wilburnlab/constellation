"""Tier A tests for :func:`constellation.massspec.search.encyclopedia.
library_export.build_library_export_args`.

Pure-function argv composition across the supported flag combinations.
Mirrors the test style in ``test_predict_library_args.py`` /
``test_process_dia_args.py`` — no jar invocation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia.library_export import (
    build_library_export_args,
)


@pytest.fixture
def base_args():
    return {
        "search_dir": Path("/path/to/searches"),
        "library": Path("/path/to/library.dlib"),
        "output_elib": Path("/path/to/quant_report.elib"),
    }


def test_default_flag_layout(base_args) -> None:
    args = build_library_export_args(**base_args)
    # First arg is always `-libexport`.
    assert args[0] == "-libexport"
    assert "-i" in args and args[args.index("-i") + 1] == "/path/to/searches"
    assert "-l" in args and args[args.index("-l") + 1] == "/path/to/library.dlib"
    assert (
        "-o" in args
        and args[args.index("-o") + 1] == "/path/to/quant_report.elib"
    )
    # Default align=True → '-a true'
    assert "-a" in args and args[args.index("-a") + 1] == "true"
    # No -blib, no -f
    assert "-blib" not in args
    assert "-f" not in args


def test_no_align(base_args) -> None:
    args = build_library_export_args(**base_args, align=False)
    assert args[args.index("-a") + 1] == "false"


def test_write_blib(base_args) -> None:
    args = build_library_export_args(**base_args, write_blib=True)
    assert "-blib" in args


def test_with_fasta_for_pecan_mode(base_args) -> None:
    args = build_library_export_args(
        **base_args, fasta=Path("/path/to/reference.fasta")
    )
    assert "-f" in args and args[args.index("-f") + 1] == "/path/to/reference.fasta"


def test_extra_args_appended(base_args) -> None:
    """`extra_args` rides at the end so users can reach unwrapped
    flags via --encyclopedia-arg without re-ordering the canonical
    flags upstream."""
    args = build_library_export_args(
        **base_args, extra_args=["-percolatorThreshold", "0.01"]
    )
    assert args[-2:] == ["-percolatorThreshold", "0.01"]


def test_all_flags_combined(base_args) -> None:
    args = build_library_export_args(
        **base_args,
        align=False,
        write_blib=True,
        fasta=Path("/ref.fa"),
        extra_args=["-customFlag", "value"],
    )
    assert args[0] == "-libexport"
    assert args[args.index("-a") + 1] == "false"
    assert "-blib" in args
    assert args[args.index("-f") + 1] == "/ref.fa"
    assert args[-2:] == ["-customFlag", "value"]
