"""Regression test pinning that the orchestrator and the standalone
``massspec predict-library`` CLI both return the SAME default for every
PTM, sourced from a single shared map.

Background: before this test (and the
``massspec.search.encyclopedia.ptm_defaults`` module it pins), the two
CLI surfaces each maintained their own defaults dictionaries and
silently disagreed on ``ProteinNTermAcetyl`` / ``PyroGluQ`` / etc. —
users hitting the orchestrator path got an under-mod-ed library and
saw lower peptide IDs than the same FASTA searched via the standalone
CLI. This test makes the next divergence loud rather than silent.
"""

from __future__ import annotations

import argparse
import re

import pytest

from constellation.cli.transcriptome_to_proteome import (
    build_parser as build_orchestrator_parser,
)
from constellation.massspec.cli import build_parser as build_massspec_parser
from constellation.massspec.search.encyclopedia.ptm_defaults import (
    PTM_DEFAULTS,
    PTM_NAMES,
    default_for,
)


def _orchestrator_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(prog="constellation")
    subs = root.add_subparsers(dest="subcommand", required=True)
    build_orchestrator_parser(subs)
    return root.parse_args([
        "transcriptome-to-proteome",
        "--demux-dir", "/tmp/x",
        "--reference", "homo_sapiens",
        "--gpf", "/tmp/x",
        "--injections", "/tmp/x",
        "--output-dir", "/tmp/x",
    ])


def _standalone_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(prog="constellation")
    subs = root.add_subparsers(dest="subcommand", required=True)
    build_massspec_parser(subs)
    return root.parse_args([
        "massspec", "predict-library",
        "--fasta", "/tmp/x",
        "--output-dlib", "/tmp/x.dlib",
        "--output-dir", "/tmp",
    ])


def _camel_to_attr(name: str) -> str:
    """``ProteinNTermAcetyl`` → ``ptm_protein_n_term_acetyl``.

    Mirrors the kebab-case derivation used by the CLI to construct the
    ``--ptm-<name>`` flag, then prefixes ``ptm_`` and swaps ``-``→``_``
    so it matches the argparse ``dest`` attribute on the namespace."""
    s = re.sub(r"(?<!^)(?=[A-Z][a-z])", "-", name)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", s)
    return "ptm_" + s.lower().replace("-", "_")


@pytest.mark.parametrize("ptm_name", PTM_NAMES)
def test_orchestrator_and_standalone_share_one_default(ptm_name: str) -> None:
    """For every known PTM, both CLI surfaces return the value from
    ``ptm_defaults.default_for``."""
    expected = default_for(ptm_name)
    attr = _camel_to_attr(ptm_name)
    orch = getattr(_orchestrator_args(), attr)
    standalone = getattr(_standalone_args(), attr)
    assert orch == expected, (
        f"orchestrator default for {ptm_name} is {orch!r}, "
        f"expected {expected!r} (from ptm_defaults)"
    )
    assert standalone == expected, (
        f"standalone predict-library default for {ptm_name} is "
        f"{standalone!r}, expected {expected!r} (from ptm_defaults)"
    )


def test_ptm_defaults_set_matches_lab_convention() -> None:
    """The shared default map encodes the lab's standard variable-mod
    set (Carb-C fix + N-term-Acetyl/PyroGluQ/M-ox variable + everything
    else off). Pinned so a future tweak to the map shows up in code
    review."""
    assert PTM_DEFAULTS == {
        "Carbamidomethyl":    "fix",
        "ProteinNTermAcetyl": "var",
        "PyroGluQ":           "var",
        "Oxidation":          "var",
    }


def test_default_for_unknown_ptm_raises() -> None:
    with pytest.raises(KeyError):
        default_for("NotAPTM")
