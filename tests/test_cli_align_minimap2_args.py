"""Tests for the ``constellation transcriptome align`` CLI's minimap2
preset / override / extra-args plumbing.

These tests verify only the argument resolution path (preset + kwarg
overrides → minimap2 arg tuple); they do not invoke the actual
minimap2 binary. End-to-end execution is covered by the existing
``test_transcriptome_align_pr1.py`` suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from constellation.cli.__main__ import _build_transcriptome_parser
from constellation.sequencing.align.map import _GENOME_MODE_ARGS
from constellation.sequencing.align.presets import resolve_minimap2_args


def _build_parser() -> argparse.ArgumentParser:
    """Construct just enough of the constellation argparse tree to parse
    a ``transcriptome align`` invocation."""
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="subcommand")
    _build_transcriptome_parser(subs)
    return parser


# ── Parser-level: the new flags exist + parse cleanly ────────────────


def test_align_parser_accepts_organism_profile_flag():
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "align",
        "--demux-dir", "/tmp/d",
        "--output-dir", "/tmp/o",
        "--reference", "x",
        "--organism-profile", "compact_eukaryote",
    ])
    assert args.organism_profile == "compact_eukaryote"


def test_align_parser_accepts_max_intron_length_override():
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "align",
        "--demux-dir", "/tmp/d",
        "--output-dir", "/tmp/o",
        "--reference", "x",
        "--max-intron-length", "3000",
    ])
    assert args.max_intron_length == 3000


def test_align_parser_accepts_all_minimap2_overrides(tmp_path):
    bed = tmp_path / "junc.bed"
    bed.write_text("")
    parser = _build_parser()
    args = parser.parse_args([
        "transcriptome", "align",
        "--demux-dir", "/tmp/d",
        "--output-dir", "/tmp/o",
        "--reference", "x",
        "--organism-profile", "intermediate_eukaryote",
        "--max-intron-length", "20000",
        "--non-canonical-cost", "9",
        "--junc-bed", str(bed),
        "--junc-bonus", "12",
        "--minimap2-extra", "--splice-flank=no -k 14",
    ])
    assert args.organism_profile == "intermediate_eukaryote"
    assert args.max_intron_length == 20000
    assert args.non_canonical_cost == 9
    assert args.junc_bed == str(bed)
    assert args.junc_bonus == 12
    assert args.minimap2_extra == "--splice-flank=no -k 14"


def test_align_parser_organism_profile_rejects_unknown_choice():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "transcriptome", "align",
            "--demux-dir", "/tmp/d",
            "--output-dir", "/tmp/o",
            "--reference", "x",
            "--organism-profile", "bogus_organism",
        ])


# ── Resolver semantics: preset values reach minimap2 args ────────────


def test_compact_eukaryote_profile_resolves_to_5kb_intron_cap():
    args = resolve_minimap2_args(
        base_args=_GENOME_MODE_ARGS,
        profile="compact_eukaryote",
        max_intron_length=None,
        non_canonical_cost=None,
        junc_bed=None,
        junc_bonus=None,
        extra_args=(),
    )
    # The base flags ride through unchanged
    assert args[: len(_GENOME_MODE_ARGS)] == _GENOME_MODE_ARGS
    # The preset's -G 5000 appears once
    g_indices = [i for i, t in enumerate(args) if t == "-G"]
    assert len(g_indices) == 1
    assert args[g_indices[0] + 1] == "5000"
    # And -C 5 from the preset
    c_indices = [i for i, t in enumerate(args) if t == "-C"]
    assert len(c_indices) == 1
    assert args[c_indices[0] + 1] == "5"


def test_explicit_intron_length_override_with_profile_yields_single_g_entry():
    args = resolve_minimap2_args(
        base_args=_GENOME_MODE_ARGS,
        profile="compact_eukaryote",
        max_intron_length=2000,
        non_canonical_cost=None,
        junc_bed=None,
        junc_bonus=None,
        extra_args=(),
    )
    g_indices = [i for i, t in enumerate(args) if t == "-G"]
    assert len(g_indices) == 1
    assert args[g_indices[0] + 1] == "2000"


def test_minimap2_extra_string_splits_via_shlex_and_appends():
    import shlex
    extra = "--splice-flank=no -k 14"
    extra_tuple = tuple(shlex.split(extra))
    args = resolve_minimap2_args(
        base_args=_GENOME_MODE_ARGS,
        profile="compact_eukaryote",
        max_intron_length=None,
        non_canonical_cost=None,
        junc_bed=None,
        junc_bonus=None,
        extra_args=extra_tuple,
    )
    # Last 3 tokens should be the extra args (splice-flank uses = form,
    # so it's one token; -k and 14 are two more — 3 total).
    assert args[-3:] == ("--splice-flank=no", "-k", "14")


def test_extra_args_conflict_with_explicit_kwarg_raises():
    import shlex
    with pytest.raises(ValueError, match=r"-G"):
        resolve_minimap2_args(
            base_args=_GENOME_MODE_ARGS,
            profile="compact_eukaryote",
            max_intron_length=2000,
            non_canonical_cost=None,
            junc_bed=None,
            junc_bonus=None,
            extra_args=tuple(shlex.split("-G 999")),
        )


# ── Manifest field round-trip ────────────────────────────────────────


def test_align_manifest_round_trips_minimap2_resolved_args(tmp_path: Path):
    from constellation.sequencing.transcriptome.manifest import (
        read_manifest,
        write_align_manifest,
    )

    manifest_path = tmp_path / "manifest.json"
    resolved = list(
        resolve_minimap2_args(
            base_args=_GENOME_MODE_ARGS,
            profile="compact_eukaryote",
            max_intron_length=3000,
            non_canonical_cost=None,
            junc_bed=None,
            junc_bonus=None,
            extra_args=(),
        )
    )
    write_align_manifest(
        manifest_path,
        reference_handle="pichia@ensembl-58",
        reference_path="/tmp/ref",
        assembly_accession="GCF_000146045.2",
        demux_dir="/tmp/demux",
        input_files=[],
        parameters={},
        stages={},
        outputs={},
        samples=None,
        minimap2_resolved_args=resolved,
    )
    loaded = read_manifest(manifest_path)
    assert loaded.kind == "align"
    assert loaded.minimap2_resolved_args == resolved


def test_align_manifest_legacy_payload_loads_with_none_resolved_args(tmp_path: Path):
    """A v4 manifest written before the resolved-args field was added
    (i.e. the JSON lacks the key) loads cleanly with the field == None."""
    import json
    from constellation.sequencing.transcriptome.manifest import (
        MANIFEST_SCHEMA_VERSION,
        read_manifest,
    )

    manifest_path = tmp_path / "legacy.json"
    legacy_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "align",
        "reference_handle": "x@y-1",
        "reference_path": "/tmp/ref",
        "assembly_accession": None,
        "created_at": "2026-01-01T00:00:00Z",
        "demux_dir": "/tmp/demux",
        "input_files": [],
        "parameters": {},
        "stages": {},
        "outputs": {},
        "samples": None,
        # NOTE: no `minimap2_resolved_args` key
    }
    manifest_path.write_text(json.dumps(legacy_payload))
    loaded = read_manifest(manifest_path)
    assert loaded.minimap2_resolved_args is None
