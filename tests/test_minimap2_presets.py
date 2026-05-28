"""Tests for ``constellation.sequencing.align.presets``."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.align.presets import (
    Minimap2Preset,
    load_presets,
    resolve_minimap2_args,
)


# Base args from constellation.sequencing.align.map._GENOME_MODE_ARGS,
# duplicated here to keep this test independent of that module's internals.
_BASE = ("-ax", "splice", "-uf", "--cs=long", "--secondary=no")


# ── load_presets ──────────────────────────────────────────────────────


def test_load_presets_returns_all_three_v1_presets():
    presets = load_presets()
    assert set(presets.keys()) == {
        "compact_eukaryote",
        "intermediate_eukaryote",
        "animal",
    }
    for p in presets.values():
        assert isinstance(p, Minimap2Preset)
        assert p.id
        assert p.name
        assert p.description
        assert len(p.flags) >= 1


def test_compact_eukaryote_has_5kb_intron_cap():
    presets = load_presets()
    flag_dict = dict(presets["compact_eukaryote"].flags)
    assert flag_dict["-G"] == "5000"
    assert flag_dict["-C"] == "5"


def test_animal_preset_keeps_stock_200k_cap():
    presets = load_presets()
    flag_dict = dict(presets["animal"].flags)
    assert flag_dict["-G"] == "200000"


def test_preset_as_arg_list_flattens_in_order():
    presets = load_presets()
    args = presets["compact_eukaryote"].as_arg_list()
    # Flags appear in JSON order: -G first, then -C
    assert args[:2] == ("-G", "5000")
    assert args[2:4] == ("-C", "5")


# ── resolve_minimap2_args: base + preset composition ──────────────────


def test_resolve_no_profile_just_returns_base_args():
    args = resolve_minimap2_args(base_args=_BASE, profile=None)
    assert args == _BASE


def test_resolve_with_profile_appends_preset_flags_after_base():
    args = resolve_minimap2_args(base_args=_BASE, profile="compact_eukaryote")
    assert args[: len(_BASE)] == _BASE
    assert "-G" in args
    g_idx = args.index("-G")
    assert args[g_idx + 1] == "5000"


def test_resolve_unknown_profile_raises_with_available_list():
    with pytest.raises(ValueError, match="unknown minimap2 splice preset 'bogus'"):
        resolve_minimap2_args(base_args=_BASE, profile="bogus")


# ── Explicit override semantics ───────────────────────────────────────


def test_explicit_max_intron_length_overrides_preset_value():
    args = resolve_minimap2_args(
        base_args=_BASE,
        profile="compact_eukaryote",  # ships -G 5000
        max_intron_length=3000,
    )
    # Only one -G entry should remain, with the user's value
    g_indices = [i for i, t in enumerate(args) if t == "-G"]
    assert len(g_indices) == 1, f"expected single -G entry, got {args}"
    assert args[g_indices[0] + 1] == "3000"


def test_explicit_overrides_apply_without_preset():
    args = resolve_minimap2_args(
        base_args=_BASE,
        profile=None,
        max_intron_length=2000,
        non_canonical_cost=7,
    )
    assert args[: len(_BASE)] == _BASE
    # No preset; just the overrides
    assert "-G" in args
    assert args[args.index("-G") + 1] == "2000"
    assert "-C" in args
    assert args[args.index("-C") + 1] == "7"


def test_junc_bed_and_bonus_emit_when_supplied(tmp_path: Path):
    bed = tmp_path / "junctions.bed"
    bed.write_text("")
    args = resolve_minimap2_args(
        base_args=_BASE,
        profile=None,
        junc_bed=bed,
        junc_bonus=9,
    )
    assert "--junc-bed" in args
    assert args[args.index("--junc-bed") + 1] == str(bed)
    assert "--junc-bonus" in args
    assert args[args.index("--junc-bonus") + 1] == "9"


# ── extra_args + conflict detection ───────────────────────────────────


def test_extra_args_appended_last():
    args = resolve_minimap2_args(
        base_args=_BASE,
        profile="compact_eukaryote",
        extra_args=("-k", "14"),
    )
    assert args[-2:] == ("-k", "14")


def test_extra_args_conflicting_with_kwarg_flag_raises():
    with pytest.raises(ValueError, match=r"-G \(use --max-intron-length"):
        resolve_minimap2_args(
            base_args=_BASE,
            profile=None,
            max_intron_length=5000,
            extra_args=("-G", "1000"),
        )


def test_extra_args_conflict_detected_even_without_explicit_kwarg():
    # The conflict check fires whenever extra_args targets a kwarg-managed
    # flag, even if the user didn't pass the kwarg — because the kwarg
    # is the documented interface for that flag.
    with pytest.raises(ValueError, match=r"--junc-bed"):
        resolve_minimap2_args(
            base_args=_BASE,
            profile=None,
            extra_args=("--junc-bed", "junctions.bed"),
        )


def test_extra_args_equals_form_detected():
    with pytest.raises(ValueError, match=r"-G"):
        resolve_minimap2_args(
            base_args=_BASE,
            profile=None,
            max_intron_length=5000,
            extra_args=("-G=1000",),
        )


def test_extra_args_unknown_flag_passes_through():
    args = resolve_minimap2_args(
        base_args=_BASE,
        profile=None,
        extra_args=("--splice-flank", "no"),
    )
    assert "--splice-flank" in args
    assert args[args.index("--splice-flank") + 1] == "no"
