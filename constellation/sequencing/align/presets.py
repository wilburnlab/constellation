"""Organism-aware minimap2 splice-mode preset resolver.

Composes the base splice-mode flags (``-ax splice -uf --cs=long
--secondary=no``, hardcoded in :mod:`constellation.sequencing.align.map`)
with an optional preset profile, explicit flag overrides, and a
caller-supplied escape-hatch arg tuple, producing the final argument
list passed to ``minimap2``.

The preset data lives at
``constellation/data/sequencing/minimap2_splice_presets.json``,
regenerable via ``scripts/build-minimap2-splice-presets-json.py``.

Resolution order (later wins for the same flag):

    1. ``base_args``  — the splice-mode invariants
    2. preset flags  — when ``profile`` is non-None
    3. explicit kwargs (``max_intron_length``, ``non_canonical_cost``,
       ``junc_bed``, ``junc_bonus``)
    4. ``extra_args`` — appended raw; no override checking

The function refuses to silently override an explicit kwarg with a
later ``extra_args`` entry that targets the same flag — that's a
``ValueError`` with the offending flag name listed.

Why this lives at ``sequencing/align/`` (not ``sequencing/transcriptome/``):
minimap2 is the substrate for any read-to-reference alignment, not just
the transcriptome pipeline. Future ``map_assembly`` / ``map_dna_to_genome``
verbs ship their own preset bundles next to this one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────
# Preset data model
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Minimap2Preset:
    """One organism-class minimap2 preset.

    ``flags`` is an ordered tuple of (flag, value) pairs preserved from
    the JSON. Iteration order matches the JSON's array order so command
    lines are reproducible.
    """

    id: str
    name: str
    description: str
    applies_to_examples: tuple[str, ...]
    flags: tuple[tuple[str, str], ...]

    def as_arg_list(self) -> tuple[str, ...]:
        """Flatten ``flags`` into the alternating-arg form minimap2 expects."""
        out: list[str] = []
        for flag, value in self.flags:
            out.append(flag)
            out.append(value)
        return tuple(out)


_PRESETS_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "sequencing"
    / "minimap2_splice_presets.json"
)


@lru_cache(maxsize=1)
def load_presets() -> dict[str, Minimap2Preset]:
    """Load + parse the JSON preset bundle once per process.

    Returns
    -------
    dict[str, Minimap2Preset]
        Keyed on preset id (``compact_eukaryote``, ...).
    """
    if not _PRESETS_PATH.is_file():
        raise FileNotFoundError(
            f"minimap2 splice presets file missing: {_PRESETS_PATH}. "
            "Regenerate via `python scripts/build-minimap2-splice-presets-json.py`."
        )
    raw = json.loads(_PRESETS_PATH.read_text())
    schema_version = str(raw.get("schema_version", ""))
    if schema_version != "1":
        raise ValueError(
            f"unsupported minimap2 splice presets schema_version "
            f"{schema_version!r} at {_PRESETS_PATH} (this code supports v1)"
        )
    out: dict[str, Minimap2Preset] = {}
    for entry in raw.get("presets", []):
        pid = str(entry["id"])
        flags = tuple(
            (str(pair[0]), str(pair[1])) for pair in entry["minimap2_flags"]
        )
        out[pid] = Minimap2Preset(
            id=pid,
            name=str(entry["name"]),
            description=str(entry["description"]),
            applies_to_examples=tuple(
                str(x) for x in entry.get("applies_to_examples", ())
            ),
            flags=flags,
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Resolver
# ──────────────────────────────────────────────────────────────────────


# Flags managed by explicit kwargs. Used by the conflict-check to refuse
# silent overrides from ``extra_args``.
_KWARG_FLAGS: dict[str, str] = {
    "-G": "max_intron_length",
    "-C": "non_canonical_cost",
    "--junc-bed": "junc_bed",
    "--junc-bonus": "junc_bonus",
}


def _flag_iter(args: Sequence[str]):
    """Yield (flag, value | None) pairs from a minimap2 arg sequence.

    Recognises both ``-G 5000`` (two tokens) and ``--junc-bed=path``
    (one token, ``=``-separated) styles. Standalone flags (boolean
    switches like ``-a``) yield ``(flag, None)``. Tokens that aren't
    flags (don't start with ``-``) are skipped silently — they're
    typically positional args appended by the caller (e.g. the index
    path).
    """
    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("-"):
            i += 1
            continue
        if "=" in tok:
            flag, _, value = tok.partition("=")
            yield flag, value
            i += 1
            continue
        # Look ahead to decide if the next token is a value or another flag
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            yield tok, args[i + 1]
            i += 2
        else:
            yield tok, None
            i += 1


def resolve_minimap2_args(
    *,
    base_args: tuple[str, ...],
    profile: str | None,
    max_intron_length: int | None = None,
    non_canonical_cost: int | None = None,
    junc_bed: Path | None = None,
    junc_bonus: int | None = None,
    extra_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Compose final minimap2 arg tuple per the documented resolution order.

    Parameters
    ----------
    base_args
        Caller-controlled invariants. For the transcriptome pipeline this
        is :data:`constellation.sequencing.align.map._GENOME_MODE_ARGS`
        (``-ax splice -uf --cs=long --secondary=no``).
    profile
        Preset id (``compact_eukaryote``, ``intermediate_eukaryote``,
        ``animal``) or ``None``.
    max_intron_length, non_canonical_cost
        Override the preset's ``-G`` / ``-C`` values. ``None`` leaves the
        preset (or base) value in place.
    junc_bed
        Path to a BED file of annotated junctions; emits ``--junc-bed PATH``.
    junc_bonus
        Annotated-junction bonus; emits ``--junc-bonus N``. Only
        meaningful in combination with ``junc_bed``.
    extra_args
        Escape-hatch raw arg tuple, appended last. Conflicts with the
        explicit kwarg-managed flags above raise ``ValueError``.

    Returns
    -------
    tuple[str, ...]
        The full minimap2 argument tuple. Order: base_args → preset →
        explicit overrides → extra_args.
    """
    if profile is not None:
        presets = load_presets()
        if profile not in presets:
            available = ", ".join(sorted(presets.keys())) or "(none)"
            raise ValueError(
                f"unknown minimap2 splice preset {profile!r}; "
                f"available: {available}"
            )
        preset_flags = list(presets[profile].flags)
    else:
        preset_flags = []

    # Apply explicit kwarg overrides. We rebuild the preset list so that
    # an override replaces a preset flag rather than appending a duplicate.
    explicit_overrides: list[tuple[str, str]] = []
    if max_intron_length is not None:
        explicit_overrides.append(("-G", str(int(max_intron_length))))
    if non_canonical_cost is not None:
        explicit_overrides.append(("-C", str(int(non_canonical_cost))))
    if junc_bed is not None:
        explicit_overrides.append(("--junc-bed", str(Path(junc_bed))))
    if junc_bonus is not None:
        explicit_overrides.append(("--junc-bonus", str(int(junc_bonus))))

    override_keys = {flag for flag, _ in explicit_overrides}
    merged: list[tuple[str, str]] = [
        (flag, value) for flag, value in preset_flags if flag not in override_keys
    ]
    merged.extend(explicit_overrides)

    # Reject extra_args that target a kwarg-managed flag — these would
    # silently override the explicit value and surprise the user.
    conflicts: list[str] = []
    for flag, _value in _flag_iter(extra_args):
        if flag in _KWARG_FLAGS:
            conflicts.append(
                f"{flag} (use --{_KWARG_FLAGS[flag].replace('_', '-')} instead)"
            )
    if conflicts:
        raise ValueError(
            "minimap2 extra-args conflict with explicit flags: "
            + "; ".join(conflicts)
        )

    out: list[str] = list(base_args)
    for flag, value in merged:
        out.append(flag)
        out.append(value)
    out.extend(extra_args)
    return tuple(out)


__all__ = [
    "Minimap2Preset",
    "load_presets",
    "resolve_minimap2_args",
]
