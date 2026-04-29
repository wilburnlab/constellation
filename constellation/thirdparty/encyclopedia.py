"""EncyclopeDIA jar adapter + modseq translation helpers.

Two responsibilities, one module:

  1. ``ToolSpec`` registration so ``constellation doctor`` finds the
     EncyclopeDIA jar on disk.
  2. ``normalize_encyclopedia_modseq`` — translate EncyclopeDIA's
     ``[+N.NNN]``-style mass-delta notation into canonical ProForma 2.0,
     including N-terminal-vs-side-chain disambiguation via UNIMOD
     specificity lookup. EncyclopeDIA collapses ``[Mod]-X`` and
     ``X[Mod]`` into one ``X[Mod]`` form, but they are chemically
     distinct (e.g. K[Acetyl]VERPD vs [Acetyl]-KVERPD — lysine ε-amine
     vs N-terminal α-amine acetylation). The reader is the right
     boundary to fix this: every consumer downstream of
     ``PEPTIDE_TABLE.modified_sequence`` then sees an unambiguous
     ProForma 2.0 string.

Cartographer's pinned version lives in ``cartographer/pipeline/__init__.py``
as ``ENCYCLOPEDIA_VERSION``. Constellation uses the same pin until we have
reason to move; ``scripts/install-encyclopedia.sh`` hash-pins that release.
"""

from __future__ import annotations

import re
from pathlib import Path

from constellation.core.chem.modifications import UNIMOD, ModVocab
from constellation.thirdparty.registry import ToolSpec, register


ENCYCLOPEDIA_VERSION = "3.0.4"
_JAR_NAME = f"encyclopedia-{ENCYCLOPEDIA_VERSION}-executable.jar"


def _probe_version(path: Path) -> str | None:
    # Jar name encodes the version; cheaper than shelling out to java -jar.
    stem = path.name
    if stem.startswith("encyclopedia-") and stem.endswith("-executable.jar"):
        return stem[len("encyclopedia-") : -len("-executable.jar")]
    return None


register(
    ToolSpec(
        name="encyclopedia",
        env_var="CONSTELLATION_ENCYCLOPEDIA_HOME",
        artifact=_JAR_NAME,
        path_bin=None,  # jar — never found via $PATH
        install_script="scripts/install-encyclopedia.sh",
        version_probe=_probe_version,
    )
)


# ──────────────────────────────────────────────────────────────────────
# EncyclopeDIA modseq → ProForma 2.0
# ──────────────────────────────────────────────────────────────────────

# EncyclopeDIA modseq grammar (informal):
#   - Bare uppercase residues
#   - Mass-delta blocks ``[+N.NNN]`` or ``[-N.NNN]`` (signed)
#     immediately follow the residue they decorate
#   - No N-term-prefix-style mod blocks; everything is residue-attached
#
# Example inputs:
#   PEPTIDE                            → "PEPTIDE"
#   PEPC[+57.02146]TIDE                 → "PEPC[UNIMOD:4]TIDE" (Cam on C)
#   K[+42.01057]VERPD                   → "[UNIMOD:1]-KVERPD"  (N-term Acetyl)
#   PEPK[+42.01057]TIDE                 → "PEPK[UNIMOD:1]TIDE" (K side-chain Ac)
#   PEPS[+79.96633]TIDE                 → "PEPS[UNIMOD:21]TIDE" (Phospho on S)

_RE_MOD_BLOCK = re.compile(r"\[([+-]?\d+(?:\.\d+)?)\]")


def normalize_encyclopedia_modseq(
    modseq: str,
    *,
    vocab: ModVocab = UNIMOD,
    tolerance_da: float = 1e-3,
) -> str:
    """Translate EncyclopeDIA-style ``X[+N.NNN]`` → ProForma 2.0.

    Rules:

      1. Walk the modseq residue-by-residue. When an ``[+N.NNN]`` block
         follows a residue, look up the mass delta in ``vocab``.
      2. If the delta resolves to exactly one UNIMOD modification:
           a. If the residue is at index 0 AND the modification has an
              N-terminal specificity AND the modification CANNOT
              attach to the residue's side chain (or has no
              ``Anywhere`` specificity for that residue), render as
              ``[UNIMOD:N]-...`` (terminal placement).
           b. Otherwise render as ``X[UNIMOD:N]`` (residue placement).
      3. If the delta does not resolve to UNIMOD (no match within
         ``tolerance_da``), pass through as a ProForma mass-delta
         (``X[+N.NNN]``) — ProForma 2.0 §4.5.2 supports bare deltas.
      4. Multiple UNIMOD candidates: prefer the one whose specificity
         best matches the residue position. Ties leave the modseq as
         a ProForma mass-delta with no claim of identity.

    The function is pure and deterministic; the reader passes its
    output straight into ``PEPTIDE_TABLE.modified_sequence``.
    """
    if not modseq:
        return ""

    # Tokenize into (residue, mod_block_or_None) pairs.
    tokens: list[tuple[str, str | None]] = []
    i = 0
    while i < len(modseq):
        c = modseq[i]
        if c.isalpha() and c.isupper():
            residue = c
            i += 1
            mod_str: str | None = None
            if i < len(modseq) and modseq[i] == "[":
                m = _RE_MOD_BLOCK.match(modseq, i)
                if m is None:
                    raise ValueError(
                        f"malformed mod block at index {i} in {modseq!r}"
                    )
                mod_str = m.group(1)
                i = m.end()
            tokens.append((residue, mod_str))
        else:
            raise ValueError(
                f"unexpected character {c!r} at index {i} in {modseq!r}; "
                "EncyclopeDIA modseq must start each residue with an uppercase letter"
            )

    if not tokens:
        return ""

    # Render — first residue may move its mod to the N-terminus.
    n_term_prefix = ""
    parts: list[str] = []
    for idx, (residue, mod_str) in enumerate(tokens):
        if mod_str is None:
            parts.append(residue)
            continue
        delta = float(mod_str)
        matches = vocab.find_by_mass(delta, tolerance_da=tolerance_da)
        chosen, place_terminal = _choose_modification(matches, residue, idx)
        if chosen is None:
            # Ambiguous or no match → pass through as ProForma mass-delta.
            sign = "+" if delta >= 0 else ""
            parts.append(f"{residue}[{sign}{delta}]")
            continue
        if place_terminal:
            n_term_prefix = f"[{chosen.id}]-"
            parts.append(residue)
        else:
            parts.append(f"{residue}[{chosen.id}]")

    return n_term_prefix + "".join(parts)


def _choose_modification(matches, residue: str, position: int):
    """Pick a UNIMOD modification + decide N-term-vs-side-chain placement.

    Returns ``(modification, place_at_n_terminus)`` or ``(None, False)``
    when no unambiguous choice can be made.

    Strategy:

      1. Filter to candidates *compatible* with the (residue, position)
         pair: must either accept the residue at "Anywhere" or, when at
         position 0, carry an N-terminal specificity.
      2. Modifications with empty specificities are treated as
         compatible at any position (the chemistry lookup didn't
         declare constraints — common for older UNIMOD entries and
         custom mods).
      3. Among the compatible set, prefer terminal placement at
         position 0 when no side-chain placement is possible for that
         residue, otherwise prefer side-chain placement.
    """
    if not matches:
        return None, False

    compatible = [m for m in matches if _is_compatible(m, residue, position)]
    if not compatible:
        return None, False

    if len(compatible) > 1:
        # Multiple chemically valid candidates of identical mass —
        # defer to ProForma mass-delta to avoid silent misassignment.
        return None, False

    mod = compatible[0]

    if position == 0 and mod.has_n_term_specificity:
        # If the residue cannot host the modification on its side chain,
        # terminal placement is the only chemically valid rendering.
        if not mod.has_residue_specificity(residue):
            return mod, True
        # Both placements are chemically valid; default to side-chain
        # because EncyclopeDIA's literal rendering is X[Mod] and we
        # have no localization evidence to override that. If the user
        # *knows* it's terminal, downstream code can override.
        return mod, False

    return mod, False


def _is_compatible(mod, residue: str, position: int) -> bool:
    """True iff `mod` could chemically attach at (residue, position).

    A modification with empty specificities (after filtering hidden /
    speculative entries from the upstream XML) is treated as **not**
    compatible — many UNIMOD substitution / artefact entries
    (Ala->Gln, Phe->Tyr, isobaric add-Gly, etc.) collide on mass with
    canonical biological PTMs but carry no real-world placement
    evidence. Excluding them here is what makes mass-delta lookup
    unambiguous in the common case.
    """
    if not mod.specificities:
        return False
    if mod.has_residue_specificity(residue):
        return True
    if position == 0 and mod.has_n_term_specificity:
        return True
    return False


__all__ = [
    "ENCYCLOPEDIA_VERSION",
    "normalize_encyclopedia_modseq",
]
