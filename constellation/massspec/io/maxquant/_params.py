"""MaxQuant ``parameters.txt`` parsing — provenance + fixed-mod specs.

``parameters.txt`` is a two-column TSV (header ``Parameter\\tValue``) of
the search settings. We keep the whole map as run provenance and parse
the ``Fixed modifications`` value into ``(unimod_name, site)`` pairs that
drive fixed-mod reconstruction in :mod:`._modseq` (fixed mods are not
written into MaxQuant's modseq strings).
"""

from __future__ import annotations

import re
from pathlib import Path

# "Carbamidomethyl (C)" / "Acetyl (Protein N-term)" → name + parenthesised site.
_MOD_SPEC_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<site>[^)]*)\)\s*$")


def parse_parameters_txt(path: Path) -> dict[str, str]:
    """Parse ``parameters.txt`` into a flat ``key -> value`` dict.

    Tab-split; the leading ``Parameter\\tValue`` header row is skipped.
    MaxQuant repeats a handful of keys (``Decoy mode``, ``Special AAs``,
    ``Include contaminants``) with identical values — duplicates keep the
    last occurrence. A missing file returns ``{}`` (the caller decides
    whether that is fatal; the reader treats it as "no fixed-mod
    reconstruction" and records a warning).
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line:
                continue
            key, _, value = line.partition("\t")
            key = key.strip()
            if lineno == 0 and key == "Parameter":
                continue
            out[key] = value.strip()
    return out


def parse_fixed_modifications(value: str | None) -> list[tuple[str, str]]:
    """``"Carbamidomethyl (C);Acetyl (Protein N-term)"`` → ``[(name, site), ...]``.

    ``site`` is the raw payload inside the parentheses: a residue letter
    (``"C"``) or a terminal spec (``"Protein N-term"`` / ``"N-term"`` /
    ``"C-term"`` / ``"Protein C-term"``). An entry with no ``(site)``
    payload keeps the bare name with an empty site. Empty / missing input
    returns ``[]``.
    """
    if not value:
        return []
    out: list[tuple[str, str]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _MOD_SPEC_RE.match(chunk)
        if m is None:
            out.append((chunk, ""))
        else:
            out.append((m.group("name").strip(), m.group("site").strip()))
    return out


__all__ = ["parse_parameters_txt", "parse_fixed_modifications"]
