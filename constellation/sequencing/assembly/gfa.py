"""Minimal GFA segment parser — extract contigs from an assembler's GFA.

GFA1 ``S`` (segment) lines carry a contig name, its sequence, and
optional typed tags (``LN:i`` length, ``rd:i`` read depth). hifiasm
writes its primary / haplotype contigs as ``S`` lines in
``<prefix>.bp.p_ctg.gfa`` / ``.hap[12].p_ctg.gfa`` with the assembled
sequence embedded and an ``rd:i`` read-coverage tag per segment —
exactly the provenance ``ASSEMBLY_CONTIG_TABLE.read_coverage`` wants
(and which a bare ``.fa`` discards).

Stdlib-only + gzip-aware, matching the ``readers/fastx.py`` convention
(no ``gfapy`` dependency for a format this simple). Pure function →
unit-testable with an inline fixture. ``circular`` is left ``None``
here — GFA1 has no standard per-segment circularity tag; the
assembler-specific convention (hifiasm's ``l``/``c`` name suffix) is
applied one layer up in :mod:`sequencing.assembly.hifiasm`.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GfaContig:
    """One parsed GFA ``S`` (segment) record."""

    name: str
    # Empty string when the GFA stored '*' (sequence held elsewhere).
    sequence: str
    length: int
    read_coverage: float | None
    circular: bool | None


def _open_text(path: Path) -> Any:
    """Open a (optionally gzipped) text file — mirrors readers/fastx.py."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_tags(fields: list[str]) -> dict[str, Any]:
    """Parse GFA ``TAG:TYPE:VALUE`` fields into a ``{tag: value}`` dict.

    ``i`` casts to int, ``f`` to float; everything else stays a string.
    Malformed fields are skipped rather than raising — assembler output
    occasionally carries vendor tags we don't model.
    """
    tags: dict[str, Any] = {}
    for raw in fields:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            continue
        tag, typ, value = parts
        if typ == "i":
            try:
                tags[tag] = int(value)
            except ValueError:
                continue
        elif typ == "f":
            try:
                tags[tag] = float(value)
            except ValueError:
                continue
        else:
            tags[tag] = value
    return tags


def parse_gfa_contigs(path: str | Path) -> list[GfaContig]:
    """Parse the ``S`` (segment) lines of a GFA file into contig records.

    Lines other than segments (``H``/``L``/``P``/``A``/...) are ignored.
    Length comes from the ``LN:i`` tag when present, else from the
    embedded sequence length; ``read_coverage`` from ``rd:i``/``rd:f``.
    """
    p = Path(path)
    out: list[GfaContig] = []
    with _open_text(p) as fh:
        for line in fh:
            if not line.startswith("S\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[1]
            seq = fields[2]
            tags = _parse_tags(fields[3:])
            seq_present = seq != "*"
            ln = tags.get("LN")
            length = int(ln) if ln is not None else (len(seq) if seq_present else 0)
            cov = tags.get("rd")
            coverage = float(cov) if cov is not None else None
            out.append(
                GfaContig(
                    name=name,
                    sequence=seq if seq_present else "",
                    length=length,
                    read_coverage=coverage,
                    circular=None,
                )
            )
    return out


__all__ = ["GfaContig", "parse_gfa_contigs"]
