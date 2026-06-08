"""AGP v2 parser → ``SCAFFOLD_TABLE`` rows.

RagTag writes the scaffold layout as ``ragtag.scaffold.agp``: per
scaffold, an ordered run of component (``W``) rows interleaved with gap
(``N`` / ``U``) rows. We project that into ``SCAFFOLD_TABLE`` (one row per
constituent contig) — pairing each contig with the gap that FOLLOWS it
(``gap_size`` = bp of Ns to the next contig; ``-1`` for the terminal
contig of a scaffold).

AGP component rows are tab-separated::

    object  obj_beg  obj_end  part_no  W  component_id  comp_beg  comp_end  orientation

and gap rows::

    object  obj_beg  obj_end  part_no  N|U  gap_length  gap_type  linkage  evidence

``contig_name_to_id`` maps AGP ``component_id`` (the draft contig name)
to the draft contig id. Stdlib-only, gzip-aware, pure → unit-testable.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

# AGP gap component-type → SCAFFOLD_TABLE.gap_type vocabulary.
_GAP_TYPE = {"N": "estimated", "U": "unknown"}


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_agp(
    path: str | Path,
    contig_name_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Parse an AGP file into ``SCAFFOLD_TABLE``-shaped row dicts.

    Raises ``ValueError`` if a component names a contig absent from
    ``contig_name_to_id`` (a real AGP only references draft contigs).
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    scaffold_ids: dict[str, int] = {}
    position_in: dict[str, int] = {}
    last_component_idx: dict[int, int] = {}

    with _open_text(p) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 5:
                continue
            obj = f[0]
            comp_type = f[4]
            if obj not in scaffold_ids:
                scaffold_ids[obj] = len(scaffold_ids)
                position_in[obj] = 0
            sid = scaffold_ids[obj]

            if comp_type in _GAP_TYPE:
                gap_length = -1
                if len(f) > 5 and f[5].lstrip("-").isdigit():
                    gap_length = int(f[5])
                idx = last_component_idx.get(sid)
                if idx is not None:
                    rows[idx]["gap_size"] = gap_length
                    rows[idx]["gap_type"] = _GAP_TYPE[comp_type]
                continue

            # component row (W, or any non-gap type)
            component_id = f[5] if len(f) > 5 else ""
            orientation = f[8] if len(f) > 8 else "+"
            if component_id not in contig_name_to_id:
                raise ValueError(
                    f"AGP component {component_id!r} (scaffold {obj!r}) not found "
                    "in the assembly's contig names"
                )
            rows.append(
                {
                    "scaffold_id": sid,
                    "name": obj,
                    "contig_id": contig_name_to_id[component_id],
                    "position": position_in[obj],
                    "orientation": orientation,
                    "gap_size": -1,  # terminal unless a following gap updates it
                    "gap_type": None,
                }
            )
            last_component_idx[sid] = len(rows) - 1
            position_in[obj] += 1

    return rows


__all__ = ["parse_agp"]
