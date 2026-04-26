"""Generate constellation/data/codon_tables.json from NCBI genetic codes.

NCBI numbers genetic codes via the `transl_table` field; the canonical
descriptions live at https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi.
This script encodes the seven tables most commonly seen in working biology:

    1   Standard
    2   Vertebrate Mitochondrial
    3   Yeast Mitochondrial
    4   Mold/Protozoan/Coelenterate Mitochondrial + Mycoplasma/Spiroplasma
    5   Invertebrate Mitochondrial
    6   Ciliate, Dasycladacean, Hexamita Nuclear
    11  Bacterial, Archaeal, Plant Plastid

Each table is encoded as the difference from the standard code (table 1):
the start/stop sets are explicit, and the forward map is built by
applying overrides on top of the standard. This keeps the source short
and makes deviations from the standard code visually obvious.

Run from project root:
    python3 scripts/build-codon-tables-json.py
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

BASES = "TCAG"
ALL_CODONS = ["".join(c) for c in product(BASES, repeat=3)]


# ── Standard genetic code (NCBI table 1) ─────────────────────────────
# 64 entries; stops are "*". Order is purely declarative (we sort
# lexicographically before serializing).
STANDARD_FORWARD: dict[str, str] = {
    # F/L
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    # S
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    # Y / stops
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    # C / stop / W
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    # L
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    # P
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    # H / Q
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    # R
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    # I / M
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    # T
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    # N / K
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    # S / R
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    # V
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    # A
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    # D / E
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    # G
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

assert len(STANDARD_FORWARD) == 64


# ── Tables expressed as deltas from STANDARD ─────────────────────────

TABLES = [
    {
        "transl_table": 1,
        "name": "Standard",
        "overrides": {},
        "starts": ["ATG", "TTG", "CTG"],
        "stops": ["TAA", "TAG", "TGA"],
    },
    {
        "transl_table": 2,
        "name": "Vertebrate Mitochondrial",
        "overrides": {
            "AGA": "*",   # stop instead of R
            "AGG": "*",   # stop instead of R
            "ATA": "M",   # Met instead of Ile
            "TGA": "W",   # Trp instead of stop
        },
        "starts": ["ATT", "ATC", "ATA", "ATG", "GTG"],
        "stops": ["TAA", "TAG", "AGA", "AGG"],
    },
    {
        "transl_table": 3,
        "name": "Yeast Mitochondrial",
        "overrides": {
            "ATA": "M",   # Met instead of Ile
            "CTT": "T",   # Thr instead of Leu (CTN → T)
            "CTC": "T",
            "CTA": "T",
            "CTG": "T",
            "TGA": "W",   # Trp instead of stop
        },
        "starts": ["ATA", "ATG", "GTG"],
        "stops": ["TAA", "TAG"],
    },
    {
        "transl_table": 4,
        "name": "Mold/Protozoan/Coelenterate Mitochondrial; Mycoplasma; Spiroplasma",
        "overrides": {
            "TGA": "W",   # Trp instead of stop
        },
        "starts": ["TTA", "TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"],
        "stops": ["TAA", "TAG"],
    },
    {
        "transl_table": 5,
        "name": "Invertebrate Mitochondrial",
        "overrides": {
            "AGA": "S",   # Ser instead of R
            "AGG": "S",   # Ser instead of R
            "ATA": "M",   # Met instead of Ile
            "TGA": "W",   # Trp instead of stop
        },
        "starts": ["TTG", "ATT", "ATC", "ATA", "ATG", "GTG"],
        "stops": ["TAA", "TAG"],
    },
    {
        "transl_table": 6,
        "name": "Ciliate, Dasycladacean and Hexamita Nuclear",
        "overrides": {
            "TAA": "Q",   # Gln instead of stop
            "TAG": "Q",   # Gln instead of stop
        },
        "starts": ["ATG"],
        "stops": ["TGA"],
    },
    {
        "transl_table": 11,
        "name": "Bacterial, Archaeal and Plant Plastid",
        "overrides": {},
        "starts": ["TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"],
        "stops": ["TAA", "TAG", "TGA"],
    },
]


def _build_forward(overrides: dict[str, str]) -> dict[str, str]:
    forward = dict(STANDARD_FORWARD)
    for codon, aa in overrides.items():
        forward[codon] = aa
    return dict(sorted(forward.items()))


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "constellation" / "data" / "codon_tables.json"
    tables_out = []
    for spec in TABLES:
        forward = _build_forward(spec["overrides"])
        # Sanity: every codon present, stops match the explicit `stops` set.
        assert set(forward) == set(ALL_CODONS), (
            f"table {spec['transl_table']} missing codons"
        )
        derived_stops = {c for c, aa in forward.items() if aa == "*"}
        if derived_stops != set(spec["stops"]):
            raise AssertionError(
                f"table {spec['transl_table']}: derived stops "
                f"{sorted(derived_stops)} != declared {sorted(spec['stops'])}"
            )
        tables_out.append(
            {
                "transl_table": spec["transl_table"],
                "name": spec["name"],
                "forward": forward,
                "starts": sorted(spec["starts"]),
                "stops": sorted(spec["stops"]),
            }
        )
    doc = {
        "schema_version": "1",
        "source": (
            "NCBI Genetic Codes "
            "(https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi). "
            "Each non-standard table encoded as overrides on top of table 1."
        ),
        "tables": tables_out,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(tables_out)} codon tables → {out_path}")


if __name__ == "__main__":
    main()
