"""Generate constellation/data/proteases.json from a curated source list.

Cleavage patterns are zero-width regex assertions — `(?<=[KR])(?!P)` means
"cleave between any K|R and the following character, unless that next
character is P". This matches the convention used by pyteomics'
`expasy_rules`, but the patterns here are inlined so `core.sequence.protein`
does not depend on pyteomics at runtime.

Patterns adapted from the ExPASy PeptideCutter rule set
(https://web.expasy.org/peptide_cutter/peptidecutter_enzymes.html), which
is the same upstream source pyteomics encodes.

Run from project root:
    python3 scripts/build-proteases-json.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROTEASES = [
    # ── Trypsin family ───────────────────────────────────────────────
    {
        "id": "Trypsin",
        "name": "Trypsin",
        "regex_pattern": r"(?<=[KR])(?!P)",
        "specificity": "high",
        "description": "Cleaves C-terminal to K or R, not before P. Default proteomics workhorse.",
    },
    {
        "id": "Trypsin/P",
        "name": "Trypsin (no proline rule)",
        "regex_pattern": r"(?<=[KR])",
        "specificity": "high",
        "description": "Cleaves C-terminal to K or R, including before P. Common in some search-engine defaults.",
    },
    # ── Single-residue specificity ───────────────────────────────────
    {
        "id": "LysC",
        "name": "Lys-C",
        "regex_pattern": r"(?<=K)(?!P)",
        "specificity": "high",
        "description": "Cleaves C-terminal to K, not before P. Endoproteinase Lys-C.",
    },
    {
        "id": "LysC/P",
        "name": "Lys-C (no proline rule)",
        "regex_pattern": r"(?<=K)",
        "specificity": "high",
        "description": "Cleaves C-terminal to K, including before P.",
    },
    {
        "id": "LysN",
        "name": "Lys-N",
        "regex_pattern": r"(?=K)",
        "specificity": "high",
        "description": "Cleaves N-terminal to K. Metalloendopeptidase from Grifola frondosa.",
    },
    {
        "id": "ArgC",
        "name": "Arg-C",
        "regex_pattern": r"(?<=R)(?!P)",
        "specificity": "high",
        "description": "Cleaves C-terminal to R, not before P. Clostripain.",
    },
    # ── Chymotrypsin ─────────────────────────────────────────────────
    {
        "id": "Chymotrypsin",
        "name": "Chymotrypsin (high specificity)",
        "regex_pattern": r"(?<=[FWY])(?!P)",
        "specificity": "high",
        "description": "Cleaves C-terminal to F|W|Y, not before P.",
    },
    {
        "id": "Chymotrypsin/low",
        "name": "Chymotrypsin (low specificity)",
        "regex_pattern": r"(?<=[FWYLM])(?!P)",
        "specificity": "low",
        "description": "Cleaves C-terminal to F|W|Y|L|M, not before P. Reflects observed cleavage at extended specificity.",
    },
    # ── Asp-N family ─────────────────────────────────────────────────
    {
        "id": "AspN",
        "name": "Asp-N",
        "regex_pattern": r"(?=D)",
        "specificity": "high",
        "description": "Cleaves N-terminal to D. Endoproteinase Asp-N.",
    },
    {
        "id": "AspN+N",
        "name": "Asp-N + Glu",
        "regex_pattern": r"(?=[DE])",
        "specificity": "low",
        "description": "Cleaves N-terminal to D or E (occurs with some Asp-N preparations).",
    },
    # ── Glu-C family ─────────────────────────────────────────────────
    {
        "id": "GluC",
        "name": "Glu-C (V8 phosphate buffer)",
        "regex_pattern": r"(?<=E)(?!P)",
        "specificity": "high",
        "description": "Cleaves C-terminal to E, not before P. Staphylococcal V8 in phosphate buffer.",
    },
    {
        "id": "GluC_bicarb",
        "name": "Glu-C (V8 bicarbonate buffer)",
        "regex_pattern": r"(?<=[DE])(?!P)",
        "specificity": "high",
        "description": "Cleaves C-terminal to D or E, not before P. Staphylococcal V8 in bicarbonate buffer.",
    },
    # ── Pepsin ───────────────────────────────────────────────────────
    {
        "id": "Pepsin_pH1.3",
        "name": "Pepsin (pH 1.3)",
        "regex_pattern": r"(?<=[FL])(?!P)",
        "specificity": "low",
        "description": "Cleaves C-terminal to F or L, not before P. Pepsin at low pH (~1.3).",
    },
    {
        "id": "Pepsin_pH2",
        "name": "Pepsin (pH > 2)",
        "regex_pattern": r"(?<=[FLWY])(?!P)",
        "specificity": "low",
        "description": "Cleaves C-terminal to F|L|W|Y, not before P. Pepsin at higher pH (>2).",
    },
    # ── Broad-specificity ────────────────────────────────────────────
    {
        "id": "ProteinaseK",
        "name": "Proteinase K",
        "regex_pattern": r"(?<=[AFILVWY])",
        "specificity": "low",
        "description": "Broad serine protease; cleaves C-terminal to A|F|I|L|V|W|Y. Useful for surface-shaving and HDX.",
    },
    {
        "id": "Thermolysin",
        "name": "Thermolysin",
        "regex_pattern": r"(?=[AFILMV])(?<!D)(?<!E)",
        "specificity": "low",
        "description": "Cleaves N-terminal to A|F|I|L|M|V; not after D|E. Heat-stable metalloprotease from Bacillus thermoproteolyticus.",
    },
    # ── Pass-through ─────────────────────────────────────────────────
    {
        "id": "No_enzyme",
        "name": "No enzyme",
        "regex_pattern": r"(?!)",
        "specificity": "high",
        "description": "Never matches; cleave() returns the whole protein unchanged. Useful for searches that consider arbitrary peptide spans.",
    },
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "constellation" / "data" / "proteases.json"
    doc = {
        "schema_version": "1",
        "source": (
            "Patterns adapted from ExPASy PeptideCutter "
            "(https://web.expasy.org/peptide_cutter/) — same upstream as "
            "pyteomics.parser.expasy_rules. Curated for Constellation."
        ),
        "proteases": PROTEASES,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(PROTEASES)} proteases → {out_path}")


if __name__ == "__main__":
    main()
