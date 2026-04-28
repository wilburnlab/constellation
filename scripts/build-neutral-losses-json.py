"""Generate constellation/data/neutral_losses.json from a curated source list.

Neutral losses are characteristic small-molecule departures (H2O, NH3, HPO3,
H3PO4, ...) that fragment ions emit during MS/MS. Each loss has a chemical
delta composition plus biochemical-rule predicates: which residues license
it, which modifications license it, and whether it's restricted to certain
backbone ion types.

The data shape mirrors `proteases.json` and `unimod.json` — a flat JSON
array loaded by `core.massspec.peptide.neutral_losses.LossRegistry._from_doc`
into the `LOSS_REGISTRY` singleton at import.

Compositions are stored as flat ``{symbol: count}`` dicts (the same shape
`Composition.from_dict` accepts), not as Hill-notation strings — keeping the
data side dependency-free of the parse_formula regex.

Run from project root:
    python3 scripts/build-neutral-losses-json.py

Source: Cartographer's `neutral_losses` dict (cartographer/masses.py)
trimmed to the chemistry-driven subset; the ubiquitin-remnant losses
(LRGG glycine, full UNIMOD:121) are punted to user-side `register_custom`
because they're K-Ub-specific. The CO loss is also dropped: `a` is now
a first-class IonType (a = b - CO), so a "CO loss applied to b" would be
double-counting.
"""

from __future__ import annotations

import json
from pathlib import Path

NEUTRAL_LOSSES = [
    # ── Water loss ──────────────────────────────────────────────────
    {
        "id": "H2O",
        "name": "Water loss",
        "delta_composition": {"H": 2, "O": 1},
        "triggering_residues": ["S", "T", "D", "E"],
        "triggering_mods": [],
        "applies_to_ion_types": [],
        "description": (
            "Hydroxyl side-chain water loss — common from S/T/D/E containing "
            "fragments. Standard 18.011 Da neutral loss."
        ),
    },
    # ── Ammonia loss ────────────────────────────────────────────────
    {
        "id": "NH3",
        "name": "Ammonia loss",
        "delta_composition": {"N": 1, "H": 3},
        "triggering_residues": ["K", "R", "N", "Q"],
        "triggering_mods": [],
        "applies_to_ion_types": [],
        "description": (
            "Amine side-chain ammonia loss — common from K/R/N/Q containing "
            "fragments. Standard 17.027 Da neutral loss."
        ),
    },
    # ── Phospho neutral losses ──────────────────────────────────────
    {
        "id": "HPO3",
        "name": "Metaphosphate loss",
        "delta_composition": {"H": 1, "P": 1, "O": 3},
        "triggering_residues": [],
        "triggering_mods": ["UNIMOD:21"],
        "applies_to_ion_types": [],
        "description": (
            "Metaphosphate (HPO3, ~79.97 Da) loss from any phosphorylated "
            "residue (S/T/Y). Licensed by the presence of UNIMOD:21 (Phospho) "
            "anywhere in the fragment."
        ),
    },
    {
        "id": "H3PO4",
        "name": "Phosphoric acid loss",
        "delta_composition": {"H": 3, "P": 1, "O": 4},
        "triggering_residues": ["S", "T"],
        "triggering_mods": ["UNIMOD:21"],
        "applies_to_ion_types": [],
        "description": (
            "Phosphoric acid (H3PO4, ~97.98 Da) loss via β-elimination from "
            "phospho-S or phospho-T. Requires the phospho mod to sit on an "
            "S or T residue inside the fragment (Y phospho cannot β-eliminate)."
        ),
    },
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "constellation" / "data" / "neutral_losses.json"
    doc = {
        "schema_version": "1",
        "source": (
            "Curated for Constellation. Chemistry sourced from Cartographer's "
            "neutral_losses dict (cartographer/masses.py); ion-type-only and "
            "K-Ub remnant losses excluded — those route through the IonType "
            "enum or LossRegistry.register_custom respectively."
        ),
        "neutral_losses": NEUTRAL_LOSSES,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(NEUTRAL_LOSSES)} neutral losses → {out_path}")


if __name__ == "__main__":
    main()
