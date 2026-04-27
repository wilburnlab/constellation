"""Generate constellation/data/elements.json from authoritative sources.

Inputs (vendored under constellation/data/_raw/):
    nist_compositions.txt
        NIST Atomic Weights & Isotopic Compositions ("ascii2" full dump):
        https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=some
        Source for: atomic_number, monoisotopic_mass, standard_atomic_weight,
        and the per-isotope (mass_number, exact_mass, abundance) tuples.

Hardcoded supplementary tables (in this script):
    PAULING_EN              Pauling electronegativity (Pauling 1960; Allred-Rochow
                            for heavy elements where Pauling did not measure).
    COVALENT_RADIUS_PM      Cordero et al., Dalton Trans., 2008, 2832-2838.
    VDW_RADIUS_PM           Bondi (J. Phys. Chem. 68, 441 (1964)) for noble gases,
                            CHNOPSF; Alvarez (Dalton Trans., 2013, 8617-8636) for
                            transition metals and elements not in Bondi.

Output:
    constellation/data/elements.json
        Single JSON document; ELEMENTS = list of element records, schema version "1".

Run from project root:
    python3 scripts/build-elements-json.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# NIST text-dump parser
# ──────────────────────────────────────────────────────────────────────

_RECORD_FIELDS = (
    "Atomic Number",
    "Atomic Symbol",
    "Mass Number",
    "Relative Atomic Mass",
    "Isotopic Composition",
    "Standard Atomic Weight",
    "Notes",
)

# Strip "(N)" / "(N#)" parenthesized uncertainty suffix from numeric fields.
_PAREN_UNCERT = re.compile(r"\(\d+#?\)$")


def _parse_record(block: str) -> dict[str, str]:
    """Parse one NIST 'Atomic Number = ...' record into a flat dict."""
    out: dict[str, str] = {}
    for line in block.strip().splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def _parse_uncert_float(s: str) -> float | None:
    """Parse '12.0000000(00)' or '6.0151228874(16)' → float. Empty → None."""
    s = s.strip()
    if not s:
        return None
    s = _PAREN_UNCERT.sub("", s)
    if not s:
        return None
    return float(s)


def _parse_standard_weight(s: str) -> float | None:
    """Standard atomic weight may be: '15.9994(3)', '[15.99903,15.99977]',
    '[209]' (mass number for radioactive-only), or empty.
    Convention: numeric value or midpoint of range; mass-number brackets
    return that integer as a float (best available scalar for the element).
    """
    s = s.strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        if "," in inner:
            lo, hi = (float(x) for x in inner.split(","))
            return (lo + hi) / 2.0
        # single number in brackets — typically a mass number for radioactive-only
        return float(inner)
    return _parse_uncert_float(s)


def parse_nist(path: Path) -> dict[int, dict]:
    """Return {atomic_number: {symbol, isotopes[], standard_atomic_weight}}."""
    text = path.read_text()
    blocks = [b for b in text.split("\n\n") if "Atomic Number" in b]
    elements: dict[int, dict] = {}
    for block in blocks:
        rec = _parse_record(block)
        z = int(rec["Atomic Number"])
        symbol = rec["Atomic Symbol"]
        mass_number = int(rec["Mass Number"])
        exact_mass = _parse_uncert_float(rec["Relative Atomic Mass"])
        abundance_str = rec.get("Isotopic Composition", "")
        abundance = _parse_uncert_float(abundance_str)
        weight = _parse_standard_weight(rec.get("Standard Atomic Weight", ""))

        if exact_mass is None:
            continue  # malformed row

        if z not in elements:
            elements[z] = {
                "atomic_number": z,
                "symbol": _canonical_symbol_for_z(z, symbol),
                "isotopes": [],
                "standard_atomic_weight": weight,
            }
        # NIST may emit D / T as Atomic Symbol for hydrogen isotopes (Z=1).
        # Always trust atomic_number for grouping; symbol comes from the
        # canonical-symbol table.
        elements[z]["isotopes"].append(
            {
                "mass_number": mass_number,
                "exact_mass": exact_mass,
                "abundance": abundance if abundance is not None else 0.0,
            }
        )
        # Prefer non-empty standard atomic weights (some isotope rows leave it
        # blank for radioactives).
        if weight is not None and elements[z]["standard_atomic_weight"] is None:
            elements[z]["standard_atomic_weight"] = weight

    for z, e in elements.items():
        e["isotopes"].sort(key=lambda i: i["mass_number"])
    return elements


def _canonical_symbol_for_z(z: int, nist_symbol: str) -> str:
    """NIST emits 'D' and 'T' as alternate symbols for H mass numbers 2/3.
    Fold those back to 'H'; pass everything else through unchanged."""
    if z == 1:
        return "H"
    return nist_symbol


def derive_monoisotopic_mass(isotopes: list[dict]) -> float:
    """Highest natural abundance wins. For radioactive-only elements (all
    abundances zero), use the lowest mass number's exact mass — a defensible
    convention; downstream code that cares should consult `isotopes` directly.
    """
    nonzero = [i for i in isotopes if i["abundance"] > 0]
    if nonzero:
        return max(nonzero, key=lambda i: i["abundance"])["exact_mass"]
    return min(isotopes, key=lambda i: i["mass_number"])["exact_mass"]


# ──────────────────────────────────────────────────────────────────────
# Hardcoded supplementary tables
# ──────────────────────────────────────────────────────────────────────

# Pauling electronegativity. None where the value is undefined or
# unmeasured (noble gases that don't form compounds, synthetic
# transactinides). Sources: Pauling (1960) for the original scale;
# Allred-Rochow / experimental updates for elements past Bk.
PAULING_EN: dict[str, float | None] = {
    "H": 2.20, "He": None,
    "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Ne": None,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": None,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
    "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01,
    "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90,
    "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10,
    "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20,
    "Tb": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27,
    "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.00, "At": 2.20, "Rn": 2.20,
    "Fr": 0.70, "Ra": 0.90, "Ac": 1.10, "Th": 1.30, "Pa": 1.50, "U": 1.38, "Np": 1.36,
    "Pu": 1.28, "Am": 1.13, "Cm": 1.28, "Bk": 1.30, "Cf": 1.30, "Es": 1.30, "Fm": 1.30,
    "Md": 1.30, "No": 1.30, "Lr": None,
    "Rf": None, "Db": None, "Sg": None, "Bh": None, "Hs": None, "Mt": None, "Ds": None,
    "Rg": None, "Cn": None, "Nh": None, "Fl": None, "Mc": None, "Lv": None, "Ts": None, "Og": None,
}

# Covalent radii in pm. Cordero et al., Dalton Trans., 2008, 2832-2838,
# single-bond values. Values past Cm are extrapolated / less reliable;
# explicitly None where no published value exists.
COVALENT_RADIUS_PM: dict[str, int | None] = {
    "H": 31, "He": 28,
    "Li": 128, "Be": 96, "B": 84, "C": 76, "N": 71, "O": 66, "F": 57, "Ne": 58,
    "Na": 166, "Mg": 141, "Al": 121, "Si": 111, "P": 107, "S": 105, "Cl": 102, "Ar": 106,
    "K": 203, "Ca": 176, "Sc": 170, "Ti": 160, "V": 153, "Cr": 139, "Mn": 139,
    "Fe": 132, "Co": 126, "Ni": 124, "Cu": 132, "Zn": 122, "Ga": 122, "Ge": 120,
    "As": 119, "Se": 120, "Br": 120, "Kr": 116,
    "Rb": 220, "Sr": 195, "Y": 190, "Zr": 175, "Nb": 164, "Mo": 154, "Tc": 147,
    "Ru": 146, "Rh": 142, "Pd": 139, "Ag": 145, "Cd": 144, "In": 142, "Sn": 139,
    "Sb": 139, "Te": 138, "I": 139, "Xe": 140,
    "Cs": 244, "Ba": 215, "La": 207,
    "Ce": 204, "Pr": 203, "Nd": 201, "Pm": 199, "Sm": 198, "Eu": 198, "Gd": 196,
    "Tb": 194, "Dy": 192, "Ho": 192, "Er": 189, "Tm": 190, "Yb": 187, "Lu": 187,
    "Hf": 175, "Ta": 170, "W": 162, "Re": 151, "Os": 144, "Ir": 141, "Pt": 136,
    "Au": 136, "Hg": 132, "Tl": 145, "Pb": 146, "Bi": 148, "Po": 140, "At": 150, "Rn": 150,
    "Fr": 260, "Ra": 221, "Ac": 215, "Th": 206, "Pa": 200, "U": 196, "Np": 190,
    "Pu": 187, "Am": 180, "Cm": 169,
    "Bk": None, "Cf": None, "Es": None, "Fm": None, "Md": None, "No": None, "Lr": None,
    "Rf": None, "Db": None, "Sg": None, "Bh": None, "Hs": None, "Mt": None, "Ds": None,
    "Rg": None, "Cn": None, "Nh": None, "Fl": None, "Mc": None, "Lv": None, "Ts": None, "Og": None,
}

# vdW radii in pm. Bondi (1964) where available; Alvarez (Dalton Trans., 2013,
# 8617-8636) for transition metals and others not in Bondi. None where
# undefined / synthetic.
VDW_RADIUS_PM: dict[str, int | None] = {
    "H": 120, "He": 140,
    "Li": 182, "Be": 153, "B": 192, "C": 170, "N": 155, "O": 152, "F": 147, "Ne": 154,
    "Na": 227, "Mg": 173, "Al": 184, "Si": 210, "P": 180, "S": 180, "Cl": 175, "Ar": 188,
    "K": 275, "Ca": 231, "Sc": 211, "Ti": 187, "V": 179, "Cr": 189, "Mn": 197,
    "Fe": 194, "Co": 192, "Ni": 163, "Cu": 140, "Zn": 139, "Ga": 187, "Ge": 211,
    "As": 185, "Se": 190, "Br": 185, "Kr": 202,
    "Rb": 303, "Sr": 249, "Y": 219, "Zr": 186, "Nb": 207, "Mo": 209, "Tc": 209,
    "Ru": 207, "Rh": 195, "Pd": 202, "Ag": 172, "Cd": 158, "In": 193, "Sn": 217,
    "Sb": 206, "Te": 206, "I": 198, "Xe": 216,
    "Cs": 343, "Ba": 268, "La": 240,
    "Ce": 235, "Pr": 239, "Nd": 229, "Pm": 236, "Sm": 229, "Eu": 233, "Gd": 237,
    "Tb": 221, "Dy": 229, "Ho": 216, "Er": 235, "Tm": 227, "Yb": 242, "Lu": 221,
    "Hf": 212, "Ta": 217, "W": 210, "Re": 217, "Os": 216, "Ir": 202, "Pt": 209,
    "Au": 166, "Hg": 209, "Tl": 196, "Pb": 202, "Bi": 207, "Po": 197, "At": 202, "Rn": 220,
    "Fr": 348, "Ra": 283, "Ac": 247, "Th": 245, "Pa": 243, "U": 241, "Np": 239,
    "Pu": 243, "Am": 244, "Cm": 245, "Bk": 244, "Cf": 245, "Es": 245, "Fm": None,
    "Md": None, "No": None, "Lr": None,
    "Rf": None, "Db": None, "Sg": None, "Bh": None, "Hs": None, "Mt": None, "Ds": None,
    "Rg": None, "Cn": None, "Nh": None, "Fl": None, "Mc": None, "Lv": None, "Ts": None, "Og": None,
}

# Group, period, valence-electron count derived from atomic number.
# `group` follows IUPAC 1-18; lanthanides (57-71) and actinides (89-103)
# are mapped to -1 (no canonical group assignment).

LANTHANIDES = set(range(57, 72))   # La..Lu
ACTINIDES = set(range(89, 104))    # Ac..Lr


def derive_group_period(z: int) -> tuple[int, int]:
    """Return (group, period). Lanthanides and actinides → group = -1."""
    if z in LANTHANIDES:
        return (-1, 6)
    if z in ACTINIDES:
        return (-1, 7)
    # Period boundaries (last Z of each period): 2, 10, 18, 36, 54, 86, 118
    if z <= 2:
        period = 1
        group = 1 if z == 1 else 18
        return (group, period)
    if z <= 10:
        period = 2
        idx = z - 2  # 1..8
        group = idx if idx <= 2 else idx + 10  # 1, 2, 13..18
        return (group, period)
    if z <= 18:
        period = 3
        idx = z - 10
        group = idx if idx <= 2 else idx + 10
        return (group, period)
    if z <= 36:
        period = 4
        idx = z - 18  # 1..18
        return (idx, period)
    if z <= 54:
        period = 5
        idx = z - 36
        return (idx, period)
    if z <= 86:
        period = 6
        # Period 6: Cs (55) → Ba (56) → [La 57..71 lanthanides] → Lu via group 3
        # → Hf (72) group 4 → … → Rn (86) group 18.
        # We've already returned for lanthanides above.
        if z <= 56:
            return (z - 54, period)  # Cs=1, Ba=2
        return (z - 71 + 3, period)  # Hf=4 ... Rn=18
    # Period 7
    period = 7
    if z <= 88:
        return (z - 86, period)  # Fr=1, Ra=2
    # Lr (103) returns group 3, Rf (104)→4, ..., Og (118)→18
    return (z - 103 + 3, period)


def derive_valence_electrons(z: int) -> int:
    """Approximate outer-shell valence electron count.

    For main-group elements this is the number of s+p electrons in the
    outermost shell. For d-block we return s+d electrons (matches the
    convention used in graph NN featurizers like RDKit's
    GetTotalValenceElectrons). For f-block (-1 group) we return s+f.
    Synthetic transactinides past Cn fall into p-block by group.
    """
    g, p = derive_group_period(z)
    if g == -1:
        # f-block: 2 (outer s) + position in series
        if z in LANTHANIDES:
            return 2 + (z - 56)
        if z in ACTINIDES:
            return 2 + (z - 88)
        return 2
    # Main-group + d-block
    if g <= 2:
        return g
    if g <= 12:
        return g  # transition metals, group 3..12 → 3..12 valence (s+d)
    return g - 10  # p-block: group 13..18 → 3..8


# ──────────────────────────────────────────────────────────────────────
# Build pipeline
# ──────────────────────────────────────────────────────────────────────


def build(repo_root: Path) -> dict:
    nist_path = repo_root / "constellation" / "data" / "_raw" / "nist_compositions.txt"
    elements = parse_nist(nist_path)

    out = []
    for z in sorted(elements):
        e = elements[z]
        symbol = e["symbol"]
        g, p = derive_group_period(z)
        record = {
            "atomic_number": z,
            "symbol": symbol,
            "monoisotopic_mass": derive_monoisotopic_mass(e["isotopes"]),
            "standard_atomic_weight": e["standard_atomic_weight"],
            "isotopes": e["isotopes"],
            "pauling_electronegativity": PAULING_EN.get(symbol),
            "covalent_radius_pm": COVALENT_RADIUS_PM.get(symbol),
            "vdw_radius_pm": VDW_RADIUS_PM.get(symbol),
            "valence_electrons": derive_valence_electrons(z),
            "group": g,
            "period": p,
        }
        out.append(record)

    return {
        "schema_version": "1",
        "source": {
            "isotopes": "NIST AME2020 (Atomic Weights and Isotopic Compositions)",
            "isotopes_url": "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl",
            "electronegativity": "Pauling 1960; Allred-Rochow for heavy elements",
            "covalent_radii": "Cordero et al., Dalton Trans., 2008, 2832-2838",
            "vdw_radii": "Bondi 1964 (J. Phys. Chem. 68, 441); Alvarez 2013 (Dalton Trans., 8617-8636)",
        },
        "elements": out,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data = build(repo_root)
    out_path = repo_root / "constellation" / "data" / "elements.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    n = len(data["elements"])
    sym = {e["symbol"] for e in data["elements"]}
    n_iso = sum(len(e["isotopes"]) for e in data["elements"])
    print(f"wrote {out_path} — {n} elements, {n_iso} isotopes")
    for must in ("H", "C", "N", "O", "P", "S", "Fe", "I", "U", "Og"):
        assert must in sym, f"missing element: {must}"
    print("sanity check passed.")


if __name__ == "__main__":
    main()
