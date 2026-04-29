"""Generate constellation/data/unimod.json from the upstream UNIMOD XML.

Input:
    constellation/data/_raw/unimod.xml
        Vendored copy of https://www.unimod.org/xml/unimod.xml. Refresh
        manually by re-running:
            curl -sL https://www.unimod.org/xml/unimod.xml \
                -o constellation/data/_raw/unimod.xml
        and committing the new file alongside the regenerated json.

Output:
    constellation/data/unimod.json
        {schema_version: "1", source: {...}, modifications: [...]}

Each modification record carries:
    id                 "UNIMOD:N"
    name               full_name from XML, falling back to title
    title              short title from XML (e.g. "Acetyl")
    delta_composition  {atom_symbol: int_count}  — light-atom skeleton only;
                       heavy isotope tags (13C, 15N, etc.) are folded into
                       their light equivalents so the composition is a
                       valid Composition over our atom table.
    mono_mass          XML's umod:delta @mono_mass — the canonical mass.
    avge_mass          XML's umod:delta @avge_mass.
    mass_override      mono_mass when it disagrees with the light-skeleton
                       composition mass beyond LIGHT_HEAVY_TOL Da; else None.
    aliases            (title,) plus any hand-curated short-codes.
    description        empty by default; filled where UNIMOD ships notes.

Each modification also carries:
    specificities      list of {position, site} pairs lifted from
                       <umod:specificity>. Position is one of Anywhere,
                       Any N-term, Any C-term, Protein N-term,
                       Protein C-term. Site is a residue letter or one
                       of "N-term" / "C-term". This drives N-terminal
                       vs. side-chain mod localization in PSI / EncyclopeDIA
                       readers (cf. K[Acetyl] vs [Acetyl]-K). Hidden
                       specificities (hidden="1") are skipped — they're
                       suppressed in the upstream UNIMOD UI and represent
                       speculative or artefact placements.

xrefs are dropped — those are out of scope for the chemistry layer.

Run from project root:
    python3 scripts/build-unimod-json.py
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

# ──────────────────────────────────────────────────────────────────────
# Element-symbol normalization
# ──────────────────────────────────────────────────────────────────────

# UNIMOD uses heavy-isotope element labels (13C, 15N, 2H, 18O, etc.).
# For the purposes of building a Composition over our atom table, we
# fold heavy isotopes into their light equivalents — the actual mass
# difference is captured by `mass_override` when present.
_HEAVY_TO_LIGHT = {
    "2H": "H", "D": "H",
    "13C": "C",
    "15N": "N",
    "17O": "O", "18O": "O",
    "33S": "S", "34S": "S", "36S": "S",
    "37Cl": "Cl",
}

# UNIMOD pseudo-elements: shorthand for a fixed atomic composition. These
# are NOT real elements; UNIMOD ships them as <umod:elem> entries with
# only a mass (no atomic decomposition), so we hardcode the canonical
# composition for each. Sources: UNIMOD documentation + IUPAC carbohydrate
# nomenclature for the sugar residues. Each entry is the dehydrated /
# residue form (i.e. minus the water lost when the sugar links into a
# chain), matching how UNIMOD applies them in glycan-mass calculations.
_PSEUDO_ELEMENT_COMP: dict[str, dict[str, int]] = {
    # Sugars (residue / dehydrated forms)
    "Hex":    {"C": 6, "H": 10, "O": 5},   # hexose
    "HexA":   {"C": 6, "H": 8,  "O": 6},   # hexuronic acid
    "HexN":   {"C": 6, "H": 11, "N": 1, "O": 4},  # hexosamine
    "HexNAc": {"C": 8, "H": 13, "N": 1, "O": 5},  # N-acetylhexosamine
    "dHex":   {"C": 6, "H": 10, "O": 4},   # deoxyhexose (e.g. fucose)
    "Pent":   {"C": 5, "H": 8,  "O": 4},   # pentose
    "Hep":    {"C": 7, "H": 12, "O": 6},   # heptose
    "NeuAc":  {"C": 11, "H": 17, "N": 1, "O": 8},  # sialic acid
    "NeuGc":  {"C": 11, "H": 17, "N": 1, "O": 9},  # N-glycolylneuraminic
    "Kdn":    {"C": 9,  "H": 14, "O": 8},  # deaminated NeuAc
    "Kdo":    {"C": 8,  "H": 12, "O": 7},  # 3-deoxy-octulosonic acid
    # Common chemical groups
    "Sulf":   {"S": 1, "O": 3},            # sulfo (delta SO3)
    "Phos":   {"H": 1, "O": 3, "P": 1},    # phospho (delta HPO3)
    "Water":  {"H": 2, "O": 1},
    "Me":     {"C": 1, "H": 2},            # methyl delta (H → CH3 = +CH2)
    "Ac":     {"C": 2, "H": 2, "O": 1},    # acetyl delta (H → COCH3 = +C2H2O)
}

# Diagnostics for unmapped UNIMOD element tokens.
_UNKNOWN_TOKENS: set[str] = set()


def _resolve_element(symbol: str) -> dict[str, int]:
    """Resolve a UNIMOD element token to an atom-symbol composition.

    Returns a dict (single-element for normal atoms, multi-element for
    pseudo-elements). Heavy isotopes fold to their light equivalent.
    Unknown tokens contribute nothing and are recorded for reporting.
    """
    if symbol in _HEAVY_TO_LIGHT:
        return {_HEAVY_TO_LIGHT[symbol]: 1}
    if symbol in _PSEUDO_ELEMENT_COMP:
        return dict(_PSEUDO_ELEMENT_COMP[symbol])
    # UNIMOD also ships an "e" pseudo-element (electron) for some
    # ionization-related mods. Skip silently.
    if symbol == "e":
        return {}
    # Standard symbols pass through.
    if re.match(r"^[A-Z][a-z]?$", symbol):
        return {symbol: 1}
    _UNKNOWN_TOKENS.add(symbol)
    return {}


def _add_into(target: dict[str, int], to_add: dict[str, int], multiplier: int) -> None:
    for s, n in to_add.items():
        target[s] = target.get(s, 0) + n * multiplier


# ──────────────────────────────────────────────────────────────────────
# Hand-curated short-code aliases (mirror cartographer's vocabulary)
# ──────────────────────────────────────────────────────────────────────

_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "UNIMOD:1":   ("Ac",),
    "UNIMOD:4":   ("Cam",),
    "UNIMOD:21":  ("Phospho",),
    "UNIMOD:34":  ("Me",),
    "UNIMOD:35":  ("Ox",),
    "UNIMOD:36":  ("diMe",),
    "UNIMOD:37":  ("triMe",),
    "UNIMOD:64":  ("Succ",),
    "UNIMOD:121": ("Ub", "GG"),
    "UNIMOD:737": ("TMT10",),
}


# ──────────────────────────────────────────────────────────────────────
# Mass calculation for sanity comparison (element-mass table, NOT a
# replacement for core.chem.elements — the build script is standalone)
# ──────────────────────────────────────────────────────────────────────


def _load_atom_masses(repo_root: Path) -> dict[str, float]:
    """Read constellation/data/elements.json so we can compute light-skeleton
    monoisotopic masses without importing the constellation package
    (which would force torch as a build-time dep)."""
    with (repo_root / "constellation" / "data" / "elements.json").open() as f:
        doc = json.load(f)
    return {e["symbol"]: e["monoisotopic_mass"] for e in doc["elements"]}


# Tolerance for declaring a UNIMOD entry to be "non-canonical decomposition"
# (i.e. heavy-isotope-bearing). Mods with light-skeleton mass within
# this many Da of the XML's mono_mass are treated as canonical.
LIGHT_HEAVY_TOL = 1e-3


# ──────────────────────────────────────────────────────────────────────
# XML parsing
# ──────────────────────────────────────────────────────────────────────

_NS = "{http://www.unimod.org/xmlns/schema/unimod_2}"


def _iter_mods(xml_path: Path) -> Iterator[ET.Element]:
    """Stream <umod:mod> elements without keeping the whole tree in RAM."""
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == f"{_NS}mod":
            yield elem
            elem.clear()


def _parse_one(elem: ET.Element, atom_masses: dict[str, float]) -> dict | None:
    record_id = elem.get("record_id")
    if record_id is None:
        return None
    title = html.unescape(elem.get("title", ""))
    full_name = html.unescape(elem.get("full_name", "") or title)

    delta = elem.find(f"{_NS}delta")
    if delta is None:
        return None

    mono = delta.get("mono_mass")
    avge = delta.get("avge_mass")
    if mono is None:
        return None
    mono = float(mono)
    avge = float(avge) if avge is not None else None

    # Build the light-skeleton composition from <umod:element> children.
    comp: dict[str, int] = {}
    for child in delta.findall(f"{_NS}element"):
        sym = child.get("symbol", "")
        n = int(child.get("number", "0"))
        if not sym or n == 0:
            continue
        _add_into(comp, _resolve_element(sym), n)

    light_mass = sum(atom_masses.get(s, 0.0) * c for s, c in comp.items())
    is_heavy = abs(light_mass - mono) > LIGHT_HEAVY_TOL
    mass_override = mono if is_heavy else None

    aliases = _ALIAS_MAP.get(f"UNIMOD:{record_id}", ())
    if title and title not in aliases:
        aliases = (title, *aliases)

    specs: list[dict[str, str]] = []
    for spec in elem.findall(f"{_NS}specificity"):
        if spec.get("hidden") == "1":
            continue
        position = spec.get("position", "")
        site = spec.get("site", "")
        if not position or not site:
            continue
        specs.append({"position": position, "site": site})

    return {
        "id": f"UNIMOD:{record_id}",
        "title": title,
        "name": full_name,
        "delta_composition": comp,
        "mono_mass": mono,
        "avge_mass": avge,
        "mass_override": mass_override,
        "aliases": list(aliases),
        "specificities": specs,
        "description": "",
    }


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def build(repo_root: Path) -> dict:
    atom_masses = _load_atom_masses(repo_root)
    xml_path = repo_root / "constellation" / "data" / "_raw" / "unimod.xml"
    mods = []
    for elem in _iter_mods(xml_path):
        rec = _parse_one(elem, atom_masses)
        if rec is not None:
            mods.append(rec)
    return {
        "schema_version": "1",
        "source": {
            "xml": "https://www.unimod.org/xml/unimod.xml",
            "vendored_at": "constellation/data/_raw/unimod.xml",
        },
        "modifications": mods,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data = build(repo_root)
    out_path = repo_root / "constellation" / "data" / "unimod.json"
    out_path.write_text(json.dumps(data, indent=1) + "\n")
    n = len(data["modifications"])
    n_heavy = sum(1 for m in data["modifications"] if m["mass_override"] is not None)
    print(f"wrote {out_path} — {n} modifications, {n_heavy} heavy-isotope")
    if _UNKNOWN_TOKENS:
        print(f"WARN: unmapped UNIMOD element tokens: {sorted(_UNKNOWN_TOKENS)}")
        print("      these mods will have incomplete delta_composition; "
              "mono_mass is still authoritative via mass_override")
    # Spot-check a few cartographer-anchored entries.
    by_id = {m["id"]: m for m in data["modifications"]}
    expected = {
        "UNIMOD:1":   ("Acetyl",      42.010565),
        "UNIMOD:4":   ("Carbamidomethyl", 57.021464),
        "UNIMOD:21":  ("Phospho",     79.966331),
        "UNIMOD:35":  ("Oxidation",   15.994915),
        "UNIMOD:121": ("GlyGly",     114.042927),
        "UNIMOD:737": ("TMT6plex",   229.162932),  # heavy-isotope path
    }
    for mid, (title, m_expected) in expected.items():
        rec = by_id.get(mid)
        if rec is None:
            print(f"  MISSING: {mid}")
            continue
        diff = abs(rec["mono_mass"] - m_expected)
        flag = " (heavy)" if rec["mass_override"] is not None else ""
        print(f"  {mid} {rec['title']:<22} mono={rec['mono_mass']:.6f} "
              f"vs expected {m_expected:.6f} (Δ={diff:.2e}){flag}")


if __name__ == "__main__":
    main()
