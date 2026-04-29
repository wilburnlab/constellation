"""Generate constellation/data/proforma.lark from the upstream HUPO-PSI EBNF.

Inputs (vendored under constellation/data/_raw/):
    proforma.ebnf
        HUPO-PSI ProForma 2.0 grammar (Final draft 15, Feb 2022):
        https://raw.githubusercontent.com/HUPO-PSI/ProForma/master/grammar/proforma.ebnf

Outputs:
    constellation/data/proforma.lark
        Hand-translated lark grammar that constellation.core.sequence.proforma
        loads at parser-construction time. Lark uses Earley with `maybe_placeholders`
        so optional productions show up as `None` in the tree (matters for the
        Transformer that builds Peptidoform / MultiPeptidoform).

The EBNF and the lark grammar are NOT kept in lockstep automatically — when the
upstream EBNF moves to a new version, re-fetch via this script's REFRESH command,
audit the diff, and update the lark grammar by hand. Adopting a pure-mechanical
EBNF → lark translator was rejected because:

  1. Lark has its own conventions for whitespace handling (`%ignore`),
     terminal priorities (`.N` suffix), and case-insensitive literals
     (`"X"i`) that don't map 1:1 from EBNF.
  2. Several EBNF productions need slight refactoring to remove
     left-recursion / ambiguity that lark's Earley parser can handle but
     which produces less-useful parse trees without a hand assist.
  3. The EBNF leaves some productions deliberately under-specified
     ("not exhaustively listed all possible characters") — the lark
     grammar makes those concrete with explicit regexes.

Run from project root:
    python3 scripts/build-proforma-grammar.py            # write the .lark file
    python3 scripts/build-proforma-grammar.py REFRESH    # also re-fetch the upstream EBNF
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

UPSTREAM_EBNF_URL = (
    "https://raw.githubusercontent.com/HUPO-PSI/ProForma/master/grammar/proforma.ebnf"
)

# ──────────────────────────────────────────────────────────────────────
# Hand-translated lark grammar
#
# Faithfully follows the upstream EBNF (proforma.ebnf in data/_raw/);
# rule names mirror the EBNF (snake_cased for lark). UPPERCASE names are
# terminals (matched by regex), lowercase are non-terminals.
#
# EBNF source comment in [...]; lark rule below.
# ──────────────────────────────────────────────────────────────────────
PROFORMA_LARK = r"""// ProForma 2.0 grammar (HUPO-PSI Final draft 15, Feb 2022).
// Hand-translated from constellation/data/_raw/proforma.ebnf via
// scripts/build-proforma-grammar.py — do not hand-edit; re-run the script.

// ── Top-level entry point ─────────────────────────────────────────────
// proforma = peptidoformIonSet
// peptidoformIonSet = [namePeptidoformIonSet], {modGlobal}, {peptidoformIon, "+"}, peptidoformIon
?start: peptidoform_ion_set

peptidoform_ion_set: name_set? mod_global* (peptidoform_ion "+")* peptidoform_ion
name_set: "(>>>" NAMETEXT ")"

// ── Global modifications (isotopes + fixed mods) ──────────────────────
// modGlobal = "<", modGlobalOption, ">"
// modGlobalOption = ISOTOPE | (modDefined, "@", modGlobalLocation, {",", modGlobalLocation})
mod_global: "<" mod_global_option ">"
?mod_global_option: isotope_label | global_fixed
isotope_label: ISOTOPE_LABEL
global_fixed: mod_defined "@" global_location ("," global_location)*

// ISOTOPE = D | T | (INT, ELEMENT)
// (deuterium "D" and tritium "T" are shorthand; otherwise mass-number + element.)
ISOTOPE_LABEL: /[DT]|[0-9]+(?:He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Th|Pa|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og|U|W|I|Y|V|K|S|B|C|N|O|F|H|P)/

global_location: cterm_loc | nterm_loc | aa_loc
cterm_loc: "C-term"i (":" AMINOACID)?
nterm_loc: "N-term"i (":" AMINOACID)?
aa_loc: AMINOACID

// ── Peptidoform-ion: //-separated chains + optional charge ────────────
// peptidoformIon = [namePeptidoformIon], {peptidoform, "//"}, peptidoform, [peptidoformCharge]
peptidoform_ion: name_ion? (peptidoform "//")* peptidoform peptidoform_charge?
name_ion: "(>>" NAMETEXT ")"

// peptidoformCharge = "/", (SIGNEDINT | "[", adductIon, {",", adductIon}, "]")
peptidoform_charge: "/" (SIGNED_INT | "[" adduct_ion ("," adduct_ion)* "]")
adduct_ion: FORMULA_TEXT CHARGE_SUFFIX occurrence_spec?

// ── Peptidoform: one peptide chain ────────────────────────────────────
// peptidoform = [namePeptidoform], {globalModUnknownPos}, {modLabile}, [modNTerm], sequence, [modCTerm]
peptidoform: name_pep? unknown_pos_block* labile_mod* nterm? sequence cterm?
name_pep: "(>" NAMETEXT ")"

// globalModUnknownPos = mod, [occurrenceSpecifier], {mod, [occurrenceSpecifier]}, '?'
unknown_pos_block: unknown_pos_mod (unknown_pos_mod)* "?"
unknown_pos_mod: mod occurrence_spec?

// modLabile = "{", modInternal, "}"
labile_mod: "{" mod_internal "}"

// ── N-term / C-term ───────────────────────────────────────────────────
// modNTerm = modOrLabel, [modOrLabel], "-"
// modCTerm = "-", modOrLabel, [modOrLabel]
nterm: mod_or_label mod_or_label? "-"
cterm: "-" mod_or_label mod_or_label?

// ── Sequence body ─────────────────────────────────────────────────────
// sequence = sequenceSection, {sequenceSection}
// sequenceSection = ambiguousAminoAcid | modRange | sequenceElement
sequence: sequence_section+
?sequence_section: ambiguous_aa | mod_range | sequence_element

// ambiguousAminoAcid = "(?", sequenceElement, {sequenceElement}, ")", {mod}
ambiguous_aa: "(?" sequence_element+ ")" mod*

// modRange = "(", sequenceElement, {sequenceElement}, ")", mod, {mod}
mod_range: "(" sequence_element+ ")" mod+

// sequenceElement = aminoAcid, {modOrLabel}
sequence_element: AMINOACID mod_or_label*

AMINOACID: /[A-Za-z]/

// ── Modifications ─────────────────────────────────────────────────────
// modOrLabel = mod | ("[", modLabel, "]")
?mod_or_label: mod | bare_label
bare_label: "[" mod_label "]"

// mod = "[", modInternal, "]"
mod: "[" mod_internal "]"

// modInternal = modSingle, [modLabel], {"|", modSingle, [modLabel]}
mod_internal: mod_single mod_label? ("|" mod_single mod_label?)*

// modDefined = "[", modInternalDefined, "]"
mod_defined: "[" mod_internal_defined "]"
mod_internal_defined: mod_single ("|" mod_single)*

// modSingle = modFormula | modGlycan | modMass | modAccession | info | modName
?mod_single: mod_formula | mod_glycan | mod_mass | mod_accession | mod_info | mod_name

// modFormula = FORMULA, ":", formula, [CHARGE]
mod_formula: "Formula"i ":" FORMULA_TEXT

// modGlycan = GLYCAN, ":", glycan composition
mod_glycan: "Glycan"i ":" GLYCAN_TEXT

// modMass = [modMassCV, ":"], SIGN, NUMBER
mod_mass: (CV_MASS_PREFIX ":")? SIGNED_NUMBER
CV_MASS_PREFIX: ("U"i | "M"i | "R"i | "X"i | "G"i | "Obs"i)

// modAccession = (CVName, ":", INT) | (G,N,O, ":", ALPHANUMERIC, {ALPHANUMERIC})
// Rendered case-sensitive in the upstream spec (UNIMOD, MOD, RESID, XLMOD, GNO);
// but the EBNF case-folds via per-letter productions, so we mirror that.
mod_accession: CV_FULL ":" ACCESSION_VALUE
CV_FULL: ("UNIMOD"i | "MOD"i | "RESID"i | "XLMOD"i | "GNO"i)
ACCESSION_VALUE: /[A-Za-z0-9]+/

// info = INFO, ":", MODTEXT
mod_info: "INFO"i ":" MOD_INNER_TEXT

// modName = [CVAbbreviation, ":"], MODTEXT
mod_name: (CV_NAME_PREFIX ":")? MOD_INNER_TEXT
CV_NAME_PREFIX: ("U"i | "M"i | "R"i | "X"i | "G"i)

// ── Labels (groups + cross-links + branches) ──────────────────────────
// modLabel = "#", (modLabelXL | modLabelBranch | modLabelAmbiguous)
mod_label: "#" LABEL_NAME LABEL_SCORE?
LABEL_NAME: /(?:XL[A-Za-z0-9]+|BRANCH|[A-Za-z0-9]+)/
LABEL_SCORE: /\([+-]?[0-9]+(?:\.[0-9]+)?\)/

// occurrenceSpecifier = "^", INT
occurrence_spec: "^" INT

// ── Terminals ─────────────────────────────────────────────────────────
// The EBNF allows arbitrary unicode-graphic characters inside [...] payloads
// so long as brackets balance. We approximate that with a regex that bans
// ']' and '|' (the latter is the modSingle separator) but allows any other
// printable char including spaces. Whitespace inside payloads is preserved
// (e.g. mass-delta '+ 0.984'); leading/trailing trim happens in the
// transformer for name comparisons.
MOD_INNER_TEXT: /[^\]\|#@\[\{\}\<\>][^\]\|#@\[\{\}\<\>]*/
FORMULA_TEXT:   /[^\]\|][^\]\|]*/
GLYCAN_TEXT:    /[^\]\|][^\]\|]*/
CHARGE_SUFFIX:  /:z[+-]?[0-9]+/i
NAMETEXT:       /[^\)]+/

INT:           /[0-9]+/
SIGNED_INT:    /[+-]?[0-9]+/
SIGNED_NUMBER: /[+-]?[0-9]+(?:\.[0-9]+)?/

%ignore /[ \t]+/
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = repo_root / "constellation" / "data" / "_raw" / "proforma.ebnf"
    out_path = repo_root / "constellation" / "data" / "proforma.lark"

    if "REFRESH" in sys.argv[1:]:
        print(f"refreshing upstream EBNF from {UPSTREAM_EBNF_URL}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UPSTREAM_EBNF_URL) as r:
            raw_bytes = r.read()
        raw_path.write_bytes(raw_bytes)
        print(f"  → wrote {len(raw_bytes)} bytes to {raw_path}")

    if not raw_path.exists():
        sys.exit(
            f"missing {raw_path}; re-run with REFRESH to fetch from upstream"
        )

    out_path.write_text(PROFORMA_LARK)
    print(f"wrote lark grammar → {out_path} ({len(PROFORMA_LARK)} bytes)")
    print(
        f"upstream EBNF reference: {raw_path} "
        f"({raw_path.stat().st_size} bytes, {sum(1 for _ in raw_path.read_text().splitlines())} lines)"
    )


if __name__ == "__main__":
    main()
