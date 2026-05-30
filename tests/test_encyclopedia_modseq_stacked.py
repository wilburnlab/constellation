"""Regression tests for stacked-bracket EncyclopeDIA modseqs.

EncyclopeDIA folds an N-terminal modification and a side-chain modification
on residue 0 into consecutive ``X[+a][+b]`` brackets (e.g. N-term Acetyl +
Cys Carbamidomethyl on the first residue). The reader must consume every
bracket on a residue, not just the first. See the bug:

    ValueError: unexpected character '[' at index 13 in
    'C[+42.010565][+57.0214635]LKMNYFSGK'

`format_encyclopedia_modseq` already *emits* the stacked form, so the reader
was the only asymmetry.
"""

from constellation.core.sequence.proforma import format_proforma
from constellation.massspec.io.encyclopedia._modseq import (
    format_encyclopedia_modseq,
    parse_encyclopedia_modseq,
)


def test_stacked_nterm_and_sidechain_on_residue_zero():
    # +42.010565 = Acetyl (N-term on C, which has no side-chain Acetyl site)
    # +57.0214635 = Carbamidomethyl (Cys side chain) — they split correctly.
    pf = parse_encyclopedia_modseq("C[+42.010565][+57.0214635]LKMNYFSGK")
    assert format_proforma(pf) == "[UNIMOD:1]-C[UNIMOD:4]LKMNYFSGK"
    assert len(pf.n_term_mods) == 1
    assert len(pf.residue_mods.get(0, ())) == 1


def test_stacked_brackets_roundtrip_through_writer():
    # Writer re-emits the stacked form; parsing it again is stable.
    s = "C[+42.010565][+57.0214635]LKMNYFSGK"
    pf = parse_encyclopedia_modseq(s)
    reparsed = parse_encyclopedia_modseq(format_encyclopedia_modseq(pf))
    assert format_proforma(reparsed) == format_proforma(pf)


def test_single_bracket_still_parses():
    # Guard against regressing the common single-mod-per-residue path.
    assert (
        format_proforma(parse_encyclopedia_modseq("PEPC[+57.02146]TIDE"))
        == "PEPC[UNIMOD:4]TIDE"
    )
