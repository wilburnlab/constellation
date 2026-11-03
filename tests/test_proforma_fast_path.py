"""`parse_proforma`'s fast path for the simple form must give exactly what
the grammar gives, and leave everything else to the grammar."""

from __future__ import annotations

import random

import pytest

from constellation.core.sequence.proforma import (
    ProFormaSyntaxError,
    _parse_simple,
    _parse_with_grammar,
    parse_proforma,
)

_RESIDUES = "ACDEFGHIKLMNPQRSTVWYXBZUOJ"
_MODS = ["UNIMOD:35", "UNIMOD:4", "UNIMOD:1", "UNIMOD:28", "Oxidation", "Carbamidomethyl", "Phospho",
         "Acetyl", "Gln-pyro-Glu", "Amidated", "U", "X_y", "+42.010565", "-17.026549", "+57.0214635",
         "+1", "-0.98", "+0"]  # fmt: skip


def _simple_strings(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)

    def mods(k: int) -> str:
        return "".join(f"[{rng.choice(_MODS)}]" for _ in range(k))

    out = []
    for _ in range(n):
        labels = "".join(rng.sample(["<13C>", "<15N>", "<18O>", "<2H>", "<D>", "<T>"], rng.randint(0, 2)))
        nterm = mods(rng.choice([0, 0, 0, 1, 2]))
        body = "".join(rng.choice(_RESIDUES) + mods(rng.choice([0] * 6 + [1, 2, 3]))
                       for _ in range(rng.randint(1, 40)))  # fmt: skip
        cterm = mods(rng.choice([0, 0, 0, 1]))
        charge = rng.choice(["", "", f"/{rng.randint(1, 9)}"])
        out.append(labels + (nterm + "-" if nterm else "") + body + ("-" + cterm if cterm else "") + charge)
    return out


def test_the_fast_path_gives_what_the_grammar_gives():
    strings = _simple_strings(600)
    for s in strings:
        fast = _parse_simple(s)
        assert fast is not None, s
        assert fast == _parse_with_grammar(s), s


@pytest.mark.parametrize(
    "s",
    [
        "pepk",
        "PEP[U:Oxidation]K",
        "PEP[U:+15.995]K",
        "[Formula:C2H2]-PEPK",
        "PEPT[INFO:hello]K",
        "PEP(TI)[+1]DE",
        "PEPT[Phospho#g1]IDES[#g1]K",
        "PEPK/+2",
        "PEP[57.02]K",
        "PEP[MOD:00046]K",
        "PEP[Obs:+1.003]K",
        "<[Carbamidomethyl]@C>PEPCK",
        "[Phospho]?PEPTIDE",
        "{Glycan:Hex}PEPTIDE",
        "PEP[Oxidation|INFO:x]K",
        "EMEVTK[XLMOD:02001#XL1]SESPEK[#XL1]",
        "PEPTIDE[ Oxidation ]K",
    ],
)
def test_everything_else_goes_to_the_grammar(s):
    assert _parse_simple(s) is None
    assert parse_proforma(s) == _parse_with_grammar(s)


@pytest.mark.parametrize("s", ["PEP]K", "PEP[K", "PEPK-", "-PEPK", "PEPK//"])
def test_bad_input_still_fails(s):
    with pytest.raises(ProFormaSyntaxError):
        parse_proforma(s)
