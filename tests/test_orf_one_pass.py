"""The one-pass ORF finder and translation give exactly what the forms they
replaced gave (2026-11-20, the EM performance review's A1/A2).

The references below are the replaced code, kept verbatim in substance:
``best_sense_orf`` translated every ATG's overlapped regex match and kept the
longest stop-free one per stop; demux enumerated every ``find_orfs`` match,
dropped those with an internal stop, kept the longest per stop and then the
longest; ``translate`` looked each codon up through ``expand_degenerate``.
"""

from __future__ import annotations

import numpy as np
import pytest

from constellation.core.sequence.nucleic import (
    CODON_TABLES,
    STANDARD,
    find_orfs,
    longest_orf,
    translate,
)
from constellation.sequencing.transcriptome.cluster.denovo.orf import (
    ORF_CODON_TABLE,
    best_sense_orf,
    is_low_complexity,
    orf_regex,
)


def _translate_reference(seq, *, codon_table=STANDARD, frame=0, to_stop=False,
                         treat_first_codon_as_start=False, partial="discard"):
    s = seq.upper().replace("U", "T")
    if frame:
        s = s[frame:]
    rem = len(s) % 3
    if rem:
        s = s + "N" * (3 - rem) if partial == "pad" else s[: len(s) - rem]
    out = []
    for i in range(0, len(s), 3):
        codon = s[i : i + 3]
        if i == 0 and treat_first_codon_as_start and codon in codon_table.starts:
            aa = "M"
        else:
            aa = codon_table.expand_degenerate(codon)
        if aa == "*" and to_stop:
            return "".join(out)
        out.append(aa)
    return "".join(out)


def _best_sense_orf_reference(seq, *, min_aa_length=30):
    if len(seq) < min_aa_length * 3 or is_low_complexity(seq):
        return None
    s = seq.upper().replace("U", "T")
    best = {}
    for m in orf_regex(min_aa_length).finditer(s, overlapped=True):
        nt = s[m.start() : m.end()]
        prot = _translate_reference(nt[:-3], codon_table=ORF_CODON_TABLE)
        if "*" in prot or len(prot) < min_aa_length:
            continue
        key = (m.start() % 3, m.end())
        prior = best.get(key)
        if prior is None or len(prot) > len(prior[0]):
            best[key] = (prot, m.start(), m.end())
    if not best:
        return None
    prot, st, en = max(best.values(), key=lambda x: len(x[0]))
    return prot, int(st), int(en)


def _demux_reference(transcript, *, codon_table, min_aa_length):
    clean = [
        o
        for o in find_orfs(
            transcript,
            codon_table=codon_table,
            min_aa_length=min_aa_length,
            both_strands=False,
            longest_per_stop=False,
        )
        if "*" not in o.protein
    ]
    if not clean:
        return None
    per_stop = {}
    for o in clean:
        prior = per_stop.get((o.strand, o.end))
        if prior is None or o.length > prior.length:
            per_stop[(o.strand, o.end)] = o
    return max(per_stop.values(), key=lambda o: o.length)


def _sequences(seed: int = 20261120, n: int = 4000) -> list[str]:
    """Random, stop-depleted (so long ORFs exist), with Ns, lowercase, U,
    low complexity, IUPAC codes, and short."""
    rng = np.random.default_rng(seed)
    sense = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
             if a + b + c not in ("TAA", "TAG", "TGA")]
    out = []
    for k in range(n):
        kind = k % 8
        length = int(rng.integers(20, 1500))
        if kind in (0, 1):
            s = "".join(rng.choice(list("ACGT"), size=length))
        else:
            codons = rng.choice(sense, size=length // 3)
            stops = rng.random(codons.size) < 0.01
            codons[stops] = rng.choice(["TAA", "TAG", "TGA"], size=int(stops.sum()))
            atg = rng.random(codons.size) < 0.03
            codons[atg] = "ATG"
            s = "".join(rng.choice(list("ACGT"), size=int(rng.integers(0, 3)))) + "".join(codons)
        if kind == 3:  # Ns and IUPAC codes
            arr = np.array(list(s))
            hit = rng.random(arr.size) < 0.004
            arr[hit] = rng.choice(list("NRYKM"), size=int(hit.sum()))
            s = "".join(arr)
        elif kind == 4:
            s = "".join(c.lower() if rng.random() < 0.5 else c for c in s)
        elif kind == 5:
            s = s.replace("T", "U")
        elif kind == 6:
            s = "A" * int(length * 0.85) + s[: int(length * 0.15)]
        elif kind == 7:
            s = s[: int(rng.integers(60, 200))]
        out.append(s)
    return out


SEQS = _sequences()


@pytest.mark.parametrize("floor", [30, 60])
def test_best_sense_orf_is_the_regex_form(floor) -> None:
    found = 0
    for s in SEQS:
        want = _best_sense_orf_reference(s, min_aa_length=floor)
        assert best_sense_orf(s, min_aa_length=floor) == want, s
        found += want is not None
    assert found > 400  # the corpus does exercise long ORFs


@pytest.mark.parametrize("table", [ORF_CODON_TABLE, STANDARD, CODON_TABLES[2]],
                         ids=["atg_only", "standard", "vert_mito"])
@pytest.mark.parametrize("floor", [30, 60])
def test_longest_orf_is_demux_s_enumerate_and_filter(table, floor) -> None:
    for s in SEQS[::2]:
        want = _demux_reference(s, codon_table=table, min_aa_length=floor)
        got = longest_orf(s, codon_table=table, min_aa_length=floor)
        assert got == want, s


@pytest.mark.parametrize("table", [STANDARD, CODON_TABLES[2], CODON_TABLES[11]],
                         ids=["standard", "vert_mito", "bacterial"])
def test_translate_is_the_codon_loop(table) -> None:
    rng = np.random.default_rng(7)
    for s in SEQS[::5]:
        for frame in (0, 1, 2):
            for to_stop in (False, True):
                for first in (False, True):
                    for partial in ("discard", "pad"):
                        kw = dict(codon_table=table, frame=frame, to_stop=to_stop,
                                  treat_first_codon_as_start=first, partial=partial)
                        assert translate(s, **kw) == _translate_reference(s, **kw), (s, kw)
    assert translate("", codon_table=table) == ""
    assert translate("AT", codon_table=table) == ""
    del rng
