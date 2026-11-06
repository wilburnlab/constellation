"""The JChronologer tokenizer against the tokens the jar itself makes.

``tests/data/jchronologer/1.0.10/`` holds the four preprocessing files
the tokenizer reads (copied from ``jchronologer-1.0.10.jar`` by
``scripts/build-jchronologer-models-json.py --test-data``) and
``tokens.tsv``, the jar's own tokens for a corpus of peptides
(``scripts/survey-jchronologer-tokens.py``, which runs the jar's
preprocessing code). Neither the jar nor torch is needed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from constellation.core.chem.modifications import UNIMOD
from constellation.core.sequence.proforma import parse_proforma
from constellation.massspec.tokenize import jchronologer as jc
from constellation.massspec.tokenize.vocabulary import (
    LENGTH_OUT_OF_RANGE,
    REASONS,
    UNKNOWN_RESIDUE,
    UNSUPPORTED_FORM,
    UNSUPPORTED_MODIFICATION,
    Sites,
    TokenizationRefused,
    Vocabulary,
)

VERSION = "1.0.10"
DATA = Path(__file__).parent / "data" / "jchronologer" / VERSION
MODELS = ("chronologer", "cartographer", "electrician", "sculptor")


def _files() -> dict[str, bytes]:
    return {m: next(DATA.glob(f"{m.capitalize()}_*.preprocessing.json")).read_bytes() for m in MODELS}


@pytest.fixture(scope="module")
def tokenizers():
    return jc.tokenizers_from_files(VERSION, _files())


def _rows():
    for line in (DATA / "tokens.tsv").read_text().splitlines():
        if line and not line.startswith(("#", "proforma\t")):
            proforma, jar_input, model, status, detail, width, tokens = line.split("\t")
            ids = [int(t) for t in tokens.split()]
            yield proforma, model, status, detail, ids + [0] * (int(width or 0) - len(ids))


def _spell(tokenizer, peptide):
    batch = tokenizer.encode([peptide])
    if batch.reasons[0] is not None:
        return None
    return tokenizer.vocabulary.decode(batch.tokens)[0]


# ── the jar's own tokens ────────────────────────────────────────────


def _sculptor_accessions() -> set[str]:
    sculptor = json.loads(_files()["sculptor"])
    return {e["unimod"] for e in sculptor["residue_unimod_map"]} | set(sculptor["nterm_unimod_map"])


def _divergence(proforma: str, model: str, detail: str) -> str | None:
    """Where Constellation's tokens differ from the jar's, on purpose."""
    if model == "sculptor" and detail.startswith("normalize: Unknown UNIMOD token"):
        accession = detail.split("'")[1]
        if accession in _sculptor_accessions():
            return "sculptor_vocabulary"  # in Sculptor's map; the jar cannot name it, so never sends it
    if model == "sculptor" and proforma in ("Q[UNIMOD:28]PEPTIDEK", "E[UNIMOD:27]PEPTIDEK"):
        return "pyro_on_first_residue"  # the jar folds it to the N-terminus for Cartographer only
    if model in ("chronologer", "cartographer") and proforma.startswith("K[UNIMOD:1]"):
        return "residue_zero_acetyl"  # patchModSequence reads it as N-terminal acetyl
    if model == "cartographer" and "-[UNIMOD:" in proforma and not detail.startswith("normalize"):
        return "c_terminal"  # the jar leaves it out of Cartographer's tokens
    return None


def test_the_test_data_is_the_snapshot():
    known = jc.load_snapshot()["versions"][VERSION]
    for model, data in _files().items():
        assert hashlib.sha256(data).hexdigest() == known["models"][model]["preprocessing_sha256"]
    header = (DATA / "tokens.tsv").read_text().splitlines()[1]
    assert known["jar_sha256"] in header


def test_tokens_are_the_jars(tokenizers):
    seen: dict[str, int] = {}
    compared = 0
    for proforma, model, status, detail, expected in _rows():
        tokenizer = tokenizers[model]
        batch = tokenizer.encode([proforma])
        ours = batch.reasons[0] is None
        kind = _divergence(proforma, model, detail)
        if kind is None:
            assert ours == (status == "accepted"), (proforma, model, batch.details[0], detail)
            if ours:
                assert batch.tokens[0].tolist() == expected, (proforma, model)
            compared += 1
            continue
        seen[kind] = seen.get(kind, 0) + 1
        spelled = tokenizer.vocabulary.decode(batch.tokens)[0] if ours else None
        if kind == "sculptor_vocabulary":
            assert status == "refused" and ours, proforma
        elif kind == "pyro_on_first_residue":
            assert status == "refused"
            n_terminal = "[UNIMOD:28]-QPEPTIDEK" if proforma.startswith("Q") else "[UNIMOD:27]-EPEPTIDEK"
            assert spelled == _spell(tokenizer, n_terminal)
        elif kind == "residue_zero_acetyl":
            assert detail.startswith("^K") and spelled.startswith("-a"), (proforma, detail, spelled)
        elif kind == "c_terminal":
            assert status == "accepted" and batch.reasons[0] == UNSUPPORTED_MODIFICATION
    assert set(seen) == {"sculptor_vocabulary", "pyro_on_first_residue", "residue_zero_acetyl", "c_terminal"}
    assert compared > 750


def test_electrician_reads_cartographers_tokens(tokenizers):
    peptides = sorted({p for p, *_ in _rows()})
    a = tokenizers["cartographer"].encode(peptides)
    b = tokenizers["electrician"].encode(peptides)
    assert np.array_equal(a.tokens, b.tokens) and a.reasons == b.reasons


def test_the_jars_masses_are_unimods():
    for accession, mass in jc.JAR_UNIMOD_MASSES.items():
        assert UNIMOD[accession].delta_mass == pytest.approx(mass, abs=1e-6), accession


def test_the_regex_spelling_agrees_with_the_accession_maps(tokenizers):
    """Every (residue, accession) Sculptor's map names and the jar can
    spell (all but TMT, which Sculptor has no token for) gets the same
    character from Cartographer's regular expressions."""
    sculptor = json.loads(_files()["sculptor"])
    checked = 0
    for entry in sculptor["residue_unimod_map"]:
        if entry["unimod"] not in jc.JAR_UNIMOD_MASSES:
            continue
        spelled = _spell(tokenizers["cartographer"], f"GLA{entry['residue']}[{entry['unimod']}]TEFAK")
        assert spelled == f"-GLA{entry['token']}TEFAK_", entry
        checked += 1
    assert checked == 20


# ── terminal forms ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("peptide", "chronologer", "cartographer", "sculptor"),
    [
        ("[UNIMOD:28]-QPEPTIDEK", "(ePEPTIDEK_", "(QPEPTIDEK_", "(QPEPTIDEK_"),
        ("[UNIMOD:27]-EPEPTIDEK", "(ePEPTIDEK_", ")EPEPTIDEK_", ")EPEPTIDEK_"),
        ("[UNIMOD:26]-CPEPTIDEK", ")dPEPTIDEK_", "(cPEPTIDEK_", "(cPEPTIDEK_"),
        ("[UNIMOD:1]-KVERPDK", "^KVERPDK_", "^KVERPDK_", "^KVERPDK_"),
        ("K[UNIMOD:1]VERPDK", "-aVERPDK_", "-aVERPDK_", "-aVERPDK_"),
        ("[UNIMOD:1]-K[UNIMOD:1]VERPDK", "^aVERPDK_", "^aVERPDK_", "^aVERPDK_"),
        ("[UNIMOD:737]-PEPTK[UNIMOD:737]IDEK", "*PEPTxIDEK_", "*PEPTxIDEK_", None),
    ],
)
def test_terminal_forms(tokenizers, peptide, chronologer, cartographer, sculptor):
    assert _spell(tokenizers["chronologer"], peptide) == chronologer
    assert _spell(tokenizers["cartographer"], peptide) == cartographer
    assert _spell(tokenizers["sculptor"], peptide) == sculptor


# ── refusals ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "peptide", "reason"),
    [
        ("cartographer", "A" * 31 + "K", LENGTH_OUT_OF_RANGE),
        ("chronologer", "A" * 50 + "K", LENGTH_OUT_OF_RANGE),
        ("sculptor", "A" * 50 + "K", LENGTH_OUT_OF_RANGE),
        ("chronologer", "PEPTI", LENGTH_OUT_OF_RANGE),
        ("cartographer", "PEPTI", LENGTH_OUT_OF_RANGE),
        ("chronologer", "PEPTW[UNIMOD:35]IDEK", UNSUPPORTED_MODIFICATION),
        ("chronologer", "PEPTN[UNIMOD:7]IDEK", UNSUPPORTED_MODIFICATION),
        ("chronologer", "PEPTN[UNIMOD:43]IDEK", UNSUPPORTED_MODIFICATION),
        ("cartographer", "PEPTUDEK", UNKNOWN_RESIDUE),
        ("sculptor", "PEPTXDEK", UNKNOWN_RESIDUE),
        ("cartographer", "[UNIMOD:58]-PEPTIDEK", UNSUPPORTED_MODIFICATION),
        ("cartographer", "PEPTK[UNIMOD:280]IDEK", UNSUPPORTED_MODIFICATION),  # Ethyl, spelled like Dimethyl
        ("sculptor", "PEPTK[UNIMOD:280]IDEK", UNSUPPORTED_MODIFICATION),
        ("cartographer", "[UNIMOD:1][UNIMOD:737]-PEPTIDEK", UNSUPPORTED_MODIFICATION),
        ("sculptor", "[UNIMOD:1][UNIMOD:737]-PEPTIDEK", UNSUPPORTED_MODIFICATION),
        ("cartographer", "PEPTK[UNIMOD:1][UNIMOD:121]IDEK", UNSUPPORTED_MODIFICATION),
        ("chronologer", "PEPTIDEK-[UNIMOD:2]", UNSUPPORTED_MODIFICATION),
        ("sculptor", "PEPTIDEK-[UNIMOD:2]", UNSUPPORTED_MODIFICATION),
        ("cartographer", "PEPTM[+15.995]IDEK", UNSUPPORTED_MODIFICATION),
        ("cartographer", "PEPTM[Oxidation]IDEK", UNSUPPORTED_MODIFICATION),
        ("cartographer", "<13C>PEPTIDEK", UNSUPPORTED_FORM),
        ("sculptor", "<15N>PEPTIDEK", UNSUPPORTED_FORM),
        ("cartographer", "[UNIMOD:21]?PEPTSIDEK", UNSUPPORTED_FORM),
    ],
)
def test_refusals(tokenizers, model, peptide, reason):
    batch = tokenizers[model].encode([peptide])
    assert batch.reasons == (reason,), batch.details
    assert batch.details[0]
    assert not batch.tokens.any()
    assert not batch.accepted[0]


def test_refusals_are_counted_by_stable_reasons(tokenizers):
    batch = tokenizers["chronologer"].encode(["PEPTIDEK", "PEPTI", "PEPTW[UNIMOD:35]IDEK", "PEPTUDEK"])
    assert batch.reasons == (None, LENGTH_OUT_OF_RANGE, UNSUPPORTED_MODIFICATION, UNKNOWN_RESIDUE)
    assert batch.accepted.tolist() == [True, False, False, False]
    assert set(r for r in batch.reasons if r) <= set(REASONS)
    with pytest.raises(ValueError):
        TokenizationRefused("because", "no")


# ── files outside the snapshot ───────────────────────────────────────


def test_a_file_outside_the_snapshot_is_refused():
    files = _files()
    files["chronologer"] = files["chronologer"].replace(b'"max_peptide_len": 50', b'"max_peptide_len": 60')
    with pytest.raises(jc.UnknownModelError, match="build-jchronologer-models-json"):
        jc.tokenizers_from_files(VERSION, files)
    with pytest.raises(jc.UnknownModelError, match="not one Constellation was checked against"):
        jc.tokenizers_from_files("9.9.9", _files())


def test_electrician_must_read_cartographers_characters(monkeypatch):
    files = _files()
    electrician = json.loads(files["electrician"])
    electrician["aa_to_int"]["Z"] = 99
    files["electrician"] = json.dumps(electrician).encode()
    snapshot = json.loads(json.dumps(jc.load_snapshot()))
    snapshot["versions"][VERSION]["models"]["electrician"]["preprocessing_sha256"] = hashlib.sha256(
        files["electrician"]
    ).hexdigest()
    monkeypatch.setattr(jc, "load_snapshot", lambda: snapshot)
    with pytest.raises(jc.UnknownModelError, match="Electrician's aa_to_int differs"):
        jc.tokenizers_from_files(VERSION, files)


# ── layout ───────────────────────────────────────────────────────────


def test_rows_are_padded_to_the_models_width(tokenizers):
    for model, width in (("chronologer", 52), ("cartographer", 33), ("sculptor", 52)):
        tokenizer = tokenizers[model]
        assert tokenizer.vocabulary.width == width
        batch = tokenizer.encode(["PEPTIDEK", "PEPTI" * 20, "A" * 30 + "K"])
        assert batch.tokens.shape == (3, width) and batch.tokens.dtype == np.int64
        ids = tokenizer.vocabulary.ids
        first = batch.tokens[0].tolist()
        assert first[:10] == [ids[c] for c in "-PEPTIDEK_"]
        assert first[10:] == [0] * (width - 10)
        assert not batch.tokens[1].any()


def test_encode_of_nothing():
    vocabulary = Vocabulary(ids={"A": 1, "-": 2, "_": 3}, max_length=4)
    assert vocabulary.encode([]).shape == (0, 6)
    assert not vocabulary.encode([None, None]).any()
    assert vocabulary.encode(["-AA_"]).tolist() == [[2, 1, 1, 3, 0, 0]]
    assert vocabulary.decode(vocabulary.encode(["-AA_"])) == ["-AA_"]
    with pytest.raises(ValueError, match="not in the vocabulary"):
        vocabulary.encode(["-AB_"])
    with pytest.raises(ValueError, match="longer than the width"):
        vocabulary.encode(["-AAAAA_"])
    with pytest.raises(ValueError, match="padding id"):
        Vocabulary(ids={"A": 0}, max_length=4)


def test_sites_are_one_per_molecule():
    a = Sites.from_peptidoform("PEPTK[UNIMOD:121][UNIMOD:1]IDEK")
    b = Sites.from_peptidoform(parse_proforma("PEPTK[UNIMOD:1][UNIMOD:121]IDEK"))
    assert a == b and a.residue_mods[4] == ("UNIMOD:1", "UNIMOD:121")
    assert Sites.from_peptidoform("PEPTIDEK/2") == Sites.from_peptidoform("PEPTIDEK")
    with pytest.raises(TokenizationRefused, match="cross-link"):
        Sites.from_peptidoform("PEPK[UNIMOD:1#XL1]TIDEK[#XL1]")
    with pytest.raises(TokenizationRefused, match="more than one chain"):
        Sites.from_peptidoform("PEPTIDEK//PEPTIDER")


def test_the_canonical_reader_is_the_parser():
    """Canonical strings read straight to sites give what parsing them
    as ProForma gives; anything else falls through to the parser."""
    corpus = {p for p, *_ in _rows()} | {
        "PEPTK[UNIMOD:121][UNIMOD:1]IDEK", "[UNIMOD:737][UNIMOD:1]-PEPTIDEK", "PEPTIDEK-[UNIMOD:2]",
        "[UNIMOD:1]-PEPTIDEK-[UNIMOD:2]", "M[UNIMOD:35]", "K",
    }
    for text in corpus:
        fast = Sites._from_canonical(text)
        assert fast is not None, text
        assert fast == Sites.from_peptidoform(parse_proforma(text)), text
    for text in (
        "<13C>PEPTIDEK", "PEPTM[Oxidation]IDEK", "PEPTM[+15.995]IDEK", "[UNIMOD:21]?PEPTSIDEK",
        "PEPTIDEK/2", "PEPTS[UNIMOD:21#g1]IDES[#g1]K", "peptidek", "PEPTIDEK-", "[UNIMOD:1]PEPTIDEK",
        "[UNIMOD:1]-[UNIMOD:2]", "{UNIMOD:43}PEPTIDEK", "PEP(TI)[UNIMOD:21]DEK", "PEPTIDEK//PEPTIDER",
        "[UNIMOD:1]-PEPTIDEK-[Amidated]",
    ):
        assert Sites._from_canonical(text) is None, text
