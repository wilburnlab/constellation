"""Content-derived ids: one number per molecule, whoever computes it."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pyarrow as pa
import pytest

from constellation.core.sequence.proforma import format_proforma, parse_proforma
from constellation.massspec.io.encyclopedia._modseq import (
    encyclopedia_residue_form,
    parse_encyclopedia_modseq,
)
from constellation.massspec.library import assign_ids
from constellation.massspec.library.ids import (
    ID_SCHEME,
    ID_SCHEME_METADATA_KEY,
    IdCollisionError,
    NoCanonicalKeyError,
    canonical_peptide_key,
    canonical_precursor_key,
    canonical_protein_key,
    content_id,
    content_ids,
    peptide_id,
    precursor_id,
    protein_id,
    with_content_ids,
)

# ── the recipe is fixed ────────────────────────────────────────────────


def test_the_id_recipe_is_pinned() -> None:
    """These numbers are what `content-v1` means. If one changes, ids
    already written to disk no longer match: bump ``ID_SCHEME`` instead."""
    assert ID_SCHEME == "content-v1"
    assert content_id("PEPTIDEK") == 1792753019647354276
    assert peptide_id("PEPC[UNIMOD:4]TIDEK") == 2515000382947725677
    assert precursor_id("PEPC[UNIMOD:4]TIDEK", 2) == 6067920219105794426
    assert protein_id("P12345") == 4585892463021593304
    assert all(0 <= content_id(k) < 2**63 for k in ("", "A", "PEPTIDEK/2", "[UNIMOD:1]-K"))
    # The recipe, spelled out independently of the module.
    import hashlib

    digest = hashlib.blake2b(b"PEPTIDEK", digest_size=8).digest()
    assert content_id("PEPTIDEK") == int.from_bytes(digest, "big") & (2**63 - 1)


# ── canonical keys ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spelling", "key"),
    [
        ("PEPTIDEK", "PEPTIDEK"),
        ("PEPM[Oxidation]TIDEK", "PEPM[UNIMOD:35]TIDEK"),  # name -> accession
        ("PEPM[UNIMOD:35]TIDEK", "PEPM[UNIMOD:35]TIDEK"),
        ("PEPM[U:Oxidation]TIDEK", "PEPM[UNIMOD:35]TIDEK"),
        ("[Acetyl]-PEPTIDEK", "[UNIMOD:1]-PEPTIDEK"),
        ("PEPTIDEK/2", "PEPTIDEK"),  # charge is not peptide identity
        ("(>label)PEPTIDEK", "PEPTIDEK"),
        ("PEPC[+57.02146]TIDEK", "PEPC[+57.0215]TIDEK"),  # a bare mass stays a mass,
        ("PEPC[+57.0214635]TIDEK", "PEPC[+57.0215]TIDEK"),  # at one precision
        ("PEPQ[-17.026549]K", "PEPQ[-17.0265]K"),
        ("<13C>PEPTIDEK", "<13C>PEPTIDEK"),
    ],
)
def test_spellings_of_one_peptide_share_a_key(spelling: str, key: str) -> None:
    assert canonical_peptide_key(spelling) == key
    # The key is ProForma, and is its own key.
    assert canonical_peptide_key(key) == key
    assert canonical_peptide_key(parse_proforma(spelling)) == key
    assert format_proforma(parse_proforma(key)) == key


def test_modifications_on_one_site_have_a_fixed_order() -> None:
    a = canonical_peptide_key("C[UNIMOD:4][UNIMOD:35]PEPK")
    b = canonical_peptide_key("C[UNIMOD:35][UNIMOD:4]PEPK")
    assert a == b


def test_terminal_and_side_chain_placement_are_different_molecules() -> None:
    n_terminal, lysine = "[UNIMOD:1]-KVERPD", "K[UNIMOD:1]VERPD"
    assert canonical_peptide_key(n_terminal) != canonical_peptide_key(lysine)
    assert peptide_id(n_terminal) != peptide_id(lysine)


@pytest.mark.parametrize(
    "spelling",
    [
        "PEPS[Phospho#g1]TIDES[#g1]K",  # ambiguous localisation
        "[Phospho]?PEPSTIDESK",  # unknown position
        "PEP(STIDE)[Phospho]SK",  # a range
        "<[Carbamidomethyl]@C>PEPCK",  # global fixed modification
        "PEPTIDEK//KEDITPEP",  # two chains
        "PEPT[INFO:odd]IDEK",
    ],
)
def test_what_names_no_single_molecule_has_no_key(spelling: str) -> None:
    with pytest.raises(NoCanonicalKeyError):
        canonical_peptide_key(spelling)


def test_precursor_key_is_the_peptide_key_plus_charge() -> None:
    assert canonical_precursor_key("PEPM[Oxidation]TIDEK", 2) == "PEPM[UNIMOD:35]TIDEK/2"
    assert canonical_precursor_key("PEPM[Oxidation]TIDEK/3") == "PEPM[UNIMOD:35]TIDEK/3"
    assert canonical_precursor_key(parse_proforma("PEPTIDEK/2")) == "PEPTIDEK/2"
    assert precursor_id("PEPTIDEK", 2) != precursor_id("PEPTIDEK", 3) != peptide_id("PEPTIDEK")
    with pytest.raises(ValueError, match="needs a charge"):
        canonical_precursor_key("PEPTIDEK")
    with pytest.raises(ValueError, match="carries /2"):
        canonical_precursor_key("PEPTIDEK/2", 3)
    with pytest.raises(ValueError, match="cannot be 0"):
        canonical_precursor_key("PEPTIDEK", 0)


def test_protein_key_is_the_accession_as_given() -> None:
    assert canonical_protein_key("  P12345\n") == "P12345"
    assert canonical_protein_key("sp|P12345|ALBU_HUMAN") != canonical_protein_key("P12345")
    with pytest.raises(NoCanonicalKeyError):
        canonical_protein_key("   ")


def test_content_ids_keeps_order_and_notices_a_collision(monkeypatch) -> None:
    from constellation.massspec.library import ids

    keys = ["PEPTIDEK", "ELVISK", "PEPTIDEK"]
    out = content_ids(keys)
    assert out.type == pa.int64()
    assert out.to_pylist() == [content_id(k) for k in keys]

    monkeypatch.setattr(ids, "content_id", lambda key: 7)
    with pytest.raises(IdCollisionError, match="'PEPTIDEK' and 'ELVISK'"):
        ids.content_ids(keys)


# ── the two EncyclopeDIA spellings ─────────────────────────────────────


@pytest.mark.parametrize(
    "entries_spelling",
    [
        "[+42.010565]MDFQHRPGGK",
        "[-17.026549]QLSALHR",
        "[+42.010565]C[+57.0214635]DFTEDK",
        "PEPC[+57.0214635]TIDEK",
        "PEPTIDEK",
    ],
)
def test_both_table_spellings_of_a_result_file_get_one_id(entries_spelling: str) -> None:
    """``entries`` and ``peptidequants`` spell N-terminal modifications
    differently; read through the EncyclopeDIA parser they are one
    peptide, hence one id."""
    quant_spelling = encyclopedia_residue_form(entries_spelling)
    a = format_proforma(parse_encyclopedia_modseq(entries_spelling))
    b = format_proforma(parse_encyclopedia_modseq(quant_spelling))
    assert peptide_id(a) == peptide_id(b)
    assert precursor_id(a, 2) == precursor_id(b, 2)


# ── re-keying a Library ────────────────────────────────────────────────


def _toy_library(peptides=("PEPTIDEK", "PEPM[Oxidation]TIDEK")):
    return assign_ids(
        proteins=[{"accession": "P1"}, {"accession": "P2"}],
        peptides=[{"modified_sequence": m} for m in peptides],
        precursors=[
            {"modified_sequence": m, "charge": z, "precursor_mz": 500.0 + z}
            for m in peptides
            for z in (2, 3)
        ],
        fragments=[
            {
                "modified_sequence": peptides[0],
                "precursor_charge": 2,
                "ion_type": 1,
                "position": 3,
                "charge": 1,
                "mz_theoretical": 300.0,
                "intensity_predicted": 1.0,
            }
        ],
        protein_peptide=[("P1", peptides[0]), ("P2", peptides[0]), ("P1", peptides[1])],
    )


def test_with_content_ids_rekeys_every_table_consistently() -> None:
    library = with_content_ids(_toy_library())
    assert library.metadata[ID_SCHEME_METADATA_KEY] == ID_SCHEME

    peptides = library.peptides.to_pylist()
    assert [p["modified_sequence"] for p in peptides] == ["PEPTIDEK", "PEPM[UNIMOD:35]TIDEK"]
    # An id can be checked against the row it sits on.
    assert all(p["peptide_id"] == content_id(p["modified_sequence"]) for p in peptides)
    assert library.proteins.column("protein_id").to_pylist() == [
        protein_id("P1"),
        protein_id("P2"),
    ]
    key = {p["peptide_id"]: p["modified_sequence"] for p in peptides}
    for row in library.precursors.to_pylist():
        assert row["precursor_id"] == content_id(f"{key[row['peptide_id']]}/{row['charge']}")
    assert library.fragments.column("precursor_id").to_pylist() == [precursor_id("PEPTIDEK", 2)]
    assert sorted(library.protein_peptide.to_pylist(), key=lambda r: (r["protein_id"], r["peptide_id"])) == sorted(
        [
            {"protein_id": protein_id("P1"), "peptide_id": peptide_id("PEPTIDEK")},
            {"protein_id": protein_id("P2"), "peptide_id": peptide_id("PEPTIDEK")},
            {"protein_id": protein_id("P1"), "peptide_id": peptide_id("PEPM[UNIMOD:35]TIDEK")},
        ],
        key=lambda r: (r["protein_id"], r["peptide_id"]),
    )
    library.validate()  # keys still unique, foreign keys still closed


def test_two_libraries_built_separately_agree_on_ids() -> None:
    """The point of the exercise: no shared library, same numbers."""
    one = with_content_ids(_toy_library(("PEPTIDEK", "PEPM[Oxidation]TIDEK")))
    other = with_content_ids(_toy_library(("PEPM[UNIMOD:35]TIDEK", "ELVISK", "PEPTIDEK")))
    shared = set(one.precursors.column("precursor_id").to_pylist()) & set(
        other.precursors.column("precursor_id").to_pylist()
    )
    assert shared == {
        precursor_id(m, z) for m in ("PEPTIDEK", "PEPM[UNIMOD:35]TIDEK") for z in (2, 3)
    }


def test_with_content_ids_refuses_two_spellings_of_one_peptide() -> None:
    library = _toy_library(("PEPM[Oxidation]TIDEK", "PEPM[UNIMOD:35]TIDEK"))
    with pytest.raises(ValueError, match="two peptide rows are the same peptide"):
        with_content_ids(library)


# ── every modified sequence in real files, when some are available ────

_SAMPLE_DIR = Path(
    os.environ.get(
        "CONSTELLATION_ENCYCLOPEDIA_SAMPLES",
        str(Path.home() / "WilburnLab/MS_Quant/test_encyclopedia_files"),
    )
)
_EXTRA = [Path(p) for p in os.environ.get("CONSTELLATION_ENCYCLOPEDIA_RESULTS", "").split(os.pathsep) if p]


def _sample_files() -> list[Path]:
    found = list(_EXTRA)
    if _SAMPLE_DIR.is_dir():
        found += sorted(p for p in _SAMPLE_DIR.iterdir() if p.suffix in (".dlib", ".elib"))
    return [p for p in found if p.is_file()]


@pytest.mark.slow
@pytest.mark.skipif(
    not _sample_files(),
    reason="no .dlib/.elib under $CONSTELLATION_ENCYCLOPEDIA_SAMPLES or in "
    "$CONSTELLATION_ENCYCLOPEDIA_RESULTS",
)
def test_every_sample_modified_sequence_has_a_stable_key() -> None:
    """Over every distinct modified sequence in the sample files (all the
    modified ones, a slice of the unmodified): the key is valid ProForma,
    is its own key, is what both table spellings give, and no two keys
    share an id."""
    modseqs: set[str] = set()
    for path in _sample_files():
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            modseqs.update(
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT PeptideModSeq FROM entries WHERE PeptideModSeq LIKE '%[%'"
                )
            )
            modseqs.update(
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT PeptideModSeq FROM entries "
                    "WHERE PeptideModSeq NOT LIKE '%[%' LIMIT 20000"
                )
            )
        finally:
            con.close()
    assert modseqs

    from constellation.massspec.library.ids import _RE_ALREADY_CANONICAL

    keys: dict[str, str] = {}
    for modseq in sorted(modseqs):
        key = canonical_peptide_key(format_proforma(parse_encyclopedia_modseq(modseq)))
        assert canonical_peptide_key(key) == key, modseq
        other = encyclopedia_residue_form(modseq)
        other_key = canonical_peptide_key(format_proforma(parse_encyclopedia_modseq(other)))
        if other_key != key:
            # The one case the residue-attached spelling cannot express: a
            # modification that fits both the N-terminus and the first
            # residue's side chain.
            assert modseq.startswith("["), modseq
            assert parse_encyclopedia_modseq(other).residue_mods.get(0), modseq
        keys[key] = modseq

    # The full ProForma parser is slow (milliseconds a string), so the
    # round trip through it is checked for every key that is not of the
    # plain accession-only shape, and for a slice of those that are.
    plain = [k for k in keys if _RE_ALREADY_CANONICAL.fullmatch(k)]
    unusual = [k for k in keys if not _RE_ALREADY_CANONICAL.fullmatch(k)]
    for key in unusual + plain[:300]:
        assert format_proforma(parse_proforma(key)) == key, keys[key]
        assert canonical_peptide_key(parse_proforma(key)) == key, keys[key]
    content_ids(list(keys))  # raises on a collision
