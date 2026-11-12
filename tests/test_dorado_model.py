"""``DoradoModel``: Dorado's model argument, checked and passed through."""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.sequencing.basecall.models import DoradoModel


@pytest.mark.parametrize(
    "spec, mods",
    [
        ("sup", ()),
        ("hac", ()),
        ("fast@latest", ()),
        ("hac@v5", ()),
        ("hac@v4.3", ()),  # Dorado reads a missing trailing part as zero
        ("sup@v5.2.0", ()),
        ("sup,5mCG_5hmCG", ("5mCG_5hmCG",)),
        ("sup@latest,6mA,5mC@v2", ("6mA", "5mC")),
        ("hac@v6.0.0,5mCG_5hmCG@v1.0.0", ("5mCG_5hmCG",)),
        ("sup,m6A_DRACH@latest", ("m6A_DRACH",)),
    ],
)
def test_variants_pass_through(spec: str, mods: tuple[str, ...]):
    model = DoradoModel.parse(spec)
    assert (model.spec, model.kind, model.mods) == (spec, "variant", mods)
    assert str(model) == spec


@pytest.mark.parametrize(
    "spec",
    [
        "dna_r10.4.1_e8.2_400bps_sup@v5.2.0",
        "dna_r10.4.1_e8.2_400bps_hac@v6.0.0",
        "dna_r10.4.1_e8.2_400bps_hac@v6.0.0_5mC_5hmC@v1",
        "rna004_130bps_sup@v6.0.0",
    ],
)
def test_full_names_pass_through(spec: str):
    model = DoradoModel.parse(spec)
    assert (model.spec, model.kind, model.mods) == (spec, "name", ())


def test_a_model_directory(tmp_path: Path):
    directory = tmp_path / "dna_r10.4.1_e8.2_400bps_hac@v6.0.0"
    directory.mkdir()
    model = DoradoModel.parse(str(directory))
    assert (model.spec, model.kind) == (str(directory.resolve()), "path")


def test_a_variant_wins_over_a_directory_of_that_name(tmp_path: Path, monkeypatch):
    (tmp_path / "sup").mkdir()
    monkeypatch.chdir(tmp_path)
    assert DoradoModel.parse("sup").kind == "variant"
    assert DoradoModel.parse("./sup").kind == "path"


def test_old_shorthand_is_refused_with_its_replacement():
    with pytest.raises(ValueError, match=r"hac@v4\.3\.0,5mC_5hmC"):
        DoradoModel.parse("hac@v4.3+5mC,5hmC")


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "super",
        "sup@5.0.0",  # no v
        "sup@v1.",  # Dorado refuses a trailing period
        "sup@v0..1",  # and an empty internal part
        "sup,,6mA",
        "/no/such/model/dir",
    ],
)
def test_anything_else_is_refused(spec: str):
    with pytest.raises(ValueError, match="not a Dorado model"):
        DoradoModel.parse(spec)


@pytest.mark.parametrize(
    "spec, mods, expected",
    [
        ("sup", (), ("sup", [])),
        ("sup", ("5mCG_5hmCG",), ("sup,5mCG_5hmCG", [])),
        (
            "dna_r10.4.1_e8.2_400bps_sup@v5.2.0",
            ("6mA", "5mC_5hmC"),
            ("dna_r10.4.1_e8.2_400bps_sup@v5.2.0", ["--modified-bases", "6mA", "5mC_5hmC"]),
        ),
        ("sup,6mA", (), ("sup,6mA", [])),
    ],
)
def test_arguments(spec, mods, expected):
    assert DoradoModel.parse(spec).arguments(mods) == expected


def test_arguments_refuse_modifications_named_twice():
    with pytest.raises(ValueError, match="one place"):
        DoradoModel.parse("sup,6mA").arguments(("5mCG_5hmCG",))
