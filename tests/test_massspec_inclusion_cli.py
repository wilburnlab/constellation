"""``constellation massspec inclusion-list`` end to end."""

from __future__ import annotations

import argparse
import json

import pytest

from constellation.massspec.cli import (
    _parse_mod_specs,
    _protease_ids,
    build_parser,
)

_FASTA = """\
>sp|TEST1|ONE Test protein one
MKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKMDEMKQPEPTIDEK
>sp|TEST2|TWO Test protein two
MKLVNELTEFAKDDSPDLPKSLGKVGTR
"""


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="command", required=True)
    build_parser(subs)
    args = parser.parse_args(argv)
    return args.func(args)


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "targets.fasta"
    path.write_text(_FASTA)
    return path


def _invoke(fasta, out_dir, *extra: str) -> int:
    return _run(
        [
            "massspec",
            "inclusion-list",
            "--fasta",
            str(fasta),
            "--output-dir",
            str(out_dir),
            *extra,
        ]
    )


# ── happy path ─────────────────────────────────────────────────────────


def test_happy_path_writes_csv_parquet_manifest(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--missed-cleavages", "0") == 0

    csv_path = out / "inclusion_list.csv"
    assert csv_path.read_bytes().split(b"\r\n")[0] == b"Compound,m/z"
    assert (out / "precursors.parquet").is_file()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tool"] == "constellation massspec inclusion-list"
    assert manifest["params"]["protease"] == "Trypsin"
    assert manifest["params"]["missed_cleavages"] == 0
    assert manifest["params"]["charges"] == [2, 3, 4]
    assert manifest["params"]["fixed_mods"] == {"C": "UNIMOD:4"}
    assert manifest["inputs"]["fasta"]["sha256"]
    # `rows` is what the CSV holds; `entries` is the pre-merge precursor
    # count, which merging can make larger.
    assert manifest["counts"]["rows"] == len(csv_path.read_text().splitlines()) - 1
    assert manifest["counts"]["entries"] >= manifest["counts"]["rows"]


def test_sidecar_can_be_skipped(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--no-sidecar") == 0
    assert not (out / "precursors.parquet").exists()
    assert json.loads((out / "manifest.json").read_text())["outputs"][
        "precursors_parquet"
    ] is None


def test_output_name_override(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--output-name", "epha3_targets.csv") == 0
    assert (out / "epha3_targets.csv").is_file()
    assert not (out / "inclusion_list.csv").exists()


def test_mz_decimals_flows_through(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--mz-decimals", "1") == 0
    for line in (out / "inclusion_list.csv").read_text().splitlines()[1:]:
        assert len(line.split(",")[1].split(".")[1]) == 1


def test_default_mz_decimals_is_three(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out) == 0
    for line in (out / "inclusion_list.csv").read_text().splitlines()[1:]:
        assert len(line.split(",")[1].split(".")[1]) == 3


# ── missed-cleavage band ───────────────────────────────────────────────


def _compounds(out) -> set[str]:
    return {
        line.split(",")[0]
        for line in (out / "inclusion_list.csv").read_text().splitlines()[1:]
    }


def test_min_missed_cleavages_selects_only_that_band(fasta, tmp_path) -> None:
    """``--min-missed-cleavages 1 --missed-cleavages 1`` lists only the
    singly-missed peptides — the fully-cleaved ones drop out entirely."""
    zero, upto_one, only_one = tmp_path / "z", tmp_path / "u", tmp_path / "o"
    assert _invoke(fasta, zero, "--missed-cleavages", "0") == 0
    assert _invoke(fasta, upto_one, "--missed-cleavages", "1") == 0
    assert (
        _invoke(
            fasta, only_one, "--missed-cleavages", "1", "--min-missed-cleavages", "1"
        )
        == 0
    )

    fully, ceiling, band = _compounds(zero), _compounds(upto_one), _compounds(only_one)
    assert band, "expected at least one singly-missed peptide"
    # The band is exactly what the ceiling adds over the fully-cleaved set.
    assert band == ceiling - fully
    assert not (band & fully)


def test_min_missed_cleavages_defaults_to_zero(fasta, tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    assert _invoke(fasta, a, "--missed-cleavages", "1") == 0
    assert _invoke(fasta, b, "--missed-cleavages", "1", "--min-missed-cleavages", "0") == 0
    assert _compounds(a) == _compounds(b)


def test_min_missed_cleavages_recorded_in_manifest(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert (
        _invoke(fasta, out, "--missed-cleavages", "2", "--min-missed-cleavages", "2")
        == 0
    )
    params = json.loads((out / "manifest.json").read_text())["params"]
    assert params["min_missed_cleavages"] == 2
    assert params["missed_cleavages"] == 2


def test_min_missed_cleavages_above_max_rejected(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert (
        _invoke(fasta, out, "--missed-cleavages", "1", "--min-missed-cleavages", "2")
        == 1
    )
    assert "--min-missed-cleavages" in capsys.readouterr().err
    assert not out.exists()


def test_summary_reports_the_missed_cleavage_band(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert (
        _invoke(fasta, out, "--missed-cleavages", "1", "--min-missed-cleavages", "1")
        == 0
    )
    assert "missed cleavages 1-1" in capsys.readouterr().err


# ── modifications ──────────────────────────────────────────────────────


def test_variable_mod_alias_canonicalizes_to_accession(fasta, tmp_path) -> None:
    """``enumerate_modforms`` accepts UNIMOD names, but builds the ProForma
    tag by splitting on ':' — an unresolved alias becomes ``[Oxidation:]``
    and only fails later, inside mass computation."""
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--variable-mod", "M:Oxidation") == 0
    text = (out / "inclusion_list.csv").read_text()
    assert "[UNIMOD:35]" in text
    assert "[Oxidation:]" not in text


def test_variable_mod_accession_and_alias_agree(fasta, tmp_path) -> None:
    by_id, by_name = tmp_path / "id", tmp_path / "name"
    assert _invoke(fasta, by_id, "--variable-mod", "M:UNIMOD:35") == 0
    assert _invoke(fasta, by_name, "--variable-mod", "M:Oxidation") == 0
    assert (by_id / "inclusion_list.csv").read_text() == (
        by_name / "inclusion_list.csv"
    ).read_text()


def test_default_fixed_mod_is_carbamidomethyl(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out) == 0
    assert "C[UNIMOD:4]" in (out / "inclusion_list.csv").read_text()


def test_no_default_mods_drops_carbamidomethyl(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--no-default-mods") == 0
    assert "UNIMOD:4]" not in (out / "inclusion_list.csv").read_text()


def test_naming_a_fixed_mod_replaces_the_default(fasta, tmp_path) -> None:
    """``action="append"`` appends on top of a non-empty ``default=``; the
    default is resolved after parsing so this stays a replacement."""
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--fixed-mod", "K:UNIMOD:737") == 0
    text = (out / "inclusion_list.csv").read_text()
    assert "UNIMOD:737]" in text
    assert "UNIMOD:4]" not in text


def test_variable_mods_on_one_site_are_alternatives(fasta, tmp_path) -> None:
    out = tmp_path / "out"
    assert (
        _invoke(
            fasta,
            out,
            "--variable-mod",
            "M:UNIMOD:35",  # Oxidation
            "--variable-mod",
            "M:UNIMOD:425",  # dihydroxy
            "--no-default-mods",
        )
        == 0
    )
    text = (out / "inclusion_list.csv").read_text()
    assert "[UNIMOD:35]" in text
    assert "[UNIMOD:425]" in text


def test_terminal_mod_site_spellings(fasta, tmp_path) -> None:
    for i, site in enumerate(("N-term", "n-term", "NTERM")):
        out = tmp_path / f"out{i}"
        assert _invoke(fasta, out, "--fixed-mod", f"{site}:Acetyl") == 0
        assert "[UNIMOD:1]-" in (out / "inclusion_list.csv").read_text()


@pytest.mark.parametrize(
    "spec, message",
    [
        ("CUNIMOD:4", "single residue letter"),  # missing the site colon
        ("C:", "expected SITE:MODKEY"),
        (":UNIMOD:4", "expected SITE:MODKEY"),
        ("Cys:UNIMOD:4", "single residue letter"),
        ("C:UNIMOD:999999", "unknown modification"),
        ("C:Nonsense", "unknown modification"),
    ],
)
def test_malformed_mod_specs_rejected(fasta, tmp_path, capsys, spec, message) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--fixed-mod", spec) == 1
    err = capsys.readouterr().err
    assert message in err
    assert spec in err
    assert not out.exists()


def test_two_fixed_mods_on_one_site_rejected(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert (
        _invoke(fasta, out, "--fixed-mod", "C:UNIMOD:4", "--fixed-mod", "C:UNIMOD:6")
        == 1
    )
    assert "takes exactly one" in capsys.readouterr().err


def test_mod_with_wrong_specificity_surfaces_the_core_message(
    fasta, tmp_path, capsys
) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--fixed-mod", "K:UNIMOD:4") == 1
    assert "specificity" in capsys.readouterr().err


def test_parse_mod_specs_resolves_names_to_accessions() -> None:
    assert _parse_mod_specs(["C:UNIMOD:4", "M:Oxidation"], kind="variable") == {
        "C": ["UNIMOD:4"],
        "M": ["UNIMOD:35"],
    }


# ── isolation-window merging ───────────────────────────────────────────


def test_merge_within_da_collapses_rows(tmp_path, capsys) -> None:
    path = tmp_path / "iso.fasta"
    path.write_text(">sp|A|A x\nMKLSKAAAAAKISKAAAAAK\n")
    plain, merged = tmp_path / "plain", tmp_path / "merged"
    assert (
        _invoke(
            path, plain, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0",
        )
        == 0
    )
    capsys.readouterr()
    assert (
        _invoke(
            path, merged, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0.5",
        )
        == 0
    )
    err = capsys.readouterr().err

    n_plain = len((plain / "inclusion_list.csv").read_text().splitlines())
    n_merged = len((merged / "inclusion_list.csv").read_text().splitlines())
    assert n_merged < n_plain
    assert ";" in (merged / "inclusion_list.csv").read_text()
    assert "merged" in err

    counts = json.loads((merged / "manifest.json").read_text())["counts"]
    assert counts["merged_away"] == n_plain - n_merged
    assert counts["rows"] == n_merged - 1  # minus the header line


def test_merge_defaults_to_half_a_dalton(fasta, tmp_path) -> None:
    """On by default at an ion-trap isolation width — pinned so the
    default can't drift silently, since it changes what gets acquired."""
    out = tmp_path / "out"
    assert _invoke(fasta, out) == 0
    assert json.loads((out / "manifest.json").read_text())["params"][
        "merge_within_da"
    ] == 0.5


def test_merge_can_be_disabled(tmp_path) -> None:
    path = tmp_path / "iso.fasta"
    path.write_text(">sp|A|A x\nMKLSKAAAAAKISKAAAAAK\n")
    out = tmp_path / "out"
    assert (
        _invoke(
            path, out, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0",
        )
        == 0
    )
    assert ";" not in (out / "inclusion_list.csv").read_text()
    counts = json.loads((out / "manifest.json").read_text())["counts"]
    assert counts["merged_away"] == 0
    assert counts["rows"] == counts["entries"]


def test_merge_does_not_touch_the_parquet_sidecar(tmp_path) -> None:
    """The grid stays the canonical record — merging is a CSV projection."""
    import pyarrow.parquet as pq

    path = tmp_path / "iso.fasta"
    path.write_text(">sp|A|A x\nMKLSKAAAAAKISKAAAAAK\n")
    out = tmp_path / "out"
    assert (
        _invoke(
            path, out, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0.5",
        )
        == 0
    )
    grid = pq.read_table(out / "precursors.parquet")
    counts = json.loads((out / "manifest.json").read_text())["counts"]
    assert grid.num_rows == counts["entries"] > counts["rows"]


# ── validation ─────────────────────────────────────────────────────────


def test_missing_fasta_returns_2(tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert _invoke(tmp_path / "nope.fasta", out) == 2
    assert "not found" in capsys.readouterr().err


def test_min_charge_greater_than_max_rejected(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--min-charge", "4", "--max-charge", "2") == 1
    assert "--min-charge" in capsys.readouterr().err
    assert not out.exists()


def test_min_mz_greater_than_max_rejected(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--min-mz", "1200", "--max-mz", "400") == 1
    assert "exceeds" in capsys.readouterr().err


def test_empty_result_errors_with_diagnostic(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--min-mz", "5000", "--max-mz", "6000") == 1
    err = capsys.readouterr().err
    assert "no precursors survived" in err
    assert "5000" in err
    # The m/z window is named as the likely cause, with the count that
    # survives without it.
    assert "Dropping the m/z window alone leaves" in err
    assert not (out / "inclusion_list.csv").exists()


def test_invalid_protease_rejected_by_argparse(fasta, tmp_path) -> None:
    with pytest.raises(SystemExit):
        _invoke(fasta, tmp_path / "out", "--protease", "NotAProtease")


def test_protease_choices_match_registry() -> None:
    """``_protease_ids`` reads proteases.json directly to keep torch out of
    parser construction; it must not drift from the registry the handler
    resolves through."""
    from constellation.core.sequence.protein import PROTEASES

    assert _protease_ids() == PROTEASES.ids()


# ── run summary ────────────────────────────────────────────────────────


def test_summary_reports_entries_peptides_peptidoforms(
    fasta, tmp_path, capsys
) -> None:
    out = tmp_path / "out"
    assert (
        _invoke(
            fasta, out, "--min-charge", "2", "--max-charge", "3",
            "--variable-mod", "M:UNIMOD:35",
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "inclusion list:" in err
    assert "peptides" in err and "peptidoforms" in err

    counts = json.loads((out / "manifest.json").read_text())["counts"]
    # A charge sweep plus a variable mod multiplies entries per peptide.
    assert counts["peptides"] < counts["peptidoforms"] < counts["entries"]


def test_no_progress_silences_the_summary(fasta, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert _invoke(fasta, out, "--no-progress") == 0
    assert capsys.readouterr().err == ""


def test_collision_warning_emitted(tmp_path, capsys) -> None:
    """LSK / ISK differ only by Leu/Ile — exactly isobaric. With merging
    disabled they stay two rows, and the scan flags them."""
    path = tmp_path / "iso.fasta"
    path.write_text(">sp|A|A x\nMKLSKAAAAAKISKAAAAAK\n")
    out = tmp_path / "out"
    assert (
        _invoke(
            path, out, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0",
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "co-isolation risk" in err
    assert json.loads((out / "manifest.json").read_text())["counts"]["mz_collisions"] > 0


def test_merging_resolves_what_the_collision_scan_would_warn_about(
    tmp_path, capsys
) -> None:
    """The scan runs *after* merging, so it reports the risk that
    survives — entries already folded into one target are not a risk."""
    path = tmp_path / "iso.fasta"
    path.write_text(">sp|A|A x\nMKLSKAAAAAKISKAAAAAK\n")
    out = tmp_path / "out"
    assert (
        _invoke(
            path, out, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0.5",
        )
        == 0
    )
    assert "co-isolation risk" not in capsys.readouterr().err
    counts = json.loads((out / "manifest.json").read_text())["counts"]
    assert counts["merged_away"] > 0
    assert counts["mz_collisions"] == 0


def test_collision_warning_disabled(tmp_path, capsys) -> None:
    path = tmp_path / "iso.fasta"
    path.write_text(">sp|A|A x\nMKLSKAAAAAKISKAAAAAK\n")
    out = tmp_path / "out"
    assert (
        _invoke(
            path, out, "--min-peptide-length", "3", "--min-mz", "300",
            "--merge-within-da", "0", "--warn-collision-ppm", "0",
        )
        == 0
    )
    assert "co-isolation risk" not in capsys.readouterr().err
