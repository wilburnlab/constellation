"""CLI-level guards for ``massspec predict-library --backend koina``.

None of these reach the network: every check they exercise is meant to
fail before the first request is sent.
"""

from __future__ import annotations

import pytest

from constellation.massspec.cli import (
    _cmd_massspec_predict_library,
    _parse_collision_energies,
)


def _args(**overrides):
    """A namespace with the parser's defaults, overridable per test."""
    import argparse

    base = dict(
        backend="koina",
        fasta=None,
        peptides=None,
        from_library=None,
        output_dlib=None,
        output_library=None,
        output_dir=None,
        resume=False,
        no_progress=True,
        ms2_model="Prosit_2020_intensity_HCD",
        rt_model="Chronologer_RT",
        koina_url=None,
        collision_energy=None,
        on_unsupported_mod="error",
        min_intensity=1e-4,
        enzyme="Trypsin",
        min_charge=2,
        max_charge=3,
        min_mz=396.4,
        max_mz=1002.7,
        max_missed_cleavage=1,
        max_variable_mods=1,
        max_variable_forms=1000,
        no_decoys=False,
        no_adjust_nce_for_dia=False,
        default_nce=33,
        default_charge=None,
        collision_energy_by_charge=None,
        generate_protein_entrapments=False,
        entrapment_seed=1,
        prediction_cache=None,
        ragged_n_term=False,
        # These must mirror the parser's declared defaults, not None —
        # the koina backend errors on any non-default EncyclopeDIA flag.
        jvm_heap_max="12g",
        jvm_heap_min=None,
        jvm_tmpdir=None,
        encyclopedia_arg=[],
    )
    from constellation.massspec.search.encyclopedia.ptm_defaults import (  # noqa: F401
        PTM_DEFAULTS,
    )
    from constellation.massspec.cli import _camel_to_kebab, _PTM_NAMES, _ptm_default_for

    for name in _PTM_NAMES:
        attr = f"ptm_{_camel_to_kebab(name).replace('-', '_')}"
        base[attr] = _ptm_default_for(name)
    base.update(overrides)
    return argparse.Namespace(**base)


# ── collision-energy parsing ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, [None]),
        ("30", [30.0]),
        ("20,25,30", [20.0, 25.0, 30.0]),
        ("20, 25 , 30", [20.0, 25.0, 30.0]),
        ("30,", [30.0]),
    ],
)
def test_parse_collision_energies(raw, expected):
    assert _parse_collision_energies(raw) == expected


def test_parse_collision_energies_rejects_garbage():
    with pytest.raises(ValueError, match="bad collision energy"):
        _parse_collision_energies("20,fast,30")


# ── source selection ──────────────────────────────────────────────────


def test_requires_exactly_one_source(tmp_path, capsys):
    rc = _cmd_massspec_predict_library(_args(output_dir=tmp_path))
    assert rc == 1
    assert "exactly one of --fasta" in capsys.readouterr().err


def test_rejects_two_sources(tmp_path, capsys):
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, fasta=tmp_path / "a.fasta", peptides=tmp_path / "b.tsv")
    )
    assert rc == 1
    assert "exactly one of --fasta" in capsys.readouterr().err


# ── backend-flag hygiene ──────────────────────────────────────────────


def test_encyclopedia_only_flags_error_rather_than_no_op(tmp_path, capsys):
    """Silently ignoring --jvm-heap-max would imply it took effect."""
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, fasta=tmp_path / "a.fasta", jvm_heap_max="8G")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "EncyclopeDIA-only" in err and "--jvm-heap-max" in err


def test_non_default_ptm_flag_errors_under_koina(tmp_path, capsys):
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, fasta=tmp_path / "a.fasta", ptm_phospho="var")
    )
    assert rc == 1
    assert "--ptm-phospho" in capsys.readouterr().err


def test_predict_library_encyclopedia_still_requires_fasta_and_dlib(tmp_path, capsys):
    """Both dropped argparse's required=True; the handler still enforces them."""
    rc = _cmd_massspec_predict_library(
        _args(backend="encyclopedia", output_dir=tmp_path, fasta=None)
    )
    assert rc == 1
    assert "--fasta is required" in capsys.readouterr().err

    rc = _cmd_massspec_predict_library(
        _args(backend="encyclopedia", output_dir=tmp_path, fasta=tmp_path / "a.fasta")
    )
    assert rc == 1
    assert "--output-dlib is required" in capsys.readouterr().err


# ── run-dir contract ──────────────────────────────────────────────────


def _completed(tmp_path, **over):
    """Mark ``tmp_path`` finished with the key the arguments make."""
    from constellation.core.stages import mark_done, stage_key
    from constellation.massspec.cli.predict_library import _koina_key_parts

    mark_done(tmp_path, stage_key(**_koina_key_parts(_args(output_dir=tmp_path, fasta=tmp_path / "a.fasta", **over))))


def test_refuses_a_completed_output_dir(tmp_path, capsys):
    _completed(tmp_path)
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, fasta=tmp_path / "a.fasta")
    )
    assert rc == 1
    assert "already complete" in capsys.readouterr().err


def test_resume_short_circuits_a_completed_dir(tmp_path, capsys):
    _completed(tmp_path, resume=True)
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, fasta=tmp_path / "a.fasta", resume=True)
    )
    assert rc == 0
    assert "already complete" in capsys.readouterr().out


def test_a_keyless_success_marker_is_not_trusted(tmp_path, capsys):
    """A marker written before runs were keyed cannot say what it holds;
    it was reused unchecked until 2026-11-07."""
    (tmp_path / "_SUCCESS").write_bytes(b"")
    rc = _cmd_massspec_predict_library(_args(output_dir=tmp_path, fasta=tmp_path / "a.fasta"))
    assert rc == 1
    assert "holds a prediction of other inputs or settings" in capsys.readouterr().err


def test_a_prediction_of_another_layout_is_not_resumed(tmp_path, capsys, monkeypatch):
    """The key carries PREDICTION_LAYOUT, so output written under an older
    rule for which precursors to keep is not taken for finished."""
    import constellation.massspec.library.prediction as prediction

    _completed(tmp_path)
    monkeypatch.setattr(prediction, "PREDICTION_LAYOUT", prediction.PREDICTION_LAYOUT + 1)
    rc = _cmd_massspec_predict_library(_args(output_dir=tmp_path, fasta=tmp_path / "a.fasta"))
    assert rc == 1
    assert "other inputs or settings" in capsys.readouterr().err


# ── energy-sweep resume ────────────────────────────────────────────────
#
# Completed per-energy runs were skipped without restoring their
# summaries, so the aggregate manifest under-reported the sweep:
# resuming an interrupted 20/30 listed only 30, and resuming after every
# energy had finished wrote "runs": [] and still reported success.


def _sweep_stubs(monkeypatch, tmp_path, *, calls):
    """Stub the JVM-free parts of a two-energy koina sweep."""
    import json

    from constellation.massspec.library.koina.assemble import AssemblyStats

    peptides = tmp_path / "peptides.txt"
    peptides.write_text("LVNELTEFAK\n")

    class _Lib:
        class _T:
            num_rows = 1

        proteins = peptides = precursors = fragments = _T()

    class _Probe:
        model_inputs = {"collision_energies": None, "peptide_sequences": None}

    # The handler imports these lazily, so patch the ORIGIN modules.
    monkeypatch.setattr(
        "constellation.massspec.library.koina.client.make_client",
        lambda *a, **k: _Probe(),
    )
    from constellation.massspec.library.digest import PrecursorSpec

    monkeypatch.setattr(
        "constellation.massspec.library.digest.precursors_from_peptide_list",
        lambda *a, **k: [PrecursorSpec("LVNELTEFAK", "LVNELTEFAK", 2, 582.32)],
    )
    monkeypatch.setattr(
        "constellation.massspec.library.save_library", lambda *a, **k: None
    )

    def _fake_predict_library(**kw):
        calls.append(kw["collision_energy"])
        return _Lib(), AssemblyStats(n_precursors=1, n_fragments=1)

    monkeypatch.setattr(
        "constellation.massspec.library.koina.api.predict_library",
        _fake_predict_library,
    )
    return peptides, json


def _interrupted_sweep(out, peptides, calls, *, keep):
    """A 20,30 sweep run to the end, then undone but for the energies in
    ``keep``, as a sweep stopped after them leaves it: each kept energy has
    its manifest and its keyed _SUCCESS, the sweep has none."""
    import shutil

    rc = _cmd_massspec_predict_library(_args(output_dir=out, peptides=peptides, collision_energy="20,30"))
    assert rc == 0
    (out / "_SUCCESS").unlink()
    (out / "manifest.json").unlink()
    for run in ("ce_20", "ce_30"):
        if run not in keep:
            shutil.rmtree(out / run)
    calls.clear()


def test_resume_restores_completed_energies_into_the_manifest(
    tmp_path, monkeypatch
) -> None:
    calls: list = []
    peptides, json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)

    out = tmp_path / "out"
    _interrupted_sweep(out, peptides, calls, keep=("ce_20",))

    rc = _cmd_massspec_predict_library(
        _args(
            output_dir=out, peptides=peptides, resume=True,
            collision_energy="20,30",
        )
    )
    assert rc == 0
    assert [c.nce for c in calls] == [30.0], "the completed energy should not be recomputed"
    assert calls[0].origin == "scaled_from_reference" and calls[0].reference_charge == 3

    manifest = json.loads((out / "manifest.json").read_text())
    energies = sorted(r["collision_energy"] for r in manifest["runs"])
    assert energies == [20.0, 30.0], "resumed run dropped from the aggregate"


def test_resume_after_every_energy_completed_still_reports_them(
    tmp_path, monkeypatch
) -> None:
    """The degenerate case that wrote "runs": [] and claimed success."""
    calls: list = []
    peptides, json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)

    out = tmp_path / "out"
    _interrupted_sweep(out, peptides, calls, keep=("ce_20", "ce_30"))

    rc = _cmd_massspec_predict_library(
        _args(
            output_dir=out, peptides=peptides, resume=True,
            collision_energy="20,30",
        )
    )
    assert rc == 0
    assert calls == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["runs"]) == 2


def test_resume_errors_when_a_completed_run_has_no_manifest(
    tmp_path, monkeypatch, capsys
) -> None:
    """_SUCCESS without its summary would silently under-report."""
    calls: list = []
    peptides, _json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)

    out = tmp_path / "out"
    _interrupted_sweep(out, peptides, calls, keep=("ce_20",))
    (out / "ce_20" / "manifest.json").unlink()

    rc = _cmd_massspec_predict_library(
        _args(
            output_dir=out, peptides=peptides, resume=True,
            collision_energy="20,30",
        )
    )
    assert rc == 1
    assert "no manifest.json" in capsys.readouterr().err


def test_a_keyless_energy_run_is_recomputed_on_resume(tmp_path, monkeypatch) -> None:
    """A per-energy _SUCCESS without a key cannot say what it holds, so a
    resumed sweep predicts that energy again instead of restoring it under
    the sweep's key (Codex review, 2026-11-07; the sweep's own marker was
    held to its key already)."""
    import json

    calls: list = []
    peptides, _json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)

    out = tmp_path / "out"
    _interrupted_sweep(out, peptides, calls, keep=("ce_20", "ce_30"))
    (out / "ce_20" / "_SUCCESS").write_bytes(b"")
    (out / "ce_30" / "_SUCCESS").write_text("some-other-key\n")

    rc = _cmd_massspec_predict_library(
        _args(output_dir=out, peptides=peptides, resume=True, collision_energy="20,30")
    )
    assert rc == 0
    assert sorted(c.nce for c in calls) == [20.0, 30.0]
    from constellation.core.stages import recorded_key

    assert recorded_key(out / "ce_20") not in (None, "")
    assert len(json.loads((out / "manifest.json").read_text())["runs"]) == 2


# ── collision energy per charge ────────────────────────────────────────


@pytest.mark.parametrize(
    ("override", "flag"),
    [
        ({"collision_energy": "30"}, "--collision-energy"),
        ({"default_charge": 2}, "--default-charge"),
        ({"no_adjust_nce_for_dia": True}, "--no-adjust-nce-for-dia"),
    ],
)
def test_energy_table_refuses_the_flags_it_replaces(tmp_path, capsys, override, flag):
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, peptides=tmp_path / "p.txt",
              collision_energy_by_charge="2=28,3=31", **override)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--collision-energy-by-charge" in err and flag in err


def test_energy_table_refuses_garbage(tmp_path, capsys):
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path, peptides=tmp_path / "p.txt",
              collision_energy_by_charge="2=28,three=31")
    )
    assert rc == 1
    assert "bad entry" in capsys.readouterr().err


def test_energy_table_must_cover_every_charge_predicted(tmp_path, monkeypatch, capsys):
    calls: list = []
    peptides, _json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path / "out", peptides=peptides,
              collision_energy_by_charge="3=31,4=33")
    )
    assert rc == 1
    assert "no energy for charge 2" in capsys.readouterr().err
    assert calls == []


def test_energy_table_is_passed_as_given_and_recorded(tmp_path, monkeypatch):
    calls: list = []
    peptides, json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)
    out = tmp_path / "out"
    rc = _cmd_massspec_predict_library(
        _args(output_dir=out, peptides=peptides, collision_energy_by_charge="2=27.5,3=31")
    )
    assert rc == 0
    (energy,) = calls
    assert energy.origin == "explicit"
    assert energy.by_charge() == {2: 27.5, 3: 31.0}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["koina"]["collision_energy_by_charge"] == {"2": 27.5, "3": 31.0}
    assert manifest["koina"]["adjust_nce_for_dia"] is False
    (run,) = manifest["runs"]
    assert run["collision_energy"] is None
    assert run["collision_energy_by_charge"]["table"] == {"2": 27.5}


def test_energy_table_is_refused_for_a_model_without_an_energy_input(tmp_path, monkeypatch, capsys):
    calls: list = []
    peptides, _json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)

    class _NoEnergy:
        model_inputs = {"peptide_sequences": None}

    monkeypatch.setattr(
        "constellation.massspec.library.koina.client.make_client", lambda *a, **k: _NoEnergy()
    )
    rc = _cmd_massspec_predict_library(
        _args(output_dir=tmp_path / "out", peptides=peptides, collision_energy_by_charge="2=28")
    )
    assert rc == 1
    assert "declares no collision-energy input" in capsys.readouterr().err
    assert calls == []


def test_default_charge_reaches_koina_as_the_reference(tmp_path, monkeypatch):
    calls: list = []
    peptides, json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)
    out = tmp_path / "out"
    rc = _cmd_massspec_predict_library(
        _args(output_dir=out, peptides=peptides, collision_energy="30", default_charge=2)
    )
    assert rc == 0
    (energy,) = calls
    assert energy.origin == "scaled_from_reference" and energy.reference_charge == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["koina"]["nce_reference_charge"] == 2
    assert manifest["runs"][0]["collision_energy_by_charge"]["table"]["2"] == 30.0


def test_no_adjust_makes_one_energy_for_every_charge(tmp_path, monkeypatch):
    calls: list = []
    peptides, json = _sweep_stubs(monkeypatch, tmp_path, calls=calls)
    out = tmp_path / "out"
    rc = _cmd_massspec_predict_library(
        _args(output_dir=out, peptides=peptides, collision_energy="30", no_adjust_nce_for_dia=True)
    )
    assert rc == 0
    (energy,) = calls
    assert energy.origin == "uniform"
    assert json.loads((out / "manifest.json").read_text())["koina"]["nce_reference_charge"] is None


def test_encyclopedia_backend_refuses_an_energy_table(tmp_path, capsys):
    rc = _cmd_massspec_predict_library(
        _args(backend="encyclopedia", output_dir=tmp_path, fasta=tmp_path / "a.fasta",
              output_dlib=tmp_path / "a.dlib", collision_energy_by_charge="2=28")
    )
    assert rc == 1
    assert "not available with --backend encyclopedia" in capsys.readouterr().err
