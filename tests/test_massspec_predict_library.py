"""Tests for ``constellation massspec predict-library``.

Two tiers:

  * **Tier A** (always runs): ``build_predict_library_args`` produces
    the right EncyclopeDIA flag layout. Pure function, no subprocess.
  * **Tier B** (gated on ``$CONSTELLATION_ENCYCLOPEDIA_HOME``): real
    jar invocation against a tiny FASTA produces a ``.dlib`` +
    auto-ingested ``library_pqdir/`` + ``manifest.json`` + ``_SUCCESS``.

The Tier B test takes ~30-60 seconds on first invocation per machine
(JChronologer model weights warm into the prediction cache) and
~10-20 seconds on subsequent invocations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia import (
    build_predict_library_args,
)


# ── Tier A: arg builder ────────────────────────────────────────────────


def test_build_args_minimal(tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    out = tmp_path / "out.dlib"
    args = build_predict_library_args(fasta=fasta, output_dlib=out)
    assert args[:2] == ["-convert", "-fastaToJChronologerLibrary"]
    assert "-i" in args
    assert str(fasta) in args
    assert "-o" in args
    assert str(out) in args


def test_build_args_default_values_match_jar(tmp_path: Path) -> None:
    """The defaults baked into the wrapper match EncyclopeDIA's own
    ``-h`` output. Catches drift if EncyclopeDIA changes a default and
    the lab silently relies on the wrapper carrying it forward."""
    args = build_predict_library_args(
        fasta=tmp_path / "x.fa", output_dlib=tmp_path / "x.dlib"
    )
    flag_value = dict(zip(args[::2], args[1::2]))
    assert flag_value["-addDecoys"] == "true"
    assert flag_value["-adjustNCEForDIA"] == "true"
    assert flag_value["-defaultCharge"] == "3"
    assert flag_value["-defaultNCE"] == "33"
    assert flag_value["-enzyme"] == "Trypsin"
    assert flag_value["-generateProteinEntrapments"] == "false"
    assert flag_value["-maxCharge"] == "6"
    assert flag_value["-maxMissedCleavage"] == "1"
    assert flag_value["-maxMz"] == "1002.7"
    assert flag_value["-maxVariableForms"] == "1000"
    assert flag_value["-maxVariableMods"] == "1"
    assert flag_value["-minCharge"] == "1"
    assert flag_value["-minMz"] == "396.4"
    assert flag_value["-raggedNTerm"] == "false"
    # No -predictionCache when None (caller relies on EncyclopeDIA's default).
    assert "-predictionCache" not in args


def test_build_args_with_ptms_and_overrides(tmp_path: Path) -> None:
    args = build_predict_library_args(
        fasta=tmp_path / "x.fa",
        output_dlib=tmp_path / "x.dlib",
        ptms={
            "Carbamidomethyl": "fix",
            "Oxidation": "var",
            "Phospho": "var",
        },
        max_charge=4,
        min_charge=2,
        enzyme="LysC",
        add_decoys=False,
        prediction_cache=tmp_path / "cache",
        extra_args=("-verbose",),
    )
    pos_carb = args.index("-ptmCarbamidomethyl")
    pos_ox = args.index("-ptmOxidation")
    pos_ph = args.index("-ptmPhospho")
    assert pos_carb < pos_ox < pos_ph
    assert args[pos_carb + 1] == "fix"
    assert args[pos_ox + 1] == "var"
    assert args[pos_ph + 1] == "var"
    flag_value = dict(zip(args[::2], args[1::2]))
    assert flag_value["-maxCharge"] == "4"
    assert flag_value["-minCharge"] == "2"
    assert flag_value["-enzyme"] == "LysC"
    assert flag_value["-addDecoys"] == "false"
    assert flag_value["-predictionCache"] == str(tmp_path / "cache")
    assert args[-1] == "-verbose"


# ── Tier B: real jar invocation ────────────────────────────────────────


def _have_encyclopedia_install() -> bool:
    home = os.environ.get("CONSTELLATION_ENCYCLOPEDIA_HOME")
    if home is None:
        return False
    home_path = Path(home).expanduser()
    return any(
        (home_path / cand).is_file()
        for cand in ("encyclopedia-6.5.15.jar", "encyclopedia-2.12.30-executable.jar")
    )


_HAVE_ENC = _have_encyclopedia_install()

_TINY_FASTA = (
    ">sp|P00001|TINY1\n"
    "MSDFLKAGSEPRKFGLALTGKVNRGQKLAPLSRLGGFNNTAQEDFFKLLR\n"
    ">sp|P00002|TINY2\n"
    "MAVPENATPRSGEDLIRYGNKVRGRELDSGFADGFEDIRSEMARFAANLPLR\n"
)


@pytest.mark.skipif(
    not _HAVE_ENC,
    reason="needs $CONSTELLATION_ENCYCLOPEDIA_HOME with the encyclopedia jar",
)
def test_run_predict_library_end_to_end(tmp_path: Path) -> None:
    """Real jar invocation against a 2-protein FASTA. Validates the
    runner + jar wiring; not prediction accuracy."""
    from constellation.massspec.search.encyclopedia import run_predict_library

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(_TINY_FASTA)
    output_dir = tmp_path / "run"
    output_dlib = output_dir / "predicted.dlib"

    result = run_predict_library(
        fasta=fasta,
        output_dlib=output_dlib,
        output_dir=output_dir,
        stream_to_stderr=False,
        jvm_heap_max="4g",
    )

    assert result.returncode == 0
    assert output_dlib.is_file()
    assert (output_dir / "logs" / "stdout.log").exists()
    assert (output_dir / "logs" / "stderr.log").exists()
    assert (output_dir / "logs" / "java-version.txt").exists()
    assert result.jar_sha256
    assert result.elapsed_seconds > 0


@pytest.mark.skipif(
    not _HAVE_ENC,
    reason="needs $CONSTELLATION_ENCYCLOPEDIA_HOME with the encyclopedia jar",
)
def test_predict_library_cli_handler_end_to_end(tmp_path: Path) -> None:
    """End-to-end via the CLI handler (covers auto-ingest + manifest + _SUCCESS)."""
    from constellation.cli.__main__ import _build_parser

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(_TINY_FASTA)
    output_dir = tmp_path / "run"
    output_dlib = output_dir / "predicted.dlib"

    parser = _build_parser()
    args = parser.parse_args(
        [
            "massspec",
            "predict-library",
            "--fasta",
            str(fasta),
            "--output-dlib",
            str(output_dlib),
            "--output-dir",
            str(output_dir),
            "--jvm-heap",
            "4g",
            "--no-progress",
        ]
    )
    rc = args.func(args)
    assert rc == 0
    assert output_dlib.is_file()
    assert (output_dir / "_SUCCESS").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "library_pqdir").is_dir()
    assert (output_dir / "library_pqdir" / "peptides.parquet").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["subcommand"] == "massspec predict-library"
    assert manifest["tool"]["version"] in ("2.12.30", "6.5.15")
    assert manifest["runtime"]["returncode"] == 0
    assert manifest["ingest"]["skipped"] is False
    assert isinstance(manifest["ingest"]["library_counts"]["peptides"], int)

    # --resume on a complete output dir short-circuits to exit 0.
    args2 = parser.parse_args(
        [
            "massspec",
            "predict-library",
            "--fasta",
            str(fasta),
            "--output-dlib",
            str(output_dlib),
            "--output-dir",
            str(output_dir),
            "--resume",
            "--no-progress",
        ]
    )
    rc2 = args2.func(args2)
    assert rc2 == 0


@pytest.mark.skipif(
    not _HAVE_ENC,
    reason="needs $CONSTELLATION_ENCYCLOPEDIA_HOME with the encyclopedia jar",
)
def test_predict_library_refuses_to_overwrite_complete_dir(
    tmp_path: Path,
) -> None:
    """When ``_SUCCESS`` exists and ``--resume`` is absent, the handler
    refuses (exit 1) rather than re-running the jar."""
    from constellation.cli.__main__ import _build_parser

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(_TINY_FASTA)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "_SUCCESS").write_bytes(b"")

    parser = _build_parser()
    args = parser.parse_args(
        [
            "massspec",
            "predict-library",
            "--fasta",
            str(fasta),
            "--output-dlib",
            str(output_dir / "predicted.dlib"),
            "--output-dir",
            str(output_dir),
            "--no-progress",
        ]
    )
    rc = args.func(args)
    assert rc == 1
