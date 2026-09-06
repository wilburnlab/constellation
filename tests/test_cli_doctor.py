"""Smoke tests for the `constellation doctor` command."""

from constellation.cli.__main__ import main


def test_doctor_runs_and_prints_registered_tools(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    # Even when no tools are installed, the header + every registered tool's
    # row should print (the genome-pipeline tools included, even though they
    # list as "not found" on a fresh checkout).
    assert "tool" in out
    assert "encyclopedia" in out
    for name in ("busco", "iqtree", "ragout", "cactus"):
        assert name in out
    # Return code: 0 if every tool resolved, 1 otherwise. We assert only
    # that the process ran cleanly (nonzero is fine for a fresh checkout).
    assert rc in (0, 1)


def test_constellation_help_shows_doctor(capsys):
    # argparse exits via SystemExit when --help is passed.
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "doctor" in out
