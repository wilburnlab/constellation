"""Tier A tests for :mod:`constellation.massspec.search.encyclopedia._common`.

Exercises the manifest-envelope builder + PTM-toggle encoder + escape-
hatch parser without invoking any jar. Catches schema regressions
before any of the implementation PRs run end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia._common import (
    MINIMUM_ENCYCLOPEDIA_VERSION,
    build_manifest_envelope,
    default_heap_for_input,
    encyclopedia_passthrough_args,
    is_supported_version,
    ptm_toggle_args,
    require_min_encyclopedia,
    sha256_file,
    write_manifest,
)


# ── sha256_file ────────────────────────────────────────────────────────


def test_sha256_file_known_digest(tmp_path: Path) -> None:
    """SHA256 of empty file is the well-known constant."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert (
        sha256_file(p)
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_file_nonempty(tmp_path: Path) -> None:
    p = tmp_path / "hi.bin"
    p.write_bytes(b"hi")
    # echo -n hi | sha256sum
    assert (
        sha256_file(p)
        == "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"
    )


# ── ptm_toggle_args ────────────────────────────────────────────────────


def test_ptm_toggle_args_empty() -> None:
    assert ptm_toggle_args(None) == []
    assert ptm_toggle_args({}) == []


def test_ptm_toggle_args_sorted_for_determinism() -> None:
    out = ptm_toggle_args(
        {"Phospho": "var", "Carbamidomethyl": "fix", "Acetyl": "off"}
    )
    # Sorted by PTM name → Acetyl, Carbamidomethyl, Phospho
    assert out == [
        "-ptmAcetyl",
        "off",
        "-ptmCarbamidomethyl",
        "fix",
        "-ptmPhospho",
        "var",
    ]


def test_ptm_toggle_args_custom_prefix() -> None:
    out = ptm_toggle_args({"Phospho": "var"}, prefix="--ptm")
    assert out == ["--ptmPhospho", "var"]


def test_ptm_toggle_args_rejects_bad_value() -> None:
    with pytest.raises(ValueError, match="off"):
        ptm_toggle_args({"Phospho": "yes"})  # type: ignore[dict-item]


# ── encyclopedia_passthrough_args ──────────────────────────────────────


def test_encyclopedia_passthrough_args_empty() -> None:
    assert encyclopedia_passthrough_args(None) == []
    assert encyclopedia_passthrough_args([]) == []


def test_encyclopedia_passthrough_args_splits_on_equals() -> None:
    out = encyclopedia_passthrough_args(
        ["-percolatorVersion=v3-05", "-foo=bar=baz", "-quiet"]
    )
    # "-foo=bar=baz" splits on first '=' → ['-foo', 'bar=baz']
    assert out == [
        "-percolatorVersion",
        "v3-05",
        "-foo",
        "bar=baz",
        "-quiet",
    ]


# ── default_heap_for_input ─────────────────────────────────────────────


def test_default_heap_for_input_scales() -> None:
    assert default_heap_for_input(500_000) == "8g"  # < 1 GiB
    assert default_heap_for_input(2 * 1024**3) == "12g"  # 2 GiB
    assert default_heap_for_input(10 * 1024**3) == "24g"  # 10 GiB
    assert default_heap_for_input(30 * 1024**3) == "48g"  # 30 GiB


# ── build_manifest_envelope ────────────────────────────────────────────


def test_build_manifest_envelope_shape(tmp_path: Path) -> None:
    mzml = tmp_path / "sample.mzML"
    mzml.write_bytes(b"<mzML>fake</mzML>")
    library = tmp_path / "library.dlib"
    library.write_bytes(b"libcontents")

    env = build_manifest_envelope(
        subcommand="massspec search",
        constellation_version="0.1.0",
        constellation_argv=[
            "constellation",
            "massspec",
            "search",
            "--mzml",
            str(mzml),
        ],
        java_argv=[
            "java",
            "-Xmx12g",
            "-jar",
            "encyclopedia-6.5.15.jar",
            "-i",
            str(mzml),
        ],
        tool={
            "name": "encyclopedia",
            "version": "6.5.15",
            "jar_path": "/path/to/encyclopedia-6.5.15.jar",
            "jar_sha256": "deadbeef" * 8,
            "source": "env",
            "env_var_set": True,
            "java_version": "openjdk 21.0.11",
            "java_source": "bundled",
        },
        inputs={"mzml": mzml, "library": library, "fasta": None},
        outputs={
            "elib": tmp_path / "sample.elib",
            "library_pqdir": tmp_path / "library_pqdir",
            "quant_pqdir": None,
        },
        runtime={"elapsed_seconds": 12.3, "returncode": 0},
        ingest={
            "skipped": False,
            "fragment_tolerance_ppm": 20.0,
            "library_counts": {"proteins": 42},
        },
        encyclopedia_passthrough_args=["-percolatorVersion", "v3-05"],
    )

    # Required top-level keys
    for key in (
        "constellation_version",
        "subcommand",
        "timestamp_utc",
        "tool",
        "argv",
        "inputs",
        "outputs",
        "runtime",
        "ingest",
        "encyclopedia_passthrough_args",
    ):
        assert key in env

    assert env["subcommand"] == "massspec search"
    assert env["constellation_version"] == "0.1.0"
    assert env["tool"]["version"] == "6.5.15"
    assert env["argv"]["constellation"][0] == "constellation"
    assert env["argv"]["java"][0] == "java"

    # Input records: present files get {path, sha256, size_bytes}; None passes through
    assert env["inputs"]["mzml"]["path"] == str(mzml)
    assert env["inputs"]["mzml"]["size_bytes"] == len(b"<mzML>fake</mzML>")
    assert len(env["inputs"]["mzml"]["sha256"]) == 64
    assert env["inputs"]["library"]["size_bytes"] == len(b"libcontents")
    assert env["inputs"]["fasta"] is None

    # Outputs stringify; None stays None
    assert env["outputs"]["elib"] == str(tmp_path / "sample.elib")
    assert env["outputs"]["quant_pqdir"] is None

    # Runtime defaults are populated (host / platform / python) PLUS the caller's overrides
    assert env["runtime"]["host"]  # non-empty
    assert env["runtime"]["python"]
    assert env["runtime"]["elapsed_seconds"] == 12.3
    assert env["runtime"]["returncode"] == 0

    # Ingest passthrough
    assert env["ingest"]["library_counts"]["proteins"] == 42

    # Timestamp shape ISO-8601 UTC
    assert env["timestamp_utc"].endswith("Z")


def test_build_manifest_envelope_missing_input_raises(
    tmp_path: Path,
) -> None:
    """Calling with a non-None Path that doesn't exist raises (we want the
    manifest to fail loudly, not invent a sha256)."""
    with pytest.raises(FileNotFoundError):
        build_manifest_envelope(
            subcommand="massspec search",
            constellation_version="0.1.0",
            constellation_argv=["constellation", "massspec", "search"],
            java_argv=None,
            tool={"name": "encyclopedia"},
            inputs={"mzml": tmp_path / "does-not-exist.mzML"},
            outputs={},
        )


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    env = build_manifest_envelope(
        subcommand="massspec search",
        constellation_version="0.1.0",
        constellation_argv=["constellation"],
        java_argv=None,
        tool={"name": "encyclopedia", "version": "6.5.15"},
        inputs={},
        outputs={"elib": tmp_path / "x.elib"},
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, env)
    assert manifest_path.read_text().endswith("\n")
    parsed = json.loads(manifest_path.read_text())
    assert parsed["subcommand"] == "massspec search"
    assert parsed["tool"]["version"] == "6.5.15"


# ── minimum-version pin (>= 6.5.15) ────────────────────────────────────


def test_minimum_encyclopedia_version_is_pinned() -> None:
    assert MINIMUM_ENCYCLOPEDIA_VERSION == "6.5.15"


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("6.5.15", True),    # the floor
        ("6.5.16", True),
        ("6.6.0", True),     # newer minor — forward-compatible
        ("7.0.0", True),     # much newer
        ("2.12.30", False),  # legacy public release — string-compare TRAP
        ("6.5.14", False),   # just below the floor
        (None, True),        # unparseable build: can't prove it's old -> allowed
    ],
)
def test_is_supported_version(version, supported) -> None:
    assert is_supported_version(version) is supported


def _encyclopedia_handle(version):
    """A ToolHandle with the given probed version, for guard tests."""
    from constellation.thirdparty.registry import ToolHandle, ToolSpec

    spec = ToolSpec(name="encyclopedia", env_var="CONSTELLATION_ENCYCLOPEDIA_HOME")
    return ToolHandle(spec, Path("/nonexistent/encyclopedia.jar"), "env", version)


def test_require_min_encyclopedia_rejects_old(capsys) -> None:
    rc = require_min_encyclopedia(_encyclopedia_handle("2.12.30"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "2.12.30" in err          # names the offending version
    assert "6.5.15" in err           # and the floor


def test_require_min_encyclopedia_accepts_current_and_unparseable(capsys) -> None:
    assert require_min_encyclopedia(_encyclopedia_handle("6.5.15")) == 0
    assert require_min_encyclopedia(_encyclopedia_handle("6.6.0")) == 0
    assert require_min_encyclopedia(_encyclopedia_handle(None)) == 0
    assert capsys.readouterr().err == ""
