"""Unit tests for the third-party tool registry."""

import re
from pathlib import Path

import pytest

from constellation.thirdparty import ToolNotFoundError, find
from constellation.thirdparty.registry import (
    ToolSpec,
    register,
    registered,
    try_find,
    version_ge,
    version_tuple,
)


def test_encyclopedia_is_registered():
    names = {s.name for s in registered()}
    assert "encyclopedia" in names


def test_genome_tools_are_registered():
    # The genome-assembly + comparative-genomics tools each declare an
    # adapter so they show up in `constellation doctor`. busco is here too:
    # it moved out of environment.yml into a dedicated-conda-env installer.
    by_name = {s.name: s for s in registered()}
    for name in ("busco", "iqtree", "ragout", "cactus"):
        assert name in by_name, f"{name} adapter not registered"
        spec = by_name[name]
        assert spec.install_script and spec.install_script.startswith("scripts/")
        assert spec.env_var.startswith("CONSTELLATION_")


def test_missing_tool_raises_with_hint(tmp_path, monkeypatch):
    spec = ToolSpec(
        name="fake-tool-for-testing",
        env_var="CONSTELLATION_FAKE_TOOL_FOR_TESTING_HOME",
        artifact="bin/fake",
        install_script="scripts/install-fake.sh",
    )
    register(spec)
    monkeypatch.delenv(spec.env_var, raising=False)

    with pytest.raises(ToolNotFoundError) as excinfo:
        find("fake-tool-for-testing")
    assert "fake-tool-for-testing" in str(excinfo.value)
    assert "install-fake.sh" in str(excinfo.value)


def test_env_var_resolves(tmp_path, monkeypatch):
    # Build a fake install tree and point env var at it.
    home = tmp_path / "fake-install"
    (home / "bin").mkdir(parents=True)
    artifact = home / "bin" / "fake"
    artifact.write_text("#!/bin/sh\necho fake\n")
    artifact.chmod(0o755)

    spec = ToolSpec(
        name="fake-env-resolver",
        env_var="CONSTELLATION_FAKE_ENV_RESOLVER_HOME",
        artifact="bin/fake",
    )
    register(spec)
    monkeypatch.setenv(spec.env_var, str(home))

    handle = find("fake-env-resolver")
    assert handle.source == "env"
    assert handle.path == artifact


def test_try_find_returns_none_when_missing(monkeypatch):
    spec = ToolSpec(
        name="fake-never-installed",
        env_var="CONSTELLATION_FAKE_NEVER_INSTALLED_HOME",
        artifact="bin/fake",
    )
    register(spec)
    monkeypatch.delenv(spec.env_var, raising=False)
    assert try_find("fake-never-installed") is None


# ── version helpers ─────────────────────────────────────────────────────


def test_version_tuple_parsing():
    assert version_tuple("6.5.15") == (6, 5, 15)
    assert version_tuple("2.12.30") == (2, 12, 30)
    assert version_tuple("7") == (7,)
    assert version_tuple("6.6-beta") == (6, 6)  # leading-digit run per field
    assert version_tuple("nightly") == ()


def test_version_ge_numeric_not_lexicographic():
    # The trap: lexicographic string compare would order "6.5.15" < "2.12.30"
    # (because '2' < '6' is false but field widths differ); numeric-tuple
    # compare gets the real ordering right.
    assert version_ge("6.5.15", "2.12.30")
    assert version_ge("6.5.15", "6.5.15")
    assert version_ge("6.6.0", "6.5.15")
    assert version_ge("7.0.0", "6.5.15")
    assert not version_ge("2.12.30", "6.5.15")
    assert not version_ge("6.5.14", "6.5.15")


# ── glob + pick-highest discovery ───────────────────────────────────────

_PROBE_RE = re.compile(r"tool-(\d+\.\d+\.\d+)\.jar$")


def _probe(path: Path) -> str | None:
    m = _PROBE_RE.match(path.name)
    return m.group(1) if m else None


def test_glob_picks_highest_version(tmp_path, monkeypatch):
    home = tmp_path / "install"
    home.mkdir()
    (home / "tool-2.12.30.jar").write_text("x")
    (home / "tool-6.5.15.jar").write_text("x")
    spec = ToolSpec(
        name="fake-glob-highest",
        env_var="CONSTELLATION_FAKE_GLOB_HIGHEST_HOME",
        artifact_glob="tool-*.jar",
        pick="highest",
        version_probe=_probe,
    )
    register(spec)
    monkeypatch.setenv(spec.env_var, str(home))

    handle = find("fake-glob-highest")
    assert handle.path.name == "tool-6.5.15.jar"  # 6.5.15 > 2.12.30 numerically
    assert handle.version == "6.5.15"

    # Forward-compat: a newer jar wins with no code change.
    (home / "tool-6.6.0.jar").write_text("x")
    assert find("fake-glob-highest").path.name == "tool-6.6.0.jar"


def test_glob_unprobeable_does_not_shadow_real_version(tmp_path, monkeypatch):
    home = tmp_path / "install"
    home.mkdir()
    (home / "tool-nightly.jar").write_text("x")  # version_probe -> None
    (home / "tool-6.5.15.jar").write_text("x")
    spec = ToolSpec(
        name="fake-glob-shadow",
        env_var="CONSTELLATION_FAKE_GLOB_SHADOW_HOME",
        artifact_glob="tool-*.jar",
        pick="highest",
        version_probe=_probe,
    )
    register(spec)
    monkeypatch.setenv(spec.env_var, str(home))
    assert find("fake-glob-shadow").path.name == "tool-6.5.15.jar"


def test_glob_no_match_raises_not_home_dir(tmp_path, monkeypatch):
    # artifact=None + artifact_glob set: the [home]-is-the-artifact fallback
    # must be suppressed, so an empty home raises rather than returning the
    # directory itself (which run_jar would feed to `java -jar <dir>`).
    home = tmp_path / "install"
    home.mkdir()
    spec = ToolSpec(
        name="fake-glob-empty",
        env_var="CONSTELLATION_FAKE_GLOB_EMPTY_HOME",
        artifact_glob="tool-*.jar",
        pick="highest",
        install_script="scripts/install-x.sh",
        version_probe=_probe,
    )
    register(spec)
    monkeypatch.setenv(spec.env_var, str(home))
    with pytest.raises(ToolNotFoundError):
        find("fake-glob-empty")


def test_user_cache_home_resolves(tmp_path, monkeypatch):
    # ~/.constellation/<user_cache_dir>/current -> <ver>/tool-<ver>.jar,
    # discovered with no env var set (source == "user_cache").
    monkeypatch.setenv("HOME", str(tmp_path))
    base = tmp_path / ".constellation" / "fake-user-cache"
    (base / "6.5.15").mkdir(parents=True)
    (base / "6.5.15" / "tool-6.5.15.jar").write_text("x")
    (base / "current").symlink_to("6.5.15")
    spec = ToolSpec(
        name="fake-user-cache",
        env_var="CONSTELLATION_FAKE_USER_CACHE_HOME",
        artifact_glob="tool-*.jar",
        pick="highest",
        user_cache_dir="fake-user-cache",
        version_probe=_probe,
    )
    register(spec)
    monkeypatch.delenv(spec.env_var, raising=False)

    handle = find("fake-user-cache")
    assert handle.source == "user_cache"
    assert handle.path.name == "tool-6.5.15.jar"
    assert handle.version == "6.5.15"
