"""Unit tests for the third-party tool registry."""

import pytest

from constellation.thirdparty import ToolNotFoundError, find
from constellation.thirdparty.registry import ToolSpec, register, registered, try_find


def test_encyclopedia_is_registered():
    names = {s.name for s in registered()}
    assert "encyclopedia" in names


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
