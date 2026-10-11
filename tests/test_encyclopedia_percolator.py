"""The Percolator pre-flight for EncyclopeDIA searches.

The jar's default Linux Percolator (3.01) is killed by a kernel booted
with ``vsyscall=none`` (WSL2), and the jar only finds out after scoring
the whole run. ``percolator_blocker`` is what makes that a one-second
refusal with the fixes listed; these tests pin when it speaks up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from constellation.massspec.search.encyclopedia import percolator
from constellation.massspec.search.encyclopedia.cli_spec import flag_default
from constellation.massspec.search.encyclopedia.percolator import (
    default_percolator_version,
    effective_percolator,
    legacy_vsyscall_mapped,
    percolator_blocker,
)


@pytest.fixture
def affected_kernel(monkeypatch):
    """Linux x86-64 booted with ``vsyscall=none``."""
    monkeypatch.setattr(percolator, "_is_linux_x86_64", lambda: True)
    monkeypatch.setattr(percolator, "legacy_vsyscall_mapped", lambda: False)


def test_default_version_is_read_from_the_captured_help() -> None:
    assert default_percolator_version() == flag_default("encyclopedia", "-percolatorVersion")
    assert default_percolator_version() == "v3-01"


@pytest.mark.parametrize(
    ("maps", "expected"),
    [
        ("7ffd000-7ffd100 r-xp 00000000 00:00 0  [vdso]\n", False),
        (
            "7ffd000-7ffd100 r-xp 00000000 00:00 0  [vdso]\n"
            "ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0  [vsyscall]\n",
            True,
        ),
    ],
)
def test_legacy_vsyscall_mapped_reads_the_memory_map(
    monkeypatch, tmp_path: Path, maps: str, expected: bool
) -> None:
    fake = tmp_path / "maps"
    fake.write_text(maps)
    monkeypatch.setattr(percolator, "_PROC_MAPS", fake)
    assert legacy_vsyscall_mapped() is expected


def test_legacy_vsyscall_mapped_is_unknown_without_proc(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(percolator, "_PROC_MAPS", tmp_path / "absent")
    assert legacy_vsyscall_mapped() is None


@pytest.mark.parametrize("version", [None, "v3-01", "V3-01", "v3p01"])
def test_default_build_is_blocked_on_a_vsyscall_none_kernel(affected_kernel, version) -> None:
    message = percolator_blocker(version)
    assert message is not None
    # Names the symptom the user would otherwise have seen, and each way out.
    assert "status: 139" in message
    assert "vsyscall" in message
    assert "--percolator-version v3-05" in message
    assert "--percolator-version /path/to/percolator" in message


@pytest.mark.parametrize("version", ["v3-05", "v2-10", "/opt/percolator/bin/percolator"])
def test_other_builds_and_paths_are_not_blocked(affected_kernel, version) -> None:
    assert percolator_blocker(version) is None


def test_kernel_fix_is_worded_for_the_host(affected_kernel, monkeypatch) -> None:
    monkeypatch.setattr(percolator, "is_wsl", lambda: True)
    assert ".wslconfig" in percolator_blocker()
    monkeypatch.setattr(percolator, "is_wsl", lambda: False)
    message = percolator_blocker()
    assert ".wslconfig" not in message and "vsyscall=xonly" in message


@pytest.mark.parametrize("mapped", [True, None])
def test_not_blocked_when_the_page_is_mapped_or_unknown(monkeypatch, mapped) -> None:
    monkeypatch.setattr(percolator, "_is_linux_x86_64", lambda: True)
    monkeypatch.setattr(percolator, "legacy_vsyscall_mapped", lambda: mapped)
    assert percolator_blocker() is None


def test_not_blocked_off_linux(monkeypatch) -> None:
    monkeypatch.setattr(percolator, "_is_linux_x86_64", lambda: False)
    monkeypatch.setattr(percolator, "legacy_vsyscall_mapped", lambda: False)
    assert percolator_blocker() is None


def test_pass_through_arguments_decide_the_build(affected_kernel) -> None:
    # A pass-through -percolatorVersion overrides the typed option...
    assert effective_percolator("v3-01", ["-percolatorVersion", "v3-05"]) == "v3-05"
    assert percolator_blocker("v3-01", ["-percolatorVersion", "v3-05"]) is None
    assert percolator_blocker("v3-05", ["-percolatorVersion", "v3-01"]) is not None
    # ...and a run with Percolator switched off has nothing to block.
    assert effective_percolator(None, ["-usePercolator", "false"]) is None
    assert percolator_blocker(None, ["-usePercolator", "false"]) is None
    assert percolator_blocker(None, ["-usePercolator", "true"]) is not None


def test_search_refuses_before_starting_the_jar(
    affected_kernel, monkeypatch, tmp_path: Path, capsys
) -> None:
    """``massspec search`` stops at the pre-flight: no JVM, no output files."""
    import constellation.massspec.search.encyclopedia as enc
    from constellation.cli.__main__ import main
    from constellation.thirdparty import registry

    spec = next(s for s in registry.registered() if s.name == "encyclopedia")
    handle = registry.ToolHandle(spec, tmp_path / "encyclopedia-6.9.23.jar", "env", "6.9.23")
    monkeypatch.setattr(registry, "find", lambda name: handle)

    def _must_not_run(**_kwargs):
        raise AssertionError("the jar was started despite the Percolator pre-flight")

    monkeypatch.setattr(enc, "run_library_search", _must_not_run)

    for name in ("run.dia", "lib.dlib", "db.fasta"):
        (tmp_path / name).touch()
    argv = [
        "massspec",
        "search",
        "--mzml",
        str(tmp_path / "run.dia"),
        "--library",
        str(tmp_path / "lib.dlib"),
        "--fasta",
        str(tmp_path / "db.fasta"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "bundled Percolator 3.01 cannot run on this kernel" in err

    # Choosing a build that runs here gets past the pre-flight.
    with pytest.raises(AssertionError, match="the jar was started"):
        main([*argv, "--percolator-version", "v3-05"])
