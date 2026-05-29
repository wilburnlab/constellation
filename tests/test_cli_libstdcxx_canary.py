"""Tests for the libstdc++ skew canary in ``constellation.cli.__main__``.

The canary's job is to detect, at process startup, whether the
process will resolve ``libstdc++.so.6`` against the conda env's copy or
a stale system copy. When skew is detected we re-exec with
``$CONDA_PREFIX/lib`` first on ``LD_LIBRARY_PATH``.

Two stages:

1. **sqlite3 import probe** — fires when importing sqlite3 raises an
   error whose message names libstdc++ / CXXABI / GLIBCXX. Catches the
   case where Python's *own* NEEDED chain dies at startup.
2. **path-resolution probe** — checks ``/proc/self/maps`` to see which
   on-disk libstdc++ is mapped into the process. Catches the more
   common case where startup succeeds but the loaded SONAME is from
   outside ``$CONDA_PREFIX``; any later ``dlopen`` (pythonnet → CoreCLR,
   matplotlib's compiled extensions, spawn-mode workers) reuses the
   wrong copy.

These tests monkeypatch the helpers' file-system surface so they can
run on any host without needing a real skew to be present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from constellation.cli.__main__ import (
    _detect_libstdcxx_skew,
    _ensure_env_lib_on_ld_library_path,
    _libstdcxx_loaded_outside_env,
)


def _null_log(msg: str) -> None:  # noqa: ARG001
    return None


# ──────────────────────────────────────────────────────────────────────
# _libstdcxx_loaded_outside_env
# ──────────────────────────────────────────────────────────────────────


class TestPathResolutionProbe:
    def test_no_conda_prefix_returns_none(self, monkeypatch):
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        assert _libstdcxx_loaded_outside_env(_null_log) is None

    def test_returns_none_when_path_is_inside_env(
        self, monkeypatch, tmp_path
    ):
        """When /proc/self/maps shows libstdc++ from $CONDA_PREFIX/lib,
        the probe says 'no skew'."""
        env_lib = tmp_path / "lib"
        env_lib.mkdir()
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        fake_maps = tmp_path / "fake_maps"
        fake_maps.write_text(
            f"7f00 r-xp 0 00:00 0 {env_lib}/libstdc++.so.6.0.32\n"
            f"7f01 r--p 0 00:00 0 {env_lib}/libstdc++.so.6.0.32\n"
        )

        # Pretend ctypes.CDLL succeeded so the probe proceeds to the
        # /proc/self/maps step.
        monkeypatch.setattr(
            "constellation.cli.__main__.ctypes" if False else "ctypes.CDLL",
            lambda name: None,
        )
        # Redirect Path("/proc/self/maps") to the fake.
        _patch_proc_maps(monkeypatch, fake_maps)

        assert _libstdcxx_loaded_outside_env(_null_log) is None

    def test_returns_path_when_outside_env(self, monkeypatch, tmp_path):
        """When /proc/self/maps shows libstdc++ from /usr/lib (not under
        $CONDA_PREFIX/lib), the probe returns the offending path."""
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        sys_path = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28"
        fake_maps = tmp_path / "fake_maps"
        fake_maps.write_text(
            f"7f00 r-xp 0 00:00 0 {sys_path}\n"
            f"7f01 r--p 0 00:00 0 {sys_path}\n"
        )

        monkeypatch.setattr("ctypes.CDLL", lambda name: None)
        _patch_proc_maps(monkeypatch, fake_maps)

        result = _libstdcxx_loaded_outside_env(_null_log)
        assert result == sys_path

    def test_ignores_pseudo_entries(self, monkeypatch, tmp_path):
        """Lines like [stack] / [vdso] / anonymous mappings must not
        be treated as libstdc++ paths."""
        env_lib = tmp_path / "lib"
        env_lib.mkdir()
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        # First mapping is pseudo (no path), second is the real one
        # under the env. Probe should return None.
        fake_maps = tmp_path / "fake_maps"
        fake_maps.write_text(
            "7f00 r-xp 0 00:00 0 [stack]\n"
            f"7f01 r-xp 0 00:00 0 {env_lib}/libstdc++.so.6\n"
        )

        monkeypatch.setattr("ctypes.CDLL", lambda name: None)
        _patch_proc_maps(monkeypatch, fake_maps)

        assert _libstdcxx_loaded_outside_env(_null_log) is None

    def test_ctypes_failure_returns_none(self, monkeypatch, tmp_path):
        """When libstdc++.so.6 cannot be loaded at all (rare), bail out."""
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

        def boom(name):
            raise OSError("no such library")

        monkeypatch.setattr("ctypes.CDLL", boom)
        assert _libstdcxx_loaded_outside_env(_null_log) is None

    def test_missing_proc_maps_returns_none(self, monkeypatch, tmp_path):
        """On non-Linux hosts /proc/self/maps doesn't exist — degrade
        silently."""
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        monkeypatch.setattr("ctypes.CDLL", lambda name: None)

        # Point the helper at a path that definitely doesn't exist.
        bogus = tmp_path / "definitely_not_proc_maps"
        _patch_proc_maps(monkeypatch, bogus)

        assert _libstdcxx_loaded_outside_env(_null_log) is None


def _patch_proc_maps(monkeypatch, fake_path: Path) -> None:
    """Redirect ``Path("/proc/self/maps")`` to ``fake_path`` for tests.

    The canary uses ``Path("/proc/self/maps")`` directly, so we
    monkeypatch :class:`Path` in the canary module to substitute the
    test fixture whenever that specific path is constructed.
    """
    import constellation.cli.__main__ as _main_mod
    real_path_cls = _main_mod.Path

    class _PatchedPath(type(Path())):
        def __new__(cls, *args, **kwargs):
            if args and str(args[0]) == "/proc/self/maps":
                return real_path_cls.__new__(real_path_cls, str(fake_path))
            return real_path_cls.__new__(real_path_cls, *args, **kwargs)

    # The canary's Path import is a module-level name, so patching it
    # via attribute substitution is the cleanest seam.
    def path_factory(*args, **kwargs):
        if args and str(args[0]) == "/proc/self/maps":
            return real_path_cls(str(fake_path))
        return real_path_cls(*args, **kwargs)

    monkeypatch.setattr(_main_mod, "Path", path_factory)


# ──────────────────────────────────────────────────────────────────────
# _detect_libstdcxx_skew (two-stage orchestrator)
# ──────────────────────────────────────────────────────────────────────


class TestDetectLibstdcxxSkew:
    def test_returns_none_when_both_stages_clean(self, monkeypatch):
        """sqlite3 imports fine AND path probe finds no skew → no re-exec."""
        # sqlite3 already imported (it's stdlib); leave alone.
        # Stub the path probe to return None.
        monkeypatch.setattr(
            "constellation.cli.__main__._libstdcxx_loaded_outside_env",
            lambda log: None,
        )
        assert _detect_libstdcxx_skew(_null_log) is None

    def test_sqlite3_libstdcxx_error_short_circuits(self, monkeypatch):
        """sqlite3 import raising a libstdc++-flavoured error returns a
        reason string immediately, regardless of the path probe."""
        # Force the sqlite3 import inside the canary to raise the
        # canonical libstdc++ error message.
        real_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict
        ) else __import__

        def fake_import(name, *args, **kwargs):
            if name == "sqlite3":
                raise ImportError(
                    "/lib/libstdc++.so.6: version `CXXABI_1.3.15' not found"
                )
            return real_import(name, *args, **kwargs)

        # Path probe should not be reached when sqlite3 fires.
        called = {"path_probe": False}

        def fake_path_probe(log):
            called["path_probe"] = True
            return None

        monkeypatch.setitem(sys.modules, "sqlite3", None)  # force re-import
        monkeypatch.setattr("builtins.__import__", fake_import)
        monkeypatch.setattr(
            "constellation.cli.__main__._libstdcxx_loaded_outside_env",
            fake_path_probe,
        )

        reason = _detect_libstdcxx_skew(_null_log)
        assert reason is not None
        assert "sqlite3 import failed" in reason
        assert "CXXABI_1.3.15" in reason
        assert called["path_probe"] is False

    def test_non_libstdcxx_sqlite3_error_falls_through_to_path_probe(
        self, monkeypatch
    ):
        """A non-libstdc++ sqlite3 ImportError should NOT short-circuit
        the path probe — sqlite3 might be genuinely broken AND we might
        still have a path skew worth fixing."""
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "sqlite3":
                raise ImportError("totally unrelated error")
            return real_import(name, *args, **kwargs)

        called = {"path_probe": False}

        def fake_path_probe(log):
            called["path_probe"] = True
            return None

        monkeypatch.setitem(sys.modules, "sqlite3", None)
        monkeypatch.setattr("builtins.__import__", fake_import)
        monkeypatch.setattr(
            "constellation.cli.__main__._libstdcxx_loaded_outside_env",
            fake_path_probe,
        )

        result = _detect_libstdcxx_skew(_null_log)
        assert result is None
        assert called["path_probe"] is True

    def test_path_probe_skew_triggers_re_exec(self, monkeypatch):
        """sqlite3 OK, path probe finds out-of-env libstdc++ → reason
        string returned so the caller can re-exec."""
        offender = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28"
        monkeypatch.setattr(
            "constellation.cli.__main__._libstdcxx_loaded_outside_env",
            lambda log: offender,
        )

        reason = _detect_libstdcxx_skew(_null_log)
        assert reason is not None
        assert offender in reason


# ──────────────────────────────────────────────────────────────────────
# Integration: _ensure_env_libstdcxx_priority decides whether to call
# the re-exec helper based on _detect_libstdcxx_skew's verdict
# ──────────────────────────────────────────────────────────────────────


class TestEnsureEnvLibOnLdLibraryPath:
    """The LD_LIBRARY_PATH prep helper — mutates os.environ in the
    parent so spawn children inherit the corrected search order."""

    def test_no_conda_prefix_no_mutation(self, monkeypatch):
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/sentinel/value")
        _ensure_env_lib_on_ld_library_path(_null_log)
        assert os.environ["LD_LIBRARY_PATH"] == "/sentinel/value"

    def test_no_env_libstdcxx_no_mutation(self, monkeypatch, tmp_path):
        """When the conda env doesn't ship its own libstdc++ (rare), the
        prep is a no-op."""
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        (tmp_path / "lib").mkdir()
        # No libstdc++.so.6 in tmp_path/lib.
        monkeypatch.setenv("LD_LIBRARY_PATH", "/sentinel/value")
        _ensure_env_lib_on_ld_library_path(_null_log)
        assert os.environ["LD_LIBRARY_PATH"] == "/sentinel/value"

    def test_prepends_env_lib_when_path_empty(self, monkeypatch, tmp_path):
        env_lib = tmp_path / "lib"
        env_lib.mkdir()
        (env_lib / "libstdc++.so.6").write_bytes(b"")  # marker only
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

        _ensure_env_lib_on_ld_library_path(_null_log)

        assert os.environ["LD_LIBRARY_PATH"] == str(env_lib)

    def test_prepends_env_lib_when_other_first(self, monkeypatch, tmp_path):
        """Simulates the HPC case: Spack lib is first on LD_LIBRARY_PATH,
        env's lib is not present. After prep, env's lib is first."""
        env_lib = tmp_path / "lib"
        env_lib.mkdir()
        (env_lib / "libstdc++.so.6").write_bytes(b"")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        monkeypatch.setenv(
            "LD_LIBRARY_PATH", "/apps/spack/.../lib:/opt/somewhere/lib"
        )

        _ensure_env_lib_on_ld_library_path(_null_log)

        assert os.environ["LD_LIBRARY_PATH"] == (
            f"{env_lib}:/apps/spack/.../lib:/opt/somewhere/lib"
        )

    def test_already_first_is_noop(self, monkeypatch, tmp_path):
        env_lib = tmp_path / "lib"
        env_lib.mkdir()
        (env_lib / "libstdc++.so.6").write_bytes(b"")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        existing = f"{env_lib}:/apps/spack/.../lib"
        monkeypatch.setenv("LD_LIBRARY_PATH", existing)

        _ensure_env_lib_on_ld_library_path(_null_log)

        assert os.environ["LD_LIBRARY_PATH"] == existing

    def test_dedupes_when_env_lib_present_but_not_first(
        self, monkeypatch, tmp_path
    ):
        """Existing env-lib entry later on the path gets moved to front,
        not duplicated."""
        env_lib = tmp_path / "lib"
        env_lib.mkdir()
        (env_lib / "libstdc++.so.6").write_bytes(b"")
        monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
        monkeypatch.setenv(
            "LD_LIBRARY_PATH", f"/apps/spack/.../lib:{env_lib}:/opt/other/lib"
        )

        _ensure_env_lib_on_ld_library_path(_null_log)

        result = os.environ["LD_LIBRARY_PATH"]
        # env_lib appears at the start exactly once.
        assert result == f"{env_lib}:/apps/spack/.../lib:/opt/other/lib"
        assert result.count(str(env_lib)) == 1


class TestEnsureEnvLibstdcxxPriority:
    def test_sentinel_set_skips_skew_detection_but_still_preps_path(
        self, monkeypatch
    ):
        """Sentinel skips ONLY the skew probes (re-exec already happened).
        LD_LIBRARY_PATH prep still runs so children inherit correct env
        even after re-exec."""
        from constellation.cli.__main__ import (
            _LIBSTDCXX_REEXEC_SENTINEL,
            _ensure_env_libstdcxx_priority,
        )

        monkeypatch.setenv(_LIBSTDCXX_REEXEC_SENTINEL, "1")
        called = {"reexec": False, "detect": False, "prep": False}
        monkeypatch.setattr(
            "constellation.cli.__main__._detect_libstdcxx_skew",
            lambda log: called.__setitem__("detect", True) or None,
        )
        monkeypatch.setattr(
            "constellation.cli.__main__._reexec_with_env_libstdcxx_first",
            lambda log: called.__setitem__("reexec", True),
        )
        monkeypatch.setattr(
            "constellation.cli.__main__._ensure_env_lib_on_ld_library_path",
            lambda log: called.__setitem__("prep", True),
        )
        _ensure_env_libstdcxx_priority()
        assert called == {"reexec": False, "detect": False, "prep": True}

    def test_no_skew_no_reexec_but_still_preps_path(self, monkeypatch):
        from constellation.cli.__main__ import (
            _LIBSTDCXX_REEXEC_SENTINEL,
            _ensure_env_libstdcxx_priority,
        )

        monkeypatch.delenv(_LIBSTDCXX_REEXEC_SENTINEL, raising=False)
        called = {"reexec": False, "prep": False}
        monkeypatch.setattr(
            "constellation.cli.__main__._detect_libstdcxx_skew",
            lambda log: None,
        )
        monkeypatch.setattr(
            "constellation.cli.__main__._reexec_with_env_libstdcxx_first",
            lambda log: called.__setitem__("reexec", True),
        )
        monkeypatch.setattr(
            "constellation.cli.__main__._ensure_env_lib_on_ld_library_path",
            lambda log: called.__setitem__("prep", True),
        )
        _ensure_env_libstdcxx_priority()
        assert called["reexec"] is False
        assert called["prep"] is True

    def test_skew_triggers_reexec(self, monkeypatch, capsys):
        from constellation.cli.__main__ import (
            _LIBSTDCXX_REEXEC_SENTINEL,
            _ensure_env_libstdcxx_priority,
        )

        monkeypatch.delenv(_LIBSTDCXX_REEXEC_SENTINEL, raising=False)
        called = {"reexec": False, "prep": False}
        monkeypatch.setattr(
            "constellation.cli.__main__._detect_libstdcxx_skew",
            lambda log: "libstdc++.so.6 mapped from /usr/lib/.../libstdc++.so.6",
        )
        monkeypatch.setattr(
            "constellation.cli.__main__._reexec_with_env_libstdcxx_first",
            lambda log: called.__setitem__("reexec", True),
        )
        monkeypatch.setattr(
            "constellation.cli.__main__._ensure_env_lib_on_ld_library_path",
            lambda log: called.__setitem__("prep", True),
        )
        _ensure_env_libstdcxx_priority()
        captured = capsys.readouterr()
        assert called["reexec"] is True
        assert called["prep"] is True
        assert "libstdc++ ABI skew" in captured.err
        assert "mapped from /usr/lib" in captured.err


# ──────────────────────────────────────────────────────────────────────
# Live host: surface what the current host's state actually is — the
# probe should run cleanly and not crash. We don't assert one outcome
# because dev/CI/HPC hosts will legitimately differ.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not Path("/proc/self/maps").is_file(),
    reason="not Linux / no /proc/self/maps",
)
def test_smoke_real_host_does_not_crash():
    """End-to-end on the live host. We don't assert which branch fires
    (clean dev host: None; HPC with skew: a string) — just that the
    probe completes."""
    result = _libstdcxx_loaded_outside_env(_null_log)
    # Either outcome is fine; just ensure the type is sensible.
    assert result is None or isinstance(result, str)
    if result is not None:
        assert result.startswith("/")
        # Confirm it really is outside the env (probe's own postcondition).
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            assert not result.startswith(str(Path(conda_prefix) / "lib"))
