"""Tier A tests for :mod:`constellation.thirdparty.jvm`.

We compile a trivial ``Hello.java`` to a jar at test-setup time and
exercise :func:`run_jar` against it. This gives end-to-end coverage of:

  * argv composition (``-Xmx`` ordering, ``-jar`` placement, ``-Xms`` /
    ``-Djava.io.tmpdir`` / extra-JVM-args inclusion)
  * exit-code propagation (jar returns 7 → ``JvmRunError.returncode`` = 7)
  * stdout-vs-stderr separation (jar writes "STDOUT" / "STDERR" → two
    distinct log files capture them)
  * log-file capture (stdout.log + stderr.log + java-version.txt land in
    ``log_dir``)
  * ``JvmNotFoundError`` raised when ``shutil.which("java")`` returns
    None AND no bundled JRE alongside the jar.

Skips if ``javac`` is not on PATH — the build_a_jar fixture compiles
real Java, no mocking. The conda env ships openjdk (which includes
``javac``), so the test runs in CI and on dev workstations alike.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from constellation.thirdparty.jvm import (
    JvmNotFoundError,
    JvmResult,
    JvmRunError,
    run_jar,
)
from constellation.thirdparty.registry import ToolSpec, _REGISTRY, register


pytestmark = pytest.mark.skipif(
    shutil.which("javac") is None or shutil.which("java") is None,
    reason="javac/java not on PATH — Tier A jvm tests need a real JDK",
)


# ── fixtures ────────────────────────────────────────────────────────────


_JAVA_SRC = r"""
public class Hello {
    public static void main(String[] args) {
        System.out.println("STDOUT:hello");
        System.err.println("STDERR:hello");
        if (args.length > 0) {
            if (args[0].equals("fail")) {
                System.err.println("STDERR:forced failure");
                System.exit(7);
            }
            if (args[0].equals("echo")) {
                for (int i = 1; i < args.length; i++) {
                    System.out.println("ECHO:" + args[i]);
                }
            }
        }
        System.exit(0);
    }
}
""".strip()


_MANIFEST = "Manifest-Version: 1.0\nMain-Class: Hello\n"


@pytest.fixture
def hello_jar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile a Hello.java → Hello.jar in a session-scoped tmp dir."""
    work = tmp_path_factory.mktemp("jvm_jar")
    (work / "Hello.java").write_text(_JAVA_SRC)
    (work / "manifest.txt").write_text(_MANIFEST)

    subprocess.run(
        ["javac", "Hello.java"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["jar", "cfm", "Hello.jar", "manifest.txt", "Hello.class"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    jar_path = work / "Hello.jar"
    assert jar_path.is_file()
    return jar_path


@pytest.fixture
def register_hello_tool(hello_jar: Path):
    """Register ``hello`` in the thirdparty registry pointing at the jar.

    Uses a per-test tool name so parallel runs don't collide. The
    artifact is the jar filename; HOME is the jar's parent directory.
    Restores the registry on teardown.
    """
    name = "_test_hello"
    env_var = "_TEST_HELLO_HOME"
    import os

    spec = ToolSpec(
        name=name,
        env_var=env_var,
        artifact=hello_jar.name,
    )
    saved = _REGISTRY.get(name)
    register(spec)
    os.environ[env_var] = str(hello_jar.parent)
    try:
        yield name
    finally:
        os.environ.pop(env_var, None)
        if saved is None:
            _REGISTRY.pop(name, None)
        else:
            _REGISTRY[name] = saved


# ── tests ───────────────────────────────────────────────────────────────


def test_run_jar_success(register_hello_tool, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    result = run_jar(
        register_hello_tool,
        args=[],
        log_dir=log_dir,
        stream_to_stderr=False,
    )
    assert isinstance(result, JvmResult)
    assert result.returncode == 0
    assert result.elapsed_seconds >= 0
    assert log_dir.exists()
    assert (log_dir / "stdout.log").read_text().strip() == "STDOUT:hello"
    assert (log_dir / "stderr.log").read_text().strip() == "STDERR:hello"
    assert (log_dir / "java-version.txt").exists()
    assert result.java_source == "system"
    # jar_sha256 is 64 hex chars
    assert len(result.jar_sha256) == 64
    assert all(c in "0123456789abcdef" for c in result.jar_sha256)


def test_run_jar_argv_composition(
    register_hello_tool, tmp_path: Path
) -> None:
    """``-Xmx`` precedes ``-jar``; tool args follow the jar path."""
    result = run_jar(
        register_hello_tool,
        args=["echo", "alpha", "beta"],
        jvm_heap_max="256m",
        jvm_heap_min="64m",
        jvm_tmpdir=tmp_path / "java-tmp",
        extra_jvm_args=["-Dconstellation.test=true"],
        log_dir=tmp_path / "logs",
        stream_to_stderr=False,
    )
    argv = result.argv
    # java path → JVM flags → -jar <jar> → tool args
    assert Path(argv[0]).name == "java"
    assert "-Xmx256m" in argv
    assert "-Xms64m" in argv
    assert any(a.startswith("-Djava.io.tmpdir=") for a in argv)
    assert "-Dconstellation.test=true" in argv
    jar_idx = argv.index("-jar")
    # All JVM flags must come before -jar
    for flag in ("-Xmx256m", "-Xms64m", "-Dconstellation.test=true"):
        assert argv.index(flag) < jar_idx
    # Tool args come after the jar path
    assert argv[jar_idx + 2 :] == ["echo", "alpha", "beta"]

    stdout = (tmp_path / "logs" / "stdout.log").read_text()
    assert "ECHO:alpha" in stdout
    assert "ECHO:beta" in stdout


def test_run_jar_nonzero_raises(
    register_hello_tool, tmp_path: Path
) -> None:
    with pytest.raises(JvmRunError) as exc:
        run_jar(
            register_hello_tool,
            args=["fail"],
            log_dir=tmp_path / "logs",
            stream_to_stderr=False,
        )
    err = exc.value
    assert err.returncode == 7
    # Failure message embeds the stderr tail
    assert "forced failure" in err.stderr_tail
    assert err.stderr_log == tmp_path / "logs" / "stderr.log"
    # And the log file still got written
    assert (tmp_path / "logs" / "stderr.log").exists()


def test_run_jar_check_false_returns_result(
    register_hello_tool, tmp_path: Path
) -> None:
    result = run_jar(
        register_hello_tool,
        args=["fail"],
        log_dir=tmp_path / "logs",
        stream_to_stderr=False,
        check=False,
    )
    assert result.returncode == 7


def test_run_jar_no_java_raises_jvmnotfounderror(
    register_hello_tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``shutil.which('java')`` returns None → JvmNotFoundError when no
    bundled JRE alongside the jar."""
    monkeypatch.setattr(
        "constellation.thirdparty.jvm.shutil.which",
        lambda name: None,
    )
    with pytest.raises(JvmNotFoundError):
        run_jar(
            register_hello_tool,
            args=[],
            log_dir=tmp_path / "logs",
            stream_to_stderr=False,
        )


def test_run_jar_bundled_jre_fallback(
    register_hello_tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no system java BUT ``<jar_parent>/jre/bin/java`` exists,
    the runner uses the bundled JRE.

    Synthesises a 'bundled' JRE by symlinking the real java into the
    expected ``jre/bin/java`` slot next to the jar.
    """
    from constellation.thirdparty.registry import find

    real_java = Path(shutil.which("java"))
    handle = find(register_hello_tool)
    jar_parent = handle.path.parent
    bundled_dir = jar_parent / "jre" / "bin"
    bundled_dir.mkdir(parents=True, exist_ok=True)
    bundled_java = bundled_dir / "java"
    if not bundled_java.exists():
        bundled_java.symlink_to(real_java)

    monkeypatch.setattr(
        "constellation.thirdparty.jvm.shutil.which",
        lambda name: None,
    )
    result = run_jar(
        register_hello_tool,
        args=[],
        log_dir=tmp_path / "logs",
        stream_to_stderr=False,
    )
    assert result.returncode == 0
    assert result.java_source == "bundled"
    assert result.java_path == bundled_java


def test_run_jar_log_dir_created(
    register_hello_tool, tmp_path: Path
) -> None:
    deep = tmp_path / "nested" / "logs"
    assert not deep.exists()
    run_jar(
        register_hello_tool,
        args=[],
        log_dir=deep,
        stream_to_stderr=False,
    )
    assert deep.is_dir()
    assert (deep / "stdout.log").exists()
    assert (deep / "stderr.log").exists()
