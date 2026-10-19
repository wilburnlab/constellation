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
import tempfile
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


def test_run_jar_prefers_bundled_jre_when_spec_asks(
    hello_jar: Path, tmp_path: Path
) -> None:
    """``prefer_bundled_jre`` swaps the order: the JRE beside the jar wins
    even though a system ``java`` is on PATH (EncyclopeDIA's case — it
    ships the Java it was built against and warns outside 11-21)."""
    import os

    from constellation.thirdparty.registry import _REGISTRY, ToolSpec, register

    home = tmp_path / "bundled-home"
    home.mkdir()
    jar = home / hello_jar.name
    jar.write_bytes(hello_jar.read_bytes())
    bundled = home / "jre" / "bin" / "java"
    bundled.parent.mkdir(parents=True)
    bundled.symlink_to(Path(shutil.which("java")))

    name, env_var = "_test_hello_bundled", "_TEST_HELLO_BUNDLED_HOME"
    register(ToolSpec(name=name, env_var=env_var, artifact=jar.name, prefer_bundled_jre=True))
    os.environ[env_var] = str(home)
    try:
        result = run_jar(name, args=[], log_dir=tmp_path / "logs", stream_to_stderr=False)
    finally:
        os.environ.pop(env_var, None)
        _REGISTRY.pop(name, None)
    assert result.returncode == 0
    assert result.java_source == "bundled"
    assert result.java_path == bundled


def test_run_jar_headless_default(register_hello_tool, tmp_path: Path) -> None:
    """``-Djava.awt.headless=true`` is on every argv by default so AWT-using
    libraries (JFreeChart, etc.) work on compute nodes without an X11
    display. Regression test for the EncyclopeDIA `AWTError: Can't connect
    to X11` failure observed in SLURM batch runs."""
    result = run_jar(
        register_hello_tool,
        args=[],
        log_dir=tmp_path / "logs",
        stream_to_stderr=False,
    )
    assert "-Djava.awt.headless=true" in result.argv
    # Flag must precede -jar so the JVM applies it before AWT init
    headless_idx = result.argv.index("-Djava.awt.headless=true")
    jar_idx = result.argv.index("-jar")
    assert headless_idx < jar_idx


def test_run_jar_headless_can_be_overridden(
    register_hello_tool, tmp_path: Path
) -> None:
    """Caller passing ``-Djava.awt.headless=false`` via extra_jvm_args wins
    over the default — later -D flags override earlier ones in the JVM's
    system-property resolution."""
    result = run_jar(
        register_hello_tool,
        args=[],
        extra_jvm_args=["-Djava.awt.headless=false"],
        log_dir=tmp_path / "logs",
        stream_to_stderr=False,
    )
    # Both flags appear; the override comes after the default
    default_idx = result.argv.index("-Djava.awt.headless=true")
    override_idx = result.argv.index("-Djava.awt.headless=false")
    assert default_idx < override_idx


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


# ── cleaning up after the JVM ──────────────────────────────────────────


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _gone_within(pid: int, seconds: float = 5.0) -> bool:
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _leader_with_helper(script: str):
    """Start ``bash -c script`` in its own session; return it and the pid it prints."""
    import subprocess

    leader = subprocess.Popen(
        ["bash", "-c", script], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    return leader, int(leader.stdout.readline())


def test_reap_process_group_stops_what_the_leader_left_behind() -> None:
    """EncyclopeDIA's failure mode: the JVM exits and the server it
    started keeps running. The leader's own exit status survives."""
    from constellation.thirdparty.jvm import _wait_for_exit, reap_process_group

    # setsid-free helper: stays in the leader's group, ignores its exit.
    leader, helper = _leader_with_helper("sleep 300 & echo $! ; exit 7")
    _wait_for_exit(leader)
    assert _alive(helper)

    assert reap_process_group(leader) == 7
    assert _gone_within(helper), "orphaned helper survived"
    # Nothing left: a second call is a no-op that still knows the status.
    assert reap_process_group(leader) == 7


def test_reap_process_group_abandons_a_running_leader() -> None:
    """The Ctrl-C / exception path: the leader is still alive."""
    import signal

    from constellation.thirdparty.jvm import reap_process_group

    leader, helper = _leader_with_helper("sleep 300 & echo $! ; wait")
    assert leader.poll() is None

    assert reap_process_group(leader) == -signal.SIGTERM
    assert _gone_within(helper)


def test_reap_process_group_escalates_to_sigkill() -> None:
    """A helper that ignores SIGTERM is killed after the grace period."""
    from constellation.thirdparty.jvm import _wait_for_exit, reap_process_group

    # The subshell ignores SIGTERM, and so does the sleep it becomes.
    leader, helper = _leader_with_helper(
        '(trap "" TERM; echo $BASHPID; exec sleep 300) & sleep 0.3; exit 0'
    )
    _wait_for_exit(leader)
    assert _alive(helper)

    assert reap_process_group(leader, grace_s=0.5) == 0
    assert _gone_within(helper), "SIGTERM-immune helper survived"


def test_capture_jar_returns_output_and_cleans_up(register_hello_tool, tmp_path: Path) -> None:
    from constellation.thirdparty.jvm import _resolve_java, capture_jar
    from constellation.thirdparty.registry import find

    handle = find(register_hello_tool)
    java, _source, _version = _resolve_java(handle)
    before = {p.name for p in Path(tempfile.gettempdir()).glob("constellation-jvm-*")}
    out = capture_jar(java, handle.path, ["one", "two"], timeout=120, cwd=tmp_path)
    assert out.strip()
    after = {p.name for p in Path(tempfile.gettempdir()).glob("constellation-jvm-*")}
    assert after <= before


def test_run_jar_private_tmpdir_is_removed(register_hello_tool, tmp_path: Path) -> None:
    """Without ``jvm_tmpdir`` the run gets its own java.io.tmpdir and it
    is gone afterwards; a caller-chosen one is left alone."""
    result = run_jar(
        register_hello_tool, args=[], log_dir=tmp_path / "logs", stream_to_stderr=False
    )
    (flag,) = [a for a in result.argv if a.startswith("-Djava.io.tmpdir=")]
    private = Path(flag.split("=", 1)[1])
    assert private.name.startswith("constellation-jvm-") and not private.exists()

    chosen = tmp_path / "java-tmp"
    chosen.mkdir()
    result = run_jar(
        register_hello_tool,
        args=[],
        jvm_tmpdir=chosen,
        log_dir=tmp_path / "logs2",
        stream_to_stderr=False,
    )
    assert f"-Djava.io.tmpdir={chosen}" in result.argv and chosen.is_dir()


# ── cancelling the wrapper stops the JVM ───────────────────────────────
# (Regression from the review of b1fe1ca. The JVM runs in its own
# session, so a signal to the wrapper no longer reaches it, and the
# default SIGTERM action kills Python without running any `finally`:
# a cancelled search kept running, with its helper and its temp files.)

_FAKE_JAVA = """#!/usr/bin/env bash
if [ "$1" = "-version" ]; then echo 'openjdk version "21-fake"' >&2; exit 0; fi
for a in "$@"; do case "$a" in -Djava.io.tmpdir=*) tmp="${a#-Djava.io.tmpdir=}";; esac; done
sleep 300 &
echo "$$ $! $tmp" > "$FAKE_JAVA_PIDFILE"
wait
"""

_WRAPPER = """
import sys, threading
from pathlib import Path
from constellation.thirdparty.jvm import cancellation_scope, run_jar
from constellation.thirdparty.registry import ToolSpec, register

register(ToolSpec(name="_fake", env_var="_FAKE_HOME", artifact="fake-1.0.0.jar",
                  prefer_bundled_jre=True))

def search():
    run_jar("_fake", args=[], log_dir=Path(sys.argv[1]), stream_to_stderr=False, check=False)

if sys.argv[2] == "main":
    search()
else:  # the orchestrated shape: jars in worker threads, the scope in the main thread
    with cancellation_scope():
        worker = threading.Thread(target=search)
        worker.start()
        while worker.is_alive():
            worker.join(0.1)
"""


def _start_wrapped_fake_java(tmp_path: Path, mode: str):
    """A wrapper process running a fake Java (which starts a helper of
    its own) through ``run_jar``. Returns the wrapper and the pids and
    temp directory the fake Java reported."""
    import os
    import sys
    import time

    import constellation

    home = tmp_path / "tool"
    (home / "jre" / "bin").mkdir(parents=True)
    (home / "fake-1.0.0.jar").write_bytes(b"")
    java = home / "jre" / "bin" / "java"
    java.write_text(_FAKE_JAVA)
    java.chmod(0o755)
    pidfile = tmp_path / "pids"
    wrapper = subprocess.Popen(
        [sys.executable, "-c", _WRAPPER, str(tmp_path / "logs"), mode],
        env={**os.environ, "_FAKE_HOME": str(home), "FAKE_JAVA_PIDFILE": str(pidfile)},
        cwd=Path(constellation.__file__).parents[1],
    )
    deadline = time.monotonic() + 30
    while not (pidfile.is_file() and pidfile.read_text().strip()):
        assert wrapper.poll() is None, "the wrapper exited before starting the fake Java"
        assert time.monotonic() < deadline, "the fake Java never started"
        time.sleep(0.05)
    java_pid, helper_pid, java_tmp = pidfile.read_text().split()
    return wrapper, int(java_pid), int(helper_pid), Path(java_tmp)


@pytest.mark.parametrize("mode", ["main", "thread"])
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_signalling_the_wrapper_stops_the_jvm_and_cleans_up(
    tmp_path: Path, mode: str, signame: str
) -> None:
    import os
    import signal

    signum = getattr(signal, signame)
    wrapper, java_pid, helper_pid, java_tmp = _start_wrapped_fake_java(tmp_path, mode)
    try:
        assert _alive(java_pid) and _alive(helper_pid) and java_tmp.is_dir()
        wrapper.send_signal(signum)  # the wrapper only, not its process group
        assert wrapper.wait(timeout=30) == 128 + signum
        assert _gone_within(java_pid), "the JVM outlived its cancelled wrapper"
        assert _gone_within(helper_pid), "the JVM's helper outlived the cancelled wrapper"
        assert not java_tmp.exists(), "the run's private temp directory was left behind"
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
        try:
            os.killpg(java_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_cancellation_scope_installs_and_restores_handlers() -> None:
    import signal

    from constellation.thirdparty import jvm

    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    with jvm.cancellation_scope():
        assert signal.getsignal(signal.SIGTERM) is jvm._on_cancel_signal
        with jvm.cancellation_scope():  # re-entrant
            assert signal.getsignal(signal.SIGTERM) is jvm._on_cancel_signal
        assert signal.getsignal(signal.SIGTERM) is jvm._on_cancel_signal
    assert {s: signal.getsignal(s) for s in before} == before


def test_cancellation_scope_leaves_an_ignored_signal_ignored() -> None:
    """``nohup constellation ...`` ignores SIGHUP; that must stay so."""
    import signal

    from constellation.thirdparty import jvm

    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with jvm.cancellation_scope():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) is jvm._on_cancel_signal
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_cancellation_scope_is_inert_off_the_main_thread() -> None:
    """Handlers can only be set from the main thread; a scope entered
    elsewhere changes nothing and does not raise."""
    import signal
    import threading

    from constellation.thirdparty import jvm

    seen: list[object] = []

    def enter() -> None:
        with jvm.cancellation_scope():
            seen.append(signal.getsignal(signal.SIGTERM))

    before = signal.getsignal(signal.SIGTERM)
    thread = threading.Thread(target=enter)
    thread.start()
    thread.join()
    assert seen == [before]
