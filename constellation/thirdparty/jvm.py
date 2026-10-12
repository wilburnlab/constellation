"""Generic JVM subprocess runner — the first jar-execution path in Constellation.

Composes a :class:`~constellation.thirdparty.registry.ToolHandle` (from
``registry.find(name)``) with subprocess machinery. Any future Java tool
(EncyclopeDIA, MSRawJava as a standalone, MSFragger, ...) consumes this
primitive; domain modules import it but never the other way round.

The runner is generic over the `java -jar <jar> ...args` shape and the
`java -cp <jar> <Main> ...args` shape. Jar discovery and version probing
stay in the tool-specific adapter; this module is only responsible for
deciding *which* `java` binary to invoke (system → bundled-JRE fallback),
how to compose JVM flags, how to spawn + stream the subprocess, and how
to surface failures in a user-actionable form.

Java resolution order:

    1. ``shutil.which("java")`` (system / conda env)
    2. ``<tool_home>/jre/bin/java`` (install4j-style bundled JRE next to the jar)
    3. raise :class:`JvmNotFoundError`

A tool whose ``ToolSpec`` sets ``prefer_bundled_jre`` swaps steps 1 and
2: its bundled JRE wins when present, and the system ``java`` is the
fallback. That is for tools tied to a Java major version (EncyclopeDIA
warns outside 11-21 and bundles 21; an unpinned conda ``openjdk`` drifts
past that).

Cleaning up after the JVM
-------------------------

A jar can start helpers of its own. EncyclopeDIA 6.x starts a .NET
"Thermo server" (MSRawJava) on every invocation, ``-h`` included, after
unpacking its runtime files (about 200 MB) into ``java.io.tmpdir``.
Measured on 6.9.23: every run leaves its ``msraw-thermo-<n>`` directory
behind, empty after a clean exit and holding 65-80 MB after an error
exit, and the server process itself is usually stopped but not always
(several were found still running hours after quick error exits made
while the machine was busy, the JVM apparently having exited while the
server was still starting). Two days of test and survey runs had left
237 such directories (1.4 GB) in ``/tmp`` and six orphan servers.

So nothing here trusts the JVM to tidy up. Every JVM is started in its
own session; when it is done, whatever remains in its process group is
stopped (:func:`reap_process_group`); and each run gets a private
``java.io.tmpdir`` that is removed afterwards, unless the caller chose
the directory, in which case it is the caller's to manage.
:func:`capture_jar` is the small sibling of :func:`run_jar` for a quick
call whose output is the point (``-h``, an argument-parsing probe):
anything that starts the jar should go through one of the two.

Being cancelled
---------------

A JVM in its own session does not receive what is sent to the wrapper:
not the terminal's Ctrl-C, not a ``kill`` of the wrapper, not a signal
to the wrapper's process group. So the wrapper has to pass a stop
request on, and it can only do that if it gets to run code when the
request arrives.

- **Ctrl-C** becomes ``KeyboardInterrupt``, the ``finally`` runs, and
  the cleanup above stops the JVM.
- **SIGTERM and SIGHUP** (``kill``, ``timeout``, a batch system
  cancelling a job, a closed terminal) by default end Python at once
  with no ``finally``, which would leave the JVM running to completion,
  writing outputs for a run nobody is waiting for. While a JVM is
  running, :func:`cancellation_scope` therefore handles them: every
  running JVM's process group is told to stop, runs in other threads
  are given time to clean up, and the main thread unwinds with
  ``SystemExit(128 + signal)`` so its own cleanup runs.
  :func:`run_jar` and :func:`capture_jar` enter the scope themselves
  when called from the main thread. Signal handlers can only be set
  from the main thread, so code that runs jars in worker threads enters
  the scope in the main thread around the whole fan-out. A signal that
  was being ignored (``nohup``) stays ignored, and a handler the
  application had installed is called after the JVMs have been told to
  stop.
- **SIGKILL of the wrapper cannot be intercepted.** The JVM then runs
  on. Nothing here can prevent that; whoever kills with SIGKILL has to
  kill the JVM's process group as well.

The bundled-JRE lookup is not tool-specific — install4j (and
InstallAnywhere, and a few others) lay out an ``app/jre/bin/java`` next
to the application jar on Linux. Any tool installed via that pattern is
self-sufficient on a machine with no system Java.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from constellation.thirdparty.registry import ToolHandle, ToolNotFoundError, find


class JvmNotFoundError(RuntimeError):
    """Raised when no usable ``java`` binary can be located.

    Both ``shutil.which("java")`` returned ``None`` and the tool's
    ``ToolHandle`` did not expose a bundled JRE at
    ``<tool_home>/jre/bin/java``. Message includes a conda hint.
    """


class JvmRunError(RuntimeError):
    """Raised when the jar exits non-zero.

    The message includes the full argv, the returncode, the last ~30
    lines of stderr (so common Java exceptions surface inline), and a
    pointer to the captured log files. ``returncode`` / ``argv`` /
    ``stderr_log`` are available as attributes for programmatic
    handling.
    """

    def __init__(
        self,
        returncode: int,
        argv: Sequence[str],
        stderr_tail: str,
        stderr_log: Path,
    ) -> None:
        super().__init__(
            f"jvm subprocess exited {returncode}: {' '.join(str(a) for a in argv)}\n"
            f"--- last lines of stderr (full: {stderr_log}) ---\n{stderr_tail}"
        )
        self.returncode = returncode
        self.argv = list(argv)
        self.stderr_tail = stderr_tail
        self.stderr_log = stderr_log


@dataclasses.dataclass(frozen=True, slots=True)
class JvmResult:
    """Outcome of a :func:`run_jar` invocation.

    Returned on success; on failure :class:`JvmRunError` is raised
    instead (unless ``check=False``, in which case the result is
    returned with the non-zero ``returncode`` exposed).
    """

    argv: list[str]
    returncode: int
    elapsed_seconds: float
    stdout_log: Path
    stderr_log: Path
    java_version: str | None
    java_source: str  # "system" or "bundled"
    java_path: Path
    jar_path: Path
    jar_sha256: str


_CONDA_HINT = (
    "no `java` on $PATH and no bundled JRE alongside the jar.\n"
    "  install openjdk in the conda env (already in environment.yml): "
    "`conda env update -f environment.yml --prune`\n"
    "  or set $CONSTELLATION_<TOOL>_HOME to an install dir containing a "
    "`jre/bin/java` (install4j bundles one)."
)


# (path, mtime_ns) → sha256 hex. Process-lifetime cache; rehashes only
# when the jar's mtime changes (e.g. user reinstalled).
_JAR_SHA_CACHE: dict[tuple[str, int], str] = {}


def _resolve_java(tool_handle: ToolHandle) -> tuple[Path, str, str | None]:
    """Return ``(java_path, source, java_version_string_or_None)``.

    ``source`` is ``"system"`` or ``"bundled"``. ``java_version`` is the
    first line of ``java -version`` stderr output (e.g.
    ``openjdk version "21.0.11" 2026-04-21 LTS``) when probing succeeded.
    """
    # install4j layout: <install_dir>/jre/bin/java sits next to the jar
    # (the jar's parent is the install dir).
    jar_parent = tool_handle.path.parent
    bundled = jar_parent / "jre" / "bin" / "java"
    if tool_handle.spec.prefer_bundled_jre and bundled.is_file():
        return bundled, "bundled", _probe_java_version(bundled)

    system_java = shutil.which("java")
    if system_java is not None:
        path = Path(system_java)
        return path, "system", _probe_java_version(path)

    if bundled.is_file():
        return bundled, "bundled", _probe_java_version(bundled)

    raise JvmNotFoundError(_CONDA_HINT)


# JVMs currently running, by pid (which is also their process-group id).
# Read by the signal handler, which runs in the main thread between
# bytecodes; plain dict operations, so no lock (a lock held by the main
# thread when the signal arrived would deadlock the handler).
_LIVE: dict[int, subprocess.Popen] = {}
#: Signals that mean "stop", beyond SIGINT (which Python already turns
#: into KeyboardInterrupt).
_CANCEL_SIGNALS: tuple[int, ...] = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)
_PREVIOUS_HANDLERS: dict[int, object] = {}
# One token per run_jar / capture_jar call in progress, (thread id, tag),
# from before its JVM starts until its cleanup has finished.
_RUNS: set[tuple[int, object]] = set()
#: How long a stop request waits for runs in other threads to finish
#: cleaning up before the main thread unwinds: the SIGTERM-then-SIGKILL
#: grace of :func:`reap_process_group`, twice over, and some slack.
_CANCEL_WAIT_S = 15.0
_scope_depth = 0
_cancelled: int | None = None  # the signal that asked us to stop, once one has
_main_is_launching = False


def _on_cancel_signal(signum: int, frame: object) -> None:
    """Pass a stop request on to every running JVM, then unwind."""
    global _cancelled
    _cancelled = signum
    for proc in list(_LIVE.values()):
        _signal_group(proc.pid, signal.SIGTERM)
    previous = _PREVIOUS_HANDLERS.get(signum)
    if callable(previous):
        # The application had its own handler: it decides what happens
        # next. If it returns, the program carries on, so later launches
        # must not be refused.
        _cancelled = None
        previous(signum, frame)
        return
    # Runs in other threads now see their JVM exit and clean up after it
    # (stop its helpers, remove its temp directory). Let them finish
    # before unwinding this thread. It is not safe to leave that to
    # interpreter shutdown: if the exception below lands while the main
    # thread is inside Thread.join(), CPython 3.12 marks that thread as
    # finished (its bpo-45274 workaround releases a lock the running
    # thread still holds), stops waiting for it, and the process exits
    # with the cleanup half done.
    me = threading.get_ident()
    deadline = time.monotonic() + _CANCEL_WAIT_S
    while time.monotonic() < deadline and any(owner != me for owner, _tag in list(_RUNS)):
        time.sleep(0.02)
    if _main_is_launching:
        # The main thread is between starting a JVM and registering it.
        # Raising here could lose track of that process; _launch raises
        # as soon as it is registered.
        return
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def cancellation_scope() -> Iterator[None]:
    """While active, SIGTERM and SIGHUP stop the running JVMs instead of
    orphaning them (module docstring, "Being cancelled").

    Re-entrant. Effective only when entered from the main thread, the
    only place signal handlers can be set; entered elsewhere it does
    nothing. The previous handlers are restored on exit, so the scope
    costs nothing outside it.
    """
    global _scope_depth, _cancelled
    active = os.name == "posix" and threading.current_thread() is threading.main_thread()
    if active:
        if _scope_depth == 0:
            _cancelled = None
            for sig in _CANCEL_SIGNALS:
                previous = signal.getsignal(sig)
                if previous is signal.SIG_IGN or previous is None:
                    continue  # ignored on purpose (nohup), or not ours to replace
                _PREVIOUS_HANDLERS[sig] = previous
                signal.signal(sig, _on_cancel_signal)
        _scope_depth += 1
    try:
        yield
    finally:
        if active:
            _scope_depth -= 1
            if _scope_depth == 0:
                for sig, previous in _PREVIOUS_HANDLERS.items():
                    signal.signal(sig, previous)  # type: ignore[arg-type]
                _PREVIOUS_HANDLERS.clear()


@contextlib.contextmanager
def _supervised() -> Iterator[None]:
    """One run, from before its JVM starts until its cleanup is done:
    inside a :func:`cancellation_scope`, and counted in ``_RUNS`` so a
    stop request can wait for it."""
    token = (threading.get_ident(), object())
    with cancellation_scope():
        _RUNS.add(token)
        try:
            yield
        finally:
            _RUNS.discard(token)


def _launch(argv: Sequence[str], **popen_kwargs: object) -> subprocess.Popen:
    """Start a JVM in its own session and register it as running.

    Its own session, so the JVM and anything it spawns form a process
    group that can be stopped as a unit. Registered, so a stop request
    reaches it. Pair with :func:`_retire`.
    """
    global _main_is_launching
    in_main = threading.current_thread() is threading.main_thread()
    if in_main:
        _main_is_launching = True
    try:
        proc = subprocess.Popen(argv, start_new_session=True, **popen_kwargs)  # type: ignore[call-overload]
        _LIVE[proc.pid] = proc
    finally:
        if in_main:
            _main_is_launching = False
    if _cancelled is not None:
        # A stop request arrived before this process was in the registry,
        # so the handler did not reach it. Stop it here.
        _retire(proc)
        raise SystemExit(128 + _cancelled)
    return proc


def _retire(proc: subprocess.Popen) -> int:
    """Stop what is left of ``proc``'s group, reap it, and forget it."""
    try:
        return reap_process_group(proc)
    finally:
        _LIVE.pop(proc.pid, None)


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to process group ``pgid``; ``False`` if it has no members."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _wait_for_exit(proc: subprocess.Popen) -> None:
    """Block until ``proc`` has exited, without reaping it where possible.

    An exited-but-unreaped process still owns its pid, so the process
    group it leads can be signalled with no chance of the id having been
    reused. :func:`reap_process_group` does the reaping.
    """
    if os.name == "posix" and hasattr(os, "waitid"):
        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
    else:  # Windows, and Pythons whose platform lacks waitid
        proc.wait()


def reap_process_group(proc: subprocess.Popen, *, grace_s: float = 5.0) -> int:
    """Stop everything in the process group ``proc`` leads; return its exit status.

    ``proc`` must have been started with ``start_new_session=True``, so
    that its pid is the id of a group holding it and its descendants and
    nothing else. Call it when ``proc`` has exited (it then only stops
    what ``proc`` left behind, and the status is ``proc``'s own) or to
    abandon a run (it stops ``proc`` too, and the status is the signal).
    SIGTERM first, SIGKILL for whatever is still there after ``grace_s``.
    """
    if os.name != "posix":
        if proc.poll() is None:
            proc.terminate()
        return proc.wait()
    pgid = proc.pid
    _signal_group(pgid, signal.SIGTERM)
    try:
        returncode = proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        _signal_group(pgid, signal.SIGKILL)
        returncode = proc.wait()
    # The leader is reaped; anything still in the group is an orphan.
    deadline = time.monotonic() + grace_s
    while _signal_group(pgid, 0):
        if time.monotonic() >= deadline:
            _signal_group(pgid, signal.SIGKILL)
            break
        time.sleep(0.05)
    return returncode


def capture_jar(
    java: Path,
    jar: Path,
    args: Sequence[str],
    *,
    timeout: float,
    jvm_args: Sequence[str] = (),
    cwd: Path | None = None,
) -> str:
    """Run ``java -jar <jar> <args>`` to completion; return stdout and stderr, interleaved.

    For short calls whose output is the result. The exit status is not
    checked (most EncyclopeDIA programs exit 1 from ``-h``). Raises
    :class:`subprocess.TimeoutExpired` after ``timeout`` seconds. Cleans
    up as described in the module docstring either way.
    """
    java_tmp = Path(tempfile.mkdtemp(prefix="constellation-jvm-"))
    proc: subprocess.Popen[str] | None = None
    with _supervised():
        try:
            proc = _launch(
                [
                    str(java),
                    "-Djava.awt.headless=true",
                    *(str(a) for a in jvm_args),
                    f"-Djava.io.tmpdir={java_tmp}",
                    "-jar",
                    str(jar),
                    *(str(a) for a in args),
                ],
                cwd=str(cwd) if cwd is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out, _ = proc.communicate(timeout=timeout)
        finally:
            if proc is not None:
                _retire(proc)
            shutil.rmtree(java_tmp, ignore_errors=True)
    return out


def _probe_java_version(java_path: Path) -> str | None:
    """Return first line of ``java -version`` (which Java writes to stderr)."""
    try:
        result = subprocess.run(
            [str(java_path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stderr or result.stdout or "").strip()
    return output.splitlines()[0] if output else None


def _jar_sha256(jar_path: Path) -> str:
    """SHA256 of the jar bytes, cached on (path, mtime_ns)."""
    stat = jar_path.stat()
    key = (str(jar_path), stat.st_mtime_ns)
    if key in _JAR_SHA_CACHE:
        return _JAR_SHA_CACHE[key]
    hasher = hashlib.sha256()
    with jar_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _JAR_SHA_CACHE[key] = digest
    return digest


def _stream_to_file_and_mirror(
    src,  # subprocess.Popen.stdout / .stderr (binary file-like)
    log_path: Path,
    mirror_stream,  # sys.stdout / sys.stderr or None
    tail_buffer: list[str],
    tail_max_lines: int,
) -> None:
    """Reader-thread body: pump bytes from ``src`` into ``log_path``,
    optionally mirror to ``mirror_stream``, keep last ``tail_max_lines``
    lines in ``tail_buffer`` for failure messages.
    """
    with log_path.open("wb") as out_fh:
        for raw in iter(src.readline, b""):
            out_fh.write(raw)
            out_fh.flush()
            if mirror_stream is not None:
                try:
                    mirror_stream.write(raw.decode("utf-8", errors="replace"))
                    mirror_stream.flush()
                except Exception:  # noqa: BLE001 — never let mirror IO kill the runner
                    pass
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            tail_buffer.append(line)
            if len(tail_buffer) > tail_max_lines:
                del tail_buffer[: len(tail_buffer) - tail_max_lines]


def run_jar(
    tool: str,
    *,
    args: Sequence[str],
    main_class: str | None = None,
    jvm_heap_max: str = "12g",
    jvm_heap_min: str | None = None,
    jvm_tmpdir: Path | None = None,
    extra_jvm_args: Sequence[str] = (),
    log_dir: Path,
    stream_to_stderr: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    check: bool = True,
    tail_lines: int = 30,
) -> JvmResult:
    """Run a registered jar tool as a subprocess.

    Resolves ``tool`` via :func:`constellation.thirdparty.registry.find`,
    picks a ``java`` binary (system → bundled JRE), composes the argv,
    spawns the process, and tees stdout/stderr to ``log_dir/stdout.log``
    and ``log_dir/stderr.log`` while optionally mirroring to host
    ``sys.stderr`` for live progress.

    Parameters
    ----------
    tool
        Registry name (``"encyclopedia"``, etc.). Must be already
        registered via :func:`constellation.thirdparty.registry.register`.
    args
        Tool-specific CLI arguments — everything after the jar / main
        class. Pass strings; the runner does not interpret them.
    main_class
        When set, invokes ``java -cp <jar> <main_class> ...args`` instead
        of ``java -jar <jar> ...args``. EncyclopeDIA's default entry
        suffices for everything we wrap today; this knob is for tools
        whose jar manifests don't declare a usable ``Main-Class``.
    jvm_heap_max
        ``-Xmx`` value. Default ``"12g"`` — comfortable for EncyclopeDIA
        library search on a workstation. Callers should size up for
        whole-proteome runs.
    jvm_heap_min
        ``-Xms`` value. ``None`` leaves the JVM at its default initial
        heap.
    jvm_tmpdir
        ``-Djava.io.tmpdir`` override, left in place afterwards.
        ``None`` gives the run a private directory under the system tmp
        and removes it when the JVM has exited.
    extra_jvm_args
        Free-form JVM args appended *after* heap/tmpdir but *before*
        ``-jar``. Use for ``-XX:`` GC tuning, ``-Dlog4j.*`` properties,
        etc. — the runner doesn't bake any opinions on these.
    log_dir
        Directory that will hold ``stdout.log``, ``stderr.log``, and
        ``java-version.txt``. Created if absent. Existing files are
        overwritten.
    stream_to_stderr
        When ``True``, the runner tees the jar's stdout and stderr to
        the host process's ``sys.stderr`` while also writing to log
        files. Disable for batched / unattended runs that want only the
        logs.
    env
        Environment for the subprocess. ``None`` inherits the parent
        environment.
    cwd
        Working directory for the subprocess.
    check
        When ``True`` (default), non-zero exit raises
        :class:`JvmRunError`. Set ``False`` to inspect the
        :class:`JvmResult` regardless.
    tail_lines
        Number of trailing stderr lines to embed in
        :class:`JvmRunError` on failure. Default 30 — enough to
        surface a Java stack trace inline without overwhelming the
        terminal.

    Returns
    -------
    JvmResult
        Captured outcome. On non-zero exit with ``check=False``, the
        result is still returned with the non-zero ``returncode``.

    Raises
    ------
    ToolNotFoundError
        ``tool`` is registered but no install satisfies the spec.
    JvmNotFoundError
        Neither system ``java`` nor a bundled JRE alongside the jar.
    JvmRunError
        ``check=True`` (default) and the jar exited non-zero.
    """
    try:
        handle = find(tool)
    except ToolNotFoundError:
        raise

    java_path, java_source, java_version = _resolve_java(handle)

    jar_path = handle.path
    jar_sha = _jar_sha256(jar_path)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    java_version_log = log_dir / "java-version.txt"
    java_version_log.write_text(
        f"{java_version or '(unknown)'}\nsource: {java_source}\npath: {java_path}\n"
    )

    jvm_flags: list[str] = [
        f"-Xmx{jvm_heap_max}",
        # Run AWT in headless mode so tools that draw plots (EncyclopeDIA's
        # mass-error PDF reports via JFreeChart, etc.) work on compute
        # nodes / containers / anywhere without an X11 display. The JVM
        # uses a software renderer instead of trying to connect to an
        # X server. Interactive Java apps that need a real display can
        # override by passing ``-Djava.awt.headless=false`` via
        # ``extra_jvm_args`` — later flags win in the JVM property
        # resolution order.
        "-Djava.awt.headless=true",
    ]
    if jvm_heap_min is not None:
        jvm_flags.append(f"-Xms{jvm_heap_min}")
    private_tmpdir: Path | None = None
    if jvm_tmpdir is None:
        private_tmpdir = Path(tempfile.mkdtemp(prefix="constellation-jvm-"))
        jvm_tmpdir = private_tmpdir
    jvm_flags.append(f"-Djava.io.tmpdir={jvm_tmpdir}")
    jvm_flags.extend(str(a) for a in extra_jvm_args)

    if main_class is None:
        invocation = ["-jar", str(jar_path)]
    else:
        invocation = ["-cp", str(jar_path), main_class]

    argv: list[str] = [
        str(java_path),
        *jvm_flags,
        *invocation,
        *(str(a) for a in args),
    ]

    stderr_tail: list[str] = []
    stdout_tail: list[str] = []  # unused for raises but symmetric

    start = time.perf_counter()
    threads: list[threading.Thread] = []
    proc: subprocess.Popen[bytes] | None = None
    with _supervised():
        try:
            proc = _launch(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                bufsize=0,
            )
            if proc.stdout is not None:
                t_out = threading.Thread(
                    target=_stream_to_file_and_mirror,
                    args=(
                        proc.stdout,
                        stdout_log,
                        sys.stderr if stream_to_stderr else None,
                        stdout_tail,
                        tail_lines,
                    ),
                    daemon=True,
                )
                t_out.start()
                threads.append(t_out)
            if proc.stderr is not None:
                t_err = threading.Thread(
                    target=_stream_to_file_and_mirror,
                    args=(
                        proc.stderr,
                        stderr_log,
                        sys.stderr if stream_to_stderr else None,
                        stderr_tail,
                        tail_lines,
                    ),
                    daemon=True,
                )
                t_err.start()
                threads.append(t_err)

            _wait_for_exit(proc)
        finally:
            # After a normal exit this stops what the JVM left running; on an
            # exception, Ctrl-C or a stop signal it stops the JVM as well.
            # Before the joins, so nothing holding a pipe open can keep a
            # reader thread alive.
            if proc is not None:
                returncode = _retire(proc)
            for t in threads:
                t.join()
            if private_tmpdir is not None:
                shutil.rmtree(private_tmpdir, ignore_errors=True)
    elapsed = time.perf_counter() - start

    result = JvmResult(
        argv=argv,
        returncode=returncode,
        elapsed_seconds=elapsed,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        java_version=java_version,
        java_source=java_source,
        java_path=java_path,
        jar_path=jar_path,
        jar_sha256=jar_sha,
    )

    if check and returncode != 0:
        raise JvmRunError(
            returncode=returncode,
            argv=argv,
            stderr_tail="\n".join(stderr_tail),
            stderr_log=stderr_log,
        )
    return result


__all__ = [
    "JvmNotFoundError",
    "JvmResult",
    "JvmRunError",
    "cancellation_scope",
    "capture_jar",
    "reap_process_group",
    "run_jar",
]
