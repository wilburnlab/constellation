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

The bundled-JRE fallback is not tool-specific — install4j (and
InstallAnywhere, and a few others) lay out an ``app/jre/bin/java`` next
to the application jar on Linux. Any tool installed via that pattern is
self-sufficient on a machine with no system Java.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
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
    system_java = shutil.which("java")
    if system_java is not None:
        path = Path(system_java)
        return path, "system", _probe_java_version(path)

    # install4j layout: <install_dir>/jre/bin/java sits next to the jar
    # (the jar's parent is the install dir).
    jar_parent = tool_handle.path.parent
    bundled = jar_parent / "jre" / "bin" / "java"
    if bundled.is_file():
        return bundled, "bundled", _probe_java_version(bundled)

    raise JvmNotFoundError(_CONDA_HINT)


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
        ``-Djava.io.tmpdir`` override. ``None`` uses the system tmp.
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
    if jvm_tmpdir is not None:
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
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        bufsize=0,
    )
    threads: list[threading.Thread] = []
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

    returncode = proc.wait()
    for t in threads:
        t.join()
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
    "run_jar",
]
