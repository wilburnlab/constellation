"""Generic mmseqs2 subprocess runner.

Composes a :class:`~constellation.thirdparty.registry.ToolHandle` (from
``registry.find("mmseqs2")``) with subprocess machinery. Mirrors
:mod:`constellation.thirdparty.jvm` shape (Popen + two reader threads
tee'ing stdout/stderr to log files + tail buffer for failure messages)
but without the bundled-runtime fallback — mmseqs2 is a native binary,
so resolution is just the registry's env-var → versioned-layout →
$PATH cascade.

mmseqs2's primary entrypoint here is ``mmseqs easy-search`` — the
one-shot search wrapper that handles indexing internally. Future
callers may compose other subcommands (``createdb``, ``search``,
``cluster``, ...) via :func:`run_mmseqs` (the lower-level dispatch).
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from constellation.thirdparty.registry import ToolNotFoundError, find


class MmseqsRunError(RuntimeError):
    """Raised when the mmseqs2 subprocess exits non-zero.

    Message includes the full argv, the returncode, the last ~30 lines
    of stderr (so mmseqs2's diagnostic output surfaces inline), and a
    pointer to the captured log files. Attributes ``returncode`` /
    ``argv`` / ``stderr_log`` are available for programmatic handling.
    """

    def __init__(
        self,
        returncode: int,
        argv: Sequence[str],
        stderr_tail: str,
        stderr_log: Path,
    ) -> None:
        super().__init__(
            f"mmseqs2 subprocess exited {returncode}: "
            f"{' '.join(str(a) for a in argv)}\n"
            f"--- last lines of stderr (full: {stderr_log}) ---\n{stderr_tail}"
        )
        self.returncode = returncode
        self.argv = list(argv)
        self.stderr_tail = stderr_tail
        self.stderr_log = stderr_log


@dataclasses.dataclass(frozen=True, slots=True)
class MmseqsResult:
    """Outcome of a :func:`run_mmseqs_search` invocation.

    Returned on success; on failure :class:`MmseqsRunError` is raised
    instead (unless ``check=False``, in which case the result is
    returned with the non-zero ``returncode`` exposed).
    """

    argv: list[str]
    returncode: int
    elapsed_seconds: float
    stdout_log: Path
    stderr_log: Path
    mmseqs_path: Path
    mmseqs_version: str | None
    mmseqs_sha256: str


# (path, mtime_ns) → sha256 hex. Process-lifetime cache; rehashes only
# when the binary's mtime changes (e.g. user reinstalled).
_MMSEQS_SHA_CACHE: dict[tuple[str, int], str] = {}


def _mmseqs_sha256(mmseqs_path: Path) -> str:
    """SHA256 of the binary bytes, cached on (path, mtime_ns)."""
    stat = mmseqs_path.stat()
    key = (str(mmseqs_path), stat.st_mtime_ns)
    if key in _MMSEQS_SHA_CACHE:
        return _MMSEQS_SHA_CACHE[key]
    hasher = hashlib.sha256()
    with mmseqs_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _MMSEQS_SHA_CACHE[key] = digest
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


# Canonical 8-column output cartographer's pipeline + Constellation's
# read_mmseqs_tab() consume. Keep the field set + order stable — the
# downstream parser is positional.
DEFAULT_FORMAT_OUTPUT = "query,target,evalue,qstart,qend,tstart,tend,cigar"


def build_easy_search_args(
    *,
    query_fasta: Path,
    target_fasta: Path,
    output_tab: Path,
    scratch_dir: Path,
    evalue: float = 1e-20,
    threads: int = 4,
    sensitivity: float = 5.7,
    format_output: str = DEFAULT_FORMAT_OUTPUT,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the argv tail for ``mmseqs easy-search`` (without the
    mmseqs binary itself prepended).

    The argv shape mmseqs2 expects:

        easy-search <queryFastaFile> <targetFastaFile> <alignmentFile> <tmpDir>
            -e <evalue> --threads <N> --format-output <fmt> -s <sensitivity>

    Pure function — no I/O. Callers prepend the resolved mmseqs path
    and feed the result to ``subprocess.Popen``.
    """
    args: list[str] = [
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tab),
        str(scratch_dir),
        "-e",
        format(evalue, "g"),
        "--threads",
        str(threads),
        "-s",
        str(sensitivity),
        "--format-output",
        format_output,
    ]
    args.extend(str(a) for a in extra_args)
    return args


def run_mmseqs_search(
    *,
    query_fasta: Path,
    target_fasta: Path,
    output_tab: Path,
    log_dir: Path,
    evalue: float = 1e-20,
    threads: int = 4,
    sensitivity: float = 5.7,
    format_output: str = DEFAULT_FORMAT_OUTPUT,
    scratch_dir: Path | None = None,
    extra_args: Sequence[str] = (),
    stream_to_stderr: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    check: bool = True,
    tail_lines: int = 30,
) -> MmseqsResult:
    """Run ``mmseqs easy-search <query> <target> <output> <tmp>`` as a
    subprocess.

    Resolves the mmseqs binary via
    :func:`constellation.thirdparty.registry.find` (env var
    ``CONSTELLATION_MMSEQS2_HOME`` → versioned layout under
    ``third_party/mmseqs2/current/bin/mmseqs`` → ``$PATH`` lookup for
    conda-installed ``mmseqs``).

    Parameters
    ----------
    query_fasta
        Path to the query FASTA (novel proteins).
    target_fasta
        Path to the target FASTA (reference proteome, or a combined
        reference+swissprot FASTA for competitive search).
    output_tab
        Output path for the alignment TSV. Headerless, columns governed
        by ``format_output``. Consumable by
        :func:`constellation.core.io.schemas.read_mmseqs_tab` when
        ``format_output`` matches its 8-column expectation (the default).
    log_dir
        Directory that will hold ``stdout.log``, ``stderr.log``, and
        ``mmseqs-version.txt``. Created if absent. Existing files are
        overwritten.
    evalue
        E-value cutoff (``-e``). Default ``1e-20`` matches cartographer.
    threads
        Worker thread count (``--threads``).
    sensitivity
        Sensitivity (``-s``). mmseqs default is 5.7.
    format_output
        Comma-separated output fields (``--format-output``). Default
        matches cartographer's 8-column layout that
        ``read_mmseqs_tab`` parses.
    scratch_dir
        Working directory for mmseqs2's intermediate index/database
        files. When ``None``, a ``tempfile.TemporaryDirectory`` is
        created and cleaned up on completion. Pass an explicit path on
        nodes with tight ``/tmp`` budgets (the pipeline orchestrator
        does this — points it at ``<output-dir>/03_alignment/.scratch/``).
    extra_args
        Free-form mmseqs args appended after the canonical flags.
        Use for ``--min-seq-id``, ``-c``, ``--alignment-mode``, etc.
        when a caller needs them.
    stream_to_stderr
        When ``True``, the runner tees stdout/stderr to the host
        ``sys.stderr`` while also writing log files. Disable for
        batched / unattended runs.
    env
        Environment for the subprocess. ``None`` inherits.
    cwd
        Working directory for the subprocess.
    check
        When ``True`` (default), non-zero exit raises
        :class:`MmseqsRunError`. Set ``False`` to inspect the
        :class:`MmseqsResult` regardless.
    tail_lines
        Number of trailing stderr lines to embed in
        :class:`MmseqsRunError` on failure. Default 30.

    Returns
    -------
    MmseqsResult
        Captured outcome. On non-zero exit with ``check=False``, the
        result is still returned with the non-zero ``returncode``.

    Raises
    ------
    ToolNotFoundError
        mmseqs2 is registered but no install satisfies the spec.
    MmseqsRunError
        ``check=True`` (default) and the binary exited non-zero.
    """
    try:
        handle = find("mmseqs2")
    except ToolNotFoundError:
        raise

    mmseqs_path = handle.path
    mmseqs_sha = _mmseqs_sha256(mmseqs_path)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    version_log = log_dir / "mmseqs-version.txt"
    version_log.write_text(
        f"{handle.version or '(unknown)'}\npath: {mmseqs_path}\n"
        f"source: {handle.source}\n"
    )

    # Manage scratch dir: caller-supplied → use as-is; None → tempdir
    # that we clean up on exit (success OR failure).
    scratch_ctx: tempfile.TemporaryDirectory | None = None
    if scratch_dir is None:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="mmseqs2_")
        scratch_path = Path(scratch_ctx.name)
    else:
        scratch_path = Path(scratch_dir)
        scratch_path.mkdir(parents=True, exist_ok=True)

    output_tab = Path(output_tab)
    output_tab.parent.mkdir(parents=True, exist_ok=True)

    try:
        args_tail = build_easy_search_args(
            query_fasta=Path(query_fasta),
            target_fasta=Path(target_fasta),
            output_tab=output_tab,
            scratch_dir=scratch_path,
            evalue=evalue,
            threads=threads,
            sensitivity=sensitivity,
            format_output=format_output,
            extra_args=extra_args,
        )
        argv: list[str] = [str(mmseqs_path), *args_tail]

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
        threads_: list[threading.Thread] = []
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
            threads_.append(t_out)
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
            threads_.append(t_err)

        returncode = proc.wait()
        for t in threads_:
            t.join()
        elapsed = time.perf_counter() - start

        result = MmseqsResult(
            argv=argv,
            returncode=returncode,
            elapsed_seconds=elapsed,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            mmseqs_path=mmseqs_path,
            mmseqs_version=handle.version,
            mmseqs_sha256=mmseqs_sha,
        )

        if check and returncode != 0:
            raise MmseqsRunError(
                returncode=returncode,
                argv=argv,
                stderr_tail="\n".join(stderr_tail),
                stderr_log=stderr_log,
            )
        return result
    finally:
        if scratch_ctx is not None:
            scratch_ctx.cleanup()


__all__ = [
    "DEFAULT_FORMAT_OUTPUT",
    "MmseqsResult",
    "MmseqsRunError",
    "build_easy_search_args",
    "run_mmseqs_search",
]
