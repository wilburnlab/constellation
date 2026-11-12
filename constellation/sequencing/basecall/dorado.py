"""``DoradoRunner`` — subprocess wrapper for the Dorado basecaller.

Dorado is the Oxford Nanopore basecaller (replaces guppy). Resolved
via :func:`constellation.thirdparty.find('dorado')`. Subcommand methods
mirror Dorado's CLI verbs (``basecaller``, ``duplex``, ``aligner``,
``demux``, ``summary``, ``download``).

Long-running calls (``basecaller`` on a single PromethION flow cell
takes ~3-4 days at sup model on an A100) return :class:`RunHandle`
rather than blocking. The handle exposes:

    wait()              block until completion
    poll()              non-blocking RunStatus check
    tail_progress()     iterator over Dorado progress events
    cancel()            send signal to the subprocess

Detached mode (``detach=True``) decouples Dorado from the Python process
and writes a PID file; ``RunHandle.attach(pid_file)`` rebinds to the
running process so a separate ``constellation basecall --follow`` session
can monitor it. The generic spawn/exit-sentinel plumbing lives in
:mod:`sequencing.basecall.dorado_run`.

License: Dorado is the Oxford Nanopore Technologies PLC. Public
License Version 1.0 — Research Purposes only. The wrapper itself is
Apache 2.0 (Constellation), but anyone *invoking* this code accepts
ONT's terms. See the upstream LICENCE.txt:
https://github.com/nanoporetech/dorado/blob/master/LICENCE.txt
"""

from __future__ import annotations

import json
import os
import re
import signal as _signal
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from constellation.sequencing.basecall.dorado_run import (
    pid_alive,
    read_exit_code,
    resolve_dorado,
    spawn_dorado,
)
from constellation.sequencing.basecall.models import DoradoModel


# ──────────────────────────────────────────────────────────────────────
# Run lifecycle
# ──────────────────────────────────────────────────────────────────────


class RunStatus(Enum):
    PENDING = "pending"  # spawned but not yet started
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"  # e.g. PID file lost


@dataclass(frozen=True)
class DoradoProgressEvent:
    """One progress event parsed from Dorado's stderr stream."""

    elapsed_s: float
    reads_processed: int
    bases_emitted: int
    eta_s: float | None
    raw: str  # original stderr line, kept for debugging


_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _parse_progress(line: str, started_at: float) -> DoradoProgressEvent:
    """Best-effort parse of one Dorado stderr line into an event.

    Dorado's progress formatting varies across versions; v1 preserves the
    raw line (the CLI renders it) and stamps elapsed time. Structured
    reads/bases/ETA extraction is refined when canned fixtures land.
    """
    elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
    return DoradoProgressEvent(
        elapsed_s=elapsed,
        reads_processed=0,
        bases_emitted=0,
        eta_s=None,
        raw=line,
    )


@dataclass(frozen=True)
class RunHandle:
    """Handle to a running (or completed) Dorado subprocess.

    Foreground runs hold the live ``Popen`` (in ``extras['_proc']``);
    detached / attached runs track liveness via the PID + an exit-code
    sentinel file (``extras['exit_file']``).
    """

    pid: int
    pid_file: Path
    output_path: Path
    stdout_log: Path
    stderr_log: Path
    started_at: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def attach(cls, pid_file: Path) -> "RunHandle":
        """Re-bind to a previously detached run from its PID file."""
        data = json.loads(Path(pid_file).read_text())
        return cls(
            pid=int(data["pid"]),
            pid_file=Path(pid_file),
            output_path=Path(data["output_path"]),
            stdout_log=Path(data["output_path"]),
            stderr_log=Path(data["stderr_log"]),
            started_at=float(data.get("started_at", 0.0)),
            extras={
                "_proc": None,
                "exit_file": data.get("exit_file"),
                "detached": True,
            },
        )

    # ── status ──────────────────────────────────────────────────────
    def poll(self) -> RunStatus:
        """Non-blocking status check."""
        proc = self.extras.get("_proc")
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                return RunStatus.RUNNING
            return RunStatus.COMPLETED if rc == 0 else RunStatus.FAILED
        exit_file = self.extras.get("exit_file")
        code = read_exit_code(Path(exit_file)) if exit_file else None
        if code is not None:
            return RunStatus.COMPLETED if code == 0 else RunStatus.FAILED
        return RunStatus.RUNNING if pid_alive(self.pid) else RunStatus.UNKNOWN

    def wait(self, *, timeout: float | None = None) -> int:
        """Block until the subprocess exits. Returns its exit code."""
        proc = self.extras.get("_proc")
        if proc is not None:
            return proc.wait(timeout=timeout)
        exit_file = self.extras.get("exit_file")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = read_exit_code(Path(exit_file)) if exit_file else None
            if code is not None:
                return code
            if not pid_alive(self.pid):
                code = read_exit_code(Path(exit_file)) if exit_file else None
                return code if code is not None else -1
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"dorado run pid={self.pid} did not finish within {timeout}s"
                )
            time.sleep(0.1)

    def tail_progress(self) -> Iterator[DoradoProgressEvent]:
        """Iterator over progress events parsed from stderr, following the
        log until the run finishes."""
        log = Path(self.stderr_log)
        # Advance from the last byte offset rather than rereading the
        # whole log each tick. At 200 ms over a multi-day run the reread
        # is quadratic in log size, and --progress / --follow got steadily
        # more expensive exactly when the run was longest.
        offset = 0
        pending = ""
        while True:
            lines: list[str] = []
            if log.exists():
                with log.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                    offset = fh.tell()
                if chunk:
                    pending += chunk.decode("utf-8", errors="replace")
                    parts = pending.split("\n")
                    pending = parts.pop()
                    lines = parts
            for line in lines:
                yield _parse_progress(line, self.started_at)
            if self.poll() in (
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.UNKNOWN,
                RunStatus.CANCELLED,
            ):
                # Drain whatever landed after the last read, plus any
                # trailing partial line the writer never terminated.
                if log.exists():
                    with log.open("rb") as fh:
                        fh.seek(offset)
                        tail = fh.read()
                    if tail:
                        pending += tail.decode("utf-8", errors="replace")
                for line in pending.split("\n"):
                    if line:
                        yield _parse_progress(line, self.started_at)
                return
            time.sleep(0.2)

    def cancel(self, *, sig: int = _signal.SIGTERM) -> None:
        """Signal the subprocess (default SIGTERM; SIGKILL escalates)."""
        proc = self.extras.get("_proc")
        if proc is not None:
            proc.send_signal(sig)
            return
        try:
            os.killpg(os.getpgid(self.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass


# ──────────────────────────────────────────────────────────────────────
# Resuming an interrupted basecall
# ──────────────────────────────────────────────────────────────────────


def partial_path(output: Path) -> Path:
    """Where an interrupted run's reads are kept while it is resumed."""
    output = Path(output)
    return output.with_name(output.name + ".partial")


def clear_partial(output: Path) -> None:
    """Delete ``<output>.partial`` after a run that resumed from it exits 0.

    Dorado writes every read of the resume file into the new output, so a
    finished output holds all of the partial's reads and the partial is
    only a second copy (often hundreds of GB).
    """
    partial_path(output).unlink(missing_ok=True)


def _non_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _resume_source(output: Path, partial: Path) -> Path | None:
    """Put the file to resume from at ``partial`` and return it, or None.

    Dorado's ``--resume-from`` writes every read of the resume file into
    the new output before it basecalls anything (``process_resume_file``
    → ``copy_completed_reads`` in its ``basecaller.cpp``), and calls every
    read the resume file lacks. So when both files exist (an earlier
    resume was itself interrupted), the larger one holds the most finished
    reads: a smaller output is an unfinished copy of the partial, and a
    larger one finished the copy and went on to new reads. Choosing wrong
    costs basecalling time, never reads.

    The partial used to be overwritten with whatever the output held, so
    a resume killed while still copying replaced days of reads with a few
    bytes; and a partial with no output beside it was ignored and the run
    started again from zero.
    """
    if _non_empty(output) and (
        not _non_empty(partial) or output.stat().st_size > partial.stat().st_size
    ):
        output.replace(partial)
        return partial
    if _non_empty(partial):
        return partial
    return None


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DoradoRunner:
    """Thin wrapper around the ``dorado`` binary.

    Resolved at call time via ``constellation.thirdparty.find('dorado')``;
    raises ``FileNotFoundError`` if Dorado isn't installed. Each
    subcommand method returns a :class:`RunHandle`; callers choose block /
    follow / detach. Default ``device='cuda:0'`` matches the lab's typical
    GPU configuration; CPU-only hosts pass ``device='cpu'``.
    """

    device: str = "cuda:0"
    threads: int = 8
    extra_args: tuple[str, ...] = ()

    # ── launch helper ───────────────────────────────────────────────
    def _launch(
        self,
        argv: list[str],
        *,
        output: Path,
        stage: str,
        detach: bool,
    ) -> RunHandle:
        output = Path(output)
        log_dir = output.parent
        stderr_log = log_dir / f"{output.name}.dorado.{stage}.stderr.log"
        pid_file = log_dir / f"{output.name}.dorado.{stage}.pid"
        started = time.time()
        pid, proc, exit_file = spawn_dorado(
            argv,
            output_path=output,
            stderr_log=stderr_log,
            detach=detach,
            pid_file=pid_file if detach else None,
            started_at=started,
        )
        return RunHandle(
            pid=pid,
            pid_file=pid_file,
            output_path=output,
            stdout_log=output,
            stderr_log=stderr_log,
            started_at=started,
            extras={
                "_proc": proc,
                "exit_file": str(exit_file) if exit_file else None,
                "stage": stage,
                "detached": detach,
            },
        )

    # ── verbs ───────────────────────────────────────────────────────
    def basecaller(
        self,
        model: DoradoModel,
        pod5_paths: list[Path],
        output: Path,
        *,
        modified_bases: tuple[str, ...] = (),
        device: str | None = None,
        emit_moves: bool = False,
        resume: bool = True,
        detach: bool = False,
    ) -> RunHandle:
        """Run ``dorado basecaller`` on the given POD5 inputs → BAM.

        Output is BAM (Dorado's default, written to stdout → ``output``).
        ``modified_bases`` defaults to the model's declared mods. When
        ``emit_moves`` is set, the ``mv`` tag is written (move-table-aware
        ``dorado polish`` models need it). Dorado reads POD5 directly — we
        do not decode POD5 ourselves.

        ``resume`` continues an interrupted run with ``--resume-from``
        from ``<output>.partial`` (see :func:`_resume_source` for which
        file that is). Without ``resume``, a non-empty ``<output>.partial``
        raises ``FileExistsError`` rather than being silently left behind
        by a fresh run. Callers delete the partial with
        :func:`clear_partial` once the run exits 0.
        """
        output = Path(output)
        # Resolve before touching any file: a missing or too-old Dorado
        # used to raise only after the output had been moved to the
        # partial, and the next --resume then found no output and started
        # from zero.
        resolve_dorado()
        partial = partial_path(output)
        resume_from: Path | None = None
        if resume:
            resume_from = _resume_source(output, partial)
        elif _non_empty(partial):
            raise FileExistsError(
                f"{partial} holds the reads of an interrupted run; pass "
                f"--resume to continue it, or remove it to start over"
            )
        dev = device if device is not None else self.device
        argv: list[str] = ["basecaller"]
        if dev:
            argv += ["--device", dev]
        mods = tuple(modified_bases) if modified_bases else tuple(model.mods)
        if emit_moves:
            argv += ["--emit-moves"]
        if resume_from is not None:
            argv += ["--resume-from", str(resume_from)]
        argv += [*self.extra_args, model.model_name(), *(str(p) for p in pod5_paths)]
        # `--modified-bases` takes a space-separated list and reads every
        # token up to the next option, so placed before the positionals it
        # swallowed the model and the POD5 paths as modification codes.
        # Last, as ONT's docs write it, it has nothing after it to take.
        if mods:
            argv += ["--modified-bases", *mods]
        return self._launch(argv, output=output, stage="basecaller", detach=detach)

    def duplex(
        self,
        model: DoradoModel,
        pod5_paths: list[Path],
        output: Path,
        *,
        device: str | None = None,
        detach: bool = False,
    ) -> RunHandle:
        """Run ``dorado duplex`` (paired-strand basecalling for Q30+
        consensus reads where complement strands are available)."""
        dev = device if device is not None else self.device
        argv: list[str] = ["duplex"]
        if dev:
            argv += ["--device", dev]
        argv += [*self.extra_args, model.model_name(), *(str(p) for p in pod5_paths)]
        return self._launch(argv, output=Path(output), stage="duplex", detach=detach)

    def aligner(
        self,
        reference: Path,
        bam: Path,
        output: Path,
        *,
        threads: int | None = None,
    ) -> RunHandle:
        """Run ``dorado aligner`` (maps an unaligned BAM against a
        reference; output BAM has alignment records populated). Used by
        the polish loop — ``dorado polish`` accepts only Dorado-aligned
        BAMs."""
        t = threads if threads is not None else self.threads
        argv = ["aligner", "-t", str(int(t)), str(reference), str(bam)]
        return self._launch(argv, output=Path(output), stage="aligner", detach=False)

    def demux(
        self,
        bam: Path,
        output_dir: Path,
        *,
        kit: str | None = None,
        emit_fastq: bool = False,
    ) -> RunHandle:
        """Run ``dorado demux`` — ONT's built-in barcoding demultiplex.

        Note: Constellation's transcriptome workflow uses its own
        segmented edlib-based demux; this wrapper is here for genomic-DNA
        workflows where the standard demux is fine. Output lands under
        ``output_dir`` (not stdout); the captured stdout is a log.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = ["demux", "--output-dir", str(output_dir)]
        if kit:
            argv += ["--kit-name", kit]
        else:
            argv += ["--no-classify"]
        if emit_fastq:
            argv += ["--emit-fastq"]
        argv += [*self.extra_args, str(bam)]
        stdout_log = output_dir / ".dorado_demux.stdout.log"
        return self._launch(argv, output=stdout_log, stage="demux", detach=False)

    def summary(
        self,
        bam: Path,
        output: Path,
    ) -> RunHandle:
        """Run ``dorado summary`` — emit a per-read summary TSV (stdout)."""
        argv = ["summary", str(bam)]
        return self._launch(argv, output=Path(output), stage="summary", detach=False)

    def polish(
        self,
        aligned_bam: Path,
        draft_fasta: Path,
        output: Path,
        *,
        device: str | None = None,
        rg: str | None = None,
        extra: tuple[str, ...] = (),
    ) -> RunHandle:
        """Run ``dorado polish <aligned_bam> <draft_fasta>`` → consensus FASTA.

        Validated against dorado 2.0.0 (no polish flag changed through
        2.1.2, per its changelog): positional order is
        ``in_aln_bam in_draft_fastx``, consensus goes to stdout when no
        ``-o`` is given, and ``-x/--device`` selects the compute device.
        ``aligned_bam`` must be reads aligned to ``draft_fasta`` with
        ``dorado aligner`` (sorted + indexed) carrying a single
        basecaller-model ``@RG`` (see
        :func:`sequencing.basecall.readgroup.harmonize_read_group`); polish
        resolves its model from that read group. ``rg`` pins ``--RG`` when a
        BAM still carries multiple read groups.
        """
        dev = device if device is not None else self.device
        argv: list[str] = ["polish"]
        if dev:
            argv += ["--device", dev]
        argv += [*extra]
        if rg:
            argv += ["--RG", rg]
        argv += [str(aligned_bam), str(draft_fasta)]
        return self._launch(argv, output=Path(output), stage="polish", detach=False)

    def fetch_model(
        self, model: DoradoModel, models_dir: Path | None = None
    ) -> Path:
        """Run ``dorado download --model <name>`` and return the resolved
        model directory."""
        argv = ["download", "--model", model.model_name()]
        if models_dir is not None:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            argv += ["--models-directory", str(models_dir)]
        subprocess.run([str(resolve_dorado()), *argv], check=True)
        base = Path(models_dir) if models_dir is not None else Path.cwd()
        return base / model.model_name()


__all__ = [
    "DoradoRunner",
    "RunHandle",
    "RunStatus",
    "DoradoProgressEvent",
    "partial_path",
    "clear_partial",
]
