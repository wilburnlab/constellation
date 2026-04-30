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

Detached mode (``detach=True``) writes a PID file and exits the Python
process, leaving Dorado running. ``RunHandle.attach(pid_file)`` rebinds
to the running process so a separate ``constellation basecall --follow``
session can monitor it.

License: Dorado is the Oxford Nanopore Technologies PLC. Public
License Version 1.0 — Research Purposes only. The wrapper itself is
Apache 2.0 (Constellation), but anyone *invoking* this code accepts
ONT's terms. See LICENCE in this directory (lands Phase 3) and the
upstream LICENCE.txt:
https://github.com/nanoporetech/dorado/blob/release-v1.4/LICENCE.txt

Status: STUB. Subcommand methods raise ``NotImplementedError`` pending
Phase 3.
"""

from __future__ import annotations

import signal as _signal
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from constellation.sequencing.basecall.models import DoradoModel


_PHASE = "Phase 3 (basecall + thirdparty/dorado + scripts/install-dorado.sh)"


# ──────────────────────────────────────────────────────────────────────
# Run lifecycle
# ──────────────────────────────────────────────────────────────────────


class RunStatus(Enum):
    PENDING = "pending"        # spawned but not yet started
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"        # e.g. PID file lost


@dataclass(frozen=True)
class DoradoProgressEvent:
    """One progress event parsed from Dorado's stderr stream.

    Dorado emits structured progress lines (reads-processed,
    bases-emitted, ETA). The handler in ``RunHandle.tail_progress``
    parses these into events the CLI can render as a status bar.
    """

    elapsed_s: float
    reads_processed: int
    bases_emitted: int
    eta_s: float | None
    raw: str  # original stderr line, kept for debugging


@dataclass(frozen=True)
class RunHandle:
    """Handle to a running (or completed) Dorado subprocess.

    Constructed by ``DoradoRunner`` subcommand methods; rebindable from
    a PID file via ``RunHandle.attach``. Long-running runs that span
    Python sessions use the PID file as the source of truth.
    """

    pid: int
    pid_file: Path
    output_path: Path
    stdout_log: Path
    stderr_log: Path
    started_at: float = 0.0  # unix timestamp
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def attach(cls, pid_file: Path) -> "RunHandle":
        """Re-bind to a previously detached run from its PID file."""
        raise NotImplementedError(f"RunHandle.attach pending {_PHASE}")

    def wait(self, *, timeout: float | None = None) -> int:
        """Block until the subprocess exits. Returns exit code."""
        raise NotImplementedError(f"RunHandle.wait pending {_PHASE}")

    def poll(self) -> RunStatus:
        """Non-blocking status check."""
        raise NotImplementedError(f"RunHandle.poll pending {_PHASE}")

    def tail_progress(self) -> Iterator[DoradoProgressEvent]:
        """Iterator over progress events parsed from stderr."""
        raise NotImplementedError(f"RunHandle.tail_progress pending {_PHASE}")

    def cancel(self, *, sig: int = _signal.SIGTERM) -> None:
        """Signal the subprocess (default SIGTERM; SIGKILL escalates)."""
        raise NotImplementedError(f"RunHandle.cancel pending {_PHASE}")


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DoradoRunner:
    """Thin wrapper around the ``dorado`` binary.

    Resolved at instantiation time via ``constellation.thirdparty.find(
    'dorado')``; raises ``ToolNotFoundError`` if Dorado isn't installed.

    Each subcommand method returns a :class:`RunHandle`; callers choose
    block / follow / detach. Default ``device='cuda:0'`` matches the
    lab's typical configuration; CPU-only hosts pass ``device='cpu'``.
    """

    device: str = "cuda:0"
    threads: int = 8
    extra_args: tuple[str, ...] = ()

    def basecaller(
        self,
        model: DoradoModel,
        pod5_paths: list[Path],
        output: Path,
        *,
        modified_bases: tuple[str, ...] = (),
        device: str | None = None,
        resume: bool = True,
        detach: bool = False,
    ) -> RunHandle:
        """Run ``dorado basecaller`` on the given POD5 inputs.

        Output is BAM (Dorado's default). ``resume=True`` passes
        ``--resume-from`` if a partial output exists. ``detach=True``
        forks Dorado, writes a PID file under ``output.parent``, and
        returns immediately.

        Modified bases default to those declared in ``model.mods`` if
        empty; pass an explicit tuple to override.
        """
        raise NotImplementedError(f"DoradoRunner.basecaller pending {_PHASE}")

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
        raise NotImplementedError(f"DoradoRunner.duplex pending {_PHASE}")

    def aligner(
        self,
        reference: Path,
        bam: Path,
        output: Path,
        *,
        threads: int | None = None,
    ) -> RunHandle:
        """Run ``dorado aligner`` (uses minimap2 internally to map an
        unaligned BAM against a reference; output BAM has alignment
        records populated)."""
        raise NotImplementedError(f"DoradoRunner.aligner pending {_PHASE}")

    def demux(
        self,
        bam: Path,
        output_dir: Path,
        *,
        kit: str | None = None,
        emit_fastq: bool = False,
    ) -> RunHandle:
        """Run ``dorado demux`` — ONT's built-in barcoding demultiplex.

        Note: Constellation's transcriptome workflow uses our own
        :mod:`sequencing.transcriptome.demux` (segmented edlib-based
        algorithm) rather than this — Dorado's full-primer Smith-
        Waterman scoring distorts on high-homopolymer-error oligo-dT
        sequences. This wrapper is here for genomic-DNA workflows
        where the standard demux is fine.
        """
        raise NotImplementedError(f"DoradoRunner.demux pending {_PHASE}")

    def summary(
        self,
        bam: Path,
        output: Path,
    ) -> RunHandle:
        """Run ``dorado summary`` — emit per-read summary TSV."""
        raise NotImplementedError(f"DoradoRunner.summary pending {_PHASE}")

    def fetch_model(self, model: DoradoModel, models_dir: Path | None = None) -> Path:
        """Run ``dorado download --model <model_name>`` and return the
        resolved model directory.

        Caches into ``models_dir`` (or a default under
        ``$CONSTELLATION_DORADO_HOME/models/``).
        """
        raise NotImplementedError(f"DoradoRunner.fetch_model pending {_PHASE}")


__all__ = [
    "DoradoRunner",
    "RunHandle",
    "RunStatus",
    "DoradoProgressEvent",
]
