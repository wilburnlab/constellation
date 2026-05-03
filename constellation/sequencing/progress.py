"""Pluggable progress reporting for long-running sequencing pipelines.

The CLI binds a callback that prints to stdout (or wraps tqdm); a
future GUI binds the same callback to a Qt signal or a websocket. The
core verbs in :mod:`sequencing.parallel` and :mod:`sequencing.transcriptome`
take ``progress_cb: ProgressCallback | None`` and emit at well-known
points without caring how the host renders them.

Three event types cover S1's needs; future event types (sub-stage
metrics, per-shard timing) can be added without breaking the Protocol
because each event carries an extras-dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


# Event kinds. Kept small in S1; expand only when callers need finer
# granularity. The host renders each kind however it likes — the
# default :class:`StreamProgress` prints them as one line each.
ProgressEventKind = Literal["stage_start", "stage_progress", "stage_done"]


@dataclass(frozen=True)
class ProgressEvent:
    """One progress notification from a running pipeline stage.

    ``stage`` is a short identifier (``"sam_ingest"``, ``"demux"``,
    ``"orf"``, ``"quant"``); the host treats it as opaque. ``completed``
    and ``total`` are integers in arbitrary units (rows, batches,
    reads); a stage may emit multiple ``stage_progress`` events with
    increasing ``completed`` values culminating in ``stage_done``.
    ``payload`` carries free-form extras (timing, shard counts,
    diagnostic metrics).
    """

    kind: ProgressEventKind
    stage: str
    completed: int = 0
    total: int = 0
    message: str = ""
    payload: dict[str, object] = field(default_factory=dict)


class ProgressCallback(Protocol):
    """Receives :class:`ProgressEvent` instances. Implementations must
    be thread-safe (worker subprocesses don't call back, but the
    multiprocessing parent may interleave calls from a result-collector
    thread)."""

    def __call__(self, event: ProgressEvent) -> None: ...


class NullProgress:
    """No-op callback. Used as the default so verbs don't have to
    handle ``None`` everywhere."""

    def __call__(self, event: ProgressEvent) -> None:  # noqa: ARG002
        return None


class StreamProgress:
    """Plain-text progress printer.

    Emits a one-line update per event to ``sys.stderr`` by default
    (so stdout stays clean for piping). Suitable for the CLI's
    ``--progress`` flag without pulling in tqdm.
    """

    def __init__(self, stream=None) -> None:
        import sys

        self._stream = stream if stream is not None else sys.stderr

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind == "stage_start":
            line = f"[{event.stage}] start{(': ' + event.message) if event.message else ''}"
        elif event.kind == "stage_progress":
            if event.total is not None and event.total > 0:
                pct = 100.0 * event.completed / event.total
                line = (
                    f"[{event.stage}] {event.completed}/{event.total} "
                    f"({pct:5.1f}%){(' ' + event.message) if event.message else ''}"
                )
            else:
                # Streaming mode (no advance total) — report counter
                # only. The message carries human context.
                line = (
                    f"[{event.stage}] {event.completed}"
                    f"{(' ' + event.message) if event.message else ''}"
                )
        else:  # stage_done
            line = f"[{event.stage}] done{(': ' + event.message) if event.message else ''}"
        print(line, file=self._stream, flush=True)


def emit_start(
    cb: ProgressCallback | None,
    stage: str,
    *,
    total: int = 0,
    message: str = "",
    payload: dict[str, object] | None = None,
) -> None:
    """Convenience helper for verbs that don't want to import the
    dataclass. ``cb=None`` is a silent no-op."""
    if cb is None:
        return
    cb(
        ProgressEvent(
            kind="stage_start",
            stage=stage,
            total=total,
            message=message,
            payload=payload or {},
        )
    )


def emit_progress(
    cb: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int = 0,
    *,
    message: str = "",
    payload: dict[str, object] | None = None,
) -> None:
    if cb is None:
        return
    cb(
        ProgressEvent(
            kind="stage_progress",
            stage=stage,
            completed=completed,
            total=total,
            message=message,
            payload=payload or {},
        )
    )


def emit_done(
    cb: ProgressCallback | None,
    stage: str,
    *,
    completed: int = 0,
    total: int = 0,
    message: str = "",
    payload: dict[str, object] | None = None,
) -> None:
    if cb is None:
        return
    cb(
        ProgressEvent(
            kind="stage_done",
            stage=stage,
            completed=completed,
            total=total,
            message=message,
            payload=payload or {},
        )
    )


__all__ = [
    "NullProgress",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressEventKind",
    "StreamProgress",
    "emit_done",
    "emit_progress",
    "emit_start",
]
