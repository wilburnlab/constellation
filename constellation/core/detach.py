"""Run a command in the background, detached from the terminal, and follow it.

For runs that outlast a terminal session (a chromatogram library over a
proteome-scale predicted library takes hours). :func:`launch` starts the
command in a session of its own, with everything it prints going to a
log, and leaves a small JSON record beside the log; a wrapping shell
writes the command's exit status to a file when it ends, so the outcome
is known without owning the process. :func:`follow` streams a run's log
from any later session until that status appears.

Records live in a directory the caller chooses (a workflow's output
directory, typically ``<output>/runs/``), one ``<stamp>-<label>.json``,
``.log`` and ``.exit`` per run.

Stopping a run: it is its own process group, so ``kill -TERM -<pid>``
reaches the command and everything it started. Constellation's JVM
runner turns that into an orderly stop (``thirdparty.jvm``). The
wrapping shell is in that group too; it traps the signal, so it lives
on to record the command's exit status.

The same idea as ``sequencing.basecall.dorado_run``'s detached Dorado
runs, without the Dorado specifics.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class DetachedRun:
    """A run started by :func:`launch`, as its record describes it."""

    record: Path
    pid: int
    argv: tuple[str, ...]
    log: Path
    exit_file: Path
    started_at: str
    label: str

    def exit_code(self) -> int | None:
        """The command's exit status, or ``None`` while it has not ended
        (or ended without its wrapper getting to write it)."""
        try:
            return int(self.exit_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def alive(self) -> bool:
        """Whether the wrapping process still runs."""
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - someone else's pid now
            return True
        return True

    def state(self) -> str:
        """``running``, ``finished`` (exit 0), ``failed`` (non-zero exit)
        or ``vanished`` (not running and no exit status: killed outright)."""
        code = self.exit_code()
        if code is not None:
            return "finished" if code == 0 else "failed"
        return "running" if self.alive() else "vanished"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def launch(
    argv: list[str], *, run_dir: Path | str, label: str, env: dict[str, str] | None = None
) -> DetachedRun:
    """Start ``argv`` detached; return its record.

    The command's stdout and stderr both go to the log. Its working
    directory is the caller's.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    base = run_dir / f"{stamp}-{label}"
    n = 1
    while base.with_suffix(".json").exists():  # two launches in one second
        n += 1
        base = run_dir / f"{stamp}-{label}-{n}"
    log, exit_file, record = base.with_suffix(".log"), base.with_suffix(".exit"), base.with_suffix(".json")
    # The shell traps TERM and HUP with a no-op so that a stop request to
    # the whole group (kill -TERM -<pid>) leaves it alive to record the
    # command's exit status; a trapped signal, unlike an ignored one, is
    # reset to its default in the command it starts, so the command
    # still receives and handles it.
    wrapper = (
        "trap 'true' TERM HUP; "
        f"{shlex.join(argv)} > {shlex.quote(str(log))} 2>&1; "
        f"echo $? > {shlex.quote(str(exit_file))}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapper],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "pid": proc.pid,
        "argv": list(argv),
        "log": log.name,
        "exit_file": exit_file.name,
        "started_at": started_at,
        "label": label,
    }
    tmp = record.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(record)
    return _from_record(record)


def _from_record(record: Path) -> DetachedRun:
    data = json.loads(record.read_text())
    return DetachedRun(
        record=record,
        pid=int(data["pid"]),
        argv=tuple(data["argv"]),
        log=record.parent / data["log"],
        exit_file=record.parent / data["exit_file"],
        started_at=str(data["started_at"]),
        label=str(data.get("label", "")),
    )


def runs(run_dir: Path | str) -> list[DetachedRun]:
    """Every run recorded in ``run_dir``, oldest first."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    out = []
    for record in sorted(run_dir.glob("*.json")):
        try:
            out.append(_from_record(record))
        except (ValueError, KeyError, OSError):
            continue  # not one of ours, or half-written
    return sorted(out, key=lambda r: (r.started_at, r.record.name))


def latest(run_dir: Path | str) -> DetachedRun | None:
    """The most recently started run in ``run_dir``."""
    found = runs(run_dir)
    return found[-1] if found else None


def follow(
    run: DetachedRun, *, out: TextIO | None = None, poll_s: float = 1.0, from_start: bool = True
) -> int | None:
    """Copy ``run``'s log to ``out`` as it grows until the run ends.

    Returns its exit status, or ``None`` when it stopped without one
    (killed outright). ``from_start=False`` begins at the log's current
    end, like ``tail -f``.
    """
    out = out if out is not None else sys.stderr
    position = 0
    if not from_start and run.log.exists():
        position = run.log.stat().st_size
    while True:
        code = run.exit_code()
        alive = code is None and run.alive()
        if run.log.exists():
            with run.log.open("r", errors="replace") as fh:
                fh.seek(position)
                chunk = fh.read()
                position = fh.tell()
            if chunk:
                out.write(chunk)
                out.flush()
        if code is not None:
            return code
        if not alive:
            # The wrapper may have exited between the two checks above.
            return run.exit_code()
        time.sleep(poll_s)


__all__ = ["DetachedRun", "follow", "latest", "launch", "runs"]
