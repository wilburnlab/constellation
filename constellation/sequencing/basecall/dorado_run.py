"""Generic ``dorado`` subprocess primitive — spawn + RunHandle plumbing.

Use-case-agnostic (mirrors ``minimap2_run`` / ``hifiasm_run``): knows
nothing about basecaller-vs-aligner-vs-summary — it takes an argv tuple,
redirects stdout to the output file (Dorado writes BAM/FASTA to stdout)
and stderr to a log, and owns the non-blocking + detach mechanics so the
``DoradoRunner`` methods stay thin.

Long simplex runs take days; ``detach=True`` fully decouples the process
(``start_new_session``) and records a PID file so a later ``basecall
--follow`` session can ``RunHandle.attach`` to it. A detached run is
wrapped in a shell that writes its real exit code to a sentinel file, so
``poll`` / ``wait`` recover success vs failure without owning the child.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import ToolNotFoundError, find, version_ge


# Constellation's genome pipeline targets Dorado 2.0.0+ exclusively: the
# `dorado polish` secondary-analysis CLI and the 26.01 output spec the
# pipeline is built against landed in the 1.x→2.0 line, and we deliberately
# carry NO compatibility shims for older Dorado — an older binary is refused
# rather than allowed to silently produce wrong results.
DORADO_MIN_VERSION = "2.0.0"


class DoradoVersionError(RuntimeError):
    """Raised when the resolved Dorado is older than ``DORADO_MIN_VERSION``."""


_INSTALL_HINT = (
    "dorado not found; install via `bash scripts/install-dorado.sh` "
    "(downloads from ONT's CDN — Oxford Nanopore PLC Public License, "
    "research use only) or set $CONSTELLATION_DORADO_HOME"
)


def resolve_dorado() -> Path:
    """Resolve the dorado binary, enforcing ``DORADO_MIN_VERSION``.

    Raises ``FileNotFoundError`` if dorado isn't installed, or
    ``DoradoVersionError`` if the resolved binary is older than 2.0.0
    (when its version can be probed).
    """
    try:
        handle = find("dorado")
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_INSTALL_HINT) from exc
    if handle.version is not None and not version_ge(handle.version, DORADO_MIN_VERSION):
        raise DoradoVersionError(
            f"dorado {handle.version} is too old — constellation requires "
            f">= {DORADO_MIN_VERSION}. Upgrade with `bash scripts/install-dorado.sh` "
            f"or point $CONSTELLATION_DORADO_HOME at a {DORADO_MIN_VERSION}+ install."
        )
    return handle.path


def dorado_version() -> str | None:
    try:
        return find("dorado").version
    except ToolNotFoundError:
        return None


def _exit_file_for(pid_file: Path | None, output_path: Path) -> Path:
    if pid_file is not None:
        return Path(str(pid_file) + ".exit")
    return output_path.with_name(output_path.name + ".exit")


def spawn_dorado(
    argv: list[str],
    *,
    output_path: Path,
    stderr_log: Path,
    detach: bool = False,
    pid_file: Path | None = None,
    started_at: float = 0.0,
    env: dict[str, str] | None = None,
) -> tuple[int, subprocess.Popen | None, Path | None]:
    """Spawn ``dorado <argv>`` with stdout→``output_path``, stderr→log.

    Returns ``(pid, popen_or_None, exit_file_or_None)``. In foreground
    mode the live ``Popen`` is returned (caller drives ``wait``/``poll``
    on it); in ``detach`` mode the process is decoupled, a PID file +
    exit-code sentinel are written, and the ``Popen`` is ``None``.
    """
    output_path = Path(output_path)
    stderr_log = Path(stderr_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    full = [str(resolve_dorado()), *argv]

    if not detach:
        out_fh = output_path.open("wb")
        err_fh = stderr_log.open("wb")
        proc = subprocess.Popen(full, stdout=out_fh, stderr=err_fh, env=env)
        # The child holds its own dup'd fds; close the parent copies so we
        # don't leak a handle per run.
        out_fh.close()
        err_fh.close()
        return proc.pid, proc, None

    exit_file = _exit_file_for(pid_file, output_path)
    exit_file.unlink(missing_ok=True)
    wrapper = (
        f"{shlex.join(full)} > {shlex.quote(str(output_path))} "
        f"2> {shlex.quote(str(stderr_log))}; "
        f"echo $? > {shlex.quote(str(exit_file))}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapper],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "argv": full,
                    "output_path": str(output_path),
                    "stderr_log": str(stderr_log),
                    "exit_file": str(exit_file),
                    "started_at": started_at,
                },
                indent=2,
            )
        )
    return proc.pid, None, exit_file


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process (signal 0 probe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_exit_code(exit_file: Path | None) -> int | None:
    """Read the detached-run exit sentinel, or ``None`` if not yet written."""
    if exit_file is None:
        return None
    p = Path(exit_file)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


__all__ = [
    "DORADO_MIN_VERSION",
    "DoradoVersionError",
    "resolve_dorado",
    "dorado_version",
    "spawn_dorado",
    "pid_alive",
    "read_exit_code",
]
