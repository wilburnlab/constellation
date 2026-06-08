"""Generic hifiasm subprocess wrapper — use-case-agnostic.

Mirrors :mod:`sequencing.align.minimap2` (generic ``minimap2_run``):
knows nothing about ONT-vs-HiFi mode or output parsing — that's the
``HiFiAsmRunner`` orchestrator's job. Accepts an arbitrary
``args: tuple[str, ...]`` so the mode flag (``--ont``) and any escape-hatch
flags compose without subclassing.

hifiasm writes its assembly graphs to ``<output_prefix>.*`` and logs
progress to stderr; the runner tees stdout+stderr to
``<output_prefix>.hifiasm.log`` so a multi-hour run stays diagnosable.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from constellation.core.progress import ProgressCallback, ProgressEvent
from constellation.thirdparty.registry import ToolNotFoundError, find


_INSTALL_HINT = (
    "hifiasm not found; install via bioconda "
    "(`conda install -c bioconda hifiasm`) or `bash scripts/install-hifiasm.sh`, "
    "or set $CONSTELLATION_HIFIASM_HOME"
)


def _resolve_hifiasm() -> Path:
    try:
        return find("hifiasm").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_INSTALL_HINT) from exc


def _emit(progress_cb: ProgressCallback | None, event: ProgressEvent) -> None:
    if progress_cb is not None:
        progress_cb(event)


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def hifiasm_run(
    read_paths: Sequence[Path],
    output_prefix: Path,
    *,
    args: tuple[str, ...] = (),
    threads: int = 16,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run ``hifiasm`` and return ``output_prefix`` (the ``-o`` value).

    Builds ``[hifiasm, *args, -t N, -o <prefix>, *reads]``. ``args`` is
    the caller-controlled flag tuple (e.g. ``('--ont',)``). Output files
    land at ``<output_prefix>.*`` — the orchestrator resolves which GFA
    to parse.

    Raises ``FileNotFoundError`` with an install hint when hifiasm is
    absent, ``subprocess.CalledProcessError`` (with a stderr tail) on a
    non-zero exit.
    """
    hifiasm_bin = _resolve_hifiasm()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(hifiasm_bin),
        *args,
        "-t",
        str(int(threads)),
        "-o",
        str(output_prefix),
        *(str(p) for p in read_paths),
    ]
    log_path = output_prefix.parent / f"{output_prefix.name}.hifiasm.log"

    _emit(
        progress_cb,
        ProgressEvent(kind="stage_start", stage="hifiasm", message=" ".join(cmd)),
    )
    with log_path.open("wb") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, stderr=_tail(log_path)
        )
    _emit(
        progress_cb,
        ProgressEvent(kind="stage_done", stage="hifiasm", message=str(output_prefix)),
    )
    return output_prefix


__all__ = ["hifiasm_run"]
