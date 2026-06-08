"""Generic RagTag subprocess wrapper — use-case-agnostic.

Mirrors ``minimap2_run`` / ``hifiasm_run``: builds the ``ragtag.py
scaffold`` command line, runs it, tees output to a log, and returns the
output dir (where ``ragtag.scaffold.fasta`` + ``ragtag.scaffold.agp``
land). The ``RagTagRunner`` orchestrator does the FASTA materialization
and AGP parsing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from constellation.core.progress import ProgressCallback, ProgressEvent
from constellation.thirdparty.registry import ToolNotFoundError, find


_INSTALL_HINT = (
    "ragtag not found; install via bioconda (`conda install -c bioconda ragtag`) "
    "or `bash scripts/install-ragtag.sh`, or set $CONSTELLATION_RAGTAG_HOME"
)


def _resolve_ragtag() -> Path:
    try:
        return find("ragtag").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_INSTALL_HINT) from exc


def _emit(progress_cb: ProgressCallback | None, event: ProgressEvent) -> None:
    if progress_cb is not None:
        progress_cb(event)


def _tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        )
    except OSError:
        return ""


def ragtag_run(
    reference_fasta: Path,
    query_fasta: Path,
    output_dir: Path,
    *,
    args: tuple[str, ...] = (),
    threads: int = 8,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run ``ragtag.py scaffold`` and return ``output_dir``.

    Builds ``[ragtag.py, scaffold, *args, -t N, -o <dir>, <ref>, <query>]``.
    Raises ``FileNotFoundError`` (install hint) if RagTag is absent,
    ``subprocess.CalledProcessError`` (stderr tail) on a non-zero exit.
    """
    ragtag = _resolve_ragtag()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ragtag),
        "scaffold",
        *args,
        "-t",
        str(int(threads)),
        "-o",
        str(output_dir),
        str(reference_fasta),
        str(query_fasta),
    ]
    log_path = output_dir / "ragtag.log"
    _emit(
        progress_cb,
        ProgressEvent(kind="stage_start", stage="ragtag", message=" ".join(cmd)),
    )
    with log_path.open("wb") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, stderr=_tail(log_path)
        )
    _emit(
        progress_cb,
        ProgressEvent(kind="stage_done", stage="ragtag", message=str(output_dir)),
    )
    return output_dir


__all__ = ["ragtag_run"]
