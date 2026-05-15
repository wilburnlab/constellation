"""Generic minimap2 subprocess wrapper — use-case-agnostic.

Knows nothing about splice / preset / DNA-vs-cDNA — that's the
orchestrator layer's job (see ``map_to_genome`` in :mod:`.map`).
The runner accepts an arbitrary ``args: tuple[str, ...]`` so future
verbs (``map_assembly``, ``map_dna_to_genome``, PAF-output workflows)
all compose this primitive with their own flag sets without subclassing
or wrapping use-case-specific helpers.

Output format is inferred from ``output_path`` extension; when ``.bam``
the runner pipes minimap2's SAM stdout through ``samtools view -b``
(no sort — sort/index is a separate concern at the orchestrator level).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from constellation.sequencing.progress import ProgressCallback, ProgressEvent
from constellation.thirdparty.registry import ToolNotFoundError, find


_BIOCONDA_HINT = (
    "minimap2 not on $PATH; install via bioconda: "
    "`conda install -c bioconda minimap2 samtools`"
)


def _resolve_minimap2() -> Path:
    try:
        return find("minimap2").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_BIOCONDA_HINT) from exc


def _resolve_samtools() -> Path:
    try:
        return find("samtools").path
    except ToolNotFoundError as exc:
        raise FileNotFoundError(_BIOCONDA_HINT) from exc


def _emit(progress_cb: ProgressCallback | None, event: ProgressEvent) -> None:
    if progress_cb is not None:
        progress_cb(event)


def minimap2_run(
    target: Path,
    queries: Sequence[Path],
    *,
    output_path: Path,
    args: tuple[str, ...] = (),
    threads: int = 8,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run ``minimap2`` against ``target`` with ``queries`` as input.

    ``args`` is the caller-controlled minimap2 argument tuple (e.g.
    ``('-ax', 'splice', '-uf')``). The runner appends ``-t {threads}``
    plus the target + queries; nothing else is implied. Output format
    is inferred from ``output_path``:

      * ``.sam`` — SAM text written directly via minimap2
      * ``.bam`` — SAM piped through ``samtools view -b``
      * ``.paf`` — caller is responsible for not passing ``-a``

    Raises ``FileNotFoundError`` with a bioconda hint if minimap2 (or
    samtools, when ``.bam`` output is requested) is not on ``$PATH``.
    Raises ``subprocess.CalledProcessError`` on minimap2 / samtools
    non-zero exit.
    """
    minimap2_bin = _resolve_minimap2()
    target = Path(target)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_mm2 = [
        str(minimap2_bin),
        *args,
        "-t",
        str(int(threads)),
        str(target),
        *(str(q) for q in queries),
    ]

    suffix = output_path.suffix.lower()
    _emit(
        progress_cb,
        ProgressEvent(
            kind="stage_start",
            stage="minimap2",
            message=" ".join(cmd_mm2),
        ),
    )

    if suffix == ".bam":
        samtools_bin = _resolve_samtools()
        cmd_view = [str(samtools_bin), "view", "-b", "-@", str(int(threads))]
        with output_path.open("wb") as out:
            mm2_proc = subprocess.Popen(cmd_mm2, stdout=subprocess.PIPE)
            try:
                view_proc = subprocess.Popen(
                    cmd_view, stdin=mm2_proc.stdout, stdout=out
                )
                # Allow mm2 process to receive SIGPIPE if view exits
                if mm2_proc.stdout is not None:
                    mm2_proc.stdout.close()
                view_rc = view_proc.wait()
                mm2_rc = mm2_proc.wait()
            finally:
                if mm2_proc.poll() is None:
                    mm2_proc.kill()
        if mm2_rc != 0:
            raise subprocess.CalledProcessError(mm2_rc, cmd_mm2)
        if view_rc != 0:
            raise subprocess.CalledProcessError(view_rc, cmd_view)
    else:
        # SAM / PAF / other text output goes straight to file
        with output_path.open("wb") as out:
            subprocess.run(cmd_mm2, stdout=out, check=True)

    _emit(
        progress_cb,
        ProgressEvent(
            kind="stage_done",
            stage="minimap2",
            message=str(output_path),
        ),
    )
    return output_path


def minimap2_build_index(
    fasta: Path,
    mmi_path: Path,
    *,
    preset: str | None = None,
    threads: int = 8,
    extra_args: tuple[str, ...] = (),
) -> Path:
    """Build a ``.mmi`` index via ``minimap2 -d``.

    ``preset`` (e.g. ``"splice"``) sets the indexing ``-k``/``-w`` via
    ``-x``. minimap2 bakes ``-k``/``-w`` into the index and warns at
    alignment time if the alignment preset wants different values, so
    callers must pass the same preset the aligner will use. ``None``
    keeps minimap2's defaults — appropriate only for callers that align
    without a preset.

    Skips rebuild if ``mmi_path`` exists and is newer than ``fasta``.
    Returns ``mmi_path``.
    """
    fasta = Path(fasta)
    mmi_path = Path(mmi_path)
    if mmi_path.exists() and mmi_path.stat().st_mtime >= fasta.stat().st_mtime:
        return mmi_path
    minimap2_bin = _resolve_minimap2()
    mmi_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(minimap2_bin),
        *(("-x", preset) if preset else ()),
        *extra_args,
        "-t",
        str(int(threads)),
        "-d",
        str(mmi_path),
        str(fasta),
    ]
    subprocess.run(cmd, check=True)
    return mmi_path


__all__ = ["minimap2_run", "minimap2_build_index"]
