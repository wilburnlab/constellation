"""Multithreaded compression of a producer command's stdout to a file.

A reusable write-side compression primitive. The motivating case: the
genome-assembly pipeline ran ``samtools fastq`` (parallel BAM decode) and
then streamed ~100 GB of decoded reads through Python's single-threaded
``gzip`` module — ~90% of a multi-hour job spent on one idle-cores-but-one
compressor. The fix is to hand the producer's stdout to a real
multithreaded compressor.

``compress_stream_to_path`` runs ``<producer> | bgzip -@ N`` as a
two-process pipe (the htslib ``bgzip`` that ships alongside ``samtools``;
output is BGZF — a valid multi-member gzip stream readable by any gzip
consumer). When ``bgzip`` can't be located it falls back to an in-process
read-loop through the stdlib ``gzip`` module, so callers never hard-depend
on the external binary — the output is always a valid gzip file, just
slower to produce.

Lives in ``thirdparty/`` (not ``core.io``) because resolving the external
``bgzip`` binary is a tool-discovery concern: ``core.io`` is a leaf that
must never import the registry, whereas ``thirdparty`` is importable by
every domain module.
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

from constellation.thirdparty.registry import try_find


def resolve_bgzip() -> Path | None:
    """Locate an ``bgzip`` binary, or ``None`` if unavailable.

    ``bgzip`` ships in the same htslib ``bin/`` as ``samtools``, so we
    prefer the sibling of the resolved ``samtools`` binary — that
    guarantees the BGZF writer matches the ``samtools`` version and is
    found even when ``samtools`` resolved via ``$CONSTELLATION_SAMTOOLS_HOME``
    rather than ``$PATH``. Otherwise fall back to a registered / ``$PATH``
    ``bgzip`` via the registry.
    """
    samtools = try_find("samtools")
    if samtools is not None:
        sibling = samtools.path.parent / "bgzip"
        if sibling.exists():
            return sibling
    handle = try_find("bgzip")
    if handle is not None:
        return handle.path
    return None


def compress_stream_to_path(
    producer_cmd: list[str],
    out_path: Path,
    *,
    threads: int = 1,
    allow_bgzip: bool = True,
    stage: str = "compress",
) -> Path:
    """Run ``producer_cmd`` and write its stdout, gzip-compressed, to ``out_path``.

    When ``bgzip`` is available (and ``allow_bgzip``), pipes
    ``producer_cmd | bgzip -@ threads -c > out_path`` so compression is
    multithreaded. Otherwise reads the producer's stdout in-process and
    compresses through the stdlib ``gzip`` module (single-threaded). Output
    is a valid gzip stream either way.

    ``threads`` is the compressor's thread budget — the producer carries
    its own threading in ``producer_cmd`` (e.g. ``samtools fastq -@ K``).

    Raises ``subprocess.CalledProcessError`` if the producer or the
    compressor exits non-zero.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(threads))

    bgzip = resolve_bgzip() if allow_bgzip else None
    if bgzip is not None:
        _producer_bgzip_pipe(producer_cmd, out_path, bgzip, threads)
    else:
        _producer_pygzip(producer_cmd, out_path)
    return out_path


def _producer_bgzip_pipe(
    producer_cmd: list[str], out_path: Path, bgzip: Path, threads: int
) -> None:
    """``producer_cmd | bgzip -@ threads -c > out_path`` with SIGPIPE teardown.

    Mirrors the two-process-pipe idiom in
    ``sequencing/align/minimap2.py::minimap2_run``: close the producer's
    stdout in the parent so it receives SIGPIPE if ``bgzip`` dies, wait on
    both, check both return codes, and kill a still-running producer in the
    ``finally``.
    """
    cmd_bgzip = [str(bgzip), "-@", str(threads), "-c"]
    with out_path.open("wb") as fout:
        producer = subprocess.Popen(producer_cmd, stdout=subprocess.PIPE)
        try:
            compressor = subprocess.Popen(
                cmd_bgzip, stdin=producer.stdout, stdout=fout
            )
            # Let the producer receive SIGPIPE if bgzip exits early.
            if producer.stdout is not None:
                producer.stdout.close()
            compress_rc = compressor.wait()
            producer_rc = producer.wait()
        finally:
            if producer.poll() is None:
                producer.kill()
    if producer_rc != 0:
        raise subprocess.CalledProcessError(producer_rc, producer_cmd)
    if compress_rc != 0:
        raise subprocess.CalledProcessError(compress_rc, cmd_bgzip)


def _producer_pygzip(producer_cmd: list[str], out_path: Path) -> None:
    """Fallback: read the producer's stdout and compress via stdlib gzip."""
    producer = subprocess.Popen(producer_cmd, stdout=subprocess.PIPE)
    try:
        with gzip.open(out_path, "wb") as gz:
            assert producer.stdout is not None
            for chunk in iter(lambda: producer.stdout.read(1 << 20), b""):
                gz.write(chunk)
    finally:
        if producer.stdout is not None:
            producer.stdout.close()
        rc = producer.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, producer_cmd)


__all__ = ["resolve_bgzip", "compress_stream_to_path"]
