"""POD5 raw-signal reader → ``RAW_SIGNAL_TABLE``.

POD5 is the canonical Oxford Nanopore raw-signal container format
(supersedes the older fast5). One file holds many reads, each with
metadata (read_id, channel, well, run_id, sampling_rate, scale,
offset) plus the int16 ADC trace.

Uses the ``pod5`` PyPI package — declared in the ``[sequencing]``
extra (already in pyproject.toml). Imports deferred to inside ``read``
so the reader module imports without pod5 installed.

Status: STUB. Pending Phase 2.
"""

from __future__ import annotations

from typing import ClassVar

from constellation.core.io.readers import RawReader, ReadResult, register_reader


_PHASE = "Phase 2 (readers/pod5)"


@register_reader
class Pod5Reader(RawReader):
    """Decodes ``.pod5`` (and legacy ``.fast5``) into RAW_SIGNAL_TABLE.

    Producers in this reader stamp ``acquisition_id`` from the source
    path's enclosing directory or from caller-supplied metadata; the
    reader itself doesn't know which acquisition the file belongs to
    until the calling layer (typically ``Project``) supplies it.
    """

    suffixes: ClassVar[tuple[str, ...]] = (".pod5",)
    modality: ClassVar[str | None] = "nanopore"

    def read(self, source) -> ReadResult:
        # Imports deferred so module-level import doesn't require pod5.
        # import pod5  # noqa: ERA001
        raise NotImplementedError(f"Pod5Reader.read pending {_PHASE}")


__all__ = ["Pod5Reader"]
