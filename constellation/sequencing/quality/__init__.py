"""Sequencing quality scores — Phred codec + per-read QC summaries.

Phred encoding is sequencing-domain idiom (BAM / FASTQ store offset-33
ASCII strings; Q-scores are on the log10 scale of error probability).
Lives here, not in :mod:`core.io`, because it isn't generic across
modalities — chromatograms and MS spectra don't have an analog.
"""

from __future__ import annotations

from constellation.sequencing.quality.phred import (
    decode_phred,
    encode_phred,
    mean_q,
    q_to_p_error,
)

__all__ = [
    "decode_phred",
    "encode_phred",
    "mean_q",
    "q_to_p_error",
]
