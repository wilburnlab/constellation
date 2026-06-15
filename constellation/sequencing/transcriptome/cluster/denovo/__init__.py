"""De novo transcriptome first-round assembly (Phase 3).

Collapses *complete* demuxed reads (full-length 5'→3' transcript windows)
into "transcripts": a per-cluster consensus + quant value, folding in
minor variation (infrequent polymorphisms + short 5'/3' length
differences at low frequency relative to a dominant variant) while
surfacing — not hiding — what was collapsed.

Pipeline (genome-free; input is an S1 demux output dir):

    0. dereplicate   exact derep → unique sequences + abundance
    1. encode        ASCII→uint8 + canonical k-mer rolling hash
    2. minimizers    window minimizers + sorted index
    3. candidates    shared-minimizer + diagonal-filtered candidate pairs
    4. verify        edlib HW verify (emits + caches the CIGAR)
    5. greedy        abundance-anchored greedy set-cover
    6. consensus     centroid-anchored CIGAR-walk PWM (reuses cached CIGARs)
    7. variants      context-aware variant calling (Cut 2)
    8. haplotypes    covariance / phasing (Cut 3)
    9. quant         per-cluster + per-sample counts/TPM (feature_origin='cluster_id')

The public entry point is :func:`cluster_transcripts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from constellation.sequencing.transcriptome.cluster.denovo.pipeline import (
        cluster_transcripts,
    )


def __getattr__(name: str) -> Any:
    # Lazy so importing a leaf module (encode / minimizers / ...) doesn't
    # drag in the whole pipeline + its heavy deps.
    if name == "cluster_transcripts":
        from constellation.sequencing.transcriptome.cluster.denovo.pipeline import (
            cluster_transcripts,
        )

        return cluster_transcripts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cluster_transcripts"]
