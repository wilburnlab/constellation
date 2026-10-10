"""Genome-browser modality.

Tracks over a reference genome: the reference sequence and annotation
from the per-user reference cache, plus coverage, read pile-ups,
transcript clusters and splice junctions from attached
``transcriptome align`` / ``cluster`` output directories.

Importing this package registers the genome track kernels.
"""

from constellation.viz.modalities.genome import tracks  # noqa: F401
