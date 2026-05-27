"""S1 demux subpackage — adapter / barcode / polyA segment location,
read classification, ORF prediction, per-sample FASTQ emission.

Files in this subpackage own the per-read decisions that translate a
raw Dorado BAM into the per-sample, per-segment shape downstream
align + cluster stages consume. See ``stages.py`` (one level up) for
the orchestrator that wires these together into the
``constellation transcriptome demultiplex`` CLI command.
"""

from __future__ import annotations
