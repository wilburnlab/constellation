"""Sequencing-domain packaged data — library designs + aligner presets.

Bundled JSON files in this package:

* ``cdna_wilburn_v1.json`` — library design: 5' SSP, 3' adapter, and
  24-barcode panel for the lab's in-house SMARTer-derived chemistry.
  Loaded by :mod:`constellation.sequencing.transcriptome.demux.designs`;
  regenerated via ``scripts/build-sequencing-primers-json.py``.

* ``minimap2_splice_presets.json`` — organism-class minimap2 splice-mode
  preset bundle (compact_eukaryote / intermediate_eukaryote / animal).
  Loaded by :mod:`constellation.sequencing.align.presets`; regenerated
  via ``scripts/build-minimap2-splice-presets-json.py``.
"""
