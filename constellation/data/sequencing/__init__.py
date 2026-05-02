"""Sequencing-domain packaged data — library designs.

Each design ships as a single JSON file (e.g. ``cdna_wilburn_v1.json``)
holding the 5' SSP, 3' adapter, and barcode panel for one library
design. The builder ``scripts/build-sequencing-primers-json.py``
regenerates them from inline primer-sequence constants.

Loaded by :mod:`constellation.sequencing.transcriptome.designs`.
"""
