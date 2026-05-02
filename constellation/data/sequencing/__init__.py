"""Sequencing-domain packaged data — primer panels, kit definitions.

Each panel ships as a single JSON file (e.g. ``cdna_wilburn_v1.json``)
holding the 5' SSP, 3' adapter, and barcode panel for one library
construct. The builder
``scripts/build-sequencing-primers-json.py`` regenerates them from
inline primer-sequence constants.

Loaded by :mod:`constellation.sequencing.transcriptome.panels`.
"""
