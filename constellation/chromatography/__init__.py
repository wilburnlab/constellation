"""Chromatography domain module — HPLC, LC, GC.

First-class target: Agilent OpenLab ``.dx`` (HPLC-DAD), reverse-
engineered in ~/projects/hplc_analysis. Format spec lives in that
repo's FORMAT_NOTES.md; the in-Constellation reader is a clean
rewrite, not a vendoring.

Imports `core` only; never cross-imports to other domain modules.
LC-MS hyphenation will land alongside the MS port — the chromatography
reader will handle the LC time-axis and the MS reader the m/z scans,
sharing the same ``Bundle`` for files that pack both into one
container.

Modules (TODO; scaffolded only):
    readers/         - agilent_dx (Signal179 / InstrumentTrace179 /
                       Spectra131 decoders + ReadResult assembly)
    peaks            - chromatogram → PeakTable workflow
                       (baseline → find → fit-EMG → integrate)
    quantitation     - calibration curves, internal-standard
                       normalisation
"""
