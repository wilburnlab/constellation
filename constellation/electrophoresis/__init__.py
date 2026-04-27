"""Electrophoresis domain module — capillary electrophoresis platforms.

First-class targets: Agilent Fragment Analyzer (nucleic acid CE) and
ProteoAnalyzer (protein CE), both producing ``.raw`` files in the same
LabVIEW-derived big-endian binary family. Reverse-engineered in
~/projects/analyzer_analysis; format spec in that repo's REPORT.md.
The in-Constellation reader is a clean rewrite.

Imports `core` only; never cross-imports to other domain modules.
Calibration models with explicit physics priors (Slater–Noolandi
biased reptation, Ogston-regime small-fragment limit) live under
``electrophoresis.physics`` once developed; the generic Sigmoidal /
Hill / polynomial / monotonic-spline calibration families come from
core (``core.stats.distributions`` and ``core.signal.calibration``).

Modules (TODO; scaffolded only):
    readers/         - agilent_fa (FA + PA share the .raw format;
                       parameterised by RunGeometry). Big-endian
                       uint16 decode via numpy boundary, immediately
                       handed off to torch.
    calibration      - LadderSpec + ladder registry; calibrate_capillary
                       drives the chosen CalibrationCurve family.
    peaks            - electropherogram → PeakTable workflow
                       (baseline → find → fit-EMG → integrate)
    physics          - Slater–Noolandi / reptation calibration models
                       (deferred; protocol-compatible with the generic
                       families).
"""
