"""Mass-spectrometry domain module (first port target from Cartographer).

DIA-focused proteomics; peaks + peptides + chromatograms + libraries +
EncyclopeDIA search. Imports `core` only; never imports other domain
modules (sequencing/structure/...).

Modules:
    schemas          - MS-domain Arrow schemas (FragmentIonTable; mzpeak/
                       chromatogram tables land here when readers ship)
    peptide/         - shipped: m/z, fragment ions, isotope envelopes,
                       biochemical neutral-loss rules. Pure physics.
    tokenize/        - scaffold: per-model peptide tokenizers (Prosit-style,
                       Chronologer-style, ...) — defined per checkpoint.

Modules (TODO; scaffolded only):
    readers/         - mzml, thermo_raw, bruker_d, mzpeak subclasses of
                       core.io.RawReader (next session)
    library/         - Koina REST client, spectral-library wrapper (session
                       after readers)
    calibration      - MzCalibration(nn.Module) via core.optim.de
    error_models     - StudentT-backed m/z error models (MzErrorModel,
                       CalibrationModel dataclasses)
    chromatogram     - XIC extraction + windowed scoring (core.shapes.peaks)
    scoring          - PSM scoring, FDR, spectral entropies (the "Counter"
                       consolidation home)
    search/          - EncyclopeDIA wrapper, MSFragger TSV reader
"""

from constellation.massspec import schemas as schemas  # noqa: F401  (registers FragmentIonTable)

# Importing `io` triggers registration of cross-tier file-format adapters
# (encyclopedia .dlib/.elib reader+writer in LIBRARY_/QUANT_/SEARCH_ READERS,
# DiaReader in core.io.readers).
from constellation.massspec import io as io  # noqa: F401, E402
