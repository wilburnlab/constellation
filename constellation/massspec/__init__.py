"""Mass-spectrometry domain module (first port target from Cartographer).

DIA-focused proteomics; peaks + peptides + chromatograms + libraries +
EncyclopeDIA search. Imports `core` only; never imports other domain
modules (sequencing/structure/...).

Modules (TODO; scaffolded only):
    schemas          - mzpeak / scanmeta / acqmeta Arrow schemas
    readers/         - mzml, thermo_raw, bruker_d, mzpeak subclasses of
                       core.io.RawReader
    peptide/         - ions, envelope, encoding (peptide-specific mass math)
    calibration      - MzCalibration(nn.Module) via core.optim.de
    error_models     - StudentT-backed m/z error models
    chromatogram     - XIC extraction + windowed scoring (core.shapes.peaks)
    scoring          - PSM scoring, FDR, spectral entropies (the "Counter"
                       consolidation home)
    library/         - Koina REST client, spectral-library wrapper
    search/          - EncyclopeDIA wrapper, MSFragger TSV reader
"""
