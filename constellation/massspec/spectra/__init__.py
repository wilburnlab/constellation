"""MS2 spectrum-level operations: similarity scoring and consensus building.

Net-new for the ProToStaR Part-I (MS2) analysis. Two submodules:

    similarity   the MS2 spectral-similarity suite — cosine / normalized
                 dot / Pearson / spectral angle / spectral-entropy
                 similarity / KL / multinomial deviance, behind one
                 ``compare_spectra`` dispatcher. The L2-geometry scores and
                 the multinomial-deviance comparator that the Part-I
                 argument contrasts.
    consensus    align replicate fragment spectra to a fixed per-precursor
                 channel basis and aggregate (sum / median) with
                 per-channel dispersion + per-replicate deviance-from-bulk.

Pure functions over aligned intensity / count vectors and PyArrow tables;
all distribution math is reused from ``core.stats`` (``Multinomial``,
``losses``). No model state lives here.
"""

from constellation.massspec.spectra.consensus import (
    ConsensusSpectrum,
    FragmentBasis,
    align_to_basis,
    build_consensus,
    fragment_basis,
)
from constellation.massspec.spectra.noise import FragmentationNoiseModel
from constellation.massspec.spectra.similarity import (
    compare_spectra,
    multinomial_deviance,
)

__all__ = [
    "compare_spectra",
    "multinomial_deviance",
    "FragmentBasis",
    "ConsensusSpectrum",
    "fragment_basis",
    "align_to_basis",
    "build_consensus",
    "FragmentationNoiseModel",
]
