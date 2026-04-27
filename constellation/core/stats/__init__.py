"""Parametric functional forms, losses, and physical units.

Two-tier ABC structure under one umbrella:

    Parametric(nn.Module)              .forward, .fit, parameters_dict
        |
        +-- Distribution                .log_prob, .cdf  (required)
        |       Normal, StudentT, GeneralizedNormal,
        |       Beta, Gamma, LogNormal,
        |       Poisson, Multinomial, Dirichlet
        |
        +-- PeakShape                   .integrate, .bounds  (required)
        |       GaussianPeak, EMGPeak    (.log_prob/.cdf optional)
        |
        +-- (calibration models — leaf, no extra trait)
                Sigmoidal, Hill, LogLinear

Fitting drives NLL minimization (for Distributions) or MSE (for
PeakShapes / calibration) through an external optimizer.
`Parametric.fit(data, *, optimizer, ...)` dispatches on optimizer
shape: an `Optimizer` (gradient-based) gets a scalar closure; a
`PopulationOptimizer` (DE) gets a vmap'd closure that returns a
`(pop_size,)` loss vector. Both Protocols live in `core.optim`.
`core.stats` itself ships zero optimizer implementations.

Submodules:
    parametric       ABC core
    distributions    9 density classes
    peaks            EMG / Gaussian peak shapes (HyperEMG / Warped /
                     Spline land in the next peak-numerics session,
                     now that DE is available)
    calibration      Sigmoidal / Hill / LogLinear standard curves
    losses           kld, spectral_angle, spectral_entropy_loss,
                     l1_normalize, l2_normalize
    units            CODATA constants + ppm/Da conversions
"""

from .parametric import Distribution, FitResult, Parametric, PeakShape
from .distributions import (
    Beta,
    Dirichlet,
    Gamma,
    GeneralizedNormal,
    LogNormal,
    Multinomial,
    NormalDistribution,
    Poisson,
    StudentT,
)
from .peaks import EMGPeak, GaussianPeak, emg_log_pdf, emg_pdf
from .calibration import Hill, LogLinear, Sigmoidal
from .losses import (
    kld,
    l1_normalize,
    l2_normalize,
    spectral_angle,
    spectral_entropy_loss,
)
from . import units

__all__ = [
    # ABCs + fit protocol
    "Parametric",
    "Distribution",
    "PeakShape",
    "FitResult",
    # Densities
    "NormalDistribution",
    "StudentT",
    "GeneralizedNormal",
    "Beta",
    "Gamma",
    "LogNormal",
    "Poisson",
    "Multinomial",
    "Dirichlet",
    # Peak shapes
    "GaussianPeak",
    "EMGPeak",
    "emg_pdf",
    "emg_log_pdf",
    # Calibration
    "Sigmoidal",
    "Hill",
    "LogLinear",
    # Losses
    "kld",
    "spectral_angle",
    "spectral_entropy_loss",
    "l1_normalize",
    "l2_normalize",
    # Submodules
    "units",
]
