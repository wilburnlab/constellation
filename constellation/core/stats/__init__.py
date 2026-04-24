"""Parametric functional forms, losses, and measurement units.

`Parametric` ABC unifies probability densities (Student-t, Normal,
GeneralizedNormal, Dirichlet, ...) and observable shapes (Gaussian,
EMG, WarpedEMG, SplinePeak, HyperEMGPeak) — both are differentiable
nn.Module subclasses with forward / log_prob / .fit(data). Fitting
drives NLL minimization through `core.optim` — no scipy in the path.

Modules (TODO; scaffolded only):
    distributions    - Parametric ABC + densities + peak shapes
    losses           - kld, spectral_angle, spectral_entropy_loss
    units            - Da/ppm/etc. constants + conversions
"""
