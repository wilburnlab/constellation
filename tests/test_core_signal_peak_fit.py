"""Peak shapes fitted to every peak at once (core.signal.peak_fit), and the
shape-fit refiner of the peak picker."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from constellation.core.signal.peak_fit import fit_peaks
from constellation.core.signal.peaks import Fraction, LocalMaxima, PeakPicker, SavitzkyGolay, ShapeFit
from constellation.core.stats.peaks import PEAK_FORMS, EMGPeak, GaussianPeak


def test_a_gaussian_fit_places_the_apex_between_irregular_samples() -> None:
    g = np.random.default_rng(5)
    x = np.cumsum(g.uniform(2.0, 3.0, size=(64, 150)), axis=1)
    apex = x[:, 70] + g.uniform(-1, 1, 64)
    y = 80 * np.exp(-0.5 * ((x - apex[:, None]) / 4.0) ** 2) + g.normal(scale=1.0, size=(64, 150))
    picker = PeakPicker(detect=LocalMaxima(prominence=30.0), smooth=SavitzkyGolay(5, 2), boundaries=Fraction(0.02),
                        refine=ShapeFit(form="gaussian"), max_peaks=1)  # fmt: skip
    p = picker(torch.tensor(y), x=torch.tensor(x))
    assert p.form == "gaussian" and p.components.shape == (64, 1, 3)
    assert np.median(np.abs(p.apex_x[:, 0].numpy() - apex)) < 0.1  # samples are 2-3 apart
    assert np.median(p.components[:, 0, 2].numpy()) == pytest.approx(4.0, rel=0.1)


def test_an_emg_fit_recovers_its_shape() -> None:
    g = np.random.default_rng(6)
    t = np.linspace(0, 60, 120)[None].repeat(32, 0)
    truth = EMGPeak(N_max=50.0, t_apex=25.0, sigma=2.0, tau=4.0)
    y = truth.forward(torch.tensor(t)).detach().numpy() + g.normal(scale=0.5, size=t.shape)
    picker = PeakPicker(detect=LocalMaxima(prominence=20.0), boundaries=Fraction(0.01), refine=ShapeFit(form="emg"),
                        max_peaks=1)  # fmt: skip
    p = picker(torch.tensor(y), x=torch.tensor(t))
    np.testing.assert_allclose(np.median(p.components[:, 0].numpy(), axis=0), [50.0, 25.0, 2.0, 4.0], rtol=0.05)


def test_the_forms_are_the_peak_classes_formulas() -> None:
    t = torch.linspace(-5, 30, 50, dtype=torch.float64)
    gp = GaussianPeak(N_max=7.0, t_apex=3.0, sigma=1.5)
    theta = torch.tensor([[7.0, 3.0, 1.5]], dtype=torch.float64)
    assert torch.allclose(PEAK_FORMS["gaussian"].evaluate(t.unsqueeze(0), theta)[0], gp.forward(t).detach())
    assert PEAK_FORMS["gaussian"].area(theta).item() == pytest.approx(gp.integrate().item())
    ep = EMGPeak(N_max=7.0, t_apex=3.0, sigma=1.5, tau=4.0)
    theta = torch.tensor([[7.0, 3.0, 1.5, 4.0]], dtype=torch.float64)
    assert torch.allclose(PEAK_FORMS["emg"].evaluate(t.unsqueeze(0), theta)[0], ep.forward(t).detach())
    assert PEAK_FORMS["emg"].area(theta).item() == pytest.approx(ep.integrate().item(), rel=1e-12)


def test_padding_peaks_get_no_fit() -> None:
    x = torch.arange(40, dtype=torch.float64).unsqueeze(0)
    y = 10 * torch.exp(-0.5 * ((x - 20) / 3) ** 2)
    fit = fit_peaks(x, y, torch.ones_like(y, dtype=torch.bool), torch.tensor([[10, -1]]), torch.tensor([[30, -1]]),
                    torch.tensor([[True, False]]))  # fmt: skip
    assert fit.converged.tolist() == [[True, False]]
    assert fit.apex[0, 0].item() == pytest.approx(20.0, abs=1e-6) and torch.isnan(fit.apex[0, 1])
