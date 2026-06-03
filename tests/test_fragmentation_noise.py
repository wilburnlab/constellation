"""FragmentationNoiseModel — the gain must be identified, not divergent.

Regression for the missing change-of-variables Jacobian: with ``v = g·x`` the
log-density of the intensity ``v`` needs ``-K·log g``; without it the likelihood
rises without bound as ``g → ∞`` (``v/g → 0``, ``DM_log_prob(0) = 0``) and a
trainable gain runs away.
"""

import torch

from constellation.core.stats.distributions import dirichlet_multinomial_log_prob
from constellation.massspec.spectra.noise import FragmentationNoiseModel


def _simulate(g_true, p, alpha0, Ns, reps, seed=0):
    torch.manual_seed(seed)
    dir_ = torch.distributions.Dirichlet(alpha0 * p)
    vs = []
    for N in Ns:
        for _ in range(reps):
            pd = dir_.sample()
            x = torch.distributions.Multinomial(int(N), pd).sample()
            vs.append(g_true * x)
    return torch.stack(vs).to(torch.float64)


def test_log_prob_includes_gain_jacobian():
    K = 5
    m = FragmentationNoiseModel(
        K, log_g_init=1.5, concentration_init=torch.tensor([2.0, 3.0, 1.0, 4.0, 2.0])
    )
    v = torch.tensor([[120.0, 240.0, 60.0, 360.0, 180.0]], dtype=torch.float64)
    expected = dirichlet_multinomial_log_prob(v / m.g, m.concentration) - K * torch.log(
        m.g
    )
    assert torch.allclose(m.log_prob(v), expected)


def test_gain_does_not_run_away():
    # Codex's bug: without the -K*log g Jacobian the NLL fell monotonically toward 0
    # as g -> inf (v/g -> 0, DM_log_prob(0) = 0), so a trainable gain ran away and
    # alpha0/probs became meaningless. With the Jacobian, growing the gain is
    # penalised. (g is only weakly identified from MS2 alone -- it trades off
    # against alpha0 through the effective N -- so we assert the divergence is
    # fixed, not tight recovery; fix/MS1-constrain g in practice.)
    p = torch.tensor([0.30, 0.24, 0.20, 0.12, 0.08, 0.06], dtype=torch.float64)
    alpha0, g_true = 40.0, 8.0
    v = _simulate(g_true, p, alpha0, Ns=[30, 60, 120, 240, 480], reps=40)
    m = FragmentationNoiseModel(len(p), concentration_init=alpha0 * p)

    def nll(g):
        with torch.no_grad():
            m.log_g.copy_(torch.log(torch.tensor(float(g), dtype=torch.float64)))
            return float(-m.log_prob(v).sum())

    big = [g_true * f for f in (1.0, 10.0, 1e2, 1e3, 1e4, 1e6)]
    nlls = [nll(g) for g in big]
    assert all(nlls[i + 1] > nlls[i] for i in range(len(nlls) - 1)), nlls
