"""Optimizers — differential evolution + PyTorch wrappers.

OPTIMIZER_REGISTRY exposes DE and torch.optim families through a
single interface so Parametric.fit() can pick an optimizer by name.

Modules (TODO; scaffolded only):
    de               - DifferentialEvolution for nn.Module params; Sobol
                       init; rand/1, best/1, current-to-best/1 mutation;
                       optional LBFGS polish step folded in. Port from
                       cartographer/optimize.py@0cee299.
    registry         - OPTIMIZER_REGISTRY uniting DE + torch optimizers
"""
