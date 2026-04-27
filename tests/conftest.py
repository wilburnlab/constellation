"""pytest configuration.

Test-only optimizer adapter for `core.stats` fit-tests. The real
optimizer wrappers (LBFGSOptimizer, DifferentialEvolution) ship in
`core.optim` next session; until then, this shim exercises the
`Parametric.fit()` interface end-to-end via `torch.optim.LBFGS`.
"""

from __future__ import annotations

import torch


class LBFGSAdapter:
    """Thin shim around `torch.optim.LBFGS` matching the duck-typed
    `optimizer.step(closure)` contract that `Parametric.fit()` expects.

    Test infrastructure only — production uses
    `constellation.core.optim.LBFGSOptimizer` once that ships.
    """

    def __init__(
        self,
        parametric: torch.nn.Module,
        *,
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-7,
    ) -> None:
        self.opt = torch.optim.LBFGS(
            list(parametric.parameters()),
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=tolerance_grad,
        )

    def step(self, closure):
        return self.opt.step(closure)
