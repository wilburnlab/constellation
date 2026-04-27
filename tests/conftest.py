"""pytest configuration.

The historical `LBFGSAdapter` shim is replaced by the production
`constellation.core.optim.LBFGSOptimizer`. The alias is kept here so
existing tests that import `LBFGSAdapter` from `tests.conftest` keep
working without rename churn.
"""

from __future__ import annotations

from constellation.core.optim import LBFGSOptimizer as LBFGSAdapter  # noqa: F401
