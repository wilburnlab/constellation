"""pytest configuration.

The historical `LBFGSAdapter` shim is replaced by the production
`constellation.core.optim.LBFGSOptimizer`. The alias is kept here so
existing tests that import `LBFGSAdapter` from `tests.conftest` keep
working without rename churn. ``fake_centroider`` serves the Thermo
reader's tests.
"""

from __future__ import annotations

import pytest

from constellation.core.optim import LBFGSOptimizer as LBFGSAdapter  # noqa: F401


@pytest.fixture
def fake_centroider(monkeypatch):
    """Stand in for ``Scan.FromFile`` + Thermo's centroider: the "centroids"
    of a profile scan are its points above half its maximum. For the
    stand-in raw files of ``_thermo_fakes``."""
    from _thermo_fakes import Obj

    from constellation.massspec.readers.thermo import _scans

    def load_scan(raw, n):
        s = raw.scans[n]
        return Obj(HasCentroidStream=s["stream"] is not None, mz=s["mz"], intensity=s["intensity"])

    def centroider(scan):
        keep = scan.intensity >= scan.intensity.max() / 2
        return scan.mz[keep], scan.intensity[keep]

    monkeypatch.setattr(_scans, "_load_scan", load_scan)
    monkeypatch.setitem(_scans.CENTROIDERS, "thermo", centroider)
