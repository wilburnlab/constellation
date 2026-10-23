"""The EncyclopeDIA replay against the jar's own fits of real searches.

Every ``*.rt_fit.txt`` in the directory named by
``CONSTELLATION_RT_FIT_REFERENCE`` is refitted from its ``library`` and
``actual`` columns in row order (the order the jar fitted them in), and
the replay must give the sidecar's ``warpToactual``, ``delta`` and
``within95p`` on every row, to the bit. Nothing runs without the
variable. The lab's reference set is ``/mnt/e/261010_u2os`` (four files:
the gas-phase-fraction chromatogram-library build and the wide-window
run searched against it, first and final pass each; 205,000 rows).

The times are read as the file prints them, in minutes, with
``read_tsv`` rather than ``read_rt_fits``, whose conversion to seconds
would round them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from constellation.massspec.io.encyclopedia._sidecar import read_tsv
from constellation.massspec.rt.encyclopedia import fit_encyclopedia

_REFERENCE = os.environ.get("CONSTELLATION_RT_FIT_REFERENCE")
_FILES = sorted(Path(_REFERENCE).glob("*.rt_fit.txt")) if _REFERENCE else []

pytestmark = pytest.mark.skipif(not _FILES, reason="set CONSTELLATION_RT_FIT_REFERENCE (module docstring)")


def _column(table, name: str) -> torch.Tensor:
    return torch.tensor(table.column(name).to_pylist(), dtype=torch.float64)


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_the_replay_reproduces_the_jars_fit(path: Path) -> None:
    table = read_tsv(path, ["library", "actual", "warpToactual", "delta", "within95p"])
    x, y = _column(table, "library"), _column(table, "actual")
    fit = fit_encyclopedia(x, y)
    warp = fit.warp(x).to(torch.float32)
    expected = _column(table, "warpToactual").to(torch.float32)
    mismatched = int((warp != expected).sum())
    assert mismatched == 0, f"{mismatched} of {x.numel()} warps differ (first at row {int((warp != expected).nonzero()[0])})"
    assert torch.equal(fit.delta(x, y).to(torch.float32), _column(table, "delta").to(torch.float32))
    assert fit.within_band(x, y).tolist() == table.column("within95p").to_pylist()
