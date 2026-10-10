"""Drift guard for the frontend's Arrow fixtures.

``constellation/viz/frontend/src/__fixtures__/genome/`` holds kernel
output that the vitest renderer snapshots are drawn from. This test
rebuilds those payloads from the live kernels and compares them with the
committed files, so a kernel change that alters the wire shape cannot
leave the frontend tests passing against a stale format.

On failure, run ``python scripts/build-viz-frontend-fixtures.py`` and
commit the result together with the renderer snapshot updates it causes.
"""

from __future__ import annotations

import json
from pathlib import Path

from _viz_frontend_fixtures import FIXTURE_DIR, build_fixture_payloads, decode


_REGEN = "run `python scripts/build-viz-frontend-fixtures.py` and commit the result"


def test_frontend_fixtures_match_kernels(tmp_path: Path, monkeypatch) -> None:
    payloads = build_fixture_payloads(tmp_path, monkeypatch)

    committed = {p.name for p in FIXTURE_DIR.iterdir() if p.is_file()}
    assert committed == set(payloads), f"fixture file set drifted — {_REGEN}"

    for name, data in payloads.items():
        on_disk = (FIXTURE_DIR / name).read_bytes()
        if name.endswith(".arrow"):
            want, got = decode(on_disk), decode(data)
            assert got.schema.equals(want.schema, check_metadata=True), (
                f"{name}: wire schema drifted — {_REGEN}"
            )
            assert got.equals(want), f"{name}: kernel output drifted — {_REGEN}"
        else:
            assert json.loads(data) == json.loads(on_disk), (
                f"{name}: kernel metadata drifted — {_REGEN}"
            )
