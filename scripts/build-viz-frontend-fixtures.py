#!/usr/bin/env python
"""Regenerate the Arrow fixtures used by the viz frontend's renderer tests.

Runs the real track kernels against a small synthetic session and writes
their output to
``constellation/viz/frontend/src/modalities/genome/__fixtures__/data/``.
The builder lives in ``tests/_viz_frontend_fixtures.py`` so the drift
guard (``tests/test_viz_frontend_fixtures.py``) and this script share one
definition of the fixture data.

    python scripts/build-viz-frontend-fixtures.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _viz_frontend_fixtures import FIXTURE_DIR, build_fixture_payloads  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        payloads = build_fixture_payloads(Path(tmp), mp)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FIXTURE_DIR.iterdir():
        if stale.is_file() and stale.name not in payloads:
            stale.unlink()
            print(f"removed {stale.name}")
    for name, data in sorted(payloads.items()):
        (FIXTURE_DIR / name).write_bytes(data)
        print(f"wrote {name} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
