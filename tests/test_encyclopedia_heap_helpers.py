"""Unit tests for the EncyclopeDIA wrapper-layer heap-sizing helpers.

Covers ``available_memory_gib`` (system probe) and
``default_heap_for_system`` (string-formatted ``-Xmx`` value derived
from current available memory).
"""

from __future__ import annotations

import pytest

from constellation.massspec.search.encyclopedia._common import (
    available_memory_gib,
    default_heap_for_input,
    default_heap_for_system,
)


def test_available_memory_gib_returns_float_on_linux() -> None:
    """On the CI / dev box (Linux / WSL), ``/proc/meminfo`` exists and
    the helper returns a sensible positive float."""
    avail = available_memory_gib()
    # Skip rather than fail on platforms without /proc and without
    # SC_AVPHYS_PAGES (Windows). The helper documents that it returns
    # None there; only the Linux path needs a positive assertion.
    if avail is None:
        pytest.skip("system memory probe returned None — non-POSIX platform")
    assert isinstance(avail, float)
    assert avail > 0
    # Sanity: development boxes have between 1 GiB and 4 TiB of RAM.
    assert 0.5 < avail < 4096


def test_default_heap_for_system_returns_g_suffixed_string() -> None:
    """The returned heap string is ``"<N>g"`` with ``N`` an integer."""
    heap = default_heap_for_system()
    assert isinstance(heap, str)
    assert heap.endswith("g")
    n = int(heap[:-1])
    assert n >= 4  # floor


def test_default_heap_for_system_respects_floor() -> None:
    """When fraction × available is below ``min_gib``, the floor wins."""
    heap = default_heap_for_system(fraction=0.0, min_gib=4, max_gib=96)
    assert heap == "4g"


def test_default_heap_for_system_respects_ceiling() -> None:
    """When fraction × available is above ``max_gib``, the ceiling wins."""
    heap = default_heap_for_system(fraction=1000.0, min_gib=4, max_gib=64)
    assert heap == "64g"


def test_default_heap_for_system_custom_bounds() -> None:
    """Custom floor / ceiling propagate."""
    heap = default_heap_for_system(fraction=0.0, min_gib=16)
    assert heap == "16g"
    heap = default_heap_for_system(fraction=1000.0, max_gib=128)
    assert heap == "128g"


def test_default_heap_for_system_fallback_when_probe_fails(monkeypatch) -> None:
    """When the system probe returns None (e.g. Windows), the helper
    falls back to the wrappers' static default ``"12g"`` so callers
    that opt into auto-sizing don't crash on unsupported platforms."""
    import constellation.massspec.search.encyclopedia._common as common

    monkeypatch.setattr(common, "available_memory_gib", lambda: None)
    assert common.default_heap_for_system() == "12g"


# ──────────────────────────────────────────────────────────────────────
# Pre-existing static heuristic — keep its behaviour pinned alongside
# the new dynamic helper.
# ──────────────────────────────────────────────────────────────────────


def test_default_heap_for_input_thresholds() -> None:
    """Input-size-based fallback bands."""
    assert default_heap_for_input(0) == "8g"
    assert default_heap_for_input(500 * 1024**2) == "8g"  # < 1 GiB
    assert default_heap_for_input(3 * 1024**3) == "12g"  # < 5 GiB
    assert default_heap_for_input(10 * 1024**3) == "24g"  # < 20 GiB
    assert default_heap_for_input(50 * 1024**3) == "48g"  # ≥ 20 GiB
