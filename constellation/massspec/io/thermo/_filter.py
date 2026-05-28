"""Thermo scan-filter-string parser.

The Thermo filter string is a space-separated token stream whose
content varies with acquisition mode; this module decodes the common
tokens into a plain dict. Each regex is applied opportunistically —
missing tokens are simply left absent from the result. Stateless,
pure Python, no .NET / DLL dependency.

Recognised tokens:

- ``analyzer``   ITMS / TQMS / SQMS / TOFMS / FTMS / SECTOR
- ``polarity``   ``+`` / ``-``
- ``point_type`` ``C`` (centroid) or ``P`` (profile)
- ``ionization`` EI / CI / FAB / APCI / ESI / NSI / TSP / FD / MALDI / GD
- ``ms_level``   int (``ms`` → 1, ``ms2`` → 2, ...)
- ``faims_cv``   float (compensation voltage)
- ``activations`` list of ``(isolation_mz, activation_type, energy | None)``
                   tuples — one entry per ``@`` cluster, so EThcD / sa /
                   supplemental-activation acquisitions produce ≥2 entries
                   all carrying the leading cluster's isolation target
- ``scan_window`` ``(lo, hi)`` tuple in m/z

Patterns lifted from ms_deisotope's ``_thermo_helper.py``; ported here as
a clean rewrite so Constellation never depends on the upstream
implementation.
"""

from __future__ import annotations

import re
from typing import Any


_ANALYZER_RE = re.compile(r"(?P<analyzer>ITMS|TQMS|SQMS|TOFMS|FTMS|SECTOR)")
_POLARITY_RE = re.compile(r"[^\w](?P<polarity>[\+\-])[^\w]")
_POINT_TYPE_RE = re.compile(r"[^\w](?P<point_type>[CPcp])[^\w]")
_IONIZATION_RE = re.compile(r"(?P<ionization>EI|CI|FAB|APCI|ESI|NSI|TSP|FD|MALDI|GD)")
_MS_LEVEL_RE = re.compile(r"\bms(?P<level>\d*)\b")
_FAIMS_CV_RE = re.compile(r"\bcv=(?P<cv>-?\d+(?:\.\d+)?)\b", re.IGNORECASE)
# Leading "mz@activation_energy" cluster (first activation; carries isolation_mz).
_ACTIVATION_LEAD_RE = re.compile(
    r"(?P<isolation_mz>\d+\.\d*)@(?P<activation>[a-zA-Z]+)(?P<energy>\d*\.?\d*)"
)
# Follow-on "@activation_energy" clusters (EThcD / sa / supplemental activation).
_ACTIVATION_FOLLOW_RE = re.compile(r"@(?P<activation>[a-zA-Z]+)(?P<energy>\d*\.?\d*)")
# Scan window: "[400.0-1600.0]"
_SCAN_WINDOW_RE = re.compile(r"\[(?P<lo>[0-9.]+)-(?P<hi>[0-9.]+)\]")


def parse_filter_string(filter_str: str) -> dict[str, Any]:
    """Decode a Thermo scan filter string into a metadata dict.

    Returns a dict with keys present only when the corresponding token
    was found. Never raises — malformed filters just produce sparser
    dicts, including ``{}`` on an empty input.
    """
    if not filter_str:
        return {}
    out: dict[str, Any] = {}

    m = _ANALYZER_RE.search(filter_str)
    if m:
        out["analyzer"] = m.group("analyzer")

    m = _POLARITY_RE.search(filter_str)
    if m:
        out["polarity"] = m.group("polarity")

    m = _POINT_TYPE_RE.search(filter_str)
    if m:
        out["point_type"] = m.group("point_type").upper()

    m = _IONIZATION_RE.search(filter_str)
    if m:
        out["ionization"] = m.group("ionization")

    m = _MS_LEVEL_RE.search(filter_str)
    if m:
        lvl_raw = m.group("level")
        out["ms_level"] = int(lvl_raw) if lvl_raw else 1

    m = _FAIMS_CV_RE.search(filter_str)
    if m:
        out["faims_cv"] = float(m.group("cv"))

    activations: list[tuple[float, str, float | None]] = []
    lead = _ACTIVATION_LEAD_RE.search(filter_str)
    if lead:
        leading_mz = float(lead.group("isolation_mz"))
        activations.append(
            (
                leading_mz,
                lead.group("activation").lower(),
                float(lead.group("energy")) if lead.group("energy") else None,
            )
        )
        for fm in _ACTIVATION_FOLLOW_RE.finditer(filter_str, lead.end()):
            activations.append(
                (
                    leading_mz,  # all follow-ons share the leading isolation target
                    fm.group("activation").lower(),
                    float(fm.group("energy")) if fm.group("energy") else None,
                )
            )
    if activations:
        out["activations"] = activations

    m = _SCAN_WINDOW_RE.search(filter_str)
    if m:
        out["scan_window"] = (float(m.group("lo")), float(m.group("hi")))

    return out


__all__ = ["parse_filter_string"]
