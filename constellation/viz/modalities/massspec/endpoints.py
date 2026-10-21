"""Mass-spec-only endpoints.

The session and track endpoints in ``viz.server.endpoints`` are
modality-neutral. These describe a mass-spec session's acquisitions and
serve what a browser navigates by:

- ``GET /api/massspec/sessions/{sid}/acquisitions``                 → each run, described
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/windows``   → its DIA isolation windows
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/scans``     → its scan table (Arrow)
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/scans/{n}`` → one scan's full header
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/identifications`` → what its searches found (Arrow)
- ``GET /api/massspec/sessions/{sid}/acquisitions/{key}/precursor``      → one identification, with what to trace
- ``GET /api/massspec/sessions/{sid}/resolve?q=PEPTIDE/2``               → a peptide typed by hand

All under ``/api/massspec/``: the genome modality's routes are mounted
first and own paths such as ``/api/sessions/{id}/search``.

Imported only through the modality's ``routers()`` factory — it needs
fastapi, which ``import constellation.viz`` must not.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
from fastapi import APIRouter, HTTPException, Query, Request

from constellation.viz.modalities.massspec.cache import identifications_for, scan_table_for
from constellation.viz.modalities.massspec.session import MassSpecSession, MsAcquisition
from constellation.viz.server.arrow_stream import batches_to_response

router = APIRouter(prefix="/api/massspec/sessions/{session_id}", tags=["massspec"])

#: Scan-table columns sent to the client for navigation.
SCAN_ROUTE_COLUMNS = (
    "scan",
    "level",
    "rt",
    "window",
    "parent_scan",
    "precursor_mz",
    "precursor_charge",
    "isolation_lower",
    "isolation_upper",
    "tic",
)


def _session(request: Request, session_id: str) -> MassSpecSession:
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    if getattr(session, "modality", None) != MassSpecSession.modality:
        raise HTTPException(404, f"session {session_id} is not a mass-spec session")
    return session


def _acquisition(request: Request, session_id: str, key: str) -> MsAcquisition:
    try:
        return _session(request, session_id).acquisition(key)
    except KeyError:
        raise HTTPException(404, f"session {session_id} has no acquisition {key!r}") from None


def _plain(value: Any) -> Any:
    """A JSON-safe copy of one Arrow row value (NaN → null, maps → dicts)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list) and value and isinstance(value[0], tuple):
        return {k: _plain(v) for k, v in value}
    return value


@router.get("/acquisitions")
def list_acquisitions(request: Request, session_id: str) -> list[dict[str, Any]]:
    return [a.describe() for a in _session(request, session_id).acquisitions]


@router.get("/acquisitions/{key}/windows")
def list_windows(request: Request, session_id: str, key: str) -> list[dict[str, Any]]:
    from constellation.massspec.scans import isolation_window_table

    acquisition = _acquisition(request, session_id, key)
    return isolation_window_table(scan_table_for(acquisition.bundle)).to_pylist()


@router.get("/acquisitions/{key}/scans")
def scan_rows(
    request: Request,
    session_id: str,
    key: str,
    level: int | None = Query(None, ge=1, le=10),
):
    """The scan table a client steps through and finds the nearest scan in."""
    scans = scan_table_for(_acquisition(request, session_id, key).bundle)
    if level is not None:
        scans = scans.filter(pc.equal(scans.column("level"), level))
    table = scans.select([c for c in SCAN_ROUTE_COLUMNS if c in scans.column_names])
    return batches_to_response(table.schema, table.to_batches())


@router.get("/acquisitions/{key}/scans/{scan}")
def scan_header(request: Request, session_id: str, key: str, scan: int) -> dict[str, Any]:
    """Everything the bundle recorded about one scan (filter string,
    analyzer, activation, injection time, trailer values, ...)."""
    acquisition = _acquisition(request, session_id, key)
    dataset = pa_ds.dataset(acquisition.bundle / "scan_metadata.parquet")
    rows = dataset.to_table(filter=pa_ds.field("scan") == scan).to_pylist()
    if not rows:
        raise HTTPException(404, f"acquisition {key!r} has no scan {scan}")
    return {k: _plain(v) for k, v in rows[0].items() if v is not None}


@router.get("/acquisitions/{key}/identifications")
def acquisition_identifications(
    request: Request,
    session_id: str,
    key: str,
    max_q: float = Query(0.01, ge=0.0, le=1.0),
):
    """What the searches of this run found, at q-value ``max_q`` or better,
    one Arrow table over every linked experiment (an ``experiment`` column
    says which)."""
    acquisition = _acquisition(request, session_id, key)
    tables = []
    for link in acquisition.links:
        found = identifications_for(Path(link["experiment"]), int(link["acquisition_id"]), max_q)
        tables.append(
            found.append_column("experiment", pa.array([link["experiment_source_id"]] * found.num_rows, pa.string()))
        )
    table = pa.concat_tables(tables) if tables else pa.table({"experiment": pa.array([], pa.string())})
    return batches_to_response(table.schema, table.to_batches())


def _plain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _plain(v) for k, v in r.items()} for r in rows]


@router.get("/acquisitions/{key}/precursor")
def precursor(
    request: Request,
    session_id: str,
    key: str,
    experiment: str,
    unit: str,
    precursor_id: int,
    top_n: int = Query(8, ge=1, le=32),
    n_isotopes: int = Query(3, ge=1, le=6),
) -> dict[str, Any]:
    """One identification of this run and what a browser traces for it:
    where it was measured (window, apex scans), the precursor's isotopes
    (MS1) and its most intense stored fragments (MS2), its ion ladder, the
    engine's quantified ions and median chromatogram, and its retention
    time in the searched library mapped onto this run."""
    import numpy as np

    from constellation.massspec.results import (
        precursor_detail,
        rt_mapping,
        searched_library_spectrum,
        trace_targets,
    )
    from constellation.massspec.scans import locate_precursor, parent_scan
    from constellation.massspec.spectra.annotate import default_fragment_charge, ion_ladder, precursor_isotopes

    acquisition = _acquisition(request, session_id, key)
    try:
        link = acquisition.link(experiment)
    except KeyError:
        raise HTTPException(404, f"acquisition {key!r} was not searched in experiment {experiment!r}") from None
    if unit not in link["units"]:
        raise HTTPException(404, f"search {unit!r} did not search acquisition {key!r}")
    root, aid = Path(link["experiment"]), int(link["acquisition_id"])
    detail = precursor_detail(root, precursor_id, unit=unit, acquisition_id=aid)
    found = detail["identification"]
    if found is None:
        raise HTTPException(404, f"search {unit!r} has no identification {precursor_id}")
    modseq, charge = found["modified_sequence"], int(found["charge"])
    scans = scan_table_for(acquisition.bundle)
    located = locate_precursor(scans, precursor_mz=found["precursor_mz"], rt=found["rt"])
    if found.get("scan") is not None:  # a spectrum-centric match names its scan
        located["ms_scan"] = int(found["scan"])
        located["parent_scan"] = parent_scan(scans, int(found["scan"]))
    library_rt = predicted_rt = None
    searched = searched_library_spectrum(root, precursor_id, unit=unit)
    if searched is not None:
        library_rt = float(searched.schema.metadata[b"x.massspec.rt_seconds"])
        mapping = rt_mapping(root, unit=unit, acquisition_id=aid)
        if mapping.num_rows > 1:
            predicted_rt = float(
                np.interp(library_rt, mapping.column("library_rt").to_numpy(), mapping.column("observed_rt").to_numpy())
            )
    return {
        "identification": {k: _plain(v) for k, v in found.items()},
        "experiment": experiment,
        "unit": unit,
        "located": {k: list(v) if isinstance(v, tuple) else v for k, v in located.items()},
        "ms1_targets": precursor_isotopes(modseq, charge=charge, n_isotopes=n_isotopes).to_pylist(),
        "ms2_targets": _plain_rows(
            trace_targets(root, precursor_id, unit=unit, acquisition_id=aid, top_n=top_n).to_pylist()
        ),
        "ladder": ion_ladder(modseq, max_fragment_charge=default_fragment_charge(charge)).to_pylist(),
        "quantified_ions": detail["quantified_ions"],
        "chromatogram": detail["chromatogram"],
        "fragment_tolerance": detail["fragment_tolerance"],
        "precursor_tolerance": detail["precursor_tolerance"],
        "library_rt": library_rt,
        "predicted_rt": predicted_rt,
    }


_CHARGE_SUFFIX = re.compile(r"^(?P<peptide>.*?)(?:/(?P<charge>\d+))?$")


@router.get("/resolve")
def resolve(
    request: Request,
    session_id: str,
    q: str,
    charge: int | None = Query(None, ge=1, le=10),
    top_n: int = Query(8, ge=1, le=32),
) -> dict[str, Any]:
    """A peptide typed by hand (ProForma 2.0, optionally ``/charge``): its
    precursor m/z at its charge (or at 1-4 when none is given), and for the
    first of those charges what to trace and its ion ladder."""
    from constellation.core.sequence.proforma import format_proforma, parse_proforma
    from constellation.massspec.library.ids import precursor_id
    from constellation.massspec.peptide.mz import precursor_mz
    from constellation.massspec.spectra.annotate import (
        default_fragment_charge,
        ion_ladder,
        precursor_isotopes,
        theoretical_fragment_targets,
    )

    _session(request, session_id)
    match = _CHARGE_SUFFIX.match(q.strip())
    text = match.group("peptide") if match else q.strip()
    given = charge or (int(match.group("charge")) if match and match.group("charge") else None)
    try:
        peptide = parse_proforma(text)
        modseq = format_proforma(peptide)
        charges = [given] if given else [1, 2, 3, 4]
        options = [
            {"charge": z, "precursor_mz": precursor_mz(peptide, charge=z), "precursor_id": precursor_id(peptide, z)}
            for z in charges
        ]
    except (ValueError, KeyError, NotImplementedError) as exc:
        raise HTTPException(400, f"not a peptide: {exc}") from exc
    z = charges[0] if given else 2
    return {
        "modified_sequence": modseq,
        "sequence": peptide.sequence,
        "charges": options,
        "charge": z,
        "ms1_targets": precursor_isotopes(modseq, charge=z).to_pylist(),
        "ms2_targets": [
            {"mz": r["mz"], "label": r["annotation"], "ion_type": r["ion_type"], "ion_number": r["ion_number"],
             "fragment_charge": r["fragment_charge"], "intensity": None, "quantified": None}
            for r in theoretical_fragment_targets(modseq, top_n=top_n).to_pylist()
        ],
        "ladder": ion_ladder(modseq, max_fragment_charge=default_fragment_charge(z)).to_pylist(),
    }


__all__ = ["SCAN_ROUTE_COLUMNS", "router"]
