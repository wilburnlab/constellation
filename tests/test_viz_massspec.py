"""The mass-spec viz modality over HTTP: sessions, kernels and its own routes.

Converted acquisitions are built from the synthetic runs in
``_massspec_fixtures``; everything else is the real server.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from _massspec_fixtures import (  # noqa: E402
    dda_scans,
    dia_scans,
    with_scan_stats,
    write_bundle,
    write_bundle_manifest,
)
from constellation.viz.modalities.massspec.kernel import MassSpecTrackKernel  # noqa: E402
from constellation.viz.server.app import create_app  # noqa: E402
from constellation.viz.server.session import derive_source_id  # noqa: E402

WINDOWS = ((400.0, 402.0), (402.0, 404.0), (404.0, 406.0))


@pytest.fixture
def runs(tmp_path: Path) -> dict[str, tuple[Path, list[dict]]]:
    dia = with_scan_stats(dia_scans(windows=WINDOWS))
    dda = with_scan_stats(dda_scans())
    return {
        "dia": (write_bundle_manifest(write_bundle(tmp_path, "dia_run", dia)), dia),
        "dda": (write_bundle_manifest(write_bundle(tmp_path, "dda_run", dda)), dda),
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app({}))


def _open(client: TestClient, *bundles: Path, label: str | None = None) -> str:
    body = {"modality": "massspec", "sources": [{"path": str(b)} for b in bundles]}
    if label:
        body["label"] = label
    response = client.post("/api/sessions/open", json=body)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _arrow(response) -> pa.Table:
    assert response.status_code == 200, response.text
    return pa.ipc.RecordBatchStreamReader(io.BytesIO(response.content)).read_all()


def _key(bundle: Path) -> str:
    return derive_source_id(bundle.resolve(), "acquisition")


def _data(client: TestClient, sid: str, kind: str, bundle: Path, **params) -> pa.Table:
    binding = f"{kind}:{_key(bundle)}"
    return _arrow(
        client.get(f"/api/tracks/{kind}/data", params={"session": sid, "binding": binding, **params})
    )


# ── sessions ──────────────────────────────────────────────────────────────


def test_a_session_lists_one_binding_per_kernel_and_acquisition(client, runs) -> None:
    dia, dda = runs["dia"][0], runs["dda"][0]
    sid = _open(client, dia, dda, label="two runs")
    summary = client.get("/api/sessions").json()[0]
    assert summary["modality"] == "massspec" and summary["n_acquisitions"] == 2

    tracks = client.get("/api/tracks", params={"session": sid}).json()
    assert sorted((t["kind"], t["binding_id"]) for t in tracks) == sorted(
        (kind, f"{kind}:{_key(b)}") for kind in ("ms_chromatogram", "ms_spectrum") for b in (dia, dda)
    )

    meta = client.get(
        "/api/tracks/ms_spectrum/metadata",
        params={"session": sid, "binding": f"ms_spectrum:{_key(dia)}"},
    ).json()
    assert meta["acquisition_type"] == "DIA"
    assert meta["default_tolerance"] == {"1": [0.35, "Da"], "2": [0.35, "Da"]}


def test_the_same_run_attached_twice_is_one_acquisition(client, runs) -> None:
    dia = runs["dia"][0]
    sid = _open(client, dia, dia)
    assert len(client.get(f"/api/massspec/sessions/{sid}/acquisitions").json()) == 1


def test_removing_the_first_source_keeps_the_session(client, runs) -> None:
    dia, dda = runs["dia"][0], runs["dda"][0]
    sid = _open(client, dia, dda, label="pair")
    first = client.get(f"/api/sessions/{sid}/manifest").json()["sources"][0]["source_id"]
    response = client.delete(f"/api/sessions/{sid}/sources/{first}")
    assert response.status_code == 200, response.text
    assert [s["session_id"] for s in client.get("/api/sessions").json()] == [sid]
    left = client.get(f"/api/massspec/sessions/{sid}/acquisitions").json()
    assert [a["key"] for a in left] == [_key(dda)]


def test_inspect_source_describes_a_run_and_refuses_anything_else(client, runs, tmp_path) -> None:
    response = client.post(
        "/api/sessions/inspect-source",
        json={"modality": "massspec", "path": str(runs["dda"][0])},
    )
    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "acquisition"
    assert response.json()["acquisition_type"] == "DDA"

    other = tmp_path / "not_a_run"
    other.mkdir()
    refused = client.post(
        "/api/sessions/inspect-source", json={"modality": "massspec", "path": str(other)}
    )
    assert refused.status_code == 400


# ── chromatograms ─────────────────────────────────────────────────────────


def test_a_tic_by_level_and_window(client, runs) -> None:
    bundle, scans = runs["dia"]
    sid = _open(client, bundle)
    ms1 = _data(client, sid, "ms_chromatogram", bundle, level=1)
    expected = [s for s in scans if s["level"] == 1]
    assert ms1.column("scan").to_pylist() == [s["scan"] for s in expected]
    assert ms1.column("intensity").to_pylist() == pytest.approx([s["tic"] for s in expected])
    assert not any(ms1.column("decimated").to_pylist())

    one_window = _data(client, sid, "ms_chromatogram", bundle, level=2, window=2)
    assert one_window.num_rows == 12
    lows = {round(s["isolation_lower"]) for s in scans if s["scan"] in one_window.column("scan").to_pylist()}
    assert len(lows) == 1

    bpc = _data(client, sid, "ms_chromatogram", bundle, level=1, trace="bpc")
    assert bpc.column("mz").to_pylist() == [s["base_peak_mz"] for s in expected]


def test_dda_scan_events_carry_their_precursor(client, runs) -> None:
    bundle, scans = runs["dda"]
    sid = _open(client, bundle)
    events = _data(client, sid, "ms_chromatogram", bundle, level=2, trace="scans")
    expected = [s for s in scans if s["level"] == 2]
    assert events.column("mz").to_pylist() == [s["precursor_mz"] for s in expected]


def test_a_long_trace_is_reduced_to_the_width_asked_for(client, runs, monkeypatch) -> None:
    bundle, _ = runs["dia"]
    monkeypatch.setattr(MassSpecTrackKernel, "decimate_above", 10)
    sid = _open(client, bundle)
    reduced = _data(client, sid, "ms_chromatogram", bundle, level=2, viewport_px=2)
    assert all(reduced.column("decimated").to_pylist())
    assert reduced.num_rows <= 8


@pytest.mark.parametrize(
    ("params", "status"),
    [({"level": 0}, 422), ({"trace": "xic"}, 422), ({"rt_start": 5.0}, 400),
     ({"rt_start": 5.0, "rt_end": 1.0}, 400)],  # fmt: skip
)
def test_bad_chromatogram_queries_are_refused(client, runs, params, status) -> None:
    bundle, _ = runs["dia"]
    sid = _open(client, bundle)
    response = client.get(
        "/api/tracks/ms_chromatogram/data",
        params={"session": sid, "binding": f"ms_chromatogram:{_key(bundle)}", **params},
    )
    assert response.status_code == status, response.text


# ── spectra ───────────────────────────────────────────────────────────────


def test_a_spectrum_and_an_m_z_window_of_it(client, runs) -> None:
    bundle, scans = runs["dia"]
    sid = _open(client, bundle)
    scan = scans[5]
    peaks = _data(client, sid, "ms_spectrum", bundle, scan=scan["scan"])
    assert peaks.column("mz").to_pylist() == sorted(scan["mz"].tolist())

    lo, hi = float(np.min(scan["mz"])), float(np.median(scan["mz"]))
    window = _data(client, sid, "ms_spectrum", bundle, scan=scan["scan"], mz_start=lo, mz_end=hi)
    assert window.num_rows and all(lo <= m <= hi for m in window.column("mz").to_pylist())


# ── the modality's own routes ─────────────────────────────────────────────


def test_acquisitions_windows_and_scans(client, runs) -> None:
    bundle, scans = runs["dia"]
    sid = _open(client, bundle)
    base = f"/api/massspec/sessions/{sid}/acquisitions"
    (described,) = client.get(base).json()
    assert described["key"] == _key(bundle) and described["levels"] == [1, 2]
    assert described["scans_per_level"] == {"1": 12, "2": 36}

    windows = client.get(f"{base}/{_key(bundle)}/windows").json()
    assert [w["window"] for w in windows] == [0, 1, 2]

    table = _arrow(client.get(f"{base}/{_key(bundle)}/scans"))
    assert table.column("scan").to_pylist() == [s["scan"] for s in scans]
    assert {"window", "parent_scan", "rt"} <= set(table.column_names)
    ms2 = _arrow(client.get(f"{base}/{_key(bundle)}/scans", params={"level": 2}))
    assert set(ms2.column("level").to_pylist()) == {2}

    header = client.get(f"{base}/{_key(bundle)}/scans/{scans[1]['scan']}").json()
    assert header["filter_string"].startswith("ITMS") and header["level"] == 2
    assert client.get(f"{base}/{_key(bundle)}/scans/99999").status_code == 404
    assert client.get(f"{base}/no-such-key/windows").status_code == 404


def test_mass_spec_routes_refuse_another_modalitys_session(client, tmp_path, monkeypatch) -> None:
    from _viz_fixtures import build_viz_session

    genome = build_viz_session(tmp_path / "genome", monkeypatch)
    app_client = TestClient(create_app(genome))
    response = app_client.get(f"/api/massspec/sessions/{genome.session_id}/acquisitions")
    assert response.status_code == 404
    assert client.get("/api/massspec/sessions/nope/acquisitions").status_code == 404


# ── the CLI ───────────────────────────────────────────────────────────────


def test_the_cli_needs_something_to_open(capsys) -> None:
    from constellation.cli.__main__ import main

    assert main(["viz", "massspec", "--no-browser"]) == 2
    assert "--acquisition" in capsys.readouterr().err


def test_a_trace_the_bundle_cannot_give_is_refused_not_cut_short(client, tmp_path) -> None:
    # A bundle imported before base peaks were derived has none.
    bare = write_bundle_manifest(write_bundle(tmp_path, "bare_run", dia_scans(windows=WINDOWS)))
    sid = _open(client, bare)
    response = client.get(
        "/api/tracks/ms_chromatogram/data",
        params={"session": sid, "binding": f"ms_chromatogram:{_key(bare)}", "level": 1, "trace": "bpc"},
    )
    assert response.status_code == 400
    assert "base peaks" in response.json()["detail"]
