"""A second modality plugs into the viz server without touching it.

This is the acceptance test for "genome is one modality". Everything a
modality supplies is defined *in this file* — a session class, a kernel
whose query is a pair of floats rather than a genomic locus, request
models, a router — and registered through the same public hooks the
genome modality uses. The test then drives it over HTTP: open, list
tracks, fetch data, mutate sources. If that needs an edit anywhere under
``constellation/viz/server`` or ``constellation/viz/tracks``, the
contracts are still genome-shaped.

The toy pieces are registered with ``monkeypatch.setitem`` on the two
registry dicts, so they exist only for the duration of each test.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import pyarrow as pa
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import APIRouter  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from _viz_fixtures import build_viz_session  # noqa: E402
from constellation.viz import modalities  # noqa: E402
from constellation.viz.modalities import (  # noqa: E402
    Modality,
    get_modality,
    register_modality,
    registered_modalities,
)
from constellation.viz.server.app import create_app  # noqa: E402
from constellation.viz.server.endpoints.tracks import _query_plan  # noqa: E402
from constellation.viz.server.session import (  # noqa: E402
    derive_session_id,
    derive_source_id,
)
from constellation.viz.tracks import base as tracks_base  # noqa: E402
from constellation.viz.tracks.base import (  # noqa: E402
    ThresholdDecision,
    TrackBinding,
    TrackKernel,
    TrackQuery,
    register_track,
)


# ----------------------------------------------------------------------
# The toy modality — everything a real one would supply
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToySource:
    path: Path
    kind: str
    label: str

    @property
    def source_id(self) -> str:
        return derive_source_id(self.path, self.kind)


@dataclass(frozen=True, slots=True)
class ToySession:
    """Deliberately shares nothing with ``GenomeSession``: no reference,
    no contigs, no slots."""

    modality: ClassVar[str] = "toy"

    session_id: str
    label: str
    title: str
    sources: tuple[ToySource, ...]
    warnings: tuple[str, ...] = ()
    saved_as: str | None = None

    @classmethod
    def open(cls, *, title: str, sources: Iterable[dict[str, Any]]) -> "ToySession":
        built = []
        for entry in sources:
            path = Path(str(entry["path"]))
            if not path.is_dir():
                raise ValueError(f"toy source is not a directory: {path}")
            built.append(
                ToySource(
                    path=path,
                    kind=str(entry.get("kind") or "trace"),
                    label=str(entry.get("label") or path.name),
                )
            )
        return cls(
            session_id=derive_session_id("toy", title),
            label=title,
            title=title,
            sources=tuple(built),
        )

    def with_sources(self, sources: Iterable[dict[str, Any]]) -> "ToySession":
        return ToySession.open(title=self.title, sources=sources)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "modality": self.modality,
            "label": self.label,
            "n_sources": len(self.sources),
            "warnings": list(self.warnings),
            "saved_as": self.saved_as,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "title": self.title,
            "sources": [
                {"source_id": s.source_id, "path": str(s.path), "kind": s.kind, "label": s.label}
                for s in self.sources
            ],
        }


@dataclass(frozen=True, kw_only=True)
class ToyQuery(TrackQuery):
    """A float window plus a repeated parameter and an enumerated one —
    none of which the old locus-shaped query could carry."""

    x0: float
    x1: float = field(metadata={"le": 1000.0})
    tags: tuple[str, ...] = ()
    scale: Literal["linear", "log"] = "linear"

    def check(self) -> None:
        super().check()
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0")


TOY_SCHEMA = pa.schema(
    [pa.field("x", pa.float64()), pa.field("y", pa.float64()), pa.field("tag", pa.string())]
)


class ToyTraceKernel(TrackKernel):
    kind = "toy_trace"
    modality = "toy"
    schema = TOY_SCHEMA
    query_model = ToyQuery

    def discover(self, session: ToySession) -> list[TrackBinding]:  # type: ignore[override]
        # Reads an attribute only a ToySession has: handed any other
        # session this would raise, which is what the modality guard
        # exists to prevent.
        prefix = session.title
        return [
            TrackBinding(
                session_id=session.session_id,
                kind=self.kind,
                binding_id=f"toy_trace-{idx}",
                label=f"{prefix} / {src.label}",
                paths={"dir": src.path},
                config={"source_id": src.source_id},
            )
            for idx, src in enumerate(session.sources)
        ]

    def metadata(self, binding: TrackBinding) -> dict[str, Any]:
        return {"kind": self.kind, "binding_id": binding.binding_id, "label": binding.label}

    def threshold(self, binding: TrackBinding, query: ToyQuery) -> ThresholdDecision:  # type: ignore[override]
        return query.force or ThresholdDecision.VECTOR

    def fetch(  # type: ignore[override]
        self, binding: TrackBinding, query: ToyQuery, mode: ThresholdDecision
    ) -> Iterator[pa.RecordBatch]:
        xs = [x / 2 for x in range(0, 20) if query.x0 <= x / 2 < query.x1]
        tag = ",".join(query.tags)
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(xs, pa.float64()),
                pa.array([x * x for x in xs], pa.float64()),
                pa.array([tag] * len(xs), pa.string()),
            ],
            schema=TOY_SCHEMA,
        )

    def response_headers(self, query: ToyQuery, mode: ThresholdDecision) -> dict[str, str]:  # type: ignore[override]
        return {"X-Toy-Scale": query.scale}


@dataclass(frozen=True, kw_only=True)
class ToyOpenRequest:
    title: str
    paths: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ToySaveRequest:
    label: str
    sources: list[dict[str, Any]]


def _toy_routers() -> list[APIRouter]:
    router = APIRouter()

    @router.get("/api/toy/ping")
    def _ping() -> dict[str, str]:
        return {"modality": "toy"}

    return [router]


def _toy_inspect(path: Path) -> dict[str, Any]:
    if not (path / "trace.txt").exists():
        raise ValueError(f"{path} has no trace.txt")
    return {"path": str(path), "kind": "trace"}


TOY = Modality(
    name="toy",
    open_request=ToyOpenRequest,
    open_session=lambda req: ToySession.open(
        title=req.title, sources=[{"path": p} for p in req.paths]
    ),
    inspect_source=_toy_inspect,
    session_from_saved=lambda saved: ToySession.open(title=saved.label, sources=saved.sources),
    save_request=ToySaveRequest,
    routers=_toy_routers,
)


@pytest.fixture
def toy(monkeypatch) -> None:
    """Register the toy modality and its kernel for one test."""
    monkeypatch.setitem(modalities._REGISTRY, TOY.name, TOY)
    monkeypatch.setitem(tracks_base._REGISTRY, ToyTraceKernel.kind, ToyTraceKernel())


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "toy-run"
    d.mkdir()
    (d / "trace.txt").write_text("1 2 3\n")
    return d


def _open(client: TestClient, source_dir: Path, title: str = "demo") -> str:
    response = client.post(
        "/api/sessions/open",
        json={"modality": "toy", "title": title, "paths": [str(source_dir)]},
    )
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _read(response) -> pa.Table:
    return pa.ipc.RecordBatchStreamReader(io.BytesIO(response.content)).read_all()


# ----------------------------------------------------------------------
# Open → list → fetch
# ----------------------------------------------------------------------


def test_toy_session_opens_lists_and_streams(toy, source_dir: Path) -> None:
    client = TestClient(create_app({}))

    opened = client.post(
        "/api/sessions/open",
        json={"modality": "toy", "title": "demo", "paths": [str(source_dir)]},
    )
    assert opened.status_code == 201
    summary = opened.json()
    assert summary["modality"] == "toy"
    sid = summary["session_id"]
    assert client.get("/api/sessions").json() == [summary]
    assert client.get(f"/api/sessions/{sid}/manifest").json()["title"] == "demo"

    # Only the toy kernel answers for a toy session.
    tracks = client.get("/api/tracks", params={"session": sid}).json()
    assert [(t["kind"], t["binding_id"], t["label"]) for t in tracks] == [
        ("toy_trace", "toy_trace-0", "demo / toy-run")
    ]
    assert tracks[0]["source_id"] == derive_source_id(source_dir, "trace")

    meta = client.get(
        "/api/tracks/toy_trace/metadata", params={"session": sid, "binding": "toy_trace-0"}
    )
    assert meta.json()["label"] == "demo / toy-run"

    # A fractional window, a repeated parameter and an enumerated one.
    data = client.get(
        "/api/tracks/toy_trace/data",
        params=[
            ("session", sid),
            ("binding", "toy_trace-0"),
            ("x0", "0.5"),
            ("x1", "2.25"),
            ("tags", "a"),
            ("tags", "b"),
            ("scale", "log"),
        ],
    )
    assert data.status_code == 200
    assert data.headers["x-track-mode"] == "vector"
    assert data.headers["x-track-kind"] == "toy_trace"
    assert data.headers["x-toy-scale"] == "log"
    assert "x-track-view" not in data.headers  # that header is the genome kernels'
    table = _read(data)
    assert table.schema == TOY_SCHEMA
    assert table.column("x").to_pylist() == [0.5, 1.0, 1.5, 2.0]
    assert set(table.column("tag").to_pylist()) == {"a,b"}


def test_toy_query_is_validated_by_its_own_model(toy, source_dir: Path) -> None:
    client = TestClient(create_app({}))
    sid = _open(client, source_dir)
    base = {"session": sid, "binding": "toy_trace-0", "x0": 0, "x1": 5}
    url = "/api/tracks/toy_trace/data"

    assert client.get(url, params=base).status_code == 200

    inverted = client.get(url, params={**base, "x0": 5, "x1": 1})
    assert inverted.status_code == 400
    assert inverted.json()["detail"] == "x1 must be greater than x0"

    for bad, loc in (
        ({"x0": "left"}, "x0"),
        ({"x1": 5000}, "x1"),
        ({"scale": "sqrt"}, "scale"),
        ({"viewport_px": 0}, "viewport_px"),
    ):
        response = client.get(url, params={**base, **bad})
        assert response.status_code == 422, bad
        assert response.json()["detail"][0]["loc"] == ["query", loc]

    missing = client.get(url, params={"session": sid, "binding": "toy_trace-0", "x1": 5})
    assert missing.status_code == 422
    assert missing.json()["detail"][0]["loc"] == ["query", "x0"]

    # Genome parameters mean nothing to this kernel and are ignored.
    assert client.get(url, params={**base, "contig": "chr1", "min_mapq": -3}).status_code == 200


def test_open_body_is_validated_by_the_modality(toy, source_dir: Path, tmp_path: Path) -> None:
    client = TestClient(create_app({}))

    missing = client.post("/api/sessions/open", json={"modality": "toy"})
    assert missing.status_code == 422
    assert missing.json()["detail"][0]["loc"] == ["body", "title"]

    # The modality's own ValueError surfaces as a 400.
    bad = client.post(
        "/api/sessions/open",
        json={"modality": "toy", "title": "t", "paths": [str(tmp_path / "absent")]},
    )
    assert bad.status_code == 400
    assert "not a directory" in bad.json()["detail"]

    # Without a modality the body is a genome one, which needs a handle.
    genome = client.post("/api/sessions/open", json={"title": "t"})
    assert genome.status_code == 422
    assert genome.json()["detail"][0]["loc"] == ["body", "reference_handle"]

    unknown = client.post("/api/sessions/open", json={"modality": "nmr"})
    assert unknown.status_code == 400
    assert "'nmr' not registered" in unknown.json()["detail"]


# ----------------------------------------------------------------------
# Modalities do not see each other's sessions
# ----------------------------------------------------------------------


def test_kernels_only_answer_for_their_own_modality(
    toy, source_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    genome = build_viz_session(
        tmp_path,
        monkeypatch,
        align_sources=[
            {"coverage": [{"contig_id": 1, "sample_id": -1, "start": 0, "end": 10, "depth": 1}]}
        ],
    )
    client = TestClient(create_app(genome), raise_server_exceptions=True)
    toy_id = _open(client, source_dir)
    genome_id = genome.session_id

    toy_kinds = {t["kind"] for t in client.get("/api/tracks", params={"session": toy_id}).json()}
    genome_kinds = {
        t["kind"] for t in client.get("/api/tracks", params={"session": genome_id}).json()
    }
    assert toy_kinds == {"toy_trace"}
    assert "toy_trace" not in genome_kinds
    assert "coverage_histogram" in genome_kinds

    # A genome kernel asked about a toy session has no such binding — a
    # 404, not a 500 from reading `session.reference_genome`.
    locus = {"contig": "chr1", "start": 0, "end": 100}
    for route, params in (
        ("data", {"session": toy_id, "binding": "coverage-0", **locus}),
        ("metadata", {"session": toy_id, "binding": "coverage-0"}),
    ):
        response = client.get(f"/api/tracks/coverage_histogram/{route}", params=params)
        assert response.status_code == 404, route

    # And the reverse.
    for route, params in (
        ("data", {"session": genome_id, "binding": "toy_trace-0", "x0": 0, "x1": 1}),
        ("metadata", {"session": genome_id, "binding": "toy_trace-0"}),
    ):
        response = client.get(f"/api/tracks/toy_trace/{route}", params=params)
        assert response.status_code == 404, route

    # Genome-only routes refuse a toy session instead of misreading it.
    for path in (f"/api/sessions/{toy_id}/contigs", f"/api/sessions/{toy_id}/search?q=gene"):
        response = client.get(path)
        assert response.status_code == 404
        assert "not a genome session" in response.json()["detail"]
    assert client.get(f"/api/sessions/{genome_id}/contigs").status_code == 200


# ----------------------------------------------------------------------
# Generic source mutation, inspect-source, modality routes
# ----------------------------------------------------------------------


def test_sources_can_be_added_and_removed_on_any_modality(
    toy, source_dir: Path, tmp_path: Path
) -> None:
    client = TestClient(create_app({}))
    sid = _open(client, source_dir)
    second = tmp_path / "toy-run-2"
    second.mkdir()

    added = client.post(f"/api/sessions/{sid}/sources", json={"path": str(second)})
    assert added.status_code == 201
    manifest = added.json()
    assert manifest["session_id"] == sid  # rebuilt in place
    assert [s["label"] for s in manifest["sources"]] == ["toy-run", "toy-run-2"]
    # The binding cache was evicted: the new source has a track.
    tracks = client.get("/api/tracks", params={"session": sid}).json()
    assert [t["binding_id"] for t in tracks] == ["toy_trace-0", "toy_trace-1"]

    refused = client.post(f"/api/sessions/{sid}/sources", json={"path": str(tmp_path / "absent")})
    assert refused.status_code == 400

    first_id = manifest["sources"][0]["source_id"]
    removed = client.delete(f"/api/sessions/{sid}/sources/{first_id}")
    assert removed.status_code == 200
    assert [s["label"] for s in removed.json()["sources"]] == ["toy-run-2"]
    tracks = client.get("/api/tracks", params={"session": sid}).json()
    assert [t["label"] for t in tracks] == ["demo / toy-run-2"]


def test_inspect_source_and_routers_dispatch_to_the_modality(
    toy, source_dir: Path, tmp_path: Path
) -> None:
    client = TestClient(create_app({}))

    ok = client.post(
        "/api/sessions/inspect-source", json={"modality": "toy", "path": str(source_dir)}
    )
    assert ok.status_code == 200
    assert ok.json() == {"path": str(source_dir), "kind": "trace"}

    empty = tmp_path / "empty"
    empty.mkdir()
    bad = client.post(
        "/api/sessions/inspect-source", json={"modality": "toy", "path": str(empty)}
    )
    assert bad.status_code == 400
    assert "no trace.txt" in bad.json()["detail"]

    # The modality's own router was mounted by create_app.
    assert client.get("/api/toy/ping").json() == {"modality": "toy"}
    # ... alongside the genome modality's.
    assert client.get("/api/references").status_code == 200


# ----------------------------------------------------------------------
# Registry contracts
# ----------------------------------------------------------------------


def test_modality_registry() -> None:
    assert "genome" in registered_modalities()
    assert get_modality("genome").name == "genome"
    with pytest.raises(KeyError, match="not registered"):
        get_modality("nmr")
    with pytest.raises(ValueError, match="already registered"):
        register_modality(get_modality("genome"))
    assert "toy" not in registered_modalities()  # the fixture cleans up


def test_a_kernel_must_name_its_modality() -> None:
    with pytest.raises(TypeError, match="modality"):

        @register_track
        class _NoModality(TrackKernel):
            kind = "no_modality_kernel"
            schema = TOY_SCHEMA

            def discover(self, session):  # type: ignore[override]
                return []

            def metadata(self, binding):  # type: ignore[override]
                return {}

            def threshold(self, binding, query):  # type: ignore[override]
                return ThresholdDecision.VECTOR

            def fetch(self, binding, query, mode):  # type: ignore[override]
                return iter(())

    assert "no_modality_kernel" not in tracks_base.registered_kinds()


def test_query_model_cannot_take_the_endpoints_own_parameter_names() -> None:
    @dataclass(frozen=True, kw_only=True)
    class _Clash(TrackQuery):
        session: str = ""

    with pytest.raises(TypeError, match="reserves"):
        _query_plan(_Clash)


def test_every_shipped_kernel_declares_a_modality_and_a_query_model() -> None:
    for kind in tracks_base.registered_kinds():
        kernel = tracks_base.get_kernel(kind)
        assert kernel.modality in registered_modalities(), kind
        assert issubclass(kernel.query_model, TrackQuery), kind
        _query_plan(kernel.query_model)  # no reserved-name clash
