"""Transcriptome browser session.

A ``TranscriptomeSession`` is a set of de novo EM runs (the directory
``transcriptome cluster --mode em-*`` writes) and what the browser needs to
know about each, read once when the session opens: its rounds, cluster and
read counts, which evidence it carries (``em.results.run_info``), and the
names of its samples where its demultiplexing output can be found.

A run's ``key`` is its ``source_id``, derived from its path, so the client's
per-run state survives source changes. The session's id is derived from an
anchor fixed when it first opens, as the mass-spec session's is.

These classes implement the ``SessionLike`` / ``SourceLike`` contracts in
``viz.server.session``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable

from constellation.viz.server.session import derive_session_id, derive_source_id

#: The one source kind: an EM run directory.
EM_RUN = "em_run"

DEFAULT_LABEL = "transcriptome session"


@dataclass(frozen=True, slots=True)
class TxSource:
    """One attached EM run directory."""

    path: Path
    kind: str
    label: str
    #: Where the run's demultiplexing output is, when not where its manifest
    #: says (a run made on a cluster and copied here).
    demux_dir: Path | None = None

    @property
    def source_id(self) -> str:
        return derive_source_id(self.path, self.kind)

    def to_entry(self) -> dict[str, Any]:
        """The input dict that reopens this source, ``demux_dir`` included
        (the server's source add / remove rebuilds a session from these)."""
        return {
            "path": str(self.path),
            "kind": self.kind,
            "label": self.label,
            "demux_dir": str(self.demux_dir) if self.demux_dir else None,
        }


@dataclass(frozen=True, slots=True)
class TxRun:
    """One run the session can show."""

    key: str
    label: str
    path: Path
    #: ``em.results.run_info`` as plain data, plus ``samples``: sample id ->
    #: name, empty when the demultiplexing output was not found.
    info: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "path": str(self.path), **self.info}


@dataclass(frozen=True, slots=True)
class TranscriptomeSession:
    """Resolved state of one transcriptome browser."""

    modality: ClassVar[str] = "transcriptome"
    session_id: str
    label: str
    anchor: str
    sources: tuple[TxSource, ...]
    runs: tuple[TxRun, ...]
    warnings: tuple[str, ...] = ()
    saved_as: str | None = None

    @classmethod
    def open(
        cls,
        *,
        sources: Iterable[dict[str, Any]],
        label: str | None = None,
        saved_as: str | None = None,
        anchor: str | None = None,
    ) -> "TranscriptomeSession":
        """Build a session from ``{"path", "kind"?, "label"?, "demux_dir"?}``
        dicts. Raises ``ValueError`` for a source that is not an EM run."""
        loaded = [_load_source(dict(entry)) for entry in sources]
        warnings: list[str] = []
        runs = []
        for src in loaded:
            run, more = _describe(src)
            runs.append(run)
            warnings.extend(more)
        resolved_label = label or (loaded[0].label if loaded else DEFAULT_LABEL)
        resolved_anchor = anchor or (str(loaded[0].path) if loaded else "")
        return cls(
            session_id=derive_session_id(resolved_anchor, resolved_label),
            label=resolved_label,
            anchor=resolved_anchor,
            sources=tuple(loaded),
            runs=tuple(runs),
            warnings=tuple(warnings),
            saved_as=saved_as,
        )

    def with_sources(self, sources: Iterable[dict[str, Any]]) -> "TranscriptomeSession":
        return TranscriptomeSession.open(
            sources=sources, label=self.label, saved_as=self.saved_as, anchor=self.anchor
        )

    def run(self, key: str) -> TxRun:
        """The run with ``key``; ``KeyError`` when there is none."""
        for run in self.runs:
            if run.key == key:
                return run
        raise KeyError(key)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "modality": self.modality,
            "label": self.label,
            "n_sources": len(self.sources),
            "n_runs": len(self.runs),
            "warnings": list(self.warnings),
            "saved_as": self.saved_as,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "sources": [
                {
                    "source_id": s.source_id,
                    "path": str(s.path),
                    "kind": s.kind,
                    "label": s.label,
                    "demux_dir": str(s.demux_dir) if s.demux_dir else None,
                }
                for s in self.sources
            ],
            "runs": [r.describe() for r in self.runs],
        }


def is_em_run(path: Path) -> bool:
    """An EM run: a ``rounds/`` directory, and a manifest, when there is
    one, of a cluster run in ``em`` mode."""
    path = Path(path)
    if not (path / "rounds").is_dir():
        return False
    manifest = path / "manifest.json"
    if not manifest.exists():
        return True  # a run whose final stage has not been written
    try:
        m = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(m, dict):
        return False
    params = m.get("parameters")
    return m.get("kind") == "cluster" and isinstance(params, dict) and params.get("mode") == "em"


def _load_source(entry: dict[str, Any]) -> TxSource:
    path = Path(entry["path"]).expanduser()
    if not path.is_dir():
        raise ValueError(f"not a directory: {path}")
    kind = entry.get("kind") or EM_RUN
    if kind != EM_RUN:
        raise ValueError(f"unknown source kind {kind!r} (the transcriptome browser opens {EM_RUN!r})")
    if not is_em_run(path):
        raise ValueError(
            f"{path} is not an EM run (the directory `transcriptome cluster --mode "
            f"em-orf / em-kmer` writes)"
        )
    demux = entry.get("demux_dir")
    return TxSource(
        path=path.resolve(),
        kind=kind,
        label=entry.get("label") or path.name,
        demux_dir=Path(demux).expanduser().resolve() if demux else None,
    )


def _sample_names(demux_dir: Path | None) -> dict[int, str]:
    if demux_dir is None:
        return {}
    path = Path(demux_dir) / "samples" / "samples.parquet"
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["sample_id", "sample_name"])
    return {
        int(i): str(n)
        for i, n in zip(table.column("sample_id").to_pylist(), table.column("sample_name").to_pylist())
    }


def _describe(src: TxSource) -> tuple[TxRun, list[str]]:
    from constellation.sequencing.transcriptome.cluster.denovo.em.results import run_info

    info = dict(run_info(src.path))
    warnings = []
    if not info.get("finished"):
        warnings.append(f"{src.label}: no final clusters (the run has not finished); templates only")
    if not info.get("has_compact_evidence"):
        warnings.append(
            f"{src.label}: no compact M-step evidence (a run from before it); arcs and "
            "column statistics come from replay alone"
        )
    demux = src.demux_dir or (Path(info["demux_dir"]) if info.get("demux_dir") else None)
    names = _sample_names(demux)
    if demux is not None and not names:
        warnings.append(
            f"{src.label}: no samples at {demux / 'samples'}; sample ids are shown "
            "(give the run's demultiplexing directory with --demux-dir)"
        )
    info["samples"] = {str(k): v for k, v in sorted(names.items())}
    info.pop("parameters", None)
    info["demux_dir"] = str(demux) if demux else None
    return TxRun(key=src.source_id, label=src.label, path=src.path, info=info), warnings


def inspect_source(path: Path) -> dict[str, Any]:
    """Describe a candidate source for an entry form."""
    src = _load_source({"path": str(path)})
    run, warnings = _describe(src)
    return {"path": str(src.path), "kind": src.kind, **run.info, "warnings": warnings}


def session_from_saved(saved: Any) -> TranscriptomeSession:
    """Build a session from a ``viz.sessions.SavedSession``."""
    return TranscriptomeSession.open(
        sources=[
            {
                "path": s["path"],
                "kind": s.get("kind"),
                "label": s.get("label"),
                "demux_dir": s.get("demux_dir"),
            }
            for s in saved.sources
        ],
        label=saved.label,
        saved_as=saved.slug,
    )


__all__ = [
    "EM_RUN",
    "TranscriptomeSession",
    "TxRun",
    "TxSource",
    "inspect_source",
    "is_em_run",
    "session_from_saved",
]
