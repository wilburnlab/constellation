"""Mass-spec browser session.

A ``MassSpecSession`` is a set of sources the user attached and the
acquisitions they resolve to. A source is a directory: a converted
acquisition (a *bundle*, from ``massspec convert`` or ``massspec
encyclopedia import-spectra``) or a search experiment (``massspec
search``). An experiment brings each searched run whose bundle can be
found (``massspec.results.acquisition_bundles``: the search's own input,
a conversion inside the experiment, or an attached bundle of the same
run), linked to the searches of it, which is what makes identifications
browsable. Every acquisition is described once, when the session opens,
from its manifest and scan table, so the kernels and routes answer from
that description.

An acquisition's ``key`` is derived from its bundle's path, so the same run
keeps its key however it was reached and across source changes; the
client keys per-acquisition state by it. The session's own id is derived
from an *anchor* fixed when the session first opens, so removing the
source that happened to be first does not orphan the client's saved layout.

These classes are the modality's implementation of the ``SessionLike`` /
``SourceLike`` contracts in ``viz.server.session``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal

from constellation.viz.server.session import derive_session_id, derive_source_id

SourceKind = Literal["acquisition", "experiment"]

#: Manifest kinds a source may have, and the source kind each is.
_SOURCE_KINDS: dict[str, SourceKind] = {
    "thermo_acquisition": "acquisition",
    "encyclopedia_dia_acquisition": "acquisition",
    "massspec_experiment": "experiment",
}

DEFAULT_LABEL = "mass-spec session"


@dataclass(frozen=True, slots=True)
class MsSource:
    """One attached directory."""

    path: Path
    kind: SourceKind
    label: str

    @property
    def source_id(self) -> str:
        return derive_source_id(self.path, self.kind)


@dataclass(frozen=True, slots=True)
class MsAcquisition:
    """One run the session can show."""

    #: Stable id of the run in this session, derived from its bundle path.
    key: str
    label: str
    bundle: Path
    #: The source this run was attached through.
    source_id: str
    #: ``massspec.scans.open_acquisition`` as plain data: ``bundle_kind``,
    #: ``source_file``, ``source_sha256``, ``acquisition_type``,
    #: ``staggered``, ``profile_mode``, ``levels``, ``scans_per_level``,
    #: ``analyzers``, ``rt_range``, ``has_index``, ``default_tolerance``.
    info: dict[str, Any]
    #: The experiments that searched this run, one dict each:
    #: ``experiment`` (path), ``experiment_source_id``, ``acquisition_id``
    #: (the run's id in that experiment) and ``units`` (its searches).
    links: tuple[dict[str, Any], ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "bundle": str(self.bundle),
            "source_id": self.source_id,
            **self.info,
            "searches": [dict(link) for link in self.links],
        }

    def link(self, experiment_source_id: str) -> dict[str, Any]:
        """This run's link to one experiment; ``KeyError`` when it has none."""
        for link in self.links:
            if link["experiment_source_id"] == experiment_source_id:
                return link
        raise KeyError(experiment_source_id)


@dataclass(frozen=True, slots=True)
class MassSpecSession:
    """Resolved state of one mass-spec browser."""

    modality: ClassVar[str] = "massspec"
    session_id: str
    label: str
    #: What ``session_id`` derives from; kept across source changes.
    anchor: str
    sources: tuple[MsSource, ...]
    acquisitions: tuple[MsAcquisition, ...]
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
    ) -> "MassSpecSession":
        """Build a session from ``{"path", "kind"?, "label"?}`` source dicts.

        Raises ``ValueError`` for a source that is not a directory, not a
        converted acquisition or search experiment, or whose stated
        ``kind`` its manifest contradicts. A searched run that cannot be
        browsed (no bundle found; fractions searched as one) is a warning.
        """
        from constellation.massspec.results import acquisition_bundles

        built = [_load_source(entry) for entry in sources]
        acquisitions: dict[str, MsAcquisition] = {}
        bundles = [s for s in built if s.kind == "acquisition"]
        for source in bundles:
            acquisition = _describe(source.path, label=source.label, source_id=source.source_id)
            acquisitions.setdefault(acquisition.key, acquisition)  # the same bundle attached twice
        warnings: list[str] = []
        for experiment in (s for s in built if s.kind == "experiment"):
            for found in acquisition_bundles(experiment.path, candidates=[b.path for b in bundles]):
                name = Path(str(found["source_file"])).name
                if found["merged_from"]:
                    warnings.append(
                        f"{experiment.label}: {name} is fractions searched as one; browse the "
                        f"fractions' own searches (identifications are not split by fraction)"
                    )
                    continue
                if found["bundle"] is None:
                    warnings.append(
                        f"{experiment.label}: no converted acquisition found for {name}; attach its "
                        f"bundle (`massspec convert`, or `massspec encyclopedia import-spectra` for a "
                        f"spectra file) to browse its peaks"
                    )
                    continue
                bundle = Path(found["bundle"]).resolve()
                key = derive_source_id(bundle, "acquisition")
                if key not in acquisitions:
                    acquisitions[key] = _describe(bundle, label=bundle.name, source_id=experiment.source_id)
                link = {
                    "experiment": str(experiment.path),
                    "experiment_source_id": experiment.source_id,
                    "experiment_label": experiment.label,
                    "acquisition_id": int(found["acquisition_id"]),
                    "units": list(found["units"]),
                }
                acquisitions[key] = replace(acquisitions[key], links=(*acquisitions[key].links, link))
        resolved_label = label or (built[0].label if built else DEFAULT_LABEL)
        resolved_anchor = anchor or (str(built[0].path) if built else "massspec")
        return cls(
            session_id=derive_session_id(resolved_anchor, resolved_label),
            label=resolved_label,
            anchor=resolved_anchor,
            sources=tuple(built),
            acquisitions=tuple(acquisitions.values()),
            warnings=tuple(warnings),
            saved_as=saved_as,
        )

    def with_sources(self, sources: Iterable[dict[str, Any]]) -> "MassSpecSession":
        """Rebuild with a new source list under the same ``session_id``."""
        return MassSpecSession.open(
            sources=sources, label=self.label, saved_as=self.saved_as, anchor=self.anchor
        )

    def acquisition(self, key: str) -> MsAcquisition:
        """The acquisition with ``key``; ``KeyError`` when there is none."""
        for acquisition in self.acquisitions:
            if acquisition.key == key:
                return acquisition
        raise KeyError(key)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "modality": self.modality,
            "label": self.label,
            "n_sources": len(self.sources),
            "n_acquisitions": len(self.acquisitions),
            "warnings": list(self.warnings),
            "saved_as": self.saved_as,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "sources": [
                {"source_id": s.source_id, "path": str(s.path), "kind": s.kind, "label": s.label}
                for s in self.sources
            ],
            "acquisitions": [a.describe() for a in self.acquisitions],
        }


def _source_kind(path: Path) -> SourceKind:
    from constellation.core.manifests import manifest_kind

    found = manifest_kind(path)
    if found not in _SOURCE_KINDS:
        raise ValueError(
            f"{path} is neither a converted acquisition nor a search experiment (its "
            f"manifest says {found!r}); convert a .raw with `massspec convert`, import an "
            f"EncyclopeDIA spectra file with `massspec encyclopedia import-spectra`, or "
            f"give the directory `massspec search` wrote"
        )
    return _SOURCE_KINDS[found]


def _load_source(entry: dict[str, Any]) -> MsSource:
    raw = entry.get("path")
    if not raw:
        raise ValueError(f"source entry missing 'path': {entry!r}")
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"source path is not a directory: {path}")
    kind = _source_kind(path)
    stated = entry.get("kind")
    if stated is not None and stated != kind:
        raise ValueError(f"source at {path} is kind={kind!r} but was given as {stated!r}")
    return MsSource(path=path, kind=kind, label=str(entry.get("label") or path.name))


def _describe(bundle: Path, *, label: str, source_id: str) -> MsAcquisition:
    from constellation.massspec.scans import default_tolerance, open_acquisition

    info = open_acquisition(bundle)
    tolerance = {
        str(level): list(default_tolerance(analyzers)) for level, analyzers in info.analyzers.items()
    }
    return MsAcquisition(
        key=derive_source_id(bundle, "acquisition"),
        label=label,
        bundle=bundle,
        source_id=source_id,
        info={
            "bundle_kind": info.kind,
            "source_file": info.source_file,
            "source_sha256": info.source_sha256,
            "acquisition_type": info.acquisition_type,
            "staggered": info.staggered,
            "profile_mode": info.profile_mode,
            "levels": list(info.levels),
            "scans_per_level": {str(k): v for k, v in info.scans_per_level.items()},
            "analyzers": {str(k): list(v) for k, v in info.analyzers.items()},
            "rt_range": list(info.rt_range),
            "has_index": info.index is not None,
            "default_tolerance": tolerance,
        },
    )


def inspect_source(path: Path) -> dict[str, Any]:
    """Describe a candidate source directory for the entry form."""
    kind = _source_kind(path)
    if kind == "experiment":
        from constellation.massspec.results import acquisition_bundles, search_units

        runs = acquisition_bundles(path)
        return {
            "path": str(path.resolve()),
            "kind": kind,
            "searches": len(search_units(path)),
            "runs": [Path(str(r["source_file"])).name for r in runs],
            "runs_with_bundle": sum(r["bundle"] is not None for r in runs),
        }
    acquisition = _describe(path.resolve(), label=path.name, source_id="")
    return {"path": str(path.resolve()), "kind": kind, **acquisition.info}


def session_from_saved(saved: Any) -> MassSpecSession:
    """Build a session from a ``viz.sessions.SavedSession``."""
    return MassSpecSession.open(
        sources=[
            {"path": s["path"], "kind": s.get("kind"), "label": s.get("label")}
            for s in saved.sources
        ],
        label=saved.label,
        saved_as=saved.slug,
    )


__all__ = [
    "MassSpecSession",
    "MsAcquisition",
    "MsSource",
    "inspect_source",
    "session_from_saved",
]
