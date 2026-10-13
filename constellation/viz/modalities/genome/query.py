"""Query models for the genome kernels.

Every genome track is drawn over one locus, so they share
:class:`GenomeQuery`; a kernel that reads more than the locus declares a
subclass carrying exactly the extra fields it uses. The server validates
the request's query string against the kernel's model
(``TrackKernel.query_model``), so a field that is not declared here is
not a parameter of that kernel's data endpoint.

Stdlib dataclasses only — this module is imported by
``import constellation.viz``, which must work without pydantic. Range
constraints ride as ``field(metadata=...)`` using pydantic ``Field``
keyword names; the endpoint's ``TypeAdapter`` enforces them (HTTP 422).
Rules that span fields go in :meth:`check` (HTTP 400).

No ``slots=True``: subclasses call ``super().check()``, and zero-argument
``super()`` does not work in a slots-dataclass subclass on Python 3.12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from constellation.viz.tracks.base import TrackQuery


@dataclass(frozen=True, kw_only=True)
class GenomeQuery(TrackQuery):
    """A locus: ``contig:[start, end)``, 0-based half-open."""

    contig: str
    start: int = field(metadata={"ge": 0})
    end: int = field(metadata={"ge": 0})

    def check(self) -> None:
        super().check()
        if self.end <= self.start:
            raise ValueError("end must be greater than start")


@dataclass(frozen=True, kw_only=True)
class CoverageQuery(GenomeQuery):
    """Coverage can be restricted to a set of samples."""

    samples: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ReadPileupQuery(GenomeQuery):
    """Pile-up adds a sample restriction and the MAPQ pushdown filter:
    alignments with ``mapq < min_mapq`` are dropped at scan time."""

    samples: tuple[str, ...] = ()
    min_mapq: int = field(default=0, metadata={"ge": 0, "le": 60})


@dataclass(frozen=True, kw_only=True)
class ClusterPileupQuery(ReadPileupQuery):
    """Cluster pile-up has two views with different wire schemas:
    ``clusters`` (one rectangle per cluster, the default) and ``members``
    (each cluster expanded into its member reads, which is where
    ``min_mapq`` applies). ``None`` means the default view."""

    cluster_view: Literal["clusters", "members"] | None = None


__all__ = [
    "ClusterPileupQuery",
    "CoverageQuery",
    "GenomeQuery",
    "ReadPileupQuery",
]
