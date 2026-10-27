"""Thermo ``.raw`` reader + streaming converter for ``constellation.massspec``.

Submodules:

- ``_filter`` — pure-Python scan filter-string regex parser
  (no .NET dependency).
- ``_trailer`` — declarative table mapping ~80 Thermo per-scan trailer
  keys to typed ``SCAN_METADATA_TABLE`` columns + coercion helpers.
- ``_netruntime`` — pythonnet + CoreCLR + DLL bootstrap. Resolves the
  Thermo CommonCore DLL pack via the standard
  ``constellation.thirdparty`` registry (env var
  ``CONSTELLATION_THERMO_HOME`` → ``third_party/thermo/current/``).
- ``_scans`` — reads a range of scans into blocks (scan-metadata table +
  peaks as numpy arrays); decides how each scan's peaks are made
  (stored centroids, the instrument's centroid stream, Thermo's
  centroider, or profile points).
- ``_read`` — :class:`ThermoReader` (in-memory ``RawReader``), run
  metadata and the per-level summary.
- ``_convert`` — :func:`convert`: scan ranges read in worker processes,
  one peak part per range, the bundle finished in the parent.
- ``manifest`` — :class:`ThermoAcquisitionManifest` dataclass + JSON
  reader/writer for the per-bundle ``manifest.json``.

Importing this package triggers the ``@register_reader`` side effect
that wires ``.raw`` into :func:`constellation.core.io.find_reader` for
the ``ms`` modality. Doing real work still requires the DLL pack
installed via ``scripts/install-thermo-dlls.sh`` —
:func:`_netruntime.require_thermo` raises ``ImportError`` with install
guidance otherwise.
"""

from __future__ import annotations

from ._batch import BatchResult, BatchStatus, convert_batch
from ._filter import parse_filter_string
from ._netruntime import is_thermo_available, load_clr, require_thermo
from constellation.massspec.bundle import DEFAULT_RT_BIN_WIDTH_S

from ._convert import DEFAULT_BATCH_SIZE, convert, default_workers, plan_ranges
from ._read import (
    SOURCE_FORMAT,
    EmptyLevelError,
    ThermoReader,
    describe_levels,
    level_summary,
)
from ._scans import CENTROIDERS
from .manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ThermoAcquisitionManifest,
    read_manifest,
    read_manifest_dir,
    write_manifest,
)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_RT_BIN_WIDTH_S",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "SOURCE_FORMAT",
    "BatchResult",
    "BatchStatus",
    "CENTROIDERS",
    "EmptyLevelError",
    "ThermoAcquisitionManifest",
    "ThermoReader",
    "convert",
    "convert_batch",
    "default_workers",
    "describe_levels",
    "is_thermo_available",
    "level_summary",
    "load_clr",
    "parse_filter_string",
    "plan_ranges",
    "read_manifest",
    "read_manifest_dir",
    "require_thermo",
    "write_manifest",
]
