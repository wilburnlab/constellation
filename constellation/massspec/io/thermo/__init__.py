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
- ``_read`` — :class:`ThermoReader` (in-memory ``RawReader``) +
  :func:`convert` (streaming-to-disk writer used by the CLI).
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
from ._read import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RT_BIN_WIDTH_S,
    SOURCE_FORMAT,
    ThermoReader,
    convert,
)
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
    "ThermoAcquisitionManifest",
    "ThermoReader",
    "convert",
    "convert_batch",
    "is_thermo_available",
    "load_clr",
    "parse_filter_string",
    "read_manifest",
    "read_manifest_dir",
    "require_thermo",
    "write_manifest",
]
