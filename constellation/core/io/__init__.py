"""I/O primitives — schema registry, reader ABC, bundle pattern.

Leaf of the core DAG — never imports other core modules. Only
format-level concerns; no sequence/chem/stats/units logic.

Public surface:
    schemas      - TRACE_1D, PEAK_TABLE, spectra_matrix_2d(n);
                   pack_metadata / unpack_metadata / with_metadata;
                   trace_to_tensor / spectra_to_tensor /
                   tensor_to_spectra; cast_to_schema; SchemaRegistry.
    readers      - RawReader ABC, ReadResult, register_reader,
                   find_reader, READER_REGISTRY (via
                   registered_readers()).
    bundle       - Bundle ABC, OpcBundle, DirBundle.
"""

from constellation.core.io.bundle import Bundle, DirBundle, OpcBundle
from constellation.core.io.readers import (
    RawReader,
    ReaderNotFoundError,
    ReadResult,
    find_reader,
    register_reader,
    registered_readers,
)
from constellation.core.io.schemas import (
    PEAK_TABLE,
    TRACE_1D,
    cast_to_schema,
    get_schema,
    pack_metadata,
    register_schema,
    registered_schemas,
    spectra_matrix_2d,
    spectra_to_tensor,
    tensor_to_spectra,
    trace_to_tensor,
    unpack_metadata,
    with_metadata,
)

__all__ = [
    # bundle
    "Bundle",
    "DirBundle",
    "OpcBundle",
    # readers
    "RawReader",
    "ReadResult",
    "ReaderNotFoundError",
    "find_reader",
    "register_reader",
    "registered_readers",
    # schemas
    "PEAK_TABLE",
    "TRACE_1D",
    "cast_to_schema",
    "get_schema",
    "pack_metadata",
    "register_schema",
    "registered_schemas",
    "spectra_matrix_2d",
    "spectra_to_tensor",
    "tensor_to_spectra",
    "trace_to_tensor",
    "unpack_metadata",
    "with_metadata",
]
