"""SQLite DDL for EncyclopeDIA's container formats.

The result library's (``.dlib`` / ``.elib``) is read from the pinned
jar's captured schema; the spectra file's (``.dia``) is written out
below and held to its capture by ``tests/test_encyclopedia_output_capture.py``.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# .dlib / .elib shared schema: container 0.1.18, as captured
# ──────────────────────────────────────────────────────────────────────
#
# The tables and indexes EncyclopeDIA 6.9.23 creates in every result
# library and library (10 tables, 22 indexes), read from the checked-in
# capture of a real file (``data/_raw/encyclopedia/<version>/schema/
# result_library.sql``, written by ``scripts/survey-encyclopedia-outputs.py``)
# rather than copied by hand, so the writer creates exactly what the
# pinned jar does. A new pin is a new capture.

#: The container version 6.9.23 writes in a result's ``metadata``.
LIBRARY_CONTAINER_VERSION = "0.1.18"


def _captured_result_library() -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    from importlib import resources

    from constellation.thirdparty.encyclopedia import MINIMUM_VERSION

    text = (
        resources.files("constellation.data")
        .joinpath("_raw", "encyclopedia", MINIMUM_VERSION, "schema", "result_library.sql")
        .read_text()
    )
    tables: list[tuple[str, str]] = []
    indexes: list[str] = []
    for line in text.splitlines():
        statement = line.strip().rstrip(";")
        if statement.startswith("CREATE TABLE "):
            tables.append((statement.split()[2].strip('"'), statement))
        elif statement.startswith("CREATE ") and " INDEX " in statement:
            indexes.append(statement)
    return tuple(tables), tuple(indexes)


LIBRARY_TABLES, LIBRARY_INDEXES = _captured_result_library()

LIBRARY_TABLE_NAMES = frozenset(name for name, _ddl in LIBRARY_TABLES)


# ──────────────────────────────────────────────────────────────────────
# .dia raw-spectra schema
# ──────────────────────────────────────────────────────────────────────
#
# Two generations exist and the reader accepts both:
#
#   0.5.0  written by EncyclopeDIA 2.12.30's own mzML converter. Four
#          tables; the ``DIA_*_DDL`` constants without a suffix.
#   0.8.0  written by MSRawJava (the library inside EncyclopeDIA 6.x and
#          MSForest). Adds ``fractions``, ion-mobility arrays, a per-MS2
#          ``TIC``, RT and ion-mobility bounds on ``ranges``, and renames
#          ``spectra.IsolationWindowCenter`` to ``IsolationWindowTarget``.
#          The ``*_V08`` constants, copied from a file that library wrote.
#
# Constellation writes 0.8.0 only (``DIA_FORMAT_VERSION``).

DIA_FORMAT_VERSION = "0.8.0"


DIA_METADATA_DDL = """
CREATE TABLE metadata (
    Key string not null,
    Value string not null,
    primary key (Key)
)
"""

DIA_PRECURSOR_DDL = """
CREATE TABLE precursor (
    Fraction int not null,
    SpectrumName string not null,
    SpectrumIndex int not null,
    ScanStartTime float not null,
    IonInjectionTime float,
    IsolationWindowLower float not null,
    IsolationWindowUpper float not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    TIC float,
    primary key (SpectrumIndex)
)
"""

DIA_RANGES_DDL = """
CREATE TABLE ranges (
    Start float not null,
    Stop float not null,
    DutyCycle float not null,
    NumWindows int
)
"""

DIA_SPECTRA_DDL = """
CREATE TABLE spectra (
    Fraction int not null,
    SpectrumName string not null,
    PrecursorName string,
    SpectrumIndex int not null,
    ScanStartTime float not null,
    IonInjectionTime float,
    IsolationWindowLower float not null,
    IsolationWindowCenter float not null,
    IsolationWindowUpper float not null,
    PrecursorCharge int not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    primary key (SpectrumIndex)
)
"""

DIA_TABLES: tuple[tuple[str, str], ...] = (
    ("metadata", DIA_METADATA_DDL),
    ("precursor", DIA_PRECURSOR_DDL),
    ("ranges", DIA_RANGES_DDL),
    ("spectra", DIA_SPECTRA_DDL),
)

DIA_FRACTIONS_DDL = """
CREATE TABLE fractions (
    Fraction int not null,
    Name string not null,
    primary key (Fraction)
)
"""

DIA_PRECURSOR_DDL_V08 = """
CREATE TABLE precursor (
    Fraction int not null,
    SpectrumName string not null,
    SpectrumIndex int not null,
    ScanStartTime float not null,
    IonInjectionTime float,
    IsolationWindowLower float not null,
    IsolationWindowUpper float not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    IonMobilityArrayEncodedLength int,
    IonMobilityArray blob,
    TIC float,
    primary key (SpectrumIndex)
)
"""

DIA_RANGES_DDL_V08 = """
CREATE TABLE ranges (
    Start float not null,
    Stop float not null,
    DutyCycle float not null,
    NumWindows int,
    IonMobilityStart float,
    IonMobilityStop float,
    RtStart float,
    RtStop float
)
"""

DIA_SPECTRA_DDL_V08 = """
CREATE TABLE spectra (
    Fraction int not null,
    SpectrumName string not null,
    PrecursorName string,
    SpectrumIndex int not null,
    ScanStartTime float not null,
    IonInjectionTime float,
    IsolationWindowLower float not null,
    IsolationWindowTarget float not null,
    IsolationWindowUpper float not null,
    PrecursorCharge int not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    IonMobilityArrayEncodedLength int,
    IonMobilityArray blob,
    TIC float,
    primary key (SpectrumIndex)
)
"""

DIA_TABLES_V08: tuple[tuple[str, str], ...] = (
    ("metadata", DIA_METADATA_DDL),
    ("fractions", DIA_FRACTIONS_DDL),
    ("precursor", DIA_PRECURSOR_DDL_V08),
    ("ranges", DIA_RANGES_DDL_V08),
    ("spectra", DIA_SPECTRA_DDL_V08),
)

# Built after the rows are in. Names and column lists as MSRawJava
# creates them.
DIA_INDEXES_V08: tuple[str, ...] = (
    'CREATE INDEX "precursor_index_isolation_window_lower" on "precursor" ("IsolationWindowLower" ASC)',
    'CREATE INDEX "precursor_index_isolation_window_upper" on "precursor" ("IsolationWindowUpper" ASC)',
    'CREATE INDEX "precursor_index_scan_start_time" on "precursor" ("ScanStartTime" ASC)',
    'CREATE INDEX "spectra_index_isolation_window_lower" on "spectra" ("IsolationWindowLower" ASC)',
    'CREATE INDEX "spectra_index_isolation_window_upper" on "spectra" ("IsolationWindowUpper" ASC)',
    'CREATE INDEX "spectra_index_scan_start_time_and_windows" on "spectra" '
    '("ScanStartTime","IsolationWindowLower","IsolationWindowUpper" ASC)',
)

DIA_TABLE_NAMES = frozenset(
    name for name, _ddl in (*DIA_TABLES, *DIA_TABLES_V08)
)


__all__ = [
    "LIBRARY_CONTAINER_VERSION",
    "LIBRARY_TABLES",
    "LIBRARY_TABLE_NAMES",
    "LIBRARY_INDEXES",
    "DIA_METADATA_DDL",
    "DIA_PRECURSOR_DDL",
    "DIA_RANGES_DDL",
    "DIA_SPECTRA_DDL",
    "DIA_TABLES",
    "DIA_FORMAT_VERSION",
    "DIA_FRACTIONS_DDL",
    "DIA_PRECURSOR_DDL_V08",
    "DIA_RANGES_DDL_V08",
    "DIA_SPECTRA_DDL_V08",
    "DIA_TABLES_V08",
    "DIA_INDEXES_V08",
    "DIA_TABLE_NAMES",
]
