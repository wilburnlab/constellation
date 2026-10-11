"""SQLite DDL constants for EncyclopeDIA's container formats.

Single source of truth — both the writer and the schema-checking
sniffers reference the same DDL strings, so format-rev drift surfaces
as a SQL error rather than a silent column-list disagreement.

Copied verbatim from EncyclopeDIA 2.12.30's ``LibraryFile.java``
schema. Older / newer versions may differ by one or two columns; we'll
expand the set when a migration target appears.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# .dlib / .elib shared schema (9 tables)
# ──────────────────────────────────────────────────────────────────────


# ``entries`` — one row per (peptide, charge, source-file) precursor.
# All sample-specific fields (chromatogram blob, RT window, source file
# distinction) live in this table; a "pure" predicted dlib leaves them
# NULL while an empirical elib populates them.
ENTRIES_DDL = """
CREATE TABLE entries (
    PrecursorMz double not null,
    PrecursorCharge int not null,
    PeptideModSeq string not null,
    PeptideSeq string not null,
    Copies int not null,
    RTInSeconds double not null,
    Score double not null,
    MassEncodedLength int not null,
    MassArray blob not null,
    IntensityEncodedLength int not null,
    IntensityArray blob not null,
    CorrelationEncodedLength int,
    CorrelationArray blob,
    QuantifiedIonsArray blob,
    RTInSecondsStart double,
    RTInSecondsStop double,
    MedianChromatogramEncodedLength int,
    MedianChromatogramArray blob,
    SourceFile string not null
)
"""

PEPTIDETOPROTEIN_DDL = """
CREATE TABLE peptidetoprotein (
    PeptideSeq string not null,
    isDecoy boolean,
    ProteinAccession string not null
)
"""

PEPTIDESCORES_DDL = """
CREATE TABLE peptidescores (
    PrecursorCharge int not null,
    PeptideModSeq string not null,
    PeptideSeq string not null,
    SourceFile string not null,
    QValue double not null,
    PosteriorErrorProbability double not null,
    IsDecoy boolean not null
)
"""

PROTEINSCORES_DDL = """
CREATE TABLE proteinscores (
    ProteinGroup int not null,
    ProteinAccession string not null,
    SourceFile string not null,
    QValue double not null,
    MinimumPeptidePEP double not null,
    IsDecoy boolean not null
)
"""

METADATA_DDL = """
CREATE TABLE metadata (
    Key string not null,
    Value string not null
)
"""

PEPTIDEQUANTS_DDL = """
CREATE TABLE peptidequants (
    PrecursorCharge int not null,
    PeptideModSeq string not null,
    PeptideSeq string not null,
    SourceFile string not null,
    RTInSecondsCenter double not null,
    RTInSecondsStart double not null,
    RTInSecondsStop double not null,
    TotalIntensity double not null,
    NumberOfQuantIons int not null,
    QuantIonMassLength int not null,
    QuantIonMassArray blob not null,
    QuantIonIntensityLength int,
    QuantIonIntensityArray blob,
    BestFragmentCorrelation double not null,
    BestFragmentDeltaMassPPM double not null,
    MedianChromatogramEncodedLength int not null,
    MedianChromatogramArray blob not null,
    MedianChromatogramRTEncodedLength int,
    MedianChromatogramRTArray blob,
    IdentifiedTICRatio double not null
)
"""

FRAGMENTQUANTS_DDL = """
CREATE TABLE fragmentquants (
    PrecursorCharge int not null,
    PeptideModSeq string not null,
    PeptideSeq string not null,
    SourceFile string not null,
    IonType string not null,
    IonIndex int not null,
    FragmentMass double not null,
    Correlation double not null,
    Background double not null,
    DeltaMassPPM double not null,
    Intensity double not null
)
"""

PEPTIDELOCALIZATIONS_DDL = """
CREATE TABLE peptidelocalizations (
    PrecursorCharge int not null,
    PeptideModSeq string not null,
    PeptideSeq string not null,
    SourceFile string not null,
    LocalizationPeptideModSeq string,
    LocalizationScore double,
    LocalizationFDR double,
    LocalizationIons string,
    NumberOfMods int,
    NumberOfModifiableResidues int,
    IsSiteSpecific boolean,
    IsLocalized boolean,
    RTInSecondsCenter double,
    LocalizedIntensity double,
    TotalIntensity double
)
"""

RETENTIONTIMES_DDL = """
CREATE TABLE retentiontimes (
    SourceFile string not null,
    Library float not null,
    Actual float not null,
    Predicted float not null,
    Delta float not null,
    Probability float not null,
    Decoy boolean,
    PeptideModSeq string
)
"""

# Indexes EncyclopeDIA's `LibraryFile.java` builds on every dlib/elib —
# matters for downstream EncyclopeDIA tooling reading our writer output.
LIBRARY_INDEXES = (
    "CREATE INDEX 'PeptideModSeq_PrecursorCharge_SourceFile_Entries_index' "
    "ON entries (PeptideModSeq ASC, PrecursorCharge ASC, SourceFile ASC)",
    "CREATE INDEX 'PeptideSeq_Entries_index' ON entries (PeptideSeq ASC)",
    "CREATE INDEX 'PrecursorMz_Entries_index' ON entries (PrecursorMz ASC)",
    "CREATE INDEX 'ProteinAccession_PeptideToProtein_index' "
    "ON peptidetoprotein (ProteinAccession ASC)",
    "CREATE INDEX 'PeptideSeq_PeptideToProtein_index' "
    "ON peptidetoprotein (PeptideSeq ASC)",
    "CREATE INDEX 'Key_Metadata_index' ON metadata (Key ASC)",
)


LIBRARY_TABLES: tuple[tuple[str, str], ...] = (
    ("entries", ENTRIES_DDL),
    ("peptidetoprotein", PEPTIDETOPROTEIN_DDL),
    ("peptidescores", PEPTIDESCORES_DDL),
    ("proteinscores", PROTEINSCORES_DDL),
    ("metadata", METADATA_DDL),
    ("peptidequants", PEPTIDEQUANTS_DDL),
    ("fragmentquants", FRAGMENTQUANTS_DDL),
    ("peptidelocalizations", PEPTIDELOCALIZATIONS_DDL),
    ("retentiontimes", RETENTIONTIMES_DDL),
)

# Whitelist for SQL-injection-safe table-name interpolation in ``_sql.py``.
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
    "ENTRIES_DDL",
    "PEPTIDETOPROTEIN_DDL",
    "PEPTIDESCORES_DDL",
    "PROTEINSCORES_DDL",
    "METADATA_DDL",
    "PEPTIDEQUANTS_DDL",
    "FRAGMENTQUANTS_DDL",
    "PEPTIDELOCALIZATIONS_DDL",
    "RETENTIONTIMES_DDL",
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
