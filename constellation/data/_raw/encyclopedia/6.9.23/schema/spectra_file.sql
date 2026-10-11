-- EncyclopeDIA 6.9.23 spectra file (.dia), format 0.8.0
CREATE TABLE fractions ( Fraction int not null, Name string not null, primary key (Fraction) );
CREATE TABLE metadata ( Key string not null, Value string not null, primary key (Key) );
CREATE TABLE precursor ( Fraction int not null, SpectrumName string not null, SpectrumIndex int not null, ScanStartTime float not null, IonInjectionTime float, IsolationWindowLower float not null, IsolationWindowUpper float not null, MassEncodedLength int not null, MassArray blob not null, IntensityEncodedLength int not null, IntensityArray blob not null, IonMobilityArrayEncodedLength int, IonMobilityArray blob, TIC float, primary key (SpectrumIndex) );
CREATE TABLE ranges ( Start float not null, Stop float not null, DutyCycle float not null, NumWindows int, IonMobilityStart float, IonMobilityStop float, RtStart float, RtStop float );
CREATE TABLE spectra ( Fraction int not null, SpectrumName string not null, PrecursorName string, SpectrumIndex int not null, ScanStartTime float not null, IonInjectionTime float, IsolationWindowLower float not null, IsolationWindowTarget float not null, IsolationWindowUpper float not null, PrecursorCharge int not null, MassEncodedLength int not null, MassArray blob not null, IntensityEncodedLength int not null, IntensityArray blob not null, IonMobilityArrayEncodedLength int, IonMobilityArray blob, TIC float, primary key (SpectrumIndex) );
CREATE INDEX "precursor_index_isolation_window_lower" on "precursor" ("IsolationWindowLower" ASC);
CREATE INDEX "precursor_index_isolation_window_upper" on "precursor" ("IsolationWindowUpper" ASC);
CREATE INDEX "precursor_index_scan_start_time" on "precursor" ("ScanStartTime" ASC);
CREATE INDEX "spectra_index_isolation_window_lower" on "spectra" ("IsolationWindowLower" ASC);
CREATE INDEX "spectra_index_isolation_window_upper" on "spectra" ("IsolationWindowUpper" ASC);
CREATE INDEX "spectra_index_scan_start_time_and_windows" on "spectra" ("ScanStartTime","IsolationWindowLower","IsolationWindowUpper" ASC);
