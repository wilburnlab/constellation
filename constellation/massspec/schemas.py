"""Mass-spectrometry-domain Arrow schemas.

Per the generic-vs-modality split in `core.io.schemas`: a schema lands in
`core.io` only if it's universal across ≥2 modalities. Anything narrower —
like fragment-ion ladders, MS peak tables, scan-level acquisition metadata —
lives here. Schemas register with `core.io.schemas.register_schema` at
import so cross-modality consumers can ask the registry "is this an
MSPeakTable?" without importing massspec.

Schemas defined here:

    FRAGMENT_ION_TABLE              one row per (peptide, position, ion_type, charge, loss)
    MS_PEAK_TABLE                      one row per observed peak (scan-keyed; ports cartographer mzpeak)
    SCAN_METADATA_TABLE             one row per scan (filter string, TIC, IIT, promoted Thermo trailer)
    ACQUISITION_METADATA_TABLE      one row per run (instrument identity, tune data)
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

# ──────────────────────────────────────────────────────────────────────
# FragmentIonTable
# ──────────────────────────────────────────────────────────────────────


# Either-or columns: callers populate `peptide_idx` (when ions belong to
# rows of an external peptide table) or `peptide_seq` (standalone use).
# Both nullable so each style is allowed.
#
# `ion_type` is stored as int8 — the IonType enum value (A=0, B=1, ...).
# `loss_id` is the string id from `LOSS_REGISTRY` (e.g., "H2O") or null
# for the no-loss baseline.
FRAGMENT_ION_TABLE: pa.Schema = pa.schema(
    [
        pa.field("peptide_idx", pa.int32(), nullable=True),
        pa.field("peptide_seq", pa.string(), nullable=True),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("ion_type", pa.int8(), nullable=False),
        pa.field("charge", pa.int32(), nullable=False),
        pa.field("loss_id", pa.string(), nullable=True),
        pa.field("mz_theoretical", pa.float64(), nullable=False),
    ],
    metadata={b"schema_name": b"FragmentIonTable"},
)


register_schema("FragmentIonTable", FRAGMENT_ION_TABLE)


# ──────────────────────────────────────────────────────────────────────
# MSPeakTable — one row per observed peak. Ports Cartographer's mzpeak SCHEMA.
# ──────────────────────────────────────────────────────────────────────

# Bumps on any additive column change. v1 = initial Constellation port.
MS_PEAK_SCHEMA_VERSION: int = 1


# Scan / level / rt unified to int32 / int8 / float64 so this table
# joins cleanly against ScanMetadataTable on `scan`. Peak-aux columns
# (resolution/noise/baseline) populate only when the reader receives
# per-peak values from a Thermo CentroidStream; mzML and other backends
# leave them null.
MS_PEAK_TABLE: pa.Schema = pa.schema(
    [
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),  # seconds
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        pa.field("collision_energy", pa.float32(), nullable=True),
        pa.field("ion_mobility", pa.float32(), nullable=True),
        pa.field("faims_cv", pa.float32(), nullable=True),  # V; null if not FAIMS
        pa.field("isolation_lower", pa.float64(), nullable=True),
        pa.field("isolation_upper", pa.float64(), nullable=True),
        pa.field("precursor_scan", pa.float64(), nullable=True),
        pa.field("precursor_mz", pa.float64(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=True),
        # Per-peak Thermo CentroidStream values. Null on mzML / other
        # backends and on profile-mode scans.
        pa.field("peak_resolution", pa.float32(), nullable=True),
        pa.field("peak_noise", pa.float32(), nullable=True),
        pa.field("peak_baseline", pa.float32(), nullable=True),
    ],
    metadata={
        b"schema_name": b"MSPeakTable",
        b"schema_version": str(MS_PEAK_SCHEMA_VERSION).encode("utf-8"),
    },
)


register_schema("MSPeakTable", MS_PEAK_TABLE)


# ──────────────────────────────────────────────────────────────────────
# ScanMetadataTable — one row per scan. Ports cartographer SCAN_METADATA_SCHEMA.
# ──────────────────────────────────────────────────────────────────────

SCAN_METADATA_SCHEMA_VERSION: int = 2


# Mirrors Cartographer's SCAN_METADATA_SCHEMA v5 column-for-column with
# `scan`/`level`/`rt` unified to the dtypes used in MS_PEAK_TABLE so the
# two join cleanly. All trailer-derived columns are nullable: MS1/MS2
# rows populate different subsets, and non-Thermo sources leave the
# trailer block entirely null.
SCAN_METADATA_TABLE: pa.Schema = pa.schema(
    [
        # Core scan identity (matches MS_PEAK_TABLE dtypes)
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("rt", pa.float64(), nullable=False),  # seconds
        pa.field("tic", pa.float64(), nullable=True),
        pa.field("iit", pa.float32(), nullable=True),  # ion injection time, ms
        pa.field("base_peak_mz", pa.float64(), nullable=True),
        pa.field("base_peak_intensity", pa.float64(), nullable=True),
        pa.field("peak_count", pa.uint32(), nullable=True),
        pa.field("filter_string", pa.string(), nullable=True),
        # Mass analyzer (FTMS, ITMS, ...) parsed from the filter string. Set on
        # every scan including MS1; the discriminator that separates, e.g., a
        # CID35 ion-trap MS2 from an HCD35 Orbitrap MS2 at identical NCE.
        pa.field("analyzer", pa.string(), nullable=True),
        pa.field("collision_energy", pa.float32(), nullable=True),
        # Primary dissociation method (hcd, cid, etd, ...) parsed from the filter
        # string; null for MS1. Supplemental activations (EThcD/ETciD) keep their
        # full sequence in filter_string — only the leading method lands here.
        pa.field("activation_type", pa.string(), nullable=True),
        pa.field("faims_cv", pa.float32(), nullable=True),
        pa.field("isolation_lower", pa.float64(), nullable=True),
        pa.field("isolation_upper", pa.float64(), nullable=True),
        pa.field("precursor_scan", pa.float64(), nullable=True),
        pa.field("precursor_mz", pa.float64(), nullable=True),
        pa.field("precursor_charge", pa.int8(), nullable=True),
        # Master-scan number reported directly by the Thermo trailer.
        # Kept distinct from `precursor_scan` so consumers can tell
        # "Thermo said this" from "we inferred this" in multi-source files.
        pa.field("master_scan", pa.float64(), nullable=True),
        # Raw trailer-reported monoisotopic m/z (before the reader
        # elected to use it as precursor_mz).
        pa.field("monoisotopic_mz_override", pa.float64(), nullable=True),
        # Full verbatim trailer extras dump. Keys preserved as Thermo
        # writes them (including trailing colon); the catch-all for
        # firmware-specific keys not promoted to typed columns.
        pa.field(
            "trailer_extras",
            pa.map_(pa.string(), pa.string()),
            nullable=True,
        ),
        # ─── Thermo trailer — promoted columns ────────────────────────
        # Ordered to match the `_TRAILER_FIELDS` table in
        # `massspec.io.thermo._trailer` so the schema reads top to
        # bottom in the same order the trailer sections appear on the
        # instrument.
        #
        # Scan identity & housekeeping
        pa.field("scan_description", pa.string(), nullable=True),
        pa.field("scan_node", pa.string(), nullable=True),
        pa.field("scan_segment", pa.uint32(), nullable=True),
        pa.field("scan_event", pa.uint32(), nullable=True),
        pa.field("master_index", pa.uint32(), nullable=True),
        pa.field("targeted_compound_index", pa.float32(), nullable=True),
        # Injection / AGC
        pa.field("multiple_injection", pa.string(), nullable=True),
        pa.field("multi_inject_info", pa.string(), nullable=True),
        pa.field("multi_inject_windows_mz", pa.string(), nullable=True),
        pa.field("agc_enabled", pa.bool_(), nullable=True),
        pa.field("microscan_count", pa.uint32(), nullable=True),
        pa.field("max_ion_time_ms", pa.float32(), nullable=True),
        pa.field("agc_target", pa.uint64(), nullable=True),
        pa.field("agc_fill", pa.float32(), nullable=True),
        pa.field("agc_target_adjust", pa.float32(), nullable=True),
        pa.field("agc_strategy", pa.uint32(), nullable=True),
        pa.field("agc_history_length", pa.uint32(), nullable=True),
        pa.field("agc_references", pa.string(), nullable=True),
        pa.field("agc_transm_quad_enabled", pa.bool_(), nullable=True),
        pa.field("agc_transm_quad_factors", pa.string(), nullable=True),
        pa.field("agc_transm_car_enabled", pa.bool_(), nullable=True),
        pa.field("agc_transm_car_factor", pa.float32(), nullable=True),
        pa.field("agc_transm_funnel_enabled", pa.bool_(), nullable=True),
        pa.field("agc_transm_funnel_factors", pa.string(), nullable=True),
        pa.field("agc_ps_mode", pa.uint32(), nullable=True),
        pa.field("agc_ps_diag", pa.uint32(), nullable=True),
        pa.field("agc_diag_1", pa.float32(), nullable=True),
        pa.field("agc_diag_2", pa.float32(), nullable=True),
        pa.field("ps_inj_time_ms", pa.float32(), nullable=True),
        pa.field("pre_accumulation_time_ms", pa.float32(), nullable=True),
        pa.field("expected_pre_accumulation_time_ms", pa.float32(), nullable=True),
        pa.field("split_lens_injection_time_ms", pa.float32(), nullable=True),
        pa.field("injection_t0", pa.float32(), nullable=True),
        pa.field("t0_flp", pa.float32(), nullable=True),
        # Peak stats (orthogonal to ScanStats-derived `peak_count` above)
        pa.field("centroid_peak_count", pa.uint32(), nullable=True),
        pa.field("raw_peak_count_ch1", pa.uint32(), nullable=True),
        pa.field("raw_peak_count_ch2", pa.uint32(), nullable=True),
        pa.field("peak_picking_enabled", pa.bool_(), nullable=True),
        pa.field("isotope_envelope_fit_error", pa.float32(), nullable=True),
        # Precursor activation (MS2-populated)
        pa.field("ms2_isolation_width", pa.float64(), nullable=True),
        pa.field("ms2_isolation_offset", pa.float64(), nullable=True),
        pa.field("hcd_energy", pa.float32(), nullable=True),
        pa.field("hcd_energy_v", pa.float32(), nullable=True),
        pa.field("hcd_abs_offset", pa.float32(), nullable=True),
        pa.field("source_cid_ev", pa.float32(), nullable=True),
        # Mass calibration (per-scan corrections; FT A/B/C polynomial)
        pa.field("ft_resolution", pa.uint32(), nullable=True),
        pa.field("ft_conversion_a", pa.float64(), nullable=True),
        pa.field("ft_conversion_b", pa.float64(), nullable=True),
        pa.field("ft_conversion_c", pa.float64(), nullable=True),
        pa.field("analyzer_temperature_c", pa.float32(), nullable=True),
        pa.field("temperature_comp_ppm", pa.float32(), nullable=True),
        pa.field("rf_comp_ppm", pa.float32(), nullable=True),
        pa.field("space_charge_comp_ppm", pa.float32(), nullable=True),
        pa.field("resolution_comp_ppm", pa.float32(), nullable=True),
        # Lock mass / internal calibrants
        pa.field("lock_mass_count", pa.uint8(), nullable=True),
        pa.field("lock_mass_1_mz", pa.float64(), nullable=True),
        pa.field("lock_mass_2_mz", pa.float64(), nullable=True),
        pa.field("lock_mass_3_mz", pa.float64(), nullable=True),
        pa.field("lock_mass_search_window_ppm", pa.float32(), nullable=True),
        pa.field("lock_mass_search_window_mmu", pa.float32(), nullable=True),
        pa.field("lock_mass_found_count", pa.uint8(), nullable=True),
        pa.field("last_locking_seconds", pa.float32(), nullable=True),
        pa.field("lock_mass_correction_ppm", pa.float32(), nullable=True),
        # Astral-specific stabilization
        pa.field("astral_stabilization_ppm", pa.float32(), nullable=True),
        pa.field("last_astral_stabilization_seconds", pa.float32(), nullable=True),
        # Ion optics
        pa.field("funnel_rf_level", pa.float32(), nullable=True),
        pa.field("injection_optics_settling_time_ms", pa.float32(), nullable=True),
        pa.field(
            "applied_injection_optics_settling_time_ms", pa.float32(), nullable=True
        ),
        pa.field("quad_settling_time_ms", pa.float32(), nullable=True),
        pa.field("applied_quad_settling_time_ms", pa.float32(), nullable=True),
        pa.field("applied_irm_gradient_delay_time_ms", pa.float32(), nullable=True),
        pa.field("applied_irm_dc_delay_time_ms", pa.float32(), nullable=True),
        pa.field("applied_trap_delay_ms", pa.float32(), nullable=True),
        # Diagnostic / acquisition behaviour
        pa.field("application_mode", pa.string(), nullable=True),
        pa.field("mild_trapping_enabled", pa.bool_(), nullable=True),
        pa.field("used_for_adaptive_rt_alignment", pa.bool_(), nullable=True),
        pa.field("delta_rt_minutes", pa.float32(), nullable=True),
        pa.field("apd_enabled", pa.bool_(), nullable=True),
        pa.field("ot_intens_comp_factor", pa.float32(), nullable=True),
        pa.field("res_dep_intens", pa.float32(), nullable=True),
        pa.field("q_trans_comp", pa.float32(), nullable=True),
        pa.field("prosa_num_f", pa.uint32(), nullable=True),
        pa.field("prosa_comp", pa.float32(), nullable=True),
        pa.field("prosa_scscr", pa.float32(), nullable=True),
        pa.field("raw_ov_ft_t", pa.float64(), nullable=True),
        pa.field("dynamic_rt_shift_minutes", pa.float32(), nullable=True),
        pa.field("lc_fwhm", pa.float32(), nullable=True),
        pa.field("iso_para_r", pa.uint32(), nullable=True),
        pa.field("inj_para_r", pa.uint32(), nullable=True),
        pa.field("analog_in_a_v", pa.float32(), nullable=True),
        pa.field("analog_in_b_v", pa.float32(), nullable=True),
        # FAIMS (faims_cv comes from the filter string; these add presence flags)
        pa.field("faims_attached", pa.bool_(), nullable=True),
        pa.field("faims_voltage_on", pa.bool_(), nullable=True),
    ],
    metadata={
        b"schema_name": b"ScanMetadataTable",
        b"schema_version": str(SCAN_METADATA_SCHEMA_VERSION).encode("utf-8"),
    },
)


register_schema("ScanMetadataTable", SCAN_METADATA_TABLE)


# ──────────────────────────────────────────────────────────────────────
# AcquisitionMetadataTable — one row per run. Ports cartographer ACQUISITION_METADATA_SCHEMA v5.
# ──────────────────────────────────────────────────────────────────────

ACQUISITION_METADATA_SCHEMA_VERSION: int = 1


# Single-row run-level metadata: instrument identity + Thermo
# GetTuneData(0) verbatim (mass-calibration parameter array, spray
# voltages, source temperatures, HCD scaling coefficients). Keeps the
# tune block verbatim now; interpret later (the FT `A` coefficient
# lives somewhere inside it).
ACQUISITION_METADATA_TABLE: pa.Schema = pa.schema(
    [
        pa.field("source_format", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("constellation_version", pa.string(), nullable=False),
        pa.field("schema_version", pa.uint32(), nullable=False),
        pa.field("instrument_model", pa.string(), nullable=True),
        pa.field("instrument_serial", pa.string(), nullable=True),
        pa.field("hardware_version", pa.string(), nullable=True),
        pa.field("software_version", pa.string(), nullable=True),
        pa.field("method_name", pa.string(), nullable=True),
        pa.field("creation_date", pa.string(), nullable=True),  # ISO-8601
        pa.field("analyzers", pa.list_(pa.string()), nullable=True),
        pa.field("n_scans", pa.uint32(), nullable=True),
        pa.field("rt_min_seconds", pa.float32(), nullable=True),
        pa.field("rt_max_seconds", pa.float32(), nullable=True),
        pa.field("profile_mode", pa.bool_(), nullable=True),
        pa.field(
            "tune_data",
            pa.map_(pa.string(), pa.string()),
            nullable=True,
        ),
    ],
    metadata={
        b"schema_name": b"AcquisitionMetadataTable",
        b"schema_version": str(ACQUISITION_METADATA_SCHEMA_VERSION).encode("utf-8"),
    },
)


register_schema("AcquisitionMetadataTable", ACQUISITION_METADATA_TABLE)


# ──────────────────────────────────────────────────────────────────────
# MSPeakIndex — derived, rebuildable mass-sorted projection of MS_PEAK_TABLE
# ──────────────────────────────────────────────────────────────────────

MS_PEAK_INDEX_SCHEMA_VERSION: int = 1


# A derived peak index: MS_PEAK_TABLE re-laid-out as a dataset
# partitioned by `(ms_level, isolation_window)` (the partition keys ride
# in the hive directory path) and sorted by `mz` then `rt` *within* each
# partition, so a query m/z resolves to a contiguous slice of (rt, scan,
# intensity). This is the structure that makes XIC-at-scale (and a future
# spectrum-centric search) cost scale with query count, not n_scans.
#
# Precision is a per-axis, function-driven knob; the index defaults to a
# lossless f64 mirror of the canonical table. `mz`, `rt`, `intensity` are
# each independently downcastable to f32 (`isolation_lower/upper` follow
# `mz`). f32 worst-case ≈ 0.06 ppm relative — far under any matching
# tolerance — and any downcast is recoverable, because MS_PEAK_TABLE
# stays f64. The realized per-axis dtypes are recorded in the build
# manifest's `x.massspec.precision`. The schema declares the lossless f64
# form; the builder casts the selected axes after the sort.
MS_PEAK_INDEX_TABLE: pa.Schema = pa.schema(
    [
        pa.field("mz", pa.float64(), nullable=False),  # within-partition sort key
        pa.field("rt", pa.float64(), nullable=False),  # seconds
        pa.field("scan", pa.int32(), nullable=False),
        pa.field("intensity", pa.float64(), nullable=False),
        # Partition key, carried as a column so a single-file read is
        # self-describing even outside the hive directory layout.
        pa.field("isolation_window_id", pa.int32(), nullable=False),
        pa.field("isolation_lower", pa.float64(), nullable=True),
        pa.field("isolation_upper", pa.float64(), nullable=True),
    ],
    metadata={
        b"schema_name": b"MSPeakIndex",
        b"schema_version": str(MS_PEAK_INDEX_SCHEMA_VERSION).encode("utf-8"),
    },
)


register_schema("MSPeakIndex", MS_PEAK_INDEX_TABLE)


__all__ = [
    "ACQUISITION_METADATA_SCHEMA_VERSION",
    "ACQUISITION_METADATA_TABLE",
    "FRAGMENT_ION_TABLE",
    "MS_PEAK_INDEX_SCHEMA_VERSION",
    "MS_PEAK_INDEX_TABLE",
    "MS_PEAK_SCHEMA_VERSION",
    "MS_PEAK_TABLE",
    "SCAN_METADATA_SCHEMA_VERSION",
    "SCAN_METADATA_TABLE",
]
