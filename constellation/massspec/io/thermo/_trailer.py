"""Declarative table mapping Thermo trailer keys → ``SCAN_METADATA_TABLE`` columns.

``_TRAILER_FIELDS`` is the canonical list of trailer keys the Thermo
reader promotes to typed scan-metadata columns (~80 fields, from AGC
timing through Astral stabilization corrections). It's ordered by
trailer section so the schema in :mod:`constellation.massspec.schemas`
mirrors the same top-to-bottom layout the instrument exposes.

Each entry carries a tuple of candidate trailer-key strings — firmware
variants (``"Conversion Parameter A:"`` vs ``"FT Conversion A:"``) fan
out across that tuple — plus the destination column name and a
coercion function. Trailer keys retain the trailing colon Thermo writes
them with so the strings round-trip through ``trailer_extras`` (the
verbatim catch-all map) without renormalisation.

Absent keys land as null in the typed columns; the full verbatim
trailer is preserved in ``SCAN_METADATA_TABLE.trailer_extras``
(``map<string, string>``) so firmware-specific keys not promoted here
remain accessible without a schema bump.

Stateless, pure Python — no .NET / DLL dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# Trailer keys whose values drive the "primary" scan-metadata columns
# (precursor_charge, precursor_mz, master_scan, iit) rather than being
# promoted as raw typed columns. Kept named for readability in
# the scan-metadata builder.
TRAILER_KEY_MONOISOTOPIC = "Monoisotopic M/Z:"
TRAILER_KEY_CHARGE = "Charge State:"
TRAILER_KEY_MASTER_SCAN = "Master Scan Number:"
TRAILER_KEY_IIT = "Ion Injection Time (ms):"


# ── coercion helpers ──────────────────────────────────────────────────


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def safe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def safe_bool(x: Any) -> bool | None:
    """Coerce a Thermo on/off-style trailer value to bool.

    Recognises ``"On"/"Off"`` and ``"Yes"/"No"`` (case-insensitive) and
    already-typed Python bools. Any other string (including empty) or
    ``None`` returns ``None`` — columns stay nullable rather than
    defaulting to ``False`` on absence.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("on", "yes", "true", "1"):
        return True
    if s in ("off", "no", "false", "0"):
        return False
    return None


def safe_str(x: Any) -> str | None:
    """Pass-through string coerce: ``None``/empty → ``None``, else stripped str."""
    if x is None:
        return None
    s = str(x).strip()
    return s or None


# ── declarative trailer-field table ───────────────────────────────────


@dataclass(frozen=True)
class TrailerField:
    """One row of the declarative trailer → scan-metadata mapping.

    ``keys`` is a tuple of candidate Thermo trailer-key strings; the
    first present + non-None coerced value wins. Firmware variants fan
    out across that tuple.
    """

    keys: tuple[str, ...]
    column: str
    coerce: Callable[[Any], Any]


_TRAILER_FIELDS: tuple[TrailerField, ...] = (
    # Scan identity & housekeeping
    TrailerField(("Scan Description:",), "scan_description", safe_str),
    TrailerField(("Scan Node:",), "scan_node", safe_str),
    TrailerField(("Scan Segment:",), "scan_segment", safe_int),
    TrailerField(("Scan Event:",), "scan_event", safe_int),
    TrailerField(("Master Index:",), "master_index", safe_int),
    TrailerField(("Targeted Compound:",), "targeted_compound_index", safe_float),
    # Injection / AGC
    TrailerField(("Multiple Injection:",), "multiple_injection", safe_str),
    TrailerField(("Multi Inject Info:",), "multi_inject_info", safe_str),
    TrailerField(
        ("Multi Inject Windows (m/z):",), "multi_inject_windows_mz", safe_str
    ),
    TrailerField(("AGC:",), "agc_enabled", safe_bool),
    TrailerField(("Micro Scan Count:",), "microscan_count", safe_int),
    TrailerField(("Max. Ion Time (ms):",), "max_ion_time_ms", safe_float),
    TrailerField(("AGC Target:",), "agc_target", safe_int),
    TrailerField(("AGC Fill:",), "agc_fill", safe_float),
    TrailerField(("AGC Target Adjust:",), "agc_target_adjust", safe_float),
    TrailerField(("AGC Strategy:",), "agc_strategy", safe_int),
    TrailerField(("AGC History length:",), "agc_history_length", safe_int),
    TrailerField(("AGC References:",), "agc_references", safe_str),
    TrailerField(("AGC Transm Quad:",), "agc_transm_quad_enabled", safe_bool),
    TrailerField(("AGC Transm Quad Factors:",), "agc_transm_quad_factors", safe_str),
    TrailerField(("AGC Transm CAR:",), "agc_transm_car_enabled", safe_bool),
    TrailerField(("AGC Transm CAR Factor:",), "agc_transm_car_factor", safe_float),
    TrailerField(("AGC Transm Funnel:",), "agc_transm_funnel_enabled", safe_bool),
    TrailerField(
        ("AGC Transm Funnel Factors:",), "agc_transm_funnel_factors", safe_str
    ),
    TrailerField(("AGC PS Mode:",), "agc_ps_mode", safe_int),
    TrailerField(("AGC PS Diag:",), "agc_ps_diag", safe_int),
    TrailerField(("AGC Diag 1:",), "agc_diag_1", safe_float),
    TrailerField(("AGC Diag 2:",), "agc_diag_2", safe_float),
    TrailerField(("PS Inj. Time (ms):",), "ps_inj_time_ms", safe_float),
    TrailerField(
        ("Pre-Accumulation Time (ms):",), "pre_accumulation_time_ms", safe_float
    ),
    TrailerField(
        ("Expected Pre-Accumulation Time (ms):",),
        "expected_pre_accumulation_time_ms",
        safe_float,
    ),
    TrailerField(
        ("Split Lens Injection Time (ms):",),
        "split_lens_injection_time_ms",
        safe_float,
    ),
    TrailerField(("Injection t0:",), "injection_t0", safe_float),
    TrailerField(("t0 FLP:",), "t0_flp", safe_float),
    # Peak stats (orthogonal to ScanStats-derived peak_count)
    TrailerField(("Number of Peaks:",), "centroid_peak_count", safe_int),
    TrailerField(("Number of Raw Peaks Ch1:",), "raw_peak_count_ch1", safe_int),
    TrailerField(("Number of Raw Peaks Ch2:",), "raw_peak_count_ch2", safe_int),
    TrailerField(("Peak Picking Select:",), "peak_picking_enabled", safe_bool),
    TrailerField(
        ("Error in isotopic envelope fit:",), "isotope_envelope_fit_error", safe_float
    ),
    # Precursor activation (MS2-populated)
    TrailerField(("MS2 Isolation Width:",), "ms2_isolation_width", safe_float),
    TrailerField(("MS2 Isolation Offset:",), "ms2_isolation_offset", safe_float),
    TrailerField(("HCD Energy:",), "hcd_energy", safe_float),
    TrailerField(("HCD Energy V:",), "hcd_energy_v", safe_float),
    TrailerField(("HCD abs. Offset:",), "hcd_abs_offset", safe_float),
    TrailerField(("Source CID eV:",), "source_cid_ev", safe_float),
    # Mass calibration (per-scan corrections; FT A/B/C polynomial)
    TrailerField(("FT Resolution:",), "ft_resolution", safe_int),
    TrailerField(
        ("Conversion Parameter A:", "FT Conversion A:"), "ft_conversion_a", safe_float
    ),
    TrailerField(
        ("Conversion Parameter B:", "FT Conversion B:"), "ft_conversion_b", safe_float
    ),
    TrailerField(
        ("Conversion Parameter C:", "FT Conversion C:"), "ft_conversion_c", safe_float
    ),
    TrailerField(("Analyzer Temperature:",), "analyzer_temperature_c", safe_float),
    TrailerField(("Temperature Comp. (ppm):",), "temperature_comp_ppm", safe_float),
    TrailerField(("RF Comp. (ppm):",), "rf_comp_ppm", safe_float),
    TrailerField(("Space Charge Comp. (ppm):",), "space_charge_comp_ppm", safe_float),
    TrailerField(("Resolution Comp. (ppm):",), "resolution_comp_ppm", safe_float),
    # Lock mass / internal calibrants
    TrailerField(("Number of Lock Masses:",), "lock_mass_count", safe_int),
    TrailerField(("Lock Mass #1 (m/z):",), "lock_mass_1_mz", safe_float),
    TrailerField(("Lock Mass #2 (m/z):",), "lock_mass_2_mz", safe_float),
    TrailerField(("Lock Mass #3 (m/z):",), "lock_mass_3_mz", safe_float),
    TrailerField(
        ("LM Search Window (ppm):",), "lock_mass_search_window_ppm", safe_float
    ),
    TrailerField(
        ("LM Search Window (mmu):",), "lock_mass_search_window_mmu", safe_float
    ),
    TrailerField(("Number of LM Found:",), "lock_mass_found_count", safe_int),
    TrailerField(("Last Locking (sec):",), "last_locking_seconds", safe_float),
    TrailerField(
        ("LM m/z-Correction (ppm):",), "lock_mass_correction_ppm", safe_float
    ),
    # Astral-specific stabilization
    TrailerField(
        ("Astral Mass Stabilization (ppm):",), "astral_stabilization_ppm", safe_float
    ),
    TrailerField(
        ("Last Astral Stabilization (sec):",),
        "last_astral_stabilization_seconds",
        safe_float,
    ),
    # Ion optics
    TrailerField(("Funnel RF Level:",), "funnel_rf_level", safe_float),
    TrailerField(
        ("Injection Optics Settling Time (ms):",),
        "injection_optics_settling_time_ms",
        safe_float,
    ),
    TrailerField(
        ("Applied Injection Optics Settling Time (ms):",),
        "applied_injection_optics_settling_time_ms",
        safe_float,
    ),
    TrailerField(("Quad Settling Time (ms):",), "quad_settling_time_ms", safe_float),
    TrailerField(
        ("Applied Quad Settling Time (ms):",),
        "applied_quad_settling_time_ms",
        safe_float,
    ),
    TrailerField(
        ("Applied IRM Gradient Delay Time (ms):",),
        "applied_irm_gradient_delay_time_ms",
        safe_float,
    ),
    TrailerField(
        ("Applied IRM DC Delay Time (ms):",),
        "applied_irm_dc_delay_time_ms",
        safe_float,
    ),
    TrailerField(("Applied Trap Delay (ms):",), "applied_trap_delay_ms", safe_float),
    # Diagnostic / acquisition behaviour
    TrailerField(("Application Mode:",), "application_mode", safe_str),
    TrailerField(("Mild Trapping Mode:",), "mild_trapping_enabled", safe_bool),
    TrailerField(
        ("Used for Adaptive RT alignment:",),
        "used_for_adaptive_rt_alignment",
        safe_bool,
    ),
    TrailerField(
        ("DeltaRT (min) = Reference - ThisFile:",), "delta_rt_minutes", safe_float
    ),
    TrailerField(("APD:",), "apd_enabled", safe_bool),
    TrailerField(("OT Intens Comp Factor:",), "ot_intens_comp_factor", safe_float),
    TrailerField(("Res. Dep. Intens:",), "res_dep_intens", safe_float),
    TrailerField(("Q Trans Comp:",), "q_trans_comp", safe_float),
    TrailerField(("PrOSA NumF:",), "prosa_num_f", safe_int),
    TrailerField(("PrOSA Comp:",), "prosa_comp", safe_float),
    TrailerField(("PrOSA ScScr:",), "prosa_scscr", safe_float),
    TrailerField(("RawOvFtT:",), "raw_ov_ft_t", safe_float),
    TrailerField(
        ("Dynamic RT Shift (min):",), "dynamic_rt_shift_minutes", safe_float
    ),
    TrailerField(("LC FWHM parameter:",), "lc_fwhm", safe_float),
    TrailerField(("Iso Para R:",), "iso_para_r", safe_int),
    TrailerField(("Inj Para R:",), "inj_para_r", safe_int),
    TrailerField(("Analog In A (V):",), "analog_in_a_v", safe_float),
    TrailerField(("Analog In B (V):",), "analog_in_b_v", safe_float),
    # FAIMS (faims_cv comes from the filter string — these add presence flags)
    TrailerField(("FAIMS Attached:",), "faims_attached", safe_bool),
    TrailerField(("FAIMS Voltage On:",), "faims_voltage_on", safe_bool),
)


def resolve_trailer_field(trailer: dict, field: TrailerField) -> Any:
    """Look up a single ``TrailerField`` in a trailer dict.

    Walks the field's candidate-key tuple in order; returns the first
    successfully-coerced non-None value, else ``None``.
    """
    for key in field.keys:
        if key in trailer:
            val = field.coerce(trailer[key])
            if val is not None:
                return val
    return None


def promote_trailer(trailer: dict) -> dict[str, Any]:
    """Project a trailer dict through every ``_TRAILER_FIELDS`` entry.

    Returns ``{column_name: coerced_value}`` for every declared field;
    keys absent from the trailer (normal for MS1 trailers missing MS2
    keys and vice versa) land as ``None``.
    """
    return {f.column: resolve_trailer_field(trailer, f) for f in _TRAILER_FIELDS}


def try_number(val: Any) -> Any:
    """Best-effort coerce a trailer string to ``int`` or ``float``, else passthrough.

    Used by :func:`trailer_to_dict` to normalise the raw CommonCore
    label/value pairs into a typed dict before downstream code
    consumes them. Empty strings collapse to ``None``.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    return s


def trailer_to_dict(trailer) -> dict[str, Any]:
    """Convert a CommonCore ``HeaderInformation`` into a Python dict.

    Keys retain the trailing colon Thermo writes them with (matching
    the ``TRAILER_KEY_*`` constants); values are coerced via
    :func:`try_number`.

    ``trailer`` is a .NET ``HeaderInformation`` instance exposing
    ``.Labels`` and ``.Values`` arrays, or ``None``. Returns ``{}`` on
    None.
    """
    if trailer is None:
        return {}
    labels = list(trailer.Labels)
    values = list(trailer.Values)
    return {label: try_number(val) for label, val in zip(labels, values)}


__all__ = [
    "TRAILER_KEY_CHARGE",
    "TRAILER_KEY_IIT",
    "TRAILER_KEY_MASTER_SCAN",
    "TRAILER_KEY_MONOISOTOPIC",
    "TrailerField",
    "_TRAILER_FIELDS",
    "promote_trailer",
    "resolve_trailer_field",
    "safe_bool",
    "safe_float",
    "safe_int",
    "safe_str",
    "trailer_to_dict",
    "try_number",
]
