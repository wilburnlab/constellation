"""Counter output + persisted-parameter schemas.

Three Arrow tables, following the `massspec.quant` schema conventions
(versioned, `schema_name` metadata, float64 for m/z-derived fields,
self-registered with `core.io.schemas`):

    COUNTER_N_TABLE                the deliverable — per (acquisition,
                                   target) integrated ion count `N_total`
                                   with its credible interval, the fitted
                                   peak shape, and provenance flags.
    COUNTER_GLOBAL_CALIBRATION     per-acquisition shared calibrants — the
                                   `GlobalCalibration` state (gain α(z),
                                   m/z offset, per-isotope d_mz, the m/z
                                   power-law α_mz / ν_mz, resolution).
    COUNTER_PEPTIDE_PARAMS         the persisted per-peptide tier — charge
                                   free-energies, ν_I, per-isotope c_mz.

Units (the ms convention — see the model docstrings): `n_total` in ions,
`d_mz` in Da, `mz_offset`/m/z errors in ppm, peak σ/τ in seconds.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema

COUNTER_N_SCHEMA_VERSION: int = 1
COUNTER_CALIBRATION_SCHEMA_VERSION: int = 1
COUNTER_PEPTIDE_PARAMS_SCHEMA_VERSION: int = 1


COUNTER_N_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("target_id", pa.int64(), nullable=False),
        pa.field("modified_sequence", pa.string(), nullable=True),
        # null = charge-summed (the progenitor partitions charge internally).
        pa.field("precursor_charge", pa.int8(), nullable=True),
        # The estimate + its uncertainty (the deliverable):
        pa.field("n_total", pa.float64(), nullable=False),  # integrated ion count
        pa.field("n_total_lo", pa.float64(), nullable=False),  # credible interval
        pa.field("n_total_hi", pa.float64(), nullable=False),
        pa.field("credible_level", pa.float64(), nullable=False),  # e.g. 0.95
        # Fitted HyperEMG shape (so N(t) is reconstructable without re-fitting):
        pa.field("rt_apex", pa.float64(), nullable=False),  # seconds
        pa.field("peak_sigma", pa.float64(), nullable=True),
        pa.field("peak_tau_r", pa.float64(), nullable=True),
        pa.field("peak_tau_l", pa.float64(), nullable=True),
        pa.field("peak_eta", pa.float64(), nullable=True),
        # Provenance:
        pa.field("inference_method", pa.string(), nullable=False),  # "vb"|"map"|...
        pa.field("converged", pa.bool_(), nullable=False),
        pa.field("final_elbo", pa.float64(), nullable=True),  # null for non-VB
        pa.field("n_scans_used", pa.int32(), nullable=False),
        pa.field("iit_corrected", pa.bool_(), nullable=False),
        pa.field("interference_flag", pa.bool_(), nullable=True),  # reserved
    ],
    metadata={
        b"schema_name": b"CounterN",
        b"schema_version": str(COUNTER_N_SCHEMA_VERSION).encode("utf-8"),
    },
)


COUNTER_GLOBAL_CALIBRATION_TABLE: pa.Schema = pa.schema(
    [
        # null acquisition_id = run-agnostic calibration.
        pa.field("acquisition_id", pa.int64(), nullable=True),
        pa.field("analyzer", pa.string(), nullable=True),
        # Gain α(z) = intensity-per-ion [a.u./ion], per the alpha_model switch.
        pa.field("alpha_model", pa.string(), nullable=False),  # linear|linear_origin|per_z
        pa.field("alpha0", pa.float64(), nullable=True),  # intercept (linear)
        pa.field("alpha1", pa.float64(), nullable=True),  # slope (linear/linear_origin)
        pa.field("alpha_z", pa.list_(pa.float64()), nullable=True),  # per-z (per_z)
        # m/z scale calibration.
        pa.field("mz_offset_ppm", pa.float64(), nullable=False),
        pa.field("d_mz_da", pa.list_(pa.float64()), nullable=False),  # per-isotope, Da
        # m/z-error power law σ_mz² = c_mz / N_k^{α_mz}; α_mz, ν_mz global.
        pa.field("mz_precision_exponent", pa.float64(), nullable=False),  # α_mz
        pa.field("nu_mz", pa.float64(), nullable=False),
        # Intensity-variance resolution scaling ρ_R = (R/R_ref)^ρ and R(m/z).
        pa.field("rho_resolution", pa.float64(), nullable=False),
        pa.field("r_ref", pa.float64(), nullable=True),
        pa.field("mz_ref", pa.float64(), nullable=True),
        # Promoted peak-shape hyperpriors (stage-3 calibration; nullable until set).
        pa.field("prior_log_sigma_mean", pa.float64(), nullable=True),
        pa.field("prior_log_sigma_std", pa.float64(), nullable=True),
        pa.field("prior_log_tau_mean", pa.float64(), nullable=True),
        pa.field("prior_log_tau_std", pa.float64(), nullable=True),
    ],
    metadata={
        b"schema_name": b"CounterGlobalCalibration",
        b"schema_version": str(COUNTER_CALIBRATION_SCHEMA_VERSION).encode("utf-8"),
    },
)


COUNTER_PEPTIDE_PARAMS_TABLE: pa.Schema = pa.schema(
    [
        pa.field("acquisition_id", pa.int64(), nullable=False),
        pa.field("target_id", pa.int64(), nullable=False),
        pa.field("peptide_id", pa.int64(), nullable=True),
        pa.field("modified_sequence", pa.string(), nullable=True),
        # Charge free-energies E_z (softmax(−E_z) = charge fractions f_z).
        pa.field("charge_free_energy", pa.list_(pa.float64()), nullable=False),
        pa.field("charges", pa.list_(pa.int32()), nullable=False),
        # Per-peptide intensity Student-t dof.
        pa.field("nu_intensity", pa.float64(), nullable=False),
        # Per-isotope m/z precision constant c_mz [ppm²].
        pa.field("c_mz", pa.list_(pa.float64()), nullable=False),
        # Theoretical (or learned-corrected) isotope fractions.
        pa.field("isotope_fractions", pa.list_(pa.float64()), nullable=True),
    ],
    metadata={
        b"schema_name": b"CounterPeptideParams",
        b"schema_version": str(COUNTER_PEPTIDE_PARAMS_SCHEMA_VERSION).encode("utf-8"),
    },
)


register_schema("CounterN", COUNTER_N_TABLE)
register_schema("CounterGlobalCalibration", COUNTER_GLOBAL_CALIBRATION_TABLE)
register_schema("CounterPeptideParams", COUNTER_PEPTIDE_PARAMS_TABLE)


__all__ = [
    "COUNTER_N_TABLE",
    "COUNTER_GLOBAL_CALIBRATION_TABLE",
    "COUNTER_PEPTIDE_PARAMS_TABLE",
    "COUNTER_N_SCHEMA_VERSION",
    "COUNTER_CALIBRATION_SCHEMA_VERSION",
    "COUNTER_PEPTIDE_PARAMS_SCHEMA_VERSION",
]
