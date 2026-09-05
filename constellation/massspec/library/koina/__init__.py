"""Koina-backed spectral-library prediction.

Koina (koina.wilhelmlab.org) serves fragment-intensity, retention-time
and CCS models behind one Triton/gRPC API. Constellation's other
predictor is EncyclopeDIA's bundled jar, which is HCD-trained and
effectively fixed-NCE; this backend adds CID models, HCD at arbitrary
collision energy, and a choice of RT models.

Three layers, each usable on its own::

    from constellation.massspec.library.koina import predict

    # raw passthrough — Arrow out, nothing else added
    predict("Prosit_2020_intensity_CID",
            {"peptide_sequences": ["LVNELTEFAK"], "precursor_charges": [2]})

``predict_fragments`` adds modseq translation and annotation parsing;
``predict_library`` adds digestion and Library assembly.

``koinapy`` is imported only inside :class:`client.KoinaClient`, so this
package imports cleanly without the ``[ms]`` extra installed — the
failure surfaces as :class:`client.KoinaUnavailableError` at call time.

Measured limits (koina.wilhelmlab.org, 2026-08-03): the Prosit 2020
models accept peptides of 1-30 residues at charge 1-6 carrying only
Carbamidomethyl-C and Oxidation-M, and reject terminal modifications in
every bracket form. ``Prosit_2020_intensity_CID`` declares no collision
energy input and is energy-independent.
"""

from constellation.massspec.library.koina._annotation import (
    parse_koina_annotation,
)
from constellation.massspec.library.koina._modseq import (
    KoinaModSeqError,
    format_koina_modseq,
    parse_koina_modseq,
)
from constellation.massspec.library.koina.api import (
    predict,
    predict_fragments,
    predict_library,
    predict_rt,
)
from constellation.massspec.library.koina.assemble import (
    AssemblyStats,
    assemble_library,
)
from constellation.massspec.library.koina.client import (
    DEFAULT_SERVER,
    ENV_SERVER,
    KoinaError,
    KoinaInputError,
    KoinaModelNotFoundError,
    KoinaUnavailableError,
    resolve_server,
)
from constellation.massspec.library.koina.models import (
    MS2_MODELS,
    RT_MODELS,
    Ms2Model,
    RtModel,
    ms2_model,
    rt_model,
)
from constellation.massspec.library.koina.nce import NCE_CHARGE_FACTORS, adjust_nce

__all__ = [
    "AssemblyStats",
    "DEFAULT_SERVER",
    "ENV_SERVER",
    "KoinaError",
    "KoinaInputError",
    "KoinaModSeqError",
    "KoinaModelNotFoundError",
    "KoinaUnavailableError",
    "MS2_MODELS",
    "Ms2Model",
    "NCE_CHARGE_FACTORS",
    "RT_MODELS",
    "RtModel",
    "adjust_nce",
    "assemble_library",
    "format_koina_modseq",
    "ms2_model",
    "parse_koina_annotation",
    "parse_koina_modseq",
    "predict",
    "predict_fragments",
    "predict_library",
    "predict_rt",
    "resolve_server",
    "rt_model",
]
