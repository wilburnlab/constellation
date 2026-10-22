"""Mass-spectrometry browser modality.

Chromatograms and spectra of converted acquisitions (bundles from
``massspec convert`` or ``massspec encyclopedia import-spectra``), with
what search experiments (``massspec search``) found in them, laid out by
the frontend as a grid of MS levels by chromatogram and spectrum.
The plan is ``docs/plans/ms-browser.md``.

Importing this package registers the ``massspec`` modality and its
kernels. Nothing here needs the ``[viz]`` extras or a domain module at
import time: the router is imported when ``routers()`` is called, and the
domain functions inside the functions that use them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from constellation.viz.modalities import Modality, register_modality
from constellation.viz.modalities.massspec import tracks  # noqa: F401
from constellation.viz.modalities.massspec.requests import (
    MassSpecOpenRequest,
    MassSpecSaveRequest,
)
from constellation.viz.modalities.massspec.session import (
    MassSpecSession,
    inspect_source,
    session_from_saved,
)

#: Every name this modality imports from a domain module, as
#: ``(module, name)``; see ``tests/test_viz_import_boundary.py``.
DOMAIN_IMPORTS = (
    ("constellation.massspec.library.ids", "precursor_id"),
    ("constellation.massspec.peptide.mz", "precursor_mz"),
    ("constellation.massspec.results", "acquisition_bundles"),
    ("constellation.massspec.results", "identifications"),
    ("constellation.massspec.results", "precursor_detail"),
    ("constellation.massspec.results", "rt_mapping"),
    ("constellation.massspec.results", "search_units"),
    ("constellation.massspec.results", "searched_library_spectrum"),
    ("constellation.massspec.results", "trace_targets"),
    ("constellation.massspec.scans", "chromatogram"),
    ("constellation.massspec.scans", "default_tolerance"),
    ("constellation.massspec.scans", "isolation_window_table"),
    ("constellation.massspec.scans", "locate_precursor"),
    ("constellation.massspec.scans", "open_acquisition"),
    ("constellation.massspec.scans", "parent_scan"),
    ("constellation.massspec.scans", "scan_table"),
    ("constellation.massspec.scans", "spectrum"),
    ("constellation.massspec.scans", "xic"),
    ("constellation.massspec.spectra.annotate", "annotate_spectrum"),
    ("constellation.massspec.spectra.annotate", "default_fragment_charge"),
    ("constellation.massspec.spectra.annotate", "ion_ladder"),
    ("constellation.massspec.spectra.annotate", "precursor_isotopes"),
    ("constellation.massspec.spectra.annotate", "theoretical_fragment_targets"),
)


def _open_session(request: MassSpecOpenRequest) -> MassSpecSession:
    return MassSpecSession.open(
        sources=[asdict(source) for source in request.sources],
        label=request.label,
        saved_as=request.saved_as,
    )


def _routers() -> Sequence[Any]:
    from constellation.viz.modalities.massspec.endpoints import router

    return (router,)


MODALITY = register_modality(
    Modality(
        name=MassSpecSession.modality,
        open_request=MassSpecOpenRequest,
        open_session=_open_session,
        inspect_source=inspect_source,
        session_from_saved=session_from_saved,
        save_request=MassSpecSaveRequest,
        routers=_routers,
    )
)
