"""Mass-spec track kernels.

Each module holds one kernel, self-registered via ``@register_track``;
importing this package pulls them all in exactly once. Every kernel has
one binding per acquisition of the session.

- ``ms_chromatogram`` — a TIC, base-peak or scan-event trace for one MS
  level (and DIA window), from the scan table.
- ``ms_spectrum``     — the peaks of one scan, annotated with a peptide's
  ions when one is given.
- ``ms_xic``          — extracted-ion chromatograms of given m/z values.
"""

from constellation.viz.modalities.massspec.tracks import (  # noqa: F401
    chromatogram,
    spectrum,
    xic,
)
