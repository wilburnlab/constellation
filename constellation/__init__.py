"""Constellation — integrative bioinformatics platform.

A shared core of physically grounded primitives (`constellation.core`)
feeding modality-specific domain modules (`massspec`, `sequencing`,
`structure`, `codon`, `nmr`).

Layer-to-layer import rule: `core` → domain modules → `cli`. Domain
modules never cross-import. See CLAUDE.md for the full DAG.
"""

try:
    from constellation._version import version as __version__
except ImportError:  # source tree without a build / source tarball without _version.py
    __version__ = "0.0.0"
