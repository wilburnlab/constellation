"""Structure + dynamics domain module (Contour replacement).

Unifies static structure (PDB/mmCIF) and MD trajectories (DCD/XTC)
under core.structure.Ensemble. Replaces Contour's pandas atom tables
with PyArrow-backed views. Imports `core` only.

Modules (TODO; scaffolded only):
    schemas          - atom/residue/chain Arrow schemas
    readers/         - pdb, mmcif, dcd, xtc subclasses of core.io.RawReader
    prep             - PDBFixer wrapper (via thirdparty.openmm)
    md               - OpenMM integration for simulation setup/analysis
"""
