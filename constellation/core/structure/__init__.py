"""Macromolecular-structure math — coords, geometry, topology, ensembles.

Imports `core.chem` and `core.sequence` (tertiary structure sits atop
primary sequence). Torch-native; replaces scipy.spatial and pandas
atom-tables used in Contour.

Modules (TODO; scaffolded only):
    coords           - CoordinateFrame (units, reference, PBCs)
    geometry         - Kabsch, centroid, superposition, RMSD (torch)
    topology         - Topology: bonds/angles/dihedrals as Arrow-backed
                       core.graph.network; residues from core.sequence
    ensemble         - Ensemble: N conformers/frames over a topology —
                       unifies NMR models, cryo-EM multi-model, MD frames
"""
