"""Abstract graph/tree/network primitives.

Generic over node and edge types; torch-tensor edge features. Used for
phylogenies, interaction graphs, molecular topologies, spectral-library
nearest-neighbor graphs — wherever cross-modality relational structure
appears.

Modules (TODO; scaffolded only):
    tree             - Tree[T] + Newick hook (IQ-TREE integration later)
    network          - Network[NodeT, EdgeT] container
"""
