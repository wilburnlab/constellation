"""Abstract graph/tree/network primitives.

Generic over node and edge types. Used for molecular topologies
(``core.structure.topology``), NCBI taxonomy
(``core.taxonomy``), and — once they land — phylogenies, spectral-
library nearest-neighbor graphs, and other cross-modality relational
structure.

Modules:
    network      - ``Network[NodeT, EdgeT]`` Arrow-backed graph
                   container with neighbor lookup, induced subgraphs,
                   connected components.
    tree         - ``Tree[T]`` Arrow-backed single-parent tree with
                   lazy children index, ancestors/descendants/lineage/
                   LCA, subtree extraction.
"""

from constellation.core.graph.network import Network
from constellation.core.graph.tree import Tree

__all__ = ["Network", "Tree"]
