"""Abstract graph/tree/network primitives.

Generic over node and edge types. Used for molecular topologies
(``core.structure.topology``), and — once they land — phylogenies
(``Tree[T]`` + Newick hook), spectral-library nearest-neighbor graphs,
and other cross-modality relational structure.

Modules:
    network      - ``Network[NodeT, EdgeT]`` Arrow-backed graph
                   container with neighbor lookup, induced subgraphs,
                   connected components.
    tree         - (TODO) ``Tree[T]`` + Newick hook (deferred until
                   phylogeny work needs it).
"""

from constellation.core.graph.network import Network

__all__ = ["Network"]
