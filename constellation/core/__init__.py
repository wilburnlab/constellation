"""Modality-agnostic primitives shared by every domain module.

Inside-core import DAG (strict; mirrors the biological hierarchy from
building blocks to tertiary structure):

    chem -> sequence -> structure -> {stats, graph, nn} -> optim
      |        |             |              |                ^
      +--------+-------------+--------------+--> io <---------+

- `chem` has no sequence/structure awareness.
- `sequence` imports `chem`; never imports `structure`.
- `structure` imports both `chem` and `sequence`.
- `stats`, `graph`, `nn` are downstream math layers.
- `optim` is the furthest downstream (drives `Parametric.fit`).
- `io` is a leaf — format/codec concerns only, never imports other core.
"""
