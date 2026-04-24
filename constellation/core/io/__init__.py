"""I/O primitives — schema registry, reader ABC, bundle pattern.

Leaf of the core DAG — never imports other core modules. Only
format-level concerns; no sequence/chem/stats/units logic.

Modules (TODO; scaffolded only):
    schemas          - SchemaRegistry + cast_to_schema (forward-compat)
    readers          - RawReader ABC + READER_REGISTRY (suffix dispatch,
                       modality-tagged; domain modules register here)
    bundle           - Bundle(primary, companions=...) for primary-plus-
                       companions I/O (e.g. .mzpeak + .scanmeta + .acqmeta)
"""
