"""Sequencing domain module (NanoporeAnalysis replacement).

Signal-to-protein pipeline for long-read cDNA + ancillary sequencing
modalities. Imports `core` only; never cross-imports to other domains.

Modules (TODO; scaffolded only):
    schemas          - pod5 / fastx / alignment Arrow schemas
    readers/         - pod5, fastx, sam_bam, paf subclasses of
                       core.io.RawReader
    quality          - Phred codec (offset-33 ASCII); used by readers.
                       Feature of sequencing, not of DNA — lives here,
                       not in core.io.
    signal           - raw POD5 signal processing (basecaller-adjacent)
    orf              - thin wrapper over core.sequence.nucleic
    basecall         - Dorado integration (via thirdparty)
    align            - edlib/parasail/minimap2 wrappers (via thirdparty)
"""
