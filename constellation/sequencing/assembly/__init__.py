"""Assembly module — de novo contigs, polishing, scaffolding, stats.

Public surface:

    Assembly                 container — contigs + scaffolds + stats
    AssemblyReader/Writer    Protocol pair + native ParquetDir form
    HiFiAsmRunner            subprocess wrapper around `hifiasm`
    PolishRunner             Dorado-based polishing with minimap2
    RagTagRunner             reference-guided scaffolding
    assembly_stats           N50 / L50 / GC / BUSCO summaries
"""

from __future__ import annotations

from constellation.sequencing.assembly.assembly import Assembly
from constellation.sequencing.assembly.hifiasm import HiFiAsmRunner
from constellation.sequencing.assembly.io import (
    ASSEMBLY_READERS,
    ASSEMBLY_WRITERS,
    AssemblyReader,
    AssemblyWriter,
    load_assembly,
    register_reader,
    register_writer,
    save_assembly,
)
from constellation.sequencing.assembly.polish import PolishRunner
from constellation.sequencing.assembly.ragtag import RagTagRunner
from constellation.sequencing.assembly.stats import assembly_stats

__all__ = [
    "Assembly",
    "AssemblyReader",
    "AssemblyWriter",
    "ASSEMBLY_READERS",
    "ASSEMBLY_WRITERS",
    "register_reader",
    "register_writer",
    "save_assembly",
    "load_assembly",
    "HiFiAsmRunner",
    "PolishRunner",
    "RagTagRunner",
    "assembly_stats",
]
