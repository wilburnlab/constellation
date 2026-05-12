"""Import-only smoke test.

Keeps CI green while the scaffold is empty. When a new subpackage is
added, extend this list — the cost is one line, the payoff is an
immediate signal if someone breaks the import DAG.
"""

# ruff: noqa: F401

import constellation
import constellation.core
import constellation.core.chem
import constellation.core.chem.elements
import constellation.core.chem.composition
import constellation.core.chem.isotopes
import constellation.core.chem.modifications
import constellation.core.graph
import constellation.core.graph.network
import constellation.core.io
import constellation.core.io.bundle
import constellation.core.io.readers
import constellation.core.io.schemas
import constellation.core.nn
import constellation.core.optim
import constellation.core.sequence
import constellation.core.sequence.alphabets
import constellation.core.sequence.nucleic
import constellation.core.sequence.ops
import constellation.core.sequence.proforma
import constellation.core.sequence.protein
import constellation.core.stats
import constellation.core.stats.calibration
import constellation.core.stats.distributions
import constellation.core.stats.losses
import constellation.core.stats.parametric
import constellation.core.stats.peaks
import constellation.core.stats.units
import constellation.core.structure
import constellation.core.structure.atoms
import constellation.core.structure.ensemble
import constellation.core.structure.geometry
import constellation.core.structure.selection
import constellation.core.structure.topology
import constellation.chromatography
import constellation.chromatography.readers
import constellation.electrophoresis
import constellation.electrophoresis.readers
import constellation.massspec
import constellation.massspec.acquisitions
import constellation.massspec.annotation
import constellation.massspec.annotation.mzpaf
import constellation.massspec.annotation.usi
import constellation.massspec.library
import constellation.massspec.library.io
import constellation.massspec.library.library
import constellation.massspec.library.schemas
import constellation.massspec.io
import constellation.massspec.io.encyclopedia
import constellation.massspec.peptide
import constellation.massspec.peptide.envelope
import constellation.massspec.peptide.ions
import constellation.massspec.peptide.match
import constellation.massspec.peptide.mz
import constellation.massspec.peptide.neutral_losses
import constellation.massspec.quant
import constellation.massspec.quant.io
import constellation.massspec.quant.quant
import constellation.massspec.quant.schemas
import constellation.massspec.readers
import constellation.massspec.schemas
import constellation.massspec.search
import constellation.massspec.search.io
import constellation.massspec.search.schemas
import constellation.massspec.search.search
import constellation.massspec.tokenize
import constellation.sequencing
import constellation.sequencing.acquisitions
import constellation.sequencing.samples
import constellation.sequencing.schemas
import constellation.sequencing.schemas.alignment
import constellation.sequencing.schemas.assembly
import constellation.sequencing.schemas.quant
import constellation.sequencing.schemas.reads
import constellation.sequencing.schemas.reference
import constellation.sequencing.schemas.signal
import constellation.sequencing.schemas.transcriptome
import constellation.sequencing.quality
import constellation.sequencing.quality.phred
import constellation.sequencing.reference
import constellation.sequencing.reference.fetch
import constellation.sequencing.reference.io
import constellation.sequencing.reference.reference
import constellation.sequencing.transcripts
import constellation.sequencing.transcripts.io
import constellation.sequencing.transcripts.transcripts
import constellation.sequencing.genetic_tools
import constellation.sequencing.assembly
import constellation.sequencing.assembly.assembly
import constellation.sequencing.assembly.hifiasm
import constellation.sequencing.assembly.io
import constellation.sequencing.assembly.polish
import constellation.sequencing.assembly.ragtag
import constellation.sequencing.assembly.stats
import constellation.sequencing.alignments
import constellation.sequencing.alignments.alignments
import constellation.sequencing.alignments.io
import constellation.sequencing.annotation
import constellation.sequencing.annotation.annotation
import constellation.sequencing.annotation.busco
import constellation.sequencing.annotation.io
import constellation.sequencing.annotation.repeats
import constellation.sequencing.annotation.telomeres
import constellation.sequencing.annotation.transcripts
import constellation.sequencing.io
import constellation.sequencing.io.sam_bam
import constellation.sequencing.readers
import constellation.sequencing.readers.bed
import constellation.sequencing.readers.fastx
import constellation.sequencing.readers.gff
import constellation.sequencing.readers.paf
import constellation.sequencing.readers.pod5
import constellation.sequencing.readers.sam_bam
import constellation.sequencing.basecall
import constellation.sequencing.basecall.dorado
import constellation.sequencing.basecall.models
import constellation.sequencing.signal
import constellation.sequencing.signal.normalize
import constellation.sequencing.signal.segment
import constellation.sequencing.signal.models
import constellation.sequencing.signal.models.physical
import constellation.sequencing.align
import constellation.sequencing.align.locate
import constellation.sequencing.align.map
import constellation.sequencing.align.minimap2
import constellation.sequencing.align.pairwise
import constellation.sequencing.quant
import constellation.sequencing.quant.genome_count
import constellation.sequencing.transcriptome
import constellation.sequencing.transcriptome.adapters
import constellation.sequencing.transcriptome.cluster
import constellation.sequencing.transcriptome.consensus
import constellation.sequencing.transcriptome.demux
import constellation.sequencing.transcriptome.network
import constellation.sequencing.transcriptome.orf
import constellation.sequencing.modifications
import constellation.sequencing.modifications.basemod
import constellation.sequencing.projects
import constellation.sequencing.projects.layout
import constellation.sequencing.projects.manifest
import constellation.transcriptome_to_proteome
import constellation.codon
import constellation.structure
import constellation.structure.readers
import constellation.nmr
import constellation.models
import constellation.cli
import constellation.thirdparty
import constellation.data
import constellation.viz
import constellation.viz.install
import constellation.viz.introspect
import constellation.viz.introspect.schema
import constellation.viz.introspect.walk
import constellation.viz.runner
import constellation.viz.runner.lock
import constellation.viz.runner.registry
import constellation.viz.runner.runner
import constellation.viz.tracks
import constellation.viz.tracks.base
import constellation.viz.tracks.coverage_histogram
import constellation.viz.tracks.gene_annotation
import constellation.viz.tracks.reference_sequence
import constellation.viz.tracks.splice_junctions
import constellation.viz.tracks.read_pileup
import constellation.viz.tracks.cluster_pileup
import constellation.viz.server
import constellation.viz.server.session
import constellation.viz.frontend
import constellation.viz.frontend.build
# `viz.server.arrow_stream`, `viz.server.app`, `viz.server.endpoints.*`,
# `viz.raster.*`, and `viz.cli` require fastapi/datashader (gated behind
# the [viz] extras) — exercised in `tests/test_viz_server.py` and
# `tests/test_viz_kernels_extended.py`, which use
# `pytest.importorskip("fastapi")` / `pytest.importorskip("datashader")`.


def test_version_exposed():
    assert hasattr(constellation, "__version__")


def test_thirdparty_public_surface():
    from constellation.thirdparty import ToolHandle, ToolNotFoundError, find, register

    assert callable(find)
    assert callable(register)
    assert ToolHandle is not None
    assert issubclass(ToolNotFoundError, Exception)
