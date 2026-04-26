"""Import-only smoke test.

Keeps CI green while the scaffold is empty. When a new subpackage is
added, extend this list — the cost is one line, the payoff is an
immediate signal if someone breaks the import DAG.
"""

# ruff: noqa: F401

import constellation
import constellation.core
import constellation.core.chem
import constellation.core.chem.atoms
import constellation.core.chem.composition
import constellation.core.chem.isotopes
import constellation.core.chem.modifications
import constellation.core.graph
import constellation.core.io
import constellation.core.nn
import constellation.core.optim
import constellation.core.sequence
import constellation.core.sequence.alphabets
import constellation.core.sequence.nucleic
import constellation.core.sequence.ops
import constellation.core.sequence.protein
import constellation.core.stats
import constellation.core.structure
import constellation.massspec
import constellation.massspec.library
import constellation.massspec.peptide
import constellation.massspec.readers
import constellation.massspec.search
import constellation.sequencing
import constellation.sequencing.readers
import constellation.codon
import constellation.structure
import constellation.structure.readers
import constellation.nmr
import constellation.models
import constellation.cli
import constellation.thirdparty
import constellation.data


def test_version_exposed():
    assert hasattr(constellation, "__version__")


def test_thirdparty_public_surface():
    from constellation.thirdparty import ToolHandle, ToolNotFoundError, find, register

    assert callable(find)
    assert callable(register)
    assert ToolHandle is not None
    assert issubclass(ToolNotFoundError, Exception)
