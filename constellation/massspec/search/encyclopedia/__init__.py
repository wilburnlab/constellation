"""Running the EncyclopeDIA jar.

Two generations live here.

**The general path** (new code uses this):

    runner    -- ``compose_argv`` + ``run_program``: any program of the
                 jar, with every flag checked against the captured help
                 before Java starts
    params    -- the settings Constellation adds meaning to: the
                 peptide-centric / spectrum-centric resolver and
                 ``SearchParameters`` (tolerances with units, ...)
    outputs   -- what a finished run must have left behind
    percolator-- a pre-flight for the Percolator executable the jar runs
    cli_spec  -- the jar's programs and flags as data

``massspec search`` is built on these through
:mod:`constellation.massspec.workflows.encyclopedia`, and
``massspec encyclopedia run`` exposes the runner directly.

**Four typed wrappers**, one per jar program, each translating a Python
signature to flags and delegating to
:func:`constellation.thirdparty.jvm.run_jar`:

==================  =================================================
Wrapper             EncyclopeDIA invocation (pinned >= 6.9.23)
==================  =================================================
``library_search``  (default) ``-i <input> -l <library> [...]``
``predict_library`` ``-convert -fastaToJChronologerLibrary``
``process_dia``     ``-convert -processDIA``
``library_export``  ``-libexport``
==================  =================================================

``predict-library`` and ``library-export`` are still served by theirs.
``library_search`` and ``process_dia`` have no command of their own any
more and are kept because the transcriptome-to-proteome pipeline calls
all four directly. They pass the caller's output directory straight
through, so the caller owns one fact about the jar: a result already
at the output path is its sign that the search is finished, and it
returns that result whatever the settings now say (:mod:`.outputs`).

The jar's programs and flags are not written down by hand: they are
captured from the pinned jar's ``-h`` output into
``constellation/data/encyclopedia_cli.json`` and queried through
:mod:`.cli_spec`. ``tests/test_encyclopedia_cli_spec.py`` checks every
flag the wrappers emit against that snapshot. The survey note is
``docs/plans/encyclopedia-6.9.23-survey.md``.
"""

from __future__ import annotations

# Importing the wrapper subpackage implies you'll be running the
# encyclopedia jar — ensure the thirdparty adapter's ``register()`` has
# fired so ``find("encyclopedia")`` resolves without the caller having
# to import the adapter themselves. (``constellation.cli.__main__``
# already does this transitively for CLI invocations; this line covers
# direct-from-Python use of the wrappers, e.g. notebooks + tests.)
from constellation.thirdparty import encyclopedia as _encyclopedia_adapter  # noqa: F401
from constellation.massspec.search.encyclopedia._common import (
    MINIMUM_ENCYCLOPEDIA_VERSION,
    PtmToggle,
    available_memory_gib,
    build_manifest_envelope,
    default_heap_for_input,
    default_heap_for_system,
    encyclopedia_passthrough_args,
    is_supported_version,
    ptm_toggle_args,
    require_min_encyclopedia,
    sha256_file,
    write_manifest,
)
from constellation.massspec.search.encyclopedia.library_export import (
    build_library_export_args,
    run_library_export,
)
from constellation.massspec.search.encyclopedia.library_search import (
    build_library_search_args,
    find_search_elib,
    run_library_search,
)
from constellation.massspec.search.encyclopedia.predict_library import (
    build_predict_library_args,
    run_predict_library,
)
from constellation.massspec.search.encyclopedia.process_dia import (
    build_process_dia_args,
    run_process_dia,
    single_input_dia_path,
)

__all__ = [
    "MINIMUM_ENCYCLOPEDIA_VERSION",
    "PtmToggle",
    "available_memory_gib",
    "build_library_export_args",
    "build_library_search_args",
    "build_manifest_envelope",
    "build_predict_library_args",
    "build_process_dia_args",
    "default_heap_for_input",
    "default_heap_for_system",
    "encyclopedia_passthrough_args",
    "find_search_elib",
    "is_supported_version",
    "ptm_toggle_args",
    "require_min_encyclopedia",
    "run_library_export",
    "run_library_search",
    "run_predict_library",
    "run_process_dia",
    "single_input_dia_path",
    "sha256_file",
    "write_manifest",
]
