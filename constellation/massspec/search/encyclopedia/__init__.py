"""Python wrappers around EncyclopeDIA jar subcommands.

Each wrapper translates a typed Python signature to the jar's CLI flags
and delegates execution to :func:`constellation.thirdparty.jvm.run_jar`.
The wrappers know about EncyclopeDIA's CLI surface (flag names,
required vs optional, default values) — they do not know about jar
discovery, JVM resolution, or subprocess plumbing.

Priority utilities (CLI verb → jar invocation):

==================  =================================================
Constellation CLI   EncyclopeDIA 6.5.15 invocation
==================  =================================================
``search``          (default) ``-i <mzml|dia> -l <library> [...]``
``predict-library`` ``-convert -fastaToJChronologerLibrary``
``process-dia``     ``-convert -processDIA``
``library-export``  ``-libexport``
==================  =================================================

Other v6.5.15 programs (``-walnut``, ``-thesaurus``, ``-xcordia``,
``-scribe``, ``-scribetwo{dda,dia}``, ``-browser``, ``-batch``, the
rest of ``-convert``) are deliberately not wrapped yet — they enter
``encyclopedia.<utility>.py`` on-demand when a lab use case appears.
The authoritative inventory lives in
``docs/plans/encyclopedia-6.5.15-utilities.md``.
"""

from __future__ import annotations

from constellation.massspec.search.encyclopedia._common import (
    SUPPORTED_VERSIONS,
    PtmToggle,
    build_manifest_envelope,
    default_heap_for_input,
    encyclopedia_passthrough_args,
    ptm_toggle_args,
    sha256_file,
    write_manifest,
)
from constellation.massspec.search.encyclopedia.library_export import (
    run_library_export,
)
from constellation.massspec.search.encyclopedia.library_search import (
    run_library_search,
)
from constellation.massspec.search.encyclopedia.predict_library import (
    run_predict_library,
)
from constellation.massspec.search.encyclopedia.process_dia import (
    run_process_dia,
)

__all__ = [
    "PtmToggle",
    "SUPPORTED_VERSIONS",
    "build_manifest_envelope",
    "default_heap_for_input",
    "encyclopedia_passthrough_args",
    "ptm_toggle_args",
    "run_library_export",
    "run_library_search",
    "run_predict_library",
    "run_process_dia",
    "sha256_file",
    "write_manifest",
]
