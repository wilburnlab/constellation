"""Adapters for external binaries/jars/DLLs (not vendored source).

`registry.find(tool_name)` returns a `ToolHandle` resolving (in order):
    1. $CONSTELLATION_<TOOL>_HOME env var
    2. <repo>/third_party/<tool>/current/ symlink
    3. $PATH lookup via `shutil.which` (for conda-installed tools)
    4. raise ToolNotFoundError with install-script hint

Per-tool adapter modules (thermo, encyclopedia, mmseqs2, dorado, ...)
go here as they're wired. Each declares REQUIRED_VERSION_RANGE and
knows how to introspect `tool --version` for compatibility checks.
"""

from constellation.thirdparty.registry import (
    ToolHandle,
    ToolNotFoundError,
    find,
    register,
)

__all__ = ["ToolHandle", "ToolNotFoundError", "find", "register"]
