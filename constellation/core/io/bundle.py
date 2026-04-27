"""Multi-file container abstraction for raw-format reads.

Vendor formats often present a single logical run as either a zip
archive (Agilent OpenLab ``.dx``, Bruker ``.tdf`` siblings, ...) or a
directory of named-by-convention companions (Agilent Fragment Analyzer
``.raw`` plus ``*.txt`` / ``*.ANAI`` / ``*.current``, Thermo ``.raw``
sometimes shipping with separate method exports).

``Bundle`` gives readers a uniform ``open(name) -> bytes`` /
``members() -> list[str]`` view over either layout. Concrete subclasses:

    OpcBundle  — zip archive, members keyed by archive entry name.
    DirBundle  — directory, members keyed by filename relative to the
                 primary file's parent dir.

``Bundle.from_path`` dispatches on disk shape (zip-by-magic, directory,
or single file — the last collapses to a degenerate ``DirBundle`` with
zero companions so reader code paths stay uniform).
"""

from __future__ import annotations

import zipfile
from abc import ABC, abstractmethod
from pathlib import Path


class Bundle(ABC):
    """Read-only view over a primary file plus its companions."""

    def __init__(self, path: Path):
        self.path = path

    @abstractmethod
    def members(self) -> list[str]:
        """Names of all readable members (primary + companions)."""

    @abstractmethod
    def open(self, name: str) -> bytes:
        """Return the raw bytes of member ``name``. Raises ``KeyError`` if absent."""

    def has(self, name: str) -> bool:
        return name in self.members()

    @classmethod
    def from_path(cls, path: Path | str) -> "Bundle":
        """Pick the right concrete bundle for ``path``.

        - Zip archives (``.dx``, ``.zip``, anything with the local-file
          header magic) → ``OpcBundle``.
        - Directories → ``DirBundle`` rooted at the directory.
        - Single regular files → ``DirBundle`` rooted at the file's
          parent directory, with the file as the primary entry.
        """
        p = Path(path).resolve()
        if p.is_dir():
            return DirBundle(p)
        if p.is_file() and zipfile.is_zipfile(p):
            return OpcBundle(p)
        if p.is_file():
            return DirBundle(p.parent, primary=p.name)
        raise FileNotFoundError(f"bundle path does not exist: {p}")


class OpcBundle(Bundle):
    """Zip-archive bundle (Open Packaging Convention or generic zip).

    Members are read into an in-memory dict on construction. For very
    large archives use ``lazy=True`` to read members on demand instead.
    """

    def __init__(self, path: Path, *, lazy: bool = False):
        super().__init__(Path(path))
        self._lazy = lazy
        self._members: dict[str, bytes] = {}
        self._namelist: list[str] = []
        with zipfile.ZipFile(self.path) as z:
            self._namelist = list(z.namelist())
            if not lazy:
                self._members = {name: z.read(name) for name in self._namelist}

    def members(self) -> list[str]:
        return list(self._namelist)

    def open(self, name: str) -> bytes:
        if name in self._members:
            return self._members[name]
        if name not in self._namelist:
            raise KeyError(f"member {name!r} not in {self.path.name}")
        # lazy path
        with zipfile.ZipFile(self.path) as z:
            data = z.read(name)
        self._members[name] = data
        return data


class DirBundle(Bundle):
    """Directory bundle — primary file plus same-directory companions.

    ``primary`` defaults to the single file in the directory if there
    is exactly one; otherwise it must be supplied. Members are
    discovered lazily by ``listdir``; ``open`` reads them on demand.
    """

    def __init__(self, path: Path, *, primary: str | None = None):
        super().__init__(Path(path))
        if not self.path.is_dir():
            raise NotADirectoryError(f"DirBundle requires a directory: {self.path}")
        files = [p.name for p in sorted(self.path.iterdir()) if p.is_file()]
        if primary is None:
            if len(files) == 1:
                primary = files[0]
            else:
                raise ValueError(
                    f"DirBundle at {self.path} contains {len(files)} files; "
                    f"specify primary=..."
                )
        if primary not in files:
            raise FileNotFoundError(f"primary {primary!r} not in {self.path}")
        self.primary = primary
        self._files = files

    def members(self) -> list[str]:
        return list(self._files)

    def open(self, name: str) -> bytes:
        if name not in self._files:
            raise KeyError(f"member {name!r} not in {self.path}")
        return (self.path / name).read_bytes()
