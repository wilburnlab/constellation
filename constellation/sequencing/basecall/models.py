"""``DoradoModel`` — Dorado's model argument, checked and passed through.

Dorado takes its model as one positional argument, in one of three forms
(Dorado's "Model selection" docs):

    variant   ``speed[@version][,mod[@version]]...`` — speed ``fast`` /
              ``hac`` / ``sup``; version ``vX[.Y[.Z]]`` (missing trailing
              parts are zero, so ``@v4.3`` is ``@v4.3.0``) or ``latest``;
              e.g. ``sup``, ``hac@v5.0.0``, ``sup@latest,5mCG_5hmCG@v1``.
              Dorado reads the chemistry (pore, kit, sample rate) from the
              POD5 and fetches the matching models.
    name      a full model name, e.g. ``dna_r10.4.1_e8.2_400bps_sup@v5.2.0``.
    path      a model directory already on disk (for offline nodes), e.g.
              ``~/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0``.

Constellation used to take a shorthand of its own (``hac@v4.3+5mC,5hmC``)
and expand it to a full name with the DNA R10.4.1 400 bps chemistry written
in, so RNA004 or 4 kHz data got a DNA model it did not match, partial
versions named models that do not exist, and the advertised example asked
for a ``5hmC`` model there is none of. The argument now goes to Dorado as
written: Dorado resolves it and validates the modification codes, against
its own catalogue, at run time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ``@v5``, ``@v5.2``, ``@v5.2.0`` or ``@latest``. Dorado refuses empty
# internal parts (``@v0..1``) and trailing periods (``@v1.``).
_VERSION = r"@(?:v\d+(?:\.\d+){0,2}|latest)"

_VARIANT_RE = re.compile(
    rf"^(?:fast|hac|sup)(?:{_VERSION})?"
    rf"(?P<mods>(?:,[A-Za-z0-9_]+(?:{_VERSION})?)*)$"
)
_MOD_RE = re.compile(rf"^(?P<code>[A-Za-z0-9_]+)(?:{_VERSION})?$")

# ``dna_r10.4.1_e8.2_400bps_sup@v5.2.0``, ``rna004_130bps_hac@v6.0.0``, or
# a simplex model with a modification model, ``..._hac@v6.0.0_5mC_5hmC@v1``.
_NAME_RE = re.compile(
    r"^(?:dna|rna)[A-Za-z0-9_.]*_(?:fast|hac|sup)@v\d+(?:\.\d+)*"
    r"(?:_[A-Za-z0-9_]+@v\d+(?:\.\d+)*)?$"
)

_FORMS = (
    "a variant (`sup`, `hac@v5.0.0`, `sup@latest,5mCG_5hmCG`), a full model "
    "name (`dna_r10.4.1_e8.2_400bps_sup@v5.2.0`), or a model directory"
)


@dataclass(frozen=True, slots=True)
class DoradoModel:
    """Dorado's model argument: ``spec`` is passed to Dorado verbatim.

    ``kind`` is which of Dorado's three forms ``spec`` is; ``mods`` are the
    modification codes a variant names (``('5mCG_5hmCG',)`` for
    ``sup,5mCG_5hmCG@v1``), kept so a caller can tell that a variant
    already carries modifications.
    """

    spec: str
    kind: Literal["variant", "name", "path"]
    mods: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def parse(cls, text: str) -> "DoradoModel":
        """Check ``text`` is one of Dorado's model forms; never rewrite it.

        A variant or a full name is taken first, so a directory that
        happens to be called ``sup`` is given as ``./sup``.
        """
        spec = text.strip()
        if not spec:  # Path("") is the working directory, not a model
            raise ValueError(f"{text!r} is not a Dorado model: expected {_FORMS}")
        if "+" in spec:
            # A tombstone for the old shorthand, naming its replacement.
            raise ValueError(
                f"{text!r}: `+modifications` was Constellation's own shorthand "
                "and is no longer read. Use Dorado's syntax, with the "
                "modifications after commas (e.g. `hac@v4.3.0,5mC_5hmC`), or "
                "pass them with --modified-bases"
            )
        m = _VARIANT_RE.match(spec)
        if m is not None:
            mods = tuple(
                _MOD_RE.match(token).group("code")  # type: ignore[union-attr]
                for token in m.group("mods").split(",")
                if token
            )
            return cls(spec=spec, kind="variant", mods=mods)
        if _NAME_RE.match(spec):
            return cls(spec=spec, kind="name")
        path = Path(spec).expanduser()
        if path.is_dir():
            return cls(spec=str(path.resolve()), kind="path")
        raise ValueError(f"{text!r} is not a Dorado model: expected {_FORMS}")

    def arguments(self, modified_bases: tuple[str, ...] = ()) -> tuple[str, list[str]]:
        """``(model positional, options for after the positionals)``.

        Modifications go where Dorado's docs put them: inside a variant
        (``hac`` + ``5mCG_5hmCG`` → ``hac,5mCG_5hmCG``); after the
        positionals as ``--modified-bases`` for a full name or a model
        directory (the option reads every token up to the next option, so
        it cannot come first). A variant that already names modifications
        cannot take more: Dorado refuses the two together as mutually
        exclusive, so this raises ``ValueError`` before anything runs.
        """
        mods = tuple(m for m in modified_bases if m)
        if not mods:
            return self.spec, []
        if self.kind == "variant":
            if self.mods:
                raise ValueError(
                    f"the model {self.spec!r} already names modifications "
                    f"({', '.join(self.mods)}), and Dorado refuses "
                    f"--modified-bases ({', '.join(mods)}) beside them; name "
                    "the modifications in one place"
                )
            return ",".join([self.spec, *mods]), []
        return self.spec, ["--modified-bases", *mods]

    def __str__(self) -> str:
        return self.spec


__all__ = ["DoradoModel"]
