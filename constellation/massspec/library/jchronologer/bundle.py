"""The installed JChronologer: which jar, its members, its models.

``locate_bundle()`` takes the EncyclopeDIA install the registry finds,
checks its version floor, and opens the JChronologer jar its manifest
``Class-Path`` names (never the highest version in ``lib/``: the 6.9.23
install also holds a leftover ``jchronologer-1.0.9.jar`` whose Sculptor
file differs). Members are read from the jar with ``zipfile``; each is
checked against the snapshot ``data/jchronologer_models.json`` when read,
and one whose SHA-256 is not there is refused (``UnknownModelError``),
since the tokenizer and decoders were checked against those files only.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import warnings
import zipfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from constellation.massspec.tokenize.jchronologer import (
    TOKENIZED_MODELS,
    JChronologerTokenizer,
    UnknownModelError,
    load_snapshot,
    tokenizers_from_files,
)

__all__ = ["JChronologerBundle", "locate_bundle", "load_model"]

_POM = "META-INF/maven/org.searlelab/jchronologer/pom.properties"


def _version(archive: zipfile.ZipFile) -> str:
    for line in archive.read(_POM).decode("utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "version":
            return value.strip()
    raise UnknownModelError(f"no version in {_POM}")


@dataclass(frozen=True)
class JChronologerBundle:
    """One JChronologer jar, as the snapshot knows it."""

    jar: Path
    version: str
    jar_sha256: str
    models: dict[str, dict[str, str]] = field(repr=False)

    def _member(self, role: str, kind: str) -> tuple[str, str]:
        if role not in self.models:
            raise UnknownModelError(f"JChronologer {self.version} has no {role} model")
        info = self.models[role]
        return info[f"{kind}_member"], info[f"{kind}_sha256"]

    def read(self, role: str, kind: str = "model") -> bytes:
        """A member's bytes (``kind`` is ``model`` or ``preprocessing``),
        refused if its SHA-256 is not the snapshot's."""
        member, expected = self._member(role, kind)
        with zipfile.ZipFile(self.jar) as archive:
            data = archive.read(member)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise UnknownModelError(
                f"{member} in {self.jar} has SHA-256 {actual}, not the {expected} Constellation was "
                "checked against; rerun scripts/build-jchronologer-models-json.py and "
                "scripts/survey-jchronologer-tokens.py"
            )
        return data

    def preprocessing(self, role: str) -> dict[str, Any]:
        return json.loads(self.read(role, "preprocessing"))

    @cached_property
    def tokenizers(self) -> dict[str, JChronologerTokenizer]:
        return tokenizers_from_files(self.version, {m: self.read(m, "preprocessing") for m in TOKENIZED_MODELS})

    def identity(self, role: str) -> dict[str, Any]:
        info = self.models[role]
        return {
            "backend": "jchronologer",
            "model": role,
            "version": self.version,
            "stem": info["stem"],
            "model_sha256": info["model_sha256"],
            "preprocessing_sha256": info["preprocessing_sha256"],
            "jar": self.jar.name,
            "jar_sha256": self.jar_sha256,
        }

    def load_model(self, role: str, device: str = "cpu"):
        return load_model(self, role, device)


def locate_bundle(jar: Path | str | None = None, *, encyclopedia=None) -> JChronologerBundle:
    """The JChronologer an EncyclopeDIA install loads: ``encyclopedia`` (a
    ``thirdparty.registry.ToolHandle``, as ``doctor`` has), else the one the
    registry finds; or ``jar`` itself."""
    if jar is None:
        from constellation.thirdparty.encyclopedia import MINIMUM_VERSION, jchronologer_jar
        from constellation.thirdparty.registry import find, meets_min_version

        handle = encyclopedia if encyclopedia is not None else find("encyclopedia")
        if not meets_min_version(handle):
            raise UnknownModelError(
                f"EncyclopeDIA {handle.version} at {handle.path} is older than {MINIMUM_VERSION}, "
                "the first to ship JChronologer's models"
            )
        jar = jchronologer_jar(handle.path)
    jar = Path(jar)
    with zipfile.ZipFile(jar) as archive:
        version = _version(archive)
    known = load_snapshot()["versions"].get(version)
    if known is None:
        raise UnknownModelError(
            f"JChronologer {version} ({jar}) is not one Constellation was checked against "
            f"(known: {', '.join(load_snapshot()['versions'])}); run "
            "scripts/build-jchronologer-models-json.py and scripts/survey-jchronologer-tokens.py on it"
        )
    return JChronologerBundle(
        jar=jar,
        version=version,
        jar_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(),
        models=known["models"],
    )


_MODELS: dict[tuple[str, str, str, str], Any] = {}
_MODELS_LOCK = threading.Lock()


def load_model(bundle: JChronologerBundle, role: str, device: str = "cpu"):
    """The TorchScript model, loaded from the jar's bytes once per process
    (per jar, member hash and device)."""
    import torch

    member, sha = bundle._member(role, "model")
    key = (str(bundle.jar), member, sha, device)
    with _MODELS_LOCK:
        model = _MODELS.get(key)
        if model is None:
            with warnings.catch_warnings():
                # torch 2.12 deprecates TorchScript loading; JChronologer ships
                # TorchScript, so there is nothing else to load yet.
                warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.* is deprecated", category=DeprecationWarning)
                model = torch.jit.load(io.BytesIO(bundle.read(role, "model")), map_location=device).eval()
            _MODELS[key] = model
    return model
