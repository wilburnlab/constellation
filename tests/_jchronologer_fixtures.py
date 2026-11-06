"""A JChronologer bundle for tests without the jar.

``FakeBundle`` serves the real preprocessing files copied under
``tests/data/jchronologer/1.0.10/`` (so tokenizing is the real thing) and
stand-in models (``fake_models()``: small torch functions of their inputs,
deterministic, with the real models' input and output shapes), so the
predictors and the CLI run end to end in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from constellation.massspec.tokenize.jchronologer import tokenizers_from_files

DATA = Path(__file__).parent / "data" / "jchronologer" / "1.0.10"
ROLES = ("chronologer", "cartographer", "electrician", "sculptor")


class FakeBundle:
    """The real preprocessing files, stand-in models (``models[role]``)."""

    version = "1.0.10"
    jar = DATA / "fake-jchronologer-1.0.10.jar"
    jar_sha256 = "0" * 64

    def __init__(self, models: dict | None = None) -> None:
        self.files = {r: next(DATA.glob(f"{r.capitalize()}_*.preprocessing.json")).read_bytes() for r in ROLES}
        self.tokenizers = tokenizers_from_files(self.version, self.files)
        self.models = fake_models() if models is None else models

    def preprocessing(self, role: str) -> dict:
        return json.loads(self.files[role])

    def identity(self, role: str) -> dict:
        return {"backend": "jchronologer", "model": role, "version": self.version, "fake": True}

    def load_model(self, role: str, device: str = "cpu"):
        return self.models[role]


def fake_models() -> dict:
    """Stand-ins with the real models' shapes: outputs that depend on the
    tokens (and charge / energy where the real model takes them)."""

    def weight(tokens: torch.Tensor) -> torch.Tensor:
        return (tokens.to(torch.float32) * torch.arange(1, tokens.shape[1] + 1)).sum(dim=1, keepdim=True)

    def chronologer(tokens):
        return weight(tokens) / 1000.0

    def cartographer(tokens, charge, nce):
        base = torch.linspace(0.0, 1.0, 174).repeat(tokens.shape[0], 1)
        scale = torch.sigmoid(weight(tokens) / 500.0) * (1 + charge.argmax(dim=1, keepdim=True)) * nce
        return torch.remainder(base * scale * 7.0, 1.0)

    def electrician(tokens):
        w = weight(tokens)
        return torch.cat([torch.remainder(w * k, 5.0) for k in (1, 2, 3, 5, 7, 11)], dim=1)

    def sculptor(tokens, charge):
        return torch.tanh(weight(tokens) / 1000.0) + charge.argmax(dim=1, keepdim=True)

    return {"chronologer": chronologer, "cartographer": cartographer, "electrician": electrician, "sculptor": sculptor}
