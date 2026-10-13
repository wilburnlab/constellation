"""Validate request data against a stdlib-dataclass model.

Modalities describe their request shapes — a kernel's query, the body
of an open-session request — as plain stdlib dataclasses, because the
modules that define them are imported by ``import constellation.viz``
and that has to work without pydantic. Range constraints ride as
``dataclasses.field(metadata={"ge": 0, ...})`` using pydantic ``Field``
keyword names.

This module is where pydantic comes in: it is only reached from the
FastAPI endpoints, which need the ``[viz]`` extras anyway.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from fastapi.exceptions import RequestValidationError
from pydantic import TypeAdapter, ValidationError


@lru_cache(maxsize=None)
def _adapter(model: type) -> TypeAdapter:
    # Cached per model class; pydantic 2 reads the dataclass fields'
    # ``metadata`` constraints when it builds the validator.
    return TypeAdapter(model)


def validate(model: type, raw: Any, *, where: Literal["query", "body"]) -> Any:
    """Return ``raw`` as an instance of ``model``.

    Raises FastAPI's ``RequestValidationError`` (HTTP 422, with the same
    ``detail`` shape FastAPI produces for its own parameters) when the
    data does not fit. ``where`` prefixes each error location, as FastAPI
    does. Keys the model does not declare are ignored.
    """
    try:
        return _adapter(model).validate_python(raw)
    except ValidationError as exc:
        raise RequestValidationError(
            [
                {**err, "loc": (where, *err["loc"])}
                for err in exc.errors(include_url=False)
            ]
        ) from exc
