"""Koina gRPC client — the only module in Constellation that imports
``koinapy``.

Everything else in ``massspec.library.koina`` speaks numpy dicts to this
module, so a future direct-``tritonclient`` implementation (or a locally
hosted Triton instance) is a drop-in replacement rather than a rewrite.
That isolation is load-bearing for a second reason: ``koinapy`` is what
pins Constellation to Python < 3.13, and it is only reachable from here.

The import happens inside ``KoinaClient.__init__``, never at module
scope, so ``import constellation.massspec.library`` keeps working
without the ``[ms]`` extra installed.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import numpy as np

DEFAULT_SERVER = "koina.wilhelmlab.org:443"

#: Environment override for the server endpoint. Deliberately *not*
#: ``CONSTELLATION_KOINA_HOME`` — the project-wide ``_HOME`` convention
#: names a filesystem install root, and a gRPC endpoint is not one.
ENV_SERVER = "CONSTELLATION_KOINA_URL"


class KoinaError(RuntimeError):
    """Base for every Koina-layer failure."""


class KoinaUnavailableError(KoinaError):
    """koinapy is not installed, or the server is unreachable."""


class KoinaModelNotFoundError(KoinaError):
    """The server does not serve a model by that name."""


class KoinaInputError(KoinaError):
    """The supplied columns don't satisfy the model's declared inputs."""


def resolve_server(server: str | None = None) -> str:
    """CLI flag > environment > default."""
    return server or os.environ.get(ENV_SERVER) or DEFAULT_SERVER


class PredictClient(Protocol):
    """Structural type the assembly layer depends on.

    Tests supply a replay-from-fixture implementation of this Protocol,
    which is why nothing downstream may reference ``KoinaClient``
    concretely.
    """

    @property
    def model_inputs(self) -> Mapping[str, Any]: ...

    @property
    def model_outputs(self) -> Mapping[str, Any]: ...

    def predict(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class KoinaClient:
    """Thin wrapper over ``koinapy.Koina``.

    Adds three things the bare client doesn't give us: a resolved server
    with env-var precedence, a typed exception surface, and input
    validation against the model's *server-declared* schema.
    """

    def __init__(
        self,
        model: str,
        *,
        server: str | None = None,
        ssl: bool = True,
    ) -> None:
        self.model = model
        self.server = resolve_server(server)
        try:
            from koinapy import Koina
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise KoinaUnavailableError(
                "koinapy is not installed — install the MS extra with "
                "`pip install 'constellation-bio[ms]'` to use --backend koina"
            ) from exc

        try:
            self._koina = Koina(model, self.server, ssl=ssl)
        except Exception as exc:
            # koinapy raises a bare ValueError for an unknown model and an
            # InferenceServerException when the endpoint is unreachable.
            # Both surface here as opaque tracebacks otherwise.
            if isinstance(exc, ValueError):
                raise KoinaModelNotFoundError(
                    f"server {self.server} does not serve a model named "
                    f"{model!r}"
                ) from exc
            raise KoinaUnavailableError(
                f"could not reach Koina at {self.server}: {exc}. Check the "
                f"network, or point --koina-url / ${ENV_SERVER} at a "
                f"self-hosted instance."
            ) from exc

    @property
    def model_inputs(self) -> Mapping[str, Any]:
        """``{name: (shape, dtype)}`` as declared by the server."""
        return dict(self._koina.model_inputs)

    @property
    def model_outputs(self) -> Mapping[str, Any]:
        """``{name: dtype}`` as declared by the server."""
        return dict(self._koina.model_outputs)

    @property
    def batchsize(self) -> int:
        return int(self._koina.batchsize)

    def validate(self, columns: Iterable[str]) -> None:
        validate_columns(columns, self.model_inputs, model=self.model)

    def predict(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        min_intensity: float = 1e-4,
        mode: str = "semi_async",
    ) -> dict[str, np.ndarray]:
        """Run inference and return the raw ``{name: array}`` response.

        ``df_output=False`` keeps the whole round trip in numpy — no
        pandas is constructed on either leg. It also skips koinapy's
        per-row ``np.repeat`` expansion of the input frame, which is
        O(N x n_fragments) at Python level.

        NOTE: that path also skips koinapy's own ``min_intensity``
        filtering, so padded positions (Prosit emits ``-1``) arrive
        intact and MUST be masked downstream.
        """
        self.validate(arrays.keys())
        try:
            out = self._koina.predict(
                dict(arrays),
                df_output=False,
                mode=mode,
                min_intensity=min_intensity,
                disable_progress_bar=True,
            )
        except Exception as exc:
            raise KoinaError(
                f"Koina inference failed for model {self.model!r} at "
                f"{self.server}: {exc}"
            ) from exc
        return {str(k): np.asarray(v) for k, v in out.items()}


def validate_columns(
    columns: Iterable[str],
    declared: Mapping[str, Any],
    *,
    model: str = "<model>",
) -> None:
    """Check supplied columns against a model's declared inputs.

    Missing declared inputs are an error. **Extra columns are a warning,
    not a silent drop** — koinapy discards undeclared inputs without
    complaint, which is how one ends up predicting a CID sweep at six
    collision energies and getting six byte-identical libraries
    (``Prosit_2020_intensity_CID`` declares no ``collision_energies``).
    """
    supplied = set(columns)
    expected = set(declared)

    missing = expected - supplied
    if missing:
        raise KoinaInputError(
            f"model {model!r} requires input(s) {sorted(missing)} that were "
            f"not supplied; declared inputs are {sorted(expected)}"
        )

    extra = supplied - expected
    if extra:
        warnings.warn(
            f"model {model!r} does not declare input(s) {sorted(extra)}; they "
            f"will be ignored by the server and have NO effect on the "
            f"prediction. Declared inputs are {sorted(expected)}.",
            UserWarning,
            stacklevel=3,
        )


#: Seam for tests — replaced with a fixture-replaying factory so the whole
#: stack above ``client`` runs offline and without koinapy installed.
_CLIENT_FACTORY: Any = KoinaClient


def make_client(model: str, **kwargs: Any) -> PredictClient:
    """Construct a client through the (test-overridable) factory."""
    return _CLIENT_FACTORY(model, **kwargs)


__all__ = [
    "DEFAULT_SERVER",
    "ENV_SERVER",
    "KoinaClient",
    "KoinaError",
    "KoinaInputError",
    "KoinaModelNotFoundError",
    "KoinaUnavailableError",
    "PredictClient",
    "make_client",
    "resolve_server",
    "validate_columns",
]
