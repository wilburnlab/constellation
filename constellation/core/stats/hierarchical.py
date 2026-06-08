"""Hierarchical parameter store + pluggable per-entity parameter providers.

Two reusable pieces for models with a *global* tier (calibrants shared across
all entities) and a *per-entity* tier (one parameter set per peptide, run in a
single vectorized forward):

  * `ParameterTier` — `globals` (scalar / small shared `nn.Parameter`s) +
    `batched` (per-entity `(E, *)` `nn.Parameter`s). Everything is a plain
    `nn.Parameter`, so `core.optim` (DE / LBFGS / Adam) and the VI guide see
    them with no special-casing. `freeze` / `thaw` toggle `requires_grad` by
    name or glob for staged calibration; `to_arrow` / `from_arrow` bridge to
    columnar Parquet (scalar params → scalar columns, vector params → list
    columns).
  * `ParameterProvider` — maps an entity index tensor to its per-entity param
    dict. `FreeParameterProvider` gathers rows from a `ParameterTier` (the
    default: one free value per entity). `EmbeddingParameterProvider` amortizes
    instead — a learned token/index → params map — so per-entity quantities
    can be *seeded as individual values OR learned embeddings* without changing
    the consumer.

`pyarrow` is imported lazily inside the Arrow-bridge methods so the module's
import-time surface stays torch-only.
"""

from __future__ import annotations

import abc
import fnmatch
import math
from typing import Mapping, Sequence

import torch
from torch import nn

__all__ = [
    "ParameterTier",
    "ParameterProvider",
    "FreeParameterProvider",
    "EmbeddingParameterProvider",
]

_ENTITY_COL = "entity_index"

# A batched-parameter spec is either an explicit init tensor of shape (E, *), a
# per-entity shape tuple (zeros-initialized), or an int = a per-entity vector
# length.
BatchedSpec = "torch.Tensor | Sequence[int] | int"
GlobalSpec = "torch.Tensor | float | Sequence[float]"


class ParameterTier(nn.Module):
    """Two-tier parameter store: shared `globals` + per-entity `batched`.

    Parameters
    ----------
    n_entities:
        Number of per-entity rows `E` (e.g. peptides).
    globals:
        `{name: init}` shared parameters. `init` is a scalar, a sequence, or a
        tensor; stored verbatim (any shape).
    batched:
        `{name: spec}` per-entity parameters, each stored as `(E, *)`. `spec`
        is an init tensor whose leading dim is `E`, a per-entity shape tuple
        (`()` → `(E,)`, `(K,)` → `(E, K)`; zeros-init), or an int `K` → `(E, K)`.
    dtype:
        Parameter dtype (float64 by default — `core.stats` convention).
    """

    def __init__(
        self,
        n_entities: int,
        *,
        globals: Mapping[str, "GlobalSpec"] | None = None,
        batched: Mapping[str, "BatchedSpec"] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_entities < 1:
            raise ValueError(f"n_entities must be ≥ 1; got {n_entities}")
        self.n_entities = int(n_entities)
        self._dtype = dtype
        self.globals = nn.ParameterDict()
        self.batched = nn.ParameterDict()

        for name, init in (globals or {}).items():
            self.globals[name] = nn.Parameter(torch.as_tensor(init, dtype=dtype))

        for name, spec in (batched or {}).items():
            self.batched[name] = nn.Parameter(self._batched_init(spec, dtype))

    def _batched_init(self, spec: "BatchedSpec", dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(spec):
            t = spec.to(dtype)
            if t.shape[0] != self.n_entities:
                raise ValueError(
                    f"batched init leading dim {t.shape[0]} != n_entities "
                    f"{self.n_entities}"
                )
            return t.clone()
        if isinstance(spec, int):
            shape: tuple[int, ...] = (spec,)
        else:
            shape = tuple(spec)
        return torch.zeros((self.n_entities, *shape), dtype=dtype)

    # -- staged-calibration freeze / thaw ---------------------------------

    def _named(self):
        for k, p in self.globals.items():
            yield f"globals.{k}", p
        for k, p in self.batched.items():
            yield f"batched.{k}", p

    def _set_requires_grad(self, patterns: tuple[str, ...], flag: bool) -> None:
        pats = patterns or ("*",)
        matched = False
        for name, p in self._named():
            leaf = name.split(".", 1)[1]
            if any(
                fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(leaf, pat)
                for pat in pats
            ):
                p.requires_grad_(flag)
                matched = True
        if not matched:
            raise KeyError(
                f"freeze/thaw pattern(s) {list(pats)} matched no parameter; "
                f"have {[n for n, _ in self._named()]}"
            )

    def freeze(self, *patterns: str) -> None:
        """Set `requires_grad=False` on params matching any pattern (default
        all). Patterns match the full name (`batched.log_c_mz`), the leaf
        (`log_c_mz`), or a glob (`batched.*`, `*c_mz`)."""
        self._set_requires_grad(patterns, False)

    def thaw(self, *patterns: str) -> None:
        """Set `requires_grad=True` on params matching any pattern (default all)."""
        self._set_requires_grad(patterns, True)

    def trainable(self) -> list[nn.Parameter]:
        """The currently-thawed parameters — pass to an optimizer to fit only
        the active stage."""
        return [p for p in self.parameters() if p.requires_grad]

    # -- Arrow bridge -----------------------------------------------------

    def to_arrow(self):
        """`(globals_table, batched_table)`. Globals → a one-row table (scalar
        params as scalar columns, vector params as length-1 list columns of the
        flattened value). Batched → an `E`-row table keyed by `entity_index`
        (scalar-per-entity params as scalar columns, vector-per-entity as list
        columns)."""
        import pyarrow as pa

        gcols: dict[str, pa.Array] = {}
        for name, p in self.globals.items():
            v = p.detach().cpu()
            if v.dim() == 0:
                gcols[name] = pa.array([float(v)], type=pa.float64())
            else:
                gcols[name] = pa.array(
                    [v.reshape(-1).tolist()], type=pa.list_(pa.float64())
                )
        globals_tbl = pa.table(gcols) if gcols else pa.table({})

        bcols: dict[str, pa.Array] = {
            _ENTITY_COL: pa.array(list(range(self.n_entities)), type=pa.int64())
        }
        for name, p in self.batched.items():
            v = p.detach().cpu()
            if v.dim() == 1:
                bcols[name] = pa.array(v.tolist(), type=pa.float64())
            else:
                bcols[name] = pa.array(
                    [row.reshape(-1).tolist() for row in v],
                    type=pa.list_(pa.float64()),
                )
        return globals_tbl, pa.table(bcols)

    @classmethod
    def from_arrow(
        cls,
        globals_table,
        batched_table,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> "ParameterTier":
        """Reconstruct a tier from the two tables produced by `to_arrow`.
        Vector params are restored as 1-D (globals) / 2-D (batched) tensors."""
        n = batched_table.num_rows
        globals_init: dict[str, torch.Tensor] = {}
        for name in globals_table.column_names:
            val = globals_table.column(name)[0].as_py()
            globals_init[name] = torch.tensor(val, dtype=dtype)

        batched_init: dict[str, torch.Tensor] = {}
        for name in batched_table.column_names:
            if name == _ENTITY_COL:
                continue
            vals = batched_table.column(name).to_pylist()
            batched_init[name] = torch.tensor(vals, dtype=dtype)
        return cls(n, globals=globals_init, batched=batched_init, dtype=dtype)


class ParameterProvider(nn.Module, abc.ABC):
    """Maps an entity-index tensor to its per-entity parameter dict
    `{name: (B, *shape)}`. Swapping the provider is how a model trades free
    per-entity values for an amortized embedding without touching its forward."""

    @property
    @abc.abstractmethod
    def names(self) -> list[str]:
        """Names of the per-entity parameters this provider supplies."""

    @abc.abstractmethod
    def forward(self, entity_index: torch.Tensor) -> dict[str, torch.Tensor]:
        """`{name: (len(entity_index), *param_shape)}` for the given entities."""


class FreeParameterProvider(ParameterProvider):
    """The default: gather per-entity rows straight from a `ParameterTier`'s
    `batched` params. One free, independently-optimized value per entity."""

    def __init__(self, tier: ParameterTier) -> None:
        super().__init__()
        self.tier = tier

    @property
    def names(self) -> list[str]:
        return list(self.tier.batched.keys())

    def forward(self, entity_index: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = torch.as_tensor(entity_index, dtype=torch.long)
        return {
            name: p.index_select(0, idx) for name, p in self.tier.batched.items()
        }


class EmbeddingParameterProvider(ParameterProvider):
    """Amortized per-entity params from a learned token/index embedding + a
    per-parameter linear head. The same forward signature as
    `FreeParameterProvider`, so a model can be seeded from free values OR from
    learned embeddings (e.g. peptide-token embeddings) interchangeably."""

    def __init__(
        self,
        n_tokens: int,
        param_shapes: Mapping[str, "Sequence[int] | int"],
        *,
        embed_dim: int = 16,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self._shapes: dict[str, tuple[int, ...]] = {}
        for name, shape in param_shapes.items():
            self._shapes[name] = (shape,) if isinstance(shape, int) else tuple(shape)
        self.embed = nn.Embedding(n_tokens, embed_dim).to(dtype)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(embed_dim, max(1, math.prod(shape))).to(dtype)
                for name, shape in self._shapes.items()
            }
        )

    @property
    def names(self) -> list[str]:
        return list(self._shapes)

    def forward(self, entity_index: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = torch.as_tensor(entity_index, dtype=torch.long)
        e = self.embed(idx)  # (B, embed_dim)
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            out[name] = head(e).reshape(idx.shape[0], *self._shapes[name])
        return out
