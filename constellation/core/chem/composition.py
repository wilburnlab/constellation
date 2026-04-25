"""`Composition` — element-count vector backed by a 1-D int32 tensor.

A Composition is a labeled bag of atoms. The class is plain-Python (NOT
a `torch.Tensor` subclass) so it can be hashed, equality-compared as a
bool, used as a dict key, and only exposes chemistry-meaningful ops.
For batched / GPU work, use the free functions `stack()`, `batched_mass`,
`batched_average_mass` over raw `(B, N_elements)` tensors.

Operators:
    a + b, a - b           element-wise count add/subtract (delta may be negative)
    a * k, k * a           stoichiometric multiply by integer
    a == b                 bool — same counts across all elements

Constructors:
    Composition.zeros()
    Composition.from_dict({"C": 6, "H": 12, "O": 6})
    Composition.from_formula("C6H12O6") / "Ca(OH)2" / "H2O"
    Composition.from_tensor(t)

Properties:
    .mass                  monoisotopic, float64 dot-product internally
    .average_mass          standard-atomic-weight weighted
    .formula               Hill notation: C, H, then alphabetical
    .atoms                 sparse {symbol: count} view (drops zeros)
    .counts                raw torch.Tensor — escape hatch to torch ops
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Iterable

import torch

from constellation.core.chem.atoms import ATOM_SYMBOLS, ATOMS

# ──────────────────────────────────────────────────────────────────────
# Formula parsing
# ──────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_GROUP_RE = re.compile(r"\(([^()]+)\)(\d*)")


def _expand_parens(formula: str) -> str:
    """Repeatedly expand innermost parenthesized groups: `Ca(OH)2` → `CaO2H2`.

    Handles arbitrary nesting because we always replace the *innermost*
    group first (the regex `\\(([^()]+)\\)(\\d*)` only matches groups
    that contain no inner parens).
    """
    while True:
        m = _GROUP_RE.search(formula)
        if m is None:
            return formula
        inner = m.group(1)
        mult = int(m.group(2)) if m.group(2) else 1
        # Multiply each element count inside the group.
        expanded_parts = []
        for sym, count in _TOKEN_RE.findall(inner):
            if not sym:
                continue
            n = int(count) if count else 1
            expanded_parts.append(f"{sym}{n * mult}")
        formula = formula[: m.start()] + "".join(expanded_parts) + formula[m.end() :]


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula into `{symbol: count}`.

    Supports element symbols (uppercase + optional lowercase), integer
    multipliers, and single- or multi-level parenthesized groups.
    Empty string and whitespace are tolerated. Unknown element symbols
    are passed through — call sites that care should validate against
    `ATOMS`.

    Examples:
        parse_formula("H2O")        → {"H": 2, "O": 1}
        parse_formula("C6H12O6")    → {"C": 6, "H": 12, "O": 6}
        parse_formula("Ca(OH)2")    → {"Ca": 1, "O": 2, "H": 2}
        parse_formula("Fe(CN)6")    → {"Fe": 1, "C": 6, "N": 6}
    """
    formula = formula.replace(" ", "")
    if "(" in formula:
        formula = _expand_parens(formula)
    counts: dict[str, int] = {}
    for sym, count in _TOKEN_RE.findall(formula):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(count) if count else 1)
    return counts


def _hill_order(symbols: Iterable[str]) -> list[str]:
    """Hill notation ordering: C first, H second, others alphabetical.

    When carbon is absent, all elements (including H) are alphabetical —
    the canonical Hill convention.
    """
    syms = list(symbols)
    if "C" in syms:
        head = ["C"]
        if "H" in syms:
            head.append("H")
        rest = sorted(s for s in syms if s not in ("C", "H"))
        return head + rest
    return sorted(syms)


# ──────────────────────────────────────────────────────────────────────
# Composition class
# ──────────────────────────────────────────────────────────────────────


_N_ELEMENTS: int = len(ATOM_SYMBOLS)


class Composition:
    """Element-count vector over the full atom table.

    Counts are stored as a 1-D int32 torch tensor of length
    `len(ATOM_SYMBOLS)`, indexed by composition-axis position
    (`ATOMS.index(symbol)`). Subtraction may produce negative counts
    — useful for representing modification *deltas*; use
    `is_physical()` to check before treating a Composition as a real
    chemical formula.
    """

    __slots__ = ("counts",)

    counts: torch.Tensor

    def __init__(self, counts: torch.Tensor):
        if counts.shape != (_N_ELEMENTS,):
            raise ValueError(
                f"counts must have shape ({_N_ELEMENTS},); got {tuple(counts.shape)}"
            )
        if counts.dtype != torch.int32:
            counts = counts.to(torch.int32)
        if counts.device.type != "cpu":
            counts = counts.cpu()
        self.counts = counts.contiguous()

    # ── constructors ──────────────────────────────────────────────
    @classmethod
    def zeros(cls) -> "Composition":
        return cls(torch.zeros(_N_ELEMENTS, dtype=torch.int32))

    @classmethod
    def from_dict(cls, mapping: Mapping[str, int]) -> "Composition":
        counts = torch.zeros(_N_ELEMENTS, dtype=torch.int32)
        for sym, n in mapping.items():
            try:
                idx = ATOMS.index(sym)
            except KeyError:
                raise KeyError(
                    f"unknown element symbol in composition: {sym!r}"
                ) from None
            counts[idx] = int(n)
        return cls(counts)

    @classmethod
    def from_formula(cls, formula: str) -> "Composition":
        return cls.from_dict(parse_formula(formula))

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "Composition":
        return cls(t)

    # ── derived properties ───────────────────────────────────────
    @property
    def mass(self) -> float:
        """Monoisotopic mass. Float64 dot-product internally for precision."""
        return float((self.counts.to(torch.float64) * ATOMS.mass_tensor).sum().item())

    @property
    def average_mass(self) -> float:
        """Standard-atomic-weight-weighted mass.

        Returns NaN if any element with non-zero count lacks a defined
        standard atomic weight (radioactive-only elements). Use `mass`
        (monoisotopic) for compositions containing such elements.
        """
        total = 0.0
        for sym, n in self.atoms.items():
            w = ATOMS[sym].standard_atomic_weight
            if w is None:
                return float("nan")
            total += w * n
        return total

    @property
    def formula(self) -> str:
        """Hill-notation formula string. Empty composition → ''."""
        present = [s for s in ATOM_SYMBOLS if int(self.counts[ATOMS.index(s)]) != 0]
        ordered = _hill_order(present)
        parts: list[str] = []
        for s in ordered:
            n = int(self.counts[ATOMS.index(s)])
            parts.append(s if n == 1 else f"{s}{n}")
        return "".join(parts)

    @property
    def atoms(self) -> dict[str, int]:
        """Sparse `{symbol: count}` view; zero-count elements dropped."""
        return {
            s: int(self.counts[i])
            for i, s in enumerate(ATOM_SYMBOLS)
            if int(self.counts[i]) != 0
        }

    @property
    def total_atoms(self) -> int:
        return int(self.counts.sum().item())

    # ── arithmetic ───────────────────────────────────────────────
    def __add__(self, other: object) -> "Composition":
        if not isinstance(other, Composition):
            return NotImplemented
        return Composition(self.counts + other.counts)

    def __sub__(self, other: object) -> "Composition":
        if not isinstance(other, Composition):
            return NotImplemented
        return Composition(self.counts - other.counts)

    def __mul__(self, k: int) -> "Composition":
        if not isinstance(k, int):
            return NotImplemented
        return Composition(self.counts * k)

    __rmul__ = __mul__

    # ── identity ─────────────────────────────────────────────────
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Composition):
            return NotImplemented
        return bool(torch.equal(self.counts, other.counts))

    def __hash__(self) -> int:
        return hash(tuple(self.counts.tolist()))

    def __repr__(self) -> str:
        f = self.formula
        return f"Composition({f!r})" if f else "Composition()"

    # ── predicates ───────────────────────────────────────────────
    def is_physical(self) -> bool:
        """All counts non-negative — i.e. represents a real formula
        rather than a delta. Modification deltas may be negative."""
        return bool((self.counts >= 0).all().item())


# ──────────────────────────────────────────────────────────────────────
# Free functions for batched / sequence-level work
# ──────────────────────────────────────────────────────────────────────


def stack(comps: Sequence[Composition]) -> torch.Tensor:
    """Stack a sequence of Compositions into a `(B, N_elements)` int32 tensor.

    The sequence layer (`core.sequence.alphabets`) uses this to build
    residue-composition tables in a single allocation.
    """
    if not comps:
        return torch.zeros((0, _N_ELEMENTS), dtype=torch.int32)
    return torch.stack([c.counts for c in comps], dim=0)


def batched_mass(counts: torch.Tensor) -> torch.Tensor:
    """Monoisotopic mass for batched count tensors of shape `(..., N_elements)`.

    Returns `(...)` float64. Any device — the mass tensor is moved to
    `counts.device` once per call.
    """
    if counts.shape[-1] != _N_ELEMENTS:
        raise ValueError(
            f"last dim must be N_elements={_N_ELEMENTS}; got {tuple(counts.shape)}"
        )
    masses = ATOMS.mass_tensor.to(counts.device)
    return (counts.to(torch.float64) * masses).sum(dim=-1)


def batched_average_mass(counts: torch.Tensor) -> torch.Tensor:
    """Standard-atomic-weight mass for batched counts. Same shape contract
    as `batched_mass`. Result is NaN at any batch position whose
    composition contains a non-zero count for an element without a
    defined standard atomic weight (radioactive-only elements)."""
    if counts.shape[-1] != _N_ELEMENTS:
        raise ValueError(
            f"last dim must be N_elements={_N_ELEMENTS}; got {tuple(counts.shape)}"
        )
    weights = torch.tensor(
        [
            0.0
            if ATOMS[s].standard_atomic_weight is None
            else ATOMS[s].standard_atomic_weight
            for s in ATOM_SYMBOLS
        ],
        dtype=torch.float64,
        device=counts.device,
    )
    undefined_mask = torch.tensor(
        [ATOMS[s].standard_atomic_weight is None for s in ATOM_SYMBOLS],
        dtype=torch.bool,
        device=counts.device,
    )
    clean = (counts.to(torch.float64) * weights).sum(dim=-1)
    # Any batch position with a non-zero count for an undefined-weight
    # element gets NaN.
    needs_nan = (counts[..., undefined_mask] != 0).any(dim=-1)
    return torch.where(
        needs_nan,
        torch.tensor(float("nan"), dtype=torch.float64, device=counts.device),
        clean,
    )


__all__ = [
    "Composition",
    "parse_formula",
    "stack",
    "batched_mass",
    "batched_average_mass",
]
