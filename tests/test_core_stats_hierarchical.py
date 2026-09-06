"""Tests for `core.stats.hierarchical` — ParameterTier + ParameterProviders."""

from __future__ import annotations

import pytest
import torch

from constellation.core.stats import (
    EmbeddingParameterProvider,
    FreeParameterProvider,
    ParameterTier,
)


def _tier(n=4) -> ParameterTier:
    return ParameterTier(
        n,
        globals={"mz_offset": 0.0, "alpha_z": [10.0, 20.0, 30.0]},
        batched={"nu_intensity": (), "c_mz": (3,)},  # scalar + per-isotope vector
    )


# ──────────────────────────────────────────────────────────────────────
# Construction + shapes
# ──────────────────────────────────────────────────────────────────────


def test_construction_shapes() -> None:
    t = _tier(5)
    assert t.n_entities == 5
    assert t.globals["mz_offset"].shape == ()
    assert t.globals["alpha_z"].shape == (3,)
    assert t.batched["nu_intensity"].shape == (5,)
    assert t.batched["c_mz"].shape == (5, 3)
    # all plain nn.Parameters visible to .parameters()
    names = {n for n, _ in t.named_parameters()}
    assert "globals.mz_offset" in names
    assert "batched.c_mz" in names


def test_batched_init_tensor_and_dim_check() -> None:
    init = torch.arange(4 * 2, dtype=torch.float64).reshape(4, 2)
    t = ParameterTier(4, batched={"w": init})
    assert torch.equal(t.batched["w"].detach(), init)
    with pytest.raises(ValueError, match="leading dim"):
        ParameterTier(3, batched={"w": init})  # init has E=4 != 3


def test_rejects_zero_entities() -> None:
    with pytest.raises(ValueError):
        ParameterTier(0)


# ──────────────────────────────────────────────────────────────────────
# freeze / thaw
# ──────────────────────────────────────────────────────────────────────


def test_freeze_thaw_by_tier_and_glob() -> None:
    t = _tier()
    t.freeze()  # everything
    assert t.trainable() == []
    t.thaw("globals.*")
    assert all(p.requires_grad for p in t.globals.values())
    assert not any(p.requires_grad for p in t.batched.values())
    # leaf-name match
    t.freeze()
    t.thaw("c_mz")
    assert t.batched["c_mz"].requires_grad
    assert not t.batched["nu_intensity"].requires_grad
    assert len(t.trainable()) == 1


def test_unmatched_pattern_raises() -> None:
    with pytest.raises(KeyError):
        _tier().freeze("no_such_param")


# ──────────────────────────────────────────────────────────────────────
# Arrow round-trip
# ──────────────────────────────────────────────────────────────────────


def test_to_from_arrow_roundtrip() -> None:
    t = _tier(4)
    with torch.no_grad():
        t.globals["mz_offset"].fill_(1.25)
        t.batched["nu_intensity"].copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
        t.batched["c_mz"].copy_(torch.arange(12, dtype=torch.float64).reshape(4, 3))
    g_tbl, b_tbl = t.to_arrow()
    assert b_tbl.num_rows == 4
    assert "entity_index" in b_tbl.column_names

    t2 = ParameterTier.from_arrow(g_tbl, b_tbl)
    assert t2.n_entities == 4
    assert float(t2.globals["mz_offset"].detach()) == pytest.approx(1.25)
    assert torch.allclose(t2.globals["alpha_z"].detach(), t.globals["alpha_z"].detach())
    assert torch.allclose(
        t2.batched["nu_intensity"].detach(), t.batched["nu_intensity"].detach()
    )
    assert torch.allclose(t2.batched["c_mz"].detach(), t.batched["c_mz"].detach())
    assert t2.batched["c_mz"].shape == (4, 3)


# ──────────────────────────────────────────────────────────────────────
# Providers
# ──────────────────────────────────────────────────────────────────────


def test_free_provider_gathers_rows_and_flows_grad() -> None:
    t = _tier(4)
    with torch.no_grad():
        t.batched["c_mz"].copy_(torch.arange(12, dtype=torch.float64).reshape(4, 3))
    prov = FreeParameterProvider(t)
    assert set(prov.names) == {"nu_intensity", "c_mz"}
    out = prov(torch.tensor([2, 0]))
    assert out["c_mz"].shape == (2, 3)
    assert torch.equal(out["c_mz"][0], t.batched["c_mz"][2])
    # gradient flows back to the gathered rows
    out["c_mz"].sum().backward()
    grad = t.batched["c_mz"].grad
    assert grad is not None
    assert torch.equal(grad[2], torch.ones(3, dtype=torch.float64))
    assert torch.equal(grad[1], torch.zeros(3, dtype=torch.float64))  # not selected


def test_embedding_provider_shapes_and_grad() -> None:
    prov = EmbeddingParameterProvider(
        n_tokens=10,
        param_shapes={"nu_intensity": (), "c_mz": 3},
        embed_dim=8,
    )
    assert set(prov.names) == {"nu_intensity", "c_mz"}
    out = prov(torch.tensor([0, 1, 2, 2]))
    assert out["nu_intensity"].shape == (4,)
    assert out["c_mz"].shape == (4, 3)
    # same token → identical params (amortization)
    assert torch.allclose(out["c_mz"][2], out["c_mz"][3])
    out["c_mz"].sum().backward()
    assert prov.embed.weight.grad is not None
