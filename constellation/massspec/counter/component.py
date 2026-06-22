"""Component co-fit — estimate a channel-overlap component (PR-E) as ONE
multi-progenitor panel, so a shared (co-isobaric) peak is soft-attributed once
across the members instead of double-claimed by independent per-target fits (the
L8/D1 cross-panel reconciliation).

`members[0]` is the **reference**: its grid defines the observation, and every
other member is scored at its own mass defect on that grid via the per-progenitor
m/z centers (PR #79). **No union grid** is built — a member's channels OUTSIDE the
reference grid's extraction window are under-scored (a documented v1 limit;
near-isobaric blends *within* the window are the target case). **No discovery**:
the component already enumerates the known co-isobaric candidates (discovery of an
UNKNOWN extra interferer remains `estimate_panel`'s job for a singleton target).

A 1-member component is exactly `estimate_panel` without the discovery loop.
"""

from __future__ import annotations

from typing import Any, Sequence

from .discover import DiscoverConfig, _panel_bounds, _target_n_from_panel, fit_panel
from .model import CounterObservation, Progenitor
from .orchestrate import seed_peak_from_observation
from .panel import Panel

__all__ = ["estimate_component"]


def estimate_component(
    members: Sequence[Progenitor],
    obs: CounterObservation,
    *,
    config: DiscoverConfig | None = None,
    credible_level: float = 0.95,
    background: bool = False,
    rt_priors_ms: Sequence[float | None] | None = None,
    mu_window_ms: float = 30000.0,
    **fit_kw: Any,
) -> list[dict[str, Any]]:
    """Co-fit a component's `members` as one panel; return one `COUNTER_N_TABLE`
    numeric-field dict per member (in `members` order — the caller maps the index
    back to its `target_id`).

    `members` must share the calibration AND the channel structure (same
    charges × isotopes count, so the additive panel stacks); `members[0]` is the
    reference whose grid `obs` is built on. `rt_priors_ms[i]` (e.g. member `i`'s PSM
    RT) seeds its `μ` AND narrows its `μ` search bound to `± mu_window_ms` around
    that RT — co-eluting same-grid members are otherwise exchangeable (one would
    absorb the whole blend), so anchoring each `μ` near its known RT keeps them
    distinguishable (a search bound, not the unsafe prior-polish). The MAP fit is
    likelihood-only (the panel point-estimate prior is routed to VB-on-panel)."""
    if len(members) == 0:
        raise ValueError("a component needs at least one member")
    n_ch = int(members[0].channel_z.numel())
    if any(int(q.channel_z.numel()) != n_ch for q in members):
        raise ValueError(
            "component members must share the channel count (charges × isotopes) so the "
            "additive panel stacks; got "
            f"{[int(q.channel_z.numel()) for q in members]}"
        )
    config = config or DiscoverConfig()
    panel = Panel(list(members), members[0].calibration, background=background)
    priors = list(rt_priors_ms) if rt_priors_ms is not None else [None] * len(members)
    if len(priors) != len(members):
        raise ValueError(
            f"rt_priors_ms has {len(priors)} entries != {len(members)} members"
        )
    for q, rt_prior in zip(members, priors):
        seed_peak_from_observation(q, obs, rt_prior_ms=rt_prior)
    # Anchor each member's μ near its PSM RT (within the obs window) so a co-eluting
    # same-grid member can't drift/swap and absorb the whole blend.
    bounds = None
    if any(p is not None for p in priors):
        bounds = _panel_bounds(panel, obs)
        rt_lo, rt_hi = float(obs.rt.min()), float(obs.rt.max())
        for i, rt_prior in enumerate(priors):
            # Only narrow when the prior is INSIDE the obs window: a prior beyond the
            # window would clamp to an inverted (lo > hi) interval and break the fit;
            # leave such a member the full default μ bound (it fits where it can).
            if rt_prior is not None and rt_lo <= float(rt_prior) <= rt_hi:
                lo = max(rt_lo, float(rt_prior) - mu_window_ms)
                hi = min(rt_hi, float(rt_prior) + mu_window_ms)
                bounds[f"progenitors.{i}.peak.mu"] = (lo, hi)
    result = fit_panel(
        panel, obs, optimizer=config.fit_optimizer, seed=config.fit_seed,
        bounds=bounds, **fit_kw,
    )
    converged = bool(getattr(result, "converged", True))
    # Each member reports its own N + Laplace CI on its peak block; the "discovered
    # interferers" for member i are the other len-1 co-isobaric members.
    return [
        _target_n_from_panel(
            panel,
            obs,
            target_index=i,
            credible_level=credible_level,
            n_discovered=len(members) - 1,
            converged=converged,
        )
        for i in range(len(members))
    ]
