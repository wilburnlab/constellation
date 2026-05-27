"""Baseline correction for 1D traces.

Three methods are shipped, splitting along whether the caller has identified
peak-free regions in advance:

Region-based (caller supplies or auto-detects baseline windows):
    polynomial   single low-degree polynomial through baseline regions
    spline       natural cubic spline through baseline regions

Peak-aware (no regions required — algorithm figures out which points
are baseline by iteratively reweighting):
    arpls        asymmetrically reweighted penalized least squares
                 (Baek et al. 2015, Analyst 140:250)

The peak-aware approach is what fixes the NH-region overshoot regression
in protein NMR spectra where outer-anchor splines deviate across the dense
middle region: arPLS does not need pre-specified anchors; the asymmetric
reweighting locates baseline points on its own.

References:
    Baek, S.-J., Park, A., Ahn, Y.-J., & Choo, J. (2015). Baseline
    correction using asymmetrically reweighted penalized least squares
    smoothing. *Analyst*, 140(1), 250-257. DOI: 10.1039/C4AN01061B
    Eilers, P. H. C. (2003). A perfect smoother. *Analytical Chemistry*,
    75(14), 3631-3636.  (Whittaker smoother — the underlying penalized
    least-squares form.)
    Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline correction
    with asymmetric least squares smoothing. *Leiden University Medical
    Centre report*.  (AsLS — predecessor to arPLS.)
"""

from __future__ import annotations

import torch


# ──────────────────────────────────────────────────────────────────────
# Region detection (heuristic; used as a default when the caller does
# not supply regions to the region-based methods)
# ──────────────────────────────────────────────────────────────────────


def detect_baseline_regions(
    spectrum: torch.Tensor,
    noise_factor: float = 3.0,
    min_region_width: int = 10,
) -> list[tuple[int, int]]:
    """Detect contiguous spectral regions likely to be baseline.

    Uses a fixed-threshold heuristic: points whose absolute value is
    below `noise_factor * estimated_noise_std` are treated as baseline
    candidates, and runs of at least `min_region_width` consecutive
    candidates become regions. Noise standard deviation is estimated
    from the trailing 10 % of the spectrum (typically peak-free for ¹H
    NMR with downfield-aliphatic ordering; user-supplied regions are
    more reliable for unusual spectra).

    Returned regions use Python half-open `[start, end)` indexing.

    Note: peak detection and baseline-region detection are dual problems
    — any peak picker can produce baseline regions by complement.

    Parameters
    ----------
    spectrum : torch.Tensor
        Real-valued 1D tensor (absorption-mode spectrum, post-phase-correction).
    noise_factor : float
        Threshold multiplier on estimated noise σ. Default 3.0.
    min_region_width : int
        Minimum contiguous baseline points to count as a region. Default 10.

    Returns
    -------
    list of (start, end) tuples
        Each tuple is a half-open index interval `[start, end)`.
    """
    spec = spectrum.detach().to(torch.float64)
    n = int(spec.shape[-1])

    tail = spec[int(0.9 * n):]
    noise_std = float(torch.std(tail))
    threshold = noise_factor * noise_std

    is_baseline = torch.abs(spec) < threshold

    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    for i in range(n):
        flag = bool(is_baseline[i].item())
        if flag and not in_region:
            start = i
            in_region = True
        elif not flag and in_region:
            if i - start >= min_region_width:
                regions.append((start, i))
            in_region = False
    if in_region and n - start >= min_region_width:
        regions.append((start, n))

    return regions


# ──────────────────────────────────────────────────────────────────────
# Region-based methods (polynomial, natural cubic spline)
# ──────────────────────────────────────────────────────────────────────


def _gather_region_points(
    spectrum: torch.Tensor,
    regions: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the index/value pairs from a list of baseline regions."""
    if not regions:
        raise ValueError(
            "No baseline regions provided or detected. Supply `regions=` "
            "manually or relax `noise_factor` for auto-detection."
        )
    xs, ys = [], []
    for start, end in regions:
        xs.append(torch.arange(start, end, dtype=torch.float64))
        ys.append(spectrum[start:end].to(torch.float64))
    return torch.cat(xs), torch.cat(ys)


def polynomial(
    spectrum: torch.Tensor,
    regions: list[tuple[int, int]] | None = None,
    degree: int = 4,
    noise_factor: float = 3.0,
) -> torch.Tensor:
    """Subtract a polynomial baseline fitted through baseline regions.

    The fit uses normalized indices in `[-1, 1]` for numerical stability
    at moderate degrees; coefficients are recovered by torch-native
    least squares (`torch.linalg.lstsq` on a Vandermonde matrix).

    Parameters
    ----------
    spectrum : torch.Tensor
        Real-valued 1D tensor (absorption-mode spectrum).
    regions : list of (int, int) or None
        Index ranges `[start, end)` of baseline-only regions. If `None`,
        auto-detected via `detect_baseline_regions`.
    degree : int
        Polynomial degree. Typical range 3-6. Too low cannot follow
        complex curvature; too high oscillates between anchor points
        (Runge's phenomenon). Default 4.
    noise_factor : float
        Forwarded to `detect_baseline_regions` if `regions is None`.

    Returns
    -------
    torch.Tensor
        Baseline-corrected spectrum, same shape and dtype as input.
    """
    if regions is None:
        regions = detect_baseline_regions(spectrum, noise_factor=noise_factor)

    n = int(spectrum.shape[-1])
    x_ctrl, y_ctrl = _gather_region_points(spectrum, regions)

    # Normalize indices to [-1, 1] for conditioning of the Vandermonde matrix.
    x_norm = (x_ctrl / (n - 1)) * 2.0 - 1.0
    V = torch.stack(
        [x_norm**k for k in range(degree + 1)], dim=1
    )  # shape (n_pts, degree+1)
    coeffs = torch.linalg.lstsq(V, y_ctrl).solution

    x_full = torch.arange(n, dtype=torch.float64)
    x_full_norm = (x_full / (n - 1)) * 2.0 - 1.0
    V_full = torch.stack([x_full_norm**k for k in range(degree + 1)], dim=1)
    baseline = V_full @ coeffs

    return spectrum - baseline.to(spectrum.dtype)


def _natural_cubic_spline(
    x_ctrl: torch.Tensor,
    y_ctrl: torch.Tensor,
    x_eval: torch.Tensor,
) -> torch.Tensor:
    """Natural cubic spline through `(x_ctrl, y_ctrl)`, evaluated at `x_eval`.

    Natural boundary conditions (M[0] = M[N] = 0) imply linear
    extrapolation beyond the outermost knots — the right behaviour for
    baseline correction. Tridiagonal system for interior moments solved
    via the Thomas algorithm (O(N) forward elimination + back-substitution).
    Implemented torch-natively; no scipy dependency for this path.
    """
    n = int(x_ctrl.shape[-1]) - 1  # number of intervals
    f64 = torch.float64

    if n == 0:
        return torch.full_like(x_eval, float(y_ctrl[0]))
    if n == 1:
        slope = (y_ctrl[1] - y_ctrl[0]) / (x_ctrl[1] - x_ctrl[0])
        return y_ctrl[0] + slope * (x_eval - x_ctrl[0])

    h = torch.diff(x_ctrl).to(f64)
    y = y_ctrl.to(f64)

    size = n - 1  # interior moments
    diag = (2.0 * (h[:-1] + h[1:])).clone()
    off = h[1:-1].to(f64).clone()
    rhs = (
        6.0
        * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    ).clone()

    # Thomas — forward sweep
    for i in range(1, size):
        m = off[i - 1] / diag[i - 1]
        diag[i] = diag[i] - m * off[i - 1]
        rhs[i] = rhs[i] - m * rhs[i - 1]

    # Back substitution
    M_int = torch.zeros(size, dtype=f64)
    M_int[-1] = rhs[-1] / diag[-1]
    for i in range(size - 2, -1, -1):
        M_int[i] = (rhs[i] - off[i] * M_int[i + 1]) / diag[i]

    M = torch.cat(
        [torch.zeros(1, dtype=f64), M_int, torch.zeros(1, dtype=f64)]
    )

    idx = torch.searchsorted(x_ctrl, x_eval, right=True) - 1
    idx = torch.clamp(idx, 0, n - 1)

    xi = x_ctrl[idx]
    xip1 = x_ctrl[idx + 1]
    hi = h[idx]
    Mi = M[idx]
    Mip1 = M[idx + 1]
    yi = y[idx]
    yip1 = y[idx + 1]

    dx_left = xip1 - x_eval
    dx_right = x_eval - xi

    return (
        (Mi / (6.0 * hi)) * dx_left**3
        + (Mip1 / (6.0 * hi)) * dx_right**3
        + (yi / hi - Mi * hi / 6.0) * dx_left
        + (yip1 / hi - Mip1 * hi / 6.0) * dx_right
    )


def spline(
    spectrum: torch.Tensor,
    regions: list[tuple[int, int]] | None = None,
    noise_factor: float = 3.0,
) -> torch.Tensor:
    """Subtract a natural cubic spline baseline fitted through baseline regions.

    A natural cubic spline is built from piecewise cubic segments joined
    smoothly at knots, with second derivative zero at the two endpoints
    (so extrapolation beyond the outermost knots is linear, not wildly
    curved — appropriate for baseline correction).

    Beware the failure mode that motivated `arpls`: with only outer
    anchor regions on a spectrum that has dense peaks in the middle, the
    spline interpolates a straight line across the unanchored interior,
    which can deviate significantly from the true baseline. Use `arpls`
    when no good interior anchors exist.

    Parameters
    ----------
    spectrum : torch.Tensor
        Real-valued 1D tensor (absorption-mode spectrum).
    regions : list of (int, int) or None
        Baseline-only index ranges. Auto-detected if `None`.
    noise_factor : float
        Forwarded to `detect_baseline_regions` if `regions is None`.

    Returns
    -------
    torch.Tensor
        Baseline-corrected spectrum, same shape and dtype as input.
    """
    if regions is None:
        regions = detect_baseline_regions(spectrum, noise_factor=noise_factor)

    n = int(spectrum.shape[-1])
    x_ctrl, y_ctrl = _gather_region_points(spectrum, regions)

    x_full = torch.arange(n, dtype=torch.float64)
    baseline = _natural_cubic_spline(x_ctrl, y_ctrl, x_full)

    return spectrum - baseline.to(spectrum.dtype)


# ──────────────────────────────────────────────────────────────────────
# Peak-aware: arPLS (Baek et al. 2015) — pure-torch via banded Cholesky
#
# The arPLS inner system  (W + λ DᵀD) z = W y  is symmetric pentadiagonal
# and positive definite. The Cholesky factor of a banded SPD matrix has
# the same band structure, so the system admits an O(N) direct solve
# using three length-O(N) arrays (main diagonal + 2 sub-diagonals). See
# context/explainers/_drafts/arpls.md (or its eventual graduation point
# under docs/explainers/nmr/) for the full derivation.
#
# Constellation invariant: numerics here stay torch-native (no scipy in
# package internals — scipy is permitted only in tests, per the design
# choice). Performance trade-off: the per-row Cholesky recurrence is
# sequential by construction, so the Python loop dominates runtime —
# roughly 2-3 s per 32k-point spectrum on CPU, vs ~100 ms for scipy's
# compiled UMFPACK. Acceptable for per-spectrum / small-batch NMR work.
# If batch throughput becomes a real bottleneck, revisit with
# `torch.compile` on the loop body or a matrix-free CG iterative solver.
# ──────────────────────────────────────────────────────────────────────


def _dtd_bands(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bands of ``DᵀD`` where D is the (N-2)×N second-difference operator.

    Returns the three banded representations:
        main:    [1, 5, 6, 6, ..., 6, 5, 1]   (length N)
        sub1:    [-2, -4, -4, ..., -4, -2]    (length N-1)
        sub2:    [1, 1, ..., 1]               (length N-2)
    """
    f64 = torch.float64
    main = torch.full((n,), 6.0, dtype=f64)
    main[0] = 1.0
    main[1] = 5.0
    main[-2] = 5.0
    main[-1] = 1.0
    sub1 = torch.full((n - 1,), -4.0, dtype=f64)
    sub1[0] = -2.0
    sub1[-1] = -2.0
    sub2 = torch.ones(n - 2, dtype=f64)
    return main, sub1, sub2


def _cholesky_pentadiag(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cholesky decomposition of a symmetric pentadiagonal SPD matrix.

    The input matrix A is described by three diagonals:

        a[i] = A[i, i]           (main, length N)
        b[i] = A[i+1, i]         (1st sub-diagonal, length N-1)
        c[i] = A[i+2, i]         (2nd sub-diagonal, length N-2)

    The Cholesky factor L (lower-triangular with bandwidth 2) is returned
    likewise as three diagonals:

        L_main[i] = L[i, i]
        L_sub1[i] = L[i+1, i]
        L_sub2[i] = L[i+2, i]

    Standard banded-Cholesky formulas (row by row from i = 0 to N-1):

        L[i, i-2] = A[i, i-2] / L[i-2, i-2]
        L[i, i-1] = (A[i, i-1] − L[i, i-2]·L[i-1, i-2]) / L[i-1, i-1]
        L[i, i]   = √(A[i, i] − L[i, i-1]² − L[i, i-2]²)

    Out-of-range terms drop at the first two rows. Cost O(N).

    Raises
    ------
    ValueError
        If the matrix is not positive-definite (negative or zero pivot
        encountered).
    """
    n = int(a.shape[-1])
    f64 = torch.float64
    L_main = torch.zeros(n, dtype=f64)
    L_sub1 = torch.zeros(max(n - 1, 0), dtype=f64)
    L_sub2 = torch.zeros(max(n - 2, 0), dtype=f64)

    if a[0] <= 0:
        raise ValueError("Matrix is not positive-definite at row 0.")
    L_main[0] = torch.sqrt(a[0])

    if n > 1:
        L_sub1[0] = b[0] / L_main[0]
        diag_sq = a[1] - L_sub1[0] ** 2
        if diag_sq <= 0:
            raise ValueError("Matrix is not positive-definite at row 1.")
        L_main[1] = torch.sqrt(diag_sq)

    for i in range(2, n):
        L_sub2[i - 2] = c[i - 2] / L_main[i - 2]
        L_sub1[i - 1] = (b[i - 1] - L_sub2[i - 2] * L_sub1[i - 2]) / L_main[i - 1]
        diag_sq = a[i] - L_sub1[i - 1] ** 2 - L_sub2[i - 2] ** 2
        if diag_sq <= 0:
            raise ValueError(f"Matrix is not positive-definite at row {i}.")
        L_main[i] = torch.sqrt(diag_sq)

    return L_main, L_sub1, L_sub2


def _banded_solve_pentadiag(
    L_main: torch.Tensor,
    L_sub1: torch.Tensor,
    L_sub2: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve ``L Lᵀ x = rhs`` given a banded Cholesky factor (O(N)).

    Forward substitution (``L y = rhs``):
        y[i] = (rhs[i] − L_sub1[i-1]·y[i-1] − L_sub2[i-2]·y[i-2]) / L_main[i]

    Back substitution (``Lᵀ x = y``):
        x[i] = (y[i] − L_sub1[i]·x[i+1] − L_sub2[i]·x[i+2]) / L_main[i]

    Edge cases at i = 0, 1, N-2, N-1 drop out-of-range terms.
    """
    n = int(rhs.shape[-1])
    f64 = torch.float64

    # Forward: L y = rhs
    y = torch.zeros(n, dtype=f64)
    y[0] = rhs[0] / L_main[0]
    if n > 1:
        y[1] = (rhs[1] - L_sub1[0] * y[0]) / L_main[1]
    for i in range(2, n):
        y[i] = (
            rhs[i] - L_sub1[i - 1] * y[i - 1] - L_sub2[i - 2] * y[i - 2]
        ) / L_main[i]

    # Back: Lᵀ x = y
    x = torch.zeros(n, dtype=f64)
    x[n - 1] = y[n - 1] / L_main[n - 1]
    if n > 1:
        x[n - 2] = (y[n - 2] - L_sub1[n - 2] * x[n - 1]) / L_main[n - 2]
    for i in range(n - 3, -1, -1):
        x[i] = (
            y[i] - L_sub1[i] * x[i + 1] - L_sub2[i] * x[i + 2]
        ) / L_main[i]

    return x


def arpls(
    spectrum: torch.Tensor,
    lam: float = 1e5,
    ratio: float = 0.05,
    max_iter: int = 50,
) -> torch.Tensor:
    """Subtract an arPLS-estimated baseline (peak-aware; no regions needed).

    Asymmetrically Reweighted Penalized Least Squares (Baek et al. 2015):
    iteratively fits a smooth curve `z` to the spectrum `y` by minimizing
    ``Σ wᵢ(yᵢ − zᵢ)² + λ Σ (Δ²zᵢ)²`` (Whittaker smoother), with weights
    `w` updated each iteration by a logistic function of the residual
    `y − z` against the standard deviation of the *negative* residual
    points (which are dominantly baseline). Points well above the current
    estimate (probable peaks) are deweighted toward zero; points near or
    below it (probable baseline) keep weight near one.

    Each iteration's linear system ``(W + λ DᵀD) z = W y`` is solved via
    a torch-native banded Cholesky of the pentadiagonal SPD matrix — see
    `_cholesky_pentadiag` / `_banded_solve_pentadiag` above and
    `context/explainers/_drafts/arpls.md` for the derivation. The
    decomposition is O(N) per iteration with three length-O(N) arrays.
    Python-loop overhead on the row recurrence dominates wallclock time
    (~2-3 s per 32k-point spectrum on CPU). For batch workloads where
    that matters, the natural next steps are `torch.compile` on the loop
    body or a matrix-free CG iterative solver.

    **Parameter guidance:**
    - `lam` controls baseline smoothness. Larger → smoother baseline.
      Typical 1e4-1e7 for NMR spectra of ~16-32k points; tune empirically.
      Default 1e5.
    - `ratio` is the convergence threshold on relative weight change.
      Default 0.05 (5 %).

    Parameters
    ----------
    spectrum : torch.Tensor
        Real-valued 1D tensor (absorption-mode spectrum).
    lam : float
        Smoothness penalty. Default 1e5.
    ratio : float
        Convergence threshold (relative L2 change in weights). Default 0.05.
    max_iter : int
        Iteration cap. Default 50.

    Returns
    -------
    torch.Tensor
        Baseline-corrected spectrum, same shape and dtype as input.

    References
    ----------
    Baek, S.-J., Park, A., Ahn, Y.-J., & Choo, J. (2015). Baseline
    correction using asymmetrically reweighted penalized least squares
    smoothing. *Analyst*, 140(1), 250-257.
    """
    f64 = torch.float64
    y = spectrum.detach().to(f64)
    n = int(y.shape[-1])

    # Constant DᵀD bands + their λ-scaled off-diagonals (those do not
    # change across iterations; only the main diagonal picks up `w`).
    dtd_main, dtd_sub1, dtd_sub2 = _dtd_bands(n)
    lam_main = lam * dtd_main      # main contribution from λ DᵀD
    lam_sub1 = lam * dtd_sub1      # constant across iterations
    lam_sub2 = lam * dtd_sub2      # constant across iterations

    w = torch.ones(n, dtype=f64)
    z = torch.zeros(n, dtype=f64)

    for _ in range(max_iter):
        # A's main diagonal = w + λ·(DᵀD main); off-diagonals are constant.
        a = w + lam_main
        L_main, L_sub1, L_sub2 = _cholesky_pentadiag(a, lam_sub1, lam_sub2)
        z = _banded_solve_pentadiag(L_main, L_sub1, L_sub2, w * y)

        d = y - z
        neg_mask = d < 0.0
        if not bool(neg_mask.any()):
            break
        d_neg = d[neg_mask]
        m = d_neg.mean()
        s = d_neg.std()
        if float(s) == 0.0:
            break

        # Baek 2015 eq. 7: logistic reweighting on the residual.
        # arg is clipped to keep exp() from overflowing at extreme residuals.
        arg = 2.0 * (d - (2.0 * s - m)) / s
        arg = torch.clamp(arg, -500.0, 500.0)
        w_new = 1.0 / (1.0 + torch.exp(arg))

        # Convergence: relative L2 change in weights.
        denom = float(torch.linalg.norm(w))
        delta = float(torch.linalg.norm(w_new - w))
        if denom > 0.0 and delta / denom < ratio:
            w = w_new
            break
        w = w_new

    baseline = z.to(dtype=spectrum.dtype, device=spectrum.device)
    return spectrum - baseline
