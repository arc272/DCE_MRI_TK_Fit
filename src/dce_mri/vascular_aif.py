# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
vascular_aif.py
===============
Vascular AIF dispersion model — Nejad-Davarani et al. (NMR Biomed 2017).

Models how a bolus measured at the carotid artery disperses and delays
as it travels through branching vessel generations to reach tissue.

Public API
----------
    single_vessel_h(t, t0)                   →  h   [Eq. 2]
    vascular_tree_h(t, t0, n_level)          →  h1n [Eq. 12]
    local_aif_at_level(aif, t, t0, g, n)    →  ca_local
    fit_vascular_aif(ca_tissue, aif, t, ...) →  VascularAIFResult
    get_local_aif(ca_tissue, aif, t, ...)   →  (local_aif, result)

Internal helpers
----------------
    _fit_single_level(ca, aif, t, n_level, ...)
    _profile_g(t0, ca, aif, t, n_level)
    _profile_g(t0, ca, aif, t, n_level)
    compute_aic(rss, n_obs, n_param)
    akaike_weights(aic_values)

Bounds
------
T0_LB/T0_UB/G_LB/G_UB below are module-level DEFAULTS only — matching
the tau_lb/tau_ub pattern used for delay fitting elsewhere in this
package (see fit_roi_with_delay_continuous). Callers should pass their
own t0_lb/t0_ub/g_lb/g_ub explicitly based on their data's expected
delay range, e.g. from a notebook:

    T0_LB, T0_UB = 0.05, 3.0   # s — adjust based on expected transit times
    G_LB,  G_UB  = 1e-3, 1e3   # adjust based on expected voxel gain range
    result = fit_vascular_aif(ca_tissue, aif, t,
                               t0_lb=T0_LB, t0_ub=T0_UB, g_lb=G_LB, g_ub=G_UB)

Fitting method
--------------
_fit_single_level is DETERMINISTIC, not a multi-start stochastic search.
local_aif_at_level(..., g, ...) is exactly linear in g, so for any fixed
t0 the optimal g has a closed-form least-squares solution (_profile_g).
This collapses the real search from 2D [g, t0] down to 1D [t0] alone,
searched via a dense grid + bounded local polish. This was chosen after
multi-start Nelder-Mead on the full 2D [g, t0] problem showed a real
local-optima trap: on one test case, 3 of 4 random restarts converged
to rss ~140-200 while only 1 found the true optimum near rss=0, and the
failure rate did not reliably go to zero even at n_restarts=25. The
profiled 1D approach had 0 failures across the same test cases and
needs no seed at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

_EPS      = 1e-12
N_LEVELS  = 6

# Module-level DEFAULTS only -- see "Bounds" note above. Pass t0_lb/
# t0_ub/g_lb/g_ub explicitly to override for your data.
T0_LB = 0.01
T0_UB = 5.0
G_LB  = 1e-6
G_UB  = 1e4

# Default grid density for the 1D t0 search in _fit_single_level.
N_GRID_DEFAULT = 200


# =============================================================================
# Result type
# =============================================================================

@dataclass(frozen=True)
class VascularAIFResult:
    """
    Result of fitting the vascular dispersion model to one voxel.

    Fields
    ------
    g           : (N_LEVELS,) optimal gain per level
    t0          : (N_LEVELS,) optimal single-vessel delay per level  (s)
    rss         : (N_LEVELS,) RSS per level
    aic         : (N_LEVELS,) AICc per level
    weights     : (N_LEVELS,) Akaike weights (sum to 1)
    best_level  : argmax(weights), 1-indexed
    delay_total : model-averaged total delay  (s)  [Paper Eq. 19]
                  = sum_i weights[i] * (i+1) * t0[i]  -- NOT the centroid
                  of h(t); the paper defines total delay through i
                  vessel segments in series as i * t0, then weights
                  that across levels by the Akaike weights.
    local_aif   : (T,) model-averaged local AIF    [Paper Eq. 18]
    h_est       : (T,) model-averaged transfer function
    success     : at least one level converged
    """
    g:           np.ndarray
    t0:          np.ndarray
    rss:         np.ndarray
    aic:         np.ndarray
    weights:     np.ndarray
    best_level:  int
    delay_total: float
    local_aif:   np.ndarray
    h_est:       np.ndarray
    success:     bool


# =============================================================================
# Public functions
# =============================================================================


def single_vessel_h(
    t:  np.ndarray,
    t0: float,
) -> np.ndarray:
    """
    Laminar-flow transfer function for a single vessel.

        h(t) = 0            for t < t0
        h(t) = 2*t0^2/t^3   for t >= t0

    t0 is the minimum transit time through the vessel (s).
    """
    t = np.asarray(t, dtype=np.float64)
    h = np.zeros_like(t)
    mask = t >= t0
    # avoid division by zero at t=0 when t0=0 (degenerate case)
    t_safe = np.where(t > 0, t, np.inf)
    h[mask] = 2.0 * t0**2 / t_safe[mask]**3
    return h


def vascular_tree_h(
    t:       np.ndarray,
    t0:      float,
    n_level: int,
) -> np.ndarray:
    """
    N-level vascular tree transfer function.

    h_n(t) = single_vessel_h convolved with itself n_level times.
    """
    t  = np.asarray(t, dtype=np.float64)
    dt = t[1] - t[0]

    h = single_vessel_h(t, t0)

    if n_level <= 1:
        return h

    h_n = h.copy()
    for _ in range(n_level - 1):
        h_n = fftconvolve(h_n, h)[:len(t)] * dt

    return h_n


def local_aif_at_level(
    global_aif: np.ndarray,
    t:          np.ndarray,
    t0:         float,
    g:          float,
    n_level:    int,
) -> np.ndarray:
    """
    Local AIF at branching level n  [Paper Eq. 13-14]:

        CA_local(t) = g · (AIF * h_{1→n})(t)

    Verified against an independent (non-fftconvolve) manual discrete
    convolution -- see local_aif_at_level_test.py.
    """
    t  = np.asarray(t, dtype=np.float64)
    dt = t[1] - t[0]
    h  = vascular_tree_h(t, t0, n_level)
    return g * fftconvolve(global_aif, h)[:len(t)] * dt


def fit_vascular_aif(
    ca_tissue:  np.ndarray,
    global_aif: np.ndarray,
    t:          np.ndarray,
    n_levels:   int   = N_LEVELS,
    t0_lb:      float = T0_LB,
    t0_ub:      float = T0_UB,
    g_lb:       float = G_LB,
    g_ub:       float = G_UB,
    n_grid:     int   = N_GRID_DEFAULT,
) -> VascularAIFResult:
    """
    Fit the vascular AIF model at each branching level independently,
    then select via AIC model averaging  [Paper Sec. 2.2].

    Deterministic -- no seed needed; see module docstring "Fitting
    method" note.

    Parameters
    ----------
    t0_lb, t0_ub : bounds on the per-vessel transit time t0 (s).
                   Pass explicitly based on your data's expected delay
                   range -- see module docstring "Bounds" note.
    g_lb, g_ub   : bounds on the gain factor g.
    n_grid       : density of the 1D t0 search grid per level.
    """
    n_obs = len(t)
    g_arr         = np.zeros(n_levels)
    t0_arr        = np.zeros(n_levels)
    rss_arr       = np.full(n_levels, np.inf)
    aic_arr       = np.full(n_levels, np.inf)
    h_per_level   = np.zeros((n_levels, n_obs))
    aif_per_level = np.zeros((n_levels, n_obs))
    ok_per_level  = np.zeros(n_levels, dtype=bool)

    for i, n_level in enumerate(range(1, n_levels + 1)):
        try:
            g, t0, rss = _fit_single_level(
                ca_tissue, global_aif, t, n_level,
                t0_lb=t0_lb, t0_ub=t0_ub, g_lb=g_lb, g_ub=g_ub,
                n_grid=n_grid,
            )
            if g is None or not np.isfinite(rss):
                continue
            g_arr[i], t0_arr[i], rss_arr[i] = g, t0, rss
            aic_arr[i] = compute_aic(rss, n_obs, n_param=2)
            h_per_level[i]   = vascular_tree_h(t, t0, n_level)
            aif_per_level[i] = local_aif_at_level(global_aif, t, t0, g, n_level)
            ok_per_level[i]  = True
        except Exception:
            continue

    if not ok_per_level.any():
        return VascularAIFResult(
            g=g_arr, t0=t0_arr, rss=rss_arr, aic=aic_arr,
            weights=np.zeros(n_levels), best_level=1,
            delay_total=np.nan,
            local_aif=np.full(n_obs, np.nan),
            h_est=np.full(n_obs, np.nan),
            success=False,
        )

    weights    = akaike_weights(aic_arr)
    best_level = int(np.argmax(weights)) + 1   # 1-indexed

    local_aif_avg = np.tensordot(weights, aif_per_level, axes=(0, 0))
    h_avg         = np.tensordot(weights, h_per_level,   axes=(0, 0))

    # Eq. 19: t_Total = sum_i w_i * i * t0_i  -- NOT a centroid of h(t).
    # The paper's simplifying assumption (Murray's law, Eq. 9-11) treats
    # every vessel segment along a path as sharing the same t0, so total
    # transit through i segments in series is just i * t0.
    level_numbers = np.arange(1, n_levels + 1)
    delay_total = float(np.sum(weights * level_numbers * t0_arr))

    return VascularAIFResult(
        g=g_arr, t0=t0_arr, rss=rss_arr, aic=aic_arr, weights=weights,
        best_level=best_level, delay_total=delay_total,
        local_aif=local_aif_avg, h_est=h_avg, success=True,
    )


def get_local_aif(
    ca_tissue:  np.ndarray,
    global_aif: np.ndarray,
    t:          np.ndarray,
    n_levels:   int   = N_LEVELS,
    t0_lb:      float = T0_LB,
    t0_ub:      float = T0_UB,
    g_lb:       float = G_LB,
    g_ub:       float = G_UB,
    n_grid:     int   = N_GRID_DEFAULT,
) -> tuple[np.ndarray, VascularAIFResult]:
    """
    Convenience wrapper: fit vascular model and return (local_aif, result).
    Falls back to global AIF if all fits fail.
    """
    result = fit_vascular_aif(
        ca_tissue, global_aif, t, n_levels,
        t0_lb=t0_lb, t0_ub=t0_ub, g_lb=g_lb, g_ub=g_ub, n_grid=n_grid,
    )
    if not result.success:
        return global_aif.copy(), result
    return result.local_aif, result


# =============================================================================
# Internal helpers
# =============================================================================

def _profile_g(
    t0:         float,
    ca_tissue:  np.ndarray,
    global_aif: np.ndarray,
    t:          np.ndarray,
    n_level:    int,
) -> tuple[float, float]:
    """
    Closed-form optimal g for a FIXED t0.

    local_aif_at_level(..., g, ...) = g * C(t0) where
    C(t0) = local_aif_at_level(..., g=1.0, ...) is the unit-gain shape.
    Minimizing sum((ca_tissue - g*C(t0))^2) over g alone is ordinary
    single-covariate least squares:

        g_opt = <C(t0), ca_tissue> / <C(t0), C(t0)>

    Returns (g_opt, rss_at_g_opt).
    """
    C = local_aif_at_level(global_aif, t, t0, g=1.0, n_level=n_level)
    if not np.all(np.isfinite(C)):
        return 0.0, np.inf
    denom = float(np.dot(C, C))
    if denom < _EPS:
        return 0.0, np.inf
    g_opt = float(np.dot(C, ca_tissue) / denom)
    rss = float(np.sum((ca_tissue - g_opt * C) ** 2))
    return g_opt, rss


def _fit_single_level(
    ca_tissue:  np.ndarray,
    global_aif: np.ndarray,
    t:          np.ndarray,
    n_level:    int,
    t0_lb:      float = T0_LB,
    t0_ub:      float = T0_UB,
    g_lb:       float = G_LB,
    g_ub:       float = G_UB,
    n_grid:     int   = N_GRID_DEFAULT,
) -> tuple[float, float, float]:
    """
    Fit [g, t0] for one branching level.

    DETERMINISTIC -- g is profiled out in closed form for every
    candidate t0 (see _profile_g); only t0 is actually searched, via a
    dense grid over [t0_lb, t0_ub] followed by a bounded local polish
    around the grid winner. No random restarts, no seed.

    g_lb/g_ub are enforced by clipping the profiled g_opt into range
    after the t0 search (the profiled solution only very rarely lands
    outside a reasonably-set [g_lb, g_ub], since it's the exact
    least-squares optimum for the winning t0).

    Returns (g_opt, t0_opt, rss)
    """
    t0_grid = np.linspace(t0_lb, t0_ub, n_grid)

    best_rss, best_t0 = np.inf, t0_grid[0]
    for t0 in t0_grid:
        _, rss = _profile_g(t0, ca_tissue, global_aif, t, n_level)
        if rss < best_rss:
            best_rss, best_t0 = rss, t0

    step = t0_grid[1] - t0_grid[0]
    lo = max(t0_lb, best_t0 - step)
    hi = min(t0_ub, best_t0 + step)

    def neg_objective(t0):
        _, rss = _profile_g(t0, ca_tissue, global_aif, t, n_level)
        return rss

    try:
        res = minimize_scalar(
            neg_objective, bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-8},
        )
        t0_final = float(res.x)
    except Exception:
        t0_final = best_t0

    g_final, rss_final = _profile_g(t0_final, ca_tissue, global_aif, t, n_level)
    g_final = float(np.clip(g_final, g_lb, g_ub))

    return g_final, t0_final, rss_final


def compute_aic(rss: float, n_obs: int, n_param: int = 2) -> float:
    """
    Corrected AIC (AICc)  [Paper Eq. 15].
    Verified in aic_weights_test.py.

    n_param is the number of free parameters actually being fit -- 2
    for vascular_aif's own [g, t0] fits, but callers combining models
    with a different parameter count (e.g. fitting.fit_dispersed_gkm,
    which fits [Ktrans, ve, vp, t0_vasc] per level) must pass n_param=4
    explicitly. This function makes no assumption about what's being
    fit; it only implements the AICc formula.
    """
    rss = max(float(rss), _EPS)
    aic = 2 * n_param + n_obs * np.log(rss / n_obs)
    denom = n_obs - n_param - 1
    if denom > 0:
        aic += (2 * n_param * (n_param + 1)) / denom
    return aic


def akaike_weights(aic_values: np.ndarray) -> np.ndarray:
    """
    Convert AIC values to Akaike weights  [Paper Eq. 16-17].
    Verified in aic_weights_test.py.

    Non-finite entries (a level that failed to converge, encoded as
    aic=inf by the caller) get weight exactly 0.0 and the remaining
    weights renormalize to sum to 1. All-non-finite input returns an
    all-zero vector rather than raising.
    """
    aic = np.asarray(aic_values, dtype=float)
    finite = np.isfinite(aic)
    if not finite.any():
        return np.zeros_like(aic)
    best = aic[finite].min()
    delta = np.where(finite, aic - best, np.inf)
    rel_lik = np.exp(-0.5 * delta)
    return rel_lik / rel_lik.sum()