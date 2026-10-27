#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Created on Tue Sep 29 15:19:37 2026

@author: allisonchen
"""

# -*- coding: utf-8 -*-
"""
bayes_mfa.py
============
Bayesian likelihood for multi-flip-angle / ACE acquisitions
(refactor Phase 5).

This module adds ONE thing to the existing Bayes layer: a likelihood in
which BOTH the equilibrium magnetization M0 and the noise SD sigma are
integrated out analytically. It sits beside bayes.py rather than inside
it, and plugs into bayes_blackjax.sample_blackjax the same way
make_log_lik_jax does.

Why M0 can be integrated out in closed form
-------------------------------------------
The model is y = M0 * g(theta) + eps, eps ~ N(0, sigma^2), where
g = jax_forward.make_predict_mfa's unit-M0 curve. It is EXACTLY linear in
M0, so with a flat prior on M0 the M0 integral is Gaussian; with the
Jeffreys prior p(sigma^2) ∝ 1/sigma^2 -- the same one
bayes.GAUSS_MARGINALIZED_SIGMA uses -- the sigma integral is a Gamma
integral. Writing N for the effective number of frames and

    S_gg     = sum w g^2
    M0_hat   = sum w g y / S_gg                 (fitting_mfa.profile_m0)
    chi2_min = sum w (y - M0_hat g)^2

the result is

    log L(theta) = log Gamma((N-1)/2) - ((N-1)/2) log(chi2_min / 2)
                   - ((N-1)/2) log(2 pi) - (1/2) log S_gg

Two differences from the existing sigma-only marginal (Orton 2007 Eq. 8):

  * N -> N - 1, one degree of freedom absorbed by M0.
  * an extra -1/2 log S_gg. This is not a constant -- S_gg depends on
    theta -- it is the Occam factor from the M0 integral.

How much the Occam term matters (measured, not assumed)
------------------------------------------------------
Plugging the profiled M0_hat into the sigma-only likelihood drops the
Occam term and uses N instead of N-1. The resulting posterior differs
from this one by an O(1/N) correction: on a B1 grid, total-variation
distance 0.064 at N=6, 0.025 at N=12, 0.003 at N=99, with N*TV roughly
constant. The Occam term varies by well under a nat across the whole
prior range, while the likelihood varies by O(N) nats.

So at typical DCE frame counts the profile plug-in is NOT meaningfully
biased. What this marginal adds is exactness, which matters at small N,
and an exact posterior for M0 itself (m0_conditional) for reporting.
Removing M0 as a sampling dimension is shared by both approaches. See
test_marginal_and_profile_converge_as_O_1_over_N.

Conventions, matching bayes.GAUSS_MARGINALIZED_SIGMA exactly
------------------------------------------------------------
* sigma prior is p(sigma^2) ∝ 1/sigma^2. Equivalent to p(sigma) ∝ 1/sigma
  up to a constant factor of 2, which never matters for sampling.
* M0 prior is flat on the whole real line. M0 is physically positive,
  but at any SNR where the data say anything, the M0 posterior sits many
  SDs from zero and truncating at 0 changes nothing measurable.
* Weights must be 0/1 (a frame mask, e.g. FrameArrays.keep). N is the
  number of KEPT frames. Real-valued weights would be heteroscedastic
  precision weights, where "the number of frames" has no clean meaning,
  so they are refused rather than silently misinterpreted.

Numerical note
--------------
chi2_min is computed as sum w (y - M0_hat g)^2, never as
S_yy - S_gy^2 / S_gg. The two are algebraically equal, but the second
subtracts two nearly equal large numbers: at SNR ~1e4 it loses ~8
significant digits and can go negative. See
test_chi2_min_is_numerically_stable_at_high_snr.

Public API
----------
    m0_sigma_marginal_loglik(g, y, w=None)   -> scalar log L(theta)
    m0_conditional(g, y, w=None)             -> (loc, scale, dof)
    make_log_lik_mfa_jax(predict_unit, weights=None) -> log_lik(p, y)
"""

from typing import Callable, Optional

import numpy as np

import jax.numpy as jnp
from jax.scipy.special import gammaln as jgammaln
from scipy.special import gammaln as sgammaln


__all__ = ["m0_sigma_marginal_loglik", "m0_conditional",
           "make_log_lik_mfa_jax", "check_mask"]

_LOG_2PI = float(np.log(2.0 * np.pi))
_TINY = 1e-300


def check_mask(w, n_frames: int) -> np.ndarray:
    """
    Validate a 0/1 frame mask and return it as float. Raises on
    non-binary entries, wrong length, or fewer than 3 kept frames
    (N - 1 >= 2 is needed for the sigma integral to give a proper,
    usable posterior).
    """
    if w is None:
        w = np.ones(n_frames)
    w = np.asarray(w, dtype=float).ravel()
    if w.shape[0] != n_frames:
        raise ValueError(f"mask length {w.shape[0]} != n_frames {n_frames}")
    if not np.all((w == 0.0) | (w == 1.0)):
        raise ValueError(
            "weights must be a 0/1 frame mask. Real-valued weights are "
            "heteroscedastic precision weights, for which the effective "
            "N in this likelihood is undefined.")
    if w.sum() < 3:
        raise ValueError(f"need at least 3 kept frames, got {int(w.sum())}")
    return w


def _sums(g, y, w, xp):
    S_gg = xp.sum(w * g * g)
    m0 = xp.sum(w * g * y) / (S_gg + _TINY)
    r = y - m0 * g
    chi2 = xp.sum(w * r * r)
    return S_gg, m0, chi2


def m0_sigma_marginal_loglik(g, y, w=None):
    """
    log L(theta), with M0 (flat prior) and sigma (Jeffreys) integrated
    out analytically. See the module docstring for the formula.

    g : (T,) unit-M0 predicted curve at theta
    y : (T,) measured curve
    w : (T,) 0/1 frame mask, or None for all frames

    Works on numpy or jax arrays; the jax path is jit/grad-ready. With
    numpy input the mask is validated; with jax input it is not (it may
    be a tracer), so validate once with check_mask before tracing.
    """
    is_jax = isinstance(g, jnp.ndarray) or isinstance(y, jnp.ndarray)
    xp = jnp if is_jax else np
    gammaln = jgammaln if is_jax else sgammaln

    if w is None:
        w = xp.ones_like(g)
    elif not is_jax:
        w = check_mask(w, np.asarray(g).shape[0])

    n = xp.sum(w)
    S_gg, _, chi2 = _sums(g, y, w, xp)
    chi2 = xp.maximum(chi2, _TINY)
    nu = n - 1.0
    return (gammaln(0.5 * nu) - 0.5 * nu * xp.log(0.5 * chi2)
            - 0.5 * nu * _LOG_2PI - 0.5 * xp.log(S_gg + _TINY))


def m0_conditional(g, y, w=None):
    """
    Posterior of M0 given theta, with sigma integrated out: a Student-t

        M0 | theta, y  ~  t_dof( loc, scale )
        loc   = M0_hat
        scale = sqrt( chi2_min / (dof * S_gg) )
        dof   = N - 1

    This is how M0 is recovered for reporting after sampling theta alone:
    draw one M0 from this distribution per posterior sample of theta.
    It is exact, not an approximation, under the same priors as the
    likelihood.
    """
    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float)
    w = check_mask(w, g.shape[0])
    S_gg, m0, chi2 = _sums(g, y, w, np)
    dof = float(w.sum() - 1.0)
    return float(m0), float(np.sqrt(chi2 / (dof * S_gg))), dof


def make_log_lik_mfa_jax(predict_unit: Callable,
                         weights: Optional[np.ndarray] = None,
                         n_frames: Optional[int] = None) -> Callable:
    """
    `log_lik(p, y) -> scalar`, the likelihood half for the blackjax path.

    predict_unit : jax_forward.make_predict_mfa's output, p -> (T,) g
    weights      : 0/1 frame mask (FrameArrays.keep), validated here
    n_frames     : required when weights is None

    The measured curve y is an ARGUMENT, for the same reason as
    fitting_mfa.MfaResidualBundle.with_target: SBC and any per-voxel
    loop evaluate many curves, and closing over y would force a fresh
    compilation per curve. Bind it for a single sampling run with
    functools.partial(log_lik, y=y) -- that reuses the compiled kernel.

    Like bayes_blackjax.make_log_lik_jax, this is the LIKELIHOOD ONLY; the
    prior is added by make_unconstrained_logdensity. Non-finite values
    map to -1e30, matching that function's convention.
    """
    if weights is None:
        if n_frames is None:
            raise ValueError("give weights or n_frames")
        weights = np.ones(n_frames)
    w_np = check_mask(weights, len(weights) if n_frames is None else n_frames)
    w = jnp.asarray(w_np)

    def log_lik(p, y):
        g = predict_unit(p)
        ll = m0_sigma_marginal_loglik(g, jnp.asarray(y), w)
        return jnp.where(jnp.isfinite(ll), ll, -1e30)

    return log_lik