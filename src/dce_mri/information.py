#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Sun Sep 27 18:46:49 2026

@author: allisonchen
"""

# -*- coding: utf-8 -*-
"""
information.py
==============
Fisher information and Cramer-Rao lower bounds for multi-flip-angle
DCE-MRI protocol design.

This module answers the question Phase 3 exists for: given a protocol,
a model and a noise level, how precisely CAN a parameter be estimated,
independent of any particular fitter? A CRLB is a property of the
experiment, not of the optimiser, so a protocol that fails here cannot
be rescued by a better fit.

Public API
----------
    fisher_matrix(predict, p, M0, sigma, ...)  -> (F, names)
    crlb(F)                                    -> (sd, cond)
    crlb_percent(predict, p, M0, snr, ...)     -> dict of % relative SD
    add_prior_precision(F, sd_by_index)        -> F + diag(1/sd^2)
    correlation_from_fisher(F)                 -> parameter correlation matrix
    sigma_from_snr(S, snr)                     -> noise SD

Conventions, all load-bearing
-----------------------------
1. M0 IS A PARAMETER. `predict` returns g(theta) with M0 factored out
   (see jax_forward.make_predict_mfa), so dS/dM0 = g is a real Jacobian
   column. It is included, the full matrix inverted, and the theta block
   read off -- algebraically identical to profiling M0 out. DROPPING the
   column instead would report theta's precision as though M0 were known,
   which it never is.

2. LOG COORDINATES by default. J_log[:, i] = p_i * dS/dp_i by the chain
   rule, so sqrt(diag(F^-1)) is directly a RELATIVE standard deviation
   (multiply by 100 for %). This also removes the ~6 orders of magnitude
   of scale spread between fp (~1e-3), tau_i (~1e-1) and M0 (~1e3) that
   would otherwise dominate the conditioning of F.

3. NOISE is sigma = baseline signal / SNR, matching Zhang 2019's
   definition (baseline signal over the noise standard deviation). With
   absolute signal that is M0 * g[0] / snr.

4. THE SETTLE MASK IS 0/1 WEIGHTS, as in jax_forward.make_residual.
   Frames immediately after a flip-angle switch carry no valid
   steady-state information and must not contribute Fisher information
   either. Pass FrameArrays.keep.

5. F IS RETURNED, NOT ITS INVERSE. Priors add as precision
   (F + diag(1/sd^2)), which is how M5 is evaluated without writing a
   sampler, so callers need F itself.

Reading the output
------------------
A CRLB is a LOWER bound on the standard deviation of any unbiased
estimator. A small bound does not promise a good fit; a large bound does
prove no fitter can do better. Report cond(F) alongside: a bound that is
enormous at full rank means ill-conditioning (parameters nearly, not
exactly, degenerate), and the condition number is the honest way to say
how nearly.
"""



from typing import Callable, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp


__all__ = [
    "fisher_matrix", "crlb", "crlb_percent", "add_prior_precision",
    "correlation_from_fisher", "sigma_from_snr", "param_names_mfa",
]


# =============================================================================
# Parameter naming -- must track make_predict_mfa's p_aug layout
# =============================================================================

def param_names_mfa(
    model_name: str = "TCXM",
    *,
    delay: bool = False,
    wx: bool = False,
    fit_tau_b: bool = False,
    fit_T10: bool = False,
    fit_B1: bool = False,
    include_M0: bool = True,
) -> list:
    """
    Names in the SAME order make_predict_mfa reads p_aug, with M0 appended
    last (it is a Jacobian column here, not a p_aug entry).

    Kept in this module rather than imported so that a layout change in
    jax_forward fails these tests loudly instead of silently relabelling
    a CRLB table.
    """
    kin = {"TCXM": ["ve", "vp", "fp", "ps"],
           "GKM":  ["Ktrans", "ve", "vp"]}[model_name]
    names = list(kin)
    if delay:
        names.append("tau_delay")
    if wx:
        if fit_tau_b:
            names.append("tau_b")
        names.append("tau_i")
    if fit_T10:
        names.append("T10")
    if fit_B1:
        names.append("B1")
    if include_M0:
        names.append("M0")
    return names


# =============================================================================
# Noise
# =============================================================================

def sigma_from_snr(S, snr: float) -> float:
    """
    Noise SD from an SNR defined on the BASELINE signal (Zhang 2019's
    convention: baseline signal / noise SD), not on the peak or the mean.
    """
    S = np.asarray(S, dtype=float)
    return float(S[0]) / float(snr)


# =============================================================================
# Fisher matrix
# =============================================================================

def fisher_matrix(
    predict:    Callable,
    p:          np.ndarray,
    M0:         float,
    sigma:      float,
    *,
    weights:    Optional[np.ndarray] = None,
    log_coords: bool = True,
    include_M0: bool = True,
):
    """
    Fisher information for a Gaussian likelihood with known, constant sigma:

        F = J^T W J / sigma^2

    Parameters
    ----------
    predict : callable p_aug -> (T,) UNIT-M0 curve, from
              jax_forward.make_predict_mfa. The M0 scaling is applied here.
    p       : (n,) parameter vector in make_predict_mfa's p_aug layout
    M0      : equilibrium magnetization; enters as its own parameter when
              include_M0 (dS/dM0 = g)
    sigma   : noise SD, same units as M0 * g
    weights : (T,) 0/1 frame weights (e.g. FrameArrays.keep). Zeroed frames
              contribute no information.
    log_coords : scale column i by p_i so the CRLB is relative
    include_M0 : append the M0 column. Turn it off only to answer the
              counterfactual "what if M0 were known exactly".

    Returns
    -------
    F : (m, m) symmetric, m = n (+1 if include_M0)
    """
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite parameter vector")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    p_j = jnp.asarray(p)
    g = np.asarray(predict(p_j), dtype=float)              # (T,) unit-M0
    J_g = np.asarray(jax.jacfwd(predict)(p_j), dtype=float)  # (T, n)

    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(J_g)):
        raise ValueError("predict or its jacobian is non-finite at p")

    J = M0 * J_g                                            # dS/dtheta
    if include_M0:
        J = np.concatenate([J, g[:, None]], axis=1)         # dS/dM0 = g

    if log_coords:
        scale = np.concatenate([p, [M0]]) if include_M0 else p
        J = J * scale[None, :]

    if weights is None:
        w = np.ones(J.shape[0])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != J.shape[0]:
            raise ValueError(
                f"weights length {w.shape[0]} != n_frames {J.shape[0]}")

    return (J.T * w) @ J / (sigma ** 2)


def add_prior_precision(F, sd_by_index: dict):
    """
    Add independent Gaussian prior precision to the diagonal.

        F_post = F + diag(1 / sd^2)

    `sd_by_index` maps a parameter index to its prior SD IN THE SAME
    COORDINATES as F -- so a relative SD (0.08 for "B1 known to 8%") when
    log_coords=True. This is the Laplace approximation to the posterior
    precision, which is how the M5 configurations are evaluated without
    running a sampler.
    """
    F = np.array(F, dtype=float, copy=True)
    for i, sd in sd_by_index.items():
        if sd is None:
            continue
        if sd <= 0:
            raise ValueError(f"prior sd for index {i} must be positive")
        F[i, i] += 1.0 / (float(sd) ** 2)
    return F


# =============================================================================
# CRLB
# =============================================================================

def crlb(F):
    """
    Cramer-Rao lower bound from a Fisher matrix.

    Returns
    -------
    sd   : (m,) sqrt(diag(F^-1)). Relative when F was built in log
           coordinates; multiply by 100 for %.
    cond : condition number of F. Report it: a huge bound at full rank
           means near-degeneracy, and cond is how near.

    A singular F (exactly degenerate parameters) yields inf entries rather
    than an exception -- that is a real, reportable answer about the
    experiment, not a numerical failure.
    """
    F = np.asarray(F, dtype=float)
    cond = float(np.linalg.cond(F))
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.full(F.shape[0], np.inf), cond
    d = np.diag(cov)
    sd = np.where(d > 0, np.sqrt(np.abs(d)), np.inf)
    return sd, cond


def correlation_from_fisher(F):
    """
    Parameter correlation matrix implied by F, i.e. the correlation of
    F^-1. This is where the off-diagonal questions live: d(tau_i)/d(B1)
    and the AIF peak-scaling degeneracy show up as |rho| near 1 between
    the relevant pair, not as a large diagonal CRLB.
    """
    F = np.asarray(F, dtype=float)
    cov = np.linalg.inv(F)
    d = np.sqrt(np.abs(np.diag(cov)))
    outer = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(outer > 0, cov / outer, np.nan)
    return R


def crlb_percent(
    predict:    Callable,
    p:          np.ndarray,
    M0:         float,
    snr:        float,
    *,
    names:      Optional[Sequence] = None,
    weights:    Optional[np.ndarray] = None,
    include_M0: bool = True,
    priors:     Optional[dict] = None,
):
    """
    Convenience wrapper: relative CRLB in percent, as a dict.

    `priors` maps a PARAMETER NAME to a relative prior SD (0.08 = 8%),
    applied via add_prior_precision. Requires `names`.

    Returns
    -------
    out  : dict name -> percent relative SD
    cond : condition number of the (possibly prior-augmented) F
    """
    p = np.asarray(p, dtype=float)
    g = np.asarray(predict(jnp.asarray(p)), dtype=float)
    sigma = sigma_from_snr(M0 * g, snr)

    F = fisher_matrix(predict, p, M0, sigma, weights=weights,
                      log_coords=True, include_M0=include_M0)

    m = F.shape[0]
    if names is None:
        names = [f"p{i}" for i in range(m - 1)] + (["M0"] if include_M0 else [])
    names = list(names)
    if len(names) != m:
        raise ValueError(f"{len(names)} names for {m} parameters")

    if priors:
        idx = {n: i for i, n in enumerate(names)}
        unknown = set(priors) - set(idx)
        if unknown:
            raise ValueError(f"prior for unknown parameter(s): {sorted(unknown)}")
        F = add_prior_precision(F, {idx[n]: sd for n, sd in priors.items()})

    sd, cond = crlb(F)
    return {n: 100.0 * s for n, s in zip(names, sd)}, cond