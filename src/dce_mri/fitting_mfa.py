#!/usr/bin/env python3
# -*- coding: utf-8 -*-\
from __future__ import annotations
"""
Created on Tue Sep 29 14:51:34 2026

@author: allisonchen
"""

# -*- coding: utf-8 -*-
"""
fitting_mfa.py
==============
Non-linear least squares for multi-flip-angle / ACE acquisitions
(refactor Phase 4).

The multi-flip-angle sibling of fitting.fit_single_voxel. Same two-stage
LHS + TRF strategy, same _lhs_starts, four differences that all follow
from the absolute-signal convention established in Phase 0:

1. M0 IS PROFILED OUT, NOT SEARCHED (variable projection).
   The model is exactly linear in M0, so at any parameter vector the
   optimal M0 has a closed form:

       M0_hat(p) = sum(w * g * y) / sum(w * g * g)

   where g = predict(p) is the unit-M0 curve make_predict_mfa returns.
   Substituting it back gives a residual in p alone. This removes a
   dimension from the search, removes the worst-conditioned direction
   (M0 spans ~3 orders of magnitude against fp's ~1e-3), and is exact
   rather than approximate.

   Variable projection normally needs the Golub-Pereyra jacobian, which
   is awkward because M0_hat itself depends on p. Differentiating the
   projected residual with jax.jacfwd produces that jacobian exactly and
   for free -- see test_varpro_jacobian_matches_finite_differences.

2. LOG COORDINATES throughout the optimiser. The parameters span six
   orders of magnitude (fp ~1e-3, tau_i ~1e-1, T10 ~1.5), which wrecks
   finite-difference jacobians and TRF's trust region alike. Fitting
   q = log(p) also makes the returned uncertainties directly comparable
   to information.crlb_percent, which works in the same coordinates.

3. T10 AND B1 ARE OPTIONAL FREE PARAMETERS, governed by NuisanceConfig
   and gated by validate_nuisance_compatibility. On single-flip-angle
   data they are not identifiable and the guardrail refuses.

4. THE SETTLE MASK IS 0/1 WEIGHTS, matching jax_forward.make_residual.
   Frames just after a flip-angle switch are not in steady state, so
   they carry no valid information and must not contribute to the cost.

Public API
----------
    profile_m0(g, y, w)                  -> closed-form M0
    build_mfa_residual(...)              -> MfaResidualBundle
    fit_curve_mfa(...)                   -> dict of results

Reading the output
------------------
`fit_curve_mfa` returns parameters in NATURAL units (the log coordinates
are internal). M0 is reported but was never searched -- it is the
projected value at the final parameter vector.
"""


from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

import jax
import jax.numpy as jnp

from dce_mri.config import (AcquisitionConfig, AifConfig, FittingConfig,
                            NuisanceConfig, RelaxationConfig, SubjectConfig,
                            validate_nuisance_compatibility)
from dce_mri.fitting import _lhs_starts, _top_finite_indices
from dce_mri.jax_forward import static_ctx_from_configs, make_predict_mfa
from dce_mri.kinetic_models import KineticModelSpec


__all__ = ["profile_m0", "build_mfa_residual", "fit_curve_mfa",
           "MfaResidualBundle"]


_TINY = 1e-300


# =============================================================================
# Variable projection
# =============================================================================

def profile_m0(g, y, w=None):
    """
    Closed-form least-squares M0 for a unit-M0 curve g against data y.

        M0_hat = sum(w g y) / sum(w g^2)

    This is the exact minimiser of sum(w (M0 g - y)^2) over M0, not an
    approximation, because the model is linear in M0.

    Works on numpy or jax arrays; the jax path is what jacfwd
    differentiates through.
    """
    xp = jnp if isinstance(g, jnp.ndarray) else np
    if w is None:
        num = xp.sum(g * y)
        den = xp.sum(g * g)
    else:
        num = xp.sum(w * g * y)
        den = xp.sum(w * g * g)
    return num / (den + _TINY)


# =============================================================================
# Residual bundle
# =============================================================================

@dataclass
class MfaResidualBundle:
    """
    Everything the multi-flip-angle optimiser needs, in LOG coordinates.

    Mirrors fitting.ResidualBundle's role: the NLLS fitter, the bootstrap
    and any Bayesian likelihood must evaluate the SAME residual, or
    comparisons between them are meaningless.

    Fields
    ------
    residual_fn  : callable q -> residual vector, q = log(params).
                   Includes any prior penalty rows.
    jac_fn       : callable q -> jacobian, or None for finite differences
    predict_fn   : callable q -> predicted curve, M0 included
    m0_fn        : callable q -> the projected M0 at q
    log_lb, log_ub : bounds on q
    param_names  : names of the FITTED parameters (M0 not among them)
    target       : the curve being fitted
    weights      : 0/1 frame weights actually used
    n_prior_rows : how many trailing residual entries are prior penalties
    """
    residual_fn:  Optional[Callable]
    jac_fn:       Optional[Callable]
    predict_fn:   Optional[Callable]
    m0_fn:        Optional[Callable]
    log_lb:       np.ndarray
    log_ub:       np.ndarray
    param_names:  tuple
    target:       np.ndarray
    weights:      np.ndarray
    n_prior_rows: int
    _kernels:     tuple = ()
    _use_jax_jac: bool  = False

    def _bind(self, sig) -> None:
        """Point the public callables at `sig`, reusing the compiled
        kernels. Called by build_mfa_residual and by with_target."""
        residual_j, jac_j, predict_j, m0_j = self._kernels
        y = jnp.asarray(np.asarray(sig, dtype=float))
        self.target = np.asarray(sig, dtype=float)
        self.residual_fn = lambda q: np.asarray(
            residual_j(jnp.asarray(q, dtype=float), y), dtype=float)
        self.jac_fn = (lambda q: np.asarray(
            jac_j(jnp.asarray(q, dtype=float), y), dtype=float)) \
            if self._use_jax_jac else None
        self.predict_fn = lambda q: np.asarray(
            predict_j(jnp.asarray(q, dtype=float), y), dtype=float)
        self.m0_fn = lambda q: float(m0_j(jnp.asarray(q, dtype=float), y))

    def with_target(self, sig) -> "MfaResidualBundle":
        """
        A bundle for a DIFFERENT measured curve, sharing this one's
        compiled kernels. This is the bootstrap / voxelwise path: build
        once, retarget many times. Rebuilding from scratch instead would
        recompile the forward model for every curve.
        """
        import copy
        other = copy.copy(self)
        other._bind(sig)
        return other


def build_mfa_residual(
    sig:       np.ndarray,
    aif:       np.ndarray,
    time_var:  np.ndarray,
    model:     KineticModelSpec,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    cfg:       FittingConfig,
    nuisance:  NuisanceConfig,
    aif_cfg:   Optional[AifConfig] = None,
    relax_cfg: Optional[RelaxationConfig] = None,
    wx:        bool = False,
    weights:   Optional[np.ndarray] = None,
    use_keep:  bool = True,
) -> MfaResidualBundle:
    """
    Assemble the projected residual for one curve.

    Parameter layout, matching make_predict_mfa exactly:

        q = log([ *theta_kin, (tau_b?), (tau_i?), (T10_s?), (B1?) ])

    M0 is absent by construction -- it is projected, not fitted.

    weights  : explicit 0/1 (or real) frame weights. When None and
               use_keep is True, they are taken from
               acq.frame_arrays(time_var).keep, which drops the frames
               after each flip-angle switch.
    """
    validate_nuisance_compatibility(nuisance, acq)

    if nuisance.m0_mode != "profile":
        raise ValueError(
            f"build_mfa_residual implements variable projection; "
            f"m0_mode={nuisance.m0_mode!r} is not this fitter's job "
            "('marginal' is Bayes-only, 'baseline' is the legacy single-FA path)"
        )
    if wx and (aif_cfg is None or relax_cfg is None):
        raise ValueError("water exchange needs aif_cfg and relax_cfg")

    sig = np.asarray(sig, dtype=float)
    time_var = np.asarray(time_var, dtype=float)

    # --- frame weights ---------------------------------------------------
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
    elif use_keep and getattr(acq, "protocol", None) is not None:
        w = acq.frame_arrays(time_var).keep.astype(float)
    else:
        w = np.ones(sig.size)
    if w.shape != sig.shape:
        raise ValueError(f"weights shape {w.shape} != signal shape {sig.shape}")
    if not np.any(w > 0):
        raise ValueError("all frames are zero-weighted")

    # --- the unit-M0 forward model --------------------------------------
    fit_tau_b = bool(relax_cfg.fit_tau_b) if (wx and relax_cfg is not None) else False
    ctx = static_ctx_from_configs(acq, subject, aif_cfg, time_var)
    predict_unit = make_predict_mfa(
        model.name, time_var, aif, ctx,
        wx=wx,
        fit_tau_b=fit_tau_b,
        tau_b_fixed=(relax_cfg.tau_b_fixed if relax_cfg is not None else 0.005),
        fit_T10=nuisance.fit_T10,
        fit_B1=nuisance.fit_B1,
        upsample=getattr(acq, "upsample", 50),
    )

    # --- bounds and names, in the p_aug order ---------------------------
    lb = list(np.asarray(model.lb, dtype=float))
    ub = list(np.asarray(model.ub, dtype=float))
    names = list(model.param_names)

    if wx:
        if fit_tau_b:
            lb.append(relax_cfg.tau_b_lb); ub.append(relax_cfg.tau_b_ub)
            names.append("tau_b")
        lb.append(relax_cfg.tau_i_lb); ub.append(relax_cfg.tau_i_ub)
        names.append("tau_i")
    if nuisance.fit_T10:                     # seconds, matching ctx.T10_s
        lb.append(nuisance.T10_bounds_ms[0] / 1000.0)
        ub.append(nuisance.T10_bounds_ms[1] / 1000.0)
        names.append("T10")
    if nuisance.fit_B1:
        lb.append(nuisance.B1_bounds[0]); ub.append(nuisance.B1_bounds[1])
        names.append("B1")

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if np.any(lb <= 0):
        raise ValueError(
            "log coordinates require strictly positive lower bounds; got "
            f"{dict(zip(names, lb))}")

    # --- prior penalty rows ---------------------------------------------
    prior_rows = []                        # (index in q, log mu, relative sd)
    if nuisance.fit_T10 and nuisance.T10_prior is not None:
        mu, sd = nuisance.T10_prior
        prior_rows.append((names.index("T10"), np.log(mu / 1000.0), sd))
    if nuisance.fit_B1 and nuisance.B1_prior is not None:
        mu, sd = nuisance.B1_prior
        prior_rows.append((names.index("B1"), np.log(mu), sd))

    w_j = jnp.asarray(w)
    idx_j = jnp.asarray([r[0] for r in prior_rows], dtype=int) if prior_rows else None
    mu_j = jnp.asarray([r[1] for r in prior_rows]) if prior_rows else None
    sd_j = jnp.asarray([r[2] for r in prior_rows]) if prior_rows else None

    # The measured curve is an ARGUMENT, not a closure constant, so one
    # compilation serves every target. Baking it in would force a fresh
    # jit trace per curve -- fatal for the bootstrap, which refits
    # hundreds of resampled targets, and for any voxelwise loop.
    def _m0(q, y):
        return profile_m0(predict_unit(jnp.exp(q)), y, w_j)

    def _predict(q, y):
        g = predict_unit(jnp.exp(q))
        return profile_m0(g, y, w_j) * g

    def _residual(q, y):
        g = predict_unit(jnp.exp(q))
        r = w_j * (profile_m0(g, y, w_j) * g - y)
        if idx_j is None:
            return r
        return jnp.concatenate([r, (q[idx_j] - mu_j) / sd_j])

    residual_j = jax.jit(_residual)
    jac_j = jax.jit(jax.jacfwd(_residual, argnums=0))
    predict_j = jax.jit(_predict)
    m0_j = jax.jit(_m0)

    bundle = MfaResidualBundle(
        residual_fn  = None, jac_fn = None, predict_fn = None, m0_fn = None,
        log_lb       = np.log(lb),
        log_ub       = np.log(ub),
        param_names  = tuple(names),
        target       = sig,
        weights      = w,
        n_prior_rows = len(prior_rows),
        _kernels     = (residual_j, jac_j, predict_j, m0_j),
        _use_jax_jac = (cfg.jacobian == "jax"),
    )
    bundle._bind(sig)
    return bundle


# =============================================================================
# Fitter
# =============================================================================

def fit_curve_mfa(
    bundle: MfaResidualBundle,
    cfg:    FittingConfig,
    seed:   Optional[int] = None,
    p0:     Optional[np.ndarray] = None,
) -> dict:
    """
    Two-stage LHS + TRF fit in log coordinates.

    Stage 1 draws cfg.n_lhs starting points uniformly in log space (which
    is what _lhs_starts' log_dims option does anyway, applied to every
    dimension here). Stage 2 refines the cfg.n_top cheapest with TRF.

    p0 : optional starting point in NATURAL units, refined in addition to
         the LHS draws. Useful for recovery tests and for warm-starting a
         bootstrap replicate; does not replace the LHS stage.

    Returns
    -------
    dict with
        params      : (n,) fitted values, NATURAL units
        log_params  : (n,) the same in log coordinates
        M0          : projected M0 at the solution
        cost        : 0.5 * sum(residual^2) at the solution, scipy's
                      convention
        success     : whether any refinement converged
        n_converged : how many of the n_top refinements converged
        param_names : names, M0 excluded (it was never searched)
    """
    lb, ub = bundle.log_lb, bundle.log_ub
    d = lb.size

    starts = _lhs_starts(
        n=cfg.n_lhs, lb=np.exp(lb), ub=np.exp(ub),
        log_dims=list(range(d)), seed=seed,
    )
    q_starts = np.log(np.clip(starts, np.exp(lb), np.exp(ub)))
    if p0 is not None:
        q_starts = np.vstack([np.log(np.asarray(p0, dtype=float)), q_starts])

    costs = np.array([
        float(np.sum(bundle.residual_fn(q) ** 2)) if np.all(np.isfinite(q))
        else np.inf
        for q in q_starts
    ])
    top = _top_finite_indices(costs, min(cfg.n_top, len(costs)))

    best_q, best_cost, n_ok = None, np.inf, 0
    for i in top:
        try:
            res = least_squares(
                fun      = bundle.residual_fn,
                x0       = q_starts[i],
                jac      = bundle.jac_fn if bundle.jac_fn is not None else "3-point",
                bounds   = (lb, ub),
                method   = "trf",
                x_scale  = cfg.x_scale,
                xtol     = cfg.xtol,
                ftol     = cfg.ftol,
                gtol     = cfg.gtol,
                max_nfev = cfg.max_nfev,
            )
        except Exception:
            continue
        if res.status > 0:
            n_ok += 1
        c = float(np.sum(res.fun ** 2))
        if c < best_cost:
            best_cost, best_q = c, res.x

    if best_q is None:
        return dict(params=np.full(d, np.nan), log_params=np.full(d, np.nan),
                    M0=np.nan, cost=np.inf, success=False, n_converged=0,
                    param_names=bundle.param_names)

    return dict(
        params      = np.exp(best_q),
        log_params  = best_q,
        M0          = bundle.m0_fn(best_q),
        cost        = 0.5 * best_cost,
        success     = n_ok > 0,
        n_converged = n_ok,
        param_names = bundle.param_names,
    )