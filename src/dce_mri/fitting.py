# -*- coding: utf-8 -*-
"""
fitting.py
==========
Single-voxel kinetic model fitting — LHS coarse search + TRF refinement.

This is the standard independent per-voxel fitter.
For TV-regularised joint fitting, see fitting_admm.py.
For whole-ROI mean-curve fitting, see fitting_roi.py.

Public API
----------
    fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg)
        →  VoxelFitResult
        Fits any KineticModelSpec in either concentration or signal space.
        Controlled by cfg.fit_space.

    fit_dispersed_gkm(ca_tissue, aif, time_var, acq, cfg, n_levels)
        →  DispersedGKMResult
        Loops over integer branch levels 1…n_levels, fits [Ktrans, ve, vp, t0]
        independently at each level, returns best by lowest cost.

Internal helpers (not part of public API)
-----------------------------------------
    _lhs_starts(n, lb, ub, constraints, seed)
    _trf_refine(p0, residual_fn, lb, ub, cfg)
    _top_finite_indices(costs, n_top)
    _residual_concentration(p, Ct_meas, time_var, aif, model_fn)
    _residual_signal(p, sig_meas, time_var, aif, model_fn, acq, T10, B1, S0)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    FittingConfig,
    VoxelFitResult,
    DispersedGKMResult,
    DelayedFitResult,
)
from dce_mri.kinetic_models import KineticModelSpec, DISPERSED_GKM
from dce_mri.signal_models import signal_to_concentration, concentration_to_signal, shift_aif


# =============================================================================
# Public: standard single-voxel fitter
# =============================================================================


def fit_single_voxel(
    sig:      np.ndarray,
    aif:      np.ndarray,
    time_var: np.ndarray,
    model:    KineticModelSpec,
    acq:      AcquisitionConfig,
    subject:  SubjectConfig,
    cfg:      FittingConfig,
    seed:     Optional[int] = None,
) -> VoxelFitResult:
    """
    Fit any KineticModelSpec to one voxel using LHS + TRF.

    The fitting domain (concentration vs signal space) is controlled by
    cfg.fit_space:
        'concentration' : convert sig → C_t first, then fit model to C_t
        'signal'        : fit model inside signal space (S0 fixed from baseline)

    Parameters
    ----------
    sig      : (T,) measured MRI signal
    aif      : (T,) arterial input function  (plasma mM)
    time_var : (T,) uniform time grid  (s)
    model    : KineticModelSpec (TCXM, GKM, or DISPERSED_GKM*)
               *For DISPERSED_GKM use fit_dispersed_gkm() instead
    acq      : AcquisitionConfig
    subject  : SubjectConfig  (T10, B1)
    cfg      : FittingConfig  (n_lhs, n_top, fit_space, ...)

    Returns
    -------
    VoxelFitResult

    """
    from dce_mri.signal_models import compute_S0

    # ── Build residual function ───────────────────────────────────────
    if cfg.fit_space == "concentration":
        Ct_meas = signal_to_concentration(sig, acq, subject.T10, subject.B1)
        target  = Ct_meas
        def residual_fn(p):
            return _residual_concentration(p, target, time_var, aif,
                                           model.fn, model.constraints)
    else:
        S0 = float(compute_S0(sig[None, :], acq)[0])
        target = sig
        def residual_fn(p):
            return _residual_signal(p, target, time_var, aif,
                                    model.fn, acq, subject.T10,
                                    subject.B1, S0, model.constraints)

    # ── Stage 1: LHS coarse search ────────────────────────────────────
    samples = _lhs_starts(cfg.n_lhs, model.lb, model.ub,
                          model.constraints, model.log_dims, seed)
    costs   = np.array([np.sum(residual_fn(p) ** 2) for p in samples])

    # ── Stage 2: TRF refinement on top candidates ─────────────────────
    top_idx = _top_finite_indices(costs, cfg.n_top)
    if len(top_idx) == 0:
        return VoxelFitResult(
            params  = model.p0.copy(),
            cost    = np.inf,
            success = False,
            S_meas  = sig,
            S_pred  = np.zeros_like(sig),
        )

    best_params, best_cost, best_status = None, np.inf, -1
    for idx in top_idx:
        params, cost, status = _trf_refine(
            samples[idx], residual_fn, model.lb, model.ub, cfg
        )
        if params is not None and cost < best_cost:
            best_params = params
            best_cost   = cost
            best_status = status

    if best_params is None:
        best_params = samples[top_idx[0]]
        best_status = -1

    # ── Reconstruct predicted signal ─────────────────────────────────
    Ct_pred = model.fn(best_params, time_var, aif)
    S0_out  = float(compute_S0(sig[None, :], acq)[0])
    S_pred  = concentration_to_signal(Ct_pred, acq, subject.T10,
                                      subject.B1, S0_out)

    return VoxelFitResult(
        params  = best_params,
        cost    = best_cost,
        success = best_status > 0,
        S_meas  = sig,
        S_pred  = S_pred,
    )

# =============================================================================
# Public: dispersed GKM fitter (loops over integer branch levels)
# =============================================================================
def fit_dispersed_gkm(
    sig:      np.ndarray,
    aif:      np.ndarray,
    time_var: np.ndarray,
    acq:      AcquisitionConfig,
    subject:  SubjectConfig,
    cfg:      FittingConfig,
    n_levels: int           = 6,
    seed:     Optional[int] = None,
) -> DispersedGKMResult:
    """
    Fit dispersed-AIF GKM at each integer branch level independently.

    For each n_level in 1…n_levels:
        - Fix n_level
        - Fit theta_kin = [Ktrans, ve, vp, t0] using LHS + TRF
        - dispersed_gkm_ct(theta_kin, time_var, aif, n_level) is the forward model
    Then select the level with lowest cost.

    Parameters
    ----------
    sig      : (T,) measured MRI signal
    aif      : (T,) global AIF
    time_var : (T,) time grid  (s)
    acq      : AcquisitionConfig
    subject  : SubjectConfig
    cfg      : FittingConfig
    n_levels : number of integer branch levels to try (default 6)

    Returns
    -------
    DispersedGKMResult
    """
    from dce_mri.kinetic_models import dispersed_gkm_ct

    per_level = {}
    for n in range(1, n_levels + 1):
        def model_fn_n(theta, t, a, _n=n):
            return dispersed_gkm_ct(theta, t, a, n_level=_n)

        # temporary spec for this level — same as DISPERSED_GKM but with fn set
        level_spec = KineticModelSpec(
            name        = f"DispersedGKM_L{n}",
            fn          = model_fn_n,
            param_names = DISPERSED_GKM.param_names,
            lb          = DISPERSED_GKM.lb,
            ub          = DISPERSED_GKM.ub,
            p0          = DISPERSED_GKM.p0,
            constraints = DISPERSED_GKM.constraints,
            log_dims    = DISPERSED_GKM.log_dims,
        )
        per_level[n] = fit_single_voxel(
            sig, aif, time_var, level_spec, acq, subject, cfg, seed
        )

    best_level = min(per_level, key=lambda n: per_level[n].cost)

    return DispersedGKMResult(
        best_level  = best_level,
        best_params = per_level[best_level].params,
        best_cost   = per_level[best_level].cost,
        per_level   = per_level,
        success     = per_level[best_level].success,
    )



# =============================================================================
# Internal helpers
# =============================================================================

def _lhs_starts(
    n:           int,
    lb:          np.ndarray,
    ub:          np.ndarray,
    constraints: Optional[callable] = None,
    log_dims:    Optional[list]     = None,
    seed:        Optional[int]      = None,
) -> np.ndarray:
    """
    Draw n Latin-Hypercube samples scaled to [lb, ub].

    Parameters
    ----------
    log_dims : list of parameter indices to sample log-uniformly
               e.g. [2, 3] for fp and ps in TCXM
               e.g. [0] for Ktrans in GKM
               None = all dimensions sampled uniformly
    """
    d       = len(lb)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit    = sampler.random(n=n)       # (n, d) in [0, 1]

    samples = np.empty_like(unit)
    for dim in range(d):
        if log_dims is not None and dim in log_dims:
            log_lb = np.log10(lb[dim])
            log_ub = np.log10(ub[dim])
            samples[:, dim] = 10.0 ** (log_lb + unit[:, dim] * (log_ub - log_lb))
        else:
            samples[:, dim] = lb[dim] + unit[:, dim] * (ub[dim] - lb[dim])

    # enforce constraints
    if constraints is not None:
        for i in range(n):
            for _ in range(20):
                if constraints(samples[i]):
                    break
                samples[i] *= 0.95
            if not constraints(samples[i]):
                samples[i] = 0.5 * (lb + ub)

    return samples

def _trf_refine(
    p0:          np.ndarray,
    residual_fn: callable,      # already plain or wrapped by the caller
    lb:          np.ndarray,
    ub:          np.ndarray,
    cfg:         FittingConfig,
) -> tuple[Optional[np.ndarray], float, int]:
    """
    Run one TRF least-squares refinement from p0.
    Returns (params, cost, status) where status > 0 means converged.
    """
    try:
        res = least_squares(
            fun      = residual_fn,
            x0       = p0,
            bounds   = (lb, ub),
            method   = "trf",
            jac      = "3-point",
            xtol     = cfg.xtol,
            ftol     = cfg.ftol,
            gtol     = cfg.gtol,
            max_nfev = cfg.max_nfev,
        )
        return res.x, float(np.sum(res.fun ** 2)), res.status
    except Exception:
        return None, np.inf, -1
    
def _top_finite_indices(
    costs: np.ndarray,
    n_top: int,
) -> np.ndarray:
    """
    Return indices of the n_top lowest finite costs.
    """
    finite = np.where(np.isfinite(costs))[0]
    if len(finite) == 0:
        return np.array([], dtype=int)
    return finite[np.argsort(costs[finite])[:n_top]]

def _residual_concentration(
    p:        np.ndarray,
    Ct_meas:  np.ndarray,
    time_var: np.ndarray,
    aif:      np.ndarray,
    model_fn: callable,
    constraints: Optional[callable] = None,
) -> np.ndarray:
    """
    Residual vector in concentration space: Ct_meas − model_fn(p, t, aif).
    Returns large values if constraints are violated.

   
    """
    if constraints is not None and not constraints(p):
       return np.ones_like(Ct_meas) * 1e6
    try:
        Ct_pred = model_fn(p, time_var, aif)
        if not np.all(np.isfinite(Ct_pred)):
            return np.ones_like(Ct_meas) * 1e6
        return Ct_meas - Ct_pred
    except Exception:
        return np.ones_like(Ct_meas) * 1e6


def _residual_signal(
    p:           np.ndarray,
    sig_meas:    np.ndarray,
    time_var:    np.ndarray,
    aif:         np.ndarray,
    model_fn:    callable,
    acq:         AcquisitionConfig,
    T10:         float,
    B1:          float,
    S0:          float,
    constraints: Optional[callable] = None,
) -> np.ndarray:
    if constraints is not None and not constraints(p):
        return np.ones_like(sig_meas) * 1e6
    try:
        Ct_pred  = model_fn(p, time_var, aif)
        if not np.all(np.isfinite(Ct_pred)):
            return np.ones_like(sig_meas) * 1e6
        S_pred = concentration_to_signal(Ct_pred, acq, T10, B1, S0)
        if not np.all(np.isfinite(S_pred)):
            return np.ones_like(sig_meas) * 1e6
        return sig_meas - S_pred
    except Exception:
        return np.ones_like(sig_meas) * 1e6    


# =============================================================================
# Public: grid search over tau values
# =============================================================================

def fit_with_delay_grid(
    sig:        np.ndarray,
    aif:        np.ndarray,
    time_var:   np.ndarray,
    model:      KineticModelSpec,
    acq:        AcquisitionConfig,
    subject:    SubjectConfig,
    cfg:        FittingConfig,
    tau_values: np.ndarray,
    seed:       Optional[int] = None,
    interp_method: str      = "linear",
) -> "DelayedFitResult":
    """
    Grid search over AIF time delays, fitting the full kinetic model
    at each tau independently.

    For each tau in tau_values:
        1. shifted_aif = shift_aif(aif, tau, time_var, acq)
        2. result = fit_single_voxel(sig, shifted_aif, ...)
    Return best tau + kinetic params by lowest cost.

    Parameters
    ----------
    sig        : (T,) measured MRI signal
    aif        : (T,) global AIF
    time_var   : (T,) time grid  (s)
    model      : KineticModelSpec
    acq        : AcquisitionConfig
    subject    : SubjectConfig
    cfg        : FittingConfig
    tau_values : 1D array of delay values to try  (seconds)
                 e.g. np.arange(-10, 31, 2.0)
                 Positive = AIF arrives later
                 Negative = AIF arrives earlier

    Returns
    -------
    DelayedFitResult

    Example tau_values
    ------------------
    Conservative:  np.arange(0, 30, 5.0)       # 6 fits
    Standard:      np.arange(-10, 31, 2.0)      # 21 fits
    Fine:          np.arange(-10, 31, 1.0)      # 41 fits
    """
    from dce_mri.config import DelayedFitResult

    per_tau = {}
    for tau in tau_values:
        shifted_aif      = shift_aif(aif, tau, time_var, acq, method=interp_method)
        per_tau[float(tau)] = fit_single_voxel(
            sig, shifted_aif, time_var, model, acq, subject, cfg, seed
        )

    best_tau = min(per_tau, key=lambda t: per_tau[t].cost)

    return DelayedFitResult(
        best_tau    = best_tau,
        best_params = per_tau[best_tau].params,
        best_cost   = per_tau[best_tau].cost,
        per_tau     = per_tau,
        success     = per_tau[best_tau].success,
        fit_mode    = "grid",
    )


# =============================================================================
# Public: continuous tau as free parameter in LHS + TRF
# =============================================================================

def fit_with_delay_continuous(
    sig:      np.ndarray,
    aif:      np.ndarray,
    time_var: np.ndarray,
    model:    KineticModelSpec,
    acq:      AcquisitionConfig,
    subject:  SubjectConfig,
    cfg:      FittingConfig,
    tau_lb:   float          = -10.0,
    tau_ub:   float          =  30.0,
    tau_p0:   float          =   0.0,
    seed:     Optional[int]  = None,
    interp_method: str      = "linear",
) -> "DelayedFitResult":
    """
    Fit kinetic model with tau as a continuous free parameter optimised
    jointly with the kinetic parameters via LHS + TRF.

    tau is appended as the last element of the parameter vector:
        theta_aug = [*theta_kin, tau]

    Inside the residual function:
        shifted_aif = shift_aif(aif, tau, time_var, acq)
        Ct = model.fn(theta_kin, time_var, shifted_aif)

    tau is sampled log-uniformly only if both bounds are positive.
    Otherwise it is sampled uniformly (since it spans zero).

    Parameters
    ----------
    tau_lb : lower bound for tau  (s).  Negative = allow earlier arrival.
    tau_ub : upper bound for tau  (s).
    tau_p0 : initial guess for tau  (s).  0 = no delay.

    Returns
    -------
    DelayedFitResult
    """
    from dce_mri.config import DelayedFitResult

    # ── Build augmented bounds and p0 ────────────────────────────────
    lb_aug = np.append(model.lb, tau_lb)
    ub_aug = np.append(model.ub, tau_ub)
    p0_aug = np.append(model.p0, tau_p0)

    # tau spans zero so never log-sample it — keep model's log_dims
    log_dims_aug = list(model.log_dims) if model.log_dims else []
    # tau is the last dimension — do not add to log_dims

    # ── Build augmented constraints ───────────────────────────────────
    # original constraints apply to kinetic params only (all but last)
    def constraints_aug(p_aug):
        if model.constraints is not None:
            return model.constraints(p_aug[:-1])
        return True

    # ── Build residual function ───────────────────────────────────────
    def _make_delayed_model_fn(base_model_fn, aif_, time_var_, acq_, method_):
        """
        Wrap base_model_fn to accept [*theta_kin, tau] and shift AIF
        internally before calling base_model_fn.
        """
        def delayed_fn(theta_aug, t, _aif_ignored):
            tau_val     = float(theta_aug[-1])
            theta_kin   = theta_aug[:-1]
            shifted_aif = shift_aif(aif_, tau_val, time_var_, acq_, method = method_)
            return base_model_fn(theta_kin, t, shifted_aif)
        return delayed_fn

    delayed_model_fn = _make_delayed_model_fn(
        model.fn, aif, time_var, acq, interp_method
    )

    # ── Build augmented KineticModelSpec ──────────────────────────────
    aug_spec = KineticModelSpec(
        name        = model.name + "_delay",
        fn          = delayed_model_fn,
        param_names = (*model.param_names, "tau"),
        lb          = lb_aug,
        ub          = ub_aug,
        p0          = p0_aug,
        constraints = constraints_aug,
        log_dims    = tuple(log_dims_aug) if log_dims_aug else None,
    )

    # ── Run standard fitter on augmented spec ─────────────────────────
    # Pass the original aif as dummy — delayed_model_fn ignores it
    # and uses the closed-over aif internally
    result = fit_single_voxel(
        sig, aif, time_var, aug_spec, acq, subject, cfg, seed
    )

    best_tau    = float(result.params[-1])
    best_params = result.params[:-1]

    return DelayedFitResult(
        best_tau    = best_tau,
        best_params = best_params,
        best_cost   = result.cost,
        per_tau     = {},        # not applicable for continuous mode
        success     = result.success,
        fit_mode    = "continuous",
    )

def _fit_concentration_curve(
    Ct_meas:  np.ndarray,
    aif:      np.ndarray,
    time_var: np.ndarray,
    model:    KineticModelSpec,
    cfg:      FittingConfig,
    seed:     Optional[int] = None,
) -> VoxelFitResult:
    """
    Fit kinetic model directly to a concentration curve.
    No signal conversion — Ct_meas is used as-is.
    Used internally by fitting_roi.py.
    """
    def residual_fn(p):
        return _residual_concentration(
            p, Ct_meas, time_var, aif, model.fn, model.constraints
        )

    samples = _lhs_starts(cfg.n_lhs, model.lb, model.ub,
                          model.constraints, model.log_dims, seed)
    costs   = np.array([np.sum(residual_fn(p) ** 2) for p in samples])
    top_idx = _top_finite_indices(costs, cfg.n_top)

    if len(top_idx) == 0:
        return VoxelFitResult(
            params  = model.p0.copy(),
            cost    = np.inf,
            success = False,
            S_meas  = Ct_meas,
            S_pred  = np.zeros_like(Ct_meas),
        )

    best_params, best_cost, best_status = None, np.inf, -1
    for idx in top_idx:
        params, cost, status = _trf_refine(
            samples[idx], residual_fn, model.lb, model.ub, cfg
        )
        if params is not None and cost < best_cost:
            best_params = params
            best_cost   = cost
            best_status = status

    if best_params is None:
        best_params = samples[top_idx[0]]
        best_status = -1

    Ct_pred = model.fn(best_params, time_var, aif)

    return VoxelFitResult(
        params  = best_params,
        cost    = best_cost,
        success = best_status > 0,
        S_meas  = Ct_meas,
        S_pred  = Ct_pred,
    )

def _fit_concentration_curve_with_delay_grid(
    Ct_meas:    np.ndarray,
    aif:        np.ndarray,
    time_var:   np.ndarray,
    model:      KineticModelSpec,
    acq:        AcquisitionConfig,
    cfg:        FittingConfig,
    tau_values: np.ndarray,
    seed:       Optional[int] = None,
    interp_method: str            = "linear",
) -> "DelayedFitResult":
    from dce_mri.config import DelayedFitResult
    from dce_mri.signal_models import shift_aif

    per_tau = {}
    for tau in tau_values:
        shifted_aif      = shift_aif(aif, float(tau), time_var, acq, method=interp_method)
        result           = _fit_concentration_curve(
            Ct_meas, shifted_aif, time_var, model, cfg, seed
        )
        per_tau[float(tau)] = result

    best_tau = min(per_tau, key=lambda t: per_tau[t].cost)
    return DelayedFitResult(
        best_tau    = best_tau,
        best_params = per_tau[best_tau].params,
        best_cost   = per_tau[best_tau].cost,
        per_tau     = per_tau,
        success     = per_tau[best_tau].success,
        fit_mode    = "grid",
    )


def _fit_concentration_curve_with_delay_continuous(
    Ct_meas:  np.ndarray,
    aif:      np.ndarray,
    time_var: np.ndarray,
    model:    KineticModelSpec,
    acq:      AcquisitionConfig,
    cfg:      FittingConfig,
    tau_lb:   float          = -10.0,
    tau_ub:   float          =  50.0,
    tau_p0:   float          =   0.0,
    seed:     Optional[int]  = None,
    interp_method: str            = "linear",
) -> "DelayedFitResult":
    from dce_mri.config import DelayedFitResult
    from dce_mri.signal_models import shift_aif

    lb_aug       = np.append(model.lb, tau_lb)
    ub_aug       = np.append(model.ub, tau_ub)
    p0_aug       = np.append(model.p0, tau_p0)
    log_dims_aug = list(model.log_dims) if model.log_dims else []

    def constraints_aug(p):
        if model.constraints is not None:
            return model.constraints(p[:-1])
        return True

    def residual_fn(p_aug):
        tau_val   = float(p_aug[-1])
        theta_kin = p_aug[:-1]
        if not constraints_aug(p_aug):
            return np.ones_like(Ct_meas) * 1e6
        try:
            shifted_aif = shift_aif(aif, tau_val, time_var, acq, method=interp_method)
            Ct_pred     = model.fn(theta_kin, time_var, shifted_aif)
            if not np.all(np.isfinite(Ct_pred)):
                return np.ones_like(Ct_meas) * 1e6
            return Ct_meas - Ct_pred
        except Exception:
            return np.ones_like(Ct_meas) * 1e6

    samples = _lhs_starts(
        cfg.n_lhs, lb_aug, ub_aug, constraints_aug,
        tuple(log_dims_aug) if log_dims_aug else None, seed
    )
    costs   = np.array([np.sum(residual_fn(p) ** 2) for p in samples])
    top_idx = _top_finite_indices(costs, cfg.n_top)

    best_params, best_cost, best_status = None, np.inf, -1
    for idx in top_idx:
        params, cost, status = _trf_refine(
            samples[idx], residual_fn, lb_aug, ub_aug, cfg
        )
        if params is not None and cost < best_cost:
            best_params = params
            best_cost   = cost
            best_status = status

    if best_params is None:
        best_params = p0_aug.copy()
        best_status = -1

    return DelayedFitResult(
        best_tau    = float(best_params[-1]),
        best_params = best_params[:-1],
        best_cost   = best_cost,
        per_tau     = {},
        success     = best_status > 0,
        fit_mode    = "continuous",
    )