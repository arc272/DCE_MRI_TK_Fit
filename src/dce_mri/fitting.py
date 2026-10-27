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
    FitModeConfig,
    VoxelFitResult,
    DispersedGKMResult,
    DelayedFitResult,
    AifConfig,
    RelaxationConfig,
)

from dce_mri.kinetic_models import KineticModelSpec, DISPERSED_GKM
from dce_mri.signal_models import signal_to_concentration, concentration_to_signal, shift_aif
from dce_mri.water_exchange import WaterExchangeModelSpec

# =============================================================================
# Public: validate fit model choice
# =============================================================================
def _validate_wx_compatibility(
    wx_model,
    fit_space:  str,
    model:      KineticModelSpec,
    time_var:   np.ndarray,
    aif:        np.ndarray,
    aif_cfg,
    relax_cfg,
    average_space:  Optional[str] = None,
) -> None:
    """
    Guardrail for water-exchange fitting. Raises ValueError on any
    incompatibility; returns None (no-op) if wx_model is None.

    Checks, in order:
      0. wx_model is None -> return immediately, nothing else runs
      1. aif_cfg was supplied (every 3-pool model needs blood T10 for
         the plasma pool), and relax_cfg was supplied when the model
         has free exchange parameters (SSM3 needs the tau bounds /
         fit_tau_b flag; NXL has n_params=0 and needs no relax_cfg)
      2. fit_space == 'signal' (wx models produce signal directly;
         they have no defined behavior in concentration space)
      3. model.fn_compartments is not None (wx needs Cv/Ce separately,
         not just their sum Ct)
      4. wx_model.n_pools matches what model.fn_compartments actually
         implies (n_curves + 1, for the intracellular pool). This is
         the check that rejects FXL: FXL.n_pools=1, but any model with
         fn_compartments implies 3. Use fit_single_voxel without
         wx_model for fast-exchange fitting instead.
      5. average_space == 'signal' when supplied (ROI callers only --
         wx models need per-voxel ve/vb before averaging makes sense;
         'concentration' mode averages Ct across voxels first, which
         throws away exactly that per-voxel split. Not checked when
         average_space is None, i.e. for fit_single_voxel.)
    """
    if wx_model is None:
        return

    if aif_cfg is None or (wx_model.n_params > 0 and relax_cfg is None):
        raise ValueError(
            f"wx_model={wx_model.name!r} requires aif_cfg"
            + (" and relax_cfg" if wx_model.n_params > 0 else "")
            + " to be supplied (aif_cfg carries blood T10 for the vascular "
            "pool baseline; relax_cfg carries tau bounds and the fit_tau_b "
            "flag)."
        )

    if fit_space != "signal":
        raise ValueError(
            f"wx_model={wx_model.name!r} requires cfg.fit_space='signal', "
            f"got {fit_space!r}. Water-exchange models produce signal "
            f"directly and have no defined concentration-space form."
        )

    if model.fn_compartments is None:
        raise ValueError(
            f"wx_model={wx_model.name!r} requires model.fn_compartments "
            f"to be set on {model.name!r} (need Cv/Ce separately, not "
            f"just their sum Ct)."
        )

    Cv, Ce = model.fn_compartments(model.p0, time_var, aif)
    n_conc_curves  = 2   # Cv, Ce -- fn_compartments always returns a pair
    n_pools_implied = n_conc_curves + 1   # + intracellular pool

    if wx_model.n_pools != n_pools_implied:
        raise ValueError(
            f"{wx_model.name} has n_pools={wx_model.n_pools}, but "
            f"{model.name} implies {n_pools_implied} pools "
            f"({n_conc_curves} concentration curves + intracellular). "
            f"For fast-exchange fitting, call this function without "
            f"wx_model rather than passing FXL."
        )
    if average_space is not None and average_space != "signal":
        raise ValueError(
            f"wx_model={wx_model.name!r} requires roi_cfg.average_space="
            f"'signal', got {average_space!r}. Averaging in concentration "
            f"space pools Ct across voxels before ve/vb are known "
            f"per-voxel, which wx_model.to_signal needs. Use "
            f"average_space='signal', or omit wx_model for concentration-"
            f"space ROI fitting."
        )

# =============================================================================
# Public: shared residual builder (used by NLLS, bootstrap AND Bayesian)
# =============================================================================
from dataclasses import dataclass as _dataclass
 
 
@_dataclass(frozen=True)
class ResidualBundle:
    """
    Everything a fitter needs to evaluate one parameter vector.
 
    Extracted from fit_single_voxel so that the NLLS optimiser, the
    bootstrap replicate fitter and the Bayesian log-likelihood all
    evaluate *the same* residual. Any drift between them would silently
    invalidate posterior-vs-bootstrap comparisons.
 
    Fields
    ------
    residual_fn : callable  p_aug -> residual vector (pred - target)
    predict_fn  : callable  p_aug -> predicted curve (no subtraction)
    lb, ub      : bounds for the FULL augmented vector (kinetic + wx)
    log_dims    : indices to sample log-uniformly, for the full vector
    n_kin       : length of the kinetic block (INCLUDES tau_delay when
                  an augmented delay spec was passed in)
    target      : the measured curve being fitted against
    param_names : names for the full augmented vector
    """
    residual_fn: object
    predict_fn:  object
    lb:          np.ndarray
    ub:          np.ndarray
    log_dims:    Optional[tuple]
    n_kin:       int
    target:      np.ndarray
    param_names: tuple
 
 
def build_residual(
    sig:      np.ndarray,
    aif:      np.ndarray,
    time_var: np.ndarray,
    model:    KineticModelSpec,
    acq:      AcquisitionConfig,
    subject:  SubjectConfig,
    cfg:      FittingConfig,
    wx_model: Optional[WaterExchangeModelSpec] = None,
    aif_cfg:  Optional[AifConfig] = None,
    relax_cfg:Optional[RelaxationConfig] = None,
    *,
    keep_mask:  Optional[np.ndarray] = None,
    on_failure: str = "penalty",
) -> ResidualBundle:
    """
    Build the residual function for one voxel.
 
    on_failure : 'penalty' -> return 1e6 everywhere (correct for
                 least-squares; a huge residual steers the optimiser away)
                 'nan'     -> return NaN everywhere, so a Bayesian caller
                 can convert to -inf. Feeding 1e6 into a Gaussian
                 log-density gives a finite but astronomically negative
                 number, which behaves differently near bounds than a
                 hard rejection -- hence the switch.
 
    keep_mask  : optional boolean mask over timepoints (bootstrap
                 frame-drop uses this). None -> all frames kept.
    """
    if on_failure not in ("penalty", "nan"):
        raise ValueError(f"on_failure must be 'penalty' or 'nan', got {on_failure!r}")
 
    _validate_wx_compatibility(wx_model, cfg.fit_space, model, time_var,
                               aif, aif_cfg, relax_cfg)
 
    from dce_mri.signal_models import compute_S0
 
    n_kin = len(model.p0)
    fail_val = 1e6 if on_failure == "penalty" else np.nan
 
    if wx_model is None:
        # ── fast-exchange path ──────────────────────────────────────
        if cfg.fit_space == "concentration":
            target = signal_to_concentration(sig, acq, subject.T10, subject.B1)
 
            def predict_fn(p):
                return model.fn(p, time_var, aif)
 
        elif cfg.fit_space == "signal":
            S0 = float(compute_S0(sig[None, :], acq)[0])
            target = sig
 
            def predict_fn(p):
                Ct = model.fn(p, time_var, aif)
                return concentration_to_signal(Ct, acq, subject.T10,
                                               subject.B1, S0)
        else:
            raise ValueError(
                "cfg.fit_space must be 'signal' or 'concentration', "
                f"got {cfg.fit_space!r}."
            )
 
        lb_use, ub_use = model.lb, model.ub
        log_dims_use   = model.log_dims
        names_use      = model.param_names
 
    else:
        # ── wx-aware path ───────────────────────────────────────────
        S0     = float(compute_S0(sig[None, :], acq)[0])
        target = sig
        ve_idx = model.param_names.index("ve")
        vp_idx = model.param_names.index("vp")

        if wx_model.n_params == 0:
            # NXL: no free exchange parameters -- p_aug IS theta_kin, so
            # bounds / log_dims / names are the kinetic model's own.
            lb_use, ub_use = model.lb, model.ub
            log_dims_use   = model.log_dims
            names_use      = model.param_names
        elif relax_cfg.fit_tau_b:
            lb_use = np.concatenate([model.lb, [relax_cfg.tau_b_lb, relax_cfg.tau_i_lb]])
            ub_use = np.concatenate([model.ub, [relax_cfg.tau_b_ub, relax_cfg.tau_i_ub]])
            log_dims_use = tuple(model.log_dims or ()) + (n_kin + 1,)
            names_use    = (*model.param_names, "tau_b", "tau_i")
        else:
            lb_use = np.concatenate([model.lb, [relax_cfg.tau_i_lb]])
            ub_use = np.concatenate([model.ub, [relax_cfg.tau_i_ub]])
            log_dims_use = tuple(model.log_dims or ()) + (n_kin,)
            names_use    = (*model.param_names, "tau_i")
 
        def predict_fn(p_aug):
            theta_kin = p_aug[:n_kin]
            if wx_model.n_params == 0:
                theta_wx = ()
            elif relax_cfg.fit_tau_b:
                theta_wx = (p_aug[n_kin], p_aug[n_kin + 1])
            else:
                theta_wx = (relax_cfg.tau_b_fixed, p_aug[n_kin])
            Cv, Ce = model.fn_compartments(theta_kin, time_var, aif)
            return wx_model.to_signal(
                Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx],
                theta_wx, acq, subject, aif_cfg, relax_cfg, S0)
 
    # ── wrap prediction into a guarded residual ─────────────────────
    tgt = target if keep_mask is None else target[keep_mask]
 
    def residual_fn(p_aug):
        p_aug = np.asarray(p_aug, dtype=float)
        if model.constraints is not None and not model.constraints(p_aug[:n_kin]):
            return np.full_like(tgt, fail_val)
        try:
            pred = predict_fn(p_aug)
        except Exception:
            return np.full_like(tgt, fail_val)
        if pred is None or not np.all(np.isfinite(pred)):
            return np.full_like(tgt, fail_val)
        if keep_mask is not None:
            pred = pred[keep_mask]
        return pred - tgt
 
    return ResidualBundle(
        residual_fn = residual_fn,
        predict_fn  = predict_fn,
        lb          = np.asarray(lb_use, dtype=float),
        ub          = np.asarray(ub_use, dtype=float),
        log_dims    = log_dims_use,
        n_kin       = n_kin,
        target      = target,
        param_names = names_use,
    )
 
def build_delay_augmented_spec(
    model:    KineticModelSpec,
    aif:      np.ndarray,
    time_var: np.ndarray,
    acq:      AcquisitionConfig,
    tau_lb:   float = -10.0,
    tau_ub:   float =  30.0,
    tau_p0:   float =   0.0,
    interp_method: str = "linear",
    wx_model: Optional[WaterExchangeModelSpec] = None,
) -> KineticModelSpec:
    """
    Build the tau-augmented KineticModelSpec used by joint (continuous)
    delay fitting.
 
    theta_aug = [*theta_kin, tau_delay]  -- tau is ALWAYS last in the
    kinetic block, and any wx parameters are appended after it by
    build_residual / fit_single_voxel.
 
    Extracted from fit_with_delay_continuous so the Bayesian sampler can
    reuse the identical augmentation instead of duplicating it.
    """
    lb_aug = np.append(model.lb, tau_lb)
    ub_aug = np.append(model.ub, tau_ub)
    p0_aug = np.append(model.p0, tau_p0)
 
    # tau spans zero so it is never log-sampled -- keep model's log_dims
    log_dims_aug = list(model.log_dims) if model.log_dims else []
 
    def constraints_aug(p_aug):
        if model.constraints is not None:
            return model.constraints(p_aug[:-1])
        return True
 
    def _wrap(base_fn):
        def wrapped(theta_aug, t, _aif_ignored):
            tau_val     = float(theta_aug[-1])
            theta_kin   = theta_aug[:-1]
            shifted_aif = shift_aif(aif, tau_val, time_var, acq,
                                    method=interp_method)
            return base_fn(theta_kin, t, shifted_aif)
        return wrapped
 
    delayed_compartments_fn = None
    if wx_model is not None:
        if model.fn_compartments is None:
            raise ValueError(
                f"wx_model={wx_model.name!r} requires model.fn_compartments, "
                f"but {model.name!r} has none set."
            )
        delayed_compartments_fn = _wrap(model.fn_compartments)
 
    return KineticModelSpec(
        name        = model.name + "_delay",
        fn          = _wrap(model.fn),
        fn_compartments = delayed_compartments_fn,
        param_names = (*model.param_names, "tau_delay"),
        lb          = lb_aug,
        ub          = ub_aug,
        p0          = p0_aug,
        constraints = constraints_aug,
        log_dims    = tuple(log_dims_aug) if log_dims_aug else None,
    )
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
    wx_model: Optional[WaterExchangeModelSpec] = None,
    aif_cfg:  Optional[AifConfig] = None,
    relax_cfg:Optional[RelaxationConfig] = None,
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
 
    # ── Build residual function (shared with bootstrap + Bayesian) ────
    bundle = build_residual(
        sig, aif, time_var, model, acq, subject, cfg,
        wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        on_failure="penalty",
    )
    residual_fn  = bundle.residual_fn
    lb_use       = bundle.lb
    ub_use       = bundle.ub
    log_dims_use = bundle.log_dims
    n_kin        = bundle.n_kin

    # ── Stage 1: LHS coarse search ────────────────────────────────────
    samples = _lhs_starts(cfg.n_lhs, lb_use, ub_use,
                          model.constraints, log_dims_use, seed)
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
            wx_params     = None if wx_model is None else np.full(ub_use.shape[0] - n_kin, np.nan),
            wx_model_name = None if wx_model is None else wx_model.name,
        )

    best_params, best_cost, best_status = None, np.inf, -1
    for idx in top_idx:
        params, cost, status = _trf_refine(
            samples[idx], residual_fn, lb_use, ub_use, cfg
        )
        if params is not None and cost < best_cost:
            best_params = params
            best_cost   = cost
            best_status = status

    if best_params is None:
        best_params = samples[top_idx[0]]
        best_status = -1
        
    # ── Reconstruct predicted signal ─────────────────────────────────
    theta_kin_best = best_params[:n_kin]
    S0_out = float(compute_S0(sig[None, :], acq)[0])

    if wx_model is None:
        Ct_pred = model.fn(theta_kin_best, time_var, aif)
        S_pred  = concentration_to_signal(Ct_pred, acq, subject.T10,
                                          subject.B1, S0_out)
        wx_params_out     = None
        wx_model_name_out = None
    else:
        if wx_model.n_params == 0:
            theta_wx_best = ()                      # NXL: nothing to unpack
        elif relax_cfg.fit_tau_b:
            theta_wx_best = (best_params[n_kin], best_params[n_kin + 1])
        else:
            theta_wx_best = (relax_cfg.tau_b_fixed, best_params[n_kin])

        #
        ve_idx = bundle.param_names.index("ve")
        vp_idx = bundle.param_names.index("vp")

        Cv_best, Ce_best = model.fn_compartments(theta_kin_best, time_var, aif)
        ve_best = theta_kin_best[ve_idx]
        vb_best = theta_kin_best[vp_idx]
        S_pred = wx_model.to_signal(
            Cv_best, Ce_best, ve_best, vb_best, theta_wx_best,
            acq, subject, aif_cfg, relax_cfg, S0_out)
        # empty array for NXL -- NOT None, which downstream code reads as
        # "no wx model was used" (VoxelFitResult.wx_model_name still says NXL)
        wx_params_out     = best_params[n_kin:]
        wx_model_name_out = wx_model.name

    return VoxelFitResult(
        params        = theta_kin_best,
        cost          = best_cost,
        success       = best_status > 0,
        S_meas        = sig,
        S_pred        = S_pred,
        wx_params     = wx_params_out,
        wx_model_name = wx_model_name_out,
    )

# =============================================================================
# Public: dispersed GKM fitter (loops over integer branch levels)
# =============================================================================
def fit_dispersed_gkm(
    sig, aif, time_var, acq, subject, cfg,
    n_levels=6, seed=None,
) -> DispersedGKMResult:
    from dce_mri.kinetic_models import DISPERSED_GKM_LEVELS
    from dce_mri.vascular_aif import compute_aic, akaike_weights

    per_level = {
        n: fit_single_voxel(sig, aif, time_var, DISPERSED_GKM_LEVELS[n],
                            acq, subject, cfg, seed)
        for n in range(1, n_levels + 1)
    }

    n_obs = len(time_var)
    aic_arr = np.array([
        compute_aic(per_level[n].cost, n_obs, n_param=4)
        for n in range(1, n_levels + 1)
    ])
    weights = akaike_weights(aic_arr)

    if not np.any(weights > 0):
        fallback_level = 1
        return DispersedGKMResult(
            best_level=fallback_level,
            best_params=per_level[fallback_level].params,
            best_cost=per_level[fallback_level].cost,
            per_level=per_level, weights=weights, aic=aic_arr,
            success=False,
        )

    best_level = int(np.argmax(weights)) + 1
    params_stack = np.stack([per_level[n].params for n in range(1, n_levels + 1)])
    cost_stack   = np.array([per_level[n].cost for n in range(1, n_levels + 1)])
    cost_stack_safe = np.where(np.isfinite(cost_stack), cost_stack, 0.0)

    best_params = np.tensordot(weights, params_stack, axes=(0, 0))
    best_cost   = float(np.sum(weights * cost_stack_safe))

    return DispersedGKMResult(
        best_level=best_level, best_params=best_params, best_cost=best_cost,
        per_level=per_level, weights=weights, aic=aic_arr,
        success=any(per_level[n].success for n in range(1, n_levels + 1)),
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
    max_redraws: int                = 30,
) -> np.ndarray:
    d       = len(lb)
    rng     = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit    = sampler.random(n=n)

    def scale(u_row):
        """Same scaling LHS itself uses -- one fresh point, correct
        marginal distribution per dimension, independent across dims."""
        out = np.empty(d)
        for dim in range(d):
            if log_dims is not None and dim in log_dims:
                log_lb, log_ub = np.log10(lb[dim]), np.log10(ub[dim])
                out[dim] = 10.0 ** (log_lb + u_row[dim] * (log_ub - log_lb))
            else:
                out[dim] = lb[dim] + u_row[dim] * (ub[dim] - lb[dim])
        return out

    samples = np.array([scale(unit[i]) for i in range(n)])

    if constraints is not None:
        for i in range(n):
            if constraints(samples[i]):
                continue
            for _ in range(max_redraws):
                candidate = scale(rng.random(d))
                if constraints(candidate):
                    samples[i] = candidate
                    break
            else:
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
    wx_model: Optional[WaterExchangeModelSpec] = None,
    aif_cfg:  Optional[AifConfig] = None,
    relax_cfg:Optional[RelaxationConfig] = None,
    
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
                 
    wx_model, aif_cfg, relax_cfg : forwarded to fit_single_voxel at
                 each tau. See fit_single_voxel / _validate_wx_compatibility
                 for requirements when wx_model is not None.

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
            sig, shifted_aif, time_var, model, acq, subject, cfg, seed,
            wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        )

    best_tau = min(per_tau, key=lambda t: per_tau[t].cost)

    return DelayedFitResult(
        best_tau    = best_tau,
        best_params = per_tau[best_tau].params,
        best_cost   = per_tau[best_tau].cost,
        per_tau     = per_tau,
        success     = per_tau[best_tau].success,
        fit_mode    = "grid",
        wx_params     = per_tau[best_tau].wx_params,
        wx_model_name = per_tau[best_tau].wx_model_name,
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
    wx_model: Optional[WaterExchangeModelSpec] = None,
    aif_cfg:  Optional[AifConfig] = None,
    relax_cfg:Optional[RelaxationConfig] = None,
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
    
    wx_model, aif_cfg, relax_cfg : forwarded to fit_single_voxel at
                 each tau. See fit_single_voxel / _validate_wx_compatibility
                 for requirements when wx_model is not None.
    Returns
    -------
    DelayedFitResult
    """
    from dce_mri.config import DelayedFitResult
    
    aug_spec = build_delay_augmented_spec(
        model, aif, time_var, acq,
        tau_lb=tau_lb, tau_ub=tau_ub, tau_p0=tau_p0,
        interp_method=interp_method, wx_model=wx_model,
    )
    # ── Run standard fitter on augmented spec ─────────────────────────
    # Pass the original aif as dummy — delayed_model_fn ignores it
    # and uses the closed-over aif internally
    result = fit_single_voxel(
        sig, aif, time_var, aug_spec, acq, subject, cfg, seed,
        wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
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
        wx_params     = result.wx_params,
        wx_model_name = result.wx_model_name,
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

# in fitting.py, near the other public/internal fitting functions

def call_fitter(
    sig_v:        np.ndarray,
    aif:          np.ndarray,
    time_var:     np.ndarray,
    model:        KineticModelSpec,
    acq:          AcquisitionConfig,
    subject:      SubjectConfig,
    fitting_cfg:  FittingConfig,
    fit_mode_cfg: "FitModeConfig",
    seed:         Optional[int] = None,
) -> dict:
    """
    Dispatch to the correct fitter based on fit_mode_cfg.mode, and
    return a normalized flat dict — used by volume.py's chunked
    parallel processing so it never needs to know about the different
    result types each fitter returns.
    """
    mode = fit_mode_cfg.mode

    if mode == "standard":
        result = fit_single_voxel(
            sig_v, aif, time_var, model, acq, subject, fitting_cfg, seed=seed
        )
    elif mode == "dispersed":
        result = fit_dispersed_gkm(
            sig_v, aif, time_var, acq, subject, fitting_cfg,
            n_levels=fit_mode_cfg.n_levels, seed=seed
        )
    elif mode == "delay_grid":
        result = fit_with_delay_grid(
            sig_v, aif, time_var, model, acq, subject, fitting_cfg,
            fit_mode_cfg.tau_values,
            interp_method=fit_mode_cfg.interp_method, seed=seed
        )
    elif mode == "delay_continuous":
        result = fit_with_delay_continuous(
            sig_v, aif, time_var, model, acq, subject, fitting_cfg,
            tau_lb=fit_mode_cfg.tau_lb,
            tau_ub=fit_mode_cfg.tau_ub,
            tau_p0=fit_mode_cfg.tau_p0,
            interp_method=fit_mode_cfg.interp_method,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown fit mode: {mode!r}")

    return normalize_fit_result(result, mode)


def normalize_fit_result(result, mode: str) -> dict:
    """
    Flatten any fit result type (VoxelFitResult, DispersedGKMResult,
    DelayedFitResult) into a common dict — used by volume.py for IPC
    across worker processes, and usable anywhere else a uniform
    result shape is convenient.
    """
    if mode == "standard":
        return {
            "params":  result.params,
            "cost":    result.cost,
            "success": result.success,
        }
    elif mode == "dispersed":
        return {
            "params":     result.best_params,
            "cost":       result.best_cost,
            "success":    result.success,
            "best_level": result.best_level,
        }
    else:  # delay_grid or delay_continuous
        return {
            "params":   result.best_params,
            "cost":     result.best_cost,
            "success":  result.success,
            "best_tau": result.best_tau,
        }

