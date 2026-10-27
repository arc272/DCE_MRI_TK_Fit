# -*- coding: utf-8 -*-
"""
fitting_roi.py
==============
Whole-ROI kinetic model fitting.

Pools signal across all ROI voxels into a single mean curve and fits
the kinetic model to that curve.  One parameter set per ROI.

Public API
----------
    fit_roi(sig_4d, mask, aif, time_var, model, acq, subject,
            fitting_cfg, roi_cfg, seed)
        →  RoiFitResult

    fit_roi_with_delay_grid(sig_4d, mask, aif, time_var, model,
                             acq, subject, fitting_cfg, roi_cfg,
                             tau_values, seed)
        →  DelayedFitResult

    fit_roi_with_delay_continuous(sig_4d, mask, aif, time_var, model,
                                   acq, subject, fitting_cfg, roi_cfg,
                                   tau_lb, tau_ub, tau_p0, seed)
        →  DelayedFitResult

    fit_multi_roi(sig_4d, seg_vol, aif, time_var, model, acq, subject,
                  fitting_cfg, roi_cfg, seed)
        →  dict[int, RoiFitResult]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    FittingConfig,
    RoiFittingConfig,
    RoiFitResult,
    DelayedFitResult,
    RoiDelayedFitResult,
    AifConfig,
    RelaxationConfig,
)
from dce_mri.kinetic_models import KineticModelSpec
from dce_mri.signal_models import (
    signal_to_concentration,
    concentration_to_signal,
    compute_S0,
    shift_aif,
)
from dce_mri.fitting import (
    fit_single_voxel,
    fit_with_delay_grid,
    fit_with_delay_continuous,
    _fit_concentration_curve,
    _fit_concentration_curve_with_delay_grid,
    _fit_concentration_curve_with_delay_continuous,
)

from dce_mri.water_exchange import WaterExchangeModelSpec
# =============================================================================
# Public: single ROI fitter
# =============================================================================


def fit_roi(
    sig_4d:      np.ndarray,
    mask:        np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    roi_cfg:     RoiFittingConfig,
    seed:        Optional[int] = None,
    wx_model:    Optional[WaterExchangeModelSpec] = None,
    aif_cfg:     Optional[AifConfig]             = None,
    relax_cfg:   Optional[RelaxationConfig]       = None,
) -> RoiFitResult:
    """
    Fit kinetic model to the mean curve of a masked ROI.

    Aggregation space is controlled by roi_cfg.average_space:
        'signal'        — average raw signal first, then convert to C(t)
        'concentration' — convert each voxel to C(t), then average
        
    Fitting space is COUPLED to roi_cfg.average_space:
    'signal'        — requires fitting_cfg.fit_space == 'signal'
                      (raises ValueError otherwise). Fits directly
                      against St_mean via fit_single_voxel.
    'concentration' — fits against Ct_mean via _fit_concentration_curve,
                      as before. fitting_cfg.fit_space is not checked
                      in this branch.
                      
    wx_model, aif_cfg, relax_cfg : forwarded to fit_single_voxel when
    roi_cfg.average_space == 'signal'. Requires average_space=
    'signal' (validated below) -- concentration-space averaging
    pools Ct across voxels before per-voxel ve/vb are available,
    which the wx forward model needs.
    """
    from dce_mri.fitting import _validate_wx_compatibility

    _validate_wx_compatibility(wx_model, fitting_cfg.fit_space, model,
                               time_var, aif, aif_cfg, relax_cfg,
                               average_space=roi_cfg.average_space)
    
    if roi_cfg.average_space == "signal" and fitting_cfg.fit_space != "signal":
        raise ValueError(
            "roi_cfg.average_space='signal' requires fitting_cfg.fit_space="
            "'signal' to fit against the aggregated signal curve directly; "
            f"got fitting_cfg.fit_space={fitting_cfg.fit_space!r}. Either "
            "build a FittingConfig with fit_space='signal', or use "
            "roi_cfg.average_space='concentration'."
        )
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(
        sig_4d, mask, acq, subject, roi_cfg
    )
    if roi_cfg.average_space == "signal" :
        print("fit_roi() is performing fit to signal time curve")
        result  = fit_single_voxel(
            St_mean, aif, time_var, model, acq, subject, fitting_cfg, seed,
            wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        )

        St_pred = result.S_pred
        if wx_model is None:
            Ct_pred = model.fn(result.params, time_var, aif)
        else:
            # Ct_pred has no single well-defined meaning under wx --
            # Cv/Ce are separate pools, not summed the FXL way. Report
            # their sum for plotting continuity, but this is NOT what
            # the wx forward model actually used to produce St_pred.
            Cv_pred, Ce_pred = model.fn_compartments(result.params, time_var, aif)
            Ct_pred = Cv_pred + Ce_pred


        return RoiFitResult(
            params        = result.params,
            cost          = result.cost,
            success       = result.success,
            Ct_mean       = Ct_mean,
            Ct_pred       = Ct_pred,
            St_mean       = St_mean,
            St_pred       = St_pred,
            n_voxels      = n_voxels,
            average_space = roi_cfg.average_space,
            wx_params     = getattr(result, "wx_params", None),
            wx_model_name = getattr(result, "wx_model_name", None),
        )
      
    else:
        print("fit_roi() is performing fit to concentration time curve")

        # force concentration space — Ct_mean is already concentration
        result = _fit_concentration_curve(
            Ct_mean, aif, time_var, model, fitting_cfg, seed
        )
    
        Ct_pred = model.fn(result.params, time_var, aif)
        S0      = float(compute_S0(St_mean[None, :], acq)[0])
        St_pred = concentration_to_signal(Ct_pred, acq, subject.T10,
                                          subject.B1, S0)

        return RoiFitResult(
            params        = result.params,
            cost          = result.cost,
            success       = result.success,
            Ct_mean       = Ct_mean,
            Ct_pred       = Ct_pred,
            St_mean       = St_mean,
            St_pred       = St_pred,
            n_voxels      = n_voxels,
            average_space = roi_cfg.average_space,
            wx_params     = getattr(result, "wx_params", None),
            wx_model_name = getattr(result, "wx_model_name", None),
        )


# =============================================================================
# Public: ROI fitting with grid delay search
# =============================================================================

def fit_roi_with_delay_grid(
    sig_4d:      np.ndarray,
    mask:        np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    roi_cfg:     RoiFittingConfig,
    tau_values:  np.ndarray,
    seed:        Optional[int] = None,
    interp_method: str         = "linear",
    wx_model: Optional[WaterExchangeModelSpec] = None,
    aif_cfg:  Optional[AifConfig] = None,
    relax_cfg:Optional[RelaxationConfig] = None,
) -> RoiDelayedFitResult:
    """
    Fit kinetic model to the mean curve of a masked ROI.
    ROI fitting with grid search over AIF time delays.

    Aggregation space is controlled by roi_cfg.average_space:
        'signal'        — average raw signal first, then convert to C(t)
        'concentration' — convert each voxel to C(t), then average
        
    Fitting space is COUPLED to roi_cfg.average_space:
    'signal'        — requires fitting_cfg.fit_space == 'signal'
                      (raises ValueError otherwise). Fits directly
                      against St_mean via fit_with_delay_grid.
    'concentration' — fits against Ct_mean via _fit_concentration_curve_with_delay_grid.
                      fitting_cfg.fit_space is not checked in this branch.
    """
    from dce_mri.fitting import _validate_wx_compatibility
    
    _validate_wx_compatibility(wx_model, fitting_cfg.fit_space, model,
                           time_var, aif, aif_cfg, relax_cfg,
                           average_space=roi_cfg.average_space)
    
    if roi_cfg.average_space == "signal" and fitting_cfg.fit_space != "signal":
        raise ValueError(
            "roi_cfg.average_space='signal' requires fitting_cfg.fit_space="
            "'signal' to fit against the aggregated signal curve directly; "
            f"got fitting_cfg.fit_space={fitting_cfg.fit_space!r}. Either "
            "build a FittingConfig with fit_space='signal', or use "
            "roi_cfg.average_space='concentration'."
        )
    
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(
        sig_4d, mask, acq, subject, roi_cfg
    )
    if roi_cfg.average_space == "signal":
        print("fit_roi_with_delay_grid() is performing fit to signal time curve")
        delayed= fit_with_delay_grid(
            St_mean, aif, time_var, model, acq, subject, fitting_cfg,
            tau_values, seed, interp_method=interp_method,
            wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        )
    else:
        print("fit_roi_with_delau_grid() is performing fit to concentration time curve")
        delayed= _fit_concentration_curve_with_delay_grid(
            Ct_mean, aif, time_var, model, acq, fitting_cfg, tau_values, seed, interp_method=interp_method
        )
    

    shifted_aif = shift_aif(aif, delayed.best_tau, time_var, acq, method=interp_method)
    S0          = float(compute_S0(St_mean[None, :], acq)[0])
   
    
    if wx_model is None:
        Ct_pred = model.fn(delayed.best_params, time_var, shifted_aif)
        St_pred = concentration_to_signal(Ct_pred, acq, subject.T10, subject.B1, S0)
    else:
        Cv_pred, Ce_pred = model.fn_compartments(delayed.best_params, time_var, shifted_aif)
        Ct_pred = Cv_pred + Ce_pred
        ve = delayed.best_params[model.param_names.index("ve")]
        vb = delayed.best_params[model.param_names.index("vp")]
        if wx_model.n_params == 0:
            theta_wx = ()                    # NXL: wx_params is empty
        else:
            tau_b = relax_cfg.tau_b_fixed if not relax_cfg.fit_tau_b else delayed.wx_params[0]
            tau_i = delayed.wx_params[-1]
            theta_wx = (tau_b, tau_i)
        St_pred = wx_model.to_signal(Cv_pred, Ce_pred, ve, vb, theta_wx,
                                     acq, subject, aif_cfg, relax_cfg, S0)

    return RoiDelayedFitResult(
        best_tau      = delayed.best_tau,
        best_params   = delayed.best_params,
        best_cost     = delayed.best_cost,
        per_tau       = delayed.per_tau,
        success       = delayed.success,
        fit_mode      = delayed.fit_mode,
        Ct_mean       = Ct_mean,
        Ct_pred       = Ct_pred,
        St_mean       = St_mean,
        St_pred       = St_pred,
        n_voxels      = n_voxels,
        average_space = roi_cfg.average_space,
        wx_params     = delayed.wx_params,
        wx_model_name = delayed.wx_model_name,
    )


# =============================================================================
# Public: ROI fitting with continuous delay
# =============================================================================

def fit_roi_with_delay_continuous(
    sig_4d:      np.ndarray,
    mask:        np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    roi_cfg:     RoiFittingConfig,
    tau_lb:      float         = -10.0,
    tau_ub:      float         =  30.0,
    tau_p0:      float         =   0.0,
    seed:        Optional[int] = None,
    interp_method: str         = "linear",
    wx_model: Optional[WaterExchangeModelSpec] = None,
    aif_cfg:  Optional[AifConfig] = None,
    relax_cfg:Optional[RelaxationConfig] = None,
) -> RoiDelayedFitResult:
    """
    Fit kinetic model to the mean curve of a masked ROI.
    ROI fitting continuously over AIF time delays.

    Aggregation space is controlled by roi_cfg.average_space:
        'signal'        — average raw signal first, then convert to C(t)
        'concentration' — convert each voxel to C(t), then average
        
    Fitting space is COUPLED to roi_cfg.average_space:
    'signal'        — requires fitting_cfg.fit_space == 'signal'
                      (raises ValueError otherwise). Fits directly
                      against St_mean via fit_with_delay_continuous.
    'concentration' — fits against Ct_mean via _fit_concentration_curve_with_delay_continuous.
                      fitting_cfg.fit_space is not checked in this branch.
    """
    from dce_mri.fitting import _validate_wx_compatibility

    _validate_wx_compatibility(wx_model, fitting_cfg.fit_space, model,
                               time_var, aif, aif_cfg, relax_cfg,
                               average_space=roi_cfg.average_space)
    
    if roi_cfg.average_space == "signal" and fitting_cfg.fit_space != "signal":
        raise ValueError(
            "roi_cfg.average_space='signal' requires fitting_cfg.fit_space="
            "'signal' to fit against the aggregated signal curve directly; "
            f"got fitting_cfg.fit_space={fitting_cfg.fit_space!r}. Either "
            "build a FittingConfig with fit_space='signal', or use "
            "roi_cfg.average_space='concentration'."
        )
    
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(
        sig_4d, mask, acq, subject, roi_cfg
    )
    if roi_cfg.average_space == "signal":
        print("fit_roi_with_delay_continuous() is performing fit to signal time curve")
        delayed= fit_with_delay_continuous(
            St_mean, aif, time_var, model, acq, subject, fitting_cfg,
            tau_lb, tau_ub, tau_p0, seed, interp_method=interp_method,
            wx_model=wx_model, aif_cfg=aif_cfg, relax_cfg=relax_cfg,
        )
    else:
        print("fit_roi_with_delay_continuous() is performing fit to concentration time curve")
        delayed= _fit_concentration_curve_with_delay_continuous(
            Ct_mean, aif, time_var, model, acq, fitting_cfg,
            tau_lb, tau_ub, tau_p0, seed, interp_method=interp_method,           
        )
    

    shifted_aif = shift_aif(aif, delayed.best_tau, time_var, acq, method=interp_method)
    S0          = float(compute_S0(St_mean[None, :], acq)[0])
    
    if wx_model is None:
        Ct_pred = model.fn(delayed.best_params, time_var, shifted_aif)
        St_pred = concentration_to_signal(Ct_pred, acq, subject.T10, subject.B1, S0)
    else:
        Cv_pred, Ce_pred = model.fn_compartments(delayed.best_params, time_var, shifted_aif)
        Ct_pred = Cv_pred + Ce_pred
        ve = delayed.best_params[model.param_names.index("ve")]
        vb = delayed.best_params[model.param_names.index("vp")]
        if wx_model.n_params == 0:
            theta_wx = ()                    # NXL: wx_params is empty
        else:
            tau_b = relax_cfg.tau_b_fixed if not relax_cfg.fit_tau_b else delayed.wx_params[0]
            tau_i = delayed.wx_params[-1]
            theta_wx = (tau_b, tau_i)
        St_pred = wx_model.to_signal(Cv_pred, Ce_pred, ve, vb, theta_wx,
                                     acq, subject, aif_cfg, relax_cfg, S0)

    return RoiDelayedFitResult(
        best_tau      = delayed.best_tau,
        best_params   = delayed.best_params,
        best_cost     = delayed.best_cost,
        per_tau       = delayed.per_tau,
        success       = delayed.success,
        fit_mode      = delayed.fit_mode,
        Ct_mean       = Ct_mean,
        Ct_pred       = Ct_pred,
        St_mean       = St_mean,
        St_pred       = St_pred,
        n_voxels      = n_voxels,
        average_space = roi_cfg.average_space,
        wx_params     = delayed.wx_params,
        wx_model_name = delayed.wx_model_name,
    )

# =============================================================================
# Public: multi-ROI fitter
# =============================================================================

def fit_multi_roi(
    sig_4d:      np.ndarray,
    seg_vol:     np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    roi_cfg:     RoiFittingConfig,
    seed:        Optional[int] = None,
) -> dict:
    """
    Fit one model per unique non-zero label in seg_vol.

    Returns {label: RoiFitResult}
    """
    labels = np.unique(seg_vol)
    labels = labels[labels != 0]
    return {
        int(label): fit_roi(
            sig_4d, seg_vol == label, aif, time_var,
            model, acq, subject, fitting_cfg, roi_cfg, seed
        )
        for label in labels
    }


# =============================================================================
# Internal: ROI signal aggregation
# =============================================================================

def _aggregate_roi_signal(
    sig_4d:  np.ndarray,
    mask:    np.ndarray,
    acq:     AcquisitionConfig,
    subject: SubjectConfig,
    cfg:     RoiFittingConfig,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Aggregate masked voxels into a single mean curve.

    Returns (Ct_mean (T,), n_voxels, St_mean (T,))
    Both Ct_mean and St_mean are always returned regardless of
    average_space so callers always have both for plotting.
    """
    sig_roi = sig_4d[mask, :]    # (N_vox, T)

    if cfg.average_space == "signal":
        # aggregate in signal space first
        if cfg.aggregation == "mean":
            St_mean = sig_roi.mean(axis=0)
        elif cfg.aggregation == "median":
            St_mean = np.median(sig_roi, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {cfg.aggregation!r}")

        Ct_mean  = signal_to_concentration(
            St_mean, acq, subject.T10, subject.B1
        )
        n_voxels = sig_roi.shape[0]

    elif cfg.average_space == "concentration":
        # convert each voxel then aggregate
        Ct_roi = signal_to_concentration(
            sig_roi, acq, subject.T10, subject.B1
        )

        # optionally keep top enhancing voxels
        if cfg.top_frac < 1.0:
            peak_enh = Ct_roi.max(axis=1)
            k        = max(1, int(np.ceil(cfg.top_frac * len(peak_enh))))
            top_idx  = np.argpartition(peak_enh, -k)[-k:]
            Ct_roi   = Ct_roi[top_idx, :]
            sig_roi  = sig_roi[top_idx, :]

        if cfg.aggregation == "mean":
            Ct_mean = Ct_roi.mean(axis=0)
            St_mean = sig_roi.mean(axis=0)
        elif cfg.aggregation == "median":
            Ct_mean = np.median(Ct_roi, axis=0)
            St_mean = np.median(sig_roi, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {cfg.aggregation!r}")

        n_voxels = Ct_roi.shape[0]

    else:
        raise ValueError(
            f"Unknown average_space: {cfg.average_space!r}. "
            f"Use 'signal' or 'concentration'."
        )

    return Ct_mean, n_voxels, St_mean

