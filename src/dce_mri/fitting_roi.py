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
)
from dce_mri.kinetic_models import KineticModelSpec
from dce_mri.signal_models import (
    signal_to_concentration,
    concentration_to_signal,
    compute_S0,
)
from dce_mri.fitting import (
    fit_single_voxel,
    fit_with_delay_grid,
    fit_with_delay_continuous,
    _fit_concentration_curve,
    _fit_concentration_curve_with_delay_grid,
    _fit_concentration_curve_with_delay_continuous,
)


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
) -> RoiFitResult:
    """
    Fit kinetic model to the mean curve of a masked ROI.

    Aggregation space is controlled by roi_cfg.average_space:
        'signal'        — average raw signal first, then convert to C(t)
        'concentration' — convert each voxel to C(t), then average

    Always fits in concentration space (fit_space='concentration')
    since Ct_mean is already a concentration curve after aggregation.
    """
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(
        sig_4d, mask, acq, subject, roi_cfg
    )

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
) -> DelayedFitResult:
    """
    ROI fitting with grid search over AIF time delays.

    Aggregates ROI to mean concentration curve first, then calls
    fit_with_delay_grid on that single curve.
    """
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(
        sig_4d, mask, acq, subject, roi_cfg
    )
    return _fit_concentration_curve_with_delay_grid(
        Ct_mean, aif, time_var, model, acq, fitting_cfg, tau_values, seed, interp_method=interp_method
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
) -> DelayedFitResult:
    """
    ROI fitting with tau as a continuous free parameter.

    Aggregates ROI to mean concentration curve first, then calls
    fit_with_delay_continuous on that single curve.
    """
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(
        sig_4d, mask, acq, subject, roi_cfg
    )
    return _fit_concentration_curve_with_delay_continuous(
        Ct_mean, aif, time_var, model, acq, fitting_cfg,
        tau_lb, tau_ub, tau_p0, seed, interp_method=interp_method
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

