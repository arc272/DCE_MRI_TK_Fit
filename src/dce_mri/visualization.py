# -*- coding: utf-8 -*-
"""
visualization.py
================
All plot_* functions for the DCE-MRI pipeline.

Every function here:
  - Accepts arrays and config objects, never file paths or dicts of files
  - Returns a plt.Figure so callers can save or display as needed
  - Has an optional save_path argument; if provided, saves automatically

Public API
----------
    plot_voxel_fit(time_var, S_meas, S_pred, params, model, ...)
    plot_aif_curves(time_var, global_aif, local_aif, ...)
    plot_parameter_maps(param_maps, z_slice, model, ...)
    plot_parameter_histograms(param_maps_or_array, model, ...)
    plot_bootstrap_envelope(sig_boot, Sig_refit, success, time_var, ...)
    plot_metric_panel(metrics_dict, sweep_values, metric_key, ...)
    plot_scatter_matrix(params_ok, cost_ok, param_names, ...)
    plot_dispersed_gkm_levels(dispersed_result, time_var, ...)
    plot_ace_signal_segments(sig, time_var, protocol, ...)
    plot_ace_voxel_fit(sig_meas, S_pred, time_var, protocol, ...)
    plot_admm_convergence(primal_res, dual_res, ...)
    plot_roi_fit(Ct_mean, Ct_pred, time_var, result, ...)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dce_mri.config import (
    VoxelFitResult,
    DispersedGKMResult,
    RoiFitResult,
)
from dce_mri.kinetic_models import KineticModelSpec


def plot_voxel_fit(
    time_var:  np.ndarray,
    result:    VoxelFitResult,
    model:     KineticModelSpec,
    title:     str           = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel plot: measured vs predicted signal/concentration + residual.

    TODO: copy from tcxm_bootstrap_pipeline.plot_voxel_fit and update
          to accept VoxelFitResult and KineticModelSpec
    """
    raise NotImplementedError


def plot_aif_curves(
    time_var:   np.ndarray,
    global_aif: np.ndarray,
    local_aif:  Optional[np.ndarray] = None,
    title:      str                  = "AIF curves",
    save_path:  Optional[str]        = None,
) -> plt.Figure:
    """
    Plot global AIF and optionally the vascular-dispersion-corrected local AIF.

    TODO: straightforward line plot
    """
    raise NotImplementedError


def plot_parameter_maps(
    param_maps: dict,
    z_slice:    Optional[int]    = None,
    model:      Optional[KineticModelSpec] = None,
    save_path:  Optional[str]    = None,
) -> plt.Figure:
    """
    2D slice view of all parameter maps side by side.

    TODO: copy from HNC_TCXM_NLLS_ANA_ST.plot_parameter_maps and update
          to accept param_maps dict + KineticModelSpec for titles
    """
    raise NotImplementedError


def plot_parameter_histograms(
    params:    np.ndarray,
    model:     KineticModelSpec,
    p_ref:     Optional[np.ndarray] = None,
    title:     str                  = "Parameter distributions",
    save_path: Optional[str]        = None,
) -> plt.Figure:
    """
    One histogram panel per parameter.

    TODO: copy from tcxm_bootstrap_pipeline.plot_parameter_histograms
          and update to use model.param_names for axis labels
    """
    raise NotImplementedError


def plot_bootstrap_envelope(
    sig_boot:     np.ndarray,
    Sig_refit:    np.ndarray,
    success_boot: np.ndarray,
    time_var:     np.ndarray,
    S_ref:        Optional[np.ndarray] = None,
    n_traces:     int                  = 20,
    ci:           float                = 0.95,
    title:        str                  = "Bootstrap signal envelope",
    save_path:    Optional[str]        = None,
) -> plt.Figure:
    """
    95% CI band + individual refit traces + reference signal.

    TODO: copy from ace_mri_integration_example.plot_bootstrap_signal_envelope
    """
    raise NotImplementedError


def plot_metric_panel(
    metrics_dict: dict,
    sweep_values: np.ndarray,
    metric_key:   str,
    param_names:  list,
    ylabel:       str           = "",
    title:        str           = "",
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Per-parameter metric (bias, CoV, etc.) vs sweep variable.

    TODO: copy from tcxm_bootstrap_pipeline.plot_metric_panel and generalise
          to accept any number of methods in metrics_dict
    """
    raise NotImplementedError


def plot_scatter_matrix(
    params_ok:   np.ndarray,
    cost_ok:     Optional[np.ndarray] = None,
    param_names: Optional[list]       = None,
    title:       str                  = "Bootstrap scatter matrix",
    save_path:   Optional[str]        = None,
) -> plt.Figure:
    """
    Corner scatter matrix; coloured by cost if provided.

    TODO: copy from tcxm_bootstrap_pipeline.plot_scatter_matrix
    """
    raise NotImplementedError


def plot_dispersed_gkm_levels(
    result:    DispersedGKMResult,
    time_var:  np.ndarray,
    aif:       np.ndarray,
    title:     str           = "Dispersed GKM — per-level fits",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show predicted C(t) for each branch level alongside measured curve,
    with AIC weights annotated.

    This is the diagnostic plot for understanding which branch level
    the model selected and how it compares to the alternatives.

    TODO: new function — loop over result.per_level, plot each C_pred,
          annotate with cost and AIC weight, highlight best_level
    """
    raise NotImplementedError


def plot_admm_convergence(
    primal_residuals: np.ndarray,
    dual_residuals:   np.ndarray,
    tol_primal:       float,
    tol_dual:         float,
    save_path:        Optional[str] = None,
) -> plt.Figure:
    """
    Plot ADMM primal and dual residuals vs iteration number.

    TODO: new function — two-panel semilogy plot with tolerance lines
    """
    raise NotImplementedError


def plot_roi_fit(
    time_var:  np.ndarray,
    result:    RoiFitResult,
    model:     KineticModelSpec,
    title:     str           = "ROI mean curve fit",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mean ROI concentration curve vs model fit + residual.

    TODO: similar to plot_voxel_fit but labels indicate ROI mean
    """
    raise NotImplementedError
