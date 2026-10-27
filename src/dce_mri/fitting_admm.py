# -*- coding: utf-8 -*-
"""
fitting_admm.py
===============
Joint TV-regularised voxelwise fitting via ADMM.

Motivation
----------
Standard independent per-voxel fitting (fitting.py) treats each voxel
as completely separate.  When SNR is low, parameter maps can be noisy
even where the underlying tissue is spatially smooth.

This module applies isotropic Total Variation (TV) regularisation across
the spatial parameter maps, encouraging piecewise-smooth solutions while
preserving genuine edges (e.g. tumour boundary).

The ADMM (Alternating Direction Method of Multipliers) solver splits the
joint problem into two alternating sub-problems:
    Step 1 (voxel update)  — for each voxel, minimise the data fit term
                              plus a quadratic coupling term  (parallel)
    Step 2 (TV update)     — proximal operator of isotropic TV
                              (closed-form shrinkage)

Reference
---------
Boyd et al. "Distributed Optimization and Statistical Learning via the
Alternating Direction Method of Multipliers." Foundations and Trends in
Machine Learning, 2011.

Public API
----------
    run_admm_fitting(sig_4d, aif, time_var, model, acq, subject,
                     fitting_cfg, admm_cfg, mask)
        →  VoxelwiseResult

Internal helpers
----------------
    _admm_voxel_update(sig_chunk, aif, time_var, model, acq, subject,
                        fitting_cfg, z_chunk, u_chunk, rho)
        →  x_chunk   (updated voxel params for one worker chunk)

    _tv_prox(x_vol, lambda_tv, rho)
        →  z_vol     (TV proximal update, closed-form)

    _compute_residuals(x_vol, z_vol, u_vol)
        →  (primal_res, dual_res)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    FittingConfig,
    AdmmConfig,
    VoxelwiseResult,
)
from dce_mri.kinetic_models import KineticModelSpec


# =============================================================================
# Public: ADMM joint fitting entry point
# =============================================================================

def run_admm_fitting(
    sig_4d:      np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    admm_cfg:    AdmmConfig,
    mask:        Optional[np.ndarray] = None,
) -> VoxelwiseResult:
    """
    Fit kinetic model to a masked volume using ADMM with TV regularisation.

    Parameters
    ----------
    sig_4d      : (X, Y, Z, T) MRI signal volume
    aif         : (T,) global AIF
    time_var    : (T,) time grid  (s)
    model       : KineticModelSpec
    acq         : AcquisitionConfig
    subject     : SubjectConfig
    fitting_cfg : FittingConfig — controls inner voxel TRF solver
    admm_cfg    : AdmmConfig — controls outer ADMM loop
    mask        : (X, Y, Z) bool — voxels to include; None = all

    Returns
    -------
    VoxelwiseResult

    Algorithm outline
    -----------------
    ADMM minimises:
        min_x  Σ_v f_v(x_v)  +  lambda_tv · TV(x)
    where f_v(x_v) is the per-voxel data-fit cost and TV is isotropic
    total variation over the spatial parameter maps.

    Variables:
        x_vol : (X, Y, Z, K)  — voxel parameters (primal)
        z_vol : (X, Y, Z, K)  — TV auxiliary variable
        u_vol : (X, Y, Z, K)  — scaled dual variable

    ADMM iterations:
        1. x-update (parallel over voxels in chunks):
               x_v ← argmin_p  f_v(p) + (rho/2)‖p − (z_v − u_v)‖²
               Implemented as a warm-started TRF with the quadratic
               coupling term added to the residual.

        2. z-update (TV proximal, closed-form):
               z ← prox_{lambda_tv/rho · TV}(x + u)
               Isotropic TV prox via iterative shrinkage (e.g. Chambolle).

        3. u-update (dual ascent):
               u ← u + x − z

        4. Check convergence:
               primal residual ‖x − z‖ < tol_primal
               dual residual   ‖rho·(z_new − z_old)‖ < tol_dual

    Warm start option:
        If admm_cfg.warm_start = True, initialise x_vol from the
        standard per-voxel fit (run_voxelwise_fitting from volume.py)
        before entering the ADMM loop.

    TODO:
    1. Build mask, initialise x_vol, z_vol, u_vol (zeros or warm start)
    2. Outer loop: for iter in range(admm_cfg.n_admm_iter):
           a. x_vol = _admm_voxel_update(sig_4d, aif, time_var, model,
                                          acq, subject, fitting_cfg,
                                          z_vol, u_vol, admm_cfg.rho,
                                          admm_cfg.chunk_size, admm_cfg.n_workers)
           b. z_old = z_vol.copy()
              z_vol  = _tv_prox(x_vol + u_vol,
                                 admm_cfg.lambda_tv, admm_cfg.rho)
           c. u_vol += x_vol - z_vol
           d. primal, dual = _compute_residuals(x_vol, z_vol, u_vol)
              if primal < admm_cfg.tol_primal and dual < admm_cfg.tol_dual:
                  break
    3. Build VoxelwiseResult from x_vol and mask
    """
    raise NotImplementedError


# =============================================================================
# Internal: voxel update step (parallel, chunked)
# =============================================================================

def _admm_voxel_update(
    sig_4d:      np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    z_flat:      np.ndarray,
    u_flat:      np.ndarray,
    rho:         float,
    chunk_size:  Optional[int]  = None,
    n_workers:   Optional[int]  = None,
) -> np.ndarray:
    """
    Parallel voxel parameter update for one ADMM iteration.

    For each voxel v, solves:
        x_v ← argmin_p  ‖residual_v(p)‖²  +  (rho/2)‖p − w_v‖²
    where w_v = z_v − u_v is the ADMM coupling target.

    The quadratic coupling term is added to the TRF residual as
    extra "virtual" observations:
        augmented_residual(p) = [residual_v(p),
                                  sqrt(rho/2) * (p − w_v)]

    Voxels are dispatched in chunks to reduce multiprocessing overhead
    (see volume.py for the chunking pattern).

    Parameters
    ----------
    sig_4d    : (X, Y, Z, T) signal volume (already masked internally)
    z_flat    : (N_vox, K) current z values for masked voxels
    u_flat    : (N_vox, K) current u values for masked voxels
    rho       : ADMM penalty

    Returns
    -------
    x_flat : (N_vox, K) updated voxel parameters

    TODO:
    1. Extract masked voxel signals into flat array  (N_vox, T)
    2. Build w_flat = z_flat - u_flat
    3. Chunk voxels and dispatch to worker pool (same pattern as volume.py)
    4. Each worker calls _single_voxel_admm_update(sig_v, aif, time_var,
                                                    model, acq, subject,
                                                    fitting_cfg, w_v, rho)
    5. Reassemble and return x_flat
    """
    raise NotImplementedError


def _single_voxel_admm_update(
    sig_v:       np.ndarray,
    aif:         np.ndarray,
    time_var:    np.ndarray,
    model:       KineticModelSpec,
    acq:         AcquisitionConfig,
    subject:     SubjectConfig,
    fitting_cfg: FittingConfig,
    w_v:         np.ndarray,
    rho:         float,
) -> np.ndarray:
    """
    Solve the augmented single-voxel sub-problem for one ADMM step.

    Augments the residual with the ADMM coupling term and runs TRF.
    Uses the previous x_v (passed as p0) as a warm start.

    TODO:
    1. Build augmented_residual_fn:
           def aug_res(p):
               r_data = _residual_signal(p, sig_v, ...)  or concentration
               r_coup = np.sqrt(rho / 2) * (p - w_v)
               return np.concatenate([r_data, r_coup])
    2. Call _trf_refine from fitting.py with aug_res and warm-start p0=w_v
    3. Return best params (or w_v as fallback if TRF fails)
    """
    raise NotImplementedError


# =============================================================================
# Internal: TV proximal operator
# =============================================================================

def _tv_prox(
    v_vol:      np.ndarray,
    lambda_tv:  float,
    rho:        float,
) -> np.ndarray:
    """
    Proximal operator of isotropic TV regularisation.

        prox_{alpha · TV}(v) = argmin_z  alpha·TV(z) + (1/2)‖z - v‖²
    where alpha = lambda_tv / rho.

    Applied independently to each parameter channel (last axis).

    Recommended implementation: Chambolle's projection algorithm
    (Chambolle 2004, "An Algorithm for Total Variation Minimization").

    Parameters
    ----------
    v_vol      : (X, Y, Z, K) — current x + u
    lambda_tv  : TV regularisation weight
    rho        : ADMM penalty parameter

    Returns
    -------
    z_vol : (X, Y, Z, K)

    TODO:
    Implement Chambolle's iterative projection for 3D isotropic TV.
    For each parameter channel k:
        z_vol[..., k] = chambolle_prox(v_vol[..., k], alpha=lambda_tv/rho)

    Reference implementation:
        https://github.com/scikit-image/scikit-image (restoration.denoise_tv_chambolle)
        or implement directly from Chambolle 2004 Algorithm 1.

    Note: scikit-image.restoration.denoise_tv_chambolle can be used
    directly here as a starting point — it solves exactly this problem.
    """
    raise NotImplementedError


# =============================================================================
# Internal: convergence check
# =============================================================================

def _compute_residuals(
    x_flat: np.ndarray,
    z_flat: np.ndarray,
    u_flat: np.ndarray,
    rho:    float,
    z_prev: np.ndarray,
) -> tuple[float, float]:
    """
    Compute ADMM primal and dual residuals.

        primal = ‖x − z‖_F / sqrt(N)
        dual   = ‖rho · (z − z_prev)‖_F / sqrt(N)

    TODO: straightforward numpy norms
    """
    raise NotImplementedError
