# -*- coding: utf-8 -*-
"""
config.py
=========
All configuration dataclasses and result types for the DCE-MRI pipeline.

Design rules
------------
- Every dataclass is frozen=True (immutable after construction).
- No imports from other dce_mri submodules here.
- No business logic — only data fields and simple derived properties.
- Result types carry raw numpy arrays; they are the outputs of internal
  functions and workflow methods.

Usage
-----
    from dce_mri.config import (
        AcquisitionConfig,
        SubjectConfig,
        FittingConfig,
        BootstrapConfig,
        VoxelwiseConfig,
        AdmmConfig,
        RoiFittingConfig,
        VoxelFitResult,
        DispersedGKMResult,
        VoxelwiseResult,
        BootstrapResult,
        RoiFitResult,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# Acquisition / sequence parameters
# =============================================================================

@dataclass(frozen=True)
class AcquisitionConfig:
    """
    MR sequence parameters — identical for every subject in a cohort.

    Fields
    ------
    TR      : repetition time  (ms)
    FA      : nominal flip angle  (degrees)
    r1      : CA relaxivity  (mM⁻¹ s⁻¹)
    nbase   : slice selecting pre-contrast baseline frames
              e.g. slice(0, 7) means frames 0-6 are baseline
    upsample: internal convolution upsampling factor for all forward models
    """
    TR:       float = 3.64
    FA:       float = 10.0
    r1:       float = 3.8
    nbase:    slice = field(default_factory=lambda: slice(0, 7))
    upsample: int   = 50
    
    def __post_init__(self):
        if not isinstance(self.nbase, slice):
            raise TypeError(
                f"nbase must be a slice object (e.g. slice(0, 7)), "
                f"got {type(self.nbase).__name__} = {self.nbase!r}. "
                f"Did you mean slice(0, {self.nbase})?"
            )


# =============================================================================
# Per-subject parameters
# =============================================================================
@dataclass(frozen=True)
class SubjectConfig:
    """
    Per-subject tissue parameters and file paths.

    Fields
    ------
    subject_id  : string identifier used in output filenames
    T10         : pre-contrast T1  (ms)
    B1          : RF transmit field scaling factor
    data_root   : base directory for all subject data
    dce_path_override  : if provided, used instead of the default pattern
    aif_path_override  : if provided, used instead of the default pattern
    mask_path_override : if provided, used instead of the default pattern
    """
    subject_id: str
    T10:        float = 1530.0
    B1:         float = 1.0
    data_root:  str   = "/mnt/labspace/projects/hnc"

    dce_path_override:  Optional[str] = None
    aif_path_override:  Optional[str] = None
    mask_path_override: Optional[str] = None

    @property
    def dce_path(self) -> str:
        if self.dce_path_override is not None:
            return self.dce_path_override
        return (f"{self.data_root}/deescal/{self.subject_id}/"
                "motion_correction/L=4e-3_syncc.nii.gz")

    @property
    def aif_path(self) -> str:
        if self.aif_path_override is not None:
            return self.aif_path_override
        return (f"{self.data_root}/deescal/{self.subject_id}/"
                "calc_AIF/mean_top5_AIF_cca.csv")

    @property
    def mask_path(self) -> str:
        if self.mask_path_override is not None:
            return self.mask_path_override
        return (f"{self.data_root}/deescal/Segmentation/"
                f"{self.subject_id}_ROI.nii.gz")

    @property
    def out_dir(self) -> str:
        return f"{self.data_root}/results/{self.subject_id}"
# =============================================================================
# AIF extraction
# =============================================================================
@dataclass(frozen=True)
class AifConfig:
    """
    Controls AIF extraction from a vascular ROI (e.g. aorta, carotid).

    Hematocrit correction and blood T1 are specific to the AIF extraction
    step — they don't apply to tissue ROIs, which is why this is a
    separate config from AcquisitionConfig and SubjectConfig.

    Fields
    ------
    T10         : blood T1  (ms).  Typically higher than tissue T10.
                  Literature values: ~1600 ms at 3T for blood.
    hematocrit  : whole-blood → plasma volume fraction correction.
                  C_plasma = C_blood / (1 - hematocrit)
                  Typical large-vessel value: 0.42
    top_frac    : fraction of ROI voxels kept by peak enhancement,
                  before averaging.  1.0 = use all voxels in the mask.
    nbase       : baseline frame slice for the AIF ROI specifically.
                  Can differ from the tissue acquisition baseline if
                  the AIF and tissue have different bolus timing
                  relative to scan start.
    """
    T10:        float = 1600.0
    hematocrit: float = 0.42
    top_frac:   float = 1.0
    nbase:      slice  = field(default_factory=lambda: slice(0, 3))

# =============================================================================
# Fitting / optimisation parameters
# =============================================================================

@dataclass(frozen=True)
class FittingConfig:
    """
    Controls the two-stage LHS + TRF single-voxel optimiser.

    Fields
    ------
    n_lhs   : number of Latin Hypercube starting points in Stage 1
    n_top   : how many of the best LHS points are refined by TRF in Stage 2
    xtol    : TRF parameter tolerance
    ftol    : TRF function tolerance
    gtol    : TRF gradient tolerance
    max_nfev: maximum TRF function evaluations per refinement
    fit_space: 'concentration' or 'signal' — which domain to fit in
    """
    n_lhs:     int   = 200
    n_top:     int   = 10
    xtol:      float = 1e-6
    ftol:      float = 1e-6
    gtol:      float = 1e-6
    max_nfev:  int   = 500
    fit_space: str   = "signal"   # "concentration" | "signal"


@dataclass(frozen=True)
class DelayedFitResult:
    """
    Result of fitting a kinetic model with AIF time delay.

    Fields
    ------
    best_tau    : best delay in seconds (positive = AIF arrives later)
    best_params : kinetic parameters at best_tau (model-specific, no tau)
    best_cost   : RSS at best solution
    per_tau     : {tau: VoxelFitResult} — populated for grid mode only
                  empty dict for continuous mode
    success     : convergence flag
    fit_mode    : 'grid' or 'continuous'
    """
    best_tau:    float
    best_params: np.ndarray
    best_cost:   float
    per_tau:     dict
    success:     bool
    fit_mode:    str
# =============================================================================
# Bootstrap uncertainty parameters
# =============================================================================

@dataclass(frozen=True)
class BootstrapConfig:
    """
    Controls bootstrap uncertainty experiments.

    Fields
    ------
    B    : number of bootstrap replicates
    seed : RNG seed for reproducibility
    """
    B:    int = 1000
    seed: int = 42


# =============================================================================
# Voxelwise volume processing parameters
# =============================================================================

@dataclass(frozen=True)
class VoxelwiseConfig:
    """
    Controls parallel voxelwise volume fitting.

    Fields
    ------
    n_workers     : number of parallel worker processes
                    (None = cpu_count - 1)
    chunk_size    : number of voxels per worker chunk.
                    Larger chunks reduce multiprocessing overhead but
                    increase memory per worker.
                    None = auto (n_voxels // n_workers)
    min_cost      : voxels with final cost > min_cost are flagged as failed
    n_qc_plots    : save QC plots for the first N successful voxels
    save_dir      : output directory for NIfTI maps and QC plots
    output_prefix : filename prefix for saved NIfTI parameter maps
    """
    n_workers:     Optional[int]  = None
    chunk_size:    Optional[int]  = None
    min_cost:      float          = 1e-6
    n_qc_plots:    int            = 5
    save_dir:      str            = "."
    output_prefix: str            = "tcxm_"


# =============================================================================
# ADMM joint TV-regularised fitting parameters
# =============================================================================

@dataclass(frozen=True)
class AdmmConfig:
    """
    Controls the joint TV-regularised voxelwise fitting via ADMM.

    This solver fits all voxels in a volume jointly, applying isotropic
    total variation regularisation across the spatial parameter maps to
    promote piecewise-smooth solutions while still allowing sharp edges.

    Fields
    ------
    n_admm_iter   : number of outer ADMM iterations
    rho           : ADMM penalty parameter (step size)
    lambda_tv     : TV regularisation weight.
                    0 → no regularisation (reduces to independent voxel fits)
                    Larger values → smoother parameter maps
    n_inner_iter  : number of inner voxelwise TRF iterations per ADMM step
    tol_primal    : primal residual convergence tolerance
    tol_dual      : dual residual convergence tolerance
    warm_start    : if True, initialise from the standard voxelwise fit
    n_workers     : parallel workers for the inner voxel update step
    chunk_size    : voxels per worker chunk (same as VoxelwiseConfig)
    save_dir      : output directory
    output_prefix : filename prefix for saved NIfTI maps
    """
    n_admm_iter:  int   = 50
    rho:          float = 1.0
    lambda_tv:    float = 0.1
    n_inner_iter: int   = 10
    tol_primal:   float = 1e-4
    tol_dual:     float = 1e-4
    warm_start:   bool  = True
    n_workers:    Optional[int] = None
    chunk_size:   Optional[int] = None
    save_dir:     str   = "."
    output_prefix: str  = "tcxm_admm_"


# =============================================================================
# Whole-ROI fitting parameters
# =============================================================================

@dataclass(frozen=True)
class RoiFittingConfig:
    """
    Controls whole-ROI mean-curve fitting.

    Fields
    ------
    aggregation   : how to pool voxels — 'mean' or 'median'
    top_frac      : fraction of voxels kept by peak enhancement before
                    aggregation.  1.0 = all voxels, 0.1 = top 10%
                    only applies when average_space = 'concentration'
    average_space : 'signal'        — average raw MRI signal across voxels
                                      then convert mean signal → concentration
                    'concentration' — convert each voxel to concentration
                                      then average concentration curves
    save_dir      : output directory
    """
    aggregation:   str   = "mean"
    top_frac:      float = 1.0
    average_space: str   = "signal"    # 'signal' or 'concentration'
    save_dir:      str   = "."

# =============================================================================
# Result types
# =============================================================================

@dataclass(frozen=True)
class VoxelFitResult:
    """
    Result of fitting one kinetic model to one voxel.

    Fields
    ------
    params  : fitted parameter vector (model-specific)
    cost    : final residual sum of squares
    success : True if the TRF converged (status > 0)
    S_meas  : measured signal or concentration curve  (T,)
    S_pred  : model-predicted signal or concentration  (T,)
    """
    params:  np.ndarray
    cost:    float
    success: bool
    S_meas:  np.ndarray
    S_pred:  np.ndarray


@dataclass(frozen=True)
class DispersedGKMResult:
    """
    Result of fitting the dispersed-AIF GKM across all integer branch levels.

    Fields
    ------
    best_level  : integer branch level with lowest fitting cost (1-indexed)
    best_params : [Ktrans, ve, vp, t0] at best_level
    best_cost   : RSS at best_level
    per_level   : dict {n_level (int): VoxelFitResult} for all levels tested
    success     : True if at least one level converged
    """
    best_level:  int
    best_params: np.ndarray     # [Ktrans, ve, vp, t0]
    best_cost:   float
    per_level:   dict           # {1: VoxelFitResult, 2: ..., ...}
    success:     bool


@dataclass(frozen=True)
class VoxelwiseResult:
    """
    Result of running any voxelwise fitter over a masked volume.

    Fields
    ------
    param_maps : dict of parameter name → (X, Y, Z) spatial map
                 keys depend on the model:
                   TCXM          : 've', 'vp', 'fp', 'ps'
                   GKM           : 'Ktrans', 've', 'vp'
                   DispersedGKM  : 'Ktrans', 've', 'vp', 't0', 'best_level'
    cost_map   : (X, Y, Z) map of final RSS per voxel
    success_map: (X, Y, Z) bool map
    n_voxels   : total voxels attempted
    n_success  : number of successful fits
    elapsed_s  : total wall-clock time (seconds)
    """
    param_maps:  dict
    cost_map:    np.ndarray
    success_map: np.ndarray
    n_voxels:    int
    n_success:   int
    elapsed_s:   float


@dataclass(frozen=True)
class BootstrapResult:
    """
    Result of a bootstrap uncertainty run for one voxel / one method.

    Fields
    ------
    params_boot  : (B, K) fitted parameters for each replicate
    cost_boot    : (B,)   final RSS for each replicate
    success_boot : (B,)   bool convergence flag per replicate
    Sig_refit    : (B, T) model-predicted signal for each successful replicate
    """
    params_boot:  np.ndarray
    cost_boot:    np.ndarray
    success_boot: np.ndarray
    Sig_refit:    np.ndarray

@dataclass(frozen=True)
class RoiFitResult:
    """
    Result of whole-ROI mean-curve fitting.

    Fields
    ------
    params      : fitted kinetic parameters (model-specific)
    cost        : final RSS
    success     : convergence flag
    Ct_mean     : mean tissue concentration curve used for fitting  (T,)
    Ct_pred     : model-predicted concentration curve  (T,)
    St_mean     : mean signal curve  (T,)  — for plotting
    St_pred     : model-predicted signal curve  (T,)  — for plotting
    n_voxels    : number of voxels pooled
    average_space : which space was used for aggregation ('signal' or 'concentration')
    """
    params:        np.ndarray
    cost:          float
    success:       bool
    Ct_mean:       np.ndarray
    Ct_pred:       np.ndarray
    St_mean:       np.ndarray
    St_pred:       np.ndarray
    n_voxels:      int
    average_space: str
    
