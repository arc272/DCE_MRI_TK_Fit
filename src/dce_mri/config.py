# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

"""
config.py
=========

Configuration dataclasses and result containers for the DCE-MRI pipeline.

## Design principles

This module is the central data-definition layer of the pipeline. It contains
configuration objects that describe how an analysis should be performed and
result objects that describe what an analysis produced.

The module follows four principles:

1. ## Configuration is explicit and immutable

   All configuration and result dataclasses use `frozen=True`. Once
   constructed, a configuration object should not be modified during a
   workflow. Changes should be made by constructing a new configuration.

2. ## Configuration is separated by scope

   Parameters are grouped according to the level at which they apply:

   ```
   AcquisitionConfig
       Cohort-/sequence-level MR acquisition parameters.

   SubjectConfig
       Subject-specific parameters and data locations.

   AifConfig
       Parameters specific to AIF extraction.

   RelaxationConfig
       Parameters specific to water-exchange / relaxation fitting.

   FittingConfig
       General single-voxel kinetic-model optimization settings.

   FitModeConfig
       Selection and parameters for alternative fitting strategies,
       such as dispersed-AIF or delayed-AIF fitting.

   BootstrapConfig
       Bootstrap uncertainty-analysis settings.

   VoxelwiseConfig
       Parallel voxelwise-volume processing settings.

   AdmmConfig
       Joint spatial TV-regularised fitting settings.

   RoiFittingConfig
       Whole-ROI curve aggregation and fitting settings.
   ```

   A configuration class should contain parameters that belong to its
   particular scope rather than duplicating the same physical quantity across
   multiple configuration objects.

3. ## Configuration contains parameters, not workflow logic

   These dataclasses describe inputs to the pipeline. They should not perform
   fitting, image processing, optimization, or other business logic.

   Simple derived properties are allowed when they provide deterministic
   information directly implied by stored fields. For example, SubjectConfig
   derives default DCE, AIF, mask, and output paths from `subject_id` and
   `data_root`.

   Actual computation belongs in the appropriate functional/workflow module.

4. ## Result objects are immutable containers for computed outputs

   Result dataclasses contain the arrays, scalar metrics, convergence flags,
   and metadata produced by fitting workflows.

   Result objects should describe the outcome of a computation rather than
   implement the computation itself.

## Model-agnostic design

The configuration layer does not hard-code the internal mathematics of any
kinetic model. Model-specific parameter names, bounds, constraints, and
forward functions are defined by KineticModelSpec in `kinetic_models.py`.

This allows the same fitting infrastructure to operate on different models
without introducing model-specific logic into the configuration classes.

In particular:

```
FittingConfig
    describes HOW optimization is performed.

FitModeConfig
    describes WHICH fitting strategy is selected.

KineticModelSpec
    describes WHICH kinetic model is being fitted and its parameter space.

VoxelFitResult / RoiFitResult / etc.
    describe WHAT the fitting procedure produced.
```

## Units and conventions

Physical units should be explicit and consistent throughout the pipeline:

```
time / delay / residence time : seconds
TR / T1                       : milliseconds
flip angle                    : degrees
concentration                 : mM
relaxivity r1                 : mM^-1 s^-1
kinetic rate constants        : s^-1
volume fractions              : dimensionless
```

Configuration objects should not silently convert units. Unit conversion
should occur at the boundary where data enter a model or fitting workflow.

## Usage

Import configuration and result types from `dce_mri.config`:

```
from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    AifConfig,
    RelaxationConfig,
    FittingConfig,
    FitModeConfig,
    BootstrapConfig,
    VoxelwiseConfig,
    AdmmConfig,
    RoiFittingConfig,
    DelayedFitResult,
    VoxelFitResult,
    DispersedGKMResult,
    VoxelwiseResult,
    BootstrapResult,
    RoiFitResult,
    RoiDelayedFitResult,
)
```

## Typical workflow

Construct configuration objects at the beginning of a workflow and pass them
explicitly to the functions that need them:

```
acq = AcquisitionConfig(...)
subject = SubjectConfig(...)
fitting = FittingConfig(...)
fit_mode = FitModeConfig(...)
```

The configuration layer should remain independent of the implementation of
the fitting algorithms. This keeps workflows reproducible, testable, and
easy to extend.

## Result types

The result classes correspond to different levels of analysis:

```
VoxelFitResult
    One kinetic-model fit for one voxel.

DelayedFitResult
    One voxel fit in which AIF delay is additionally estimated or
    evaluated over a delay grid.

DispersedGKMResult
    One voxel dispersed-GKM analysis containing results for multiple
    vascular branch levels and their Akaike weights.

VoxelwiseResult
    Collection of fitted parameter maps and QC information from a
    volume-wide voxelwise analysis.

BootstrapResult
    Parameter and prediction distributions from repeated bootstrap fits.

RoiFitResult
    Fit to an aggregated whole-ROI concentration/signal curve.

RoiDelayedFitResult
    Whole-ROI fit with the additional AIF-delay outputs.
```

Result objects intentionally contain NumPy arrays rather than file paths or
serialization logic. 

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dce_mri.ace_protocol import ACEProtocol, FrameArrays


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
                Used directly by legacy single-FA signal functions
                (signal_models.concentration_to_signal etc.) regardless
                of `protocol`. If `protocol` is set, TR/FA no longer
                describe the real per-frame schedule — see frame_arrays().
    FA      : nominal flip angle  (degrees)  — same caveat as TR.
    r1      : CA relaxivity  (mM⁻¹ s⁻¹)
    nbase   : slice selecting pre-contrast baseline frames
              e.g. slice(0, 7) means frames 0-6 are baseline
    upsample: internal convolution upsampling factor for all forward models
    protocol: optional multi-flip-angle / ACE acquisition schedule.
              When None, frame_arrays() falls back to a one-segment
              schedule built from TR/FA — i.e. single-FA is the `None`
              case of frame_arrays(), not a separate code path.
              NOTE: setting `protocol` alone does not yet change what
              legacy single-FA signal functions compute; only code that
              calls frame_arrays() (Step 1.3 onward) respects it.
    """
    TR:       float = 3.64
    FA:       float = 10.0
    r1:       float = 3.8
    nbase:    slice = field(default_factory=lambda: slice(0, 7))
    upsample: int   = 50
    protocol: Optional[ACEProtocol] = None

    def __post_init__(self):
        if not isinstance(self.nbase, slice):
            raise TypeError(
                f"nbase must be a slice object (e.g. slice(0, 7)), "
                f"got {type(self.nbase).__name__} = {self.nbase!r}. "
                f"Did you mean slice(0, {self.nbase})?"
            )
        if self.protocol is not None and not isinstance(self.protocol, ACEProtocol):
            raise TypeError(
                f"protocol must be an ACEProtocol or None, "
                f"got {type(self.protocol).__name__}"
            )

    def frame_arrays(self, time_var: np.ndarray) -> FrameArrays:
        """
        Per-frame (alpha_deg, TR_ms, seg_id, keep) for `time_var`.

        Single source of truth for acquisition timing: with `protocol`
        set, defers to it entirely; with `protocol=None`, synthesises the
        equivalent one-segment schedule from TR/FA, so callers never need
        to branch on whether a protocol is present.
        """
        if self.protocol is not None:
            return self.protocol.frame_arrays(time_var)
        t = np.asarray(time_var, dtype=np.float64)
        t_max = float(t[-1]) + 1.0 if t.size else 1.0
        return ACEProtocol.single_fa(alpha=self.FA, TR_ms=self.TR,
                                     t_max=t_max).frame_arrays(t)


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

@dataclass(frozen=True)
class RelaxationConfig:
    """
    Bounds for the water-exchange fit's free exchange-time parameters.

    Baseline relaxation quantities (blood T1, tissue T1, relaxivity,
    hematocrit) are NOT duplicated here -- reuse AifConfig.T10
    (blood T1, ms), subject.T10 (tissue T1, ms), acq.r1 (relaxivity),
    and AifConfig.hematocrit directly. This config exists only for the
    genuinely new quantities water exchange introduces: the tau_b/tau_i
    search bounds, matching the tau_lb/tau_ub pattern already used for
    AIF delay fitting -- pass explicitly, don't hardcode module
    constants inside the fitting functions.

    ...
    Fields
    ------
    tau_b_fixed, fit_tau_b: value and option of fixing tau_b (not indenfiable in tests) 
    
    tau_i_lb, tau_i_ub : bounds on intracellular water residence time (s).
                  Diffusion/membrane-limited -- can be genuinely slow.
    tau_b_lb, tau_b_ub : bounds on vascular water residence time (s).
                  Flow-limited (blood is actively replaced) -- much
                  faster than tau_i. Matches the MATLAB reference
                  (bmri_dce_model.m: LB=[...0 0], UB=[...2 0.01]) --
                  note the two orders of magnitude gap between the two
                  upper bounds is not a typo, it reflects genuinely
                  different physical rate-limiting mechanisms.
    """
   
    fit_tau_b:   bool  = False     # False = WX-4c default (tau_i only)
                                   # True  = fit both taus
    tau_b_fixed: float = 0.005     # s; used only when fit_tau_b=False
    
    tau_i_lb: float = 0.01
    tau_i_ub: float = 1.5
    tau_b_lb: float = 1e-4
    tau_b_ub: float = 0.01
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
    n_lhs:     int   = 1000
    n_top:     int   = 20
    xtol:      float = 1e-6
    ftol:      float = 1e-6
    gtol:      float = 1e-6
    max_nfev:  int   = 500
    fit_space: str   = "signal"   # "concentration" | "signal"
    jacobian:  str   = "3-point"  # "3-point" | "jax"
    x_scale:   object = "jac"     # scipy least_squares x_scale


@dataclass(frozen=True)
class NuisanceConfig:
    """
    Policy for the three NUISANCE parameters: M0, T10 and B1.

    A nuisance parameter is one that must be estimated for the answer to
    be right, but whose value is not the measurement. None of these are
    what a DCE study is for, yet all three multiply into every fitted
    quantity -- and the CRLB analysis showed how hard: a 14% B1 error
    propagates to a +233% tau_i error, because tau_i and B1 correlate at
    +0.94 on ACE and +0.97 on DFA.

    Separate from RelaxationConfig on purpose. That config holds the
    fix-or-fit policy for the EXCHANGE times (tau_b, tau_i), which are
    physiological quantities fixed for identifiability. These three are
    instrument nuisances. The structural similarity is real but the
    reasons differ, so they are kept apart.

    Fields
    ------
    fit_T10, fit_B1 : free them rather than assume them. Requires a
                  multi-flip-angle protocol -- see
                  validate_nuisance_compatibility. On single-FA data both
                  are provably non-identifiable (Fisher condition number
                  ~7e15), so the guardrail refuses rather than returning
                  a confident wrong answer.
    T10_bounds_ms, B1_bounds : search bounds, used only when fitting.
    T10_prior, B1_prior : (mu, relative_sd) or None for flat. mu is in ms
                  for T10 and dimensionless for B1; the sd is RELATIVE
                  (0.08 = 8%) to match the log coordinates the fitter and
                  the CRLB both work in.

                  A WARNING that the CRLB cannot express: a prior costs
                  accuracy when its centre is wrong, and buys almost no
                  precision once the data are informative. At ROI SNR on
                  ACE, a B1 prior improved the tau_i bound by 0.5
                  percentage points while a centre wrong by 0.18 (Sung's
                  right-breast 0.82 vs a naive 1.0) introduced +31% bias.
                  Leave these None unless the centre is genuinely known
                  for THIS acquisition -- the default is flat for that
                  reason.
    m0_mode     : how the equilibrium magnetization is handled.
                  'profile'  -- variable projection; M0 solved in closed
                                form at each parameter vector, never
                                searched. The default, and what the
                                multi-flip-angle fitter uses.
                  'marginal' -- integrated out analytically (Bayes only).
                  'baseline' -- legacy S0 = mean(baseline frames). ONLY
                                valid with no protocol: with several flip
                                angles the baseline ratio cancels the
                                flip-angle encoding exactly (verified: a
                                C=0 curve across an 8deg->25deg switch
                                returns [1,1,1,1,1,1]).
    """
    fit_T10:       bool  = False
    fit_B1:        bool  = False
    T10_bounds_ms: tuple = (300.0, 4000.0)
    B1_bounds:     tuple = (0.5, 1.5)
    T10_prior:     Optional[tuple] = None
    B1_prior:      Optional[tuple] = None
    m0_mode:       str   = "profile"

    def __post_init__(self) -> None:
        if self.m0_mode not in ("profile", "marginal", "baseline"):
            raise ValueError(
                f"m0_mode must be 'profile', 'marginal' or 'baseline', "
                f"got {self.m0_mode!r}")
        for nm, b in (("T10_bounds_ms", self.T10_bounds_ms),
                      ("B1_bounds", self.B1_bounds)):
            if len(b) != 2 or not (0 < b[0] < b[1]):
                raise ValueError(f"{nm} must be (lb, ub) with 0 < lb < ub, got {b}")
        for nm, pr in (("T10_prior", self.T10_prior), ("B1_prior", self.B1_prior)):
            if pr is None:
                continue
            if len(pr) != 2 or pr[0] <= 0 or pr[1] <= 0:
                raise ValueError(
                    f"{nm} must be (mu, relative_sd) with both positive, got {pr}")


def validate_nuisance_compatibility(nuisance: "NuisanceConfig", acq) -> None:
    """
    Refuse provably degenerate configurations, in the style of
    fitting._validate_wx_compatibility.

    Raises rather than warns: on single-flip-angle data, freeing T10 or B1
    is not merely imprecise, it is rank-deficient. The fitter would still
    return a number, and that number would be whatever the bounds and the
    starting point implied.
    """
    protocol = getattr(acq, "protocol", None)

    if protocol is None or protocol.is_single_fa:
        if nuisance.fit_T10 or nuisance.fit_B1:
            raise ValueError(
                "fit_T10/fit_B1 require a multi-flip-angle protocol: with a "
                "single flip angle the signal depends only on M0*B1 and "
                "R1/B1^2, so T10 and B1 are not identifiable (Fisher "
                "condition number ~7e15). Set acq.protocol to a "
                "multi-segment ACEProtocol, or leave both False."
            )
        if nuisance.m0_mode == "baseline":
            return

    if protocol is not None and not protocol.is_single_fa \
            and nuisance.m0_mode == "baseline":
        raise ValueError(
            "m0_mode='baseline' is invalid with a multi-flip-angle protocol: "
            "normalising each frame by a baseline at its own flip angle "
            "cancels the flip-angle encoding the protocol exists to provide. "
            "Use m0_mode='profile' (NLLS) or 'marginal' (Bayes)."
        )


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
# 
# =============================================================================

@dataclass(frozen=True)
class FitModeConfig:
    """
    Selects which fitting function to call in voxelwise volume fitting.

    Fields
    ------
    mode          : 'standard' | 'dispersed' | 'delay_grid' | 'delay_continuous'
    n_levels      : dispersed GKM branch levels to try (mode='dispersed')
    tau_values    : grid of delay values in seconds (mode='delay_grid')
    tau_lb        : lower bound for tau in seconds (mode='delay_continuous')
    tau_ub        : upper bound for tau in seconds (mode='delay_continuous')
    tau_p0        : initial guess for tau in seconds (mode='delay_continuous')
    interp_method : AIF shift interpolation method ('linear' | 'pchip')
                    only used when mode involves a delay (grid/continuous)
    """
    mode:          str                  = "standard"
    n_levels:      int                  = 6
    tau_values:    Optional[np.ndarray] = None
    tau_lb:        float                = -10.0
    tau_ub:        float                =  30.0
    tau_p0:        float                =   0.0
    interp_method: str                  = "linear"

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
    wx_params     : fitted water-exchange parameters, or None if no
                    wx_model was used. For grid mode, this is copied
                    up from per_tau[best_tau].wx_params — see fitting.py.
    wx_model_name : name of the wx_model used, or None 
    """
    params:  np.ndarray
    cost:    float
    success: bool
    S_meas:  np.ndarray
    S_pred:  np.ndarray
    wx_params:      Optional[np.ndarray] = None
    wx_model_name:  Optional[str]        = None
    

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
    wx_params     : fitted water-exchange parameters, or None if no
                    wx_model was used. For grid mode, this is copied
                    up from per_tau[best_tau].wx_params — see fitting.py.
    wx_model_name : name of the wx_model used, or None
    """
    best_tau:    float
    best_params: np.ndarray
    best_cost:   float
    per_tau:     dict
    success:     bool
    fit_mode:    str
    wx_params:      Optional[np.ndarray] = None
    wx_model_name:  Optional[str]        = None

@dataclass(frozen=True)
class DispersedGKMResult:
    """
    ...
    best_level  : integer branch level with the HIGHEST AKAIKE WEIGHT
                  (argmax(weights)), 1-indexed. Kept for reference/QC;
                  it is no longer what best_params/best_cost are taken
                  from directly.
    best_params : AIC-weighted AVERAGE of [Ktrans, ve, vp, t0_vasc]
                  across all levels that converged -- NOT the single
                  best level's raw fitted params.
    best_cost   : AIC-weighted average of per-level RSS (a summary
                  statistic, not a literal re-evaluated cost).
    per_level   : unchanged.
    weights     : (n_levels,) Akaike weights, sums to 1.
    aic         : (n_levels,) AICc per level.
    success     : True if at least one level converged.
    """
    best_level:  int
    best_params: np.ndarray     # AIC-weighted average [Ktrans, ve, vp, t0_vasc]
    best_cost:   float          # AIC-weighted average RSS
    per_level:   dict           # {1: VoxelFitResult, 2: ..., ...}
    weights:     np.ndarray     # Akaike weights per level, sums to 1
    aic:         np.ndarray     # AICc per level
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
    Ct_mean       : mean tissue concentration curve used for fitting  (T,)
    Ct_pred       : model-predicted concentration curve  (T,) -- under
                    wx_model, this is Cv+Ce, not literally what generated St_pred
    St_mean     : mean signal curve  (T,)  — for plotting
    St_pred     : model-predicted signal curve  (T,)  — for plotting
    n_voxels    : number of voxels pooled
    average_space : which space was used for aggregation ('signal' or 'concentration')
    wx_params     : fitted water-exchange parameters, or None if no wx_model was used
    wx_model_name : name of the wx_model used, or None
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
    wx_params:     Optional[np.ndarray] = None
    wx_model_name: Optional[str]        = None
    
@dataclass(frozen=True)
class RoiDelayedFitResult:
    """
    Result of fitting a kinetic model to a masked ROI's mean curve,
    with AIF time delay. Combines DelayedFitResult's delay-search
    fields with RoiFitResult's ROI-aggregation fields.
    
    Fields
    ------
    best_tau      : best delay in seconds (positive = AIF arrives later)
    best_params   : kinetic parameters at best_tau (model-specific, no tau)
    best_cost     : RSS at best solution
    per_tau       : {tau: VoxelFitResult} — grid mode only, empty for continuous
    success       : convergence flag
    fit_mode      : 'grid' or 'continuous'
    Ct_mean       : mean tissue concentration curve used for fitting  (T,)
    Ct_pred       : model-predicted concentration curve  (T,) -- under
                    wx_model, this is Cv+Ce, not literally what generated St_pred
    St_mean       : mean signal curve  (T,)  — for plotting
    St_pred       : model-predicted signal curve  (T,)  — for plotting
    n_voxels      : number of voxels pooled
    average_space : 'signal' or 'concentration'
    wx_params     : fitted water-exchange parameters, or None if no wx_model was used
    wx_model_name : name of the wx_model used, or None
    """
    # everything DelayedFitResult already has
    best_tau:    float
    best_params: np.ndarray
    best_cost:   float
    per_tau:     dict
    success:     bool
    fit_mode:    str
    # everything RoiFitResult adds
    Ct_mean:       np.ndarray
    Ct_pred:       np.ndarray
    St_mean:       np.ndarray
    St_pred:       np.ndarray
    n_voxels:      int
    average_space: str
    wx_params:     Optional[np.ndarray] = None
    wx_model_name: Optional[str]        = None