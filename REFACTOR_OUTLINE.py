# DCE-MRI Pipeline — Refactor Outline
# =====================================
# This document is both the original build plan AND a living status
# tracker. Each phase header carries a STATUS line:
#     DONE      — implemented and verified
#     PARTIAL   — some functions implemented, some still stubs
#     NOT STARTED — file/functions exist only as docstring + TODO
#
# Last status review: 2026-07-27
# -----------------------------------------------------------------------


# ══════════════════════════════════════════════════════════════════════
# CURRENT STATUS SUMMARY (read this first)
# ══════════════════════════════════════════════════════════════════════
#
# DONE
#   Phase 0  — package skeleton
#   Phase 1  — config layer (expanded beyond original plan, see Phase 1)
#   Phase 2  — signal models (expanded, see Phase 2)
#   Phase 3  — kinetic models (TCXM, GKM, DispersedGKM all working)
#   Phase 4A — vascular transfer functions (single_vessel_h, vascular_tree_h)
#   Phase 4C — AIF extraction from ROI                          ← NEW, not in original plan
#   Phase 5  — single-voxel fitting (LHS + TRF, signal & concentration space)
#   Phase 5B — AIF time-delay fitting (grid + continuous)        ← NEW, not in original plan
#   Phase 6  — chunked parallel voxelwise fitting (one stub fn remains)
#   Phase 8  — whole-ROI fitting (incl. delay variants)          ← expanded beyond plan
#
#   Verified by: tests/ROI_fit_test.py (5/5 assertions pass — TCXM & GKM,
#   no-delay / grid-delay / continuous-delay, signal vs concentration
#   aggregation). All modules import cleanly.
#
# PARTIAL
#   Phase 4B — per-voxel vascular AIF model fitting (transfer functions done,
#              fitting routines are stubs)
#   Phase 6  — create_binary_masks() still a stub
#
# NOT STARTED
#   Phase 7  — ADMM joint TV-regularised fitting (entire file is stubs)
#   Phase 9  — bootstrap uncertainty (entire file is stubs)
#   Phase 10 — ACE protocol (entire file is stubs)
#   Phase 11 — statistics.py and visualization.py (entirely stubs);
#              io.py done except load_results_npz()
#   Phase 12 — none of the four planned entry-point scripts exist yet
#              (scripts/ currently holds an unrelated per-patient ROI
#              tightening tool — useful, just not part of this plan)
#   Phase 13 — none of the planned 01-06 concept notebooks exist;
#              notebooks/ currently has 06_roi_fitting.ipynb and
#              07_voxelwise_fitting.ipynb (numbering predates this plan
#              and likely needs to be reconciled/renamed once phases
#              7-11 exist to write notebooks against)
#
# BUG LOG
#   2026-07-27  volume.py / run_voxelwise_fitting(): the per-voxel
#     param_maps fill loop used `model.param_names` instead of
#     `current_param_names`. In "dispersed" mode this silently dropped
#     t0_vasc (index 3) from every voxel's output, leaving that map at
#     zero everywhere with no error. Fixed: loop now uses
#     current_param_names, which is already computed correctly earlier
#     in the function. Verified with a synthetic dispersed-GKM voxel.
#
# DEPENDENCY NOTE
#   pyproject.toml was missing scikit-image, which fitting_admm.py's
#   docstring names as the intended TV-prox implementation
#   (restoration.denoise_tv_chambolle). Added scikit-image>=0.24.
#
# HOUSEKEEPING NOTE
#   src/test_utils.py imports torch/ants/VAE/UNet_FS — this looks like
#   it belongs to a different (segmentation/diffusion) project, not
#   dce_mri. Consider moving it out of src/ or deleting if stale.
#   signal_models.py also has two apparently-dead helpers, _spgr_E1 and
#   _spgr_signal, not called anywhere else in the package — the real
#   logic is inlined in signal_to_concentration/concentration_to_signal.
#   Safe to remove once you've confirmed no notebook imports them.


# ══════════════════════════════════════════════════════════════════════
# PHASE 0 — Package skeleton
# STATUS: DONE
# ══════════════════════════════════════════════════════════════════════
#
# Goal: make the repo installable with `pip install -e .`
#       so all subsequent imports work from anywhere.
#
# Files to create (all empty or minimal):
#
#   dce_mri/
#   ├── pyproject.toml              ← package name, version, dependencies
#   ├── README.md                   ← install instructions + structure map
#   └── src/
#       └── dce_mri/
#           └── __init__.py         ← empty file, just makes it a package
#
# Dependencies to list in pyproject.toml:
#   numpy, scipy, nibabel, matplotlib, pandas, scikit-image (added — see
#   dependency note above), plus whatever the rest of your stack needs
#   optional dev: pytest, ipykernel, jupyter
#
# Verification:
#   cd dce_mri && pip install -e ".[dev]"
#   python -c "import dce_mri; print('ok')"


# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — Config layer
# STATUS: DONE — expanded beyond original plan
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/config.py
# Rule: frozen dataclasses only. No imports from other dce_mri modules.
#       No methods with logic. Only data fields and simple @property paths.
#
# ── Configs written (as originally planned) ──
#
# AcquisitionConfig  — TR, FA, r1, nbase, upsample
# SubjectConfig      — subject_id, T10, B1, data_root + path properties
#                      (dce_path, aif_path, mask_path, out_dir), plus
#                      *_override fields for ad-hoc paths not in the plan
# FittingConfig      — n_lhs, n_top, xtol, ftol, gtol, max_nfev, fit_space
# BootstrapConfig    — B, seed
# VoxelwiseConfig    — n_workers, chunk_size, min_cost, n_qc_plots,
#                      save_dir, output_prefix
# AdmmConfig         — n_admm_iter, rho, lambda_tv, n_inner_iter,
#                      tol_primal, tol_dual, warm_start, n_workers,
#                      chunk_size, save_dir, output_prefix
# RoiFittingConfig   — aggregation, top_frac, save_dir, plus
#                      average_space ('signal'|'concentration') — new
#                      field needed once ROI fitting supported both
#                      aggregation spaces (see Phase 8)
#
# ── Configs added beyond the original plan ──
#
# AifConfig                                                    ← NEW
#   Fields: T10 (blood T1), hematocrit, top_frac, nbase
#   Why: AIF extraction from a vascular ROI needs its own T10/nbase
#        (blood T1 and bolus timing differ from tissue), which doesn't
#        fit AcquisitionConfig or SubjectConfig. See Phase 4C.
#
# FitModeConfig                                                ← NEW
#   Fields: mode ('standard'|'dispersed'|'delay_grid'|'delay_continuous'),
#           n_levels, tau_values, tau_lb, tau_ub, tau_p0, interp_method
#   Why: volume.py's run_voxelwise_fitting needs to dispatch to different
#        fitters (standard / dispersed / delay search) without every
#        caller having to know the details. See Phase 5B and fitting.py's
#        call_fitter()/normalize_fit_result().
#
# ── Result types written (as planned) ──
#
# VoxelFitResult, DispersedGKMResult, VoxelwiseResult, BootstrapResult
#
# ── Result types added beyond the original plan ──
#
# DelayedFitResult                                             ← NEW
#   Fields: best_tau, best_params, best_cost, per_tau, success, fit_mode
#   Why: output of fit_with_delay_grid / fit_with_delay_continuous
#        (Phase 5B) and their ROI equivalents (Phase 8).
#
# RoiFitResult — expanded beyond the original plan:
#   Fields: params, cost, success, Ct_mean, Ct_pred, n_voxels, PLUS
#           St_mean, St_pred, average_space
#   Why: needed once ROI fitting supported signal-space aggregation
#        (St_mean/St_pred are needed for plotting either way).
#
# Verification:
#   from dce_mri.config import AcquisitionConfig, FittingConfig
#   cfg = AcquisitionConfig(); print(cfg)   # prints frozen dataclass — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — Signal models
# STATUS: DONE — expanded beyond original plan
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/signal_models.py
#
# Functions written (as planned):
#   signal_to_concentration(sig, acq, T10, B1)   →  C_t
#   concentration_to_signal(C_t, acq, T10, B1, S0)  →  sig
#   compute_S0(sig, acq)  →  S0
#
# Functions added beyond the original plan:
#
# extract_aif(sig_4d, aif_mask, acq, aif_cfg, B1)              ← NEW
#   Pulls voxel signals from a vascular ROI mask, optionally keeps only
#   top-enhancing voxels (aif_cfg.top_frac), averages, converts to
#   whole-blood concentration using AIF-specific T10/nbase (not the
#   tissue subject.T10/acq.nbase), then applies hematocrit correction
#   to get plasma concentration. Returns (aif_plasma, aif_signal).
#
# shift_aif(aif, tau, time_var, acq, method='linear'|'pchip')  ← NEW
#   Shifts an AIF by tau seconds via interpolation (linear or
#   monotonicity-preserving PCHIP). Baseline-fills the left edge using
#   acq.nbase mean rather than aif[0], so enhancement never leaks into
#   the pre-contrast baseline regardless of shift size. Powers all of
#   Phase 5B's delay-fitting functions.
#
# Housekeeping: _spgr_E1 / _spgr_signal appear to be dead code (see
# housekeeping note above) — candidates for deletion.
#
# Verification:
#   from dce_mri.signal_models import signal_to_concentration
#   import numpy as np
#   acq = AcquisitionConfig()
#   sig = np.ones(90); sig[7:] *= 1.2
#   C = signal_to_concentration(sig, acq, T10=1530.0, B1=1.0)
#   assert C.shape == (90,)                                     — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — Kinetic models
# STATUS: DONE
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/kinetic_models.py
#
# Functions written (all per original plan):
#
# tcxm_ct_analytic(theta_kin, time_var, aif, upsample)  →  C_t
#   theta_kin = [ve, vp, fp, ps]
#
# gkm_ct(theta_kin, time_var, aif, upsample)  →  C_t
#   theta_kin = [Ktrans, ve, vp]
#
# dispersed_gkm_ct(theta_kin, time_var, aif, n_level, upsample)  →  C_t
#   theta_kin = [Ktrans, ve, vp, t0]  (4 params, n_level passed separately
#   as planned — not folded into theta_kin)
#
# KineticModelSpec dataclass — name, fn, param_names, lb, ub, p0,
#   constraints, PLUS log_dims (not in original plan — records which
#   parameter indices should be LHS-sampled log-uniformly, since fp/ps/
#   Ktrans span orders of magnitude; used by fitting.py's _lhs_starts)
#
# Module-level instances: TCXM, GKM, DISPERSED_GKM (fn=None, as planned)
#
# Design fix beyond the original plan:
#
# make_dispersed_gkm_spec(n_level) + DISPERSED_GKM_LEVELS dict  ← NEW
#   The plan's approach (closures capturing n_level per-level inside
#   fitting.py) would not survive multiprocessing.Pool — closures aren't
#   picklable. Instead, six module-level functions
#   (_dispersed_gkm_fn_level_1..6) are defined once, and
#   make_dispersed_gkm_spec() builds a fully callable, picklable
#   KineticModelSpec per level. fit_dispersed_gkm() in fitting.py (Phase 5)
#   loops over DISPERSED_GKM_LEVELS instead of building closures inline.
#   This matters because Phase 6 (volume.py) runs everything through
#   multiprocessing.Pool.
#
# Verification:
#   from dce_mri.kinetic_models import TCXM, tcxm_ct_analytic
#   t = np.linspace(0, 300, 90)
#   aif = np.clip(((t/15)**4) * np.exp(-4*(t/15-1)), 0, None)
#   C = tcxm_ct_analytic(TCXM.p0, t, aif)
#   assert C.shape == (90,) and np.all(np.isfinite(C))          — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 4A — Vascular transfer functions
# STATUS: DONE
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/vascular_aif.py
#
# single_vessel_h(t, t0)  →  h                                  [Eq. 2]
# vascular_tree_h(t, t0, n_level)  →  h_n                       [Eq. 12]
#   Implemented as n_level-fold self-convolution of single_vessel_h.
#   Already consumed by kinetic_models.dispersed_gkm_ct — this is the
#   load-bearing piece for the dispersed model and it works.
#
# Verification:
#   from dce_mri.vascular_aif import single_vessel_h
#   h = single_vessel_h(np.linspace(0,5,100), t0=0.5)
#   assert h[0] == 0.0 and h[-1] > 0.0                          — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 4B — Per-voxel vascular AIF model fitting
# STATUS: NOT STARTED (stubs only)
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/vascular_aif.py (same file as 4A)
#
# This is the per-voxel local-AIF estimation half of the Nejad-Davarani
# model — given a tissue concentration curve, find the branching level
# and delay that best explain the local AIF via AIC model averaging.
# All of the following currently `raise NotImplementedError`:
#
# local_aif_at_level(global_aif, t, t0, g, n_level)  →  ca_local
#   CA_local(t) = g · (AIF * h_{1→n})(t)                   [Eq. 13-14]
#
# fit_vascular_aif(ca_tissue, global_aif, t, n_levels, n_restarts, seed)
#   →  VascularAIFResult
#   Fit [g, t0] per branching level via multi-start optimization, then
#   AIC-model-average across levels.                      [Sec. 2.2]
#
# get_local_aif(ca_tissue, global_aif, t, ...)  →  (local_aif, result)
#   Convenience wrapper; falls back to global AIF if all fits fail.
#
# _fit_single_level(ca, aif, t, n_level, n_restarts, seed)
#   →  (g_opt, t0_opt, rss)   — multi-start Nelder-Mead per level
#
# _compute_aic(rss, n_obs, n_param)  →  AICc                [Eq. 15]
# _akaike_weights(aic_values)  →  weights                   [Eq. 16-17]
#
# Note: VascularAIFResult dataclass already exists in this file (not
# moved to config.py — that's fine, it only matters if you want every
# result type centralized).
#
# Verification (once implemented):
#   result = fit_vascular_aif(ca_tissue, global_aif, t)
#   assert result.weights.sum() == pytest.approx(1.0)
#   assert 1 <= result.best_level <= 6


# ══════════════════════════════════════════════════════════════════════
# PHASE 4C — AIF extraction from ROI
# STATUS: DONE
# (not in the original plan — added because you needed it in practice)
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/signal_models.py (see Phase 2 — extract_aif)
# Config: src/dce_mri/config.py — AifConfig (see Phase 1)
#
# This phase didn't exist in the original outline. It was needed to go
# from "a vascular ROI mask + 4D signal" to "a plasma-concentration AIF
# curve," which every fitting phase downstream assumes you already have.
#
# Pipeline: pull ROI voxels → optional top-enhancing trim (top_frac) →
# average → signal-to-concentration using AIF-specific T10/nbase →
# hematocrit correction → plasma concentration.
#
# Verification:
#   aif_plasma, aif_signal = extract_aif(sig_4d, aif_mask, acq, aif_cfg)
#   assert aif_plasma.shape == (sig_4d.shape[-1],)


# ══════════════════════════════════════════════════════════════════════
# PHASE 5 — Standard single-voxel fitting
# STATUS: DONE
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/fitting.py
#
# Internal helpers written (all per plan):
#   _lhs_starts(n, lb, ub, constraints, seed) — extended with log_dims
#     (samples log-uniformly for parameters that span orders of
#     magnitude, per KineticModelSpec.log_dims — Phase 3 addition)
#   _trf_refine(p0, residual_fn, lb, ub, cfg)
#   _top_finite_indices(costs, n_top)
#   _residual_concentration(p, Ct_meas, time_var, aif, model_fn, constraints)
#   _residual_signal(p, sig_meas, time_var, aif, model_fn, acq, T10, B1,
#                     S0, constraints)
#
# Public functions written (all per plan):
#
# fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg, seed)
#   →  VoxelFitResult
#   cfg.fit_space selects concentration vs signal domain, as planned.
#
# fit_dispersed_gkm(sig, aif, time_var, acq, subject, cfg, n_levels, seed)
#   →  DispersedGKMResult
#   Loops DISPERSED_GKM_LEVELS (Phase 3's picklable per-level specs)
#   instead of building closures inline as the original plan sketched —
#   same algorithm, safer implementation.
#
# Verification (tests/ROI_fit_test.py exercises this indirectly via
# fitting_roi.py; ran clean — TCXM and GKM both recover known params):
#   result = fit_single_voxel(sig, aif, t, TCXM, acq, subject, cfg)
#   assert isinstance(result, VoxelFitResult)
#   assert result.params.shape == (4,)                          — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 5B — AIF time-delay fitting
# STATUS: DONE
# (not in the original plan — added to handle AIF/tissue bolus timing
#  mismatch, which turned out to matter in practice)
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/fitting.py
#
# fit_with_delay_grid(sig, aif, time_var, model, acq, subject, cfg,
#                      tau_values, seed, interp_method)
#   →  DelayedFitResult
#   Grid search over tau: shift_aif() at each candidate tau, run
#   fit_single_voxel(), keep the lowest-cost tau. per_tau dict holds
#   every candidate's VoxelFitResult for inspection.
#
# fit_with_delay_continuous(sig, aif, time_var, model, acq, subject, cfg,
#                            tau_lb, tau_ub, tau_p0, seed, interp_method)
#   →  DelayedFitResult
#   Appends tau as an extra free parameter, builds an augmented
#   KineticModelSpec whose fn shifts the AIF internally before calling
#   the base model, and runs the same LHS+TRF machinery as
#   fit_single_voxel on the augmented parameter vector.
#
# _fit_concentration_curve(...), _fit_concentration_curve_with_delay_grid
# (...), _fit_concentration_curve_with_delay_continuous(...)
#   Concentration-space equivalents of the three functions above, used
#   internally by fitting_roi.py (Phase 8) since ROI mean curves are
#   already in concentration space and shouldn't be re-converted.
#
# call_fitter(sig_v, aif, time_var, model, acq, subject, fitting_cfg,
#             fit_mode_cfg, seed)  →  dict
#   Dispatches to fit_single_voxel / fit_dispersed_gkm /
#   fit_with_delay_grid / fit_with_delay_continuous based on
#   fit_mode_cfg.mode, then flattens the result via
#   normalize_fit_result(). This is what volume.py's _process_chunk
#   actually calls — it's the single entry point that lets Phase 6 stay
#   ignorant of which fitter variant is running.
#
# normalize_fit_result(result, mode)  →  dict
#   Flattens VoxelFitResult / DispersedGKMResult / DelayedFitResult into
#   a common {params, cost, success, [best_level|best_tau]} shape.
#
# Verification: tests/ROI_fit_test.py Tests 3-5 — grid delay recovers
# tau to within 2s, continuous delay recovers tau to within 0.1s and
# outperforms grid cost when true tau falls between grid points — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 6 — Chunked parallel voxelwise volume fitting
# STATUS: PARTIAL (one stub remaining)
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/volume.py
#
# Functions written and verified:
#
# run_voxelwise_fitting(sig_4d, aif, time_var, model, acq, subject,
#                       fitting_cfg, voxelwise_cfg, mask, fit_mode_cfg)
#   →  VoxelwiseResult
#   Auto-generates a mask from baseline percentile if none given, builds
#   coords via np.argwhere, chunks them, dispatches to Pool via
#   _process_chunk, assembles param_maps/cost_map/success_map. Handles
#   "dispersed" and "delay_grid"/"delay_continuous" modes by adding
#   extra param_maps keys (best_level / best_tau).
#   BUG FIXED 2026-07-27: the per-voxel fill loop used model.param_names
#   instead of current_param_names, silently dropping t0_vasc in
#   dispersed mode. See BUG LOG above. Verified fixed with a synthetic
#   dispersed-GKM voxel.
#
# _build_chunks(coords, chunk_size) — np.array_split, as planned
# _process_chunk(args_dict) — top-level worker fn (picklable), calls
#   call_fitter() per voxel, catches per-voxel exceptions and returns
#   NaN params + success=False rather than killing the whole chunk
# save_nifti_maps(result, ref_img, cfg) — saves every param map + cost +
#   success as NIfTI, as planned
# load_nifti_canonical(path) — nib.as_closest_canonical, as planned
#
# Still a stub:
#
# create_binary_masks(seg, labels, return_list)
#   TODO: copy from tcxm_bootstrap_pipeline.create_binary_masks
#   (per-label binary mask extraction from a labelled segmentation —
#   needed by fit_multi_roi-style workflows and QC mask export)
#
# Good chunk_size values to try:
#   Small volume (< 5000 voxels): chunk_size = 50
#   Medium volume (5000-50000):   chunk_size = 200
#   Large volume (> 50000):       chunk_size = 500
#
# Verification:
#   result = run_voxelwise_fitting(sig_4d, aif, t, TCXM, acq, subject,
#                                   FittingConfig(), VoxelwiseConfig(), mask)
#   assert 've' in result.param_maps
#   assert result.param_maps['ve'].shape == sig_4d.shape[:3]     — OK
#   (dispersed mode additionally verified: t0_vasc map now populates)


# ══════════════════════════════════════════════════════════════════════
# PHASE 7 — ADMM joint TV-regularised fitting
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/fitting_admm.py
#
# Every function in this file is currently docstring + TODO +
# `raise NotImplementedError`:
#   run_admm_fitting(...)  →  VoxelwiseResult
#   _admm_voxel_update(...)
#   _single_voxel_admm_update(...)
#   _tv_prox(...)
#   _compute_residuals(...)
#
# This is entirely new. No existing code to copy from.
# Purpose: fit all voxels jointly with spatial TV regularisation.
# Use to compare against standard independent per-voxel fitting.
#
# ADMM problem being solved:
#   min_{x}  Σ_v f_v(x_v)  +  lambda_tv · TV(x)
#   where:
#     x_v   = kinetic parameters for voxel v  [ve, vp, fp, ps]
#     f_v   = per-voxel data fit cost (RSS from forward model)
#     TV(x) = isotropic total variation over 3D spatial parameter maps
#
# ADMM variables:
#   x_vol : (X, Y, Z, K) — voxel parameters (primal variable)
#   z_vol : (X, Y, Z, K) — auxiliary variable (TV proximal target)
#   u_vol : (X, Y, Z, K) — scaled dual variable
#
# Outer loop (n_admm_iter iterations):
#   Step 1 — x-update (parallel, chunked):
#     For each voxel v:
#       x_v = argmin_p  f_v(p) + (rho/2) * ||p - (z_v - u_v)||^2
#     The quadratic term is implemented by appending sqrt(rho/2)*(p-w_v)
#     to the TRF residual vector, where w_v = z_v - u_v
#     Uses same chunked parallel pattern as volume.py — call_fitter/
#     _process_chunk are good models for how to structure this.
#
#   Step 2 — z-update (TV proximal, closed-form):
#     z = prox_{(lambda_tv/rho) * TV}(x + u)
#     Now that scikit-image is a declared dependency (see pyproject.toml
#     update), skimage.restoration.denoise_tv_chambolle can be called
#     directly per parameter channel rather than hand-implementing
#     Chambolle's algorithm.
#
#   Step 3 — u-update:
#     u = u + x - z
#
#   Step 4 — convergence check:
#     primal = ||x - z||_F / sqrt(N) < tol_primal
#     dual   = ||rho * (z_new - z_old)||_F / sqrt(N) < tol_dual
#
# Warm start (admm_cfg.warm_start = True):
#   Run run_voxelwise_fitting first, use its param_maps as x_vol init
#   Strongly recommended — cold start from zeros is slow
#
# Key hyperparameters and typical starting values:
#   rho        = 1.0     (ADMM penalty; tune if convergence is slow)
#   lambda_tv  = 0.1     (TV weight; increase for smoother maps)
#   n_admm_iter = 50     (usually converges in 20-40 iterations)
#
# Verification (once implemented):
#   result_admm = run_admm_fitting(sig_4d, aif, t, TCXM, acq, subject,
#                                   FittingConfig(), AdmmConfig(), mask)
#   result_std  = run_voxelwise_fitting(...)
#   # Compare to standard result — ADMM maps should be spatially smoother


# ══════════════════════════════════════════════════════════════════════
# PHASE 8 — Whole-ROI fitting
# STATUS: DONE — expanded beyond original plan
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/fitting_roi.py
#
# Public functions written (as planned, plus delay variants added):
#
# fit_roi(sig_4d, mask, aif, time_var, model, acq, subject, fitting_cfg,
#         roi_cfg, seed)  →  RoiFitResult
#   Aggregates ROI to a mean curve via _aggregate_roi_signal, fits via
#   the concentration-space internal fitter, and reconstructs a
#   predicted signal curve for plotting (St_pred).
#
# fit_multi_roi(sig_4d, seg_vol, aif, time_var, model, acq, subject,
#               fitting_cfg, roi_cfg, seed)  →  dict[int, RoiFitResult]
#   Loops over unique non-zero labels in seg_vol, calls fit_roi per label.
#
# fit_roi_with_delay_grid / fit_roi_with_delay_continuous            ← NEW
#   ROI equivalents of Phase 5B's delay-fitting functions — aggregate
#   to mean concentration curve first, then delegate to the
#   concentration-space delay fitters in fitting.py.
#
# _aggregate_roi_signal(sig_4d, mask, acq, subject, cfg)
#   →  (Ct_mean, n_voxels, St_mean)
#   Supports both aggregation spaces via cfg.average_space:
#     'signal'        — average raw signal, then convert to C(t)
#     'concentration' — convert each voxel to C(t), then average
#                       (with optional top_frac trimming by peak
#                       enhancement before averaging)
#
# Verification: tests/ROI_fit_test.py — all 5 tests pass. TCXM and GKM
# both recover ground-truth parameters under both aggregation spaces;
# grid and continuous delay search both correctly recover a known
# tau=8.4s AIF shift; continuous delay beats grid cost when true tau
# falls between grid points, as expected                        — OK


# ══════════════════════════════════════════════════════════════════════
# PHASE 9 — Bootstrap uncertainty
# STATUS: NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/bootstrap.py
#
# All of the following currently `raise NotImplementedError`:
#
# gaussian_noise_many(y, sigma, B, seed)  →  (B, T) array
#   Source: tcxm_bootstrap_pipeline.gaussian_noise_many — copy as-is
#
# run_bootstrap(sig_orig, aif, time_var, model, acq, subject,
#               fitting_cfg, bootstrap_cfg, sigma)
#   →  BootstrapResult
#   Source: tcxm_bootstrap_pipeline.bootstrap_concentration or
#   bootstrap_signal, unified so fitting_cfg.fit_space picks the domain.
#   Internally should call fit_single_voxel per replicate — can now use
#   call_fitter() from fitting.py if you also want dispersed/delay modes
#   inside the bootstrap loop (not in the original plan, but the
#   dispatch machinery from Phase 5B/6 already supports it).
#
# run_sigma_sweep(S_est, aif, time_var, sigmas, model, acq, subject,
#                 fitting_cfg, bootstrap_cfg, out_dir)
#   →  dict[sigma: BootstrapResult]
#   Loops run_bootstrap over sigmas; vascular AIF correction optional
#   via a use_local_aif flag (depends on Phase 4B being done first if
#   you want that path — otherwise skip it initially and use the
#   global AIF for every sigma).
#
# _bootstrap_loop(fit_fn, sig_boot, aif, time_var, acq, subject,
#                 fitting_cfg, verbose)  →  BootstrapResult
#
# Not yet stubbed out but planned:
# run_ace_bootstrap(...) — depends on Phase 10 (ACE protocol) existing
# first, so build this after ace_protocol.py.
#
# Verification (once implemented):
#   result = run_bootstrap(sig, aif, t, TCXM, acq, subject,
#                           FittingConfig(), BootstrapConfig(), sigma=0.02)
#   assert result.params_boot.shape == (1000, 4)


# ══════════════════════════════════════════════════════════════════════
# PHASE 10 — ACE protocol
# STATUS: NOT STARTED (entire file is stubs)
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/ace_protocol.py
#
# FitMode enum
#   Source: ace_mri_model.FitMode — copy exactly
#
# ACEProtocol dataclass
#   Fields: t_edges, alphas, TRs, r1
#   Methods: n_segments, segment_of(t), alpha_at(t), TR_at(t)
#   Classmethod: paper_protocol(t_max) — encoding schedule from Zhang et al.
#   Source: ace_mri_model.ACEProtocol — copy exactly
#
# spgr_signal_vec(R1_t, alpha_t, TR_t_ms, B1, S0)  →  S_t
#   Source: ace_mri_model.spgr_signal_vec — copy exactly
#
# baseline_S0(B1, T10, protocol)  →  S0
#   Source: ace_mri_model._baseline_S0 — copy exactly
#
# ace_forward(t, Cp, B1, T10, theta_kin, protocol, kinetic_model_fn, S0)
#   →  S_t
#   Source: ace_mri_model.ace_forward — copy exactly
#   Note: kinetic_model_fn is plug-and-play — can be tcxm_ct_analytic,
#         gkm_ct, or (with n_level fixed) dispersed_gkm_ct
#
# fit_ace_model_based(sig, aif, time_var, protocol, kinetic_model_fn,
#                     kin_lb, kin_ub, kin_p0, kin_constraints,
#                     fit_mode, B1_fixed, T10_fixed, n_lhs, n_top, seed)
#   →  (b1_t10, theta_kin, cost, success, S_pred)
#   Source: ace_mri_model.fit_ace_model_based — copy exactly
#
# bootstrap_ace_mb(sig_boot, ...)  →  BootstrapResult
#   Source: ace_mri_model.bootstrap_ace_mb — copy; update return type
#
# Verification (once implemented):
#   from dce_mri.ace_protocol import ACEProtocol, ace_forward
#   protocol = ACEProtocol.paper_protocol()
#   S = ace_forward(t, aif, B1=1.0, T10=1.53, theta_kin=TCXM.p0,
#                   protocol=protocol, kinetic_model_fn=tcxm_ct_analytic)
#   assert S.shape == t.shape


# ══════════════════════════════════════════════════════════════════════
# PHASE 11 — Statistics, I/O, Visualization
# STATUS: io.py nearly done; statistics.py and visualization.py NOT STARTED
# ══════════════════════════════════════════════════════════════════════
#
# File: src/dce_mri/statistics.py  — all three functions are stubs
#   param_stats(params, p_ref)
#     Source: tcxm_bootstrap_pipeline.param_stats — copy as-is
#   compute_over_sweep(results, sweep_values, method_key, p_ref, success_only)
#     Source: tcxm_bootstrap_pipeline.compute_over_sweep — copy as-is
#   ace_param_stats(b1t10_boot, kin_boot, ok_boot, b1t10_ref, kin_ref, kin_names)
#     Source: ace_mri_model.ace_param_stats — copy as-is
#     (depends on Phase 10 existing to be meaningful, but the function
#      itself has no import-time dependency on ace_protocol.py)
#
# File: src/dce_mri/io.py  — done except one function
#   load_aif(path, skip_rows) — DONE
#   load_bootstrap_npz(path) — DONE
#   save_results_npz(results, out_path, extra_arrays) — DONE
#   load_results_npz(path) — STILL A STUB
#     Source: tcxm_bootstrap_pipeline.load_results_from_npz — copy as-is
#
# File: src/dce_mri/visualization.py — all ten functions are stubs
#   plot_voxel_fit            — tcxm_bootstrap_pipeline.plot_voxel_fit
#   plot_parameter_maps       — HNC_TCXM_NLLS_ANA_ST plot block
#   plot_parameter_histograms — tcxm_bootstrap_pipeline.plot_parameter_histograms
#   plot_bootstrap_envelope   — ace_mri_integration_example envelope plot (new)
#   plot_metric_panel         — tcxm_bootstrap_pipeline.plot_metric_panel
#   plot_scatter_matrix       — tcxm_bootstrap_pipeline.plot_scatter_matrix
#   plot_aif_curves           — new, simple line plot
#   plot_dispersed_gkm_levels — new, shows all 6 level fits + AIC weights
#   plot_admm_convergence     — new, primal/dual residuals vs iteration
#   plot_roi_fit              — new, mean ROI curve + model prediction
#
# Suggested order within this phase: io.load_results_npz first (small,
# unblocks nothing else but is a 10-minute copy job), then
# statistics.py (bootstrap/sweep phases will want these once Phase 9
# exists), then visualization.py last since several plot_* functions
# depend on result types from phases that don't exist yet
# (plot_admm_convergence needs Phase 7, plot_bootstrap_envelope needs
# Phase 9).


# ══════════════════════════════════════════════════════════════════════
# PHASE 12 — Scripts (entry points)
# STATUS: NOT STARTED — scripts/ currently holds unrelated tooling
# ══════════════════════════════════════════════════════════════════════
#
# Directory: scripts/
# These are thin wrappers. They parse arguments, build config objects,
# call the workflow, and save results. No business logic here.
#
# Currently in scripts/: tighten_lesion_rois.py — a real, working
# per-patient ROI-cleanup tool (splits AIF/lesion masks, computes an
# enhancement-ratio map, drops non-enhancing voxels from lesion masks).
# This is legitimate infrastructure but is not one of the four planned
# entry points below — keep it, just don't confuse it with Phase 12
# progress.
#
# scripts/run_voxelwise.py
#   1. Parse: subject_id, model name, fit_space, n_workers, chunk_size
#   2. Build: AcquisitionConfig, SubjectConfig, FittingConfig, VoxelwiseConfig
#   3. Load:  sig_4d, aif, mask (using io.py + volume.load_nifti_canonical)
#   4. Run:   run_voxelwise_fitting(...)
#   5. Save:  save_nifti_maps(result, ref_img, cfg)
#
# scripts/run_voxelwise_admm.py
#   Same as above but calls run_admm_fitting instead (needs Phase 7)
#   Extra args: rho, lambda_tv, n_admm_iter, warm_start
#
# scripts/run_roi_fitting.py
#   1. Parse: subject_id, model, aggregation, top_frac
#   2. Build configs
#   3. Load: sig_4d, aif, seg_vol
#   4. Run: fit_multi_roi(...)   — already implemented, ready to wrap
#   5. Print per-label parameter tables, save plots (needs Phase 11 viz)
#
# scripts/run_bootstrap_sweep.py
#   1. Parse: subject_id, model, sigmas, B, n_lhs, n_top
#   2. Build configs
#   3. Load: sig, aif
#   4. Run: run_sigma_sweep(...)   — needs Phase 9
#   5. Save: save_results_npz; generate metric panel plots (needs Phase 11)
#
# Note: run_roi_fitting.py is the one script you could write today
# without waiting on any other phase — fit_multi_roi is fully done.


# ══════════════════════════════════════════════════════════════════════
# PHASE 13 — Notebooks
# STATUS: diverged from plan — needs reconciliation
# ══════════════════════════════════════════════════════════════════════
#
# Directory: notebooks/
# Plan called for six notebooks numbered 01-06 covering one concept
# each. What actually exists: 06_roi_fitting.ipynb and
# 07_voxelwise_fitting.ipynb (plus a scratch test_notebook.ipynb).
# The numbering doesn't match the original scheme and 01-05 were never
# created. Given ROI fitting and voxelwise fitting (Phases 8 and 6) are
# both done and tested, the existing notebooks are plausibly covering
# real ground — worth reviewing their actual content against the list
# below before renumbering, rather than assuming they need to be
# replaced outright.
#
# Original planned set, for reference:
#
# 01_signal_to_concentration.ipynb
#   Load one voxel signal, convert, plot C(t) vs S(t)
#   Sweep T10 and B1 to show sensitivity
#
# 02_single_voxel_fitting.ipynb
#   Load one voxel, fit TCXM and GKM, compare results
#   Show LHS coarse landscape + TRF refinement path
#
# 03_dispersed_gkm.ipynb
#   Demonstrate fit_dispersed_gkm on one voxel
#   Plot per-level costs (plot_dispersed_gkm_levels — needs Phase 11)
#   Compare best-level local AIF vs global AIF (needs Phase 4B)
#
# 04_ace_protocol.ipynb
#   Show ACEProtocol FA/TR schedule (needs Phase 10)
#   Simulate ACE signal for known params
#   Fit back and recover B1, T10, kinetics
#
# 05_bootstrap_uncertainty.ipynb
#   Run sigma sweep on one voxel (needs Phase 9)
#   Plot bias, CoV, scatter matrix (needs Phase 11)
#
# 06_voxelwise_maps.ipynb  — could plausibly be today's 07? worth checking
#   Load a subject, run voxelwise fitting, display maps
#   Compare standard vs ADMM results side by side (ADMM needs Phase 7)


# ══════════════════════════════════════════════════════════════════════
# DEAD CODE TO DELETE
# ══════════════════════════════════════════════════════════════════════
#
# signal_models.py: _spgr_E1, _spgr_signal — appear unused; the real
#   logic is inlined directly in signal_to_concentration and
#   concentration_to_signal. Confirm no notebook imports them, then delete.
#
# src/test_utils.py — imports torch/ants/VAE/UNet_FS; looks like it
#   belongs to a different project entirely. Move out of src/ or delete.
#
# kinetic_models.py: commented-out lambda constraints (e.g.
#   `constraints = _tcxm_constraints, #lambda p: p[0] + p[1] <= 1.0,`)
#   — the named functions are in place and working; the trailing
#   comments are now just clutter, safe to delete.


# ══════════════════════════════════════════════════════════════════════
# DEPENDENCY MAP (which module imports from which)
# ══════════════════════════════════════════════════════════════════════
#
#   config.py           ← imports nothing from dce_mri
#   signal_models.py    ← config
#   kinetic_models.py   ← config, vascular_aif (for dispersed_gkm_ct)
#   vascular_aif.py     ← config (VascularAIFResult defined locally)
#   ace_protocol.py     ← config, kinetic_models
#   fitting.py          ← config, kinetic_models, signal_models
#   fitting_roi.py       ← config, kinetic_models, signal_models, fitting
#   fitting_admm.py     ← config, kinetic_models
#   bootstrap.py        ← config, kinetic_models, fitting, ace_protocol
#   volume.py           ← config, kinetic_models, fitting
#   statistics.py       ← config  (numpy only)
#   io.py               ← config  (numpy, nibabel only)
#   visualization.py    ← config, kinetic_models, statistics
#
# No circular imports. config.py is the only module with zero dce_mri
# imports — confirmed still true after Phases 4C/5B additions.


# ══════════════════════════════════════════════════════════════════════
# RECOMMENDED NEXT STEPS (build order for what's left)
# ══════════════════════════════════════════════════════════════════════
#
# You have a working, tested TCXM/GKM pipeline end to end for both
# single-voxel and whole-ROI fitting, including AIF-delay correction.
# Suggested order for what remains, roughly cheapest/most-unblocking
# first:
#
#   1. io.load_results_npz()        — ~10 min, small, unblocks nothing
#                                       else but easy to clear off the list
#   2. volume.create_binary_masks() — small, needed for multi-label mask
#                                       workflows
#   3. scripts/run_roi_fitting.py   — can write today, fit_multi_roi is done
#   4. Phase 4B (vascular_aif fitting) — unlocks local-AIF-corrected
#      dispersed-GKM fitting; you already have the transfer functions
#   5. Phase 9 (bootstrap)          — high value for your uncertainty
#      work; can be written against TCXM/GKM today without waiting on
#      Phase 4B or 10
#   6. Phase 11 statistics.py       — needed once Phase 9 exists
#   7. Phase 7 (ADMM)               — independent of 4B/9/10; scikit-image
#      is now a declared dependency so _tv_prox can call
#      denoise_tv_chambolle directly instead of hand-rolling Chambolle
#   8. Phase 10 (ACE protocol)      — only if you're actively using B1/T10
#      joint estimation right now; otherwise fine to defer
#   9. Phase 11 visualization.py    — write incrementally as each
#      upstream phase lands rather than all at once at the end
#   10. Phase 12 remaining scripts, Phase 13 notebook reconciliation —
#      last, once the functions they wrap actually exist