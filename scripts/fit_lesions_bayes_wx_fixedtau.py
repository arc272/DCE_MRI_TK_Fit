from __future__ import annotations

"""
Created on Tue Sep 22 15:01:11 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_lesions_bayes_wx_fixedtau.py
================================
Bayesian TCXM + FIXED delay + WATER EXCHANGE (SSM3, tau_i free) fit of the
LARGEST lesion in each subject listed in tumor_identification.csv, using the
AIF CSVs produced by extract_aifs.py.

Three things differ from fit_lesions_bayes_wx.py:

 1. tau_delay is NOT sampled. It is read per subject from an earlier
    TCXM + joint-delay (no water exchange) run -- TAU_FIXED_FROM_RUN,
    column TAU_FIXED_COL -- and held fixed, so the sampled vector is
    [ve, vp, fp, ps, tau_i]: 5 parameters instead of 6. The delay's own
    uncertainty is NOT propagated; the source 95% CrI width is copied into
    tau_delay_src_ci_width so the discarded uncertainty stays visible.

 2. The DCE volume comes from the reconstruction chosen by RECON
    (default "grasp_8spf": the single 4-D file in
    GRASP_NIfTI/fastMRI_breast_{nnn}_{SCAN_INDEX}_8spf_NII/, whose
    slice_000_frame_000 name is a misnomer). Lesion masks still come from
    MASK_ROOT, and the mask/volume grid match is checked per subject.

 3. The baseline window is FIXED at NBASE_FALLBACK = frames 0-6
    (AUTO_NBASE = False).

The lesion curve itself is unchanged: _aggregate_roi_signal with
roi_cfg.average_space="signal" takes the MEAN SIGNAL ACROSS THE LESION'S
VOXELS at each time point, giving one signal-intensity-vs-time curve per
lesion, which is what both the Bayesian and NLLS fits see.

Model / sampler: verify_03 / verify_04b setup PLUS water exchange
    TCXM, delay_mode='fixed', wx_model=SSM3 with relax_cfg.fit_tau_b=False
    (so tau_i is sampled and tau_b is held at relax_cfg.tau_b_fixed),
    PRIOR_CHOICES = {rates: loguniform, volfrac: beta, tau: flat},
    gauss_marginalized_sigma, BlackJAX NUTS, dense mass matrix.
    Set WX_MODEL = None to recover the plain TCXM fit.

    The sampled vector is [ve, vp, fp, ps, tau_i] -- 5 parameters. Results are
    NOT directly comparable to a joint-delay run's CSV. The `model` and
    `delay_mode` columns record which ran.
    SSM3 uses vi = 1 - vp - ve as the intracellular fraction (vb := vp), so vi
    is reported as a derived quantity and flagged when it approaches 0, where
    tau_i has almost no pool left to describe.

MCSE: every parameter AND derived quantity now carries R-hat, ESS, MCSE(mean)
    and MCSE(median), plus mcse_median_rel_ci = MCSE(median) / 95% CrI width.
    That last one gates qc_pass via MCSE_REL_MAX and is the scale-free answer
    to "is the reported median resolved well enough", which BARG step 2C asks
    for and a flat ESS cutoff cannot give. BARG also asks for the convergence
    and resolution of every DERIVED value, which is why the derived quantities
    get the same four columns (computed from the per-chain derived draws).
    Note BARG's ESS >= 10,000 recommendation is for HDI limits; this script
    reports medians and equal-tailed intervals, for which Kruschke states ESS
    "can be lower".
Tissue signal model: 06_roi_fitting.ipynb cell [4]
    AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9), SubjectConfig(T10=1530, B1=1.0),
    FittingConfig(n_lhs=200, n_top=10, fit_space="signal"), mean signal over
    the lesion (average_space="signal").

Baseline window: detected PER LESION from its own mean signal curve, with the
    same algorithm and the same priority order as extract_aif_roi.py
    (NBASE_OVERRIDE > auto > NBASE_FALLBACK). The window enters the fit as a
    per-subject AcquisitionConfig (dataclasses.replace(acq, nbase=...)), so
    S0, Ct, both fits and the residual checks all use it. Every subject's
    window, its source and its QC flags are columns in the results CSV.

    NBASE_METHOD picks the arrival rule. On synthetic exponential-uptake
    curves (36 frames, baseline 100, true arrival 12; the expected detected
    arrival is 13, because the arrival frame itself sits exactly at baseline):
      enhancement sweep at 1% noise, median detected arrival --
        peak enh   1.02x 1.05x 1.10x 1.25x 1.50x 2.00x 3.00x 5.00x
        peak_frac      9    13    13    13    13    13    13    13
        noise_sd      26    33    28    14    13    13    13    13
      noise sweep at 1.50x enhancement, median [min, max] over 40 seeds --
        0.5%   peak_frac 13 [13,13]    noise_sd 13 [13,13]
        1.0%   peak_frac 13 [13,14]    noise_sd 13 [13,14]
        2.0%   peak_frac 13 [12,14]    noise_sd 14 [13,19]
        5.0%   peak_frac 14 [ 6,15]    noise_sd 30 [14,35]
         10%   peak_frac 14 [ 5,17]    noise_sd 31 [19,35]
    peak_frac is the default: it held the right answer across the amplitude
    range and degraded more gracefully with noise. These are simulations of
    one uptake shape, NOT a validation on breast data -- inspect the detected
    windows and the flags on the real cohort before trusting either rule.

NLLS comparison: 06_roi_fitting.ipynb cell [13] pipeline
    fit_roi_with_delay_continuous(TCXM, same acq / subject / cfg_fit / roi_cfg,
    tau bounds = DELAY_BOUNDS). With NLLS_INTERP = "linear" and NLLS_SEED = SEED
    this is the same computation as the Bayesian fit's internal NLLS anchor
    (res.point_estimate) -- nlls_vs_bayes_anchor_maxdiff checks that they agree.

Per subject:
  * {sid}_lesion_inspection.png      lesion slice with the fitted component
  * {sid}_prior_post.png             prior vs posterior, every parameter (+ NLLS line)
  * {sid}_fit.png                    data vs posterior-median and NLLS predictions
                                     + residuals
  * {sid}_draws.npz                  all chains (for later BEST / derived checks)
  * rows/{sid}.csv                   one-row result (lets a crashed run resume)
Cohort:
  * lesion_bayes_results.csv         every estimate + QC metric, one row / subject
                                     (NLLS columns prefixed nlls_; MCSE columns
                                     {param}_mcse_mean / _mcse_median /
                                     _mcse_median_rel_ci / _mcse_rel_flag)
  * group_BvsM_all.png               mean +/- SD per metric, B vs M, all fits
  * group_BvsM_qcpass.png            same, fits passing QC only
  * group_BvsM_nlls.png              same, NLLS estimates
  * nlls_vs_bayes.png                NLLS vs posterior median, per metric
  * group_summary.csv                n / mean / SD / median per metric, method, group

Units: fp, ps, Ktrans in s^-1 (the model's units); tau_delay, Tc, Te, tau_i
and tau_b in s. tau_i is log-sampled (the "flat" tau prior is log-uniform over
the relax_cfg bounds), so it is in LOG_SCALE for the IQR-ratio QC metric.
fp and ps are PLOTTED on linear axes (PLOT_LINEAR); their prior-dominated IQR
ratio is still computed in log10 (LOG_SCALE), because that is a QC metric, not
a display choice.

Run:
    PYTHONPATH=src python scripts/fit_lesions_bayes_wx.py
"""

import glob
import os
import time
import warnings
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as nd_label

warnings.filterwarnings("ignore")


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            RoiFittingConfig, AifConfig, RelaxationConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.signal_models import compute_S0, shift_aif
from dce_mri.fitting_roi import (_aggregate_roi_signal, fit_roi,
                                fit_roi_with_delay_continuous)
from dce_mri.volume import load_nifti_canonical
from dce_mri.synthetic import simulate_curve
from dce_mri.bayes import (BayesConfig, fit_curve_bayes, build_param_layout,
                           NOISE_OPTIONS)
from dce_mri.bayes_priors import assemble_prior

# =============================================================================
# settings
# =============================================================================

TUMOR_CSV   = "/home/allisonchen/tumor_identification.csv"
MASK_ROOT   = "/mnt/labspace/users/allison/fastMRI_data/tight_masks_2/canonical"
AIF_DIR     = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"  # extract_aif_roi.py
AIF_SUMMARY = os.path.join(AIF_DIR, "AIF_extraction_summary.csv")   # optional
RESULTS_ROOT = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results"

# --- which reconstruction supplies the DCE volume ---------------------------
# Lesion masks (and the vascular ROIs behind the AIF CSVs) ALWAYS come from
# MASK_ROOT. Only the 4-D image volume changes with RECON:
#   "grasp_8spf"     GRASP_NIfTI/fastMRI_breast_{nnn}_{SCAN_INDEX}_8spf_NII/
#                    holds ONE 4-D file whose name (slice_000_frame_000.nii.gz)
#                    is a misnomer -- it is the whole reconstruction, not one
#                    slice/frame. find_dce globs the directory and requires
#                    exactly one match, so a folder that really did contain
#                    per-slice files fails loudly instead of fitting one slice.
#   "tight_masks_2"  the canonical volumes, {sid}_tv*.nii.gz in MASK_ROOT
RECON      = "grasp_8spf"        # "grasp_8spf" | "tight_masks_2"
GRASP_ROOT = "/mnt/labspace/users/allison/fastMRI_data/GRASP_NIfTI"
SCAN_INDEX = 2                   # the "_2_" field: same scan as the
                                 # tight_masks_2 fits

ONLY = []            # e.g. ["FMB056"] to run a single subject first
RESUME_DIR = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/run_20261021_160427/"# None      # path of an earlier run dir to resume (skips done subjects)

# --- tau_delay: FIXED per subject from an earlier TCXM + joint-delay run -----
# The delay is read from that run's lesion_bayes_results.csv and held fixed
# here, so this fit samples 5 parameters (ve, vp, fp, ps, tau_i) instead of 6.
#
# How it enters the model: BayesConfig(delay_mode="fixed", delay_fixed=tau)
# makes bayes.fit_curve_bayes pre-shift the AIF by tau
# (signal_models.shift_aif) and build the predictor with delay=False, so
# tau_delay is NOT in the sampled vector -- verified to give exactly
# ('ve','vp','fp','ps','tau_i'). The NLLS arm below is given the SAME shifted
# AIF, which is what bayes.py's own NLLS anchor does for this delay mode.
#
# What this buys and costs: one less dimension and a cleaner wx posterior,
# at the price of treating tau_delay as known. The delay's uncertainty is NOT
# propagated, so ve/vp/fp/ps credible intervals here are narrower than a joint
# fit's by an amount this run cannot measure. The source run's tau_delay 95%
# CrI width is copied into tau_delay_src_ci_width so the discarded uncertainty
# is at least visible per lesion.
TAU_FIXED_FROM_RUN = ("/mnt/labspace/users/allison/fastMRI_data/"
                      "lesion_bayes_results/run_grasp_8spf_20261013_180711_allflatprior")
TAU_FIXED_COL = "tau_delay_median"   # or "tau_delay_mean", "nlls_tau_delay"
# What to do when a subject has no usable delay in the source run:
#   "error"         : record an error row and skip the subject (default --
#                     never invent a delay)
#   "cohort_median" : use the median tau over the source run's usable subjects
#   "zero"          : use 0 s (i.e. no delay)
TAU_MISSING = "error"
# Also refuse a source value whose own fit failed QC there. Set False to accept
# any finite value (the source QC is still recorded either way).
TAU_REQUIRE_SRC_QC = False

# files in the subject folder that are NOT lesion candidates (case-insensitive)
LESION_EXCLUDE = ("orta", "idif")    # aorta ROI, FireVoxel IDIF ROI

# --- baseline window, auto-detected per lesion ------------------------------
# Same structure and the same default algorithm as extract_aif_roi.py
# (priority: NBASE_OVERRIDE > auto > fallback), but run on the LESION mean
# signal curve instead of the vascular ROI curve.
# AUTO_NBASE = False: every lesion uses NBASE_FALLBACK, i.e. frames 0-6.
# The detection machinery below is kept (and still reports baseline CV/drift
# QC) but is not used to CHOOSE the window while this is False.
# Keeping the window fixed also keeps the AIF padding safe: with
# delay_mode="fixed", signal_models.shift_aif fills the frames before the
# shift with mean(aif[acq.nbase]), so a window that reached past the AIF's
# arrival would inject real AIF upslope into the pre-bolus frames. With
# frames 0-6 and an AIF that arrives at frame ~8 that mean is 0. The per
# subject aif_pad_mM column records it so a subject where it is not 0 is
# visible rather than silent.
AUTO_NBASE     = False          # True: detect per subject from the lesion curve
NBASE_FALLBACK = slice(0, 7)    # used if AUTO_NBASE=False or detection fails
NBASE_OVERRIDE = {              # per-subject manual windows, win over auto
    # "FMB194": slice(0, 9),
}
# "peak_frac": extract_aif_roi.detect_nbase's rule, unchanged -- arrival is the
#              first frame above base + ARRIVAL_FRAC x (peak - base), so the
#              threshold scales with ENHANCEMENT AMPLITUDE and is self-scaling:
#              in simulation it tracked a known arrival across 1.05x-5x
#              enhancement. Its failure mode is firing EARLY on noise once
#              enhancement is barely above the noise floor (~1.02x), and it
#              degrades gradually as noise grows.
# "noise_sd" : arrival is the first frame above base + ARRIVAL_K_SD x (SD of
#              the current baseline), so the threshold scales with the curve's
#              own NOISE. It needs enhancement well clear of that noise: in the
#              same simulation it matched peak_frac above ~1.25x enhancement at
#              <=2% noise, but below that it pushed arrival to the END of the
#              scan (frame 26-33 of 36 for a true arrival of 12), letting the
#              baseline swallow the uptake. Pair it with N_MAX_BASE if used.
# peak_frac is the default on that evidence; the numbers are in the docstring.
NBASE_METHOD   = "peak_frac"
ARRIVAL_FRAC   = 0.10   # "peak_frac" threshold: fraction of (peak - base)
ARRIVAL_K_SD   = 4.0    # "noise_sd" threshold: multiples of the baseline SD
N_MIN_BASE     = 5      # minimum baseline length (frames)
N_CONFIRM      = 2      # frames that must stay above the threshold
BASE_MARGIN    = 1      # frames dropped immediately before arrival
N_MAX_BASE     = None   # hard cap on baseline length, or None. A lesion that
                        # barely enhances can push arrival to the end of the
                        # scan; the cap stops the baseline eating the uptake.
BASE_CV_FLAG   = 0.10   # flag baselines noisier than this CV
DRIFT_T_FLAG   = 3.0    # flag if |baseline slope| > 3 x its standard error
BASE_LONG_FRAC = 0.40   # flag baselines longer than this fraction of the scan
AIF_PAD_FLAG   = 0.01   # flag if mean(aif[nbase]) exceeds this (mM): the value
                        # shift_aif pads the pre-shift frames with

# The tissue curve cannot start rising before contrast reaches the feeding
# artery, and DELAY_BOUNDS starts at 0 s, so the AIF arrival frame is a lower
# bound on the lesion arrival frame. A detected lesion arrival EARLIER than
# the AIF arrival means noise tripped the detector (or the AIF arrival itself
# is wrong).
#   "flag"   : record the disagreement, keep the detected window
#   "clamp"  : move arrival up to the AIF arrival frame when it is earlier
#   "ignore" : do not compare
AIF_ARRIVAL_MODE = "flag"

# --- acquisition / tissue: 06_roi_fitting.ipynb cell [4] --------------------
# acq.nbase here is only the FALLBACK window. Every subject gets its own acq
# via dataclasses.replace(acq, nbase=<detected window>) -- the same pattern
# extract_aif_roi.py uses -- so nothing downstream silently keeps the
# fallback. This works because average_space="signal" makes the lesion mean
# signal curve St independent of nbase (fitting_roi._aggregate_roi_signal:
# St_mean = sig_roi.mean(axis=0)), so St can be computed first, the window
# detected from it, and the resulting acq used for every nbase-dependent
# quantity (S0, Ct, both fits, the residual checks).
acq     = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE_FALLBACK)
T10_TISSUE, B1 = 1530.0, 1.0
cfg_fit = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
roi_cfg = RoiFittingConfig(average_space="signal")
SPOKES_PER_FRAME, NSPOKE, SCANTIME = 8, 288, 150



# --- water exchange ---------------------------------------------------------
# TCXM + SSM3 (3-pool shutter-speed) with tau_i free and tau_b FIXED, which is
# what relax_cfg.fit_tau_b=False selects (bayes.build_param_layout appends only
# "tau_i"; jax_forward.make_predict then uses relax_cfg.tau_b_fixed). Set
# WX_MODEL = None to fall back to the plain TCXM fit -- every wx argument is
# passed through as None in that case, which the package treats as "no wx".
#
# Requirements the package enforces (fitting._validate_wx_compatibility), all
# already satisfied here: fit_space="signal", average_space="signal",
# TCXM.fn_compartments is not None, and SSM3.n_pools == 3.
WX_MODEL  = SSM3
relax_cfg = RelaxationConfig(
    fit_tau_b   = False,     # tau_i only (tau_b was not identifiable in tests)
    tau_b_fixed = 0.005,     # s, used because fit_tau_b=False
    tau_i_lb    = 0.01,      # s   (must be > 0: the tau block is log-uniform,
    tau_i_ub    = 1.5,       # s    so blackjax log-transforms these bounds)
)
# SSM3 needs blood T1 and the hematocrit for the vascular pool. These must match
# extract_aif_roi.py's AifConfig, or the AIF and the forward model disagree.
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)

# --- Bayesian model: validated verify_03 / verify_04b configuration ---------
# "tau" is the water-exchange block: "flat" is log-uniform over the relax_cfg
# bounds, and is the only wx choice the blackjax transform implements
# (bayes_blackjax._LOG_FLAT_SOURCES). It is also the choice that lets a
# tau_i posterior which has learned nothing stay visible as prior-dominated.
PRIOR_CHOICES = {"rates": "uniform", "volfrac": "flat_masked", "tau": "flat"}
DELAY_BOUNDS  = (0, 25.0)
N_DRAWS, N_TUNE, N_CHAINS = 4000, 4000, 16
BACKEND, DENSE_MASS, SEED = "blackjax", True, 1
N_PRIOR_SAMPLES, PRIOR_SEED = 20000, 999

# --- NLLS comparison: 06_roi_fitting.ipynb cell [13] pipeline ----------------
RUN_NLLS     = True
NLLS_INTERP  = "linear"  # "linear": the SAME AIF shift the Bayesian joint-delay
                         #   model uses (jnp.interp), so NLLS vs Bayes compares
                         #   estimators, not forward models.
                         # "pchip": what cell [13] used.
NLLS_TAU_P0  = 0.0       # s
NLLS_SEED    = SEED      # same LHS draw as the Bayesian fit's NLLS anchor
AT_BOUND_TOL = 1e-3      # NLLS estimate within this fraction of the bound range
                         # (log range for log-sampled params) = "at bound"

# --- QC thresholds ------------------------------------------------------------
RHAT_MAX       = 1.01
ESS_MIN        = 400
IQR_RATIO_FLAT = 0.8     # posterior/prior IQR above this = prior-dominated
E_FLOW_LIMITED = 0.4     # verify_03: ps recovery degrades once E=ps/(ps+fp) > ~0.3-0.5
CHAIN_SPLIT    = 0.5     # per-chain median spread > 0.5 x within-chain IQR = disagree
# MCSE of the REPORTED point estimate (the posterior median), as a fraction of
# that parameter's own 95% CrI width. This is the quantity BARG step 2C is
# really after: it says how precisely the MCMC resolved the number being
# reported, and unlike a flat ESS cutoff it is per-quantity and scale-free.
# 0.05 = the Monte Carlo error is at most 5% of the interval width.
MCSE_REL_MAX   = 0.05
VI_SMALL       = 0.05    # vi = 1 - ve - vp below this: tau_i is unidentifiable
                         # because there is almost no intracellular pool left

# tau_i is a SAMPLED parameter when WX_MODEL is not None, so it joins PARAMS
# (it gets R-hat / ESS / MCSE / prior-dominated like any other).
# tau_delay is FIXED, so it is not sampled and not in PARAMS; it is reported
# as a constant (tau_delay_fixed) instead.
PARAMS  = ["ve", "vp", "fp", "ps"] + (["tau_i"] if WX_MODEL else [])
DERIVED = ["vtot", "vi", "E", "Ktrans", "Tc", "Te"]
# tau_i is log-sampled (the "flat" tau prior is log-uniform), so its IQR ratio
# is computed in log10 like the other log-scale quantities.
LOG_SCALE = {"fp", "ps", "Ktrans", "Tc", "Te"} | ({"tau_i"} if WX_MODEL else set())
PLOT_LINEAR = {"fp", "ps"}                          # drawn on linear axes anyway
PLOT_LOG    = LOG_SCALE - PLOT_LINEAR               # everything else stays log
UNITS = {"fp": "s$^{-1}$", "ps": "s$^{-1}$", "Ktrans": "s$^{-1}$",
         "tau_delay": "s", "Tc": "s", "Te": "s", "tau_i": "s"}
GROUP_METRICS = (["ve", "vp", "fp", "ps", "vtot", "E", "Ktrans"]
                 + (["tau_i", "vi"] if WX_MODEL else []))
MODEL_TAG = ("TCXM + fixed delay"
             + (f" + {WX_MODEL.name} (tau_i)" if WX_MODEL else ""))
DELAY_MODE = "fixed"

C_BAYES, C_MEDIAN, C_PRIOR = "#B22234", "#E67E22", "0.75"
C_NLLS = "#1E8449"
C_GROUP = {"B": "#1B4F72", "M": "#B22234"}


# =============================================================================
# inputs
# =============================================================================

def mask_dir_for(sid):
    """Lesion / vascular masks always live in the canonical folder."""
    d = os.path.join(MASK_ROOT, sid)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{sid}: no mask folder at {d}")
    return d


def grasp_subj_dir(sid):
    """FMB001 -> GRASP_NIfTI/fastMRI_breast_001_{SCAN_INDEX}_8spf_NII."""
    try:
        n = int(sid[3:])
    except ValueError:
        raise ValueError(f"{sid}: subject ID is not FMB + digits, so it cannot "
                         f"be mapped to a {RECON} folder name")
    d = os.path.join(GRASP_ROOT, f"fastMRI_breast_{n:03d}_{SCAN_INDEX}_8spf_NII")
    if not os.path.isdir(d):
        near = sorted(os.path.basename(p) for p in
                      glob.glob(os.path.join(GRASP_ROOT, f"fastMRI_breast_{n:03d}_*")))
        raise FileNotFoundError(f"{sid}: no {os.path.basename(d)} in {GRASP_ROOT}; "
                                f"found for this subject: {near}")
    return d


def find_dce(sid):
    """The 4-D DCE volume for the selected recon.

    For grasp_8spf the file name (slice_000_frame_000.nii.gz) does not describe
    the contents -- it is the whole 4-D reconstruction -- so the directory is
    globbed and exactly one volume is required. A folder that really did hold
    per-slice/per-frame files fails here instead of silently fitting one slice.
    """
    if RECON == "tight_masks_2":
        d = mask_dir_for(sid)
        hits = sorted(glob.glob(os.path.join(d, f"{sid}_tv*.nii.gz")))
    elif RECON == "grasp_8spf":
        d = grasp_subj_dir(sid)
        hits = sorted(glob.glob(os.path.join(d, "*.nii.gz"))
                      + glob.glob(os.path.join(d, "*.nii")))
    else:
        raise ValueError(f"unknown RECON {RECON!r}")
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{sid}: expected one volume in {d}, found {len(hits)}: "
            f"{[os.path.basename(h) for h in hits]}")
    return hits[0]


# --- fixed tau_delay from an earlier run -------------------------------------

_TAU_TABLE = None


def load_tau_table():
    """{sid: {...}} from TAU_FIXED_FROM_RUN's cohort CSV, read once.

    Carries the source run's own QC alongside the value, so a delay copied
    from a fit that did not converge there is visible here rather than silently
    treated as known.
    """
    global _TAU_TABLE
    if _TAU_TABLE is not None:
        return _TAU_TABLE
    path = os.path.join(TAU_FIXED_FROM_RUN, "lesion_bayes_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no lesion_bayes_results.csv in TAU_FIXED_FROM_RUN ({path}). "
            f"tau_delay cannot be fixed without the source run.")
    df = pd.read_csv(path)
    if TAU_FIXED_COL not in df.columns:
        raise KeyError(f"{path} has no column {TAU_FIXED_COL!r}; "
                       f"available tau columns: "
                       f"{[c for c in df.columns if 'tau_delay' in c]}")
    err = df["error"].fillna("").astype(str).str.strip() if "error" in df else ""
    df = df[err == ""]
    tb = {}
    for _, r in df.iterrows():
        v = float(r[TAU_FIXED_COL]) if pd.notna(r[TAU_FIXED_COL]) else np.nan
        lo, hi = r.get("tau_delay_ci2.5", np.nan), r.get("tau_delay_ci97.5", np.nan)
        tb[str(r["subject_id"]).strip()] = dict(
            tau=v,
            src_qc_pass=str(r.get("qc_pass", "")).strip().lower() == "true",
            src_prior_dominated=str(
                r.get("tau_delay_prior_dominated", "")).strip().lower() == "true",
            src_ci_width=(float(hi) - float(lo)
                          if pd.notna(lo) and pd.notna(hi) else np.nan),
            src_rhat=float(r.get("tau_delay_rhat", np.nan)),
            src_recon=str(r.get("recon", "")).strip())
    good = [v["tau"] for v in tb.values()
            if np.isfinite(v["tau"]) and (v["src_qc_pass"] or not TAU_REQUIRE_SRC_QC)]
    _TAU_TABLE = dict(table=tb, cohort_median=float(np.median(good)) if good else np.nan,
                      n=len(tb), n_good=len(good), path=path)
    return _TAU_TABLE


def tau_for(sid):
    """(tau_seconds, info dict). Applies TAU_MISSING when there is no value."""
    tt = load_tau_table()
    e = tt["table"].get(sid)
    info = dict(tau_delay_src_run=os.path.basename(TAU_FIXED_FROM_RUN.rstrip("/")),
                tau_delay_src_col=TAU_FIXED_COL)
    usable = (e is not None and np.isfinite(e["tau"])
              and (e["src_qc_pass"] or not TAU_REQUIRE_SRC_QC))
    if usable:
        info.update(tau_delay_fixed=float(e["tau"]), tau_delay_src="run",
                    tau_delay_src_qc_pass=e["src_qc_pass"],
                    tau_delay_src_prior_dominated=e["src_prior_dominated"],
                    tau_delay_src_ci_width=e["src_ci_width"],
                    tau_delay_src_rhat=e["src_rhat"],
                    tau_delay_src_recon=e["src_recon"])
        return float(e["tau"]), info
    why = ("absent from the source run" if e is None else
           "not finite in the source run" if not np.isfinite(e["tau"]) else
           "source fit failed QC (TAU_REQUIRE_SRC_QC=True)")
    if TAU_MISSING == "error":
        raise ValueError(f"{sid}: no usable {TAU_FIXED_COL} -- {why}. Set "
                         f"TAU_MISSING to 'cohort_median' or 'zero' to proceed "
                         f"anyway.")
    tau = tt["cohort_median"] if TAU_MISSING == "cohort_median" else 0.0
    if not np.isfinite(tau):
        raise ValueError(f"{sid}: {why}, and the source run has no usable "
                         f"delays to take a cohort median from")
    info.update(tau_delay_fixed=float(tau), tau_delay_src=TAU_MISSING,
                tau_delay_src_qc_pass=False, tau_delay_src_prior_dominated=False,
                tau_delay_src_ci_width=np.nan, tau_delay_src_rhat=np.nan,
                tau_delay_src_recon="")
    print(f"  !! {sid}: {why}; using {TAU_MISSING} tau = {tau:.3f} s")
    return float(tau), info


def select_largest_lesion(subj_dir, sid, dce_path):
    """
    Candidates = every .nii / .nii.gz in the subject folder except the vascular
    ROIs (any LESION_EXCLUDE fragment in the name: aorta, IDIF) and the DCE
    volume itself. The file with the most nonzero voxels is used, whole, as
    the lesion mask.
    """
    # the tight_masks_2 DCE volume lives in the mask folder, so it has to be
    # excluded by NAME -- the dce_path identity check only catches it when
    # RECON == "tight_masks_2"
    dce_names = {os.path.basename(p)
                 for p in glob.glob(os.path.join(subj_dir, f"{sid}_tv*.nii.gz"))
                 + glob.glob(os.path.join(subj_dir, f"{sid}_tv*.nii"))}
    files = sorted(
        f for f in os.listdir(subj_dir)
        if f.lower().endswith((".nii", ".nii.gz"))
        and not any(x in f.lower() for x in LESION_EXCLUDE)
        and f not in dce_names
        and os.path.join(subj_dir, f) != dce_path
    )
    if not files:
        raise FileNotFoundError(f"{sid}: no lesion masks (non-vascular .nii) in "
                                f"{subj_dir}")

    counts = {}
    for f in files:
        vol, _ = load_nifti_canonical(os.path.join(subj_dir, f))
        if vol.ndim != 3:
            raise ValueError(f"{sid}: '{f}' is {vol.ndim}-D, expected a 3-D mask")
        counts[f] = int(np.count_nonzero(vol))

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    best_file, n_vox = ranked[0]
    mask = load_nifti_canonical(os.path.join(subj_dir, best_file))[0] != 0

    info = dict(lesion_file=best_file, n_vox=n_vox,
                n_lesion_files=len(files),
                all_lesion_files="; ".join(f"{f} ({n})" for f, n in ranked),
                second_largest_vox=(ranked[1][1] if len(ranked) > 1 else 0))
    return mask, info


def load_aif_csv(sid, t):
    path = os.path.join(AIF_DIR, f"{sid}_AIF_plasma.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{sid}: no AIF CSV at {path} -- run extract_aifs.py "
                                f"for this subject first")
    df = pd.read_csv(path)
    aif = df["aif_plasma_mM"].to_numpy(float)
    if aif.size != t.size:
        raise ValueError(f"{sid}: AIF has {aif.size} frames, DCE has {t.size}")
    if "time_s" in df and not np.allclose(df["time_s"].to_numpy(float), t, atol=1e-3):
        raise ValueError(f"{sid}: AIF time_s does not match the DCE time grid")
    return aif, path


def detect_nbase(sig, method=None, frac=None, k_sd=None, n_min=None,
                 n_confirm=None, margin=None, n_max=None, n_iter=3):
    """
    Baseline window = the frames before contrast arrival in a signal curve.
    Returns (slice(0, stop), arrival_frame).

    "peak_frac" is extract_aif_roi.detect_nbase, unchanged: iterate the
    baseline median, threshold at base + frac x (peak - base), and take the
    first frame from n_min onwards that stays above it for n_confirm frames.
    "noise_sd" replaces that threshold with base + k_sd x SD(baseline).

    Raises ValueError if the curve cannot support an n_min-frame baseline.
    """
    method    = NBASE_METHOD if method is None else method
    frac      = ARRIVAL_FRAC if frac is None else frac
    k_sd      = ARRIVAL_K_SD if k_sd is None else k_sd
    n_min     = N_MIN_BASE if n_min is None else n_min
    n_confirm = N_CONFIRM if n_confirm is None else n_confirm
    margin    = BASE_MARGIN if margin is None else margin
    n_max     = N_MAX_BASE if n_max is None else n_max
    if method not in ("peak_frac", "noise_sd"):
        raise ValueError(f"NBASE_METHOD must be 'peak_frac' or 'noise_sd', "
                         f"got {method!r}")

    s = np.asarray(sig, float)
    if not np.all(np.isfinite(s)):
        raise ValueError("curve contains non-finite samples")
    peak_idx = int(np.argmax(s))
    if peak_idx < n_min + margin + 1:
        raise ValueError(f"peak at frame {peak_idx}: too early for a "
                         f"{n_min}-frame baseline")

    stop, arrival = n_min, peak_idx
    for _ in range(n_iter):
        base = float(np.median(s[:stop]))
        if method == "peak_frac":
            thresh = base + frac * (s[peak_idx] - base)
        else:
            sd = float(np.std(s[:stop], ddof=1)) if stop > 1 else 0.0
            if not sd > 0:
                raise ValueError(f"baseline SD is {sd} over frames 0-{stop - 1}: "
                                 f"no noise scale for method 'noise_sd'")
            thresh = base + k_sd * sd
        above = s[:peak_idx + 1] > thresh
        arrival = next((i for i in range(n_min, peak_idx + 1)
                        if above[i:i + n_confirm].all()), peak_idx)
        new_stop = max(n_min, arrival - margin)
        if new_stop == stop:
            break
        stop = new_stop
    if n_max is not None:
        stop = min(stop, int(n_max))
    return slice(0, stop), arrival


def baseline_drift(sig, nbase):
    """
    Straight-line fit over the baseline window (extract_aif_roi.baseline_drift).
    Returns (slope as % of the baseline mean per frame, slope / its standard
    error). NaN for windows shorter than 4 frames.
    """
    y = np.asarray(sig[nbase], float)
    if y.size < 4:
        return np.nan, np.nan
    x = np.arange(y.size, dtype=float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se_b = np.sqrt(np.sum(resid ** 2) / (y.size - 2) / np.sum((x - x.mean()) ** 2))
    t = b / se_b if se_b > 0 else np.inf
    return float(100.0 * b / y.mean()), float(t)


def choose_nbase(sid, St, aif_arr=np.nan):
    """
    (nbase, info) for one lesion curve. Priority: NBASE_OVERRIDE > auto >
    fallback, matching extract_aif_roi.choose_nbase. info records which branch
    was taken, the arrival frame, and every baseline QC quantity, so the
    choice is auditable per subject in lesion_bayes_results.csv.
    """
    n_frames = int(np.asarray(St).size)
    arrival, err, clamped = np.nan, "", False

    if sid in NBASE_OVERRIDE:
        nb, source = NBASE_OVERRIDE[sid], "override"
    elif AUTO_NBASE:
        try:
            nb, arrival = detect_nbase(St)
            source = f"auto:{NBASE_METHOD}"
            if (AIF_ARRIVAL_MODE != "ignore" and np.isfinite(aif_arr)
                    and arrival < aif_arr):
                if AIF_ARRIVAL_MODE == "clamp":
                    arrival = float(aif_arr)
                    nb = slice(0, max(N_MIN_BASE, int(arrival) - BASE_MARGIN))
                    if N_MAX_BASE is not None:
                        nb = slice(0, min(nb.stop, int(N_MAX_BASE)))
                    source += "+aif_clamped"
                    clamped = True
        except Exception as e:                 # never lose a fit to detection
            nb, source, err = NBASE_FALLBACK, "fallback:detect_failed", repr(e)
    else:
        nb, source = NBASE_FALLBACK, "fallback"

    nb = slice(0, int(min(nb.stop, n_frames)))
    if nb.stop < 2:
        raise ValueError(f"{sid}: baseline window {nb} has fewer than 2 frames")

    y = np.asarray(St, float)[nb]
    cv = float(np.std(y) / np.mean(y))
    drift_pct, drift_t = baseline_drift(St, nb)
    info = dict(
        nbase_start=int(nb.start), nbase_stop=int(nb.stop),
        n_base_frames=int(nb.stop - nb.start), nbase_source=source,
        nbase_method=(NBASE_METHOD if source.startswith("auto") else ""),
        nbase_detect_error=err,
        lesion_arrival_frame=float(arrival),
        lesion_arrival_clamped_to_aif=bool(clamped),
        lesion_arrival_before_aif_arrival=bool(
            np.isfinite(arrival) and np.isfinite(aif_arr) and arrival < aif_arr),
        baseline_cv=cv, baseline_cv_flag=bool(cv > BASE_CV_FLAG),
        baseline_drift_pct_per_frame=drift_pct, baseline_drift_t=drift_t,
        baseline_drift_flag=bool(np.isfinite(drift_t)
                                 and abs(drift_t) > DRIFT_T_FLAG),
        baseline_long_flag=bool(nb.stop > BASE_LONG_FRAC * n_frames),
    )
    return nb, info


def aif_arrival(sid):
    """AIF arrival frame from the AIF extraction summary, if available
    (extract_aif_roi.py: arrival of whichever source -- IDIF or aorta --
    produced the AIF CSV)."""
    if not os.path.exists(AIF_SUMMARY):
        return np.nan
    s = pd.read_csv(AIF_SUMMARY)
    r = s[s["subject_id"] == sid]
    if r.empty or "arrival_frame" not in r:
        return np.nan
    return float(r["arrival_frame"].iloc[0])


# =============================================================================
# posterior summaries
# =============================================================================

def derived_draws(d):
    """d: dict name -> draws. Returns dict of derived quantities.

    vi is the intracellular volume fraction SSM3 actually uses:
    water_exchange._ssm3_to_signal sets vi = 1 - vb - ve with vb = vp, so it
    is reported here rather than left implicit. As vi -> 0 there is no
    intracellular pool for tau_i to describe, hence the VI_SMALL flag.
    """
    ve, vp, fp, ps = d["ve"], d["vp"], d["fp"], d["ps"]
    return dict(vtot=ve + vp, vi=1.0 - ve - vp,
                E=ps / (ps + fp), Ktrans=ps * fp / (ps + fp),
                Tc=vp / fp, Te=ve / ps)


def _iqr(x, log=False):
    x = np.asarray(x, float)
    if log:
        x = np.log10(x[x > 0])
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def chain_disagree(chains_1d):
    """chains_1d: (n_chains, n_draws). Spread of chain medians vs mean within-IQR."""
    meds = np.median(chains_1d, axis=1)
    within = np.mean(np.percentile(chains_1d, 75, axis=1)
                     - np.percentile(chains_1d, 25, axis=1))
    spread = float(meds.max() - meds.min())
    return spread > CHAIN_SPLIT * within, spread, float(within)


def _diagnostics(chains_1d):
    """R-hat / ESS / MCSE(mean) / MCSE(median) for one (n_chains, n_draws)
    array, via ArviZ -- the same functions bayes._rhat_ess_mcse uses, called
    here so DERIVED quantities get their own diagnostics too.

    BARG step 2B/2C: "Every parameter and derived value has a distinct PSRF
    and ESS. Therefore, the convergence and resolution values of every
    parameter should be reported." For sampled parameters these agree with
    res.rhat / res.ess by construction (same arrays, same functions);
    nlls-style scripts that lack ArviZ get NaNs instead of a crash.
    """
    try:
        import arviz as az
        x = np.asarray(chains_1d, float)
        return dict(
            rhat=float(np.asarray(az.rhat(x))),
            ess=float(np.asarray(az.ess(x))),
            mcse_mean=float(np.asarray(az.mcse(x, method="mean"))),
            mcse_median=float(np.asarray(az.mcse(x, method="quantile", prob=0.5))),
        )
    except Exception:
        return dict(rhat=np.nan, ess=np.nan, mcse_mean=np.nan, mcse_median=np.nan)


def summarise(res, prior_d):
    """Everything per parameter + derived quantity, as a flat dict."""
    names = list(res.param_names)
    post = {p: res.flat()[:, names.index(p)] for p in PARAMS}
    post.update(derived_draws(post))
    # per-chain arrays for BOTH sampled and derived quantities: derived chains
    # are built from the sampled chains draw by draw, so every diagnostic
    # below sees the same chain structure the sampler produced.
    ch = {p: res.chains[:, :, names.index(p)] for p in PARAMS}
    ch.update(derived_draws(ch))

    out = {}
    for p in PARAMS + DERIVED:
        x, log = post[p], p in LOG_SCALE
        q = np.percentile(x, [2.5, 25, 50, 75, 97.5])
        out.update({f"{p}_median": q[2], f"{p}_q25": q[1], f"{p}_q75": q[3],
                    f"{p}_ci2.5": q[0], f"{p}_ci97.5": q[4], f"{p}_mean": float(np.mean(x)),
                    f"{p}_iqr_ratio": _iqr(x, log) / _iqr(prior_d[p], log)})
        out[f"{p}_prior_dominated"] = bool(out[f"{p}_iqr_ratio"] > IQR_RATIO_FLAT)

        # --- convergence / resolution, every parameter AND derived value ----
        d = _diagnostics(ch[p])
        if p in PARAMS:
            # prefer the sampler's own numbers where it has them, so these
            # columns cannot drift from res.rhat / res.ess
            d["rhat"] = float(res.rhat.get(p, d["rhat"]))
            d["ess"] = float(res.ess.get(p, d["ess"]))
            d["mcse_mean"] = float(getattr(res, "mcse_mean", {}).get(p, d["mcse_mean"]))
            d["mcse_median"] = float(
                getattr(res, "mcse_median", {}).get(p, d["mcse_median"]))
            dis, spread, within = chain_disagree(ch[p])
            out[f"{p}_chains_disagree"] = bool(dis)
        out[f"{p}_rhat"] = d["rhat"]
        out[f"{p}_ess"] = d["ess"]
        out[f"{p}_mcse_mean"] = d["mcse_mean"]
        out[f"{p}_mcse_median"] = d["mcse_median"]
        # MCSE of the reported median relative to this quantity's own 95% CrI
        # width: scale-free, so it is comparable across parameters and lesions
        width = float(q[4] - q[0])
        rel = d["mcse_median"] / width if width > 0 else np.nan
        out[f"{p}_mcse_median_rel_ci"] = float(rel)
        out[f"{p}_mcse_rel_flag"] = bool(np.isfinite(rel) and rel > MCSE_REL_MAX)
    return out, post


# =============================================================================
# NLLS summaries
# =============================================================================

def nlls_params(nres):
    """Kinetic parameter vector, whichever ROI fitter produced it.
    fit_roi                       -> RoiFitResult.params / cost
    fit_roi_with_delay_continuous -> RoiDelayedFitResult.best_params / best_cost
    """
    v = getattr(nres, "best_params", None)
    return np.asarray(v if v is not None else nres.params, float)


def nlls_cost(nres):
    v = getattr(nres, "best_cost", None)
    return float(v if v is not None else nres.cost)


def nlls_estimates(nres):
    """ROI fit result -> {name: value} for PARAMS + DERIVED.

    With WX_MODEL set the water-exchange parameters come back in wx_params;
    wx_params[-1] is tau_i whether or not tau_b was also fitted (see
    fitting_roi: `tau_i = delayed.wx_params[-1]`). tau_delay is not fitted
    here -- it is held at the value read from the source run.
    """
    kin = dict(zip(TCXM.param_names, nlls_params(nres)))
    est = {p: float(kin[p]) for p in PARAMS if p not in ("tau_delay", "tau_i")}
    if "tau_i" in PARAMS:
        wxp = getattr(nres, "wx_params", None)
        if wxp is None:
            raise ValueError("WX_MODEL is set but the NLLS result has no "
                             "wx_params -- tau_i cannot be reported")
        est["tau_i"] = float(np.asarray(wxp, float)[-1])
    est.update({k: float(v) for k, v in derived_draws(est).items()})
    return est


def at_bound(x, lo, hi, log):
    """True if x is within AT_BOUND_TOL of either bound, measured as a
    fraction of the bound range (log range for log-sampled parameters)."""
    if log:
        x, lo, hi = np.log(x), np.log(lo), np.log(hi)
    tol = AT_BOUND_TOL * (hi - lo)
    return bool(x - lo <= tol or hi - x <= tol)


def compare_nlls(est, post, summ):
    """Where each NLLS estimate falls in the corresponding posterior."""
    out = {}
    for p in PARAMS + DERIVED:
        x, med = est[p], summ[f"{p}_median"]
        out[f"nlls_{p}"] = x
        out[f"{p}_nlls_post_pctile"] = float(100.0 * np.mean(post[p] < x))
        out[f"{p}_nlls_in_ci95"] = bool(summ[f"{p}_ci2.5"] <= x <= summ[f"{p}_ci97.5"])
        out[f"{p}_nlls_minus_median"] = x - med
        # ratio only for strictly positive quantities (tau_delay spans zero)
        out[f"{p}_nlls_over_median"] = (x / med if p != "tau_delay" and med > 0
                                        else np.nan)
    return out


# =============================================================================
# figures
# =============================================================================

def plot_lesion(sig_4d, lesion, sid, outdir, time_idx=15):
    z = int(np.argmax(lesion.sum(axis=(0, 1))))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sig_4d[:, :, z, time_idx].T, cmap="gray", origin="lower")
    ax.imshow(lesion[:, :, z].T, cmap="Reds", alpha=0.5 * lesion[:, :, z].T,
              origin="lower")
    ax.set_title(f"{sid}: fitted lesion, max slice z={z}, t={time_idx}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{sid}_lesion_inspection.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_prior_vs_posterior(res, names, log_names, lb, ub, prior_samples, tag,
                            save_path, nlls_vals=None):
    """verify_03 figure (log10 bins for log-sampled params, except PLOT_LINEAR),
    no truth line. nlls_vals: optional {name: NLLS estimate}, drawn dashed.

    log_names is a set of PARAMETER NAMES to bin in log10, not kinetic-spec
    indices: with water exchange the sampled vector is
    [ve, vp, fp, ps, tau_delay, tau_i] and tau_i is log-sampled by the "flat"
    tau prior, which an index into TCXM.log_dims cannot express.
    """
    post = res.flat()
    res_names = list(res.param_names)
    n, ncol = len(res_names), 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, pname in enumerate(res_names):
        ax = axes[k // ncol][k % ncol]
        pk = post[:, k]
        if pname in names:
            j = names.index(pname)
            log = (pname in log_names) and (pname not in PLOT_LINEAR)
            lo_k, hi_k = float(lb[j]), float(ub[j])
            prior_col = prior_samples[:, j]
        else:
            log, prior_col = False, None
            lo_k, hi_k = float(pk.min()), float(pk.max())
        tr = np.log10 if log else (lambda v: v)
        bins = np.linspace(tr(lo_k), tr(hi_k), 60)
        if prior_col is not None:
            ax.hist(tr(prior_col), bins=bins, density=True, color=C_PRIOR,
                    alpha=0.9, label="prior", zorder=1)
        ax.hist(tr(pk), bins=bins, density=True, color=C_BAYES, alpha=0.75,
                label="posterior", zorder=2)
        med = float(np.median(pk))
        ax.axvline(tr(med), color=C_MEDIAN, lw=1.8, zorder=3,
                   label=f"post. median\n{med:.4g}")
        if nlls_vals is not None and pname in nlls_vals:
            v = nlls_vals[pname]
            ax.axvline(tr(v), color=C_NLLS, lw=1.6, ls="--", zorder=4,
                       label=f"NLLS\n{v:.4g}")
        mcse_med = getattr(res, "mcse_median", {}).get(pname, np.nan)
        ax.set_title(f"{pname}   R-hat={res.rhat.get(pname, np.nan):.3f}   "
                     f"ESS={res.ess.get(pname, np.nan):.0f}   "
                     f"MCSE(med)={mcse_med:.3g}", fontsize=10)
        unit = f" ({UNITS[pname]})" if pname in UNITS else ""
        ax.set_xlabel(f"log10({pname}{unit})" if log else pname + unit)
        ax.set_ylabel("density")
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.2)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{tag}   {MODEL_TAG}"
                 + ("   (dashed green = NLLS)" if nlls_vals else ""), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fit(t, St, pred, sid, diagnosis, outdir, nbase, pred_nlls=None):
    """Data vs posterior-median prediction (and NLLS prediction, if given)."""
    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, St, "o", ms=4, color="k", label="lesion mean signal")
    ax[0].plot(t, pred, "-", color=C_BAYES, lw=2,
               label=f"posterior-median prediction (RMSE {rms(St - pred):.3g})")
    if pred_nlls is not None:
        ax[0].plot(t, pred_nlls, "--", color=C_NLLS, lw=1.8,
                   label=f"NLLS fit (RMSE {rms(St - pred_nlls):.3g})")
    ax[0].axvspan(t[nbase.start], t[nbase.stop - 1], color="0.6", alpha=0.25,
                  label=f"baseline frames {nbase.start}-{nbase.stop - 1}")
    ax[0].set_ylabel("Signal (a.u.)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="0.4", lw=1)
    ax[1].plot(t, St - pred, "o-", ms=3, color=C_BAYES, label="Bayes")
    if pred_nlls is not None:
        ax[1].plot(t, St - pred_nlls, "s--", ms=3, color=C_NLLS, label="NLLS")
        ax[1].legend(fontsize=7)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("residual")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{sid} ({diagnosis})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{sid}_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bayes_flag(d, m):
    """Hollow marker: failed sampler QC, or prior-dominated for this metric."""
    return ((~d["qc_pass"].to_numpy(bool))
            | d[f"{m}_prior_dominated"].to_numpy(bool))


def _nlls_flag(d, m):
    """Hollow marker: NLLS did not converge, or the estimate (or, for a derived
    metric, any fitted parameter) sits on a bound."""
    col = f"nlls_{m}_at_bound" if f"nlls_{m}_at_bound" in d else "nlls_any_at_bound"
    return (~d["nlls_success"].to_numpy(bool)) | d[col].to_numpy(bool)


def plot_groups(df, metrics, save_path, title, col="{m}_median",
                flag_fn=_bayes_flag,
                flag_text="failed QC or prior-dominated for that metric"):
    """Mean +/- SD per group, every tumor as a point; hollow = flagged.
    col is the value column pattern ("{m}_median" Bayes, "nlls_{m}" NLLS)."""
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for k, m in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        c = col.format(m=m)
        allv = df[c].to_numpy(float)
        allv = allv[np.isfinite(allv)]
        for gx, g in enumerate(("B", "M")):
            d = df[(df["diagnosis"] == g) & np.isfinite(df[c].astype(float))]
            y = d[c].to_numpy(float)
            flag = flag_fn(d, m)
            x = gx + rng.uniform(-0.12, 0.12, y.size)
            ax.scatter(x[~flag], y[~flag], s=22, color=C_GROUP[g], alpha=0.8, zorder=2)
            ax.scatter(x[flag], y[flag], s=22, facecolors="none",
                       edgecolors=C_GROUP[g], alpha=0.8, zorder=2)
            if y.size:
                mu = y.mean()
                sd = y.std(ddof=1) if y.size > 1 else 0.0
                lo = min(sd, 0.999 * mu) if (m in PLOT_LOG and mu > 0) else sd
                ax.errorbar(gx + 0.28, mu, yerr=[[lo], [sd]], fmt="s", ms=7,
                            color="k", capsize=5, lw=1.6, zorder=3)
        if m in PLOT_LOG and allv.size and np.all(allv > 0):
            ax.set_yscale("log")
            ax.set_ylim(allv.min() / 2.0, allv.max() * 2.0)
        ax.set_xticks([0, 1])
        nB = int(((df["diagnosis"] == "B") & np.isfinite(df[c].astype(float))).sum())
        nM = int(((df["diagnosis"] == "M") & np.isfinite(df[c].astype(float))).sum())
        ax.set_xticklabels([f"B (n={nB})", f"M (n={nM})"])
        ax.set_xlim(-0.5, 1.6)
        ax.set_title(m + (f" ({UNITS[m]})" if m in UNITS else ""))
        ax.grid(alpha=0.25, which="both", axis="y")
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title + f"\nsquares = mean ± SD; hollow points = {flag_text}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_nlls_vs_bayes(df, metrics, save_path):
    """NLLS estimate vs posterior median (IQR bars), one panel per metric.
    Hollow = flagged by either method."""
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow),
                             squeeze=False)
    for k, m in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        d = df[np.isfinite(df[f"{m}_median"].astype(float))
               & np.isfinite(df[f"nlls_{m}"].astype(float))]
        if d.empty:
            ax.axis("off")
            continue
        for g in ("B", "M"):
            dg = d[d["diagnosis"] == g]
            if dg.empty:
                continue
            x = dg[f"{m}_median"].to_numpy(float)
            y = dg[f"nlls_{m}"].to_numpy(float)
            xerr = np.vstack([x - dg[f"{m}_q25"].to_numpy(float),
                              dg[f"{m}_q75"].to_numpy(float) - x])
            flag = _bayes_flag(dg, m) | _nlls_flag(dg, m)
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=C_GROUP[g],
                        alpha=0.35, lw=1, zorder=1)
            ax.scatter(x[~flag], y[~flag], s=24, color=C_GROUP[g], zorder=2,
                       label=g)
            ax.scatter(x[flag], y[flag], s=24, facecolors="none",
                       edgecolors=C_GROUP[g], zorder=2)
        v = np.concatenate([d[f"{m}_median"].to_numpy(float),
                            d[f"nlls_{m}"].to_numpy(float)])
        if m in PLOT_LOG and np.all(v > 0):
            lim = [v.min() / 1.5, v.max() * 1.5]
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            pad = 0.05 * (v.max() - v.min() or 1.0)
            lim = [v.min() - pad, v.max() + pad]
        ax.plot(lim, lim, "k--", lw=0.8, zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        n_in = int(d[f"{m}_nlls_in_ci95"].sum())
        unit = f" ({UNITS[m]})" if m in UNITS else ""
        ax.set_title(f"{m}{unit}   NLLS in 95% CI: {n_in}/{len(d)}", fontsize=10)
        ax.set_xlabel("posterior median (IQR bar)")
        ax.set_ylabel("NLLS")
        ax.grid(alpha=0.25, which="both")
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("NLLS vs Bayesian posterior median, per lesion\n"
                 "dashed = identity; hollow = failed Bayes QC, prior-dominated, "
                 "NLLS not converged or at a bound", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# one subject
# =============================================================================

def fit_one(sid, diagnosis, outdir):
    mask_dir = mask_dir_for(sid)           # masks: always MASK_ROOT
    dce_path = find_dce(sid)               # images: whichever RECON is selected
    sig_4d, _ = load_nifti_canonical(dce_path)
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    t = np.arange(sig_4d.shape[-1]) * dt

    lesion, linfo = select_largest_lesion(mask_dir, sid, dce_path)
    # same-shape/different-grid is the one mismatch that would otherwise index
    # silently and give a lesion curve from the wrong voxels
    if lesion.shape != sig_4d.shape[:3]:
        raise ValueError(f"{sid}: lesion mask {lesion.shape} does not match the "
                         f"{RECON} volume {sig_4d.shape[:3]} "
                         f"({os.path.basename(dce_path)})")
    aif, aif_path = load_aif_csv(sid, t)
    subject = SubjectConfig(subject_id=sid, T10=T10_TISSUE, B1=B1)
    arr = aif_arrival(sid)

    # --- baseline window ------------------------------------------------------
    # St is nbase-independent (average_space="signal"), so it is computed with
    # the fallback acq purely to detect the window; every nbase-dependent
    # quantity below uses acq_s, and Ct is recomputed from it.
    _, n_vox, St = _aggregate_roi_signal(sig_4d, lesion, acq, subject, roi_cfg)
    nb, binfo = choose_nbase(sid, St, arr)
    acq_s = replace(acq, nbase=nb)
    Ct, _, St_chk = _aggregate_roi_signal(sig_4d, lesion, acq_s, subject, roi_cfg)
    if not np.allclose(St_chk, St):
        raise ValueError(f"{sid}: lesion mean signal changed with nbase -- "
                         f"roi_cfg.average_space must be 'signal'")
    plot_lesion(sig_4d, lesion, sid, outdir)

    base = St[nb]
    base_cv = binfo["baseline_cv"]
    peak_enh = float(St.max() / np.mean(base))

    # --- fixed delay ----------------------------------------------------------
    # The Bayesian fit gets delay_mode="fixed"/delay_fixed and shifts the AIF
    # itself; aif_shift here is the SAME computation (signal_models.shift_aif
    # with the same acq), built once so the NLLS arm and the posterior-median
    # prediction below use exactly the AIF the sampler used.
    tau_fix, tinfo = tau_for(sid)
    aif_shift = shift_aif(aif, float(tau_fix), t, acq_s)
    aif_pad = float(np.mean(np.asarray(aif)[nb]))
    tinfo.update(aif_pad_mM=aif_pad,
                 aif_pad_flag=bool(abs(aif_pad) > AIF_PAD_FLAG))

    # --- NLLS: 06_roi_fitting.ipynb cell [13] pipeline ------------------------
    # Isolated: an NLLS failure is recorded but never costs the Bayesian fit.
    nres, est, nlls_row = None, None, {}
    if RUN_NLLS:
        try:
            t1 = time.time()
            # fixed delay: no tau search. fit_roi on the PRE-SHIFTED AIF is
            # the same computation as bayes.py's own NLLS anchor for
            # delay_mode="fixed" (fit_single_voxel on shift_aif(...)), so
            # nlls_vs_bayes_anchor_maxdiff stays meaningful.
            nres = fit_roi(
                sig_4d, lesion, aif_shift, t, TCXM, acq_s, subject, cfg_fit,
                roi_cfg, seed=NLLS_SEED,
                wx_model=WX_MODEL,
                aif_cfg=(aif_cfg if WX_MODEL else None),
                relax_cfg=(relax_cfg if WX_MODEL else None))
            nlls_s = time.time() - t1
            if not np.allclose(nres.St_mean, St):
                raise ValueError("NLLS aggregated a different lesion curve than "
                                 "the Bayesian fit")
            est = nlls_estimates(nres)
        except Exception as e:
            print(f"  !! NLLS FAILED: {e!r}")
            nres, est, nlls_row = None, None, dict(nlls_error=repr(e))

    # --- fit -----------------------------------------------------------------
    cfg_b = BayesConfig(prior_choices=PRIOR_CHOICES, n_draws=N_DRAWS,
                        n_tune=N_TUNE, n_chains=N_CHAINS,
                        noise="gauss_marginalized_sigma", seed=SEED,
                        backend=BACKEND, step="nuts", dense_mass_matrix=DENSE_MASS,
                        delay_mode=DELAY_MODE, delay_fixed=float(tau_fix),
                        delay_bounds=DELAY_BOUNDS)
    t0 = time.time()
    res = fit_curve_bayes(St, aif, t, TCXM, acq_s, subject, cfg_fit, cfg_b,
                          wx_model=WX_MODEL,
                          aif_cfg=(aif_cfg if WX_MODEL else None),
                          relax_cfg=(relax_cfg if WX_MODEL else None))
    fit_s = time.time() - t0

    # --- prior (same object the sampler used) ---------------------------------
    layout = build_param_layout(TCXM, cfg_b, aif=aif, time_var=t, acq=acq_s,
                                wx_model=WX_MODEL,
                                relax_cfg=(relax_cfg if WX_MODEL else None))
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, (relax_cfg if WX_MODEL else None),
                           wx_model=WX_MODEL, delay_mode=DELAY_MODE,
                           n_noise=noise.n_params, noise_names=noise.param_names,
                           choices=PRIOR_CHOICES, delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(PRIOR_SEED),
                                 N_PRIOR_SAMPLES)[:, :layout.n_model]
    # layout.param_names covers the SAMPLED vector in order, including tau_i;
    # layout.model.param_names stops at the delay-augmented kinetic block, so
    # it would silently drop tau_i. Noise params (none here, sigma is
    # marginalised) sit after n_model and are excluded by the slice.
    names_fit = list(layout.param_names[:layout.n_model])
    lb_fit = np.asarray(layout.lb, float)[:layout.n_model]
    ub_fit = np.asarray(layout.ub, float)[:layout.n_model]
    prior_d = {p: prior_samples[:, names_fit.index(p)] for p in PARAMS}
    prior_d.update(derived_draws(prior_d))

    plot_prior_vs_posterior(res, names_fit, LOG_SCALE, lb_fit, ub_fit,
                            prior_samples, f"{sid} ({diagnosis})",
                            os.path.join(outdir, f"{sid}_prior_post.png"),
                            nlls_vals=est)

    summ, post = summarise(res, prior_d)

    # --- posterior-median prediction + residual checks ------------------------
    theta_med = np.array([summ[f"{p}_median"] for p in names_fit])
    S0 = float(compute_S0(St[None, :], acq_s)[0])
    # aif_shift, not aif: with delay_mode="fixed" the delay lives in the AIF,
    # not in theta_med, so simulating from the unshifted AIF would silently
    # drop it and inflate the residuals
    pred = simulate_curve(theta_med, aif_shift, t, layout.model, acq_s, subject,
                          cfg_fit, wx_model=WX_MODEL,
                          aif_cfg=(aif_cfg if WX_MODEL else None),
                          relax_cfg=(relax_cfg if WX_MODEL else None), S0=S0)
    resid = St - pred
    post_resid = resid[nb.stop:]
    lag1 = float(np.corrcoef(post_resid[:-1], post_resid[1:])[0, 1])
    base_sd = float(np.std(base, ddof=1))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    plot_fit(t, St, pred, sid, diagnosis, outdir, nb,
             pred_nlls=(nres.St_pred if nres is not None else None))

    # --- NLLS vs posterior ----------------------------------------------------
    if nres is not None:
        lb_m, ub_m = lb_fit, ub_fit
        at = {p: at_bound(est[p], lb_m[j], ub_m[j], p in LOG_SCALE)
              for j, p in enumerate(names_fit)}
        rmse_nlls = float(np.sqrt(np.mean((St - nres.St_pred) ** 2)))
        nlls_row = dict(nlls_success=bool(nres.success),
                        nlls_cost=nlls_cost(nres), nlls_rmse=rmse_nlls,
                        nlls_rmse_over_baseline_sd=(rmse_nlls / base_sd
                                                    if base_sd > 0 else np.nan),
                        nlls_interp="fixed_delay_preshift", nlls_seconds=nlls_s,
                        nlls_any_at_bound=any(at.values()), nlls_error="")
        nlls_row.update({f"nlls_{p}_at_bound": v for p, v in at.items()})
        nlls_row.update(compare_nlls(est, post, summ))
        # same computation as the sampler's NLLS anchor when NLLS_INTERP is
        # "linear" -> expect ~0; reported as a fraction of each bound range
        if res.point_estimate is not None:
            pe = np.asarray(res.point_estimate, float)[:len(names_fit)]
            th = np.array([est[p] for p in names_fit])
            nlls_row["nlls_vs_bayes_anchor_maxdiff"] = float(
                np.max(np.abs(th - pe) / (ub_m - lb_m)))

    np.savez_compressed(os.path.join(outdir, f"{sid}_draws.npz"),
                        chains=res.chains,
                        param_names=np.array(res.param_names, dtype=object),
                        St=St, aif=aif, aif_shift=aif_shift, t=t,
                        tau_delay_fixed=float(tau_fix), lesion_nvox=n_vox)

    rhat_max = max(summ[f"{p}_rhat"] for p in PARAMS)
    ess_min = min(summ[f"{p}_ess"] for p in PARAMS)
    # worst MCSE(median)/CrI-width over the SAMPLED parameters; derived
    # quantities have their own columns but do not gate qc_pass
    mcse_rel = [summ[f"{p}_mcse_median_rel_ci"] for p in PARAMS]
    mcse_rel_max = float(np.nanmax(mcse_rel)) if len(mcse_rel) else np.nan
    n_div = int(res.n_divergences) if res.n_divergences is not None else -1
    any_disagree = any(summ[f"{p}_chains_disagree"] for p in PARAMS)
    mcse_ok = not (np.isfinite(mcse_rel_max) and mcse_rel_max > MCSE_REL_MAX)
    qc_pass = bool(rhat_max <= RHAT_MAX and ess_min >= ESS_MIN
                   and n_div == 0 and not any_disagree and mcse_ok)

    row = dict(subject_id=sid, diagnosis=diagnosis, recon=RECON,
               dce_file=os.path.basename(dce_path),
               dce_dir=os.path.dirname(dce_path), **tinfo,
               aif_file=os.path.basename(aif_path), **linfo,
               peak_enhancement=peak_enh,
               aif_arrival_frame=arr,
               baseline_overlaps_aif_arrival=bool(np.isfinite(arr) and arr < nb.stop),
               **binfo,
               S0=S0, rmse=rmse, baseline_sd=base_sd,
               rmse_over_baseline_sd=rmse / base_sd if base_sd > 0 else np.nan,
               resid_lag1_autocorr=lag1,
               rhat_max=rhat_max, ess_min=ess_min,
               mcse_median_rel_ci_max=mcse_rel_max, mcse_rel_flag=not mcse_ok,
               n_divergences=n_div,
               any_chains_disagree=any_disagree, qc_pass=qc_pass,
               ps_flow_limited=bool(summ["E_median"] > E_FLOW_LIMITED),
               # provenance: which model actually produced this row
               model=MODEL_TAG, delay_mode=DELAY_MODE,
               wx_model=(WX_MODEL.name if WX_MODEL else ""),
               wx_fit_tau_b=(bool(relax_cfg.fit_tau_b) if WX_MODEL else False),
               wx_tau_b_fixed=(float(relax_cfg.tau_b_fixed) if WX_MODEL else np.nan),
               wx_tau_i_lb=(float(relax_cfg.tau_i_lb) if WX_MODEL else np.nan),
               wx_tau_i_ub=(float(relax_cfg.tau_i_ub) if WX_MODEL else np.nan),
               wx_blood_T10=(float(aif_cfg.T10) if WX_MODEL else np.nan),
               wx_hematocrit=(float(aif_cfg.hematocrit) if WX_MODEL else np.nan),
               # vi = 1 - ve - vp is the pool tau_i describes; near 0 there is
               # nothing left for tau_i to be identified from
               vi_small_flag=bool(WX_MODEL and summ["vi_median"] < VI_SMALL),
               fit_seconds=fit_s, error="")
    row.update(summ)
    row.update(nlls_row)
    return row


# =============================================================================
# main
# =============================================================================

def main():
    outdir = RESUME_DIR or os.path.join(
        RESULTS_ROOT, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    rowdir = os.path.join(outdir, "rows")
    os.makedirs(rowdir, exist_ok=True)
    print(f"results -> {outdir}")
    print(f"model: {MODEL_TAG}")
    print(f"model: {MODEL_TAG}   sampled: {PARAMS}")
    print(f"recon: {RECON}   masks: {MASK_ROOT}")
    tt = load_tau_table()
    print(f"tau_delay FIXED from {tt['path']}")
    print(f"  column {TAU_FIXED_COL!r}: {tt['n_good']}/{tt['n']} subjects usable, "
          f"cohort median {tt['cohort_median']:.3f} s; "
          f"missing -> {TAU_MISSING!r}, require source QC = {TAU_REQUIRE_SRC_QC}")
    print(f"baseline: FIXED {NBASE_FALLBACK} (AUTO_NBASE={AUTO_NBASE})")
    print(f"TCXM bounds at run time: lb={TCXM.lb}, ub={TCXM.ub}")
    if WX_MODEL:
        print(f"water exchange: {WX_MODEL.name} "
              f"(n_pools={WX_MODEL.n_pools}, params={WX_MODEL.param_names}); "
              f"fit_tau_b={relax_cfg.fit_tau_b}, "
              f"tau_b fixed at {relax_cfg.tau_b_fixed} s, "
              f"tau_i in [{relax_cfg.tau_i_lb}, {relax_cfg.tau_i_ub}] s, "
              f"prior tau='{PRIOR_CHOICES.get('tau', 'flat')}'")
        print(f"                blood T10={aif_cfg.T10} ms, "
              f"hematocrit={aif_cfg.hematocrit} "
              f"(MUST match extract_aif_roi.py)")
    print(f"QC: R-hat<={RHAT_MAX}, ESS>={ESS_MIN}, 0 divergences, "
          f"chains agree, MCSE(median)/CrI<={MCSE_REL_MAX}")
    print(f"baseline window: "
          + (f"AUTO ({NBASE_METHOD}, n_min={N_MIN_BASE}, "
             f"n_confirm={N_CONFIRM}, margin={BASE_MARGIN}, "
             f"cap={N_MAX_BASE}, aif_arrival={AIF_ARRIVAL_MODE})"
             if AUTO_NBASE else f"FIXED {NBASE_FALLBACK}")
          + (f", {len(NBASE_OVERRIDE)} override(s)" if NBASE_OVERRIDE else ""))

    tumors = pd.read_csv(TUMOR_CSV)
    tumors.columns = [c.strip() for c in tumors.columns]
    tumors["Patient_ID"] = tumors["Patient_ID"].str.strip()
    tumors["Diagnosis"] = tumors["Diagnosis"].str.strip().str.upper()
    if ONLY:
        tumors = tumors[tumors["Patient_ID"].isin(ONLY)]
    print(f"{len(tumors)} subjects: {tumors['Diagnosis'].value_counts().to_dict()}")

    for _, r in tumors.iterrows():
        sid, dx = r["Patient_ID"], r["Diagnosis"]
        row_path = os.path.join(rowdir, f"{sid}.csv")
        if os.path.exists(row_path):
            print(f"{sid}: already done, skipping")
            continue
        print(f'\n{"=" * 80}\n{sid} ({dx})\n{"=" * 80}')
        t_subj = time.time()
        try:
            row = fit_one(sid, dx, outdir)
            print(f"  lesion {row['lesion_file']}: {row['n_vox']} voxels "
                  f"(largest of {row['n_lesion_files']} mask file(s))")
            print(f"  DCE {row['dce_file']}  [{row['recon']}]")
            print(f"  tau_delay FIXED at {row['tau_delay_fixed']:.3f} s "
                  f"(from {row['tau_delay_src']}"
                  + (f", source 95% CrI width {row['tau_delay_src_ci_width']:.2f} s"
                     if np.isfinite(row.get('tau_delay_src_ci_width', np.nan)) else "")
                  + ")"
                  + ("  [source failed QC]" if not row.get('tau_delay_src_qc_pass', True) else "")
                  + ("  [source tau prior-dominated]" if row.get('tau_delay_src_prior_dominated') else "")
                  + (f"  [AIF pad {row['aif_pad_mM']:.3f} mM]" if row.get('aif_pad_flag') else ""))
            print(f"  baseline frames 0-{row['nbase_stop'] - 1} "
                  f"({row['n_base_frames']} frames, {row['nbase_source']})"
                  f"  lesion arrival {row['lesion_arrival_frame']:.0f}"
                  f"  AIF arrival {row['aif_arrival_frame']:.0f}"
                  f"  CV {row['baseline_cv']:.4f}"
                  f"  drift {row['baseline_drift_pct_per_frame']:+.3f}%/frame"
                  f" (t={row['baseline_drift_t']:+.1f})"
                  + "".join(f"  [{k}]" for k in
                            ("baseline_cv_flag", "baseline_drift_flag",
                             "baseline_long_flag", "baseline_overlaps_aif_arrival",
                             "lesion_arrival_before_aif_arrival")
                            if row.get(k)))
            print(f"  R-hat max {row['rhat_max']:.3f}  ESS min {row['ess_min']:.0f}  "
                  f"MCSE(med)/CrI max {row['mcse_median_rel_ci_max']:.3f}"
                  f"{' [FLAG]' if row['mcse_rel_flag'] else ''}  "
                  f"div {row['n_divergences']}  QC {'PASS' if row['qc_pass'] else 'FAIL'}"
                  f"  ({row['fit_seconds']:.0f}s)")
            print("  MCSE(median): " + "  ".join(
                f"{p}={row[f'{p}_mcse_median']:.3g}"
                f" ({row[f'{p}_mcse_median_rel_ci'] * 100:.1f}% CrI)"
                for p in PARAMS))
            if WX_MODEL:
                print(f"  vi median {row['vi_median']:.3f}"
                      f"{'  [vi_small: tau_i weakly identifiable]' if row['vi_small_flag'] else ''}")
            print("  " + "  ".join(f"{p}={row[f'{p}_median']:.4g}"
                                   f"{'*' if row[f'{p}_prior_dominated'] else ''}"
                                   for p in PARAMS + ["E", "Ktrans"])
                  + "   (* = prior-dominated)")
            if "nlls_ve" in row:
                n_in = sum(bool(row[f"{p}_nlls_in_ci95"]) for p in PARAMS)
                print(f"  NLLS {'converged' if row['nlls_success'] else 'NOT CONVERGED'}"
                      f"  RMSE {row['nlls_rmse']:.4g} (Bayes {row['rmse']:.4g})"
                      f"  inside 95% CI {n_in}/{len(PARAMS)}"
                      f"  anchor diff {row.get('nlls_vs_bayes_anchor_maxdiff', np.nan):.1e}")
                print("  NLLS " + "  ".join(
                    f"{p}={row[f'nlls_{p}']:.4g}"
                    f"{'!' if row.get(f'nlls_{p}_at_bound') else ''}"
                    for p in PARAMS + ["E", "Ktrans"]) + "   (! = at bound)")
        except Exception as e:           # one bad subject must not stop the batch
            print(f"  !! FAILED: {e!r}")
            row = dict(subject_id=sid, diagnosis=dx, recon=RECON,
                       model=MODEL_TAG, delay_mode=DELAY_MODE, error=repr(e))
        subj_s = time.time() - t_subj
        row["total_seconds"] = subj_s
        print(f"  total time for {sid}: {subj_s / 60:.1f} min ({subj_s:.0f} s)")
        pd.DataFrame([row]).to_csv(row_path, index=False)

    # --- cohort table ---------------------------------------------------------
    rows = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(rowdir, "*.csv")))]
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(outdir, "lesion_bayes_results.csv"), index=False)
    ok = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in [c for c in ok.columns if c.endswith(("_prior_dominated", "qc_pass"))]:
        ok[c] = ok[c].astype(bool)
    # NLLS flags: NaN (NLLS failed, or a resumed row from a pre-NLLS run) must
    # read as False, not astype(bool)'s True
    for c in [c for c in ok.columns
              if c.endswith(("_at_bound", "_in_ci95")) or c == "nlls_success"]:
        ok[c] = ok[c].map(lambda v: str(v).strip().lower() == "true")
    has_nlls = "nlls_ve" in ok.columns
    nl = (ok[np.isfinite(ok["nlls_ve"].astype(float))] if has_nlls
          else ok.iloc[0:0])
    if "n_base_frames" in ok.columns and len(ok):
        nbf = ok["n_base_frames"].astype(float)
        print(f"\nbaseline windows: {nbf.min():.0f}-{nbf.max():.0f} frames "
              f"(median {nbf.median():.0f}); "
              f"sources {ok['nbase_source'].value_counts().to_dict()}")
        for c, lbl in (("baseline_cv_flag", "noisy baseline"),
                       ("baseline_drift_flag", "drifting baseline"),
                       ("baseline_long_flag", "long baseline"),
                       ("lesion_arrival_before_aif_arrival",
                        "lesion arrival before AIF arrival")):
            if c in ok.columns:
                n = int(ok[c].map(lambda v: str(v).strip().lower() == "true").sum())
                if n:
                    print(f"  {n} lesion(s) flagged: {lbl}")

    print(f"\n{len(ok)}/{len(df)} subjects fitted, "
          f"{int(ok['qc_pass'].sum()) if len(ok) else 0} pass QC")
    if len(ok) and "mcse_median_rel_ci_max" in ok.columns:
        # BARG step 2C: report the ACHIEVED resolution, not just the threshold
        e = ok["ess_min"].astype(float)
        m = ok["mcse_median_rel_ci_max"].astype(float)
        print(f"achieved ESS (min over parameters): min {e.min():.0f}, "
              f"median {e.median():.0f}, max {e.max():.0f}")
        print(f"achieved MCSE(median)/CrI (max over parameters): "
              f"median {m.median():.4f}, worst {m.max():.4f} "
              f"({int((m > MCSE_REL_MAX).sum())} lesion(s) over "
              f"{MCSE_REL_MAX})")
        worst = (ok[[f"{p}_ess" for p in PARAMS]].astype(float)
                 .idxmin(axis=1).value_counts().to_dict())
        print(f"  parameter setting ESS min: {worst}")
        if "vi_small_flag" in ok.columns:
            n = int(ok["vi_small_flag"].map(
                lambda v: str(v).strip().lower() == "true").sum())
            if n:
                print(f"  {n} lesion(s) flagged: vi < {VI_SMALL} "
                      f"(tau_i weakly identifiable)")
    if has_nlls:
        print(f"NLLS: {len(nl)} fitted, {int(nl['nlls_success'].sum())} converged, "
              f"{int(nl['nlls_any_at_bound'].sum())} with a parameter at a bound")

    if len(ok):
        plot_groups(ok, GROUP_METRICS, os.path.join(outdir, "group_BvsM_all.png"),
                    "All fitted lesions")
        qc = ok[ok["qc_pass"]]
        if len(qc):
            plot_groups(qc, GROUP_METRICS, os.path.join(outdir, "group_BvsM_qcpass.png"),
                        "QC-passing fits only")
        if len(nl):
            plot_groups(nl, GROUP_METRICS, os.path.join(outdir, "group_BvsM_nlls.png"),
                        f"NLLS (interp={NLLS_INTERP})", col="nlls_{m}",
                        flag_fn=_nlls_flag,
                        flag_text="NLLS not converged or at a bound")
            plot_nlls_vs_bayes(nl, GROUP_METRICS,
                               os.path.join(outdir, "nlls_vs_bayes.png"))

        nlls_clean = (nl["nlls_success"] & ~nl["nlls_any_at_bound"]) if len(nl) \
            else pd.Series(dtype=bool)
        subsets = [("bayes", "{m}_median", "all", ok),
                   ("bayes", "{m}_median", "qc_pass", ok[ok["qc_pass"]]),
                   ("nlls", "nlls_{m}", "all", nl),
                   ("nlls", "nlls_{m}", "converged_not_at_bound", nl[nlls_clean])]
        summ = []
        for m in GROUP_METRICS:
            for method, col, subset, d in subsets:
                if col.format(m=m) not in d.columns:
                    continue
                for g in ("B", "M"):
                    y = d.loc[d["diagnosis"] == g, col.format(m=m)].dropna()
                    summ.append(dict(metric=m, method=method, group=g,
                                     subset=subset, n=len(y), mean=y.mean(),
                                     sd=y.std(ddof=1), median=y.median()))
        pd.DataFrame(summ).to_csv(os.path.join(outdir, "group_summary.csv"), index=False)

    print(f"\nresults written to: {outdir}")


if __name__ == "__main__":
    main()