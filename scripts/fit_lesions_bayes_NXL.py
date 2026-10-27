

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
fit_lesions_bayes_NXL.py
========================
Bayesian TCXM + NXL (no-exchange limit) + joint-delay fit of the LARGEST
lesion in each subject listed in tumor_identification.csv, using the AIF
CSVs produced by extract_aifs.py.

Derived from fit_lesions_bayes.py (the FXL run) with exactly three changes:

  1. wx_model = NXL  (thesis Ch. 3 Eqs 3.36-3.37) is passed to
     fit_curve_bayes / build_param_layout / assemble_prior / simulate_curve,
     together with aif_cfg (NXL needs blood T10 for the plasma pool).
     NXL has NO free water-exchange parameters, so the sampled vector is
     still exactly (ve, vp, fp, ps, tau_delay) -- same PARAMS list, same
     prior blocks, same QC metrics as the FXL run. relax_cfg is NOT needed
     (that is SSM3/CNXL only).
  2. The NLLS comparison is REMOVED: no fit_roi_with_delay_continuous call,
     no nlls_* columns, no group_BvsM_nlls.png / nlls_vs_bayes.png.
     NOTE: fit_curve_bayes still runs ONE internal NLLS anchor per lesion
     (cfg_bayes.init_from_nlls=True) to initialise the chains. That is not
     the comparison fit; it cannot be removed without changing the sampler
     init, so the FXL and NXL runs stay initialised the same way. It is
     reported as point_* columns only.
  3. sigma is RECONSTRUCTED post hoc (new sigma_* columns). The
     marginalised-sigma likelihood is a Gaussian with p(sigma) ~ 1/sigma
     integrated out, so it implies the exact conditional

         sigma^2 | theta, y  ~  Inv-Gamma(n/2, SSE(theta)/2)

     Drawing one sigma per posterior draw of theta gives the sigma
     posterior the fit already implies -- no refit, and identical to what
     noise='gauss_free_jeffreys' would have sampled (same model: the
     marginalised form is its analytic integral). This is what the
     FXL/NXL gap calibration needs sigma for.

Everything else is unchanged from fit_lesions_bayes.py: the same
reconstruction switch, the same lesion selection, the same acquisition /
tissue settings, the same sampler settings, the same QC thresholds and the
same per-subject and cohort outputs.

IMPORTANT -- aif_cfg must match the AIF extraction
    aif_cfg.hematocrit has to be the value extract_aif_roi.py used to
    convert whole-blood to plasma, or the AIF and the forward model
    disagree. aif_cfg.T10 is the BLOOD T1 and is genuinely distinct from
    subject.T10 (tissue); NXL uses it for the plasma pool's baseline R1.
    Both are copied from fit_lesions_bayes_WX.py here -- CHECK them.

Model / sampler: as the FXL run
    TCXM + NXL, delay_mode='joint', PRIOR_CHOICES as set below,
    gauss_marginalized_sigma, BlackJAX NUTS, dense mass matrix.

!! CONVERGENCE WARNING -- READ BEFORE A FULL COHORT RUN !!
    On synthetic data, NXL + delay_mode='joint' converged far worse than
    FXL + delay_mode='joint' at the same settings:

        FXL data + FXL fit + joint delay   R-hat 1.001, ESS 3447
        NXL data + NXL fit + joint delay   R-hat 1.182, ESS   16
        NXL data + NXL fit + FIXED delay   R-hat 1.004, ESS 1221

    with fp's 95% CrI 3.3x its own value in the joint-delay NXL run (0.09x
    in the FXL run) -- an fp/tau_delay ridge, not a coding error. So run
    ONE subject first (ONLY = ["FMBxxx"]) and look at rhat_max / ess_min
    before committing to the cohort. If it does not converge, the options
    are: raise N_TUNE, switch PRIOR_CHOICES["rates"] to "loguniform" (log
    sampling of fp/ps), or fall back to DELAY_MODE='fixed' with the delay
    taken from the FXL run (DELAY_FROM_FXL_CSV below does that).

Per subject:
  * {sid}_lesion_inspection.png      lesion slice with the fitted component
  * {sid}_prior_post.png             prior vs posterior, every parameter
  * {sid}_fit.png                    data vs posterior-median prediction
                                     + residuals
  * {sid}_draws.npz                  all chains (+ the reconstructed sigma)
  * rows/{sid}.csv                   one-row result (lets a crashed run resume)
Cohort:
  * lesion_bayes_results_nxl.csv     every estimate + QC metric, one row / subject
  * group_BvsM_all.png               mean +/- SD per metric, B vs M, all fits
  * group_BvsM_qcpass.png            same, fits passing QC only
  * group_summary.csv                n / mean / SD / median per metric, group

Units: fp, ps, Ktrans in s^-1 (the model's units); tau_delay, Tc, Te in s.

Run:
    PYTHONPATH=src python scripts/fit_lesions_bayes_NXL.py
"""

import glob
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            RoiFittingConfig, AifConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL
from dce_mri.signal_models import compute_S0, shift_aif
from dce_mri.fitting import build_residual, build_delay_augmented_spec
from dce_mri.fitting_roi import _aggregate_roi_signal
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
AIF_DIR     = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"
AIF_SUMMARY = os.path.join(AIF_DIR, "AIF_extraction_summary.csv")   # optional
RESULTS_ROOT = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results"

# --- which reconstruction supplies the DCE volume ---------------------------
RECON      = "grasp_8spf"        # "grasp_8spf" | "tight_masks_2"
GRASP_ROOT = "/mnt/labspace/users/allison/fastMRI_data/GRASP_NIfTI"
SCAN_INDEX = 2

ONLY = []            # e.g. ["FMB056"] -- DO THIS FIRST, see the warning above
RESUME_DIR = None    # path of an earlier NXL run dir to resume (skips done subjects)

LESION_EXCLUDE = ("orta", "idif")    # aorta ROI, FireVoxel IDIF ROI

# --- acquisition / tissue: 06_roi_fitting.ipynb cell [4] --------------------
NBASE = slice(0, 7)
acq     = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)
T10_TISSUE, B1 = 1530.0, 1.0
cfg_fit = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
roi_cfg = RoiFittingConfig(average_space="signal")
SPOKES_PER_FRAME, NSPOKE, SCANTIME = 8, 288, 150

# --- water exchange: NXL ----------------------------------------------------
WX_MODEL = NXL
# BLOOD T10 (ms) and the hematocrit used by extract_aif_roi.py. Copied from
# fit_lesions_bayes_WX.py -- confirm both before a cohort run.
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)
# NXL has n_params = 0, so no RelaxationConfig is needed (SSM3/CNXL only).
relax_cfg = None

# --- Bayesian model: same as the FXL run ------------------------------------
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_MODE    = "fixed"       # "joint" | "fixed"  (see the warning above)
DELAY_BOUNDS  = (0, 25.0)
# Only used when DELAY_MODE == "fixed": take each lesion's delay from the
# FXL run's cohort CSV (column tau_delay_median) instead of sampling it.
# None -> DELAY_FIXED_DEFAULT for every lesion.
DELAY_FROM_FXL_CSV  = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/run_grasp_8spf_2026100_beta&logflat/lesion_bayes_results.csv"
DELAY_FIXED_DEFAULT = 0.0
N_DRAWS, N_TUNE, N_CHAINS = 6000, 6000, 20
BACKEND, DENSE_MASS, SEED = "blackjax", True, 1
N_PRIOR_SAMPLES, PRIOR_SEED = 20000, 999
SIGMA_SEED = 7                # for the post-hoc sigma reconstruction

# --- QC thresholds ------------------------------------------------------------
RHAT_MAX       = 1.01
ESS_MIN        = 400
IQR_RATIO_FLAT = 0.8
E_FLOW_LIMITED = 0.4
CHAIN_SPLIT    = 0.5

PARAMS  = ["ve", "vp", "fp", "ps"] + (["tau_delay"] if DELAY_MODE == "joint" else [])
DERIVED = ["vtot", "E", "Ktrans", "Tc", "Te"]
LOG_SCALE = {"fp", "ps", "Ktrans", "Tc", "Te"}
PLOT_LINEAR = {"fp", "ps"}
PLOT_LOG    = LOG_SCALE - PLOT_LINEAR
UNITS = {"fp": "s$^{-1}$", "ps": "s$^{-1}$", "Ktrans": "s$^{-1}$",
         "tau_delay": "s", "Tc": "s", "Te": "s", "sigma": "a.u."}
GROUP_METRICS = (["ve", "vp", "fp", "ps"]
                 + (["tau_delay"] if DELAY_MODE == "joint" else [])
                 + ["vtot", "E", "Ktrans"])

C_BAYES, C_MEDIAN, C_PRIOR = "#B22234", "#E67E22", "0.75"
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
    """The 4-D DCE volume for the selected recon."""
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


def select_largest_lesion(subj_dir, sid, dce_path):
    """
    Candidates = every .nii / .nii.gz in the subject folder except the vascular
    ROIs (any LESION_EXCLUDE fragment in the name) and any DCE volume. The file
    with the most nonzero voxels is used, whole, as the lesion mask.
    """
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


def aif_arrival(sid):
    """AIF arrival frame from the AIF extraction summary, if available."""
    if not os.path.exists(AIF_SUMMARY):
        return np.nan
    s = pd.read_csv(AIF_SUMMARY)
    r = s[s["subject_id"] == sid]
    if r.empty or "arrival_frame" not in r:
        return np.nan
    return float(r["arrival_frame"].iloc[0])


_FXL_DELAY_CACHE = {}


def fixed_delay_for(sid):
    """
    DELAY_MODE == 'fixed': this lesion's delay. Taken from the FXL run's
    cohort CSV when DELAY_FROM_FXL_CSV is set, so the NXL fit sees the same
    AIF shift the FXL fit inferred; otherwise DELAY_FIXED_DEFAULT.
    """
    if DELAY_FROM_FXL_CSV is None:
        return float(DELAY_FIXED_DEFAULT)
    if not _FXL_DELAY_CACHE:
        d = pd.read_csv(DELAY_FROM_FXL_CSV)
        if "tau_delay_median" not in d.columns:
            raise ValueError(f"{DELAY_FROM_FXL_CSV} has no tau_delay_median column")
        _FXL_DELAY_CACHE.update(
            dict(zip(d["subject_id"].astype(str).str.strip(),
                     d["tau_delay_median"].astype(float))))
    if sid not in _FXL_DELAY_CACHE:
        raise KeyError(f"{sid}: no tau_delay_median in {DELAY_FROM_FXL_CSV}")
    v = _FXL_DELAY_CACHE[sid]
    if not np.isfinite(v):
        raise ValueError(f"{sid}: FXL tau_delay_median is not finite ({v})")
    return float(v)


# =============================================================================
# posterior summaries
# =============================================================================

def derived_draws(d):
    """d: dict name -> draws. Returns dict of derived quantities."""
    ve, vp, fp, ps = d["ve"], d["vp"], d["fp"], d["ps"]
    return dict(vtot=ve + vp, E=ps / (ps + fp), Ktrans=ps * fp / (ps + fp),
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


def summarise(res, prior_d):
    """Everything per parameter + derived quantity, as a flat dict."""
    names = list(res.param_names)
    post = {p: res.flat()[:, names.index(p)] for p in PARAMS}
    post.update(derived_draws(post))
    ch = {p: res.chains[:, :, names.index(p)] for p in PARAMS}

    out = {}
    for p in PARAMS + DERIVED:
        x, log = post[p], p in LOG_SCALE
        q = np.percentile(x, [2.5, 25, 50, 75, 97.5])
        out.update({f"{p}_median": q[2], f"{p}_q25": q[1], f"{p}_q75": q[3],
                    f"{p}_ci2.5": q[0], f"{p}_ci97.5": q[4],
                    f"{p}_mean": float(np.mean(x)),
                    f"{p}_iqr_ratio": _iqr(x, log) / _iqr(prior_d[p], log)})
        out[f"{p}_prior_dominated"] = bool(out[f"{p}_iqr_ratio"] > IQR_RATIO_FLAT)
        if p in PARAMS:
            out[f"{p}_rhat"] = float(res.rhat.get(p, np.nan))
            out[f"{p}_ess"]  = float(res.ess.get(p, np.nan))
            dis, spread, within = chain_disagree(ch[p])
            out[f"{p}_chains_disagree"] = bool(dis)
    return out, post


def sigma_draws(res, St, aif_use, t, subject, layout):
    """
    sigma recovered from a gauss_marginalized_sigma run.

    That likelihood is  N(y | f(theta), sigma^2) x p(sigma), p(sigma) ~ 1/sigma,
    with sigma integrated out analytically. The same algebra gives the exact
    conditional posterior

        sigma^2 | theta, y  ~  Inv-Gamma(n/2, SSE(theta)/2)

    so one draw of sigma per posterior draw of theta yields the joint
    posterior of (theta, sigma) the fit already implies -- no refit needed,
    and identical to what noise='gauss_free_jeffreys' would have sampled.
    Verified against direct numerical integration of the un-marginalised
    density (max quantile difference ~1e-9).

    CAUTION for the gap calibration: this sigma is the residual scale, so it
    absorbs MODEL ERROR as well as noise. On a misspecified fit it over-
    estimates the true noise (2-3x on synthetic G2 data). The pre-contrast
    baseline SD (reported as baseline_sd) is the model-independent
    alternative, though it is noisy with only nbase frames.
    """
    spec = (build_delay_augmented_spec(TCXM, aif_use, t, acq,
                                       tau_lb=DELAY_BOUNDS[0],
                                       tau_ub=DELAY_BOUNDS[1],
                                       wx_model=WX_MODEL)
            if DELAY_MODE == "joint" else TCXM)
    bundle = build_residual(St, aif_use, t, spec, acq, subject, cfg_fit,
                            WX_MODEL, aif_cfg, relax_cfg)
    draws = res.flat()[:, :layout.n_model]
    sse = np.array([np.sum(bundle.residual_fn(p) ** 2) for p in draws])
    n = St.size
    rng = np.random.default_rng(SIGMA_SEED)
    sig = np.sqrt(sse / 2.0 / rng.standard_gamma(n / 2.0, size=sse.size))
    q = np.percentile(sig, [2.5, 25, 50, 75, 97.5])
    return sig, sse, dict(sigma_median=q[2], sigma_q25=q[1], sigma_q75=q[3],
                          sigma_ci2_5=q[0], sigma_ci97_5=q[4],
                          sigma_mean=float(np.mean(sig)),
                          sse_at_post_median=float(np.median(sse)))


# =============================================================================
# figures
# =============================================================================

def plot_lesion(sig_4d, lesion, sid, outdir, time_idx=15):
    z = int(np.argmax(lesion.sum(axis=(0, 1))))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sig_4d[:, :, z, time_idx].T, cmap="gray", origin="lower")
    ax.imshow(lesion[:, :, z].T, cmap="Reds", alpha=0.5 * lesion[:, :, z].T,
              origin="lower")
    ax.set_title(f"{sid}: fitted lesion, max slice z={z}, t={time_idx}  [{RECON}]")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{sid}_lesion_inspection.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_prior_vs_posterior(res, names, log_dims, lb, ub, prior_samples, tag,
                            save_path, sigma=None):
    """
    verify_03 figure (log10 bins for log-sampled params, except PLOT_LINEAR),
    no truth line. sigma: the reconstructed sigma draws, drawn in an extra
    panel with no prior (its prior is improper, p(sigma) ~ 1/sigma).
    """
    post = res.flat()
    res_names = list(res.param_names)
    panels = res_names + (["sigma"] if sigma is not None else [])
    n, ncol = len(panels), 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, pname in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        if pname == "sigma":
            ax.hist(sigma, bins=60, density=True, color=C_BAYES, alpha=0.75)
            med = float(np.median(sigma))
            ax.axvline(med, color=C_MEDIAN, lw=1.8,
                       label=f"median\n{med:.4g}")
            ax.set_title("sigma  (reconstructed, Inv-Gamma(n/2, SSE/2))",
                         fontsize=10)
            ax.set_xlabel(f"sigma ({UNITS['sigma']})")
            ax.set_ylabel("density")
            ax.legend(fontsize=6.5, loc="best")
            ax.grid(alpha=0.2)
            continue
        pk = post[:, res_names.index(pname)]
        if pname in names:
            j = names.index(pname)
            log = (j in log_dims) and (pname not in PLOT_LINEAR)
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
        ax.set_title(f"{pname}   R-hat={res.rhat.get(pname, np.nan):.3f}   "
                     f"ESS={res.ess.get(pname, np.nan):.0f}", fontsize=10)
        unit = f" ({UNITS[pname]})" if pname in UNITS else ""
        ax.set_xlabel(f"log10({pname}{unit})" if log else pname + unit)
        ax.set_ylabel("density")
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.2)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{tag}   TCXM + NXL + {DELAY_MODE} delay   [{RECON}]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fit(t, St, pred, sid, diagnosis, outdir):
    """Data vs posterior-median prediction."""
    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, St, "o", ms=4, color="k", label="lesion mean signal")
    ax[0].plot(t, pred, "-", color=C_BAYES, lw=2,
               label=f"posterior-median prediction (RMSE {rms(St - pred):.3g})")
    ax[0].axvspan(t[NBASE.start], t[NBASE.stop - 1], color="0.6", alpha=0.25,
                  label="baseline frames")
    ax[0].set_ylabel("Signal (a.u.)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="0.4", lw=1)
    ax[1].plot(t, St - pred, "o-", ms=3, color=C_BAYES)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("residual")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{sid} ({diagnosis})   TCXM + NXL   [{RECON}]")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{sid}_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bayes_flag(d, m):
    """Hollow marker: failed sampler QC, or prior-dominated for this metric."""
    return ((~d["qc_pass"].to_numpy(bool))
            | d[f"{m}_prior_dominated"].to_numpy(bool))


def plot_groups(df, metrics, save_path, title, col="{m}_median",
                flag_fn=_bayes_flag,
                flag_text="failed QC or prior-dominated for that metric"):
    """Mean +/- SD per group, every tumor as a point; hollow = flagged."""
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
    fig.suptitle(f"{title}   TCXM + NXL   [{RECON}]"
                 + f"\nsquares = mean ± SD; hollow points = {flag_text}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# one subject
# =============================================================================

def fit_one(sid, diagnosis, outdir):
    mask_dir = mask_dir_for(sid)           # masks: always tight_masks_2/canonical
    dce_path = find_dce(sid)               # images: whichever RECON is selected
    sig_4d, _ = load_nifti_canonical(dce_path)
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    t = np.arange(sig_4d.shape[-1]) * dt

    lesion, linfo = select_largest_lesion(mask_dir, sid, dce_path)
    if lesion.shape != sig_4d.shape[:3]:
        raise ValueError(f"{sid}: lesion mask {lesion.shape} does not match the "
                         f"{RECON} volume {sig_4d.shape[:3]} "
                         f"({os.path.basename(dce_path)})")
    aif, aif_path = load_aif_csv(sid, t)
    subject = SubjectConfig(subject_id=sid, T10=T10_TISSUE, B1=B1)

    Ct, n_vox, St = _aggregate_roi_signal(sig_4d, lesion, acq, subject, roi_cfg)
    plot_lesion(sig_4d, lesion, sid, outdir)

    base = St[NBASE]
    base_cv = float(np.std(base) / np.mean(base))
    peak_enh = float(St.max() / np.mean(base))
    arr = aif_arrival(sid)

    # --- fit -----------------------------------------------------------------
    delay_kw = (dict(delay_bounds=DELAY_BOUNDS) if DELAY_MODE == "joint"
                else dict(delay_fixed=fixed_delay_for(sid)))
    cfg_b = BayesConfig(prior_choices=PRIOR_CHOICES, n_draws=N_DRAWS,
                        n_tune=N_TUNE, n_chains=N_CHAINS,
                        noise="gauss_marginalized_sigma", seed=SEED,
                        backend=BACKEND, step="nuts", dense_mass_matrix=DENSE_MASS,
                        delay_mode=DELAY_MODE, **delay_kw)
    t0 = time.time()
    res = fit_curve_bayes(St, aif, t, TCXM, acq, subject, cfg_fit, cfg_b,
                          wx_model=WX_MODEL, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
    fit_s = time.time() - t0

    # The AIF the forward model actually saw: unshifted for joint delay (the
    # sampler shifts it per draw), pre-shifted for fixed delay.
    aif_use = (aif if DELAY_MODE == "joint"
               else shift_aif(aif, float(delay_kw["delay_fixed"]), t, acq))

    # --- prior (same object the sampler used) ---------------------------------
    layout = build_param_layout(TCXM, cfg_b, aif=aif, time_var=t, acq=acq,
                                wx_model=WX_MODEL, relax_cfg=relax_cfg)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, relax_cfg, wx_model=WX_MODEL,
                           delay_mode=DELAY_MODE,
                           n_noise=noise.n_params, noise_names=noise.param_names,
                           choices=PRIOR_CHOICES, delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(PRIOR_SEED),
                                 N_PRIOR_SAMPLES)[:, :layout.n_model]
    names_fit = list(layout.model.param_names)
    prior_d = {p: prior_samples[:, names_fit.index(p)] for p in PARAMS}
    prior_d.update(derived_draws(prior_d))

    summ, post = summarise(res, prior_d)

    # --- sigma, reconstructed from the marginalised run ----------------------
    sig_draws, sse_draws, sigma_row = sigma_draws(res, St, aif_use, t, subject,
                                                  layout)

    plot_prior_vs_posterior(res, names_fit, layout.model.log_dims or (),
                            np.asarray(layout.model.lb, float),
                            np.asarray(layout.model.ub, float),
                            prior_samples, f"{sid} ({diagnosis})",
                            os.path.join(outdir, f"{sid}_prior_post.png"),
                            sigma=sig_draws)

    # --- posterior-median prediction + residual checks ------------------------
    theta_med = np.array([summ[f"{p}_median"] for p in names_fit])
    S0 = float(compute_S0(St[None, :], acq)[0])
    pred = simulate_curve(theta_med, aif_use, t, layout.model, acq, subject,
                          cfg_fit, wx_model=WX_MODEL, aif_cfg=aif_cfg,
                          relax_cfg=relax_cfg, S0=S0)
    resid = St - pred
    post_resid = resid[NBASE.stop:]
    lag1 = float(np.corrcoef(post_resid[:-1], post_resid[1:])[0, 1])
    base_sd = float(np.std(base, ddof=1))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    plot_fit(t, St, pred, sid, diagnosis, outdir)

    np.savez_compressed(os.path.join(outdir, f"{sid}_draws.npz"),
                        chains=res.chains,
                        param_names=np.array(res.param_names, dtype=object),
                        sigma=sig_draws, sse=sse_draws,
                        St=St, aif=aif, t=t, lesion_nvox=n_vox)

    rhat_max = max(summ[f"{p}_rhat"] for p in PARAMS)
    ess_min = min(summ[f"{p}_ess"] for p in PARAMS)
    n_div = int(res.n_divergences) if res.n_divergences is not None else -1
    any_disagree = any(summ[f"{p}_chains_disagree"] for p in PARAMS)
    qc_pass = bool(rhat_max <= RHAT_MAX and ess_min >= ESS_MIN
                   and n_div == 0 and not any_disagree)

    row = dict(subject_id=sid, diagnosis=diagnosis, recon=RECON,
               dce_file=os.path.basename(dce_path),
               dce_dir=os.path.dirname(dce_path),
               aif_file=os.path.basename(aif_path), **linfo,
               wx_model=WX_MODEL.name,
               wx_blood_T10=float(aif_cfg.T10),
               wx_hematocrit=float(aif_cfg.hematocrit),
               delay_mode=DELAY_MODE,
               delay_fixed=(np.nan if DELAY_MODE == "joint"
                            else float(delay_kw["delay_fixed"])),
               baseline_cv=base_cv, peak_enhancement=peak_enh,
               aif_arrival_frame=arr,
               baseline_overlaps_aif_arrival=bool(np.isfinite(arr) and arr < NBASE.stop),
               S0=S0, rmse=rmse, baseline_sd=base_sd,
               rmse_over_baseline_sd=rmse / base_sd if base_sd > 0 else np.nan,
               resid_lag1_autocorr=lag1,
               rhat_max=rhat_max, ess_min=ess_min, n_divergences=n_div,
               any_chains_disagree=any_disagree, qc_pass=qc_pass,
               ps_flow_limited=bool(summ["E_median"] > E_FLOW_LIMITED),
               fit_seconds=fit_s, error="")
    row.update(summ)
    row.update(sigma_row)
    # the sampler's internal NLLS anchor (init only -- not a comparison fit)
    if res.point_estimate is not None:
        pe = np.asarray(res.point_estimate, float)
        row.update({f"point_{p}": float(pe[j])
                    for j, p in enumerate(layout.param_names[:len(pe)])})
    return row


# =============================================================================
# main
# =============================================================================

def main():
    outdir = RESUME_DIR or os.path.join(
        RESULTS_ROOT,
        f"run_{RECON}_nxl_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    rowdir = os.path.join(outdir, "rows")
    os.makedirs(rowdir, exist_ok=True)
    print(f"results -> {outdir}")
    print(f"recon: {RECON}   masks: {MASK_ROOT}")
    print(f"wx model: {WX_MODEL.name} (n_params={WX_MODEL.n_params}), "
          f"blood T10 {aif_cfg.T10} ms, hematocrit {aif_cfg.hematocrit}")
    print(f"delay_mode: {DELAY_MODE}"
          + (f"  bounds {DELAY_BOUNDS}" if DELAY_MODE == "joint"
             else f"  from {DELAY_FROM_FXL_CSV or DELAY_FIXED_DEFAULT}"))
    print(f"priors: {PRIOR_CHOICES}   sampler: {N_CHAINS} chains, "
          f"{N_TUNE} tune, {N_DRAWS} draws, seed {SEED}")
    print(f"TCXM bounds at run time: lb={TCXM.lb}, ub={TCXM.ub}")

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
            print(f"  R-hat max {row['rhat_max']:.3f}  ESS min {row['ess_min']:.0f}  "
                  f"div {row['n_divergences']}  QC {'PASS' if row['qc_pass'] else 'FAIL'}"
                  f"  ({row['fit_seconds']:.0f}s)")
            print("  " + "  ".join(f"{p}={row[f'{p}_median']:.4g}"
                                   f"{'*' if row[f'{p}_prior_dominated'] else ''}"
                                   for p in PARAMS + ["E", "Ktrans"])
                  + "   (* = prior-dominated)")
            print(f"  sigma {row['sigma_median']:.3g} "
                  f"[{row['sigma_ci2_5']:.3g}, {row['sigma_ci97_5']:.3g}]"
                  f"   baseline SD {row['baseline_sd']:.3g}"
                  f"   RMSE {row['rmse']:.3g}")
        except Exception as e:           # one bad subject must not stop the batch
            print(f"  !! FAILED: {e!r}")
            row = dict(subject_id=sid, diagnosis=dx, recon=RECON,
                       wx_model=WX_MODEL.name, error=repr(e))
        subj_s = time.time() - t_subj
        row["total_seconds"] = subj_s
        print(f"  total time for {sid}: {subj_s / 60:.1f} min ({subj_s:.0f} s)")
        pd.DataFrame([row]).to_csv(row_path, index=False)

    # --- cohort table ---------------------------------------------------------
    rows = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(rowdir, "*.csv")))]
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(outdir, "lesion_bayes_results_nxl.csv"), index=False)
    ok = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in [c for c in ok.columns if c.endswith(("_prior_dominated", "qc_pass"))]:
        ok[c] = ok[c].astype(bool)
    print(f"\n{len(ok)}/{len(df)} subjects fitted, "
          f"{int(ok['qc_pass'].sum()) if len(ok) else 0} pass QC")

    if len(ok):
        plot_groups(ok, GROUP_METRICS, os.path.join(outdir, "group_BvsM_all.png"),
                    "All fitted lesions")
        qc = ok[ok["qc_pass"]]
        if len(qc):
            plot_groups(qc, GROUP_METRICS,
                        os.path.join(outdir, "group_BvsM_qcpass.png"),
                        "QC-passing fits only")

        summ = []
        for m in GROUP_METRICS:
            for subset, d in (("all", ok), ("qc_pass", ok[ok["qc_pass"]])):
                for g in ("B", "M"):
                    y = d.loc[d["diagnosis"] == g, f"{m}_median"].dropna()
                    summ.append(dict(metric=m, method="bayes_nxl", group=g,
                                     subset=subset, recon=RECON, n=len(y),
                                     mean=y.mean(), sd=y.std(ddof=1),
                                     median=y.median()))
        pd.DataFrame(summ).to_csv(os.path.join(outdir, "group_summary.csv"),
                                  index=False)

    print(f"\nresults written to: {outdir}")


if __name__ == "__main__":
    main()