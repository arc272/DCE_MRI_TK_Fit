#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Tue Sep 22 15:01:11 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_lesions_bayes.py
====================
Bayesian TCXM + joint-delay fit of the LARGEST lesion in each subject listed
in tumor_identification.csv, using the AIF CSVs produced by extract_aifs.py.

Model / sampler: identical to the validated verify_03 / verify_04b setup
    TCXM, delay_mode='joint', PRIOR_CHOICES = {rates: loguniform, volfrac: beta},
    gauss_marginalized_sigma, BlackJAX NUTS, dense mass matrix.
Tissue signal model: 06_roi_fitting.ipynb cell [4]
    AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9), SubjectConfig(T10=1530, B1=1.0),
    FittingConfig(n_lhs=200, n_top=10, fit_space="signal"), mean signal over
    the lesion (average_space="signal"), nbase = frames 0..6.

Per subject:
  * {sid}_lesion_inspection.png      lesion slice with the fitted component
  * {sid}_prior_post.png             prior vs posterior, every parameter
  * {sid}_fit.png                    data vs posterior-median prediction + residuals
  * {sid}_draws.npz                  all chains (for later BEST / derived checks)
  * rows/{sid}.csv                   one-row result (lets a crashed run resume)
Cohort:
  * lesion_bayes_results.csv         every estimate + QC metric, one row / subject
  * group_BvsM_all.png               mean +/- SD per metric, B vs M, all fits
  * group_BvsM_qcpass.png            same, fits passing QC only
  * group_summary.csv                n / mean / SD / median per metric and group

Units: fp, ps, Ktrans in s^-1 (the model's units); tau_delay, Tc, Te in s.

Run:
    PYTHONPATH=src python scripts/fit_lesions_bayes.py
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
from scipy.ndimage import label as nd_label

warnings.filterwarnings("ignore")

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            RoiFittingConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.signal_models import compute_S0
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
DATA_ROOT   = "/mnt/labspace/users/allison/fastMRI_data/tight_masks_2/canonical"
AIF_DIR     = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"
AIF_SUMMARY = os.path.join(AIF_DIR, "AIF_extraction_summary.csv")   # optional
RESULTS_ROOT = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results"

ONLY = []            # e.g. ["FMB056"] to run a single subject first
RESUME_DIR = None      # path of an earlier run dir to resume (skips done subjects)

# --- acquisition / tissue: 06_roi_fitting.ipynb cell [4] --------------------
NBASE = slice(0, 7)
acq     = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)
T10_TISSUE, B1 = 1530.0, 1.0
cfg_fit = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
roi_cfg = RoiFittingConfig(average_space="signal")
SPOKES_PER_FRAME, NSPOKE, SCANTIME = 8, 288, 150



# --- Bayesian model: validated verify_03 / verify_04b configuration ---------
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS  = (0, 25.0)
N_DRAWS, N_TUNE, N_CHAINS = 3000, 3000, 16
BACKEND, DENSE_MASS, SEED = "blackjax", True, 1
N_PRIOR_SAMPLES, PRIOR_SEED = 20000, 999

# --- QC thresholds ------------------------------------------------------------
RHAT_MAX       = 1.01
ESS_MIN        = 400
IQR_RATIO_FLAT = 0.8     # posterior/prior IQR above this = prior-dominated
E_FLOW_LIMITED = 0.4     # verify_03: ps recovery degrades once E=ps/(ps+fp) > ~0.3-0.5
CHAIN_SPLIT    = 0.5     # per-chain median spread > 0.5 x within-chain IQR = disagree

PARAMS  = ["ve", "vp", "fp", "ps", "tau_delay"]
DERIVED = ["vtot", "E", "Ktrans", "Tc", "Te"]
LOG_SCALE = {"fp", "ps", "Ktrans", "Tc", "Te"}      # IQR ratio + axes on log scale
GROUP_METRICS = ["ve", "vp", "fp", "ps", "tau_delay", "vtot", "E", "Ktrans"]

C_BAYES, C_MEDIAN, C_PRIOR = "#B22234", "#E67E22", "0.75"
C_GROUP = {"B": "#1B4F72", "M": "#B22234"}


# =============================================================================
# inputs
# =============================================================================

def find_dce(subj_dir, sid):
    hits = sorted(glob.glob(os.path.join(subj_dir, f"{sid}_tv*.nii.gz")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{sid}: expected one '{sid}_tv*.nii.gz', "
                                f"found {len(hits)}: {hits}")
    return hits[0]

def select_largest_lesion(subj_dir, sid, dce_path):
    """
    Candidates = every .nii / .nii.gz in the subject folder except the aorta
    ROI ('orta' in the name) and the DCE volume itself. The file with the
    most nonzero voxels is used, whole, as the lesion mask.
    """
    files = sorted(
        f for f in os.listdir(subj_dir)
        if f.lower().endswith((".nii", ".nii.gz"))
        and "orta" not in f.lower()
        and "idif" not in f.lower()
        and os.path.join(subj_dir, f) != dce_path
    )
    if not files:
        raise FileNotFoundError(f"{sid}: no lesion masks (non-aorta .nii) in {subj_dir}")

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
    """Aorta arrival frame from extract_aifs.py's summary, if available."""
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
                    f"{p}_ci2.5": q[0], f"{p}_ci97.5": q[4], f"{p}_mean": float(np.mean(x)),
                    f"{p}_iqr_ratio": _iqr(x, log) / _iqr(prior_d[p], log)})
        out[f"{p}_prior_dominated"] = bool(out[f"{p}_iqr_ratio"] > IQR_RATIO_FLAT)
        if p in PARAMS:
            out[f"{p}_rhat"] = float(res.rhat.get(p, np.nan))
            out[f"{p}_ess"]  = float(res.ess.get(p, np.nan))
            dis, spread, within = chain_disagree(ch[p])
            out[f"{p}_chains_disagree"] = bool(dis)
    return out, post


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


def plot_prior_vs_posterior(res, names, log_dims, lb, ub, prior_samples, tag,
                            save_path):
    """verify_03 figure (log10 bins for log-sampled params), no truth line."""
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
            log = j in log_dims
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
        ax.set_xlabel(f"log10({pname})" if log else pname)
        ax.set_ylabel("density")
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.2)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{tag}   TCXM + joint delay", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fit(t, St, pred, sid, diagnosis, outdir):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, St, "o", ms=4, color="k", label="lesion mean signal")
    ax[0].plot(t, pred, "-", color=C_BAYES, lw=2, label="posterior-median prediction")
    ax[0].axvspan(t[NBASE.start], t[NBASE.stop - 1], color="0.6", alpha=0.25,
                  label="baseline frames")
    ax[0].set_ylabel("Signal (a.u.)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="0.4", lw=1)
    ax[1].plot(t, St - pred, "o-", ms=3, color="0.3")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("residual")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{sid} ({diagnosis})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{sid}_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_groups(df, metrics, save_path, title):
    """Mean +/- SD per group, every tumor as a point; hollow = flagged."""
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for k, m in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        allv = df[f"{m}_median"].to_numpy(float)
        allv = allv[np.isfinite(allv)]
        for gx, g in enumerate(("B", "M")):
            d = df[(df["diagnosis"] == g) & np.isfinite(df[f"{m}_median"])]
            y = d[f"{m}_median"].to_numpy(float)
            flag = (~d["qc_pass"].to_numpy(bool)) | d[f"{m}_prior_dominated"].to_numpy(bool)
            x = gx + rng.uniform(-0.12, 0.12, y.size)
            ax.scatter(x[~flag], y[~flag], s=22, color=C_GROUP[g], alpha=0.8, zorder=2)
            ax.scatter(x[flag], y[flag], s=22, facecolors="none",
                       edgecolors=C_GROUP[g], alpha=0.8, zorder=2)
            if y.size:
                mu = y.mean()
                sd = y.std(ddof=1) if y.size > 1 else 0.0
                lo = min(sd, 0.999 * mu) if (m in LOG_SCALE and mu > 0) else sd
                ax.errorbar(gx + 0.28, mu, yerr=[[lo], [sd]], fmt="s", ms=7,
                            color="k", capsize=5, lw=1.6, zorder=3)
        if m in LOG_SCALE and allv.size and np.all(allv > 0):
            ax.set_yscale("log")
            ax.set_ylim(allv.min() / 2.0, allv.max() * 2.0)
        ax.set_xticks([0, 1])
        nB = int(((df["diagnosis"] == "B") & np.isfinite(df[f"{m}_median"])).sum())
        nM = int(((df["diagnosis"] == "M") & np.isfinite(df[f"{m}_median"])).sum())
        ax.set_xticklabels([f"B (n={nB})", f"M (n={nM})"])
        ax.set_xlim(-0.5, 1.6)
        unit = {"fp": " (s$^{-1}$)", "ps": " (s$^{-1}$)", "Ktrans": " (s$^{-1}$)",
                "tau_delay": " (s)"}.get(m, "")
        ax.set_title(m + unit)
        ax.grid(alpha=0.25, which="both", axis="y")
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title + "\nsquares = mean ± SD; hollow points = failed QC "
                 "or prior-dominated for that metric", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# one subject
# =============================================================================

def fit_one(sid, diagnosis, outdir):
    subj_dir = os.path.join(DATA_ROOT, sid)
    dce_path = find_dce(subj_dir, sid)
    sig_4d, _ = load_nifti_canonical(dce_path)
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    t = np.arange(sig_4d.shape[-1]) * dt

    lesion, linfo = select_largest_lesion(subj_dir, sid, dce_path)
    aif, aif_path = load_aif_csv(sid, t)
    subject = SubjectConfig(subject_id=sid, T10=T10_TISSUE, B1=B1)

    Ct, n_vox, St = _aggregate_roi_signal(sig_4d, lesion, acq, subject, roi_cfg)
    plot_lesion(sig_4d, lesion, sid, outdir)

    base = St[NBASE]
    base_cv = float(np.std(base) / np.mean(base))
    peak_enh = float(St.max() / np.mean(base))
    arr = aif_arrival(sid)

    # --- fit -----------------------------------------------------------------
    cfg_b = BayesConfig(prior_choices=PRIOR_CHOICES, n_draws=N_DRAWS,
                        n_tune=N_TUNE, n_chains=N_CHAINS,
                        noise="gauss_marginalized_sigma", seed=SEED,
                        backend=BACKEND, step="nuts", dense_mass_matrix=DENSE_MASS,
                        delay_mode="joint", delay_bounds=DELAY_BOUNDS)
    t0 = time.time()
    res = fit_curve_bayes(St, aif, t, TCXM, acq, subject, cfg_fit, cfg_b)
    fit_s = time.time() - t0

    # --- prior (same object the sampler used) ---------------------------------
    layout = build_param_layout(TCXM, cfg_b, aif=aif, time_var=t, acq=acq)
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, None, delay_mode="joint",
                           n_noise=noise.n_params, noise_names=noise.param_names,
                           choices=PRIOR_CHOICES, delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(PRIOR_SEED),
                                 N_PRIOR_SAMPLES)[:, :layout.n_model]
    names_fit = list(layout.model.param_names)
    prior_d = {p: prior_samples[:, names_fit.index(p)] for p in PARAMS}
    prior_d.update(derived_draws(prior_d))

    plot_prior_vs_posterior(res, names_fit, layout.model.log_dims or (),
                            np.asarray(layout.model.lb, float),
                            np.asarray(layout.model.ub, float),
                            prior_samples, f"{sid} ({diagnosis})",
                            os.path.join(outdir, f"{sid}_prior_post.png"))

    summ, post = summarise(res, prior_d)

    # --- posterior-median prediction + residual checks ------------------------
    theta_med = np.array([summ[f"{p}_median"] for p in names_fit])
    S0 = float(compute_S0(St[None, :], acq)[0])
    pred = simulate_curve(theta_med, aif, t, layout.model, acq, subject,
                          cfg_fit, S0=S0)
    resid = St - pred
    post_resid = resid[NBASE.stop:]
    lag1 = float(np.corrcoef(post_resid[:-1], post_resid[1:])[0, 1])
    base_sd = float(np.std(base, ddof=1))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    plot_fit(t, St, pred, sid, diagnosis, outdir)

    np.savez_compressed(os.path.join(outdir, f"{sid}_draws.npz"),
                        chains=res.chains,
                        param_names=np.array(res.param_names, dtype=object),
                        St=St, aif=aif, t=t, lesion_nvox=n_vox)

    rhat_max = max(summ[f"{p}_rhat"] for p in PARAMS)
    ess_min = min(summ[f"{p}_ess"] for p in PARAMS)
    n_div = int(res.n_divergences) if res.n_divergences is not None else -1
    any_disagree = any(summ[f"{p}_chains_disagree"] for p in PARAMS)
    qc_pass = bool(rhat_max <= RHAT_MAX and ess_min >= ESS_MIN
                   and n_div == 0 and not any_disagree)

    row = dict(subject_id=sid, diagnosis=diagnosis,
               dce_file=os.path.basename(dce_path),
               aif_file=os.path.basename(aif_path), **linfo,
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
        try:
            row = fit_one(sid, dx, outdir)
            print(f"  lesion {row['lesion_file']}: {row['n_vox']} voxels "
                  f"(largest of {row['n_lesion_files']} mask file(s))")
            print(f"  R-hat max {row['rhat_max']:.3f}  ESS min {row['ess_min']:.0f}  "
                  f"div {row['n_divergences']}  QC {'PASS' if row['qc_pass'] else 'FAIL'}"
                  f"  ({row['fit_seconds']:.0f}s)")
            print("  " + "  ".join(f"{p}={row[f'{p}_median']:.4g}"
                                   f"{'*' if row[f'{p}_prior_dominated'] else ''}"
                                   for p in PARAMS + ["E", "Ktrans"])
                  + "   (* = prior-dominated)")
        except Exception as e:           # one bad subject must not stop the batch
            print(f"  !! FAILED: {e!r}")
            row = dict(subject_id=sid, diagnosis=dx, error=repr(e))
        pd.DataFrame([row]).to_csv(row_path, index=False)

    # --- cohort table ---------------------------------------------------------
    rows = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(rowdir, "*.csv")))]
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(outdir, "lesion_bayes_results.csv"), index=False)
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
            plot_groups(qc, GROUP_METRICS, os.path.join(outdir, "group_BvsM_qcpass.png"),
                        "QC-passing fits only")

        summ = []
        for m in GROUP_METRICS:
            for g in ("B", "M"):
                for subset, d in (("all", ok), ("qc_pass", ok[ok["qc_pass"]])):
                    y = d.loc[d["diagnosis"] == g, f"{m}_median"].dropna()
                    summ.append(dict(metric=m, group=g, subset=subset, n=len(y),
                                     mean=y.mean(), sd=y.std(ddof=1),
                                     median=y.median()))
        pd.DataFrame(summ).to_csv(os.path.join(outdir, "group_summary.csv"), index=False)

    print(f"\nresults written to: {outdir}")


if __name__ == "__main__":
    main()