#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
group_stats_analysis.py
=======================
Benign-vs-malignant statistics on a fit_lesions_bayes.py run, and the three
artifact tests. Library code (definitions of every statistic) is in
src/dce_mri/group_stats.py.

  1. SIZE CHECK          lesion voxel counts by group; AUC of size alone
  2. DISCRIMINATION      per parameter, Bayes posterior medians AND NLLS:
                         AUC (DeLong CI), Mann-Whitney p, Cliff's delta,
                         Hodges-Lehmann shift, size-adjusted OR + LR p;
                         Holm and BH within the parameter family
  3. BLOCK TESTS         groups of parameters jointly beyond size (LR test)
                         + cross-validated AUC gain over size alone
  4. IDENTIFIABILITY     prior-dominated / at-bound / flow-limited rates by
                         group, Fisher p, and whether size explains them
  5. PRIOR SENSITIVITY   (RUN_PRIOR_SENS) importance reweighting of the
                         stored draws under alternative priors: location
                         (median x f), width, family. Per setting: AUC and
                         prior-dominated rates. 'Transmitted fraction' per
                         lesion = how much of a prior-median shift reaches the
                         posterior median (0 = data-dominated, 1 = prior-
                         dominated); B vs M tests DIFFERENTIAL SHRINKAGE.
                         The predicted sensitivity order (stated in advance,
                         PREDICTED_ORDER) is checked against the observed one.
  6. SUBSAMPLE COMPARE   (SUBSAMPLE_RUN_DIR) benign vs malignant-full vs
                         malignant-subsampled-to-benign-size: AUC change with
                         bootstrap CI, prior-dominated rates, and an SNR-
                         matching check on baseline CV
  7. SUBTYPES            (SUBTYPE_CSV) every pairwise comparison among
                         B, DCIS, IDC, ILC, IDC/ILC: each malignant subtype vs
                         benign AND each subtype vs each other. Same statistics
                         as section 2 per pair (AUC, Mann-Whitney, Cliff's
                         delta, Hodges-Lehmann, size-adjusted LR) + a Kruskal-
                         Wallis omnibus per parameter. Holm across the pairs
                         (within a parameter) and across parameters (within a
                         pair).
  8. SIZE BAND           (BAND_MM3) the same statistics as section 2 on the
                         MATCHED SUBSET of lesions whose volume falls inside
                         an overlapping band -- no refitting, so it is the
                         cheap counterpart of section 6. Reports the residual
                         size AUC and the baseline-CV match INSIDE the band
                         (both must be checked before the parameter AUCs mean
                         anything), a sweep over BAND_CANDIDATES so the band
                         can be chosen on size grounds alone, a per-lesion CSV
                         and volume box plots.

Fit RMSE (rmse, rmse_over_baseline_sd; and the nlls_ versions) is plotted by
group (B vs M, and by subtype) as DESCRIPTIVE box plots only. It is not in
PARAMS / FAMILY and no univariate tests are run on it.

Outputs, in OUT_DIR: one CSV per section, figures, and a printed summary.

Run:
    PYTHONPATH=src python scripts/group_stats_analysis.py
"""

import os

# --- which device JAX may touch -- MUST come before anything imports jax ------
# (dce_mri.hierarchical imports jax; if a kernel has already imported it, restart
#  the kernel for this to take effect.) This script is mostly numpy/scipy, so
#  GPU_ID = "" (CPU only) is a fine choice and cannot disturb a running fit.
#GPU_ID = "1"                                     # "0", "1", or "" for CPU only
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # indices match nvidia-smi
#os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
#os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
#if GPU_ID == "":
#    os.environ["JAX_PLATFORMS"] = "cpu"

import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from dce_mri.config import AcquisitionConfig
from dce_mri import group_stats as G
from dce_mri import hierarchical as H

# =============================================================================
# settings
# =============================================================================

RUN_DIR = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/run_grasp_8spf_20261015_150449_TCXM"
SUBSAMPLE_RUN_DIR = None     # a subsample_lesions.py run dir, or None
OUT_DIR = None               # None -> RUN_DIR/group_stats_<timestamp>

SUBSET   = "all"         # "qc_pass" or "all" (non-error)
SIZE_COL = "n_vox"
VOXEL_MM3  = 1.0 * 1.0 * 1.1   # mm^3 per voxel (1 x 1 x 1.1 mm)
VOLUME_LOG = True              # log y-axis for the tumor-volume box plots

# Rates are stored in s^-1 by fit_lesions_bayes.py. Everything this script reports
# (tables, printed output, figures) is converted to min^-1 at load time.
# {} -> leave in s^-1. Prior sensitivity (section 5) still computes in s^-1
# internally (the priors are defined there) and only its reported medians are scaled.
RATE_SCALE = {"fp": 60.0, "ps": 60.0, "Ktrans": 60.0}
UNITS      = {"fp": "min$^{-1}$", "ps": "min$^{-1}$", "Ktrans": "min$^{-1}$", "tau_delay": "s", "tau_i": "s"}
PARAMS   = ("ve", "vp", "fp", "ps", "tau_delay", "tau_i")
LOG_PARAMS = ("fp", "ps", "Ktrans", "tau_i")          # log scale for regression / HL
FAMILY   = ("ve", "vp", "fp", "ps", "tau_delay", "tau_i", "vtot", "E", "Ktrans")  # multiplicity
                                             # family (tau_delay = nuisance)
BLOCKS = {"vp+fp+ps": ("vp", "fp", "ps"),
          "ve+vp+fp+ps": ("ve", "vp", "fp", "ps"),
          "ve+vp+fp+ps+tau_delay": ("ve", "vp", "fp", "ps", "tau_delay" ),
          "ve+vp+fp+ps+tau_i": ("ve", "vp", "fp", "ps","tau_i")}
CV_REPEATS = 20

# --- prior sensitivity (must match the fit run's prior) ----------------------
RUN_PRIOR_SENS = False
PRIOR_CHOICES  = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS   = (0, 25.0)
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
SENS_PARAMS   = ("ve", "vp", "fp", "ps")
LOC_FACTORS   = {"fp": (0.5, 2.0), "ps": (0.5, 2.0),
                 "vp": (0.5, 1.5), "ve": (0.5, 1.5)}  # x2 puts ve/vp's
                                                     # median on its bound
WIDTH_FACTORS = (0.5, 2.0)
T_DF          = 3
PREDICTED_ORDER = ("ps", "vp", "fp")   # most -> least prior-sensitive,
                                       # stated BEFORE looking
ESS_MIN_W  = 200                       # weight ESS below this = unreliable
KHAT_BAD   = 0.7
EXCLUDE_UNRELIABLE = True              # drop unreliable lesions per setting
IQR_RATIO_FLAT = 0.8                   # same threshold as the fit script
LOG_IQR = ("fp", "ps", "Ktrans", "Tc", "Te")   # = fit script's LOG_SCALE

# --- subsample comparison -----------------------------------------------------
N_BOOT = 2000
SEED   = 0

# --- malignant subtype comparison (section 7) ---------------------------------
# CSV with Patient_ID, Diagnosis, Malignant_lesion_type. None -> skip section 7.
SUBTYPE_CSV     = "/home/allisonchen/tumor_identification.csv"
SUBTYPE_ORDER   = ("B", "DCIS", "IDC", "ILC", "IDC/ILC")   # B first = reference
SUBTYPE_MIN_N   = 4      # a pair is skipped if either group has fewer lesions
SUBTYPE_SMALL_N = 10     # pairs with a group below this are flagged small_n

# Merged grouping: all invasive types pooled into one box; B and DCIS unchanged.
# Section 7 is run a second time on these groups (files get a "_merged" suffix).
SUBTYPE_MERGE = {"IDC": "Invasive", "ILC": "Invasive", "IDC/ILC": "Invasive"}
MERGED_ORDER  = ("B", "DCIS", "Invasive")

# --- size band (section 8): matched subset, NO refitting ----------------------
# The band the analysis uses, in mm^3 of fitted-ROI volume (n_vox x VOXEL_MM3).
# None -> skip section 8. CHOOSE THE BAND FROM THE SWEEP (and from the section-1
# size distributions) BEFORE looking at any parameter result: picking the band
# that gives the nicest parameter AUC is selective reporting.
BAND_MM3        = (100.0, 800.0)
# Bands reported side by side in 8_size_band_sweep.csv (n per group + the size
# AUC that remains inside each band). Descriptive only; no parameter tests.
BAND_CANDIDATES = ((100.0, 800.0), (100.0, 600.0), (150.0, 800.0),
                   (200.0, 800.0), (150.0, 500.0))
BAND_MIN_N      = 4      # a band with fewer lesions than this in either group
                         # is reported in the sweep but not tested

C_GROUP = {"B": "#1B4F72", "M": "#B22234", "Msub": "#E67E22"}
C_SUB = {"B": "#1B4F72", "DCIS": "#E6A23C", "IDC": "#B22234",
         "ILC": "#7D3C98", "IDC/ILC": "#1E8449", "Invasive": "#B22234"}


# =============================================================================
# loading
# =============================================================================

def _bool(v):
    return str(v).strip().lower() == "true"


def load_results(run_dir, subset, fname="lesion_bayes_results.csv"):
    df = pd.read_csv(os.path.join(run_dir, fname))
    err = df["error"].fillna("").astype(str).str.strip() if "error" in df else ""
    df = df[err == ""].copy()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    df = df[df["diagnosis"].isin(["B", "M"])]
    if subset == "qc_pass":
        df = df[df["qc_pass"].map(_bool)]
    elif subset != "all":
        raise ValueError("SUBSET must be 'qc_pass' or 'all'")
    for c in [c for c in df.columns if c.endswith(("_prior_dominated", "_at_bound",
                                                   "_flow_limited", "qc_pass",
                                                   "nlls_success"))]:
        df[c] = df[c].map(_bool)
    return to_per_min(df.reset_index(drop=True))


def to_per_min(df):
    """Scale the VALUE columns of the rate parameters (s^-1 -> min^-1). Ratios and
    flags are unit-free and are left alone; differences scale like the values."""
    df = df.copy()
    n = 0
    for p, f in RATE_SCALE.items():
        for c in (f"{p}_median", f"{p}_q25", f"{p}_q75", f"{p}_ci2.5", f"{p}_ci97.5",
                  f"{p}_mean", f"nlls_{p}", f"{p}_nlls_minus_median"):
            if c in df:
                df[c] = df[c].astype(float) * f
                n += 1
    if n:
        print(f"  rates converted s^-1 -> min^-1 for: {', '.join(RATE_SCALE)} "
              f"({n} columns)")
    return df


# =============================================================================
# sections 1-4
# =============================================================================

def section_size(df, y, out):
    s = df[SIZE_COL].astype(float).to_numpy()
    auc, p = G.auc_mw(s[y == 0], s[y == 1])
    lo, hi, _ = G.delong_ci(s[y == 0], s[y == 1])
    r = pd.DataFrame([dict(
        n_B=int((y == 0).sum()), n_M=int((y == 1).sum()),
        median_vox_B=float(np.median(s[y == 0])), median_vox_M=float(np.median(s[y == 1])),
        iqr_B=f"{np.percentile(s[y == 0], 25):.0f}-{np.percentile(s[y == 0], 75):.0f}",
        iqr_M=f"{np.percentile(s[y == 1], 25):.0f}-{np.percentile(s[y == 1], 75):.0f}",
        auc_size=auc, auc_ci2_5=lo, auc_ci97_5=hi, mw_p=p)])
    r.to_csv(os.path.join(out, "1_size_check.csv"), index=False)
    print(f"\n1. SIZE: median voxels B {r['median_vox_B'][0]:.0f} vs M "
          f"{r['median_vox_M'][0]:.0f}; AUC of size alone {auc:.3f} "
          f"[{lo:.3f}, {hi:.3f}], p={p:.2g}")
    return r


def section_discrimination(df, y, logs, out):
    tabs = {}
    for label, fmt in (("bayes", "{p}_median"), ("nlls", "nlls_{p}")):
        cols = {p: fmt.format(p=p) for p in PARAMS if fmt.format(p=p) in df}
        if not cols:
            continue
        vals = {p: df[c].astype(float).to_numpy() for p, c in cols.items()}
        t = G.discrimination_table(vals, y, logs, log_params=LOG_PARAMS,
                                   family=[p for p in FAMILY if p in vals])
        t.insert(0, "estimate", label)
        t.to_csv(os.path.join(out, f"2_discrimination_{label}.csv"), index=False)
        tabs[label] = t
        print(f"\n2. DISCRIMINATION ({label}; AUC>0.5 = malignant higher)")
        print(t[["param", "median_B", "median_M", "auc", "auc_ci2_5", "auc_ci97_5",
                 "mw_p", "mw_p_holm", "sizeadj_or_per_sd", "sizeadj_lr_p",
                 "sizeadj_lr_p_holm", "sizeadj_separation"]].to_string(index=False,
                                                 float_format=lambda v: f"{v:.3g}"))
    return tabs


def section_blocks(df, y, logs, out):
    rows = []
    for label, fmt in (("bayes", "{p}_median"), ("nlls", "nlls_{p}")):
        for name, ps in BLOCKS.items():
            cols = [fmt.format(p=p) for p in ps]
            if not all(c in df for c in cols):
                continue
            X = df[cols].astype(float).to_numpy()
            r = G.block_test(y, logs, X, log_mask=[p in LOG_PARAMS for p in ps],
                             n_repeats=CV_REPEATS)
            rows.append(dict(estimate=label, block=name, **r))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "3_block_tests.csv"), index=False)
    print("\n3. BLOCK TESTS (beyond log size)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return t


def section_identifiability(df, y, logs, out):
    flags = {}
    for p in PARAMS:
        if f"{p}_prior_dominated" in df:
            flags[f"{p}_prior_dominated"] = df[f"{p}_prior_dominated"].to_numpy()
    for p in ("ve", "vp", "fp", "ps", "tau_delay"):
        if f"nlls_{p}_at_bound" in df:
            flags[f"nlls_{p}_at_bound"] = df[f"nlls_{p}_at_bound"].to_numpy()
    if "ps_flow_limited" in df:
        flags["ps_flow_limited"] = df["ps_flow_limited"].to_numpy()
    t = G.flag_table(flags, y, logs)
    t.to_csv(os.path.join(out, "4_identifiability.csv"), index=False)
    print("\n4. IDENTIFIABILITY ASYMMETRY")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return t


# =============================================================================
# section 5: prior sensitivity
# =============================================================================

def section_prior_sensitivity(df, y, logs, out):
    first = os.path.join(RUN_DIR, f"{df['subject_id'].iloc[0]}_draws.npz")
    z0 = np.load(first, allow_pickle=True)
    prior, tr = H.tcxm_joint_delay_prior(PRIOR_CHOICES, DELAY_BOUNDS, acq,
                                         np.asarray(z0["t"]), np.asarray(z0["aif"]))
    names = list(tr.names)
    base = G.base_factors(PRIOR_CHOICES, tr)
    sd, mean = G.check_base_factors(prior, base, tr)
    if sd > 1e-8:
        raise ValueError(f"base prior factors disagree with the package prior "
                         f"(SD of difference {sd:.2e}) -- PRIOR_CHOICES wrong?")
    settings, skipped = G.default_settings(prior, tr, SENS_PARAMS, LOC_FACTORS,
                                           WIDTH_FACTORS, T_DF)
    for s in skipped:
        print(f"  skipped setting {s}")

    draws, keep = [], []
    for i, sid in enumerate(df["subject_id"]):
        f = os.path.join(RUN_DIR, f"{sid}_draws.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        nm = [str(s) for s in z["param_names"]]
        ch = np.asarray(z["chains"], float)
        draws.append(ch.reshape(-1, ch.shape[-1])[:, [nm.index(p) for p in names]])
        keep.append(i)
    d = df.iloc[keep].reset_index(drop=True)
    yk, lk = y[keep], logs[keep]
    print(f"\n5. PRIOR SENSITIVITY: {len(draws)} lesions with draws, "
          f"{len(settings)} prior settings")
    quantities = [p for p in PARAMS]
    long = G.reweight_cohort(draws, names, settings, base, quantities,
                             H.pareto_khat, log_iqr=LOG_IQR)
    long.insert(1, "subject_id", d["subject_id"].to_numpy()[long["lesion"]])
    long.insert(2, "diagnosis", d["diagnosis"].to_numpy()[long["lesion"]])
    long["unreliable"] = (long["ess"] < ESS_MIN_W) | (long["khat"] > KHAT_BAD)
    long.to_csv(os.path.join(out, "5_prior_sens_lesions.csv"), index=False)

    rows = []
    for st in settings:
        sub = long[long["setting"] == st.name].sort_values("lesion")
        use = ~sub["unreliable"].to_numpy() if EXCLUDE_UNRELIABLE else np.ones(len(sub), bool)
        piqr = G.prior_iqr(st, prior, tr, quantities, log_iqr=LOG_IQR)
        for q in quantities:
            v = sub[f"{q}_median"].to_numpy()
            vb, vm = v[use & (yk == 0)], v[use & (yk == 1)]
            auc, p = G.auc_mw(vb, vm)
            lo, hi, _ = G.delong_ci(vb, vm)
            pdom = sub[f"{q}_iqr"].to_numpy() / piqr[q] > IQR_RATIO_FLAT
            rows.append(dict(setting=st.name, kind=st.kind, target=st.target,
                             quantity=q, n_used=int(use.sum()),
                             n_unreliable=int((~use).sum()), auc=auc,
                             auc_ci2_5=lo, auc_ci97_5=hi, mw_p=p,
                             cliffs_delta=2 * auc - 1,
                             median_B=float(np.median(vb)) * RATE_SCALE.get(q, 1.0),
                             median_M=float(np.median(vm)) * RATE_SCALE.get(q, 1.0),
                             prior_dom_B=float(pdom[use & (yk == 0)].mean()),
                             prior_dom_M=float(pdom[use & (yk == 1)].mean())))
    auc_t = pd.DataFrame(rows)
    auc_t.to_csv(os.path.join(out, "5_prior_sens_auc.csv"), index=False)

    # transmitted fraction of a prior-median shift, per lesion, on z
    summ = []
    for p in SENS_PARAMS:
        j = names.index(p)
        loc = [s for s in settings if s.kind == "location" and s.target == p]
        if len(loc) < 2:
            continue
        lo_s, hi_s = loc[0], loc[-1]
        mu_lo, mu_hi = lo_s.meta["mu"], hi_s.meta["mu"]
        a = long[long["setting"] == lo_s.name].sort_values("lesion")
        b = long[long["setting"] == hi_s.name].sort_values("lesion")
        ok = ~(a["unreliable"].to_numpy() | b["unreliable"].to_numpy())
        za = G._zcol(tr, j, a[f"{p}_median"].to_numpy())[0]
        zb = G._zcol(tr, j, b[f"{p}_median"].to_numpy())[0]
        frac = (zb - za) / (mu_hi - mu_lo)
        fB, fM = frac[ok & (yk == 0)], frac[ok & (yk == 1)]
        own = auc_t[(auc_t["quantity"] == p) & (auc_t["target"] == p)]
        own_loc = own[own["kind"].isin(["location", "reference"])]["auc"]
        own_w = own[own["kind"].isin(["width", "reference"])]["auc"]
        fam = auc_t[(auc_t["quantity"] == p) & (auc_t["kind"] == "family")].set_index("setting")["auc"]
        base_auc = auc_t[(auc_t["quantity"] == p) & (auc_t["setting"] == "base")]["auc"].iloc[0]
        summ.append(dict(
            param=p, auc_base=base_auc,
            auc_range_own_location=float(own_loc.max() - own_loc.min()),
            auc_range_own_width=float(own_w.max() - own_w.min()),
            auc_family_t_minus_normal=float(fam.get(f"all:t{T_DF}", np.nan)
                                            - fam.get("all:ref", np.nan)),
            transmitted_B_median=float(np.median(fB)),
            transmitted_M_median=float(np.median(fM)),
            transmitted_B_vs_M_p=float(stats.mannwhitneyu(fB, fM).pvalue)
            if fB.size > 1 and fM.size > 1 else np.nan))
    sens = pd.DataFrame(summ)
    obs = list(sens.sort_values("auc_range_own_location", ascending=False)["param"])
    obs_pred = [p for p in obs if p in PREDICTED_ORDER]
    sens.attrs["observed_order"] = obs_pred
    sens["rank_observed"] = sens["auc_range_own_location"].rank(ascending=False).astype(int)
    sens.to_csv(os.path.join(out, "5_prior_sens_summary.csv"), index=False)
    match = obs_pred == list(PREDICTED_ORDER)
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"  predicted sensitivity order {' > '.join(PREDICTED_ORDER)}; observed "
          f"{' > '.join(obs_pred)} -> {'MATCHES' if match else 'does not match'}")
    print("  transmitted fraction: 0 = posterior median ignores the prior shift, "
          "1 = follows it fully. B > M = benign shrunk more.")
    fig_prior_sens(auc_t, os.path.join(out, "5_prior_sens_auc.png"))
    return auc_t, sens, match


def fig_prior_sens(t, path):
    qs = [q for q in ("ve", "vp", "fp", "ps", "vtot", "E", "Ktrans") if q in set(t["quantity"])]
    order = list(dict.fromkeys(t["setting"]))
    fig, axes = plt.subplots(len(qs), 1, figsize=(max(8, 0.45 * len(order)), 2.4 * len(qs)),
                             sharex=True, squeeze=False)
    for ax, q in zip(axes[:, 0], qs):
        d = t[t["quantity"] == q].set_index("setting").loc[order]
        x = np.arange(len(order))
        own = np.array([s.split(":")[0] in (q, "all") or s == "base" for s in order])
        ax.errorbar(x[own], d["auc"][own], yerr=[d["auc"][own] - d["auc_ci2_5"][own],
                    d["auc_ci97_5"][own] - d["auc"][own]], fmt="o", color="k", ms=4)
        ax.plot(x[~own], d["auc"][~own], "o", color="0.65", ms=3)
        ax.axhline(0.5, color="crimson", lw=0.8, ls="--")
        ax.axhline(d.loc["base", "auc"], color="0.4", lw=0.8)
        ax.set_ylabel(f"AUC {q}")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xticks(np.arange(len(order)))
    axes[-1, 0].set_xticklabels(order, rotation=70, ha="right", fontsize=8)
    fig.suptitle("AUC under each prior setting (black = this parameter's own prior "
                 "changed; grey line = base prior; red = 0.5)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# section 6: size-matched subsample comparison
# =============================================================================

def _auc_fast(b, m):
    r = stats.rankdata(np.r_[b, m])
    return (r[b.size:].sum() - m.size * (m.size + 1) / 2) / (b.size * m.size)


def section_subsample(df, out):
    sub = load_results(SUBSAMPLE_RUN_DIR, "all", fname="subsample_results.csv")
    if SUBSET == "qc_pass":
        n0 = sub["subject_id"].nunique()
        sub = sub[sub["qc_pass"]] if "qc_pass" in sub else sub
        print(f"  subsample fits failing QC dropped: lesions {n0} -> "
              f"{sub['subject_id'].nunique()}")
    agg = {}
    for c in [f"{p}_median" for p in PARAMS] + [f"nlls_{p}" for p in PARAMS] + \
             ["baseline_cv", "n_vox"] + [f"{p}_prior_dominated" for p in PARAMS]:
        if c in sub:
            agg[c] = "median" if not c.endswith("_prior_dominated") else "mean"
    a = sub.groupby("subject_id").agg(agg)
    B = df[df["diagnosis"] == "B"]
    Mf = df[(df["diagnosis"] == "M") & df["subject_id"].isin(a.index)].set_index("subject_id")
    Ms = a.loc[Mf.index]
    print(f"\n6. SIZE-MATCHED SUBSAMPLE: {len(B)} benign, {len(Mf)} malignant "
          f"(full and subsampled)")
    rng = np.random.default_rng(SEED)
    rows = []
    for label, fmt in (("bayes", "{p}_median"), ("nlls", "nlls_{p}")):
        for p in PARAMS:
            c = fmt.format(p=p)
            if c not in B or c not in Ms:
                continue
            b = B[c].astype(float).to_numpy()
            mf = Mf[c].astype(float).to_numpy()
            ms = Ms[c].astype(float).to_numpy()
            ok = np.isfinite(mf) & np.isfinite(ms)
            b, mf, ms = b[np.isfinite(b)], mf[ok], ms[ok]
            af, asb = _auc_fast(b, mf), _auc_fast(b, ms)
            boot = []
            for _ in range(N_BOOT):
                ib = rng.integers(0, b.size, b.size)
                im = rng.integers(0, mf.size, mf.size)
                boot.append(abs(_auc_fast(b[ib], ms[im]) - 0.5)
                            - abs(_auc_fast(b[ib], mf[im]) - 0.5))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            shift = (np.median(np.log(ms / mf)) if p in LOG_PARAMS and np.all(mf > 0)
                     and np.all(ms > 0) else np.median(ms - mf))
            r = dict(estimate=label, param=p, auc_B_vs_Mfull=af, auc_B_vs_Msub=asb,
                     change_in_abs_auc_minus_half=abs(asb - 0.5) - abs(af - 0.5),
                     change_ci2_5=lo, change_ci97_5=hi,
                     median_B=float(np.median(b)), median_Mfull=float(np.median(mf)),
                     median_Msub=float(np.median(ms)),
                     median_lesion_shift=float(shift),
                     shift_is_log_ratio=p in LOG_PARAMS)
            if label == "bayes" and f"{p}_prior_dominated" in B:
                r.update(prior_dom_B=float(B[f"{p}_prior_dominated"].mean()),
                         prior_dom_Mfull=float(Mf[f"{p}_prior_dominated"].mean()),
                         prior_dom_Msub=float(Ms[f"{p}_prior_dominated"].mean()))
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "6_subsample_comparison.csv"), index=False)

    snr = dict(baseline_cv_B=float(B["baseline_cv"].median()),
               baseline_cv_Mfull=float(Mf["baseline_cv"].median()),
               baseline_cv_Msub=float(Ms["baseline_cv"].median()),
               snr_match_p_B_vs_Msub=float(stats.mannwhitneyu(
                   B["baseline_cv"], Ms["baseline_cv"]).pvalue),
               n_vox_B=float(B[SIZE_COL].median()), n_vox_Mfull=float(Mf[SIZE_COL].median()),
               n_vox_Msub=float(Ms["n_vox"].median()))
    pd.DataFrame([snr]).to_csv(os.path.join(out, "6_subsample_snr_check.csv"), index=False)
    print(f"  SNR check, median baseline CV: B {snr['baseline_cv_B']:.4f}  "
          f"M full {snr['baseline_cv_Mfull']:.4f}  M sub {snr['baseline_cv_Msub']:.4f}  "
          f"(B vs M sub p={snr['snr_match_p_B_vs_Msub']:.2g}; a small p means the "
          f"SNR was NOT matched)")
    print("  change in |AUC - 0.5| after subsampling (negative = moves toward 0.5, "
          "i.e. toward a size/SNR artifact):")
    print(t[["estimate", "param", "auc_B_vs_Mfull", "auc_B_vs_Msub",
             "change_in_abs_auc_minus_half", "change_ci2_5", "change_ci97_5"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    fig_subsample(B, Mf, Ms, os.path.join(out, "6_subsample_values.png"))
    fig_subsample_size(B, Mf, Ms, os.path.join(out, "6_subsample_sizes.png"))
    return t, snr


def fig_subsample(B, Mf, Ms, path):
    ps = [p for p in ("ve", "vp", "fp", "ps", "E", "Ktrans") if f"{p}_median" in Ms]
    fig, axes = plt.subplots(1, len(ps), figsize=(3.2 * len(ps), 3.8), squeeze=False)
    for ax, p in zip(axes[0], ps):
        c = f"{p}_median"
        data = [B[c].dropna(), Mf[c].dropna(), Ms[c].dropna()]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], (C_GROUP["B"], C_GROUP["M"], C_GROUP["Msub"])):
            patch.set_facecolor(col)
            patch.set_alpha(0.45)
        for k, v in enumerate(data):
            ax.scatter(np.full(v.size, k + 1) + np.random.default_rng(k).uniform(-0.15, 0.15, v.size),
                       v, s=8, color="k", alpha=0.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["B", "M full", "M sub"])
        ax.set_title(f"{p} ({UNITS[p]})" if p in UNITS else p)
        ax.grid(alpha=0.3)
    fig.suptitle("Posterior medians: benign vs malignant (full ROI) vs malignant "
                 "subsampled to benign size", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_subsample_size(B, Mf, Ms, path):
    data = [B[SIZE_COL].astype(float).to_numpy(),
            Mf[SIZE_COL].astype(float).to_numpy(),
            Ms["n_vox"].astype(float).to_numpy()]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    _box_panel(ax, data, [C_GROUP["B"], C_GROUP["M"], C_GROUP["Msub"]],
               [f"B\nn={len(data[0])}", f"M full\nn={len(data[1])}",
                f"M sub\nn={len(data[2])}"])
    ax.set_yscale("log")
    ax.set_ylabel("ROI voxels")
    ax.set_title("ROI size: benign vs malignant full vs subsampled", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# section 7: malignant subtype comparisons
# =============================================================================

def load_groups(df):
    """Per-lesion group label: 'B', or the malignant subtype from SUBTYPE_CSV."""
    t = pd.read_csv(SUBTYPE_CSV)
    t["Patient_ID"] = t["Patient_ID"].astype(str).str.strip()
    t["Diagnosis"] = t["Diagnosis"].astype(str).str.strip().str.upper()
    t["Malignant_lesion_type"] = (t["Malignant_lesion_type"].fillna("")
                                  .astype(str).str.strip().str.upper())
    sub = dict(zip(t["Patient_ID"], t["Malignant_lesion_type"]))
    dx = dict(zip(t["Patient_ID"], t["Diagnosis"]))
    grp, clash = [], []
    for sid, d in zip(df["subject_id"], df["diagnosis"]):
        if dx.get(sid, d) != d:
            clash.append(sid)
        grp.append("B" if d == "B" else (sub.get(sid, "") or "UNKNOWN"))
    if clash:
        print(f"  WARNING: Diagnosis in {os.path.basename(SUBTYPE_CSV)} disagrees with "
              f"the results for {clash[:8]}{'...' if len(clash) > 8 else ''}")
    return np.array(grp)


def section_subtypes(df, logs, out, order_cfg=SUBTYPE_ORDER, merge=None, suffix=""):
    """merge: optional {subtype: merged_label}; suffix is appended to every output
    file name so the merged run does not overwrite the per-subtype one."""
    grp = load_groups(df)
    if merge:
        grp = np.array([merge.get(g, g) for g in grp])
    n_unk = int((grp == "UNKNOWN").sum())
    if n_unk:
        print(f"  WARNING: {n_unk} malignant lesions have no subtype and are left out "
              f"of section 7")
    order = [g for g in order_cfg if (grp == g).sum() > 0]
    extra = sorted(set(grp) - set(order) - {"UNKNOWN"})
    if extra:
        print(f"  WARNING: groups not in the order list are left out: {extra}")
    counts = pd.DataFrame({"group": order, "n": [int((grp == g).sum()) for g in order],
                           "median_vox": [float(np.median(np.exp(logs[grp == g])))
                                          for g in order]})
    counts.to_csv(os.path.join(out, f"7_subtype_counts{suffix}.csv"), index=False)
    print(f"\n7. SUBTYPES{suffix}: lesions per group (after the SUBSET filter)")
    print(counts.to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    # descriptive tumor-volume box plot (merged run only: B vs DCIS vs Invasive)
    if merge:
        fig_box_volume(df, grp, order, C_SUB,
                       "Tumor volume: benign vs DCIS vs invasive (IDC, ILC, IDC/ILC)",
                       os.path.join(out, f"box_volume_subtypes{suffix}.png"))

    # descriptive RMSE box plots by subtype (no tests)
    for label in ("bayes", "nlls"):
        fig_box_rmse(df, grp, order, C_SUB, label,
                     os.path.join(out, f"box_rmse_subtypes{suffix}_{label}.png"))

    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    res = {}
    for label, fmt in (("bayes", "{p}_median"), ("nlls", "nlls_{p}")):
        cols = {p: fmt.format(p=p) for p in PARAMS if fmt.format(p=p) in df}
        if not cols:
            continue
        vals = {p: df[c].astype(float).to_numpy() for p, c in cols.items()}
        fam = [p for p in FAMILY if p in vals]

        # omnibus: does the parameter differ anywhere among the groups?
        om = []
        for p, v in vals.items():
            samp = [v[grp == g][np.isfinite(v[grp == g])] for g in order]
            ok = [s for s in samp if s.size >= 2]
            H_, pk = stats.kruskal(*ok) if len(ok) >= 2 else (np.nan, np.nan)
            om.append(dict(param=p, kw_H=H_, kw_p=pk,
                           **{f"median_{g}": float(np.median(s)) if s.size else np.nan
                              for g, s in zip(order, samp)}))
        om = pd.DataFrame(om)
        om["kw_p_holm_params"] = np.nan
        f_om = om["param"].isin(fam)
        om.loc[f_om, "kw_p_holm_params"] = G.holm(om.loc[f_om, "kw_p"])
        om.insert(0, "estimate", label)
        om.to_csv(os.path.join(out, f"7_subtype_omnibus{suffix}_{label}.csv"), index=False)

        # every pair, same statistics as section 2
        rows = []
        for a, b in pairs:
            sel = (grp == a) | (grp == b)
            na, nb = int((grp == a).sum()), int((grp == b).sum())
            if min(na, nb) < SUBTYPE_MIN_N:
                print(f"  skipped {a} vs {b} ({label}): n = {na}, {nb} "
                      f"(< SUBTYPE_MIN_N = {SUBTYPE_MIN_N})")
                continue
            yy = (grp[sel] == b).astype(int)          # 1 = the second group
            t = G.discrimination_table({p: v[sel] for p, v in vals.items()}, yy,
                                       logs[sel], log_params=LOG_PARAMS, family=fam)
            t = t.drop(columns=[c for c in t.columns if c.endswith("_bh")])
            t = t.rename(columns={"n_B": "n_ref", "n_M": "n_cmp",
                                  "median_B": "median_ref", "median_M": "median_cmp",
                                  "mw_p_holm": "mw_p_holm_params",
                                  "sizeadj_lr_p_holm": "sizeadj_lr_p_holm_params"})
            t.insert(0, "small_n", min(na, nb) < SUBTYPE_SMALL_N)
            t.insert(0, "cmp", b)
            t.insert(0, "ref", a)
            t.insert(0, "kind", "vs_benign" if a == "B" else "between_malignant")
            rows.append(t)
        if not rows:
            continue
        long = pd.concat(rows, ignore_index=True)
        # Holm across the pairs, separately for each parameter
        for col in ("mw_p", "sizeadj_lr_p"):
            long[f"{col}_holm_pairs"] = np.nan
            for p in fam:
                idx = (long["param"] == p).to_numpy()
                long.loc[idx, f"{col}_holm_pairs"] = G.holm(long.loc[idx, col].to_numpy())
            # Holm over EVERY (pair x parameter) test in the family at once
            long[f"{col}_holm_all"] = np.nan
            ia = long["param"].isin(fam).to_numpy()
            long.loc[ia, f"{col}_holm_all"] = G.holm(long.loc[ia, col].to_numpy())
        long.insert(0, "estimate", label)
        long.to_csv(os.path.join(out, f"7_subtype_pairwise{suffix}_{label}.csv"), index=False)
        res[label] = (om, long)
        fig_box_groups(df, grp, order, label, fmt, long,
                       os.path.join(out, f"box_subtypes{suffix}_{label}.png"))

        print(f"\n7. SUBTYPES ({label}): omnibus Kruskal-Wallis across "
              f"{', '.join(order)}")
        print(om[["param", "kw_p", "kw_p_holm_params"]].to_string(
            index=False, float_format=lambda v: f"{v:.3g}"))
        hit = long[(long["mw_p"] < 0.05) & long["param"].isin(fam)]
        print(f"  pairwise rows with raw Mann-Whitney p < 0.05 "
              f"({len(hit)} of {int(long['param'].isin(fam).sum())}; AUC > 0.5 = "
              f"'cmp' higher than 'ref'; full table in 7_subtype_pairwise_{label}.csv):")
        if len(hit):
            print(hit[["ref", "cmp", "param", "n_ref", "n_cmp", "auc", "auc_ci2_5",
                       "auc_ci97_5", "mw_p", "mw_p_holm_pairs", "mw_p_holm_params",
                       "mw_p_holm_all", "small_n"]].to_string(index=False,
                                             float_format=lambda v: f"{v:.3g}"))
    fig_subtypes(df, grp, order, os.path.join(out, f"7_subtype_values{suffix}.png"))
    return counts, res


def fig_subtypes(df, grp, order, path):
    ps = [p for p in ("ve", "vp", "fp", "ps", "E", "Ktrans") if f"{p}_median" in df]
    fig, axes = plt.subplots(1, len(ps), figsize=(3.4 * len(ps), 4.0), squeeze=False)
    for ax, p in zip(axes[0], ps):
        c = f"{p}_median"
        data = [df.loc[grp == g, c].astype(float).dropna() for g in order]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(C_SUB.get(g, "0.6"))
            patch.set_alpha(0.45)
        for k, v in enumerate(data):
            ax.scatter(np.full(v.size, k + 1)
                       + np.random.default_rng(k).uniform(-0.15, 0.15, v.size),
                       v, s=8, color="k", alpha=0.5)
        if p in LOG_PARAMS:
            ax.set_yscale("log")
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels([f"{g}\nn={len(d)}" for g, d in zip(order, data)], fontsize=8)
        ax.set_title(f"{p} ({UNITS[p]})" if p in UNITS else p)
        ax.grid(alpha=0.3)
    fig.suptitle("Posterior medians by group (benign and malignant subtypes)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# section 8: overlapping size band -- MATCHED SUBSET, no refitting
# =============================================================================
# Section 6 removes the size/SNR advantage by refitting the malignant lesions
# smaller. This section instead KEEPS ONLY the lesions whose volume falls in a
# band where the two groups overlap, and reruns section 2's statistics there.
# No refitting, so it costs nothing -- but it drops most lesions, and the
# malignant ones that survive are the small ones, which may differ
# biologically from the rest (that selection is the method's weak point, and
# is why section 6 is still worth running).
#
# Read, in this order:
#   auc_size_in_band     near 0.5 = size really is matched inside the band;
#                        well above 0.5 = size is still confounded, narrow it
#   baseline_cv_p        small = the SNR was NOT matched even though the
#                        volumes were (voxel count is only an SNR proxy)
#   then the parameter AUCs

def _band_mask(df, band_mm3):
    """Boolean mask of lesions whose fitted-ROI volume is inside band_mm3."""
    vol = df[SIZE_COL].astype(float).to_numpy() * VOXEL_MM3
    return (vol >= band_mm3[0]) & (vol <= band_mm3[1]), vol


def section_size_band(df, y, logs, out):
    # --- sweep: n per group and the residual size AUC for every candidate -----
    rows = []
    for lo_mm, hi_mm in BAND_CANDIDATES:
        m, vol = _band_mask(df, (lo_mm, hi_mm))
        nB, nM = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        if min(nB, nM) >= 2:
            a, p = G.auc_mw(vol[m & (y == 0)], vol[m & (y == 1)])
        else:
            a, p = np.nan, np.nan
        rows.append(dict(band_lo_mm3=lo_mm, band_hi_mm3=hi_mm, n_B=nB, n_M=nM,
                         median_mm3_B=float(np.median(vol[m & (y == 0)])) if nB else np.nan,
                         median_mm3_M=float(np.median(vol[m & (y == 1)])) if nM else np.nan,
                         auc_size_in_band=a, size_mw_p=p,
                         testable=bool(min(nB, nM) >= BAND_MIN_N),
                         is_selected=(lo_mm, hi_mm) == tuple(BAND_MM3)))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out, "8_size_band_sweep.csv"), index=False)
    print("\n8. SIZE BAND -- candidate bands (descriptive; pick on size grounds "
          "BEFORE looking at parameters)")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # --- the selected band ----------------------------------------------------
    m, vol = _band_mask(df, BAND_MM3)
    nB, nM = int((y[m] == 0).sum()), int((y[m] == 1).sum())
    print(f"\n8. SIZE BAND {BAND_MM3[0]:.0f}-{BAND_MM3[1]:.0f} mm^3: {nB} B, {nM} M "
          f"kept (of {int((y == 0).sum())} B, {int(y.sum())} M)")
    if min(nB, nM) < BAND_MIN_N:
        print(f"  too few lesions in the band (< BAND_MIN_N = {BAND_MIN_N}); "
              f"no tests run")
        return sweep, None, {}

    auc_s, p_s = G.auc_mw(vol[m & (y == 0)], vol[m & (y == 1)])
    lo_s, hi_s, _ = G.delong_ci(vol[m & (y == 0)], vol[m & (y == 1)])
    cv = df["baseline_cv"].astype(float).to_numpy()
    p_cv = float(stats.mannwhitneyu(cv[m & (y == 0)], cv[m & (y == 1)]).pvalue)
    check = pd.DataFrame([dict(
        band_lo_mm3=BAND_MM3[0], band_hi_mm3=BAND_MM3[1], subset=SUBSET,
        n_B=nB, n_M=nM, n_B_all=int((y == 0).sum()), n_M_all=int(y.sum()),
        median_mm3_B=float(np.median(vol[m & (y == 0)])),
        median_mm3_M=float(np.median(vol[m & (y == 1)])),
        iqr_mm3_B=f"{np.percentile(vol[m & (y == 0)], 25):.0f}-"
                  f"{np.percentile(vol[m & (y == 0)], 75):.0f}",
        iqr_mm3_M=f"{np.percentile(vol[m & (y == 1)], 25):.0f}-"
                  f"{np.percentile(vol[m & (y == 1)], 75):.0f}",
        auc_size_in_band=auc_s, auc_ci2_5=lo_s, auc_ci97_5=hi_s, size_mw_p=p_s,
        baseline_cv_B=float(np.median(cv[m & (y == 0)])),
        baseline_cv_M=float(np.median(cv[m & (y == 1)])), baseline_cv_p=p_cv)])
    check.to_csv(os.path.join(out, "8_size_band_check.csv"), index=False)
    print(f"  median volume B {check['median_mm3_B'][0]:.0f} vs M "
          f"{check['median_mm3_M'][0]:.0f} mm^3; AUC of size WITHIN the band "
          f"{auc_s:.3f} [{lo_s:.3f}, {hi_s:.3f}], p={p_s:.2g} "
          f"(near 0.5 = size matched)")
    print(f"  median baseline CV B {check['baseline_cv_B'][0]:.4f} vs M "
          f"{check['baseline_cv_M'][0]:.4f} (p={p_cv:.2g}; a small p means the SNR "
          f"was NOT matched even though the volumes were)")

    # --- per-lesion CSV of what the band kept ---------------------------------
    keep_cols = ["subject_id", "diagnosis", SIZE_COL, "baseline_cv",
                 "peak_enhancement", "qc_pass"]
    lesions = df[[c for c in keep_cols if c in df]].copy()
    lesions["volume_mm3"] = vol
    lesions["in_band"] = m
    for p in PARAMS:
        if f"{p}_median" in df:
            lesions[f"{p}_median"] = df[f"{p}_median"].to_numpy()
    lesions.sort_values(["in_band", "diagnosis", "volume_mm3"],
                        ascending=[False, True, True]).to_csv(
        os.path.join(out, "8_size_band_lesions.csv"), index=False)

    # --- volume box plot: whole cohort next to the band ------------------------
    fig_box_volume_band(df, y, m, os.path.join(out, "8_box_volume_band.png"))

    # --- section 2's statistics, inside the band ------------------------------
    tabs = {}
    for label, fmt in (("bayes", "{p}_median"), ("nlls", "nlls_{p}")):
        cols = {p: fmt.format(p=p) for p in PARAMS if fmt.format(p=p) in df}
        if not cols:
            continue
        vals = {p: df[c].astype(float).to_numpy()[m] for p, c in cols.items()}
        t = G.discrimination_table(vals, y[m], logs[m], log_params=LOG_PARAMS,
                                   family=[p for p in FAMILY if p in vals])
        t.insert(0, "band_hi_mm3", BAND_MM3[1])
        t.insert(0, "band_lo_mm3", BAND_MM3[0])
        t.insert(0, "estimate", label)
        t.to_csv(os.path.join(out, f"8_size_band_discrimination_{label}.csv"),
                 index=False)
        tabs[label] = t
        print(f"\n8. SIZE BAND ({label}; AUC>0.5 = malignant higher; compare with "
              f"section 2 on the whole cohort)")
        print(t[["param", "n_B", "n_M", "median_B", "median_M", "auc", "auc_ci2_5",
                 "auc_ci97_5", "mw_p", "mw_p_holm", "sizeadj_or_per_sd",
                 "sizeadj_lr_p"]].to_string(index=False,
                                            float_format=lambda v: f"{v:.3g}"))
        fig_box_BM(df[m].reset_index(drop=True), y[m], label, fmt, t,
                   os.path.join(out, f"8_box_size_band_{label}.png"))
    return sweep, check, tabs


def fig_box_volume_band(df, y, m, path):
    """Tumor volume (mm^3): the whole cohort and the in-band subset, side by
    side, with the band drawn in. Descriptive only."""
    vol = df[SIZE_COL].astype(float).to_numpy() * VOXEL_MM3
    data = [vol[y == 0], vol[y == 1], vol[m & (y == 0)], vol[m & (y == 1)]]
    labels = [f"B all\nn={len(data[0])}", f"M all\nn={len(data[1])}",
              f"B in band\nn={len(data[2])}", f"M in band\nn={len(data[3])}"]
    colors = [C_GROUP["B"], C_GROUP["M"], C_GROUP["B"], C_GROUP["M"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _box_panel(ax, data, colors, labels)
    ax.axhspan(BAND_MM3[0], BAND_MM3[1], color="0.5", alpha=0.15, zorder=0)
    for yv in BAND_MM3:
        ax.axhline(yv, color="0.35", lw=0.9, ls="--", zorder=1)
    if VOLUME_LOG:
        ax.set_yscale("log")
    ax.axvline(2.5, color="0.3", lw=0.8)
    ax.set_ylabel("Tumor volume (mm$^3$)")
    ax.set_title(f"Tumor volume: whole cohort vs the "
                 f"{BAND_MM3[0]:.0f}-{BAND_MM3[1]:.0f} mm$^3$ band", fontsize=11)
    for k, d in enumerate(data):
        if len(d):
            print(f"  volume {labels[k].splitlines()[0]:>10s}: n={len(d)}  "
                  f"median={np.median(d):.0f} mm^3  "
                  f"IQR={np.percentile(d, 25):.0f}-{np.percentile(d, 75):.0f}")
    fig.text(0.5, 0.005, "box = IQR, line = median, whiskers = 1.5 x IQR, dots = "
             "lesions; shaded = the band; descriptive only", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# box-and-whisker figures: B vs M, and every group vs every other group
# =============================================================================
# Box = IQR (25th-75th percentile), line = median, whiskers = most extreme point
# within 1.5 x IQR; every lesion is overlaid as a dot (so nothing is hidden).

BOX_PARAMS     = ("ve", "vp", "fp", "ps", "tau_delay")
BOX_LOG_PARAMS = ()                 # e.g. ("fp", "ps") for a log y-axis
BOX_UNITS      = UNITS                                      # data already in min^-1
EST_TITLE      = {"bayes": "Bayes posterior median", "nlls": "NLLS"}


def _box_panel(ax, data, colors, labels):
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, whis=1.5,
                    medianprops=dict(color="k", lw=2.0))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for k, v in enumerate(data):
        ax.scatter(np.full(len(v), k + 1)
                   + np.random.default_rng(k).uniform(-0.16, 0.16, len(v)),
                   v, s=9, color="k", alpha=0.45, zorder=3)
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def fig_box_BM(df, y, label, fmt, tab, path):
    """One figure, one panel per parameter: benign vs all malignant."""
    ps = [p for p in BOX_PARAMS if fmt.format(p=p) in df]
    if not ps:
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(3.6 * len(ps), 4.4), squeeze=False)
    for ax, p in zip(axes[0], ps):
        v = df[fmt.format(p=p)].astype(float).to_numpy()
        b, m = v[y == 0], v[y == 1]
        b, m = b[np.isfinite(b)], m[np.isfinite(m)]
        _box_panel(ax, [b, m], [C_GROUP["B"], C_GROUP["M"]],
                   [f"Benign\nn={b.size}", f"Malignant\nn={m.size}"])
        if p in BOX_LOG_PARAMS:
            ax.set_yscale("log")
        else:                                   # headroom so the label clears the dots
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.12 * (hi - lo))
        ax.set_title(f"{p} ({BOX_UNITS[p]})" if p in BOX_UNITS else p, fontsize=11)
        if tab is not None and p in set(tab["param"]):
            r = tab[tab["param"] == p].iloc[0]
            ph = r["mw_p_holm"] if np.isfinite(r.get("mw_p_holm", np.nan)) else r["mw_p"]
            kind = "Holm" if np.isfinite(r.get("mw_p_holm", np.nan)) else "raw"
            ax.text(0.5, 0.985, f"AUC {r['auc']:.2f}   p = {ph:.2g} ({kind})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    fig.suptitle(f"{EST_TITLE.get(label, label)}: benign vs malignant   "
                 f"(box = IQR, line = median, whiskers = 1.5 x IQR, dots = lesions; "
                 f"Mann-Whitney p, Holm across parameters)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_box_groups(df, grp, order, label, fmt, long, path):
    """One figure, one panel per parameter: B and every subtype side by side, with
    brackets on every pair whose Holm-across-pairs Mann-Whitney p is < 0.05."""
    ps = [p for p in BOX_PARAMS if fmt.format(p=p) in df]
    if not ps:
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(4.4 * len(ps), 5.4), squeeze=False)
    pos = {g: i + 1 for i, g in enumerate(order)}
    for ax, p in zip(axes[0], ps):
        v = df[fmt.format(p=p)].astype(float).to_numpy()
        data = [v[(grp == g) & np.isfinite(v)] for g in order]
        _box_panel(ax, data, [C_SUB.get(g, "0.6") for g in order],
                   [f"{g}\nn={len(d)}" for g, d in zip(order, data)])
        if p in BOX_LOG_PARAMS:
            ax.set_yscale("log")
        ax.set_title(f"{p} ({BOX_UNITS[p]})" if p in BOX_UNITS else p, fontsize=11)
        sig = long[(long["param"] == p) & (long["mw_p_holm_pairs"] < 0.05)].copy()
        sig["span"] = [abs(pos[b] - pos[a]) for a, b in zip(sig["ref"], sig["cmp"])]
        sig = sig.sort_values(["span", "ref"])
        lo, hi = ax.get_ylim()
        top = max([float(np.max(d)) for d in data if len(d)] + [hi])
        step = 0.075 * (top - lo)
        for k, (_, r) in enumerate(sig.iterrows()):
            yb = top + (k + 0.6) * step
            x1, x2 = pos[r["ref"]], pos[r["cmp"]]
            ax.plot([x1, x1, x2, x2], [yb - 0.25 * step, yb, yb, yb - 0.25 * step],
                    color="k", lw=0.9)
            ax.text((x1 + x2) / 2, yb, _stars(r["mw_p_holm_pairs"]), ha="center",
                    va="bottom", fontsize=10)
        if len(sig):
            ax.set_ylim(lo, top + (len(sig) + 1.2) * step)
    fig.suptitle(f"{EST_TITLE.get(label, label)}: benign and each malignant subtype "
                 f"(box = IQR, line = median, whiskers = 1.5 x IQR, dots = lesions). "
                 f"Brackets: pairs with Holm-adjusted (across the pairs) Mann-Whitney "
                 f"p < 0.05; * <0.05  ** <0.01  *** <0.001",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# fit RMSE by group -- DESCRIPTIVE ONLY (not in PARAMS/FAMILY; no tests)
# =============================================================================
# rmse is in signal units (scales with each lesion's S0 / enhancement);
# rmse_over_baseline_sd is relative to that subject's noise (~1 = residual at
# the noise floor), so it is the one that is comparable across lesions.

RMSE_COLS = {"bayes": ("rmse", "rmse_over_baseline_sd"),
             "nlls":  ("nlls_rmse", "nlls_rmse_over_baseline_sd")}
RMSE_TITLE = {"rmse": "RMSE (signal units)",
              "rmse_over_baseline_sd": "RMSE / baseline SD",
              "nlls_rmse": "RMSE (signal units)",
              "nlls_rmse_over_baseline_sd": "RMSE / baseline SD"}


def fig_box_rmse(df, groups, order, colors, label, path):
    """Descriptive box plots of fit RMSE by group. No tests, no p-values."""
    cols = [c for c in RMSE_COLS[label] if c in df]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols) + 1, 4.8),
                             squeeze=False)
    for ax, c in zip(axes[0], cols):
        v = df[c].astype(float).to_numpy()
        data = [v[(groups == g) & np.isfinite(v)] for g in order]
        _box_panel(ax, data, [colors.get(g, "0.6") for g in order],
                   [f"{g}\nn={len(d)}" for g, d in zip(order, data)])
        if c.endswith("over_baseline_sd"):
            ax.axhline(1.0, color="crimson", lw=0.8, ls="--")   # residual = noise
        ax.set_title(RMSE_TITLE[c], fontsize=11)
    fig.suptitle(f"{EST_TITLE.get(label, label)}: fit RMSE by group "
                 f"(box = IQR, line = median, whiskers = 1.5 x IQR, dots = lesions; "
                 f"descriptive only)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# tumor volume by group -- DESCRIPTIVE ONLY (the size test is section 1)
# =============================================================================

def fig_box_volume(df, groups, order, colors, title, path):
    """Box plot of fitted-ROI volume (n_vox * VOXEL_MM3, mm^3) by group."""
    vol = df[SIZE_COL].astype(float).to_numpy() * VOXEL_MM3
    data = [vol[(groups == g) & np.isfinite(vol)] for g in order]
    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2.5, 4.8))
    _box_panel(ax, data, [colors.get(g, "0.6") for g in order],
               [f"{g}\nn={len(d)}" for g, d in zip(order, data)])
    if VOLUME_LOG:
        ax.set_yscale("log")
    ax.set_ylabel("Tumor volume (mm$^3$)")
    ax.set_title(title, fontsize=11)
    for k, d in enumerate(data):
        if len(d):
            print(f"  volume {order[k]:>9s}: n={len(d)}  median={np.median(d):.0f} mm^3  "
                  f"IQR={np.percentile(d, 25):.0f}-{np.percentile(d, 75):.0f}")
    fig.text(0.5, 0.005, "box = IQR, line = median, whiskers = 1.5 x IQR, dots = lesions; "
             "descriptive only", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    out = OUT_DIR or os.path.join(RUN_DIR, "group_stats_" +
                                  datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"results -> {out}")
    cfg = {k: v for k, v in globals().items()
           if k.isupper() and k not in ("C_GROUP",)}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    df = load_results(RUN_DIR, SUBSET)
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    logs = np.log(df[SIZE_COL].astype(float).to_numpy())
    print(f"{len(df)} lesions ({int((y == 0).sum())} B, {int(y.sum())} M), subset={SUBSET}")

    res = dict(out_dir=out)
    res["size"] = section_size(df, y, out)
    print("\nTUMOR VOLUME (n_vox x VOXEL_MM3)")
    fig_box_volume(df, df["diagnosis"].to_numpy(), ("B", "M"), C_GROUP,
                   "Tumor volume: benign vs malignant",
                   os.path.join(out, "box_volume_B_vs_M.png"))
    res["discrimination"] = section_discrimination(df, y, logs, out)
    for label, fmt in (("bayes", "{p}_median"), ("nlls", "nlls_{p}")):
        fig_box_BM(df, y, label, fmt, res["discrimination"].get(label),
                   os.path.join(out, f"box_B_vs_M_{label}.png"))
        fig_box_rmse(df, df["diagnosis"].to_numpy(), ("B", "M"), C_GROUP, label,
                     os.path.join(out, f"box_rmse_B_vs_M_{label}.png"))
    res["blocks"] = section_blocks(df, y, logs, out)
    res["identifiability"] = section_identifiability(df, y, logs, out)
    if RUN_PRIOR_SENS:
        res["prior_auc"], res["prior_summary"], res["order_match"] = \
            section_prior_sensitivity(df, y, logs, out)
    if SUBSAMPLE_RUN_DIR:
        res["subsample"], res["snr"] = section_subsample(df, out)
    if SUBTYPE_CSV:
        res["subtypes"] = section_subtypes(df, logs, out)
        res["subtypes_merged"] = section_subtypes(df, logs, out, order_cfg=MERGED_ORDER,
                                                  merge=SUBTYPE_MERGE, suffix="_merged")
    if BAND_MM3:
        res["band_sweep"], res["band_check"], res["band"] = \
            section_size_band(df, y, logs, out)
    print(f"\nresults written to: {out}")
    return res


if __name__ == "__main__":
    main()