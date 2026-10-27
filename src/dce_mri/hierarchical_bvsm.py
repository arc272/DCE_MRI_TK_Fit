#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
hierarchical_bvsm.py
====================
Two-stage hierarchical Bayesian benign-vs-malignant comparison on the
per-tumor posteriors of a fit_lesions_bayes.py run. Library code lives in
src/dce_mri/hierarchical.py (method and assumptions documented there).

Steps
  1. load each tumor's draws ({sid}_draws.npz) -- QC-pass tumors by default
  2. rebuild the EXACT per-tumor prior the fit used (PRIOR_CHOICES,
     DELAY_BOUNDS below must match that run)
  3. STAGE-1 DIAGNOSTIC: within- vs between-tumor spread per parameter and
     group -- is pooling likely to help at all?
     (set RUN_HIERARCHICAL = False to stop here)
  4. sample the hyperparameters (NUTS) under one or both likelihoods:
       "gaussian" -- per-tumor Gaussian summary with the prior divided out
                     analytically; no Monte Carlo noise, fast; approximate
                     where stage-1 posteriors are non-Gaussian
       "mc"       -- importance reweighting of the raw draws; exact in the
                     limit of many draws, but noisy for prior-dominated
                     tumors (Pareto k-hat > 0.7) and expensive
       "both"     -- run both and report where they agree (recommended)
  5. per-tumor shrunken estimates and diagnostics
  6. CSVs + figures

Outputs, in OUT_DIR ({m} = gaussian / mc):
  stage1_within_between.csv / .png   step 3
  hier_group_summary_{m}.csv         population medians B / M, M-B, ratio,
                                     z-scale delta with CI and P(delta > 0)
  hier_hyper_draws_{m}.npz           hyperparameter chains + names
  hier_hyper_diagnostics_{m}.csv     R-hat / ESS per hyperparameter
  hier_tumor_estimates_{m}.csv       stage-1 vs hierarchical medians per
                                     tumor + per-tumor flags
  hier_group_effects_{m}.png         population medians, B vs M
  hier_shrinkage_{m}.png             stage-1 vs hierarchical tumor medians
  hier_khat.png                      per-tumor k-hat (mc only)
  hier_method_comparison.csv         gaussian vs mc, per hyperparameter (both)
  hier_config.json                   every setting used

Run:
    PYTHONPATH=src python scripts/hierarchical_bvsm.py
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dce_mri.config import AcquisitionConfig
from dce_mri import hierarchical as H

# =============================================================================
# settings
# =============================================================================

RUN_DIR = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/run_20260929_141445"
OUT_DIR = None            # None -> RUN_DIR/hierarchical_<timestamp>

# --- must match fit_lesions_bayes.py for the run being analysed --------------
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS  = (-10.0, 30.0)
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))

# --- which tumors / how many draws ---------------------------------------------
SUBSET            = "qc_pass"   # "qc_pass" or "all" (all = non-error rows)
N_DRAWS_PER_TUMOR = 8000        # random subset of each tumor's pooled chains;
                                # None = all. The mc likelihood's noise falls
                                # with more draws but its cost rises with them,
                                # and evaluating p0 at every draw (~180 us
                                # each) is a one-off ~4 min per 230 x 8000.
DRAW_SEED         = 0

RUN_HIERARCHICAL = True         # False: stop after the stage-1 diagnostic
LIKELIHOOD       = "both"       # "gaussian", "mc", or "both"
GAUSS_MIN_INFO   = 0.05         # z-scale precision below which a direction
                                # is treated as carrying no information
AGREE_TOL_SD     = 0.25         # gaussian vs mc: |mean diff| / posterior SD

# --- population model --------------------------------------------------------
# All five parameters are population-modelled (the importance ratio needs a
# population density for the whole joint draw). tau_delay is a per-patient
# nuisance: one shared population, no group effect.
PARAMS         = ("ve", "vp", "fp", "ps", "tau_delay")
GROUP_EFFECT   = {"ve": True, "vp": True, "fp": True, "ps": True, "tau_delay": False}
SEPARATE_SIGMA = {"ve": True, "vp": True, "fp": True, "ps": True, "tau_delay": False}
FAMILY, NU     = "normal", 5.0  # "student_t" for heavier-tailed populations
HYPER = dict(mu_sd=2.0,         # Normal(0, .) on reference means (z-scale)
             delta_sd=1.0,      # Normal(0, .) on M - B differences (z-scale)
             sigma_scale=1.0,   # HalfNormal(.) on population SDs (z-scale)
             beta_sd=1.0)       # Normal(0, .) on covariate effects (z-scale)

# --- covariates (act on the group-effect parameters) -------------------------
COVARIATES = ()                 # e.g. ("aif_idif",) -- 1 if the AIF came from
                                # the IDIF ROI, 0 if from the aorta ROI
AIF_DIR = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"

# --- hyperparameter sampler --------------------------------------------------
N_WARMUP, N_DRAWS, N_CHAINS = 2000, 2000, 4
SEED, TARGET_ACCEPT, DENSE  = 1, 0.9, True

# --- diagnostics -------------------------------------------------------------
K_PHI_REWEIGHT  = 40       # posterior phi draws used for per-tumor weights
KHAT_BAD        = 0.7
RHAT_MAX        = 1.01
ESS_MIN         = 400
PRIOR_Z_SAMPLES = 200_000

C_GROUP = {"B": "#1B4F72", "M": "#B22234"}
UNITS = {"fp": "s$^{-1}$", "ps": "s$^{-1}$", "tau_delay": "s"}


# =============================================================================
# covariates
# =============================================================================

def build_covariates(sids, malignant):
    if not COVARIATES:
        return None
    cols = []
    for cov in COVARIATES:
        if cov != "aif_idif":
            raise ValueError(f"unknown covariate {cov!r}")
        v = []
        for sid in sids:
            f = os.path.join(AIF_DIR, f"{sid}_AIF_plasma.csv")
            if not os.path.exists(f):
                raise FileNotFoundError(f"{sid}: no AIF CSV at {f}")
            d = pd.read_csv(f)
            if "aif_source" not in d:
                raise ValueError(f"{f} has no aif_source column "
                                 f"(made by extract_aifs.py, not extract_aif_roi.py?)")
            v.append(1.0 if str(d["aif_source"].iloc[0]).strip() == "idif" else 0.0)
        v = np.asarray(v)
        tab = pd.crosstab(pd.Series(np.where(malignant > 0.5, "M", "B"), name="dx"),
                          pd.Series(v.astype(int), name=cov))
        print(f"covariate {cov} by diagnosis:\n{tab}\n")
        if v.min() == v.max():
            raise ValueError(f"{cov} is constant across tumors -- drop it")
        if np.array_equal(v, malignant) or np.array_equal(v, 1 - malignant):
            raise ValueError(f"{cov} is identical to diagnosis -- the group "
                             f"effect would be unidentifiable")
        cols.append(v)
    return np.column_stack(cols)


# =============================================================================
# figures
# =============================================================================

def fig_stage1(df, path):
    params = list(dict.fromkeys(df["param"]))
    fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.8),
                             squeeze=False)
    for ax, p in zip(axes[0], params):
        d = df[df["param"] == p]
        for k, (_, r) in enumerate(d.iterrows()):
            c = C_GROUP[r["group"]]
            for x, col, mk in ((k - 0.12, "within_sd", "o"),
                               (k + 0.12, "between_sd", "s")):
                if col in r and np.isfinite(r.get(col, np.nan)):
                    ax.plot(x, r[col], mk, color=c, ms=8,
                            mfc=c if mk == "o" else "white")
            txt = (f"pred {r['pred_sd_ratio']:.2f}" if np.isfinite(r.get("pred_sd_ratio", np.nan))
                   else "n/a")
            ax.annotate(f"{txt}\nuninf {100 * r['frac_uninformative']:.0f}%",
                        (k, 0), xytext=(0, -38), textcoords="offset points",
                        ha="center", fontsize=7, annotation_clip=False)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["group"])
        ax.set_title(p)
        ax.set_ylabel("SD on population (z) scale")
        ax.grid(alpha=0.3)
    fig.suptitle("Stage-1 diagnostic: filled = within-tumor (likelihood) SD, "
                 "hollow = between-tumor SD\npred = predicted SD ratio of the "
                 "group mean, hierarchical / equal-weight (<1 = pooling helps)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.9])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_group_effects(chains, layout, spec, tr, path):
    flat = chains.reshape(-1, chains.shape[-1])
    a, d, _, _, _ = layout.unpack(flat)
    P = len(spec.param_names)
    fig, axes = plt.subplots(1, P, figsize=(3.6 * P, 3.6), squeeze=False)
    for j, (ax, p) in enumerate(zip(axes[0], spec.param_names)):
        mu_B = a[:, j]
        if j in layout.ge:
            delta = d[:, int(np.flatnonzero(layout.ge == j)[0])]
            series = (("B", tr.from_z_col(mu_B, j)),
                      ("M", tr.from_z_col(mu_B + delta, j)))
            lo, hi = np.percentile(delta, [2.5, 97.5])
            ttl = (f"{p}\n\u0394z 95% CI [{lo:.2f}, {hi:.2f}], "
                   f"P(\u0394>0)={np.mean(delta > 0):.2f}")
        else:
            series = (("all", tr.from_z_col(mu_B, j)),)
            ttl = f"{p} (shared, no group effect)"
        allv = np.concatenate([v for _, v in series])
        bins = np.linspace(*np.percentile(allv, [0.5, 99.5]), 50)
        for g, v in series:
            ax.hist(v, bins=bins, density=True, alpha=0.55,
                    color=C_GROUP.get(g, "0.4"), label=g)
        unit = f" ({UNITS[p]})" if p in UNITS else ""
        ax.set_xlabel(f"population median {p}{unit}")
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Hierarchical model: posterior of each group's population median",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_shrinkage(tt, params, path, flag_col=None):
    P = len(params)
    fig, axes = plt.subplots(1, P, figsize=(3.6 * P, 3.6), squeeze=False)
    bad = (tt[flag_col].to_numpy() > KHAT_BAD if flag_col
           else np.zeros(len(tt), bool))
    for ax, p in zip(axes[0], params):
        x = tt[f"{p}_stage1_median"].to_numpy()
        y = tt[f"{p}_hier_median"].to_numpy()
        for g in ("B", "M"):
            s = (tt["diagnosis"] == g).to_numpy()
            ax.scatter(x[s & ~bad], y[s & ~bad], s=18, color=C_GROUP[g], label=g)
            ax.scatter(x[s & bad], y[s & bad], s=18, facecolors="none",
                       edgecolors=C_GROUP[g])
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        unit = f" ({UNITS[p]})" if p in UNITS else ""
        ax.set_xlabel(f"stage-1 median{unit}")
        ax.set_ylabel(f"hierarchical median{unit}")
        ax.set_title(p)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Per-tumor shrinkage" + (f" (hollow = k-hat > {KHAT_BAD}: "
                 f"reweighted estimate unreliable)" if flag_col else ""), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_khat(tt, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    k = tt["khat_max"].to_numpy()
    ax.hist(k[np.isfinite(k)], bins=30, color="0.5")
    ax.axvline(KHAT_BAD, color="crimson", ls="--", label=f"{KHAT_BAD}")
    ax.set_xlabel(f"per-tumor k-hat (max over {K_PHI_REWEIGHT} phi draws)")
    ax.set_ylabel("tumors")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    out = OUT_DIR or os.path.join(
        RUN_DIR, "hierarchical_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"results -> {out}")
    cfg = {k: (v if not isinstance(v, (tuple, slice)) else str(v))
           for k, v in globals().items()
           if k.isupper() and not k.startswith("_") and k not in ("C_GROUP", "UNITS")}
    with open(os.path.join(out, "hier_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # --- 1. draws -------------------------------------------------------------
    t0 = time.time()
    sids, mal, draws, skipped = H.load_run_draws(
        RUN_DIR, PARAMS, N_DRAWS_PER_TUMOR, subset=SUBSET, seed=DRAW_SEED)
    print(f"{len(sids)} tumors ({int((mal == 0).sum())} B, {int(mal.sum())} M), "
          f"{N_DRAWS_PER_TUMOR} draws each, subset={SUBSET}")
    if len(skipped):
        print(f"skipped {len(skipped)}:\n{skipped.to_string(index=False)}")
    if (mal == 0).sum() < 3 or mal.sum() < 3:
        raise ValueError("need at least 3 tumors per group")

    # --- 2. exact prior + transform -------------------------------------------
    z0 = np.load(os.path.join(RUN_DIR, f"{sids[0]}_draws.npz"), allow_pickle=True)
    prior, tr = H.tcxm_joint_delay_prior(PRIOR_CHOICES, DELAY_BOUNDS, acq,
                                         np.asarray(z0["t"]), np.asarray(z0["aif"]))
    if tuple(tr.names) != tuple(PARAMS):
        raise ValueError(f"PARAMS {PARAMS} must equal the model's {tr.names}")
    X = build_covariates(sids, mal)
    need_c = RUN_HIERARCHICAL and LIKELIHOOD in ("mc", "both")
    if need_c:
        print("computing log p0 for every draw (one-off, mc likelihood only)...")
    cohort = H.build_cohort(sids, mal, draws, tr,
                            H.prior_logpdf_rows(prior) if need_c else None,
                            X=X, cov_names=COVARIATES)
    print(f"  cohort ready ({time.time() - t0:.0f} s)")

    # --- 3. stage-1 diagnostic ------------------------------------------------
    pm, psd = H.prior_z_moments(prior.sample, tr, n=PRIOR_Z_SAMPLES)
    s1 = H.stage1_within_between(cohort, pm, psd)
    s1.to_csv(os.path.join(out, "stage1_within_between.csv"), index=False)
    fig_stage1(s1, os.path.join(out, "stage1_within_between.png"))
    cols = [c for c in ("param", "group", "n", "n_informative", "frac_uninformative",
                        "within_sd", "between_sd", "heterogeneity_P90_P10",
                        "pred_sd_ratio") if c in s1]
    print("\nSTAGE-1 DIAGNOSTIC (z-scale; pred_sd_ratio < 1 = pooling helps)")
    print(s1[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    result = dict(out_dir=out, stage1=s1, n_tumors=len(sids))
    if not RUN_HIERARCHICAL:
        print("\nRUN_HIERARCHICAL = False -- stopping after the diagnostic")
        return result

    # --- 4-6. hyperparameters, per method -------------------------------------
    if LIKELIHOOD not in ("gaussian", "mc", "both"):
        raise ValueError(f"LIKELIHOOD must be 'gaussian', 'mc' or 'both'")
    methods = ("gaussian", "mc") if LIKELIHOOD == "both" else (LIKELIHOOD,)
    spec = H.HierSpec(param_names=tuple(PARAMS),
                      group_effect=tuple(GROUP_EFFECT[p] for p in PARAMS),
                      separate_sigma=tuple(SEPARATE_SIGMA[p] for p in PARAMS),
                      family=FAMILY, nu=NU, cov_names=tuple(COVARIATES), **HYPER)
    layout = H.PhiLayout(spec)
    phi0 = H.init_phi(cohort, layout)
    lp_rows = H.prior_logpdf_rows(prior)
    rng = np.random.default_rng(SEED)
    result.update(layout=layout, spec=spec, methods={})

    for meth in methods:
        print(f"\n{'=' * 72}\nLIKELIHOOD = {meth}\n{'=' * 72}")
        if meth == "gaussian":
            if FAMILY != "normal":
                print("  ! Gaussian method needs FAMILY='normal' (closed-form "
                      "marginal) -- skipping")
                continue
            sep = H.check_prior_separable(lp_rows, tr, cohort.z[0].mean(0))
            if sep > 1e-6:
                print(f"  ! prior not separable on z (mixed derivative {sep:.2e}); "
                      f"the diagonal curvature is approximate")
            glik = H.gaussian_likelihood(cohort, lp_rows, min_info=GAUSS_MIN_INFO)
            data = glik.data(cohort)
            logdensity, _, cond_post = H.make_gaussian_model(spec, layout)
            print(f"  per-tumor directions dropped as uninformative: "
                  f"{dict(zip(*np.unique(glik.n_dropped, return_counts=True)))}")
            print(f"  tumors with |skew| of z-draws > 1 (Gaussian summary "
                  f"questionable): {int((glik.max_abs_skew > 1).sum())}/{len(sids)}")
        else:
            if cohort.c is None:
                raise RuntimeError("cohort has no c")
            data = cohort.data()
            logdensity, log_weights = H.make_two_stage_model(spec, layout)

        print(f"  sampling {layout.dim} hyperparameters "
              f"({N_CHAINS} chains x {N_WARMUP}+{N_DRAWS})...")
        t1 = time.time()
        chains, n_div, acc, steps = H.sample_hyper(
            logdensity, data, phi0, n_warmup=N_WARMUP, n_draws=N_DRAWS,
            n_chains=N_CHAINS, seed=SEED, dense=DENSE, target_accept=TARGET_ACCEPT)
        rhat, ess = H.hyper_diagnostics(chains, layout.names)
        rmax, emin = max(rhat.values()), min(ess.values())
        ok = rmax <= RHAT_MAX and emin >= ESS_MIN and n_div == 0
        print(f"  {time.time() - t1:.0f} s  R-hat max {rmax:.4f}  ESS min {emin:.0f}  "
              f"divergences {n_div}  accept {acc:.2f}  -> "
              f"{'converged' if ok else 'NOT CONVERGED -- do not interpret'}")
        pd.DataFrame(dict(name=layout.names, rhat=[rhat[n] for n in layout.names],
                          ess=[ess[n] for n in layout.names])).to_csv(
            os.path.join(out, f"hier_hyper_diagnostics_{meth}.csv"), index=False)
        np.savez_compressed(os.path.join(out, f"hier_hyper_draws_{meth}.npz"),
                            chains=chains, names=np.array(layout.names, dtype=object),
                            subject_ids=np.array(sids, dtype=object), malignant=mal,
                            spec=json.dumps(spec.__dict__, default=str))

        summ = H.summarize_hyper(chains, layout, spec, tr, rhat, ess)
        summ.insert(1, "sampler_converged", ok)
        summ.insert(1, "likelihood", meth)
        summ.to_csv(os.path.join(out, f"hier_group_summary_{meth}.csv"), index=False)
        print("\n  GROUP-LEVEL RESULTS (population medians, natural units; 95% CI)")
        for _, r in summ.iterrows():
            p = r["param"]
            if r["group_effect"]:
                print(f"    {p:9s} B {r['popmedian_B_median']:.4g} "
                      f"[{r['popmedian_B_ci2.5']:.4g}, {r['popmedian_B_ci97.5']:.4g}]   "
                      f"M {r['popmedian_M_median']:.4g} "
                      f"[{r['popmedian_M_ci2.5']:.4g}, {r['popmedian_M_ci97.5']:.4g}]   "
                      f"P(M>B)={r['P_delta_gt_0']:.3f}"
                      f"{'  CI excludes 0' if r['ci95_excludes_0'] else ''}")
            else:
                print(f"    {p:9s} all {r['popmedian_B_median']:.4g} "
                      f"[{r['popmedian_B_ci2.5']:.4g}, {r['popmedian_B_ci97.5']:.4g}]")

        flat = chains.reshape(-1, chains.shape[-1])
        pick = flat[rng.choice(flat.shape[0], min(K_PHI_REWEIGHT, flat.shape[0]),
                               replace=False)]
        res_m = dict(summary=summ, chains=chains, converged=ok, n_div=n_div,
                     rhat_max=rmax, ess_min=emin)
        if meth == "gaussian":
            zs = H.gaussian_tumor_samples(pick, cond_post, data, n_per=50, seed=SEED)
            tt = H.tumor_table_gaussian(cohort, zs, glik)
        else:
            W, khat, wess = H.tumor_reweighting(pick, log_weights, data)
            tt = H.tumor_table(cohort, W, khat, wess)
            n_bad = int((tt["khat_max"] > KHAT_BAD).sum())
            print(f"\n  per-tumor k-hat > {KHAT_BAD}: {n_bad}/{len(tt)} tumors -- their "
                  f"reweighted contributions are noisy (see hier_khat.png)")
            fig_khat(tt, os.path.join(out, "hier_khat.png"))
            res_m["n_khat_bad"] = n_bad
        tt.to_csv(os.path.join(out, f"hier_tumor_estimates_{meth}.csv"), index=False)
        res_m["tumors"] = tt
        fig_group_effects(chains, layout, spec, tr,
                          os.path.join(out, f"hier_group_effects_{meth}.png"))
        fig_shrinkage(tt, PARAMS, os.path.join(out, f"hier_shrinkage_{meth}.png"),
                      flag_col="khat_max" if meth == "mc" else None)
        result["methods"][meth] = res_m

    # --- method agreement ------------------------------------------------------
    if len(result["methods"]) == 2:
        cmp = H.compare_methods(result["methods"]["gaussian"]["chains"],
                                result["methods"]["mc"]["chains"], layout,
                                tol_sd=AGREE_TOL_SD)
        cmp = cmp.rename(columns={"mean_a": "mean_gaussian", "mean_b": "mean_mc",
                                  "sd_a": "sd_gaussian", "sd_b": "sd_mc",
                                  "sd_ratio_a_over_b": "sd_ratio_gaussian_over_mc"})
        cmp.to_csv(os.path.join(out, "hier_method_comparison.csv"), index=False)
        bad = cmp[~cmp["agree"]]
        dl = cmp[cmp["name"].str.startswith("delta")]
        print(f"\nGAUSSIAN vs MC: group effects (delta) agree within "
              f"{AGREE_TOL_SD} SD: {int(dl['agree'].sum())}/{len(dl)}; "
              f"all components: {int(cmp['agree'].sum())}/{len(cmp)}")
        if len(bad):
            print("  disagreeing components:\n" + bad[["name", "mean_gaussian",
                  "mean_mc", "mean_diff_in_sd"]].to_string(index=False,
                  float_format=lambda v: f"{v:.3f}"))
        result["comparison"] = cmp

    print(f"\nresults written to: {out}")
    return result


if __name__ == "__main__":
    main()
