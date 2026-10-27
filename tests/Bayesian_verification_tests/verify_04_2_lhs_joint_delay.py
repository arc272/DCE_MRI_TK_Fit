#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Mon Sep 21 23:14:14 2026

@author: allisonchen
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_04_2_lhs_joint_delay.py
============================
VERIFICATION 4b -- how does JOINTLY estimating an unknown arrival delay
affect recovery of the TCXM parameters (ve, vp, fp, ps)?

Design
------
  * TRUTHS: N_COMBOS LHS draws over the 4-D TCXM box only (identical to
    verify_04 -- same LHS_SEED, same inset box), so results are directly
    comparable to the no-delay run.
  * INJECTED DELAY: each combo is simulated at every value in
    TAU_DELAY_TRUE_VALUES (fixed, NOT LHS-sampled). The delay is a pure
    AIF shift; the kinetics are identical across a combo's delay levels.
  * FIT: always delay_mode='joint' -- tau_delay is a free, jointly
    estimated 5th parameter, uniform prior on DELAY_BOUNDS. The question
    is not whether tau_delay itself is identifiable (real-data geometry
    makes it so) but whether estimating it degrades ve/vp/fp/ps.
  * The recovery / bias figures are GROUPED BY DELAY, so the same tissue
    can be compared across 2.5 / 5 / 10 s.

Implementation note: truth simulation uses the delay-augmented spec
(layout.model), but fit_curve_bayes is always given the BASE TCXM --
it re-augments internally from cfg_bayes.delay_mode. Passing
layout.model would double-augment.

Run:
    PYTHONPATH=src python tests/Bayesian_verification_tests/verify_04_lhs_joint_delay.py
"""


import os
import time
import warnings
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")          # must precede the pyplot import, not follow it
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

warnings.filterwarnings("ignore")

from dce_mri.synthetic import default_test_setup, simulate_curve, add_noise
from dce_mri.kinetic_models import TCXM, GKM
from dce_mri.bayes import (BayesConfig, fit_curve_bayes, build_param_layout,
                           NOISE_OPTIONS)
from dce_mri.bayes_priors import assemble_prior
from dce_mri.fitting import _lhs_starts   # reused for TRUTH generation

# house colors -- consistent with verify_02 / verify_03 / bayes_figures.py
C_BAYES  = "#B22234"     # posterior (crimson)
C_TRUTH  = "#1B4F72"     # truth / identity line (navy)
C_MEDIAN = "#E67E22"     # posterior median marker (orange)
C_TC     = "#2E7D32"     # Tc = vp/fp, ridge summary (green)
C_TE     = "#7B3F9E"     # Te = ve/ps, ridge summary (purple)
DELAY_COLORS = ["#2E7D32", "#B22234", "#1B4F72"]   # short / mid / long

# =============================================================================
# settings
# =============================================================================

N_COMBOS   = 15
LHS_SEED   = 7
EDGE_INSET = 0.05

SIGMA = 50.0

# Seeds per combo -- bump to 3-5 before making BIAS claims.
N_SEEDS = 1
SEED0   = 1

PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}

TAU_DELAY_TRUE_VALUES =[10]   # s; fixed, crossed with every combo
DELAY_BOUNDS = (-10.0, 30.0)                # fit-side prior box for tau_delay

N_PRIOR_SAMPLES = 20000
PRIOR_SEED = 999

N_DRAWS  = 2000
N_TUNE   = 2000
N_CHAINS = 8

BACKEND    = "blackjax"    # "pymc" | "blackjax"
DENSE_MASS = True

OUTDIR = os.path.join(
    os.path.expanduser("~"), "dce_bayes_results",
    "verify04_jointdelay_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
)


# =============================================================================
# helpers
# =============================================================================

def _inset_box(lb, ub, log_dims, inset):
    """
    Shrink [lb, ub] inward by `inset` fraction of the (log-)range on each
    side, so no LHS-drawn truth can land exactly on a bound.
    """
    lb2, ub2 = np.array(lb, float), np.array(ub, float)
    for j in range(len(lb2)):
        if j in log_dims:
            llo, lhi = np.log(lb2[j]), np.log(ub2[j])
            pad = inset * (lhi - llo)
            lb2[j] = np.exp(llo + pad)
            ub2[j] = np.exp(lhi - pad)
        else:
            pad = inset * (ub2[j] - lb2[j])
            lb2[j] += pad
            ub2[j] -= pad
    return lb2, ub2


def nearest_neighbor_separation(curves, sigma):
    """
    RMS distance from each curve to its closest OTHER curve, in units of
    the noise sd. < 1 means at least one other sampled truth is
    indistinguishable from this one at this noise level.
    """
    n = len(curves)
    out = np.empty(n)
    for i in range(n):
        d = [np.sqrt(np.mean((curves[i] - curves[j]) ** 2))
             for j in range(n) if j != i]
        out[i] = (min(d) / sigma) if d else np.nan
    return out


def _iqr_err(med, lo, hi):
    """Asymmetric yerr for errorbar(), clipped so log axes never get <= 0."""
    return np.vstack([np.maximum(med - lo, 0.0), np.maximum(hi - med, 0.0)])


def _bimodality_coeff(x):
    """
    Sarle's bimodality coefficient, BC = (skew^2 + 1) / kurtosis.
    Uniform gives 0.555. Only trustworthy on the LOG scale for a positive
    ratio; a hint that directs your eye to the histogram, never the answer.
    """
    from scipy.stats import skew, kurtosis
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.nan
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)
    return float((g ** 2 + 1.0) / (k + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))))


def tc_te_diagnostic(res, tag, outdir, truth_map=None):
    """
    Is the vp/ps bimodality a COORDINATE artefact or a real second mode?
    UNCHANGED from verify_04 -- looks parameters up by name, so the extra
    tau_delay column is simply ignored.
    """
    names = list(res.param_names)
    need = ("ve", "vp", "fp", "ps")
    if not all(nm in names for nm in need):
        print(f"  Tc/Te diagnostic: skipped (needs {need}; row has {tuple(names)})")
        return None

    flat = res.flat()
    ve, vp = flat[:, names.index("ve")], flat[:, names.index("vp")]
    fp, ps = flat[:, names.index("fp")], flat[:, names.index("ps")]
    Tc, Te = vp / fp, ve / ps

    tc_true = te_true = None
    if truth_map:
        try:
            tc_true = float(truth_map["vp"]) / float(truth_map["fp"])
            te_true = float(truth_map["ve"]) / float(truth_map["ps"])
        except (KeyError, TypeError, ZeroDivisionError):
            pass

    out = {}
    print(f"\n  Tc/Te reparameterisation check   (BC on LOG scale; "
          f"uniform=0.555, higher => possibly bimodal)")
    print(f"    {'quantity':<10}{'median':>12}{'IQR':>12}{'BC(log)':>10}"
          f"{'truth':>10}   reading")
    for nm, v, tv in (("Tc", Tc, tc_true), ("Te", Te, te_true),
                      ("vp", vp, truth_map.get("vp") if truth_map else None),
                      ("ps", ps, truth_map.get("ps") if truth_map else None)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)]
        bc = _bimodality_coeff(np.log(v))
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        flag = "possibly bimodal" if bc > 0.555 else "unimodal"
        tvs = f"{float(tv):10.4g}" if tv is not None else f"{'-':>10}"
        print(f"    {nm:<10}{med:12.4g}{iqr:12.4g}{bc:10.3f}{tvs}   {flag}")
        out[nm] = dict(median=med, iqr=iqr, bc_log=bc,
                       truth=(float(tv) if tv is not None else None))

    ch = res.chains
    cTc = ch[:, :, names.index("vp")] / ch[:, :, names.index("fp")]
    cTe = ch[:, :, names.index("ve")] / ch[:, :, names.index("ps")]
    for nm, arr in (("Tc", cTc), ("Te", cTe)):
        meds = np.median(arr, axis=1)
        within = float(np.mean(np.percentile(arr, 75, axis=1)
                               - np.percentile(arr, 25, axis=1)))
        spread = float(meds.max() - meds.min())
        ok = "OK" if spread <= 0.5 * within else "DISAGREE"
        print(f"    per-chain {nm}: spread={spread:.4g}  within={within:.4g}  {ok}")
        out[nm]["chain_spread"] = spread
        out[nm]["chain_within"] = within

    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    ax[0][3].axis("off")
    for a, (v, nm, tv) in zip(ax[0],
                              ((vp, "vp", truth_map.get("vp") if truth_map else None),
                               (ps, "ps", truth_map.get("ps") if truth_map else None),
                               (fp, "fp", truth_map.get("fp") if truth_map else None))):
        a.hist(v, bins=90, color=C_BAYES, alpha=0.85)
        if tv is not None:
            a.axvline(float(tv), ls="--", color=C_TRUTH, lw=1.6, label="truth")
            a.legend(fontsize=8)
        a.set_xlabel(nm); a.set_ylabel("count")
        a.set_title(f"{nm}  (sampled coordinate)")

    for a, (v, nm, tv) in zip(ax[1], ((Tc, "Tc = vp/fp  (s)", tc_true),
                                      (Te, "Te = ve/ps  (s)", te_true))):
        vv = v[np.isfinite(v) & (v > 0)]
        a.hist(vv, bins=np.geomspace(vv.min(), vv.max(), 90),
               color=C_TC, alpha=0.85)
        a.set_xscale("log")
        if tv is not None:
            a.axvline(float(tv), ls="--", color=C_TRUTH, lw=1.6, label="truth")
            a.legend(fontsize=8)
        a.set_xlabel(nm); a.set_ylabel("count")
        a.set_title(f"{nm.split()[0]}  (derived -- log axis)")

    a = ax[1][2]
    a.scatter(vp, fp, s=2, alpha=0.12, color="#444444")
    if truth_map and truth_map.get("vp") is not None:
        a.plot(float(truth_map["vp"]), float(truth_map["fp"]), "*",
               ms=16, color=C_TRUTH, label="truth")
        a.legend(fontsize=8)
    a.set_yscale("log"); a.set_xlabel("vp"); a.set_ylabel("fp")
    a.set_title("the ridge: curved streak => coordinate artefact,\n"
                "two separate clouds => real second mode")

    a = ax[1][3]
    a.scatter(ve, ps, s=2, alpha=0.12, color="#444444")
    if truth_map and truth_map.get("ve") is not None:
        a.plot(float(truth_map["ve"]), float(truth_map["ps"]), "*",
               ms=16, color=C_TRUTH, label="truth")
        a.legend(fontsize=8)
    a.set_yscale("log"); a.set_xlabel("ve"); a.set_ylabel("ps")
    a.set_title("the Te ridge: curved streak => coordinate artefact,\n"
                "two separate clouds => real second mode")

    fig.suptitle(f"{tag}: are the vp/ps humps a coordinate artefact?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, f"{tag}_tc_te_check.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
    return out


def plot_true_signal_with_noise(time_var, signal_true, signal_noisy, sigma,
                                title="True signal with added noise",
                                save_path=None):
    """Not called automatically -- kept for ad hoc sanity checks."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_var, signal_true, lw=2.0, label="true signal")
    ax.plot(time_var, signal_noisy, "o-", ms=3.5, lw=0.8, alpha=0.7,
            label=f"noisy signal ($\\sigma={sigma:g}$)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Signal")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# =============================================================================
# figures
# =============================================================================

def plot_truth_design(truths, names, save_path):
    """(vp,fp), (ve,ps), (ve,vp) for the 4-D kinetic truths, numbered."""
    idx = {nm: names.index(nm) for nm in ("ve", "vp", "fp", "ps") if nm in names}
    if len(idx) < 4:
        print("  truth design plot: skipped (missing one of ve/vp/fp/ps)")
        return

    n = len(truths)
    colors = [cm.viridis(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    pairs = [("vp", "fp"), ("ve", "ps"), ("ve", "vp")]
    for (a_name, b_name), ax in zip(pairs, axes):
        xs = truths[:, idx[a_name]]
        ys = truths[:, idx[b_name]]
        for i in range(n):
            ax.scatter(xs[i], ys[i], color=colors[i], s=60, zorder=3)
            ax.annotate(str(i + 1), (xs[i], ys[i]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        if a_name in ("fp", "ps"):
            ax.set_xscale("log")
        if b_name in ("fp", "ps"):
            ax.set_yscale("log")
        ax.set_xlabel(a_name); ax.set_ylabel(b_name)
        ax.grid(alpha=0.25)
        ax.set_title(f"{a_name} vs {b_name}")

    fig.suptitle(f"The {n} LHS truth combos (kinetics only; each is fit at "
                 f"every injected delay)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combo_curves(time_var, curves, sep, sigma, save_path):
    """All UNDELAYED truth curves -- a view of the kinetic design only."""
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    n = len(curves)
    mid = curves[n // 2]
    ax.fill_between(time_var, mid - sigma, mid + sigma,
                    color="0.6", alpha=0.30, zorder=0,
                    label=fr"$\pm\sigma$ ({sigma:g}) about combo {n // 2 + 1}")
    norm = mcolors.Normalize(vmin=0, vmax=max(n - 1, 1))
    for i, (c, s) in enumerate(zip(curves, sep)):
        ax.plot(time_var, c, lw=1.8, color=cm.viridis(norm(i)), zorder=3,
                label=f"combo {i + 1}  (nn sep {s:.1f}$\\sigma$)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Signal")
    ax.set_title(f"TCXM curves for all {n} LHS truth combos, no delay  "
                 f"(\u03c3 = {sigma:g})")
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prior_vs_posterior(res, truth_map, names, log_dims, lb, ub,
                            prior_samples, tag, save_path):
    """
    One panel per fitted parameter (now including tau_delay): prior
    (grey) vs posterior (crimson), density-normalised on shared bins,
    truth (navy dashed) and posterior median (orange) marked.

    prior_samples comes from assemble_prior(...).sample() -- the same
    prior the sampler uses (Beta volfrac, log-uniform rates, uniform
    tau_delay) -- generated once in main().
    """
    post = res.flat()
    res_names = list(res.param_names)
    n = len(res_names)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow),
                             squeeze=False)

    for k, pname in enumerate(res_names):
        ax = axes[k // ncol][k % ncol]

        if pname in names:
            j = names.index(pname)
            log = j in log_dims
            lo_k, hi_k = float(lb[j]), float(ub[j])
            prior_col = prior_samples[:, j]
        else:
            # a parameter outside the fit layout -- no prior overlay
            log = False
            lo_k, hi_k = float(np.min(post[:, k])), float(np.max(post[:, k]))
            prior_col = None

        bins = (np.geomspace(lo_k, hi_k, 60) if log else np.linspace(lo_k, hi_k, 60))

        if prior_col is not None:
            ax.hist(prior_col, bins=bins, density=True,
                    color="0.75", alpha=0.9, label="prior", zorder=1)
        ax.hist(post[:, k], bins=bins, density=True,
                color=C_BAYES, alpha=0.75, label="posterior", zorder=2)

        med = float(np.median(post[:, k]))
        ax.axvline(med, color=C_MEDIAN, lw=1.8, zorder=3,
                   label=f"post. median\n{med:.4g}")
        if pname in truth_map:
            ax.axvline(float(truth_map[pname]), color=C_TRUTH, ls="--", lw=1.8,
                       zorder=3, label=f"truth {float(truth_map[pname]):.4g}")

        if log:
            ax.set_xscale("log")
        rhat = res.rhat.get(pname, np.nan)
        ess  = res.ess.get(pname, np.nan)
        ax.set_title(f"{pname}   R-hat={rhat:.3f}   ESS={ess:.0f}", fontsize=10)
        ax.set_xlabel(pname); ax.set_ylabel("density")
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.2)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"{tag}   TCXM + joint delay   (densities normalised to unit area)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_across_combos(rows, names, log_param_names, delay_values, outdir):
    """
    Estimated (median, IQR) vs true, one figure per parameter, points
    COLORED BY INJECTED DELAY -- shows whether ve/vp/fp/ps recovery
    degrades as the jointly-fitted delay grows.
    """
    for pname in names:
        r_all = [d for d in rows if d["param"] == pname]
        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        all_truth = np.array([d["truth"] for d in r_all])
        lims = [all_truth.min(), all_truth.max()]
        ax.plot(lims, lims, "--", color="0.4", lw=1.4, zorder=1,
                label="perfect recovery")

        for delay_true, c in zip(delay_values, DELAY_COLORS):
            r = [d for d in r_all if d["delay"] == delay_true]
            truth = np.array([d["truth"] for d in r])
            med   = np.array([d["bayes_med"] for d in r])
            lo    = np.array([d["bayes_q25"] for d in r])
            hi    = np.array([d["bayes_q75"] for d in r])
            ax.errorbar(truth, med, yerr=_iqr_err(med, lo, hi),
                        fmt="o", ms=5.5, lw=1.2, capsize=3, color=c,
                        zorder=3, label=f"delay={delay_true:g}s")

        if pname in log_param_names:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"true {pname}")
        ax.set_ylabel(f"estimated {pname}  (median, bars = IQR)")
        ax.set_title(f"Recovery of {pname}, jointly fit with delay\n"
                     f"grouped by injected delay -- same kinetics per combo")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"recovery_{pname}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_bias_summary_across_combos(rows, names, log_param_names, delay_values,
                                    save_path):
    """
    Relative bias vs truth, one panel per parameter, one line per delay.
    tau_delay truths are 2.5-10 s, well away from zero, so est/truth - 1
    is well defined for that panel too.
    """
    n = len(names)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                             squeeze=False)

    for k, pname in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        ax.axhline(0.0, ls="--", color="0.4", lw=1.3, zorder=1)
        for delay_true, c in zip(delay_values, DELAY_COLORS):
            r = sorted([d for d in rows if d["param"] == pname
                        and d["delay"] == delay_true], key=lambda d: d["truth"])
            truth = np.array([d["truth"] for d in r])
            med   = np.array([d["bayes_med"] for d in r])
            ax.plot(truth, med / truth - 1.0, "o-", ms=5, lw=1.2,
                    color=c, zorder=3, label=f"delay={delay_true:g}s")

        if pname in log_param_names:
            ax.set_xscale("log")
        ax.set_xlabel(f"true {pname}")
        ax.set_ylabel("relative bias  (est/truth - 1)")
        ax.set_title(pname)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25, which="both")

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Relative bias, jointly fit with delay -- grouped by injected "
                 f"delay  (rates prior = {PRIOR_CHOICES.get('rates', 'loguniform')},"
                 f" volfrac = {PRIOR_CHOICES.get('volfrac', 'flat_masked')})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_summary(diag_rows, save_path):
    """
    Log-scale BC of Tc and Te per (combo, delay, seed). With several
    delays per combo, points at the same combo # overplot -- deliberately
    left as-is until it's clear whether ridge behaviour is delay-sensitive.
    """
    if not diag_rows:
        print("  ridge summary: no rows (need ve,vp,fp,ps in the fit) -- skipped")
        return

    combos = [d["combo"] for d in diag_rows]
    bc_tc  = [d["Tc"]["bc_log"] for d in diag_rows]
    bc_te  = [d["Te"]["bc_log"] for d in diag_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0.555, ls="--", color="0.4", lw=1.2,
               label="uniform reference (0.555)")
    ax.scatter(combos, bc_tc, s=50, color=C_TC, label="Tc = vp/fp", zorder=3)
    ax.scatter(combos, bc_te, s=50, color=C_TE, marker="^", label="Te = ve/ps",
               zorder=3)
    ax.set_xlabel("combo #")
    ax.set_ylabel("bimodality coefficient (log scale)")
    ax.set_title("Tc/Te bimodality across LHS truth combos (all delays)\n"
                 "above the line = possibly bimodal -- a hint, check histograms")
    ax.set_xticks(sorted(set(combos)))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_bimodal_tc = sum(1 for b in bc_tc if b > 0.555)
    n_bimodal_te = sum(1 for b in bc_te if b > 0.555)
    print(f"\n  ridge summary: Tc possibly-bimodal in {n_bimodal_tc}/{len(bc_tc)} "
          f"rows, Te possibly-bimodal in {n_bimodal_te}/{len(bc_te)} rows")


# =============================================================================
# main
# =============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"figures -> {OUTDIR}")
    print(f"prior choices: {PRIOR_CHOICES}")
    print(f"injected delays: {TAU_DELAY_TRUE_VALUES} s   "
          f"(fit: delay_mode='joint', bounds={DELAY_BOUNDS})")

    setup = default_test_setup(n_frames=36, dt=4.17, t_arrival=32.0)   # Parker AIF, 0-150 s
    acq, subject, cfg_fit = setup["acq"], setup["subject"], setup["cfg"]
    aif, tv = setup["aif"], setup["time_var"]

    # --- 4-dim: drives the truth LHS + truth-design figure ONLY --------
    names    = list(GKM.param_names)
    lb, ub   = np.asarray(GKM.lb, float), np.asarray(GKM.ub, float)
    log_dims = GKM.log_dims or ()

    # --- 5-dim fit-side layout: ve, vp, fp, ps, tau_delay --------------
    cfg_shape = BayesConfig(
        prior_choices=PRIOR_CHOICES, noise="gauss_marginalized_sigma",
        delay_mode="joint", delay_bounds=DELAY_BOUNDS,
        backend=BACKEND, seed=SEED0)
    layout = build_param_layout(GKM, cfg_shape, aif=aif, time_var=tv, acq=acq)
    names_fit    = list(layout.model.param_names)          # ve,vp,fp,ps,tau_delay
    lb_fit, ub_fit = (np.asarray(layout.model.lb, float),
                      np.asarray(layout.model.ub, float))
    log_dims_fit = layout.model.log_dims or ()
    log_param_names_fit = {names_fit[j] for j in log_dims_fit}

    n_fits = N_COMBOS * len(TAU_DELAY_TRUE_VALUES) * N_SEEDS
    print(f"\n{n_fits} Bayesian fits queued = {N_COMBOS} combos x "
          f"{len(TAU_DELAY_TRUE_VALUES)} delays x {N_SEEDS} seed(s), "
          f"each {N_DRAWS}x{N_CHAINS} draws.\n")

    # --- (a) 4-D LHS truths, identical design to verify_04 --------------
    lb_in, ub_in = _inset_box(lb, ub, log_dims, EDGE_INSET)
    truths = _lhs_starts(N_COMBOS, lb_in, ub_in, GKM.constraints, log_dims,
                         seed=LHS_SEED)

    # undelayed curves: kinetic-design view + nn separation only
    curves = [simulate_curve(t, aif, tv, GKM, acq, subject, cfg_fit)
              for t in truths]
    sep = nearest_neighbor_separation(curves, SIGMA)

    print(f"{'='*86}\nLHS TRUTH DESIGN   {N_COMBOS} combos, seed={LHS_SEED}")
    header = (f"  {'combo':>6}" + "".join(f"{nm:>12}" for nm in names) +
              f"{'peak enh':>11}{'nn sep':>9}   note")
    print(header)
    for i, (t, c, s) in enumerate(zip(truths, curves, sep)):
        vals = "".join(f"{v:12.4g}" for v in t)
        note = "separable" if s >= 1.0 else "BELOW NOISE FLOOR (nn)"
        print(f"  {i+1:6d}{vals}{c.max()/c[:7].mean():11.2f}{s:9.2f}   {note}")

    plot_truth_design(truths, names, os.path.join(OUTDIR, "truth_design.png"))
    plot_combo_curves(tv, curves, sep, SIGMA,
                      os.path.join(OUTDIR, "curves_all_combos.png"))

    # --- prior render from the SAME prior the sampler uses --------------
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, None, delay_mode="joint",
                           n_noise=noise.n_params,
                           noise_names=noise.param_names,
                           choices=PRIOR_CHOICES, delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(PRIOR_SEED),
                                 N_PRIOR_SAMPLES)[:, :layout.n_model]

    rows = []
    tc_te_rows = []
    fits = []
    t_start = time.time()

    print(f"\n{'='*86}\nFITTING   (delay_mode='joint' always -- tau_delay is a "
          f"free, jointly-estimated nuisance parameter)")
    print(f"  {'combo':>6}{'delay':>7}{'seed':>5}" +
          "".join(f"{nm+' med':>12}{nm+' IQR':>11}" for nm in names_fit) +
          f"{'R-hat(max)':>11}{'ESS(min)':>9}{'div':>6}")

    for i, truth4 in enumerate(truths):
        s = float(sep[i])
        for delay_true in TAU_DELAY_TRUE_VALUES:
            truth5 = np.append(truth4, delay_true)
            truth_map = dict(zip(names_fit, truth5))

            # delay-augmented spec shifts the AIF by tau_delay internally
            S_clean = simulate_curve(truth5, aif, tv, layout.model,
                                     acq, subject, cfg_fit)

            for k in range(N_SEEDS):
                seed = SEED0 + k
                sig = add_noise(S_clean, SIGMA, "gaussian",
                                np.random.default_rng(seed))

                cfg_b = BayesConfig(
                    prior_choices=PRIOR_CHOICES,
                    n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS,
                    noise="gauss_marginalized_sigma", seed=seed,
                    backend=BACKEND, step="nuts", dense_mass_matrix=DENSE_MASS,
                    delay_mode="joint", delay_bounds=DELAY_BOUNDS)
                res = fit_curve_bayes(sig, aif, tv, GKM, acq, subject,   # BASE TCXM --
                                      cfg_fit, cfg_b)                     # re-augmented
                                                                           # internally

                post = res.flat()
                meds, iqrs = [], []
                for pname in names_fit:
                    j = res.param_names.index(pname)
                    by = post[:, j]
                    med = float(np.median(by))
                    q25 = float(np.percentile(by, 25))
                    q75 = float(np.percentile(by, 75))
                    ci_lo, ci_hi = res.ci(pname)
                    truth_v = float(truth_map[pname])
                    rows.append(dict(
                        combo=i + 1, delay=delay_true, seed=seed, param=pname,
                        truth=truth_v, bayes_med=med,
                        bayes_q25=q25, bayes_q75=q75,
                        ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                        in95=bool(ci_lo <= truth_v <= ci_hi),
                        rhat=float(res.rhat.get(pname, np.nan)),
                        ess=float(res.ess.get(pname, np.nan)),
                        sep=s,
                    ))
                    meds.append(med); iqrs.append(q75 - q25)

                rhat_max = max(res.rhat.get(p, np.nan) for p in names_fit)
                ess_min  = min(res.ess.get(p, np.nan) for p in names_fit)
                n_div = res.n_divergences if res.n_divergences is not None else -1
                fits.append(dict(combo=i + 1, delay=delay_true, seed=seed,
                                 rhat_max=rhat_max, ess_min=ess_min, n_div=n_div))
                line = f"  {i+1:6d}{delay_true:7.1f}{seed:5d}" + "".join(
                    f"{m:12.5g}{iqr:11.5g}" for m, iqr in zip(meds, iqrs))
                print(line + f"{rhat_max:11.3f}{ess_min:9.0f}{n_div:6d}")

                tag = f"combo{i+1:02d}_delay{delay_true:g}s_seed{seed}"

                # --- (c) prior vs posterior --------------------------------
                plot_prior_vs_posterior(
                    res, truth_map, names_fit, log_dims_fit, lb_fit, ub_fit,
                    prior_samples, tag,
                    os.path.join(OUTDIR, f"prior_post_{tag}.png"))

                # --- (d) ridge check ----------------------------------------
                diag = tc_te_diagnostic(res, tag, OUTDIR, truth_map=truth_map)
                if diag is not None:
                    diag["combo"] = i + 1
                    diag["delay"] = delay_true
                    diag["seed"] = seed
                    tc_te_rows.append(diag)

                # --- save draws for later derived-quantity checks ----------
                np.savez_compressed(
                    os.path.join(OUTDIR, f"draws_{tag}.npz"),
                    chains=res.chains,
                    param_names=np.array(res.param_names, dtype=object),
                    truth=truth5, n_divergences=np.array(n_div))

    elapsed = time.time() - t_start

    plot_recovery_across_combos(rows, names_fit, log_param_names_fit,
                                TAU_DELAY_TRUE_VALUES, OUTDIR)
    plot_bias_summary_across_combos(
        rows, names_fit, log_param_names_fit, TAU_DELAY_TRUE_VALUES,
        os.path.join(OUTDIR, "bias_summary.png"))
    plot_ridge_summary(tc_te_rows, os.path.join(OUTDIR, "ridge_summary.png"))

    # =========================================================================
    # summary -- split by injected delay
    # =========================================================================
    print(f"\n{'='*86}\nSUMMARY   ({N_COMBOS} LHS combos x "
          f"{len(TAU_DELAY_TRUE_VALUES)} delays x {N_SEEDS} seed(s), "
          f"sigma={SIGMA:g}, {elapsed:.0f}s)")
    for delay_true in TAU_DELAY_TRUE_VALUES:
        print(f"\n  delay = {delay_true:g} s")
        print(f"    {'param':<10}{'coverage':>10}{'mean |bias|':>13}"
              f"{'median |bias|':>15}{'n below noise':>15}")
        for pname in names_fit:
            r = [d for d in rows if d["param"] == pname
                 and d["delay"] == delay_true]
            cov = np.mean([d["in95"] for d in r])
            ab = [abs(d["bayes_med"] / d["truth"] - 1.0) for d in r]
            n_below = sum(1 for d in r if d["sep"] < 1.0)
            print(f"    {pname:<10}{cov:9.0%}{np.mean(ab):13.3f}"
                  f"{np.median(ab):15.3f}{n_below:15d}")
        ff = [f for f in fits if f["delay"] == delay_true]
        print(f"    fits with R-hat>1.01: "
              f"{sum(f['rhat_max'] > 1.01 for f in ff)}/{len(ff)}   "
              f"with divergences: {sum(f['n_div'] > 0 for f in ff)}/{len(ff)}")

    print("\n  coverage      : fraction of combos with truth inside the 95% CI.")
    print("  mean |bias|   : mean |posterior median / truth - 1| across combos.")
    print("  median |bias| : same, median -- robust to prior-dominated low-ps "
          "outliers.")
    print("  n below noise : combos whose nearest-neighbour (undelayed) curve "
          "differs by < 1 sigma.")
    print("\n  Compare ve/vp/fp/ps rows ACROSS delays, and against the no-delay "
          "verify_04 run\n  (same LHS_SEED => same kinetic truths): widening or "
          "new bias here is the\n  cost of estimating the delay jointly.")
    print(f"\nfigures written to: {OUTDIR}")


if __name__ == "__main__":
    main()