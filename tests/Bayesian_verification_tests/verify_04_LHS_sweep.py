#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Mon Sep 21 14:29:47 2026

@author: allisonchen
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_04_lhs_ridge_check.py
=============================
VERIFICATION 4 -- does recovery / ridge behaviour hold across the JOINT
parameter space, not just along one axis through p0?

verify_03 swept ONE TCXM parameter at a time, holding the rest at the
single point p0. That's fine for asking "is this one axis identifiable
near this one operating point", but it can't see anything that only
exists jointly -- and the vp/ps (equivalently Tc=vp/fp, Te=ve/ps) ridge
that verify_02/03 already flagged is exactly that kind of thing. A ridge
in a 4-D likelihood is a property of the JOINT surface; moving one
coordinate at a time from a single p0 can walk straight along it, or
never see it at all, purely depending on where p0 happens to sit.

What changed relative to verify_03
-----------------------------------
  (a) TRUTH GENERATION: instead of sweep_values() stepping one parameter
      across its range with the rest pinned at p0, this draws
      N_COMBOS=15 full parameter vectors via Latin Hypercube Sampling
      over the ENTIRE TCXM box (all parameters vary together,
      log-uniform on fp/ps per TCXM.log_dims). It reuses
      dce_mri.fitting._lhs_starts -- the exact routine the NLLS fitter
      already uses to seed its multi-start search -- just pointed at the
      TRUTH values instead of optimiser starting points.

  (b) NO BOOTSTRAP: residual_bootstrap_curve is gone. verify_02/03 used
      it as an independent check against the Bayesian posterior; here
      the posterior itself is the thing being examined for ridges, and
      bootstrap adds cost without adding to that answer. This also
      roughly halves the work per truth point.

  (c) PRIOR VS POSTERIOR: every (combo, seed) gets the histogram grid
      you were looking at for "ROW 2" -- windowed prior (grey) vs
      posterior (crimson), truth (navy dashed) and posterior median
      (orange) marked, R-hat/ESS in each panel title.

  (d) RIDGE CHECK AT EVERY COMBO: the existing Tc/Te reparameterisation
      diagnostic (tc_te_diagnostic, unchanged from verify_03) now runs
      once per combo instead of once at p0, and a new summary figure
      plots its bimodality coefficient across all 15 combos so you can
      see whether the ridge is a broad feature of this acquisition /
      noise level or localised to particular regions of parameter
      space.

Interpretation carries over unchanged from verify_02/03: R-hat~1 + high
ESS + chains agreeing means a wide posterior is an honest read of a flat
likelihood, not a stuck sampler; a nearest-neighbour curve separation
below ~1 sigma means no method -- no prior, no sampler, no amount of
resampling -- could have told two combos apart at this noise level.

Run:
    PYTHONPATH=src python tests/Bayesian_verification_tests/verify_04_lhs_ridge_check.py
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
from dce_mri.kinetic_models import TCXM
from dce_mri.bayes import BayesConfig, fit_curve_bayes
from dce_mri.fitting import _lhs_starts   # reused for TRUTH generation, not just optimiser starts

# house colors -- consistent with verify_02 / verify_03 / bayes_figures.py
C_BAYES  = "#B22234"     # posterior (crimson)
C_TRUTH  = "#1B4F72"     # truth / identity line (navy)
C_MEDIAN = "#E67E22"     # posterior median marker (orange)
C_TC     = "#2E7D32"     # Tc = vp/fp, ridge summary (green)
C_TE     = "#7B3F9E"     # Te = ve/ps, ridge summary (purple)

# =============================================================================
# settings
# =============================================================================

N_COMBOS   = 15           # LHS truth combinations spanning the FULL joint box
LHS_SEED   = 7             # seed for the truth-generating LHS design
EDGE_INSET = 0.05          # inset the box before drawing truths so no combo
                           # rails on a bound -- same rationale verify_03's
                           # sweep_values() used, just applied to the whole
                           # box instead of one axis at a time

SIGMA = 50.0               # fixed noise sd (verify_02's mid level; peak ~2950)

# Seeds per combo. 1 is enough to see gross failure, but a BIAS claim
# (e.g. "ps runs low") needs several -- with one seed you cannot separate
# "median sits high" from "this noise draw pushed it high". Bump to 3-5
# before telling your PI a parameter is biased rather than just unlucky.
N_SEEDS = 1
SEED0   = 1

PRIOR_CHOICES = {"rates": "loguniform"}   # or {"rates": "uniform"}; see
                                          # verify_03's docstring for why
                                          # this matters for fp/ps

N_PRIOR_SAMPLES = 20000    # size of the Monte-Carlo prior render used for
                           # every "prior (windowed)" histogram
PRIOR_SEED = 999           # fixed, so every combo's figure shows the SAME
                           # prior draw -- only the posterior should move
                           # from panel to panel

N_DRAWS  = 2000
N_TUNE   = 2000
N_CHAINS = 8

BACKEND    = "blackjax"    # "pymc" | "blackjax"
DENSE_MASS = True

OUTDIR = os.path.join(
    os.path.expanduser("~"), "dce_bayes_results",
    "verify04_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
)


# =============================================================================
# helpers
# =============================================================================

def _inset_box(lb, ub, log_dims, inset):
    """
    Shrink [lb, ub] inward by `inset` fraction of the (log-)range on each
    side, so no LHS-drawn truth can land exactly on a bound. Mirrors the
    padding verify_03's sweep_values() used for the same reason -- a
    railed truth makes recovery look bad for reasons that have nothing
    to do with the prior or the sampler.
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
    For an UNORDERED set of curves (the LHS combos have no natural 1-D
    order), the RMS distance from each curve to its closest OTHER curve,
    in units of the noise sd.

    Same read as verify_02/03's curve_separation: < 1 means at least one
    other sampled truth is indistinguishable from this one at this noise
    level -- no method could recover them separately, full stop. Full
    pairwise search (O(n^2); n=15 here is trivial).
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
    Uniform gives 0.555; above that is the conventional "possibly
    bimodal" flag, below is "unimodal".

    CAUTION, verified numerically before use: BC is fooled by SKEW.
    A ratio of uniform-over-loguniform -- exactly Tc's and Te's shape --
    scores 0.70 on the raw scale while being unimodal by construction,
    and a plain lognormal scores 0.47. So this is only trustworthy on
    the LOG scale for a positive ratio, which is how it's used below.
    Treat it as a hint that directs your eye to the histogram, never as
    the answer on its own.
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

    Tc = vp/fp and Te = ve/ps are the two mean transit times the TCXM
    impulse response is actually built from (jax_forward.tcxm_compartments
    computes them on every call, lines ~225-227). The measured curve
    depends on ve,vp,fp,ps only through these ratios plus the vp/ve
    prefactors -- so a CURVED ridge in (vp,fp) shows up as two humps in
    the vp marginal while being one clean hump in Tc.

        Tc/Te unimodal  -> the humps were a coordinate artefact;
                           reparameterising to (Tc,Te,vp,ve) should help.
        Tc/Te bimodal   -> two genuinely separate solutions. NO bijective
                           reparameterisation can merge them; next step is
                           a multi-start NLLS check, not new coordinates.

    Returns the dict it prints, or None if the row has no TCXM block.

    UNCHANGED from verify_03 -- now just called once per LHS combo
    instead of once at p0.
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

    # per-chain medians of Tc/Te -- do the chains agree in THESE coordinates?
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

    # ---- figure: marginals in both coordinate systems + the ridge --------
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    ax[0][3].axis("off")   # nothing goes here; keeps the grid rectangular
    for a, (v, nm, tv) in zip(ax[0],
                              ((vp, "vp", truth_map.get("vp") if truth_map else None),
                               (ps, "ps", truth_map.get("ps") if truth_map else None),
                               (fp, "fp", truth_map.get("fp") if truth_map else None))):
        a.hist(v, bins=90, color="#B22234", alpha=0.85)
        if tv is not None:
            a.axvline(float(tv), ls="--", color="#1B4F72", lw=1.6, label="truth")
            a.legend(fontsize=8)
        a.set_xlabel(nm); a.set_ylabel("count")
        a.set_title(f"{nm}  (sampled coordinate)")

    for a, (v, nm, tv) in zip(ax[1], ((Tc, "Tc = vp/fp  (s)", tc_true),
                                      (Te, "Te = ve/ps  (s)", te_true))):
        vv = v[np.isfinite(v) & (v > 0)]
        a.hist(vv, bins=np.geomspace(vv.min(), vv.max(), 90),
               color="#2E7D32", alpha=0.85)
        a.set_xscale("log")
        if tv is not None:
            a.axvline(float(tv), ls="--", color="#1B4F72", lw=1.6, label="truth")
            a.legend(fontsize=8)
        a.set_xlabel(nm); a.set_ylabel("count")
        a.set_title(f"{nm.split()[0]}  (derived -- log axis)")

    a = ax[1][2]
    a.scatter(vp, fp, s=2, alpha=0.12, color="#444444")
    if truth_map and truth_map.get("vp") is not None:
        a.plot(float(truth_map["vp"]), float(truth_map["fp"]), "*",
               ms=16, color="#1B4F72", label="truth")
        a.legend(fontsize=8)
    a.set_yscale("log"); a.set_xlabel("vp"); a.set_ylabel("fp")
    a.set_title("the ridge: curved streak => coordinate artefact,\n"
                "two separate clouds => real second mode")

    a = ax[1][3]
    a.scatter(ve, ps, s=2, alpha=0.12, color="#444444")
    if truth_map and truth_map.get("ve") is not None:
        a.plot(float(truth_map["ve"]), float(truth_map["ps"]), "*",
               ms=16, color="#1B4F72", label="truth")
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


def plot_true_signal_with_noise(
    time_var,
    signal_true,
    signal_noisy,
    sigma,
    title="True signal with added noise",
    save_path=None,
):
    """
    Plot the noiseless simulated signal and the noisy signal used for
    fitting. Not called automatically -- kept around for ad hoc sanity
    checks on a specific combo/seed while iterating.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_var, signal_true, lw=2.0, label="true signal")
    ax.plot(time_var, signal_noisy, "o-", ms=3.5, lw=0.8, alpha=0.7,
            label=f"noisy signal ($\\sigma={sigma:g}$)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
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
    """
    Quick look at the LHS design itself: the two ridge-relevant pairs
    (vp,fp) and (ve,ps), plus vp vs ve, each combo numbered so you can
    cross-reference against its per-combo prior/posterior and Tc/Te
    figures.
    """
    idx = {nm: names.index(nm) for nm in ("ve", "vp", "fp", "ps") if nm in names}
    if len(idx) < 4:
        print("  truth design plot: skipped (TCXM.param_names is missing "
              "one of ve/vp/fp/ps)")
        return

    n = len(truths)
    cmap = cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

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

    fig.suptitle(f"The {n} LHS truth combos, numbered "
                 "(cross-ref against per-combo figures)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combo_curves(time_var, curves, sep, sigma, save_path):
    """
    All simulated curves for all combos on one axes, coloured by combo
    index, with a +/- sigma band about one representative curve (not a
    true multivariate median -- just a scale reference).
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    n = len(curves)
    mid = curves[n // 2]
    ax.fill_between(time_var, mid - sigma, mid + sigma,
                    color="0.6", alpha=0.30, zorder=0,
                    label=fr"$\pm\sigma$ ({sigma:g}) about combo {n // 2 + 1}")

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=max(n - 1, 1))
    for i, (c, s) in enumerate(zip(curves, sep)):
        ax.plot(time_var, c, lw=1.8, color=cmap(norm(i)), zorder=3,
                label=f"combo {i + 1}  (nn sep {s:.1f}$\\sigma$)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.set_title(f"TCXM curves for all {n} LHS truth combos  (\u03c3 = {sigma:g})")
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prior_vs_posterior(res, truth_map, names, log_dims, lb, ub,
                            prior_samples, tag, save_path):
    """
    One panel per fitted parameter: windowed prior histogram (grey) vs
    posterior histogram (crimson), both density-normalised, with truth
    (navy dashed) and posterior median (orange solid) marked and
    R-hat/ESS in the title -- the same read as the ve/vp/fp/ps/tau_delay
    grid for "ROW 2", just reusable per LHS combo instead of one fixed
    p0.

    "windowed" prior = a large iid draw from the same box the model was
    fit with (uniform on log_dims, uniform elsewhere), histogrammed over
    the SAME bin edges as the posterior so the two read as directly
    comparable areas. That draw is generated ONCE in main() and passed
    in here -- it's the same every call, so only the posterior should
    visibly move from panel to panel.
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
            # e.g. tau_delay from a delay-augmented spec that isn't part
            # of TCXM's own box -- fall back to the posterior's own range
            # so the panel still renders; no prior overlay for it.
            log = False
            lo_k, hi_k = float(np.min(post[:, k])), float(np.max(post[:, k]))
            prior_col = None

        bins = (np.geomspace(lo_k, hi_k, 60) if log else np.linspace(lo_k, hi_k, 60))

        if prior_col is not None:
            ax.hist(prior_col, bins=bins, density=True,
                    color="0.75", alpha=0.9, label="prior (windowed)", zorder=1)
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

    fig.suptitle(f"{tag}   TCXM   (densities normalised to unit area)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_across_combos(rows, names, log_param_names, outdir):
    """
    Estimated (Bayesian posterior median, IQR bars) vs true, one panel
    per parameter, pooling all combo x seed rows. Unlike verify_03's
    version, the OTHER parameters are not held at p0 here -- they vary
    combo to combo, so this is recovery under realistic joint variation
    rather than an idealised one-axis-at-a-time test.
    """
    for pname in names:
        r = [d for d in rows if d["param"] == pname]
        truth = np.array([d["truth"] for d in r])
        med   = np.array([d["bayes_med"] for d in r])
        lo    = np.array([d["bayes_q25"] for d in r])
        hi    = np.array([d["bayes_q75"] for d in r])
        log   = pname in log_param_names

        fig, ax = plt.subplots(figsize=(6.0, 5.6))
        lims = [truth.min(), truth.max()]
        ax.plot(lims, lims, "--", color=C_TRUTH, lw=1.4, zorder=1,
                label="perfect recovery")
        ax.errorbar(truth, med, yerr=_iqr_err(med, lo, hi),
                    fmt="o", ms=5.5, lw=1.4, capsize=3, color=C_BAYES,
                    zorder=3, label="Bayesian posterior")
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"true {pname}")
        ax.set_ylabel(f"estimated {pname}  (median, bars = IQR)")
        ax.set_title(f"Recovery of {pname} across {len(r)} joint LHS combos\n"
                     f"(all other parameters varying too, not fixed at p0)")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"recovery_{pname}.png"),
                   dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_bias_summary_across_combos(rows, names, log_param_names, save_path):
    """Relative bias (est/truth - 1) vs truth, one panel per parameter."""
    n = len(names)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                             squeeze=False)

    for k, pname in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        r = sorted([d for d in rows if d["param"] == pname], key=lambda d: d["truth"])
        truth = np.array([d["truth"] for d in r])
        med   = np.array([d["bayes_med"] for d in r])

        ax.axhline(0.0, ls="--", color=C_TRUTH, lw=1.3, zorder=1)
        ax.plot(truth, med / truth - 1.0, "o-", ms=5, lw=1.2,
                color=C_BAYES, zorder=3)

        if pname in log_param_names:
            ax.set_xscale("log")
        ax.set_xlabel(f"true {pname}")
        ax.set_ylabel("relative bias  (est/truth - 1)")
        ax.set_title(pname)
        ax.grid(alpha=0.25, which="both")

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Relative bias across the joint LHS truth combos  "
                 f"(rates prior = {PRIOR_CHOICES.get('rates', 'loguniform')})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_summary(diag_rows, save_path):
    """
    One point per (combo, seed) for the log-scale bimodality coefficient
    of Tc=vp/fp and Te=ve/ps, so you can see at a glance whether the
    vp/ps <-> ve/ps ridge shows up broadly across the sampled parameter
    space or only in a few pockets -- the question one-truth-at-p0 could
    never answer.
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
    ax.set_title("Tc/Te bimodality across the LHS truth combos\n"
                 "above the line = possibly bimodal (a real second mode, "
                 "not just a marginal-coordinate artefact)")
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

    setup = default_test_setup(n_frames=36, dt=4.17, t_arrival=32.0)          # Parker AIF, 0-150 s
    acq, subject, cfg_fit = setup["acq"], setup["subject"], setup["cfg"]
    aif, tv = setup["aif"], setup["time_var"]

    names    = list(TCXM.param_names)
    lb, ub   = np.asarray(TCXM.lb, float), np.asarray(TCXM.ub, float)
    log_dims = TCXM.log_dims or ()
    log_param_names = {names[j] for j in log_dims}

    n_fits = N_COMBOS * N_SEEDS
    print(f"\n{n_fits} Bayesian fits queued = {n_fits} x "
          f"({N_DRAWS}x{N_CHAINS} draws).")
    print("(no bootstrap this time -- see module docstring for why)\n")

    # --- (a) LHS over the FULL joint box, inset so no combo rails -------
    lb_in, ub_in = _inset_box(lb, ub, log_dims, EDGE_INSET)
    truths = _lhs_starts(N_COMBOS, lb_in, ub_in, TCXM.constraints, log_dims,
                         seed=LHS_SEED)

    curves = [simulate_curve(t, aif, tv, TCXM, acq, subject, cfg_fit)
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

    # --- fixed windowed-prior draw, reused across every combo's figure --
    prior_samples = _lhs_starts(N_PRIOR_SAMPLES, lb, ub, TCXM.constraints,
                                log_dims, seed=PRIOR_SEED)

    rows = []
    tc_te_rows = []
    t_start = time.time()

    print(f"\n{'='*86}\nFITTING")
    print(f"  {'combo':>6}{'seed':>5}" +
          "".join(f"{nm+' med':>12}{nm+' IQR':>11}" for nm in names) +
          f"{'R-hat(max)':>11}{'ESS(min)':>9}")

    for i, (truth_vec, curve, s) in enumerate(zip(truths, curves, sep)):
        truth_map = dict(zip(names, truth_vec))

        for k in range(N_SEEDS):
            seed = SEED0 + k
            sig = add_noise(curve, SIGMA, "gaussian",
                            np.random.default_rng(seed))

            cfg_b = BayesConfig(
                prior_choices=PRIOR_CHOICES,
                n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS,
                noise="gauss_marginalized_sigma", seed=seed, backend=BACKEND,
                step="nuts", dense_mass_matrix=DENSE_MASS)
            res = fit_curve_bayes(sig, aif, tv, TCXM, acq, subject,
                                  cfg_fit, cfg_b)

            post = res.flat()
            meds, iqrs = [], []
            for pname in names:
                j = res.param_names.index(pname)
                by = post[:, j]
                med = float(np.median(by))
                q25 = float(np.percentile(by, 25))
                q75 = float(np.percentile(by, 75))
                ci_lo, ci_hi = res.ci(pname)
                truth_v = float(truth_map[pname])
                rows.append(dict(
                    combo=i + 1, seed=seed, param=pname, truth=truth_v,
                    bayes_med=med, bayes_q25=q25, bayes_q75=q75,
                    ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                    in95=bool(ci_lo <= truth_v <= ci_hi),
                    rhat=float(res.rhat.get(pname, np.nan)),
                    ess=float(res.ess.get(pname, np.nan)),
                    sep=float(s),
                ))
                meds.append(med); iqrs.append(q75 - q25)

            rhat_max = max(res.rhat.get(p, np.nan) for p in names)
            ess_min  = min(res.ess.get(p, np.nan) for p in names)
            line = f"  {i+1:6d}{seed:5d}" + "".join(
                f"{m:12.5g}{iqr:11.5g}" for m, iqr in zip(meds, iqrs))
            print(line + f"{rhat_max:11.3f}{ess_min:9.0f}")

            # --- (c) prior vs posterior, this combo/seed -----------------
            tag = f"combo {i+1:02d}/{N_COMBOS}  seed={seed}"
            plot_prior_vs_posterior(
                res, truth_map, names, log_dims, lb, ub, prior_samples, tag,
                os.path.join(OUTDIR, f"prior_post_combo{i+1:02d}_seed{seed}.png"))

            # --- (d) ridge check, this combo/seed -------------------------
            diag = tc_te_diagnostic(
                res, f"combo{i+1:02d}_seed{seed}", OUTDIR, truth_map=truth_map)
            if diag is not None:
                diag["combo"] = i + 1
                diag["seed"] = seed
                tc_te_rows.append(diag)

    elapsed = time.time() - t_start

    plot_recovery_across_combos(rows, names, log_param_names, OUTDIR)
    plot_bias_summary_across_combos(
        rows, names, log_param_names, os.path.join(OUTDIR, "bias_summary.png"))
    plot_ridge_summary(tc_te_rows, os.path.join(OUTDIR, "ridge_summary.png"))

    # =========================================================================
    # summary
    # =========================================================================
    print(f"\n{'='*86}\nSUMMARY   ({N_COMBOS} LHS combos x {N_SEEDS} seed(s), "
          f"sigma={SIGMA:g}, rates prior="
          f"{PRIOR_CHOICES.get('rates', 'loguniform')}, {elapsed:.0f}s)")
    print(f"\n  {'param':<8}{'coverage':>10}{'mean |bias|':>13}{'n below noise':>15}")
    for pname in names:
        r = [d for d in rows if d["param"] == pname]
        cov = np.mean([d["in95"] for d in r])
        relbias = np.mean([abs(d["bayes_med"] / d["truth"] - 1.0) for d in r])
        n_below = sum(1 for d in r if d["sep"] < 1.0)
        print(f"  {pname:<8}{cov:9.0%}{relbias:13.3f}{n_below:15d}")

    print("\n  coverage      : fraction of the 15 (x N_SEEDS) combos with truth "
          "inside the 95%")
    print("                  credible interval for that parameter. Should sit "
          "near 0.95 if")
    print("                  calibrated across the JOINT space, not just near p0.")
    print("  mean |bias|   : mean |posterior median / truth - 1| across combos.")
    print("  n below noise : combos whose nearest-neighbour curve differs by "
          "< 1 sigma --")
    print("                  recovery there is an acquisition limit, not a "
          "prior/sampler failure.")
    print(f"\nfigures written to: {OUTDIR}")


if __name__ == "__main__":
    main()