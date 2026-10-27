#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Sep 22 09:24:21 2026

@author: allisonchen
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_03_truth_sweep.py
========================
VERIFICATION 3 -- does recovery hold ACROSS the parameter range, or only
at the one operating point p0?

verify_02 asks "at truth = p0, do bootstrap and Bayes agree, and how does
that change with noise?". It sweeps NOISE at a FIXED truth. This script
sweeps TRUTH at a FIXED noise: for each TCXM parameter in turn, the true
value is stepped across its prior range (all other parameters held at p0),
a curve is simulated, noise is added, and both methods are asked to
recover it.

CHANGES in this version
------------------------
  (a) PRIOR VS POSTERIOR FIGURES: every fit now also saves a per-parameter
      grid (prior grey, posterior crimson, truth navy, median orange,
      R-hat/ESS per panel) -- the same figure style as verify_04/verify_05,
      generated from assemble_prior(...).sample(), i.e. the SAME prior
      object the sampler used (Beta volfrac, log-uniform rates), not a
      uniform box draw that would misrepresent a Beta prior. Log-sampled
      parameters (fp, ps, tau_delay's neighbours if WX is ever added) are
      histogrammed as log10(x) on linear bins so a log-uniform prior
      renders FLAT, not as a decaying curve.

  (b) OPTIONAL JOINT DELAY: set WITH_DELAY = True to cross every sweep
      point with TAU_DELAY_TRUE_VALUES and fit delay_mode='joint' always
      -- tau_delay becomes a free, jointly-estimated 5th parameter, and
      is tracked/plotted alongside ve/vp/fp/ps. Mirrors verify_04b.

      BOOTSTRAP IS SKIPPED WHEN WITH_DELAY=True. residual_bootstrap_curve
      has no delay-fitting capability (it calls fit_single_voxel with a
      fixed, unshifted AIF) -- giving it the truth-shifted AIF would hand
      it free information the Bayesian fit has to estimate jointly, and
      NOT shifting it would just measure model misspecification, not
      ps/fp identifiability. Same reasoning verify_04 used to drop
      bootstrap once delay entered the picture. With WITH_DELAY=False
      this script is unchanged from the original: bootstrap included,
      no delay, boot-vs-bayes plots as before.

Why this matters
----------------
A single favourable truth value cannot distinguish "this prior/method is
well calibrated" from "this prior happens to concentrate mass where I put
the truth". That distinction is invisible at one operating point and
obvious across a sweep -- which is the whole point of running this.

The sharpest version of the question is the rate parameters. A LINEAR-
uniform prior over ps = [1e-5/60, 2.5/60] puts ~90% of its mass in the top
decade; a LOG-uniform prior spreads it evenly per decade. If truth sits
high, linear-uniform will look better; if truth sits low, it should look
much worse. Run this script twice --

    PRIOR_CHOICES = {"rates": "uniform"}       # linear-uniform
    PRIOR_CHOICES = {"rates": "loguniform"}    # log-uniform  (assemble_prior default)

-- and compare the two recovery figures. If linear-uniform only wins in
the upper part of the range, that is the coincidence, not a result.

Read the CURVE SEPARATION column first
--------------------------------------
For each sweep value the script reports the RMS difference between that
curve and its neighbour, in units of the noise sd. If that number is below
~1, the two truth values produce curves the noise cannot tell apart, so
NEITHER method can recover the parameter there -- no prior, no sampler, no
amount of resampling. Poor recovery in a low-separation region is a
statement about the acquisition, not about the estimator, and should not
be read as a prior/sampler failure.

Run:
    PYTHONPATH=src python tests/Bayesian_verification_tests/verify_03_truth_sweep.py
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
from dce_mri.bayes import (BayesConfig, fit_curve_bayes, build_param_layout,
                           NOISE_OPTIONS)
from dce_mri.bayes_priors import assemble_prior
from dce_mri.bayes_validation import residual_bootstrap_curve

# house colors -- same as verify_02 / bayes_figures.py
C_BOOT   = "#2E7D32"     # bootstrap  (green)
C_BAYES  = "#B22234"     # Bayesian   (crimson)
C_TRUTH  = "#1B4F72"     # truth / identity line (navy)
C_MEDIAN = "#E67E22"     # posterior median marker (orange)
DELAY_COLORS = ["#2E7D32", "#B22234", "#1B4F72"]   # short / mid / long, when WITH_DELAY

# =============================================================================
# settings
# =============================================================================

SWEEP_PARAMS = ["ps", "fp"]   # trim this while iterating
N_SWEEP      = 10        # truth values per parameter
SIGMA        = 50.0     # fixed noise sd (verify_02's mid level; peak ~2950)

# Seeds per (parameter, truth value[, delay]). 1 is enough to see gross
# failure, but a BIAS claim needs several -- with one seed you cannot
# separate "median sits high" from "this noise draw pushed it high". Bump
# to 3-5 before telling your PI a method is biased.
N_SEEDS = 3
SEED0   = 1

# Inset from the bounds so no swept truth sits ON a bound -- a railed truth
# makes recovery look bad for reasons that have nothing to do with the prior.
EDGE_INSET = 0.05       # fraction of the range (log range for rates)

# Which parameters are swept geometrically rather than linearly. Matches
# TCXM.log_dims = (2, 3) -- fp and ps span orders of magnitude.
LOG_SWEPT = {"fp", "ps"}

# THE knob this script exists to test. See module docstring.
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}       # or {"rates": "uniform"}

# --- (b) joint delay -------------------------------------------------------
WITH_DELAY = True                       # True: cross every point with the
                                         # delays below, fit delay_mode='joint',
                                         # SKIP bootstrap (see module docstring)
TAU_DELAY_TRUE_VALUES = [ 5.0, 10.0]  # s; only used if WITH_DELAY
DELAY_BOUNDS = (-10.0, 30.0)              # fit-side prior box for tau_delay

N_REP    = 100          # bootstrap replicates (verify_02 uses 1000); unused if WITH_DELAY
N_DRAWS  = 2000
N_TUNE   = 2000
N_CHAINS = 8

BACKEND = "blackjax"      # "pymc" | "blackjax"
DENSE_MASS = True

# --- (a) prior render for the prior-vs-posterior figures -------------------
N_PRIOR_SAMPLES = 20000
PRIOR_SEED = 999

OUTDIR = os.path.join(
    os.path.expanduser("~"), "dce_bayes_results",
    "verify03_" + ("delay_" if WITH_DELAY else "") +
    datetime.now().strftime("%Y%m%d_%H%M%S"),
)


# =============================================================================
# helpers
# =============================================================================

def sweep_values(lo, hi, n, log=False, inset=EDGE_INSET):
    """n truth values spanning [lo, hi], inset from both ends."""
    if log:
        llo, lhi = np.log(lo), np.log(hi)
        pad = inset * (lhi - llo)
        return np.exp(np.linspace(llo + pad, lhi - pad, n))
    pad = inset * (hi - lo)
    return np.linspace(lo + pad, hi - pad, n)


def curve_separation(curves, sigma):
    """
    RMS difference between each curve and its nearest swept neighbour,
    expressed in units of the noise sd.

    < 1 means neighbouring truth values are indistinguishable under this
    noise level: the sweep step is smaller than the measurement can see.
    """
    n = len(curves)
    out = np.empty(n)
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        d = [np.sqrt(np.mean((curves[i] - curves[j]) ** 2)) for j in neighbours]
        out[i] = min(d) / sigma if d else np.nan
    return out


def _iqr_err(med, lo, hi):
    """Asymmetric yerr for errorbar(), clipped so log axes never get <= 0."""
    return np.vstack([np.maximum(med - lo, 0.0), np.maximum(hi - med, 0.0)])


# =============================================================================
# figures
# =============================================================================

def plot_sweep_curves(time_var, curves, values, pname, sigma, sep, save_path,
                      delay_true=None):
    """
    ALL simulated curves for one swept parameter on ONE axes, coloured by
    truth value, with a colorbar instead of a legend entry per curve.

    The shaded band is +/- sigma around the middle curve: if the curves sit
    inside each other's band, the sweep is below the noise floor and the
    recovery figure for this parameter cannot be read as an estimator
    property.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    log = pname in LOG_SWEPT
    norm = (mcolors.LogNorm(vmin=values.min(), vmax=values.max()) if log
            else mcolors.Normalize(vmin=values.min(), vmax=values.max()))
    cmap = cm.viridis

    mid = curves[len(curves) // 2]
    ax.fill_between(time_var, mid - sigma, mid + sigma,
                    color="0.6", alpha=0.30, zorder=0,
                    label=fr"$\pm\sigma$ ({sigma:g}) about the middle curve")

    for c, v, s in zip(curves, values, sep):
        ax.plot(time_var, c, lw=2.0, color=cmap(norm(v)), zorder=3,
                label=f"{pname}={v:.4g}  (sep {s:.1f}$\\sigma$)")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(f"true {pname}")

    dtag = f", tau_delay={delay_true:g}s" if delay_true is not None else ""
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.set_title(f"TCXM curves across the {pname} sweep "
                 f"(all other parameters at p0{dtag}, \u03c3 = {sigma:g})")
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prior_vs_posterior(res, truth_map, names, log_dims, lb, ub,
                            prior_samples, tag, save_path):
    """
    (a) One panel per fitted parameter: prior (grey) vs posterior
    (crimson), density-normalised on shared bins, truth (navy dashed) and
    posterior median (orange) marked, R-hat/ESS per title.

    prior_samples comes from assemble_prior(...).sample() -- the SAME
    prior the sampler used (Beta volfrac, log-uniform rates, uniform
    tau_delay if present) -- generated once in main().

    Log-sampled parameters are shown as log10(x) on LINEAR bins, so a
    log-uniform prior renders flat (density-on-geomspace-bins makes it
    look like a decaying curve instead).
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
        pk = post[:, k]

        if pname in names:
            j = names.index(pname)
            log = j in log_dims
            lo_k, hi_k = float(lb[j]), float(ub[j])
            prior_col = prior_samples[:, j]
        else:
            log = False
            lo_k, hi_k = float(np.min(pk)), float(np.max(pk))
            prior_col = None

        tr = np.log10 if log else (lambda v: v)
        bins = np.linspace(tr(lo_k), tr(hi_k), 60)

        if prior_col is not None:
            ax.hist(tr(prior_col), bins=bins, density=True, color="0.75",
                    alpha=0.9, label="prior", zorder=1)
        ax.hist(tr(pk), bins=bins, density=True, color=C_BAYES, alpha=0.75,
                label="posterior", zorder=2)

        med = float(np.median(pk))
        ax.axvline(tr(med), color=C_MEDIAN, lw=1.8, zorder=3,
                   label=f"post. median\n{med:.4g}")
        if pname in truth_map:
            tv = float(truth_map[pname])
            ax.axvline(tr(tv), color=C_TRUTH, ls="--", lw=1.8, zorder=3,
                       label=f"truth {tv:.4g}")

        rhat = res.rhat.get(pname, np.nan)
        ess  = res.ess.get(pname, np.nan)
        ax.set_title(f"{pname}   R-hat={rhat:.3f}   ESS={ess:.0f}", fontsize=10)
        ax.set_xlabel(f"log10({pname})" if log else pname)
        ax.set_ylabel("density")
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.2)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"{tag}   TCXM{' + joint delay' if WITH_DELAY else ''}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(rows, pname, save_path):
    """
    Estimated vs true, IQR as error bars, identity line.

    WITH_DELAY=False: both methods (bootstrap + Bayes), as original.
    WITH_DELAY=True:  Bayes only, points COLOURED BY injected delay, so
                      you can see whether recovery degrades as the
                      jointly-fitted delay grows.
    """
    r = [d for d in rows if d["param"] == pname]
    log = pname in LOG_SWEPT
    truth = np.array([d["truth"] for d in r])

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lims = [truth.min(), truth.max()]
    ax.plot(lims, lims, "--", color=C_TRUTH, lw=1.4, zorder=1,
            label="perfect recovery")

    if WITH_DELAY:
        for delay_true, c in zip(TAU_DELAY_TRUE_VALUES, DELAY_COLORS):
            rd = [d for d in r if d["delay"] == delay_true]
            t = np.array([d["truth"] for d in rd])
            med = np.array([d["bayes_med"] for d in rd])
            lo = np.array([d["bayes_q25"] for d in rd])
            hi = np.array([d["bayes_q75"] for d in rd])
            ax.errorbar(t, med, yerr=_iqr_err(med, lo, hi), fmt="o", ms=5.5,
                        lw=1.2, capsize=3, color=c, zorder=3,
                        label=f"delay={delay_true:g}s")
    else:
        for key, color, label, dx in (("boot", C_BOOT, "residual bootstrap", 0.985),
                                      ("bayes", C_BAYES, "Bayesian posterior", 1.015)):
            med = np.array([d[f"{key}_med"] for d in r])
            lo = np.array([d[f"{key}_q25"] for d in r])
            hi = np.array([d[f"{key}_q75"] for d in r])
            x = truth * dx if log else truth + (dx - 1.0) * (lims[1] - lims[0])
            ax.errorbar(x, med, yerr=_iqr_err(med, lo, hi),
                        fmt="o", ms=5, lw=1.4, capsize=3,
                        color=color, label=label, zorder=3)

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(f"true {pname}")
    ax.set_ylabel(f"estimated {pname}  (median, bars = IQR)")
    ax.set_title(f"Recovery of {pname} across its range\n"
                 f"prior: rates={PRIOR_CHOICES.get('rates', 'loguniform')}, "
                 f"volfrac={PRIOR_CHOICES.get('volfrac', 'flat_masked')}, "
                 f"\u03c3={SIGMA:g}")
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bias_summary(rows, params, save_path):
    """
    Relative bias (est/truth - 1) vs truth, one panel per parameter.
    WITH_DELAY=True: one line per delay, Bayes only.
    WITH_DELAY=False: boot vs Bayes, as original.
    """
    n = len(params)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                             squeeze=False)

    for k, pname in enumerate(params):
        ax = axes[k // ncol][k % ncol]
        r = [d for d in rows if d["param"] == pname]
        ax.axhline(0.0, ls="--", color=C_TRUTH, lw=1.3, zorder=1)

        if WITH_DELAY:
            for delay_true, c in zip(TAU_DELAY_TRUE_VALUES, DELAY_COLORS):
                rd = sorted([d for d in r if d["delay"] == delay_true],
                           key=lambda d: d["truth"])
                truth = np.array([d["truth"] for d in rd])
                med = np.array([d["bayes_med"] for d in rd])
                ax.plot(truth, med / truth - 1.0, "o-", ms=4.5, lw=1.3,
                        color=c, label=f"delay={delay_true:g}s", zorder=3)
        else:
            truth = np.array([d["truth"] for d in r])
            for key, color, label in (("boot", C_BOOT, "bootstrap"),
                                      ("bayes", C_BAYES, "Bayesian")):
                med = np.array([d[f"{key}_med"] for d in r])
                ax.plot(truth, med / truth - 1.0, "o-", ms=4.5, lw=1.3,
                        color=color, label=label, zorder=3)

        if pname in LOG_SWEPT:
            ax.set_xscale("log")
        ax.set_xlabel(f"true {pname}")
        ax.set_ylabel("relative bias  (est/truth - 1)")
        ax.set_title(pname)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Relative bias across each parameter's range  "
                 f"(rates prior = {PRIOR_CHOICES.get('rates', 'loguniform')}"
                 f"{', joint delay' if WITH_DELAY else ''})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def _bimodality_coeff(x):
    """
    Sarle's bimodality coefficient. UNCHANGED / still unused in main() --
    kept for parity with the original file. See tc_te_diagnostic's own
    caution about skew before wiring this into the fitting loop.
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
    """UNCHANGED from the original file -- defined but not called in
    main(), exactly as before. Left untouched; not part of this edit."""
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
    for nm, v, tv in (("Tc", Tc, tc_true), ("Te", Te, te_true),
                      ("vp", vp, truth_map.get("vp") if truth_map else None),
                      ("ps", ps, truth_map.get("ps") if truth_map else None)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)]
        bc = _bimodality_coeff(np.log(v))
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        out[nm] = dict(median=med, iqr=iqr, bc_log=bc,
                       truth=(float(tv) if tv is not None else None))
    return out


# =============================================================================
# main
# =============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"figures -> {OUTDIR}")
    print(f"prior choices: {PRIOR_CHOICES}")
    if WITH_DELAY:
        print(f"WITH_DELAY=True: crossing every sweep point with "
              f"{TAU_DELAY_TRUE_VALUES} s, delay_mode='joint'.")
        print("Bootstrap is SKIPPED in this mode -- see module docstring.")

    setup = default_test_setup()          # Parker AIF, 0-150 s
    acq, subject, cfg_fit = setup["acq"], setup["subject"], setup["cfg"]
    aif, tv = setup["aif"], setup["time_var"]

    p0 = np.asarray(TCXM.p0, float)
    names = list(TCXM.param_names)         # 4-dim; indexes the SWEEP axis

    # --- (a)+(b): fit-side layout, 5-dim if WITH_DELAY, else 4-dim ------
    cfg_shape = BayesConfig(
        prior_choices=PRIOR_CHOICES, noise="gauss_marginalized_sigma",
        delay_mode=("joint" if WITH_DELAY else "none"),
        delay_bounds=DELAY_BOUNDS, backend=BACKEND, seed=SEED0)
    layout = build_param_layout(TCXM, cfg_shape, aif=aif, time_var=tv, acq=acq)
    names_fit    = list(layout.model.param_names)
    lb_fit, ub_fit = (np.asarray(layout.model.lb, float),
                      np.asarray(layout.model.ub, float))
    log_dims_fit = layout.model.log_dims or ()

    # --- (a) prior render from the SAME prior the sampler uses ----------
    noise = NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = assemble_prior(layout.model, None,
                           delay_mode=("joint" if WITH_DELAY else "none"),
                           n_noise=noise.n_params, noise_names=noise.param_names,
                           choices=PRIOR_CHOICES, delay_bounds=DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(PRIOR_SEED),
                                 N_PRIOR_SAMPLES)[:, :layout.n_model]

    delays = TAU_DELAY_TRUE_VALUES if WITH_DELAY else [None]

    n_fits = len(SWEEP_PARAMS) * N_SWEEP * len(delays) * N_SEEDS
    print(f"\n{n_fits} Bayesian fit(s) queued "
          f"({N_DRAWS}x{N_CHAINS} draws each"
          f"{', + bootstrap' if not WITH_DELAY else ''}).")
    print("Trim SWEEP_PARAMS or N_SWEEP first if you are still iterating.\n")

    rows = []
    t_start = time.time()

    for pname in SWEEP_PARAMS:
        j = names.index(pname)
        log = pname in LOG_SWEPT
        vals = sweep_values(float(TCXM.lb[j]), float(TCXM.ub[j]),
                            N_SWEEP, log=log)

        for delay_true in delays:
            # --- simulate the whole sweep at this delay first -----------
            if WITH_DELAY:
                base = np.append(p0, delay_true)
                sim_model = layout.model            # delay-augmented; shifts AIF
            else:
                base = p0.copy()
                sim_model = TCXM

            curves = []
            for v in vals:
                tvec = base.copy()
                tvec[j] = v
                curves.append(simulate_curve(tvec, aif, tv, sim_model, acq,
                                             subject, cfg_fit))
            sep = curve_separation(curves, SIGMA)

            dtag = f"_delay{delay_true:g}s" if WITH_DELAY else ""
            plot_sweep_curves(tv, curves, vals, pname, SIGMA, sep,
                              os.path.join(OUTDIR, f"curves_sweep_{pname}{dtag}.png"),
                              delay_true=delay_true)

            dline = f"   tau_delay={delay_true:g}s" if WITH_DELAY else ""
            print(f"\n{'='*86}\nSWEEP {pname}   "
                  f"[{vals[0]:.4g} .. {vals[-1]:.4g}]   "
                  f"{'log' if log else 'linear'}-spaced{dline}")
            print(f"  {'truth':>12}{'peak enh':>11}{'sep/sigma':>11}   note")
            for v, c, s in zip(vals, curves, sep):
                note = "curves separable" if s >= 1.0 else "BELOW NOISE FLOOR"
                print(f"  {v:12.5g}{c.max()/c[:7].mean():11.2f}{s:11.2f}   {note}")

            # --- fit each truth value at this delay -----------------------
            if WITH_DELAY:
                print(f"\n  {'truth':>12}{'seed':>5}"
                      f"{'bayes med':>12}{'bayes IQR':>11}"
                      f"{'R-hat':>8}{'ESS':>7}{'in95':>6}{'sep':>7}")
            else:
                print(f"\n  {'truth':>12}{'seed':>5}"
                      f"{'boot med':>12}{'boot IQR':>11}"
                      f"{'bayes med':>12}{'bayes IQR':>11}"
                      f"{'R-hat':>8}{'ESS':>7}{'in95':>6}{'sep':>7}")

            for v, c, s in zip(vals, curves, sep):
                tvec = base.copy()
                tvec[j] = v
                truth_map = dict(zip(names_fit, tvec))

                for k in range(N_SEEDS):
                    seed = SEED0 + k
                    sig = add_noise(c, SIGMA, "gaussian",
                                    np.random.default_rng(seed))

                    if not WITH_DELAY:
                        boot = residual_bootstrap_curve(
                            sig, aif, tv, TCXM, acq, subject, cfg_fit,
                            n_rep=N_REP, seed=seed, verbose=False)
                        bs = np.asarray(boot[pname], float)
                        bs = bs[np.isfinite(bs)]

                    cfg_b_kwargs = dict(
                        prior_choices=PRIOR_CHOICES,
                        n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS,
                        noise="gauss_marginalized_sigma", seed=seed,
                        backend=BACKEND, step="nuts", dense_mass_matrix=DENSE_MASS)
                    if WITH_DELAY:
                        cfg_b_kwargs.update(delay_mode="joint",
                                            delay_bounds=DELAY_BOUNDS)
                    cfg_b = BayesConfig(**cfg_b_kwargs)
                    res = fit_curve_bayes(sig, aif, tv, TCXM, acq, subject,   # base
                                          cfg_fit, cfg_b)                     # TCXM

                    by = res.flat()[:, res.param_names.index(pname)]
                    ci_lo, ci_hi = res.ci(pname)
                    in95 = bool(ci_lo <= v <= ci_hi)

                    row = dict(
                        param=pname, truth=float(v), seed=seed, sep=float(s),
                        delay=delay_true,
                        bayes_med=float(np.median(by)),
                        bayes_q25=float(np.percentile(by, 25)),
                        bayes_q75=float(np.percentile(by, 75)),
                        ci_lo=float(ci_lo), ci_hi=float(ci_hi), in95=in95,
                        rhat=float(res.rhat.get(pname, np.nan)),
                        ess=float(res.ess.get(pname, np.nan)),
                    )
                    row["bayes_iqr"] = row["bayes_q75"] - row["bayes_q25"]
                    if not WITH_DELAY:
                        row.update(
                            boot_med=float(np.median(bs)) if bs.size else np.nan,
                            boot_q25=float(np.percentile(bs, 25)) if bs.size else np.nan,
                            boot_q75=float(np.percentile(bs, 75)) if bs.size else np.nan,
                            boot_failed=int(boot["n_failed"]))
                        row["boot_iqr"] = row["boot_q75"] - row["boot_q25"]
                    rows.append(row)

                    if WITH_DELAY:
                        print(f"  {v:12.5g}{seed:5d}"
                              f"{row['bayes_med']:12.5g}{row['bayes_iqr']:11.5g}"
                              f"{row['rhat']:8.3f}{row['ess']:7.0f}"
                              f"{'YES' if in95 else 'NO':>6}{s:7.1f}")
                    else:
                        print(f"  {v:12.5g}{seed:5d}"
                              f"{row['boot_med']:12.5g}{row['boot_iqr']:11.5g}"
                              f"{row['bayes_med']:12.5g}{row['bayes_iqr']:11.5g}"
                              f"{row['rhat']:8.3f}{row['ess']:7.0f}"
                              f"{'YES' if in95 else 'NO':>6}{s:7.1f}")

                    # --- (a) prior vs posterior, every fit ------------------
                    tag = f"sweep_{pname}_{v:.4g}{dtag}_seed{seed}"
                    plot_prior_vs_posterior(
                        res, truth_map, names_fit, log_dims_fit, lb_fit, ub_fit,
                        prior_samples, tag,
                        os.path.join(OUTDIR, f"prior_post_{tag}.png"))

            plot_recovery(rows, pname,
                          os.path.join(OUTDIR, f"recovery_{pname}{dtag}.png"))

    plot_bias_summary(rows, SWEEP_PARAMS,
                      os.path.join(OUTDIR, "bias_summary.png"))

    # =========================================================================
    # summary
    # =========================================================================
    print(f"\n{'='*86}\nSUMMARY   (rates prior = "
          f"{PRIOR_CHOICES.get('rates', 'loguniform')}, \u03c3 = {SIGMA:g}, "
          f"{time.time()-t_start:.0f}s)")

    if WITH_DELAY:
        for delay_true in TAU_DELAY_TRUE_VALUES:
            print(f"\n  delay = {delay_true:g} s")
            print(f"  {'param':<8}{'coverage':>10}{'|bias| lo-half':>16}"
                  f"{'|bias| hi-half':>16}{'n below noise':>15}")
            for pname in SWEEP_PARAMS:
                r = [d for d in rows if d["param"] == pname and d["delay"] == delay_true]
                cov = np.mean([d["in95"] for d in r])
                truth = np.array([d["truth"] for d in r])
                med = np.array([d["bayes_med"] for d in r])
                relbias = np.abs(med / truth - 1.0)
                half = np.median(truth)
                lo_h = relbias[truth <= half].mean() if np.any(truth <= half) else np.nan
                hi_h = relbias[truth > half].mean() if np.any(truth > half) else np.nan
                n_below = sum(1 for d in r if d["sep"] < 1.0)
                print(f"  {pname:<8}{cov:9.0%}{lo_h:16.3f}{hi_h:16.3f}{n_below:15d}")
    else:
        print(f"\n  {'param':<8}{'coverage':>10}{'|bias| lo-half':>16}"
              f"{'|bias| hi-half':>16}{'n below noise':>15}")
        for pname in SWEEP_PARAMS:
            r = [d for d in rows if d["param"] == pname]
            cov = np.mean([d["in95"] for d in r])
            truth = np.array([d["truth"] for d in r])
            med = np.array([d["bayes_med"] for d in r])
            relbias = np.abs(med / truth - 1.0)
            half = np.median(truth)
            lo_h = relbias[truth <= half].mean() if np.any(truth <= half) else np.nan
            hi_h = relbias[truth > half].mean() if np.any(truth > half) else np.nan
            n_below = sum(1 for d in r if d["sep"] < 1.0)
            print(f"  {pname:<8}{cov:9.0%}{lo_h:16.3f}{hi_h:16.3f}{n_below:15d}")

    print("\n  coverage       : fraction of sweep points with truth inside the "
          "95% credible interval.")
    print("                   Should sit near 0.95 if the posterior is "
          "calibrated ACROSS the range,")
    print("                   not just at p0. Well below that = the interval "
          "is too narrow somewhere.")
    print("  |bias| lo/hi   : mean |relative bias| over the lower vs upper "
          "half of the swept range.")
    print("                   A LARGE lo/hi ASYMMETRY on fp or ps is the "
          "linear-uniform signature:")
    print("                   the prior's mass sits high, so low truths get "
          "pulled upward.")
    print("  n below noise  : sweep points whose neighbouring curves differ by "
          "< 1 sigma.")
    print("                   Bias there is an ACQUISITION limit, not an "
          "estimator or prior failure.")
    print(f"\nfigures written to: {OUTDIR}")


if __name__ == "__main__":
    main()
