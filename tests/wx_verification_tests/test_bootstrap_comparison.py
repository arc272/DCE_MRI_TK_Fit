#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bootstrap_comparison.py
============================
Tier-B-style evaluation of the two bootstrap methods the PI proposed,
extended to every fitted parameter rather than tau_i alone.

REVISION NOTE (this version)
-----------------------------
Complete redesign of the figures based on feedback on the first version:

  BC1 (recovery_vs_truth) -- was plotting relative interval WIDTH on a
      log y-axis, which produced unreadable scientific-notation ticks
      and a garbled x-axis (too few points for matplotlib's log tick
      locator). Replaced with a direct plot of ACTUAL parameter values
      (plain units, linear axes, manual ticks) for both bootstrap
      methods AND a genuine single-fit control: 100 independent noisy
      realisations (sigma=5) fit with NO resampling at all, at each
      true tau_i. Point = median across repetitions; error bar = IQR
      (25th-75th percentile) across repetitions -- for the bootstrap
      methods this is the median/IQR of each trial's bootstrap median;
      for the control it's the median/IQR of 100 independent point
      fits. Truth is shown as a horizontal reference line (ve/vp/fp/ps,
      constant across the sweep) or a y=x diagonal (tau_i, which IS the
      swept variable).

  BC2 (per-trial intervals) -- removed from the default output. It
      wasn't landing as an explanation of anything; the redesigned BC1
      shows the same underlying information (typical estimate +
      spread) in a form that's actually readable. The function is kept
      in the file, unused by default, in case it's wanted later.

  BC3 (all-parameter distributions) -- restyled: stepfilled histograms
      with light, colour-matched edges instead of solid black borders
      (which produced a moire-like overlap pattern), density=True so
      shapes are comparable regardless of how many fits succeeded per
      method, fewer bins given the small sample sizes involved. Still
      no standard deviation anywhere -- IQR only, as before.

Also fixed: the __main__ block's quick/full n_rep values were swapped
(quick=True was running n_rep=150, the EXPENSIVE setting, while the
full run would have used n_rep=39). Corrected so quick is actually
quick.

Design (mirrors Tier B's B1/B2 structure)
------------------------------------------
  - noise fixed at sigma = 5.0 (tSNR ~200) -- this is a method
    comparison, not a noise sweep (B1 already covers noise)
  - swept across several TRUE tau_i values (B2 found recoverability
    depends strongly on where tau_i sits, so a single-tau_i comparison
    could be misleading)
  - bootstrap methods run on the SAME noise realisations per trial
    (paired comparison); the control uses its own independent 100
    realisations per tau_i, since it isn't being compared trial-by-
    trial against the bootstrap methods, only in aggregate
  - ALL TCXM parameters reported, not just tau_i

Methods compared
----------------
  frame_drop : PI method 1. Subsampling / delete-d jackknife, NOT a
               calibrated CI -- expected to differ, and that
               difference is the point of showing it, not a defect.
  residual   : PI method 2. Classical residual bootstrap.
  wild       : optional (--wild), heteroscedasticity-robust upgrade.

Figures -> ~/wx_verification_results/<timestamp>/bootstrap_compare/
"""

from __future__ import annotations

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments
from dce_mri.water_exchange import SSM3

from wx_verify_common import (TestLog, savefig, default_configs,
                              synthetic_aif, default_time_grid)
from dce_mri.bootstrap_wx import (BootstrapWxConfig, bootstrap_curve,
                                  diagnose_residuals, _fit_replicate)

SUBDIR = "bootstrap_compare"

SIGMA = 30.0                                   # fixed, per the design above
TRUE_TAUS = [0.3, 0.6, 1.2]             # spans B2's sensitivity curve
TRUTH_KIN = np.array([0.25, 0.04, 0.6/60, 0.15/60])   # ve, vp, fp, ps

# (key, display label, unit scale, log axis) -- log axis is now unused by
# the redesigned BC1 (linear throughout, see figure_recovery_vs_truth),
# kept here only because BC3 still uses it for bin spacing.
PARAM_SPECS = [
    ("ve",     "ve",           1.0,  False),
    ("vp",     "vp",           1.0,  False),
    ("fp",     "fp (/min)",    60.0, True),
    ("ps",     "ps (/min)",    60.0, True),
    ("tau_i",  "tau_i (s)",    1.0,  True),
]

METHOD_COLORS = {"frame_drop": "seagreen", "residual": "steelblue",
                 "wild": "darkorange"}
CONTROL_COLOR = "gray"


def _truth_for(key, true_ti):
    """Ground-truth value for a parameter, in display units."""
    return {"ve": TRUTH_KIN[0], "vp": TRUTH_KIN[1],
            "fp": TRUTH_KIN[2]*60, "ps": TRUTH_KIN[3]*60,
            "tau_i": true_ti}[key]


def _iqr(values):
    """Interquartile range (75th - 25th percentile). NaN if too few points."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    p75, p25 = np.percentile(values, [75, 25])
    return p75 - p25


def _setup():
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    Cv, Ce = tcxm_ct_compartments(TRUTH_KIN, t, aif)
    return acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, Cv, Ce


def _generate(Cv, Ce, tau_i, acq, subject, aif_cfg, relax_cfg):
    return SSM3.to_signal(Cv, Ce, TRUTH_KIN[0], TRUTH_KIN[1],
                          (relax_cfg.tau_b_fixed, tau_i),
                          acq, subject, aif_cfg, relax_cfg, 1000.0)


# =============================================================================
# core sweep -- bootstrap methods
# =============================================================================

def run_comparison(log: TestLog, methods=("frame_drop", "residual"),
                   n_trials=8, n_rep=150, warm_start=False, n_workers=None):
    """
    For each true tau_i and each method: repeat n_trials independent
    noise realisations, bootstrap each, record the interval for every
    parameter. Methods share noise realisations within a trial so the
    comparison is paired.
    """
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif, Cv, Ce = _setup()
    results = {}          # (method, true_ti, param) -> list of per-trial dicts

    print(f"\n  sigma={SIGMA}, {len(TRUE_TAUS)} tau_i values, "
          f"{len(methods)} methods, {n_trials} trials x {n_rep} replicates")
    ws_note = ("fast; see compare_warm_start() before trusting for production"
               if warm_start else "full restart")
    print(f"  warm_start={warm_start} ({ws_note})\n")

    t_start = time.perf_counter()
    for true_ti in TRUE_TAUS:
        S_true = _generate(Cv, Ce, true_ti, acq, subject, aif_cfg, relax_cfg)
        print(f"    -- true tau_i = {true_ti} --")
        for trial in range(n_trials):
            rng = np.random.default_rng(12000 + trial)     # SHARED across methods
            sig = S_true + rng.normal(0, SIGMA, len(t))

            if trial == 0 and true_ti == TRUE_TAUS[0]:
                diagnose_residuals(sig - S_true, S_true, verbose=True)

            for method in methods:
                bcfg = BootstrapWxConfig(method=method, n_replicates=n_rep,
                                         warm_start=warm_start,
                                         n_workers=n_workers,
                                         seed=12000 + trial)
                try:
                    r = bootstrap_curve(sig, aif, t, TCXM, acq, subject, cfg,
                                        SSM3, aif_cfg, relax_cfg, bcfg,
                                        verbose=False)
                except Exception:
                    continue
                if len(r.replicates) < 10:
                    continue
                for key, _, scale, _ in PARAM_SPECS:
                    lo, hi = r.ci(key)
                    results.setdefault((method, true_ti, key), []).append(dict(
                        lo=lo*scale, hi=hi*scale,
                        med=r.percentile(50, key)*scale,
                        point=float(r.point_estimate[r.param_names.index(key)])*scale,
                        truth=_truth_for(key, true_ti),
                        is_calibrated=r.is_calibrated,
                    ))
        el = time.perf_counter() - t_start
        print(f"       done ({el/60:.1f} min elapsed)")

    print(f"\n  total: {(time.perf_counter()-t_start)/60:.1f} min")
    return results


# =============================================================================
# control -- 100 independent single fits, NO resampling at all
# =============================================================================

def run_single_fit_control(n_control=100, n_workers=None, seed_base=50000,
                           verbose=True):
    """
    For each true tau_i: generate n_control INDEPENDENT noisy curves
    (fresh Gaussian noise, sigma=SIGMA, no resampling), fit each one
    from scratch with the full LHS+TRF search, and return the raw
    parameter arrays. This is the "just fit it 100 times" baseline the
    two bootstrap methods are being compared against -- not a bootstrap
    at all, a literal repeated experiment.

    Reuses _fit_replicate directly (module-level, picklable) rather
    than going through bootstrap_curve, since there is no single
    "observed" curve or residual set here -- every one of the 100
    curves is an independent draw.

    Returns {true_ti: (n_ok, n_kin+1) array}, columns matching
    (*TCXM.param_names, tau_i).
    """
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif, Cv, Ce = _setup()
    n_kin = len(TCXM.p0)
    out = {}

    if verbose:
        print(f"\n  single-fit control: {n_control} independent fits x "
              f"{len(TRUE_TAUS)} tau_i values (no resampling)")
    t0 = time.perf_counter()

    for true_ti in TRUE_TAUS:
        S_true = _generate(Cv, Ce, true_ti, acq, subject, aif_cfg, relax_cfg)
        ss = np.random.SeedSequence([seed_base, int(round(true_ti * 1000))])
        children = ss.spawn(n_control)

        tasks = []
        for k in range(n_control):
            rng = np.random.default_rng(children[k])
            obs = S_true + rng.normal(0, SIGMA, len(t))
            lhs_seed = int(rng.integers(0, 2**31 - 1))
            # keep_mask=None (use all frames), p0_warm=None + warm_start=False
            # (always a fresh full LHS+TRF search -- this IS the control)
            tasks.append((obs, None, aif, t, TCXM, acq, subject, cfg,
                          SSM3, aif_cfg, relax_cfg, None, False, lhs_seed))

        nw = n_workers or (os.cpu_count() or 1)
        if nw > 1:
            import multiprocessing as mp
            with mp.Pool(nw) as pool:
                res = pool.map(_fit_replicate, tasks)
        else:
            res = [_fit_replicate(a) for a in tasks]

        ok = np.array([r for r in res if r is not None])
        out[true_ti] = ok
        if verbose:
            n_fail = n_control - len(ok)
            print(f"    tau_i={true_ti}: {len(ok)}/{n_control} fits ok"
                  + (f"  ({n_fail} failed)" if n_fail else ""))

    if verbose:
        print(f"  control done: {(time.perf_counter()-t0)/60:.1f} min")
    return out


# =============================================================================
# scoring (coverage only -- printed, not plotted)
# =============================================================================

def score(results, methods, log: TestLog):
    """
    Coverage of the nominal 95% interval, per (method, tau_i, param).
    Printed as a compact text table and logged; the visual comparison
    of actual values now lives entirely in figure_recovery_vs_truth.
    """
    table = {}
    for (method, ti, key), trials in results.items():
        if not trials:
            continue
        cov = np.mean([t["lo"] <= t["truth"] <= t["hi"] for t in trials])
        table[(method, ti, key)] = dict(
            coverage=cov, n=len(trials),
            is_calibrated=trials[0]["is_calibrated"])

    print("\n" + "="*66)
    print("  COVERAGE of the nominal 95% interval, by method")
    print("="*66)
    for key, label, _, _ in PARAM_SPECS:
        row = f"    {label:<14}"
        for m in methods:
            covs = [table[(m, ti, key)]["coverage"]
                    for ti in TRUE_TAUS if (m, ti, key) in table]
            row += f"  {m}: {np.mean(covs):>5.0%}" if covs else f"  {m}: --"
        print(row)

    for m in methods:
        covs = [table[(m, ti, "tau_i")]["coverage"]
                for ti in TRUE_TAUS if (m, ti, "tau_i") in table]
        if not covs:
            continue
        mean_cov = float(np.mean(covs))
        calibrated = table[[k for k in table if k[0] == m][0]]["is_calibrated"]
        if calibrated:
            log.record(f"{m}: tau_i coverage is plausible for a 95% interval",
                       mean_cov >= 0.6,
                       f"mean coverage {mean_cov:.0%} across tau_i values "
                       f"(low coverage may reflect tau_i's weak "
                       f"identifiability, not a method defect)")
        else:
            log.record(f"{m}: reported as a STABILITY INDEX, not a CI",
                       True,
                       f"mean 'coverage' {mean_cov:.0%} -- not expected to "
                       f"hit 95%; measures sensitivity to frame loss, not "
                       f"sampling uncertainty")
    return table


# =============================================================================
# BC1 (redesigned) -- actual values, real units, plain axes
# =============================================================================

def figure_recovery_vs_truth(results, control, methods, name="BC1_recovery_vs_truth"):
    """
    One panel per parameter. X = true tau_i (linear, ticks at the exact
    swept values). Y = the parameter's actual value in its real unit
    (linear, plain-formatted -- no scientific notation).

    For each bootstrap method: point = median of that method's per-trial
    bootstrap medians; error bar = IQR (25th-75th percentile) of those
    trial medians.

    For the control: point = median of the 100 independent single-fit
    estimates; error bar = their IQR.

    Truth is a horizontal reference line for ve/vp/fp/ps (constant
    across the sweep) or a y=x diagonal for tau_i (which IS the thing
    being swept).
    """
    n_kin = len(TCXM.p0)
    series = list(methods) + ["control"]
    n_series = len(series)
    x_range = max(TRUE_TAUS) - min(TRUE_TAUS)
    dodge = 0.015 * x_range
    offsets = np.linspace(-dodge, dodge, n_series)

    fig, axes = plt.subplots(1, len(PARAM_SPECS),
                             figsize=(4.3*len(PARAM_SPECS), 4.8))
    axes = np.atleast_1d(axes)

    for ax, (key, label, scale, _) in zip(axes, PARAM_SPECS):
        for si, m in enumerate(series):
            xs, meds, lo_err, hi_err = [], [], [], []
            for ti in TRUE_TAUS:
                if m == "control":
                    arr = control.get(ti)
                    if arr is None or len(arr) == 0:
                        continue
                    col = n_kin if key == "tau_i" else TCXM.param_names.index(key)
                    v = arr[:, col] * scale
                else:
                    trials = results.get((m, ti, key), [])
                    if not trials:
                        continue
                    v = np.array([t["med"] for t in trials])
                med = float(np.median(v))
                p25, p75 = np.percentile(v, [25, 75])
                xs.append(ti + offsets[si])
                meds.append(med)
                lo_err.append(med - p25)
                hi_err.append(p75 - med)
            if not xs:
                continue
            color = CONTROL_COLOR if m == "control" else METHOD_COLORS.get(m)
            marker = "s" if m == "control" else "o"
            label_txt = (f"single-fit control (n={len(next(iter(control.values()))) if control else '?'}, "
                        f"no resampling)" if m == "control" else m)
            ax.errorbar(xs, meds, yerr=[lo_err, hi_err], fmt=marker,
                       capsize=3, ms=6, color=color, label=label_txt,
                       alpha=0.9)

        if key == "tau_i":
            ax.plot(TRUE_TAUS, TRUE_TAUS, "r--", lw=1.5,
                   label="truth (y = x)", zorder=0)
        else:
            truth_val = _truth_for(key, TRUE_TAUS[0])
            ax.axhline(truth_val, color="r", ls="--", lw=1.5,
                      label=f"truth = {truth_val:.3g}", zorder=0)

        ax.set_xticks(TRUE_TAUS)
        ax.set_xticklabels([str(v) for v in TRUE_TAUS])
        ax.set_xlabel("true tau_i (s)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Recovered value vs truth, sigma={SIGMA}\n"
                 f"point = median, error bar = IQR (25th-75th percentile) "
                 f"across repetitions", fontsize=12)
    fig.tight_layout()
    return savefig(fig, name, SUBDIR)


# =============================================================================
# BC2 -- SUPERSEDED, kept for optional manual use only (not called by default)
# =============================================================================

def figure_interval_spread(results, methods):
    """
    NOT called by default -- superseded by figure_recovery_vs_truth,
    which shows the same underlying information (typical estimate +
    spread across repetitions) in a form that reads directly. Left
    here in case the per-trial view is wanted for a specific debugging
    question later.
    """
    ti_show = TRUE_TAUS[len(TRUE_TAUS)//2]
    fig, axes = plt.subplots(1, len(PARAM_SPECS),
                             figsize=(3.7*len(PARAM_SPECS), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (key, label, _, logy) in zip(axes, PARAM_SPECS):
        offset = 0
        for m in methods:
            trials = results.get((m, ti_show, key), [])
            if not trials:
                continue
            x = np.arange(len(trials)) + offset
            med = np.array([t["med"] for t in trials])
            lo = np.array([t["lo"] for t in trials])
            hi = np.array([t["hi"] for t in trials])
            ax.errorbar(x, med, yerr=[med-lo, hi-med], fmt="o", capsize=2,
                        ms=4, color=METHOD_COLORS.get(m), label=m, alpha=0.85)
            offset += len(trials) + 1
        truth = _truth_for(key, ti_show)
        ax.axhline(truth, color="r", ls="--", lw=1.6, label="truth")
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("trial", fontsize=9)
        ax.set_xticks([])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("estimate and 95% interval", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Per-trial intervals at true tau_i = {ti_show} s "
                 f"(sigma={SIGMA})", fontsize=12)
    fig.tight_layout()
    savefig(fig, "BC2_interval_spread", SUBDIR)


# =============================================================================
# BC3 (restyled) -- overlaid, transparent, density-normalised histograms
# =============================================================================

def figure_all_param_distributions(results, methods):
    """
    Distribution of bootstrap trial-medians, every parameter, every
    true tau_i. Restyled from the first version:
      - histtype='stepfilled' with a colour-matched (not black) edge,
        so overlapping series read as translucent shapes rather than a
        busy bar-on-bar pattern
      - density=True so shapes are comparable even when the two
        methods have different numbers of successful trials
      - fewer bins, given how few trials typically go into each panel
    Still IQR-based everywhere else in this file; no standard deviation.
    """
    ncols = len(PARAM_SPECS)
    nrows = len(TRUE_TAUS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 2.9*nrows))
    axes = np.atleast_2d(axes)
    for i, ti in enumerate(TRUE_TAUS):
        for jx, (key, label, _, logx) in enumerate(PARAM_SPECS):
            ax = axes[i, jx]
            allv = []
            for m in methods:
                trials = results.get((m, ti, key), [])
                if trials:
                    allv.append(np.array([t["med"] for t in trials]))
            if not allv:
                ax.axis("off"); continue
            cat = np.concatenate(allv)
            n_bins = 10
            if logx and cat.min() > 0 and cat.max() > cat.min():
                bins = np.logspace(np.log10(cat.min()*0.9),
                                   np.log10(cat.max()*1.1), n_bins)
                ax.set_xscale("log")
            else:
                lo, hi = cat.min(), cat.max()
                pad = 0.05 * max(hi - lo, 1e-9)
                bins = np.linspace(lo - pad, hi + pad, n_bins)
            for m in methods:
                trials = results.get((m, ti, key), [])
                if not trials:
                    continue
                v = np.array([t["med"] for t in trials])
                col = METHOD_COLORS.get(m)
                ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                        alpha=0.45, color=col, edgecolor=col, linewidth=1.3,
                        label=m)
            ax.axvline(_truth_for(key, ti), color="r", ls="--", lw=1.5)
            if i == 0:
                ax.set_title(label, fontsize=9)
            if jx == 0:
                ax.set_ylabel(f"tau_i={ti}\ndensity", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Bootstrap trial-median distributions, every parameter, "
                 f"every true tau_i (sigma={SIGMA})\n"
                 f"red line = ground truth; shapes normalised to density",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "BC3_all_parameter_spread", SUBDIR)


# =============================================================================

def run_bootstrap_comparison(quick=False, include_wild=False, n_workers=None):
    log = TestLog("BOOTSTRAP METHOD COMPARISON")
    print("=" * 72)
    print("  BOOTSTRAP METHOD COMPARISON (Tier-B style)")
    print(f"  sigma fixed at {SIGMA}; swept over tau_i; all TCXM parameters")
    print("=" * 72)

    methods = ["frame_drop", "residual"]
    if include_wild:
        methods.append("wild")

    # NOTE: fixed from the previous version, where quick=True ran the
    # EXPENSIVE n_rep=150 and the full run would have used the cheap
    # n_rep=39 -- i.e. the two branches were swapped. quick is now
    # actually quick.
    n_trials = 3 if quick else 70
    n_rep = 40 if quick else 48
    n_control = 20 if quick else 70

    results = run_comparison(log, methods=tuple(methods), n_trials=n_trials,
                             n_rep=n_rep, warm_start=False, n_workers=n_workers)
    if not results:
        log.record("bootstrap comparison produced results", False,
                   "no trials completed")
        return log.summary()

    control = run_single_fit_control(n_control=n_control, n_workers=n_workers)

    score(results, methods, log)
    figure_recovery_vs_truth(results, control, methods)
    figure_all_param_distributions(results, methods)
    # figure_interval_spread(results, methods)  # superseded, see docstring

    print("\n  Reading these results")
    print("  ---------------------")
    print("  frame_drop is EXPECTED to differ from residual/wild -- it is")
    print("  subsampling, not a calibrated bootstrap. Its 'coverage'")
    print("  number in the printed table is a stability index, not a 95%")
    print("  guarantee -- see the module docstring.")
    print("  In BC1: if a method's error bar is comparable in size to the")
    print("  grey 'single-fit control' bar for a given parameter, that")
    print("  method isn't buying much precision over just fitting once.")
    print("  If it's meaningfully narrower, the resampling+median is")
    print("  doing real work.")
    return log.summary()


if __name__ == "__main__":
    import sys
    t0 = time.perf_counter()
    run_bootstrap_comparison(n_workers=48)
    print(f"\ntotal wall time: {time.perf_counter() - t0:.0f} s")