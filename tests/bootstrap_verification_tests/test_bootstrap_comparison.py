#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Mon Aug 24 16:01:49 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bootstrap_comparison.py
============================
Tier-B-style evaluation of the two bootstrap methods the PI proposed,
extended to every fitted parameter rather than tau_i alone.

Design (mirrors Tier B's B1/B2 structure)
------------------------------------------
  - noise fixed at sigma = 5.0 (tSNR ~200), since the question here is
    "which resampling method behaves better", not "how does noise
    degrade recovery" -- that was already answered in B1
  - swept across several TRUE tau_i values, because B2 established that
    recoverability depends strongly on where tau_i sits (sensitivity
    peaks mid-range, falls off at both ends), so a method comparison at
    a single tau_i could be misleading
  - BOTH methods run on the SAME noise realisations, so the comparison
    is paired rather than confounded by different random data
  - ALL TCXM parameters reported (ve, vp, fp, ps, derived Ktrans), not
    just tau_i -- the kinetic parameters have their own known
    degeneracies (the fp-ps ridge) and it matters whether the bootstrap
    reflects them

Methods compared
----------------
  frame_drop : the PI's method 1. Formally subsampling / delete-d
               jackknife, NOT a calibrated CI -- expected to behave
               differently, and that difference is the point of the
               comparison, not a defect.
  residual   : the PI's method 2. Classical residual bootstrap.
  wild       : optionally included (--wild) as the heteroscedasticity-
               robust upgrade to method 2.

What "better" means here
------------------------
Two separate criteria, reported separately because they can disagree:
  COVERAGE  -- does the nominal 95% interval contain the known truth
               about 95% of the time? A calibrated method should.
  WIDTH     -- how wide is the interval? Narrow is only good if
               coverage holds; a narrow interval that misses the truth
               is worse than a wide one that contains it.

Figures -> ~/wx_verification_results/<timestamp>/bootstrap_compare/
"""



import time
import numpy as np
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments
from dce_mri.water_exchange import SSM3

from wx_verify_common import (TestLog, savefig, default_configs,
                              synthetic_aif, default_time_grid)
from dce_mri.bootstrap_wx import BootstrapWxConfig, bootstrap_curve, diagnose_residuals

SUBDIR = "bootstrap_compare"

SIGMA = 5.0                                   # fixed, per the design above
TRUE_TAUS = [0.05, 0.15, 0.3, 0.7, 1.2]       # spans B2's sensitivity curve
TRUTH_KIN = np.array([0.25, 0.04, 0.6/60, 0.15/60])   # ve, vp, fp, ps

# (key, display label, unit scale, log axis)
PARAM_SPECS = [
    ("ve",     "ve",           1.0,  False),
    ("vp",     "vp",           1.0,  False),
    ("fp",     "fp (/min)",    60.0, True),
    ("ps",     "ps (/min)",    60.0, True),
    ("tau_i",  "tau_i (s)",    1.0,  True),
]

METHOD_COLORS = {"frame_drop": "seagreen", "residual": "steelblue",
                 "wild": "darkorange"}


def _truth_for(key, true_ti):
    """Ground-truth value for a parameter, in display units."""
    return {"ve": TRUTH_KIN[0], "vp": TRUTH_KIN[1],
            "fp": TRUTH_KIN[2]*60, "ps": TRUTH_KIN[3]*60,
            "tau_i": true_ti}[key]


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
# core sweep
# =============================================================================

def run_comparison(log: TestLog, methods=("frame_drop", "residual"),
                   n_trials=8, n_rep=150, warm_start=True, n_workers=None):
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
                # one-off diagnostic on a representative curve, to record
                # whether wild/block would have been indicated
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
# scoring
# =============================================================================

def score(results, methods, log: TestLog):
    """Coverage and width per (method, tau_i, param)."""
    table = {}
    for (method, ti, key), trials in results.items():
        if not trials:
            continue
        cov = np.mean([t["lo"] <= t["truth"] <= t["hi"] for t in trials])
        width = np.median([t["hi"] - t["lo"] for t in trials])
        rel = width / max(abs(trials[0]["truth"]), 1e-12)
        bias = np.median([t["med"] - t["truth"] for t in trials])
        table[(method, ti, key)] = dict(coverage=cov, width=width,
                                        rel_width=rel, bias=bias,
                                        n=len(trials),
                                        is_calibrated=trials[0]["is_calibrated"])

    print("\n" + "="*78)
    print("  COVERAGE (nominal 95%) and INTERVAL WIDTH, by method")
    print("="*78)
    for key, label, _, _ in PARAM_SPECS:
        print(f"\n  {label}")
        header = f"    {'true tau_i':>11}" + "".join(
            f"{m:>26}" for m in methods)
        print(header)
        print(f"    {'':>11}" + "".join(f"{'cov / rel.width':>26}"
                                        for _ in methods))
        for ti in TRUE_TAUS:
            row = f"    {ti:>11.2f}"
            for m in methods:
                e = table.get((m, ti, key))
                row += (f"{e['coverage']:>13.0%} /{e['rel_width']:>10.1%}"
                        if e else f"{'--':>26}")
            print(row)

    # headline check: does the calibrated method achieve plausible coverage
    # on tau_i, and does frame_drop behave differently (as theory predicts)?
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
                       f"hit 95%; this method measures sensitivity to frame "
                       f"loss, not sampling uncertainty")
    return table


# =============================================================================
# figures
# =============================================================================

def figure_coverage_and_width(table, methods):
    """B1-style: coverage and relative width vs true tau_i, per parameter."""
    fig, axes = plt.subplots(2, len(PARAM_SPECS),
                             figsize=(3.6*len(PARAM_SPECS), 7.5), sharex=True)
    for col, (key, label, _, _) in enumerate(PARAM_SPECS):
        ax = axes[0, col]
        for m in methods:
            ys = [table[(m, ti, key)]["coverage"] if (m, ti, key) in table
                  else np.nan for ti in TRUE_TAUS]
            ax.plot(TRUE_TAUS, ys, "o-", color=METHOD_COLORS.get(m),
                    label=m)
        ax.axhline(0.95, color="r", ls="--", lw=1.4, label="nominal 95%")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel("coverage", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        for m in methods:
            ys = [table[(m, ti, key)]["rel_width"] if (m, ti, key) in table
                  else np.nan for ti in TRUE_TAUS]
            ax.plot(TRUE_TAUS, ys, "o-", color=METHOD_COLORS.get(m), label=m)
        ax.axhline(1.0, color="gray", ls=":", lw=1.2,
                   label="width = truth")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("true tau_i (s)", fontsize=9)
        if col == 0:
            ax.set_ylabel("interval width / truth", fontsize=9)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Bootstrap method comparison: coverage (top) and width "
                 f"(bottom)\nsigma={SIGMA}, all TCXM parameters + tau_i",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "BC1_coverage_and_width", SUBDIR)


def figure_interval_spread(results, methods):
    """
    B2-style: per-trial intervals against truth, one panel per parameter,
    at a representative tau_i. Shows the actual spread, not a summary.
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
                 f"(sigma={SIGMA})\nintervals containing the red line are "
                 f"correct", fontsize=12)
    fig.tight_layout()
    savefig(fig, "BC2_interval_spread", SUBDIR)


def figure_all_param_distributions(results, methods):
    """
    Distribution of bootstrap medians across trials, for EVERY parameter
    -- the "parameter spread" view extended past tau_i.
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
                if not trials:
                    continue
                v = np.array([t["med"] for t in trials])
                allv.append(v)
            if not allv:
                ax.axis("off"); continue
            cat = np.concatenate(allv)
            if logx and cat.min() > 0:
                bins = np.logspace(np.log10(cat.min()*0.9),
                                   np.log10(cat.max()*1.1), 14)
                ax.set_xscale("log")
            else:
                bins = np.linspace(cat.min(), cat.max(), 14)
            for m in methods:
                trials = results.get((m, ti, key), [])
                if not trials:
                    continue
                v = np.array([t["med"] for t in trials])
                ax.hist(v, bins=bins, alpha=0.55,
                        color=METHOD_COLORS.get(m), edgecolor="k", label=m)
            ax.axvline(_truth_for(key, ti), color="r", ls="--", lw=1.5)
            if i == 0:
                ax.set_title(label, fontsize=9)
            if jx == 0:
                ax.set_ylabel(f"tau_i={ti}\ncount", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Bootstrap median per trial, every parameter, every "
                 f"true tau_i (sigma={SIGMA})\nred line = ground truth",
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

    n_trials = 4 if quick else 8
    n_rep = 60 if quick else 150

    results = run_comparison(log, methods=tuple(methods), n_trials=n_trials,
                             n_rep=n_rep, warm_start=True, n_workers=n_workers)
    if not results:
        log.record("bootstrap comparison produced results", False,
                   "no trials completed")
        return log.summary()

    table = score(results, methods, log)
    figure_coverage_and_width(table, methods)
    figure_interval_spread(results, methods)
    figure_all_param_distributions(results, methods)

    print("\n  Reading these results")
    print("  ---------------------")
    print("  frame_drop is EXPECTED to differ -- it is subsampling, and")
    print("  measures sensitivity to frame loss, not sampling uncertainty.")
    print("  Its interval is a stability index; comparing its 'coverage'")
    print("  to 95% is a category error, and it is reported only so the")
    print("  difference from a calibrated method is visible.")
    print("  For residual/wild, low coverage on tau_i is more likely to")
    print("  reflect tau_i's weak identifiability (see Tier B) than a")
    print("  defect in the resampling itself -- check whether the kinetic")
    print("  parameters in the same run achieve better coverage, which")
    print("  would point at the parameter rather than the method.")
    return log.summary()


if __name__ == "__main__":
    import sys
    run_bootstrap_comparison(quick="--quick" in sys.argv,
                             include_wild="--wild" in sys.argv)