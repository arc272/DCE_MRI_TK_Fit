#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_verify_b_recovery.py
=======================
TIER B -- synthetic recovery, optimizer choice, LHS budget.

Covers WX-8 items 5, 7, 8.

  B0  Noise-level visualisation
  B1  Synthetic tau_i recovery vs noise      (WX-8 #5)
  B2  Recovery vs where true tau_i sits      (WX-8 #7, second half)
  B3  TRF vs Nelder-Mead for tau_i           (WX-8 #7, first half)
  B4  n_lhs / n_top budget sensitivity, swept across sigma  (WX-8 #8)

CHANGELOG (this revision) -- seeding fix
-----------------------------------------
Previously, the noise realisation for repetition k used seed (base+k),
and the SAME integer k was passed as fit_single_voxel's seed, which
controls LHS sampling internally. This coupled two things that should
vary independently: "which noisy curve am I fitting" and "how does the
optimizer explore parameter space." It also meant every budget level
in B4 reused the identical small set of noise draws (n_reps=4-6),
giving very little statistical power to detect a real budget effect
even if one existed.

Fixed via _make_seed_pair(), which uses numpy's SeedSequence to spawn
two INDEPENDENT, reproducible child seeds per (base, repetition) pair
-- one for the noise draw, one for the LHS search -- so a fit's
optimizer exploration is no longer tied to which noise realisation
it happens to be fitting.

B4 additionally now:
  - reports the fit's final COST alongside tau_i at every budget, and
  - explicitly checks, per repetition, whether a larger budget ever
    finds a LOWER cost than n_lhs=50 for that SAME noise realisation.
This directly distinguishes "the optimum genuinely sits at a bound
for this data" (cost flat across budget) from "the search just isn't
trying hard enough" (cost should improve with budget if so) --
answering the question with the actual objective function instead of
inferring it from parameter-value patterns.

n_reps for B4 raised from 6 to 10 by default for more independent
noise realisations per budget/sigma cell.

Figures saved to ~/wx_verification_results/<timestamp>/tier_b/
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.stats import qmc

from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments
from dce_mri.water_exchange import SSM3
from dce_mri.fitting import fit_single_voxel

from wx_verify_common import (
    TestLog, savefig, default_configs, synthetic_aif, default_time_grid,
)

SUBDIR = "tier_b"

SIGMAS_EXPANDED = [0.0, 2.0, 5.0, 10.0, 15.0, 30]
SIGMAS_B4 = [5.0, 10.0, 30.0]


# =============================================================================
# seeding helper -- independent noise / LHS streams
# =============================================================================

def _make_seed_pair(base: int, rep: int) -> tuple[int, int]:
    """
    Returns (noise_seed, lhs_seed): two independent, reproducible
    integer seeds derived from (base, rep) via numpy's SeedSequence.

    Independence matters here because noise_seed controls which noisy
    curve gets fit, and lhs_seed controls how the optimizer explores
    parameter space for that curve -- these are conceptually unrelated
    and should not be tied to the same small counter, or "does more
    search help" and "is this a hard noise draw" become impossible to
    tell apart.
    """
    ss = np.random.SeedSequence([base, rep])
    noise_child, lhs_child = ss.spawn(2)
    noise_seed = int(noise_child.generate_state(1)[0])
    lhs_seed   = int(lhs_child.generate_state(1)[0])
    return noise_seed, lhs_seed


# =============================================================================
# helpers
# =============================================================================

def _make_truth(ve=0.25, vp=0.04, fp=0.6/60, ps=0.15/60):
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([ve, vp, fp, ps])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif)
    return acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce


def _generate(Cv, Ce, theta, tau_i, acq, subject, aif_cfg, relax_cfg,
              tau_b=0.005, S0=1000.0):
    return SSM3.to_signal(Cv, Ce, theta[0], theta[1], (tau_b, tau_i),
                          acq, subject, aif_cfg, relax_cfg, S0)


def _fit_tau_i(sig, aif, t, acq, subject, aif_cfg, relax_cfg, fitting_cfg,
               seed=0):
    """
    Fit via the REAL fit_single_voxel wx path.
    Returns (tau_i or nan, cost or nan, VoxelFitResult or None).
    """
    try:
        r = fit_single_voxel(sig, aif, t, TCXM, acq, subject, fitting_cfg,
                             seed=seed, wx_model=SSM3, aif_cfg=aif_cfg,
                             relax_cfg=relax_cfg)
        if r.wx_params is None:
            return np.nan, np.nan, r
        return float(r.wx_params[-1]), float(r.cost), r
    except Exception:
        return np.nan, np.nan, None


# =============================================================================
# B0 -- what does each tested sigma actually look like?
# =============================================================================

def test_b0_noise_visualization(log: TestLog, true_ti=0.3, seed=0):
    print("\n--- B0: what does each tested noise level actually look like? ---")
    acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce = _make_truth()
    S_true = _generate(Cv, Ce, theta, true_ti, acq, subject, aif_cfg, relax_cfg)
    baseline = S_true[:9].mean()

    rng_master = np.random.default_rng(seed)
    noisy = {}
    for sg in SIGMAS_EXPANDED:
        noisy[sg] = S_true + (rng_master.normal(0, sg, len(t)) if sg > 0 else 0.0)

    print(f"    true tau_i={true_ti}, baseline S0={baseline:.1f}, "
          f"peak={S_true.max():.1f}")
    print(f"    {'sigma':>6} {'% of baseline':>14} {'% of peak rise':>15}")
    rise = S_true.max() - baseline
    for sg in SIGMAS_EXPANDED:
        print(f"    {sg:>6.1f} {100*sg/baseline:>13.2f}% {100*sg/rise:>14.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(SIGMAS_EXPANDED)))
    for sg, col in zip(SIGMAS_EXPANDED, cmap):
        ax.plot(t, noisy[sg], "-", lw=1.2, color=col, alpha=0.85,
                label=f"sigma={sg:g}")
    ax.plot(t, S_true, "k-", lw=2.5, label="clean (sigma=0)", zorder=10)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("signal (a.u.)", fontsize=9)
    ax.set_title(f"(a) Same curve, increasing noise\n"
                 f"(true tau_i={true_ti}, one realisation per sigma)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pct_of_rise = [100 * sg / rise for sg in SIGMAS_EXPANDED]
    ax.plot(SIGMAS_EXPANDED, pct_of_rise, "o-", color="crimson")
    for sg, p in zip(SIGMAS_EXPANDED, pct_of_rise):
        ax.annotate(f"{p:.1f}%", (sg, p), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("sigma (a.u.)", fontsize=9)
    ax.set_ylabel("noise as % of peak-baseline rise", fontsize=9)
    ax.set_title("(b) Noise magnitude relative to the\n"
                 "signal's own enhancement", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("B0  What each tested sigma looks like on the actual curve",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "B0_noise_levels", SUBDIR)
    log.record("B0 noise-level visualisation generated", True,
               f"{len(SIGMAS_EXPANDED)} sigma levels plotted")


# =============================================================================
# B1 -- synthetic recovery vs noise
# =============================================================================

def test_b1_recovery_vs_noise(log: TestLog, n_reps=8):
    print("\n--- B1: synthetic tau_i recovery vs noise level ---")
    acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce = _make_truth()

    true_taus = [0.05, 0.3, 1.0]
    sigmas = SIGMAS_EXPANDED

    results = {}
    for ti in true_taus:
        S_true = _generate(Cv, Ce, theta, ti, acq, subject, aif_cfg, relax_cfg)
        for sg in sigmas:
            rec = []
            for k in range(n_reps):
                noise_seed, lhs_seed = _make_seed_pair(base=1000, rep=k)
                rng = np.random.default_rng(noise_seed)
                obs = S_true + (rng.normal(0, sg, len(t)) if sg > 0 else 0.0)
                val, _, _ = _fit_tau_i(obs, aif, t, acq, subject, aif_cfg,
                                       relax_cfg, fitting_cfg, seed=lhs_seed)
                rec.append(val)
            results[(ti, sg)] = np.array(rec)
            med = np.nanmedian(rec)
            print(f"    true={ti:5.2f}  sigma={sg:5.1f}  ->  "
                  f"median={med:7.4f}  IQR=[{np.nanpercentile(rec,25):.4f},"
                  f"{np.nanpercentile(rec,75):.4f}]")

    for ti in true_taus:
        rec = results[(ti, 0.0)]
        ok = np.all(np.abs(rec - ti) / ti < 0.05)
        log.record(f"B1 noiseless recovery of tau_i={ti}", ok,
                   f"median={np.nanmedian(rec):.4f}, truth={ti}")

    for ti in true_taus:
        for sg in sigmas[1:]:
            rec = results[(ti, sg)]
            rel_err = abs(np.nanmedian(rec) - ti) / ti
            spread  = (np.nanpercentile(rec, 75) - np.nanpercentile(rec, 25)) / ti
            print(f"    [report] true={ti:5.2f} sigma={sg:5.1f}: "
                  f"median rel.err={rel_err:6.1%}  IQR/truth={spread:6.1%}")

    fig, axes = plt.subplots(1, len(true_taus), figsize=(4.6*len(true_taus), 4.2),
                             sharey=False)
    for ax, ti in zip(np.atleast_1d(axes), true_taus):
        data = [results[(ti, sg)] for sg in sigmas]
        data = [d[np.isfinite(d)] for d in data]
        ax.boxplot(data, labels=[f"{s:g}" for s in sigmas])
        ax.axhline(ti, color="r", ls="--", lw=1.5, label=f"truth = {ti}")
        ax.axhline(relax_cfg.tau_i_lb, color="gray", ls=":", lw=1, label="bounds")
        ax.axhline(relax_cfg.tau_i_ub, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("noise sigma (a.u.)", fontsize=9)
        ax.set_ylabel("recovered tau_i (s)", fontsize=9)
        ax.set_title(f"true tau_i = {ti} s", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("B1  Synthetic tau_i recovery vs noise "
                 f"({n_reps} realisations each)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "B1_recovery_vs_noise", SUBDIR)
    return results


# =============================================================================
# B2 -- recovery vs where true tau_i sits
# =============================================================================

def test_b2_recovery_vs_location(log: TestLog, n_reps=6, sigma=5.0):
    print("\n--- B2: recovery reliability vs where true tau_i sits ---")
    acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce = _make_truth()

    true_taus = np.logspace(np.log10(0.02), np.log10(1.5), 7)
    med, iqr, sens = [], [], []

    for ti in true_taus:
        S_true = _generate(Cv, Ce, theta, ti, acq, subject, aif_cfg, relax_cfg)
        S_up = _generate(Cv, Ce, theta, ti*1.5, acq, subject, aif_cfg, relax_cfg)
        sens.append(float(np.sqrt(np.mean((S_up - S_true)**2))))

        rec = []
        for k in range(n_reps):
            noise_seed, lhs_seed = _make_seed_pair(base=2000, rep=k)
            rng = np.random.default_rng(noise_seed)
            obs = S_true + rng.normal(0, sigma, len(t))
            val, _, _ = _fit_tau_i(obs, aif, t, acq, subject, aif_cfg,
                                   relax_cfg, fitting_cfg, seed=lhs_seed)
            rec.append(val)
        rec = np.array(rec)

        med.append(np.nanmedian(rec))
        iqr.append(np.nanpercentile(rec, 75) - np.nanpercentile(rec, 25))
        print(f"    true={ti:6.3f}  median={np.nanmedian(rec):7.4f}  "
              f"IQR={iqr[-1]:7.4f}  local sens={sens[-1]:6.3f} a.u.")

    med, iqr, sens = np.array(med), np.array(iqr), np.array(sens)
    detectable = sens > sigma / np.sqrt(len(t))
    log.record("B2 at least one tau_i region is above the detection floor",
               bool(np.any(detectable)),
               f"{int(detectable.sum())}/{len(true_taus)} sampled tau_i values "
               f"have local sensitivity above sigma/sqrt(T)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(true_taus, true_taus, "k--", lw=1, label="perfect recovery")
    axes[0].errorbar(true_taus, med, yerr=iqr/2, fmt="o-", capsize=3,
                     color="steelblue", label="recovered (median +/- IQR/2)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("true tau_i (s)", fontsize=9)
    axes[0].set_ylabel("recovered tau_i (s)", fontsize=9)
    axes[0].set_title(f"recovery vs truth (sigma={sigma})", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].semilogx(true_taus, sens, "o-", color="darkorange",
                     label="local sensitivity (RMSE per 1.5x tau_i)")
    axes[1].axhline(sigma/np.sqrt(len(t)), color="r", ls="--", lw=1.5,
                    label=f"detection floor (sigma/sqrt(T))")
    axes[1].set_xlabel("true tau_i (s)", fontsize=9)
    axes[1].set_ylabel("curve RMSE (a.u.)", fontsize=9)
    axes[1].set_title("where is tau_i actually detectable?", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("B2  Recovery reliability depends on where true tau_i sits",
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, "B2_recovery_vs_location", SUBDIR)


# =============================================================================
# B3 -- TRF vs Nelder-Mead
# =============================================================================

def test_b3_optimizer_comparison(log: TestLog, n_reps=8, sigma=5.0):
    print("\n--- B3: TRF vs Nelder-Mead for the augmented (kinetic + tau_i) fit ---")
    acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce = _make_truth()
    true_ti = 0.3
    S_true = _generate(Cv, Ce, theta, true_ti, acq, subject, aif_cfg, relax_cfg)

    n_kin = len(TCXM.p0)
    lb = np.concatenate([TCXM.lb, [relax_cfg.tau_i_lb]])
    ub = np.concatenate([TCXM.ub, [relax_cfg.tau_i_ub]])
    log_dims = tuple(TCXM.log_dims or ()) + (n_kin,)

    def scale(u):
        o = np.empty(len(lb))
        for d in range(len(lb)):
            if d in log_dims:
                a, b = np.log10(lb[d]), np.log10(ub[d])
                o[d] = 10 ** (a + u[d]*(b-a))
            else:
                o[d] = lb[d] + u[d]*(ub[d]-lb[d])
        return o

    def make_resid(obs):
        def resid(p):
            th, ti = p[:n_kin], p[n_kin]
            if th[0] + th[1] > 1.0:
                return np.full(len(t), 1e6)
            try:
                cv, ce = tcxm_ct_compartments(th, t, aif)
                S = SSM3.to_signal(cv, ce, th[0], th[1],
                                   (relax_cfg.tau_b_fixed, ti),
                                   acq, subject, aif_cfg, relax_cfg, 1000.0)
            except Exception:
                return np.full(len(t), 1e6)
            if not np.all(np.isfinite(S)):
                return np.full(len(t), 1e6)
            return S - obs
        return resid

    trf_rec, nm_rec, trf_time, nm_time = [], [], [], []
    for k in range(n_reps):
        noise_seed, lhs_seed = _make_seed_pair(base=3000, rep=k)
        rng = np.random.default_rng(noise_seed)
        obs = S_true + rng.normal(0, sigma, len(t))
        resid = make_resid(obs)

        u = qmc.LatinHypercube(d=len(lb), seed=lhs_seed).random(n=fitting_cfg.n_lhs)
        starts = np.array([scale(u[i]) for i in range(fitting_cfg.n_lhs)])
        costs = np.array([np.sum(resid(p)**2) for p in starts])
        top = np.argsort(costs)[:fitting_cfg.n_top]

        t0 = time.perf_counter()
        best = (None, np.inf)
        for i in top:
            try:
                r = least_squares(resid, starts[i], bounds=(lb, ub),
                                  method="trf", jac="2-point", max_nfev=400)
                c = float(np.sum(r.fun**2))
                if c < best[1]:
                    best = (r.x, c)
            except Exception:
                pass
        trf_time.append(time.perf_counter() - t0)
        trf_rec.append(best[0][n_kin] if best[0] is not None else np.nan)

        t0 = time.perf_counter()
        def nm_cost(p):
            if np.any(p < lb) or np.any(p > ub):
                return 1e12
            return float(np.sum(resid(p)**2))
        best_nm = (None, np.inf)
        for i in top:
            try:
                r = minimize(nm_cost, starts[i], method="Nelder-Mead",
                             options=dict(maxiter=2000, xatol=1e-8, fatol=1e-8))
                if r.fun < best_nm[1]:
                    best_nm = (r.x, float(r.fun))
            except Exception:
                pass
        nm_time.append(time.perf_counter() - t0)
        nm_rec.append(best_nm[0][n_kin] if best_nm[0] is not None else np.nan)

    trf_rec, nm_rec = np.array(trf_rec), np.array(nm_rec)
    trf_err = np.abs(np.nanmedian(trf_rec) - true_ti)
    nm_err  = np.abs(np.nanmedian(nm_rec) - true_ti)

    print(f"    TRF        : median={np.nanmedian(trf_rec):.4f}  "
          f"IQR={np.nanpercentile(trf_rec,75)-np.nanpercentile(trf_rec,25):.4f}  "
          f"mean time={np.mean(trf_time):.2f}s")
    print(f"    Nelder-Mead: median={np.nanmedian(nm_rec):.4f}  "
          f"IQR={np.nanpercentile(nm_rec,75)-np.nanpercentile(nm_rec,25):.4f}  "
          f"mean time={np.mean(nm_time):.2f}s")

    log.record("B3 TRF is not materially worse than Nelder-Mead for tau_i",
               trf_err <= nm_err * 1.5,
               f"TRF |err|={trf_err:.4f} vs NM |err|={nm_err:.4f} "
               f"(truth={true_ti})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].boxplot([trf_rec[np.isfinite(trf_rec)], nm_rec[np.isfinite(nm_rec)]],
                    labels=["TRF", "Nelder-Mead"])
    axes[0].axhline(true_ti, color="r", ls="--", lw=1.5, label=f"truth={true_ti}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("recovered tau_i (s)", fontsize=9)
    axes[0].set_title(f"recovery accuracy (sigma={sigma}, {n_reps} reps)", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].bar(["TRF", "Nelder-Mead"], [np.mean(trf_time), np.mean(nm_time)],
                color=["steelblue", "darkorange"], edgecolor="k", alpha=0.85)
    axes[1].set_ylabel("mean wall time per fit (s)", fontsize=9)
    axes[1].set_title("cost", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("B3  Optimizer comparison for the augmented wx fit", fontsize=11)
    fig.tight_layout()
    savefig(fig, "B3_optimizer_comparison", SUBDIR)


# =============================================================================
# B4 -- n_lhs / n_top budget, swept across sigma, with cost tracking
# =============================================================================

def test_b4_lhs_budget(log: TestLog, n_reps=10, sigmas=None):
    """
    For each (sigma, budget) cell, uses n_reps INDEPENDENT noise
    realisations (via _make_seed_pair, decoupled from the LHS seed).
    Noise realisations are matched ACROSS budgets within a sigma (same
    n_reps seeds reused for every budget) so the comparison is
    apples-to-apples: same set of noisy curves, different search
    effort -- while being independent ACROSS sigma levels and
    independent of the LHS search seed within each cell.

    Tracks final fit COST alongside tau_i, and explicitly checks
    whether larger budgets ever find a lower cost than the smallest
    budget for the SAME noise realisation. This directly answers
    "would more search help" using the objective function itself,
    rather than inferring it from whether the recovered tau_i moved.
    """
    print("\n--- B4: n_lhs / n_top budget, swept across sigma ---")
    from dce_mri.config import FittingConfig
    acq, subject, aif_cfg, relax_cfg, _, t, aif, theta, Cv, Ce = _make_truth()
    true_ti = 0.3
    S_true = _generate(Cv, Ce, theta, true_ti, acq, subject, aif_cfg, relax_cfg)

    if sigmas is None:
        sigmas = SIGMAS_B4

    budgets = [(50, 5), (100, 5), (200, 10), (400, 15), (800, 20)]

    # pre-generate the noise+lhs seed pairs ONCE per sigma, reused
    # identically across every budget within that sigma
    seed_pairs = {sg: [_make_seed_pair(base=4000 + int(sg*100), rep=k)
                       for k in range(n_reps)]
                 for sg in sigmas}

    all_meds, all_iqrs, all_times = ({sg: [] for sg in sigmas} for _ in range(3))
    all_costs = {sg: [] for sg in sigmas}
    # per (sigma,budget): fraction of reps where this budget beat n_lhs=50's cost
    improved_frac = {sg: [] for sg in sigmas}

    for sigma in sigmas:
        print(f"\n    -- sigma = {sigma} --")
        cost_at_50 = None
        for bi, (n_lhs, n_top) in enumerate(budgets):
            cfg = FittingConfig(n_lhs=n_lhs, n_top=n_top, fit_space="signal")
            rec, tt, costs = [], [], []
            for k in range(n_reps):
                noise_seed, lhs_seed = seed_pairs[sigma][k]
                rng = np.random.default_rng(noise_seed)
                obs = S_true + rng.normal(0, sigma, len(t))
                t0 = time.perf_counter()
                val, cost, _ = _fit_tau_i(obs, aif, t, acq, subject, aif_cfg,
                                          relax_cfg, cfg, seed=lhs_seed)
                tt.append(time.perf_counter() - t0)
                rec.append(val)
                costs.append(cost)
            rec, costs = np.array(rec), np.array(costs)

            if bi == 0:
                cost_at_50 = costs.copy()
                frac_improved = 0.0
            else:
                with np.errstate(invalid="ignore"):
                    improved = costs < (cost_at_50 - 1e-9)
                frac_improved = float(np.nanmean(improved))

            print(f"      n_lhs={n_lhs:4d} n_top={n_top:3d}  "
                  f"tau_i={np.round(rec,4).tolist()}")
            print(f"      n_lhs={n_lhs:4d} n_top={n_top:3d}  "
                  f"cost ={np.round(costs,4).tolist()}  "
                  f"n_nan={int(np.sum(np.isnan(rec)))}/{n_reps}  "
                  f"frac_beat_n50_cost={frac_improved:.0%}")

            med = np.nanmedian(rec)
            iqr = np.nanpercentile(rec, 75) - np.nanpercentile(rec, 25)
            all_meds[sigma].append(med)
            all_iqrs[sigma].append(iqr)
            all_times[sigma].append(np.mean(tt))
            all_costs[sigma].append(np.nanmedian(costs))
            improved_frac[sigma].append(frac_improved)
            print(f"      n_lhs={n_lhs:4d} n_top={n_top:3d}  ->  "
                  f"median tau_i={med:7.4f}  IQR={iqr:7.4f}  "
                  f"median cost={np.nanmedian(costs):8.3f}  "
                  f"{np.mean(tt):5.2f}s/fit")

    # ---- the decisive check: did ANY larger budget beat n_lhs=50's cost,
    # for ANY sigma, on a meaningful fraction of independent noise draws? ---
    max_improved = max(max(improved_frac[sg][1:]) for sg in sigmas)
    log.record("B4 larger search budgets do not meaningfully find lower-cost "
               "solutions than n_lhs=50 (supports genuine floor, not "
               "under-search)",
               max_improved < 0.2,
               f"largest observed improvement rate over any budget/sigma = "
               f"{max_improved:.0%} of {n_reps} independent noise draws")

    ref_sigma = sigmas[len(sigmas)//2]
    iqrs_ref = np.array(all_iqrs[ref_sigma])
    idx_default = [i for i, b in enumerate(budgets) if b == (200, 10)][0]
    improved = (iqrs_ref[-1] < iqrs_ref[idx_default] * 0.7)
    log.record(f"B4 default n_lhs=200/n_top=10 is adequate at sigma={ref_sigma}",
               not improved,
               f"IQR at default={iqrs_ref[idx_default]:.4f}, "
               f"at n_lhs=800={iqrs_ref[-1]:.4f}")

    meds_by_sigma = np.array([all_meds[sg] for sg in sigmas])
    identical_across_sigma = np.allclose(meds_by_sigma, meds_by_sigma[0], atol=1e-6)
    log.record("B4 recovery varies with sigma (sanity check)",
               not identical_across_sigma,
               "medians identical across ALL sigma AND ALL budgets -- "
               "re-check seeding/wiring"
               if identical_across_sigma else
               "medians differ across sigma as expected")

    # ---- figure ----
    n_sig = len(sigmas)
    fig, axes = plt.subplots(n_sig, 3, figsize=(15, 3.6*n_sig))
    if n_sig == 1:
        axes = axes[None, :]
    labels = [f"{n}/{k}" for n, k in budgets]
    for row, sigma in enumerate(sigmas):
        meds, iqrs, times = all_meds[sigma], all_iqrs[sigma], all_times[sigma]
        costs_med = all_costs[sigma]

        ax0 = axes[row, 0]
        ax0.errorbar(range(len(budgets)), meds, yerr=np.array(iqrs)/2,
                     fmt="o-", capsize=3, color="steelblue")
        ax0.axhline(true_ti, color="r", ls="--", lw=1.5, label=f"truth={true_ti}")
        ax0.set_xticks(range(len(budgets))); ax0.set_xticklabels(labels, fontsize=8)
        ax0.set_ylabel("recovered tau_i (s)", fontsize=9)
        ax0.set_title(f"sigma={sigma}: tau_i vs budget", fontsize=10)
        ax0.legend(fontsize=7); ax0.grid(alpha=0.3)

        ax1 = axes[row, 1]
        ax1.plot(range(len(budgets)), costs_med, "o-", color="seagreen")
        ax1.set_xticks(range(len(budgets))); ax1.set_xticklabels(labels, fontsize=8)
        ax1.set_ylabel("median final cost (RSS)", fontsize=9)
        ax1.set_title(f"sigma={sigma}: cost vs budget\n"
                      f"(flat = genuine floor, decreasing = under-search)",
                      fontsize=9)
        ax1.grid(alpha=0.3)

        ax2 = axes[row, 2]
        ax2.bar(range(len(budgets)), improved_frac[sigma], color="darkorange",
                edgecolor="k", alpha=0.85)
        ax2.set_xticks(range(len(budgets))); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("frac. reps beating n_lhs=50 cost", fontsize=9)
        ax2.set_title(f"sigma={sigma}: does more budget\never find a better fit?",
                      fontsize=9)
        ax2.grid(alpha=0.3, axis="y")

        if row == n_sig - 1:
            ax0.set_xlabel("n_lhs / n_top", fontsize=9)
            ax1.set_xlabel("n_lhs / n_top", fontsize=9)
            ax2.set_xlabel("n_lhs / n_top", fontsize=9)

    fig.suptitle(f"B4  LHS budget sensitivity, swept across sigma "
                 f"({n_reps} independent noise draws per cell)", fontsize=12)
    fig.tight_layout()
    savefig(fig, "B4_lhs_budget", SUBDIR)


# =============================================================================

def run_tier_b(quick=False):
    log = TestLog("TIER B -- recovery, optimizer, LHS budget")
    print("=" * 72)
    print("  TIER B  --  synthetic recovery, optimizer choice, LHS budget")
    if quick:
        print("  (quick mode: fewer repetitions, results are indicative only)")
    print("=" * 72)
    n = 4 if quick else 100
    test_b0_noise_visualization(log)
    test_b1_recovery_vs_noise(log, n_reps=n)
    test_b2_recovery_vs_location(log, n_reps=max(3, n//2))
    test_b3_optimizer_comparison(log, n_reps=n)
    test_b4_lhs_budget(log, n_reps=5 if quick else 50,
                       sigmas=[5.0] if quick else None)
    return log.summary()


if __name__ == "__main__":
    import sys
    run_tier_b(quick="--quick" in sys.argv)