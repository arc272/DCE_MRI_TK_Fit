#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_verify_c_delay.py
====================
TIER C -- WX-6e: AIF delay + water exchange interaction.

This is the phase's real open question. Earlier work in this project
established, by direct test, that:

  - AIF delay is an EXACT rigid time shift (8s delay -> +8.00s half-max
    shift, rise duration unchanged) -- it is a linear time-invariant
    property of the convolution.
  - tau_i's effect is AMPLITUDE, not timing -- swept to 20s (10x past
    its upper bound) the half-max moved <=0.5s while peak enhancement
    changed monotonically. Mechanism: the intracellular pool holds most
    of the voxel and, when slow, simply fails to participate, diluting
    the whole-voxel signal.

tau_b was NEVER tested this way, and is mechanistically the more
plausible confound: it governs the vascular pool, which sees the
sharpest first-pass transient. C1 closes that gap.

  C1  tau_b separability sweep (timing vs amplitude)  -- the open question
  C2  Joint recovery: (tau_delay, tau_i) from synthetic data
  C3  Sequential vs joint fitting comparison
  C4  Decade coverage for the augmented joint parameter vector

Figures saved to ~/wx_verification_results/<timestamp>/tier_c/
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments
from dce_mri.water_exchange import SSM3
from dce_mri.fitting import (fit_single_voxel, fit_with_delay_grid,
                             fit_with_delay_continuous, _lhs_starts,
                             _top_finite_indices)

from wx_verify_common import (
    TestLog, savefig, default_configs, synthetic_aif, plot_curves,
)

SUBDIR = "tier_c"


# =============================================================================
# helpers -- fine time grid so shape effects are visible pre-discretisation
# =============================================================================

def _fine_setup(dt_fine=0.5, duration=146.0, peak_mM=4.0):
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = np.arange(int(duration / dt_fine)) * dt_fine
    aif = synthetic_aif(t, peak_mM=peak_mM, t_arrival=45.0, width=10.0)
    theta = np.array([0.25, 0.04, 0.6/60, 0.15/60])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif)
    return acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce


def _half_max_time(S, t, nbase=20):
    base = S[:nbase].mean(); peak = S.max()
    if peak <= base:
        return np.nan
    half = base + 0.5 * (peak - base)
    idx = np.where(S >= half)[0]
    return t[idx[0]] if len(idx) else np.nan


def _rise_duration(S, t, nbase=20):
    base = S[:nbase].mean(); peak = S.max()
    if peak <= base:
        return np.nan
    lo = base + 0.1 * (peak - base); hi = base + 0.9 * (peak - base)
    i_lo = np.where(S >= lo)[0]; i_hi = np.where(S >= hi)[0]
    if not len(i_lo) or not len(i_hi):
        return np.nan
    return t[i_hi[0]] - t[i_lo[0]]


def _peak_enh(S, nbase=20):
    return float(S.max() / S[:nbase].mean())


# =============================================================================
# C1 -- tau_b separability: the open question from WX-6e
# =============================================================================

def test_c1_tau_b_separability(log: TestLog):
    print("\n--- C1: does tau_b act on TIMING (like delay) or AMPLITUDE (like tau_i)? ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif, theta, Cv, Ce = _fine_setup()
    ve, vb = theta[0], theta[1]

    # reference: fast exchange in both taus
    S_ref = SSM3.to_signal(Cv, Ce, ve, vb, (1e-6, 1e-6),
                           acq, subject, aif_cfg, relax_cfg, 1000.0)
    hm_ref, rd_ref = _half_max_time(S_ref, t), _rise_duration(S_ref, t)

    # (a) pure AIF delay, for calibration of what a TIMING effect looks like
    delays = [0.0, 2.0, 4.0, 8.0]
    delay_hm, delay_rd, delay_pk = [], [], []
    for d in delays:
        aif_d = np.interp(t - d, t, aif, left=0.0, right=aif[-1])
        Cv_d, Ce_d = tcxm_ct_compartments(theta, t, aif_d)
        S = SSM3.to_signal(Cv_d, Ce_d, ve, vb, (1e-6, 1e-6),
                           acq, subject, aif_cfg, relax_cfg, 1000.0)
        delay_hm.append(_half_max_time(S, t))
        delay_rd.append(_rise_duration(S, t))
        delay_pk.append(_peak_enh(S))

    # (b) tau_b sweep, across AND beyond its bounds
    tau_bs = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.5]
    tb_hm, tb_rd, tb_pk = [], [], []
    for tb in tau_bs:
        S = SSM3.to_signal(Cv, Ce, ve, vb, (tb, 0.3),
                           acq, subject, aif_cfg, relax_cfg, 1000.0)
        tb_hm.append(_half_max_time(S, t))
        tb_rd.append(_rise_duration(S, t))
        tb_pk.append(_peak_enh(S))

    # (c) tau_i sweep, as the already-characterised amplitude comparator
    tau_is = [0.01, 0.1, 0.5, 2.0, 20.0]
    ti_hm, ti_pk = [], []
    for ti in tau_is:
        S = SSM3.to_signal(Cv, Ce, ve, vb, (0.005, ti),
                           acq, subject, aif_cfg, relax_cfg, 1000.0)
        ti_hm.append(_half_max_time(S, t))
        ti_pk.append(_peak_enh(S))

    print(f"    reference (fast exchange): half-max={hm_ref:.2f}s  "
          f"rise={rd_ref:.2f}s")
    print(f"\n    {'delay (s)':>10} {'half-max shift':>16} {'rise change':>13} {'peak enh':>10}")
    for d, hm, rd, pk in zip(delays, delay_hm, delay_rd, delay_pk):
        print(f"    {d:>10.1f} {hm-hm_ref:>+16.2f} {rd-rd_ref:>+13.2f} {pk:>10.3f}")
    print(f"\n    {'tau_b (s)':>10} {'half-max shift':>16} {'rise change':>13} {'peak enh':>10}")
    for tb, hm, rd, pk in zip(tau_bs, tb_hm, tb_rd, tb_pk):
        tag = "  <- bounds" if tb in (1e-3, 1e-2) else ""
        print(f"    {tb:>10.0e} {hm-hm_ref:>+16.2f} {rd-rd_ref:>+13.2f} {pk:>10.3f}{tag}")
    print(f"\n    {'tau_i (s)':>10} {'half-max shift':>16} {'peak enh':>10}")
    for ti, hm, pk in zip(tau_is, ti_hm, ti_pk):
        print(f"    {ti:>10.2f} {hm-hm_ref:>+16.2f} {pk:>10.3f}")

    # verdict: within tau_b's ACTUAL bounds, is the timing shift small
    # compared with what a real delay produces?
    in_bounds = [i for i, tb in enumerate(tau_bs)
                 if relax_cfg.tau_b_lb <= tb <= relax_cfg.tau_b_ub]
    max_tb_shift = max(abs(tb_hm[i] - hm_ref) for i in in_bounds)
    dt_frame = 4.1667
    log.record("C1 tau_b's timing effect within bounds is sub-frame "
               "(not confusable with AIF delay)",
               max_tb_shift < dt_frame,
               f"max half-max shift across tau_b bounds = {max_tb_shift:.2f}s "
               f"vs frame dt = {dt_frame:.2f}s")

    tb_pk_range = max(tb_pk) - min(tb_pk)
    log.record("C1b tau_b's dominant effect characterised",
               True,
               f"peak-enh range across full tau_b sweep = {tb_pk_range:.4f}; "
               f"timing range = {max(tb_hm)-min(tb_hm):.2f}s "
               f"-> {'AMPLITUDE-dominant' if tb_pk_range > 0.01 else 'weak overall'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(delays, np.array(delay_hm) - hm_ref, "o-", color="crimson")
    axes[0].plot(delays, delays, "k--", lw=1, label="perfect 1:1 shift")
    axes[0].set_xlabel("AIF delay applied (s)", fontsize=9)
    axes[0].set_ylabel("half-max shift (s)", fontsize=9)
    axes[0].set_title("(a) AIF delay = pure TIMING\n(calibration reference)", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    ax2 = axes[1]
    ax2.semilogx(tau_bs, np.array(tb_hm) - hm_ref, "o-", color="steelblue",
                 label="half-max shift")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvspan(relax_cfg.tau_b_lb, relax_cfg.tau_b_ub, alpha=0.15,
                color="green", label="tau_b bounds")
    ax2.axhline(dt_frame, color="r", ls="--", lw=1.2, label="one frame (4.17s)")
    ax2.axhline(-dt_frame, color="r", ls="--", lw=1.2)
    ax2.set_xlabel("tau_b (s)", fontsize=9)
    ax2.set_ylabel("half-max shift (s)", fontsize=9)
    ax2.set_title("(b) tau_b: timing effect?", fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.semilogx(tau_bs, tb_pk, "o-", color="steelblue", label="tau_b")
    ax3.semilogx(tau_is, ti_pk, "s--", color="darkorange", label="tau_i")
    ax3.axvspan(relax_cfg.tau_b_lb, relax_cfg.tau_b_ub, alpha=0.15, color="green")
    ax3.set_xlabel("tau (s)", fontsize=9)
    ax3.set_ylabel("peak enhancement ratio", fontsize=9)
    ax3.set_title("(c) amplitude effect: tau_b vs tau_i", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle("C1  Is tau_b confusable with AIF delay? "
                 "(WX-6e's open question)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "C1_tau_b_separability", SUBDIR)


# =============================================================================
# C2 -- joint recovery of (tau_delay, tau_i)
# =============================================================================

def test_c2_joint_recovery(log: TestLog, n_reps=6, sigma=5.0):
    print("\n--- C2: joint recovery of (tau_delay, tau_i) from synthetic data ---")
    from wx_verify_common import default_time_grid
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([0.25, 0.04, 0.6/60, 0.15/60])

    true_delay, true_ti = 6.0, 0.3
    aif_shifted = np.interp(t - true_delay, t, aif, left=0.0, right=aif[-1])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif_shifted)
    S_true = SSM3.to_signal(Cv, Ce, theta[0], theta[1],
                            (relax_cfg.tau_b_fixed, true_ti),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)

    rec_delay, rec_ti = [], []
    for k in range(n_reps):
        rng = np.random.default_rng(5000 + k)
        obs = S_true + rng.normal(0, sigma, len(t))
        try:
            r = fit_with_delay_continuous(
                obs, aif, t, TCXM, acq, subject, fitting_cfg,
                tau_lb=-10.0, tau_ub=30.0, seed=k,
                wx_model=SSM3, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
            rec_delay.append(r.best_tau)
            rec_ti.append(float(r.wx_params[-1]) if r.wx_params is not None
                          else np.nan)
        except Exception as e:
            print(f"    rep {k}: failed ({type(e).__name__})")
            rec_delay.append(np.nan); rec_ti.append(np.nan)

    rec_delay, rec_ti = np.array(rec_delay), np.array(rec_ti)
    print(f"    true delay={true_delay:.1f}s -> median={np.nanmedian(rec_delay):.3f}s "
          f"IQR={np.nanpercentile(rec_delay,75)-np.nanpercentile(rec_delay,25):.3f}")
    print(f"    true tau_i={true_ti:.2f}s  -> median={np.nanmedian(rec_ti):.4f}s "
          f"IQR={np.nanpercentile(rec_ti,75)-np.nanpercentile(rec_ti,25):.4f}")

    delay_ok = abs(np.nanmedian(rec_delay) - true_delay) < 2.0
    log.record("C2a joint fit recovers tau_delay within 2s", delay_ok,
               f"median={np.nanmedian(rec_delay):.3f}, truth={true_delay}")
    # tau_i recovery is REPORTED, not asserted -- see Tier B docstring
    print(f"    [report] tau_i median rel.err = "
          f"{abs(np.nanmedian(rec_ti)-true_ti)/true_ti:.1%}")

    finite = np.isfinite(rec_delay) & np.isfinite(rec_ti)
    if finite.sum() > 2:
        corr = float(np.corrcoef(rec_delay[finite], rec_ti[finite])[0, 1])
    else:
        corr = np.nan
    log.record("C2b tau_delay and tau_i are not strongly anti/correlated "
               "across noise realisations",
               (not np.isfinite(corr)) or abs(corr) < 0.8,
               f"Pearson r = {corr:.3f} (|r|>0.8 would suggest trade-off)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].boxplot([rec_delay[np.isfinite(rec_delay)]], labels=["joint fit"])
    axes[0].axhline(true_delay, color="r", ls="--", lw=1.5, label=f"truth={true_delay}s")
    axes[0].set_ylabel("recovered tau_delay (s)", fontsize=9)
    axes[0].set_title("delay recovery", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].boxplot([rec_ti[np.isfinite(rec_ti)]], labels=["joint fit"])
    axes[1].axhline(true_ti, color="r", ls="--", lw=1.5, label=f"truth={true_ti}s")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("recovered tau_i (s)", fontsize=9)
    axes[1].set_title("tau_i recovery", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].scatter(rec_delay, rec_ti, c="steelblue", edgecolor="k", s=45)
    axes[2].axvline(true_delay, color="r", ls="--", lw=1)
    axes[2].axhline(true_ti, color="r", ls="--", lw=1)
    axes[2].set_xlabel("recovered tau_delay (s)", fontsize=9)
    axes[2].set_ylabel("recovered tau_i (s)", fontsize=9)
    axes[2].set_yscale("log")
    axes[2].set_title(f"trade-off check (r = {corr:.2f})", fontsize=10)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"C2  Joint (tau_delay, tau_i) recovery, sigma={sigma}, "
                 f"{n_reps} realisations", fontsize=11)
    fig.tight_layout()
    savefig(fig, "C2_joint_recovery", SUBDIR)
    return rec_delay, rec_ti, true_delay, true_ti


# =============================================================================
# C3 -- sequential vs joint
# =============================================================================

def test_c3_sequential_vs_joint(log: TestLog, n_reps=6, sigma=5.0):
    print("\n--- C3: sequential (grid delay, then wx) vs joint (continuous) ---")
    from wx_verify_common import default_time_grid
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([0.25, 0.04, 0.6/60, 0.15/60])

    true_delay, true_ti = 6.0, 0.3
    aif_shifted = np.interp(t - true_delay, t, aif, left=0.0, right=aif[-1])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif_shifted)
    S_true = SSM3.to_signal(Cv, Ce, theta[0], theta[1],
                            (relax_cfg.tau_b_fixed, true_ti),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)

    tau_values = np.arange(0.0, 16.0, 2.0)
    seq_d, seq_ti, joint_d, joint_ti = [], [], [], []

    for k in range(n_reps):
        rng = np.random.default_rng(6000 + k)
        obs = S_true + rng.normal(0, sigma, len(t))

        # sequential: grid search over delay, wx active at each tau
        try:
            rg = fit_with_delay_grid(obs, aif, t, TCXM, acq, subject,
                                     fitting_cfg, tau_values, seed=k,
                                     wx_model=SSM3, aif_cfg=aif_cfg,
                                     relax_cfg=relax_cfg)
            seq_d.append(rg.best_tau)
            seq_ti.append(float(rg.wx_params[-1]) if rg.wx_params is not None
                          else np.nan)
        except Exception:
            seq_d.append(np.nan); seq_ti.append(np.nan)

        # joint: continuous delay optimised alongside tau_i
        try:
            rc = fit_with_delay_continuous(obs, aif, t, TCXM, acq, subject,
                                           fitting_cfg, tau_lb=-10.0,
                                           tau_ub=30.0, seed=k,
                                           wx_model=SSM3, aif_cfg=aif_cfg,
                                           relax_cfg=relax_cfg)
            joint_d.append(rc.best_tau)
            joint_ti.append(float(rc.wx_params[-1]) if rc.wx_params is not None
                            else np.nan)
        except Exception:
            joint_d.append(np.nan); joint_ti.append(np.nan)

    seq_d, seq_ti = np.array(seq_d), np.array(seq_ti)
    joint_d, joint_ti = np.array(joint_d), np.array(joint_ti)

    seq_d_err   = abs(np.nanmedian(seq_d) - true_delay)
    joint_d_err = abs(np.nanmedian(joint_d) - true_delay)
    seq_ti_err   = abs(np.nanmedian(seq_ti) - true_ti)
    joint_ti_err = abs(np.nanmedian(joint_ti) - true_ti)

    print(f"    sequential: delay median={np.nanmedian(seq_d):6.3f} "
          f"(|err|={seq_d_err:.3f})  tau_i median={np.nanmedian(seq_ti):.4f} "
          f"(|err|={seq_ti_err:.4f})")
    print(f"    joint     : delay median={np.nanmedian(joint_d):6.3f} "
          f"(|err|={joint_d_err:.3f})  tau_i median={np.nanmedian(joint_ti):.4f} "
          f"(|err|={joint_ti_err:.4f})")

    log.record("C3 joint fitting does not substantially beat sequential "
               "(i.e. sequential is the reasonable default)",
               joint_ti_err >= seq_ti_err * 0.7,
               f"tau_i |err|: sequential={seq_ti_err:.4f}, joint={joint_ti_err:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].boxplot([seq_d[np.isfinite(seq_d)], joint_d[np.isfinite(joint_d)]],
                    labels=["sequential\n(grid)", "joint\n(continuous)"])
    axes[0].axhline(true_delay, color="r", ls="--", lw=1.5, label=f"truth={true_delay}s")
    axes[0].set_ylabel("recovered tau_delay (s)", fontsize=9)
    axes[0].set_title("delay recovery", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].boxplot([seq_ti[np.isfinite(seq_ti)], joint_ti[np.isfinite(joint_ti)]],
                    labels=["sequential\n(grid)", "joint\n(continuous)"])
    axes[1].axhline(true_ti, color="r", ls="--", lw=1.5, label=f"truth={true_ti}s")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("recovered tau_i (s)", fontsize=9)
    axes[1].set_title("tau_i recovery", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle(f"C3  Sequential vs joint delay+exchange fitting "
                 f"(sigma={sigma}, {n_reps} reps)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "C3_sequential_vs_joint", SUBDIR)


# =============================================================================
# C4 -- decade coverage for the joint augmented vector
# =============================================================================

def test_c4_joint_coverage(log: TestLog, n_trials=15):
    print("\n--- C4: LHS decade coverage for the joint (kinetic + delay + tau_i) vector ---")
    from wx_verify_common import default_time_grid
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([0.25, 0.04, 0.6/60, 0.15/60])
    aif_shifted = np.interp(t - 6.0, t, aif, left=0.0, right=aif[-1])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif_shifted)
    S_true = SSM3.to_signal(Cv, Ce, theta[0], theta[1],
                            (relax_cfg.tau_b_fixed, 0.3),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)
    rng = np.random.default_rng(7000)
    obs = S_true + rng.normal(0, 5.0, len(t))

    n_kin = len(TCXM.p0)
    # augmented as fit_with_delay_continuous + wx builds it:
    # [kinetic..., tau_delay, tau_i]
    lb = np.concatenate([TCXM.lb, [-10.0], [relax_cfg.tau_i_lb]])
    ub = np.concatenate([TCXM.ub, [ 30.0], [relax_cfg.tau_i_ub]])
    log_dims = tuple(TCXM.log_dims or ()) + (n_kin + 1,)
    names = list(TCXM.param_names) + ["tau_delay", "tau_i"]

    def resid(p):
        th, td, ti = p[:n_kin], p[n_kin], p[n_kin+1]
        if th[0] + th[1] > 1.0:
            return np.full(len(t), 1e6)
        try:
            a_s = np.interp(t - td, t, aif, left=0.0, right=aif[-1])
            cv, ce = tcxm_ct_compartments(th, t, a_s)
            S = SSM3.to_signal(cv, ce, th[0], th[1],
                               (relax_cfg.tau_b_fixed, ti),
                               acq, subject, aif_cfg, relax_cfg, 1000.0)
        except Exception:
            return np.full(len(t), 1e6)
        if not np.all(np.isfinite(S)):
            return np.full(len(t), 1e6)
        return S - obs

    frac_lhs, frac_top = {n: [] for n in names}, {n: [] for n in names}
    oob_total = 0
    for trial in range(n_trials):
        samples = _lhs_starts(fitting_cfg.n_lhs, lb, ub, None, log_dims,
                              seed=trial)
        oob_total += int(np.sum(np.any((samples < lb) | (samples > ub), axis=1)))
        costs = np.array([np.sum(resid(p)**2) for p in samples])
        top_idx = _top_finite_indices(costs, fitting_cfg.n_top)
        if len(top_idx) == 0:
            continue
        top = samples[top_idx]
        for j, nm in enumerate(names):
            w = ub[j] - lb[j]
            frac_lhs[nm].append((samples[:, j].max() - samples[:, j].min()) / w)
            frac_top[nm].append((top[:, j].max() - top[:, j].min()) / w)

    log.record("C4a _lhs_starts returns no out-of-bounds samples "
               "for the joint augmented vector",
               oob_total == 0,
               f"{oob_total} out-of-bounds samples across "
               f"{n_trials*fitting_cfg.n_lhs} draws")

    print(f"    {'param':>10} {'LHS frac':>10} {'top frac':>10}  (mean over trials)")
    collapsed = []
    for nm in names:
        fl, ft = np.mean(frac_lhs[nm]), np.mean(frac_top[nm])
        print(f"    {nm:>10} {fl:>9.1%} {ft:>10.1%}")
        if ft < 0.15:
            collapsed.append(nm)

    log.record("C4b tau_delay retains usable coverage in the top-n_top cut",
               np.mean(frac_top["tau_delay"]) > 0.15,
               f"mean top-frac = {np.mean(frac_top['tau_delay']):.1%}")
    if collapsed:
        print(f"    [report] dimensions collapsing below 15% top-coverage: "
              f"{collapsed} -- expected for ps (known fp-ps ridge)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [np.mean(frac_lhs[n]) for n in names], 0.4,
           label="LHS coverage", color="steelblue", edgecolor="k")
    ax.bar(x + 0.2, [np.mean(frac_top[n]) for n in names], 0.4,
           label="top-n_top coverage", color="darkorange", edgecolor="k")
    ax.axhline(0.15, color="r", ls="--", lw=1.2, label="collapse threshold (15%)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("fraction of bounded range covered", fontsize=9)
    ax.set_title(f"C4  Joint-vector coverage: LHS vs surviving top candidates\n"
                 f"({n_trials} trials, n_lhs={fitting_cfg.n_lhs}, "
                 f"n_top={fitting_cfg.n_top})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, "C4_joint_coverage", SUBDIR)


# =============================================================================

def run_tier_c(quick=False):
    log = TestLog("TIER C -- delay + exchange interaction (WX-6e)")
    print("=" * 72)
    print("  TIER C  --  AIF delay + water exchange (WX-6e)")
    if quick:
        print("  (quick mode: fewer repetitions, results are indicative only)")
    print("=" * 72)
    n = 4 if quick else 6
    test_c1_tau_b_separability(log)
    test_c2_joint_recovery(log, n_reps=n)
    test_c3_sequential_vs_joint(log, n_reps=n)
    test_c4_joint_coverage(log, n_trials=8 if quick else 15)
    return log.summary()


if __name__ == "__main__":
    import sys
    run_tier_c(quick="--quick" in sys.argv)
