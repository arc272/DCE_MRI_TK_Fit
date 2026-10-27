#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_verify_e_nxl.py
==================
TIER E -- the No eXchange Limit (NXL) water-exchange model.

NXL = thesis Ch. 3 Eqs 3.36-3.37 (Bains et al. 2010 Eq. 10): the same
three pools as SSM3 (plasma b, EES e, intracellular i), each relaxing
independently, signals summed with weights (vb, ve, vi). Implemented in
closed form (water_exchange._nxl_to_signal), NOT as an expm limit.

  E1  closed form == thesis Eq 3.37 written out here, and == the suite's
      own independent sxl_limit_reference (run at hematocrit=0, the
      convention the production SSM3/NXL code uses)
  E2  SSM3(tau_b, tau_i -> inf) -> NXL, error shrinking ~1/tau
  E3  FXL(vol-weighted) >= SSM3(tau) >= NXL bracket, across a tau grid
  E4  guardrails: NXL needs aif_cfg but not relax_cfg; theta_wx must be
      empty; un-extended paths (bootstrap_wx, MFA Bayes, unknown JAX
      wx_name) refuse NXL loudly instead of silently running SSM3
  E5  parameter layout: build_residual / build_param_layout /
      assemble_prior all give exactly theta_kin (+ tau_delay)
  E6  JAX nxl_to_signal / make_predict == numpy (incl. joint delay) and
      a finite autodiff gradient
  E7  NLLS recovery (fit_single_voxel, wx_model=NXL): noiseless + noisy
  E8  ROI path (fit_roi_with_delay_continuous, wx_model=NXL)
  E9  BlackJAX NUTS (fit_curve_bayes, wx_model=NXL): convergence and
      truth inside the 95% CrI   [slow; skipped with --quick]

Figures saved to ~/wx_verification_results/<timestamp>/tier_e/

Run (from this directory):
    JAX_ENABLE_X64=1 python wx_verify_e_nxl.py [--quick]
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import time
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments
from dce_mri.water_exchange import SSM3, NXL, FXL, _nxl_to_signal
from dce_mri.fitting import (fit_single_voxel, build_residual,
                             _validate_wx_compatibility,
                             build_delay_augmented_spec)
from dce_mri.synthetic import make_parker_aif, simulate_curve, add_noise
from dce_mri.config import RoiFittingConfig

from wx_verify_common import (
    TestLog, savefig, default_configs, default_time_grid,
    sxl_limit_reference, spgr_single_pool, plot_curves,
)

SUBDIR = "tier_e"
QUICK = "--quick" in sys.argv

# well-identified operating point (verify_04: Tc = vp/fp = 5 s, Te = ve/ps = 60 s)
THETA = np.array([0.30, 0.05, 0.010, 0.005])          # ve, vp, fp, ps  (1/s)
S0 = 1000.0


def _setup():
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    acq = replace(acq, nbase=slice(0, 7))
    t = default_time_grid()                            # 36 x 4.1667 s
    aif = make_parker_aif(t, t_arrival=35.0, hematocrit=aif_cfg.hematocrit)
    return acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif


def thesis_eq_3_37(Cv, Ce, ve, vb, acq, subject, aif_cfg, S0=1000.0):
    """
    Thesis Eq 3.37, written out from the page -- with C_p = Cv/vp and
    C_e = Ce/ve (the F_p prefactor in the printed third line is a
    carry-over of the Eq 3.12 typo; see water_exchange._nxl_to_signal).
    Baselines: plasma R1 from blood T10, EES/IC from tissue T10.
    """
    vi = 1.0 - vb - ve
    TR = acq.TR / 1000.0
    a = np.deg2rad(acq.FA * subject.B1)
    R10t, R10b = 1000.0 / subject.T10, 1000.0 / aif_cfg.T10
    S = lambda R: np.sin(a) * (1 - np.exp(-TR * R)) / (1 - np.cos(a) * np.exp(-TR * R))
    num = (vi * S(R10t) + vb * S(R10b + acq.r1 * Cv / vb)
           + ve * S(R10t + acq.r1 * Ce / ve))
    den = vi * S(R10t) + vb * S(R10b) + ve * S(R10t)
    return S0 * num / den


def fxl_volume_weighted(Cv, Ce, ve, vb, acq, subject, aif_cfg, S0=1000.0):
    """tau -> 0 limit of SSM3: one pool, R1 = vb R1b + ve R1e + vi R1i."""
    vi = 1.0 - vb - ve
    TR = acq.TR / 1000.0
    a = np.deg2rad(acq.FA * subject.B1)
    R10t, R10b = 1000.0 / subject.T10, 1000.0 / aif_cfg.T10
    R = vb * (R10b + acq.r1 * Cv / vb) + ve * (R10t + acq.r1 * Ce / ve) + vi * R10t
    R0 = vb * R10b + (ve + vi) * R10t
    return S0 * spgr_single_pool(R, TR, a) / spgr_single_pool(R0, TR, a)


# =============================================================================
def test_e1_closed_form(log: TestLog):
    print("\n--- E1: NXL closed form vs independent references ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif = _setup()
    for nm, th in {"typical": THETA,
                   "high vp": np.array([0.20, 0.15, 0.03, 0.01]),
                   "small vi": np.array([0.48, 0.48, 0.01, 0.004])}.items():
        Cv, Ce = tcxm_ct_compartments(th, t, aif)
        S = NXL.to_signal(Cv, Ce, th[0], th[1], (), acq, subject, aif_cfg, None, S0)
        log.check_close(f"E1a [{nm}] NXL == thesis Eq 3.37 (written out)",
                        S, thesis_eq_3_37(Cv, Ce, th[0], th[1], acq, subject, aif_cfg, S0),
                        rtol=1e-12, atol=1e-9)
        ref = sxl_limit_reference(Cv, Ce, th[0], th[1], acq, subject,
                                  replace(aif_cfg, hematocrit=0.0), S0)
        log.check_close(f"E1b [{nm}] NXL == wx_verify_common.sxl_limit_reference(Hct=0)",
                        S, ref, rtol=1e-12, atol=1e-9)
    # baseline frames (C=0) must reproduce S0 exactly
    z = np.zeros_like(t)
    S = NXL.to_signal(z, z, THETA[0], THETA[1], (), acq, subject, aif_cfg, None, S0)
    log.check_close("E1c C=0 gives S0 at every frame", S, np.full_like(t, S0),
                    rtol=0, atol=1e-9)


def test_e2_ssm3_limit(log: TestLog):
    print("\n--- E2: SSM3(tau_b, tau_i -> inf) -> NXL ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif = _setup()
    Cv, Ce = tcxm_ct_compartments(THETA, t, aif)
    S_nxl = NXL.to_signal(Cv, Ce, THETA[0], THETA[1], (), acq, subject, aif_cfg, None, S0)
    taus = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    errs = []
    for tau in taus:
        S = SSM3.to_signal(Cv, Ce, THETA[0], THETA[1], (tau, 10 * tau),
                           acq, subject, aif_cfg, relax_cfg, S0)
        errs.append(np.max(np.abs(S - S_nxl)))
    errs = np.array(errs)
    slope = np.polyfit(np.log10(taus[1:]), np.log10(errs[1:]), 1)[0]
    log.record("E2a max|SSM3 - NXL| at tau_b=1e6 s < 1e-4 a.u.", errs[-1] < 1e-4,
               f"{errs[-1]:.2e} a.u.")
    log.record("E2b convergence is first order in 1/tau (log-log slope ~ -1)",
               abs(slope + 1) < 0.05, f"slope = {slope:.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(taus, errs, "o-k")
    ax.set_xlabel(r"$\tau_b$ (s)   [$\tau_i = 10\,\tau_b$]")
    ax.set_ylabel("max |SSM3 − NXL|  (a.u., S0=1000)")
    ax.set_title(f"E2  SSM3 → NXL as τ → ∞  (slope {slope:.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    savefig(fig, "E2_ssm3_to_nxl", SUBDIR)


def test_e3_bracket(log: TestLog):
    print("\n--- E3: FXL(vol-weighted) >= SSM3(tau) >= NXL ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif = _setup()
    Cv, Ce = tcxm_ct_compartments(THETA, t, aif)
    ve, vb = THETA[0], THETA[1]
    S_hi = fxl_volume_weighted(Cv, Ce, ve, vb, acq, subject, aif_cfg, S0)
    S_lo = NXL.to_signal(Cv, Ce, ve, vb, (), acq, subject, aif_cfg, None, S0)
    worst = np.inf
    curves = {"FXL (vol-weighted R10)": S_hi}
    for tb in (1e-4, 5e-3, 1e-1):
        for ti in (1e-3, 0.1, 0.5, 2.0, 20.0):
            S = SSM3.to_signal(Cv, Ce, ve, vb, (tb, ti), acq, subject, aif_cfg,
                               relax_cfg, S0)
            worst = min(worst, np.min(S_hi - S), np.min(S - S_lo))
            if tb == 5e-3 and ti in (0.1, 2.0, 20.0):
                curves[f"SSM3 τ_b=5 ms, τ_i={ti:g} s"] = S
    curves["NXL"] = S_lo
    log.record("E3 SSM3 bracketed by [NXL, FXL] at every frame and tau",
               worst > -1e-9, f"worst margin {worst:.2e} a.u.")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plot_curves(ax, t, curves, title="E3  exchange limits bracket SSM3")
    savefig(fig, "E3_bracket", SUBDIR)


def test_e4_guardrails(log: TestLog):
    print("\n--- E4: guardrails ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif = _setup()
    Cv, Ce = tcxm_ct_compartments(THETA, t, aif)
    try:
        _validate_wx_compatibility(NXL, "signal", TCXM, t, aif, aif_cfg, None)
        log.record("E4a NXL accepted with relax_cfg=None", True)
    except Exception as e:
        log.record("E4a NXL accepted with relax_cfg=None", False, str(e)[:80])
    log.check_raises("E4b NXL without aif_cfg raises",
                     lambda: _validate_wx_compatibility(NXL, "signal", TCXM, t, aif, None, None))
    log.check_raises("E4c SSM3 without relax_cfg still raises",
                     lambda: _validate_wx_compatibility(SSM3, "signal", TCXM, t, aif, aif_cfg, None))
    log.check_raises("E4d NXL.to_signal with a (tau_b, tau_i) tuple raises",
                     lambda: NXL.to_signal(Cv, Ce, THETA[0], THETA[1], (0.005, 0.5),
                                           acq, subject, aif_cfg, None, S0))
    log.check_raises("E4e NXL concentration-space fit raises",
                     lambda: _validate_wx_compatibility(NXL, "concentration", TCXM,
                                                        t, aif, aif_cfg, None))
    from dce_mri.bootstrap_wx import bootstrap_curve
    log.check_raises("E4f bootstrap_wx refuses NXL (not extended)",
                     lambda: bootstrap_curve(np.ones_like(t), aif, t, TCXM, acq, subject,
                                             cfg, NXL, aif_cfg, None, None),
                     exc=NotImplementedError)
    from dce_mri.bayes_fit_mfa import fit_curve_bayes_mfa
    from dce_mri.bayes import BayesConfig
    from dce_mri.config import NuisanceConfig
    log.check_raises("E4g MFA Bayes path refuses NXL (not extended)",
                     lambda: fit_curve_bayes_mfa(np.ones_like(t), aif, t, TCXM, acq,
                                                 subject, cfg,
                                                 BayesConfig(noise="gauss_marginalized_sigma"),
                                                 nuisance=NuisanceConfig(),
                                                 wx_model=NXL, aif_cfg=aif_cfg,
                                                 relax_cfg=relax_cfg),
                     exc=NotImplementedError)
    from dce_mri import jax_forward as jf
    ctx = jf.static_ctx_from_configs(acq, subject, aif_cfg)
    log.check_raises("E4h JAX make_predict refuses an unknown wx_name",
                     lambda: jf.make_predict("TCXM", t, aif, ctx, S0, wx=True,
                                             wx_name="SSM2"))


def test_e5_layout(log: TestLog):
    print("\n--- E5: parameter layout (NLLS residual, Bayes layout, prior) ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif = _setup()
    from dce_mri.bayes import BayesConfig, build_param_layout
    from dce_mri.bayes_priors import assemble_prior
    sig = simulate_curve(THETA, aif, t, TCXM, acq, subject, cfg, NXL, aif_cfg, None, S0)
    b = build_residual(sig, aif, t, TCXM, acq, subject, cfg, NXL, aif_cfg, None)
    log.record("E5a build_residual names == TCXM names",
               b.param_names == TCXM.param_names, str(b.param_names))
    log.record("E5b build_residual bounds/log_dims == TCXM's",
               np.array_equal(b.lb, TCXM.lb) and np.array_equal(b.ub, TCXM.ub)
               and b.log_dims == TCXM.log_dims)
    log.check_close("E5c residual at truth == 0 (noiseless)", b.residual_fn(THETA),
                    np.zeros_like(t), rtol=0, atol=1e-9)
    for dm, want in (("none", ("ve", "vp", "fp", "ps")),
                     ("joint", ("ve", "vp", "fp", "ps", "tau_delay"))):
        cb = BayesConfig(delay_mode=dm, noise="gauss_marginalized_sigma")
        lay = build_param_layout(TCXM, cb, aif=aif, time_var=t, acq=acq,
                                 wx_model=NXL, relax_cfg=None)
        pr = assemble_prior(lay.model, None, wx_model=NXL, delay_mode=dm,
                            choices={"rates": "uniform"})
        ok = lay.param_names == want and lay.n_wx == 0
        try:
            pr.validate_against(lay.param_names); ok_p = True
        except ValueError:
            ok_p = False
        log.record(f"E5d [{dm}] Bayes layout == {want}", ok, str(lay.param_names))
        log.record(f"E5e [{dm}] prior has no tau block and matches layout",
                   ok_p and "tau" not in pr.sources, str(pr.sources))


def test_e6_jax(log: TestLog):
    print("\n--- E6: JAX vs numpy ---")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dce_mri import jax_forward as jf
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif = _setup()
    ctx = jf.static_ctx_from_configs(acq, subject, aif_cfg)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        th = np.array([rng.uniform(0.05, 0.45), rng.uniform(0.01, 0.2),
                       np.exp(rng.uniform(np.log(1e-3), np.log(0.04))),
                       np.exp(rng.uniform(np.log(3e-4), np.log(0.03)))])
        Cv, Ce = tcxm_ct_compartments(th, t, aif)
        a = NXL.to_signal(Cv, Ce, th[0], th[1], (), acq, subject, aif_cfg, None, S0)
        bj = np.asarray(jf.nxl_to_signal(jnp.asarray(Cv), jnp.asarray(Ce),
                                         th[0], th[1], ctx, S0))
        worst = max(worst, np.max(np.abs(a - bj)) / np.max(np.abs(a)))
    log.record("E6a jf.nxl_to_signal == water_exchange._nxl_to_signal", worst < 1e-12,
               f"max rel err {worst:.2e}")

    # full predict incl. joint delay vs numpy build_residual on the augmented spec
    sig = add_noise(simulate_curve(THETA, aif, t, TCXM, acq, subject, cfg, NXL,
                                   aif_cfg, None, S0), 5.0, rng=rng)
    spec = build_delay_augmented_spec(TCXM, aif, t, acq, wx_model=NXL)
    b = build_residual(sig, aif, t, spec, acq, subject, cfg, NXL, aif_cfg, None)
    from dce_mri.signal_models import compute_S0
    S0m = float(compute_S0(sig[None, :], acq)[0])
    pred = jf.make_predict("TCXM", t, aif, ctx, S0m, wx=True, wx_name="NXL",
                           delay=True, baseline_val=float(np.mean(aif[acq.nbase])),
                           upsample=acq.upsample)
    res = jf.make_residual(pred, sig)
    worst = 0.0
    for d in (-3.0, 0.0, 2.7, 8.0):
        p = np.append(THETA, d)
        worst = max(worst, np.max(np.abs(np.asarray(res(jnp.asarray(p))) - b.residual_fn(p))))
    log.record("E6b JAX residual == numpy build_residual (NXL + joint delay)",
               worst < 1e-8, f"max abs diff {worst:.2e} a.u.")
    g = jax.grad(lambda p: jnp.sum(res(p) ** 2))(jnp.asarray(np.append(THETA, 2.7)))
    log.record("E6c autodiff gradient finite", bool(np.all(np.isfinite(np.asarray(g)))),
               np.array2string(np.asarray(g), precision=3))


def test_e7_nlls(log: TestLog):
    print("\n--- E7: NLLS recovery with wx_model=NXL ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif = _setup()
    S_clean = simulate_curve(THETA, aif, t, TCXM, acq, subject, cfg, NXL,
                             aif_cfg, None, S0)
    r = fit_single_voxel(S_clean, aif, t, TCXM, acq, subject, cfg, seed=0,
                         wx_model=NXL, aif_cfg=aif_cfg, relax_cfg=None)
    rel = np.abs(r.params - THETA) / THETA
    log.record("E7a noiseless: every parameter within 0.5% of truth",
               bool(np.all(rel < 5e-3)),
               "rel err " + np.array2string(rel, precision=1, floatmode="maxprec"))
    log.record("E7b result tags: wx_model_name == 'NXL', wx_params empty",
               r.wx_model_name == "NXL" and r.wx_params is not None
               and r.wx_params.size == 0)
    n_rep = 4 if QUICK else 12
    rng = np.random.default_rng(1)
    est = np.array([fit_single_voxel(add_noise(S_clean, 5.0, rng=rng), aif, t, TCXM,
                                     acq, subject, cfg, seed=k, wx_model=NXL,
                                     aif_cfg=aif_cfg).params for k in range(n_rep)])
    med_bias = np.median(est, axis=0) / THETA - 1
    log.record(f"E7c sigma=5 (S0=1000), {n_rep} reps: |median bias| < 10% for every param",
               bool(np.all(np.abs(med_bias) < 0.10)),
               "median rel bias " + np.array2string(med_bias, precision=3))

    # model-mismatch sanity: FXL and NXL fits of the SAME NXL data differ
    rf = fit_single_voxel(S_clean, aif, t, TCXM, acq, subject, cfg, seed=0)
    log.record("E7d FXL fit of NXL data has higher cost than NXL fit",
               rf.cost > 10 * max(r.cost, 1e-12),
               f"FXL cost {rf.cost:.3g} vs NXL cost {r.cost:.3g}")


def test_e8_roi(log: TestLog):
    print("\n--- E8: ROI path, continuous delay ---")
    from dce_mri.fitting_roi import fit_roi_with_delay_continuous
    from dce_mri.signal_models import shift_aif
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif = _setup()
    true_delay = 4.0
    aif_d = shift_aif(aif, true_delay, t, acq)
    S = simulate_curve(THETA, aif_d, t, TCXM, acq, subject, cfg, NXL, aif_cfg, None, S0)
    sig4d = np.tile(S, (2, 2, 1, 1))
    mask = np.ones((2, 2, 1), bool)
    r = fit_roi_with_delay_continuous(sig4d, mask, aif, t, TCXM, acq, subject, cfg,
                                      RoiFittingConfig(average_space="signal"),
                                      tau_lb=0.0, tau_ub=15.0, seed=0,
                                      wx_model=NXL, aif_cfg=aif_cfg, relax_cfg=None)
    rel = np.abs(np.asarray(r.best_params) - THETA) / THETA
    log.record("E8a ROI+delay: params within 1% and delay within 0.1 s",
               bool(np.all(rel < 1e-2)) and abs(r.best_tau - true_delay) < 0.1,
               f"rel err {np.array2string(rel, precision=1)}, tau={r.best_tau:.3f}")
    log.check_close("E8b St_pred reproduces the noiseless curve", r.St_pred, S,
                    rtol=0, atol=0.5)


def test_e9_blackjax(log: TestLog):
    print("\n--- E9: BlackJAX NUTS with wx_model=NXL ---")
    from dce_mri.bayes import BayesConfig, fit_curve_bayes
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif = _setup()
    rng = np.random.default_rng(2)
    sig = add_noise(simulate_curve(THETA, aif, t, TCXM, acq, subject, cfg, NXL,
                                   aif_cfg, None, S0), 5.0, rng=rng)
    cb = BayesConfig(backend="blackjax", noise="gauss_marginalized_sigma",
                     delay_mode="none", n_chains=4, n_tune=600, n_draws=1000,
                     seed=3, dense_mass_matrix=True, target_accept=0.8,
                     prior_choices={"rates": "uniform", "volfrac": "flat_masked"})
    t0 = time.time()
    res = fit_curve_bayes(sig, aif, t, TCXM, acq, subject, cfg, cb,
                          wx_model=NXL, aif_cfg=aif_cfg, relax_cfg=None)
    el = time.time() - t0
    log.record("E9a sampled vector is exactly (ve, vp, fp, ps)",
               tuple(res.param_names) == ("ve", "vp", "fp", "ps"), str(res.param_names))
    rh = max(res.rhat.values()); es = min(res.ess.values())
    log.record("E9b R-hat < 1.01, ESS > 400, 0 divergences",
               rh < 1.01 and es > 400 and res.n_divergences == 0,
               f"R-hat max {rh:.4f}, ESS min {es:.0f}, div {res.n_divergences}, {el:.0f} s")
    flat = res.chains.reshape(-1, 4)
    lo, hi = np.percentile(flat, 2.5, axis=0), np.percentile(flat, 97.5, axis=0)
    inside = (THETA >= lo) & (THETA <= hi)
    log.record("E9c truth inside 95% CrI for every parameter", bool(np.all(inside)),
               " ".join(f"{n}:[{a:.3g},{b:.3g}]" for n, a, b in
                        zip(res.param_names, lo, hi)))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for j, (ax, nm) in enumerate(zip(axes, res.param_names)):
        ax.hist(flat[:, j], 60, density=True, histtype="stepfilled", alpha=0.6,
                color="#B22234")
        ax.axvline(THETA[j], color="k", ls="--", lw=1.2, label="truth")
        ax.set_title(nm, fontsize=10)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8)
    fig.suptitle("E9  TCXM + NXL posterior (BlackJAX NUTS, σ=5, S0=1000)", fontsize=11)
    savefig(fig, "E9_blackjax_posterior", SUBDIR)


def main():
    log = TestLog("TIER E -- NXL (no-exchange limit)")
    test_e1_closed_form(log)
    test_e2_ssm3_limit(log)
    test_e3_bracket(log)
    test_e4_guardrails(log)
    test_e5_layout(log)
    test_e6_jax(log)
    test_e7_nlls(log)
    test_e8_roi(log)
    if QUICK:
        print("\n--- E9 skipped (--quick) ---")
    else:
        test_e9_blackjax(log)
    return log.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
