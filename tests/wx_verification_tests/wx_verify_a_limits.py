#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_verify_a_limits.py
=====================
TIER A -- analytical limits, structural bounds, conventions, guardrails.

Fast (seconds), fully deterministic, no real data required. These are
the checks that should NEVER fail; if any of them does, something in
water_exchange.py or fitting.py is structurally wrong and the
synthetic-recovery work in Tier B is meaningless until it's fixed.

Covers WX-8 items 1, 2, 3, 4, 6.

  A1  FXL-limit consistency   (WX-8 #1)
  A2  SXL-limit consistency   (WX-8 #1)
  A3  FXL >= signal >= SXL structural bound, across a TR/FA/R1 grid
                              (WX-8 #2)
  A4  Ce/ve density + hematocrit convention regression  (WX-8 #4)
  A5  Compartment-sum regression: Cv+Ce == model.fn    (WX-8 #3)
  A6  Guardrail checks, all five rules                 (WX-8 #6)

Figures saved to ~/wx_verification_results/<timestamp>/tier_a/
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import (TCXM, GKM, tcxm_ct_analytic,
                                    tcxm_ct_compartments,
                                    gkm_ct, gkm_ct_compartments)
from dce_mri.water_exchange import SSM3, FXL, WaterExchangeModelSpec
from dce_mri.fitting import fit_single_voxel, _validate_wx_compatibility

from wx_verify_common import (
    TestLog, savefig, default_configs, synthetic_aif, default_time_grid,
    fxl_limit_reference, sxl_limit_reference, ssm3_manual, plot_curves,
)

SUBDIR = "tier_a"
TINY = 1e-6      # stand-in for tau -> 0
HUGE = 1e6       # stand-in for tau -> infinity


# =============================================================================

def _setup(ve=0.25, vp=0.04):
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([ve, vp, 0.6/60, 0.15/60])       # TCXM: ve, vp, fp, ps
    Cv, Ce = tcxm_ct_compartments(theta, t, aif)
    return acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce


# =============================================================================
# A1 -- fast-exchange limit
# =============================================================================

def test_a1_fxl_limit(log: TestLog):
    print("\n--- A1: fast-exchange limit (tau -> 0) vs volume-weighted reference ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif, theta, Cv, Ce = _setup()
    ve, vb = theta[0], theta[1]

    S_ssm3 = SSM3.to_signal(Cv, Ce, ve, vb, (TINY, TINY),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)
    S_ref  = fxl_limit_reference(Cv, Ce, ve, vb, acq, subject, aif_cfg, 1000.0)

    log.check_close("A1 SSM3(tau->0) == volume-weighted single-pool reference",
                    S_ssm3, S_ref, rtol=1e-5, atol=1e-4)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_curves(axes[0], t, {"SSM3 (tau -> 0)": S_ssm3,
                             "volume-weighted reference": S_ref},
                title="A1: fast-exchange limit")
    axes[1].plot(t, S_ssm3 - S_ref, "k-", lw=1.5)
    axes[1].axhline(0, color="r", ls=":", lw=1)
    axes[1].set_title("residual (SSM3 - reference)", fontsize=10)
    axes[1].set_xlabel("time (s)", fontsize=9)
    axes[1].set_ylabel("difference (a.u.)", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle("A1  Fast-exchange limit consistency", fontsize=11)
    savefig(fig, "A1_fxl_limit", SUBDIR)


# =============================================================================
# A2 -- slow-exchange limit
# =============================================================================

def test_a2_sxl_limit(log: TestLog):
    print("\n--- A2: slow-exchange limit (tau -> inf) vs independent-pools reference ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif, theta, Cv, Ce = _setup()
    ve, vb = theta[0], theta[1]

    S_ssm3 = SSM3.to_signal(Cv, Ce, ve, vb, (HUGE, HUGE),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)
    S_ref  = sxl_limit_reference(Cv, Ce, ve, vb, acq, subject, aif_cfg, 1000.0)

    log.check_close("A2 SSM3(tau->inf) == independent-pools reference",
                    S_ssm3, S_ref, rtol=1e-5, atol=1e-4)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_curves(axes[0], t, {"SSM3 (tau -> inf)": S_ssm3,
                             "independent-pools reference": S_ref},
                title="A2: slow-exchange limit")
    axes[1].plot(t, S_ssm3 - S_ref, "k-", lw=1.5)
    axes[1].axhline(0, color="r", ls=":", lw=1)
    axes[1].set_title("residual (SSM3 - reference)", fontsize=10)
    axes[1].set_xlabel("time (s)", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle("A2  Slow-exchange limit consistency", fontsize=11)
    savefig(fig, "A2_sxl_limit", SUBDIR)


# =============================================================================
# A3 -- structural bound: FXL >= signal >= SXL, across parameter grid
# =============================================================================

def test_a3_structural_bounds(log: TestLog):
    print("\n--- A3: FXL >= signal >= SXL across TR/FA/R1 grid ---")
    from dce_mri.config import AcquisitionConfig, SubjectConfig

    _, _, aif_cfg, relax_cfg, _, t, aif, theta, Cv, Ce = _setup()
    ve, vb = theta[0], theta[1]

    TRs   = [3.0, 4.87, 8.0]
    FAs   = [8.0, 10.0, 20.0]
    T10s  = [1000.0, 1530.0, 2000.0]
    taus  = [0.05, 0.3, 1.0]

    n_viol, n_tot = 0, 0
    margins = []
    for TR in TRs:
        for FA in FAs:
            for T10 in T10s:
                acq = AcquisitionConfig(TR=TR, FA=FA, r1=3.9, nbase=slice(0, 11))
                subj = SubjectConfig(subject_id="grid", T10=T10, B1=1.0)
                S_fxl = SSM3.to_signal(Cv, Ce, ve, vb, (TINY, TINY),
                                       acq, subj, aif_cfg, relax_cfg, 1000.0)
                S_sxl = SSM3.to_signal(Cv, Ce, ve, vb, (HUGE, HUGE),
                                       acq, subj, aif_cfg, relax_cfg, 1000.0)
                for ti in taus:
                    S = SSM3.to_signal(Cv, Ce, ve, vb, (0.005, ti),
                                       acq, subj, aif_cfg, relax_cfg, 1000.0)
                    hi = np.maximum(S_fxl, S_sxl)
                    lo = np.minimum(S_fxl, S_sxl)
                    # tolerance absorbs float noise at baseline where all
                    # three coincide exactly
                    viol = np.any(S > hi + 1e-6) or np.any(S < lo - 1e-6)
                    n_viol += int(viol)
                    n_tot  += 1
                    margins.append(float(np.min(np.minimum(hi - S, S - lo))))

    log.record("A3 signal bracketed by FXL/SXL limits at every grid point",
               n_viol == 0,
               f"{n_tot - n_viol}/{n_tot} grid points OK, "
               f"worst margin {min(margins):.2e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margins, bins=30, color="steelblue", edgecolor="k", alpha=0.8)
    ax.axvline(0, color="r", ls="--", lw=1.5, label="violation threshold")
    ax.set_xlabel("min margin to nearest limit (a.u.)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(f"A3  Structural bound margins ({n_tot} grid points)\n"
                 f"all margins > 0 means signal always bracketed by FXL/SXL",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, "A3_structural_bounds", SUBDIR)


# =============================================================================
# A4 -- density / hematocrit convention regression
# =============================================================================

def test_a4_conventions(log: TestLog):
    print("\n--- A4: Ce/ve density + hematocrit convention regression ---")
    acq, subject, aif_cfg, relax_cfg, _, t, aif, theta, Cv, Ce = _setup()
    ve, vb = theta[0], theta[1]
    tau_b, tau_i = 0.005, 0.3

    S_real = SSM3.to_signal(Cv, Ce, ve, vb, (tau_b, tau_i),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)

    S_correct = ssm3_manual(Cv, Ce, ve, vb, tau_b, tau_i, acq, subject,
                            aif_cfg, apply_density=True, apply_hematocrit=True)
    S_no_dens = ssm3_manual(Cv, Ce, ve, vb, tau_b, tau_i, acq, subject,
                            aif_cfg, apply_density=False, apply_hematocrit=True)
    S_no_hct  = ssm3_manual(Cv, Ce, ve, vb, tau_b, tau_i, acq, subject,
                            aif_cfg, apply_density=True, apply_hematocrit=False)

    log.check_close("A4a SSM3 matches density+hematocrit convention",
                    S_real, S_correct, rtol=1e-6, atol=1e-6)

    d_dens = float(np.max(np.abs(S_correct - S_no_dens)))
    d_hct  = float(np.max(np.abs(S_correct - S_no_hct)))
    log.record("A4b density division is load-bearing (removing it changes signal)",
               d_dens > 1.0, f"max|diff| without /ve,/vb = {d_dens:.2f} a.u.")
    log.record("A4c hematocrit dilution is load-bearing",
               d_hct > 0.1, f"max|diff| without (1-Hct) = {d_hct:.2f} a.u.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_curves(ax, t, {
        "SSM3 (real module)":            S_real,
        "manual: density + hematocrit":  S_correct,
        "manual: NO density division":   S_no_dens,
        "manual: NO hematocrit":         S_no_hct,
    }, title="A4  Convention regression -- real SSM3 must match the\n"
             "density+hematocrit variant, and differ from the others")
    savefig(fig, "A4_conventions", SUBDIR)


# =============================================================================
# A5 -- compartment-sum regression
# =============================================================================

def test_a5_compartment_sum(log: TestLog):
    print("\n--- A5: compartment sum Cv+Ce == model.fn ---")
    _, _, _, _, _, t, aif, theta, _, _ = _setup()

    Cv_t, Ce_t = tcxm_ct_compartments(theta, t, aif)
    Ct_t = tcxm_ct_analytic(theta, t, aif)
    log.check_close("A5a TCXM: Cv+Ce == tcxm_ct_analytic",
                    Cv_t + Ce_t, Ct_t, rtol=1e-9, atol=1e-12)

    theta_g = np.array([0.15/60, 0.25, 0.04])          # GKM: Ktrans, ve, vp
    Cv_g, Ce_g = gkm_ct_compartments(theta_g, t, aif)
    Ct_g = gkm_ct(theta_g, t, aif)
    log.check_close("A5b GKM: Cv+Ce == gkm_ct",
                    Cv_g + Ce_g, Ct_g, rtol=1e-9, atol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_curves(axes[0], t, {"Cv": Cv_t, "Ce": Ce_t, "Cv+Ce": Cv_t + Ce_t,
                             "tcxm_ct_analytic (reference)": Ct_t},
                title="A5a  TCXM compartments", ylabel="[Gd] (mM)")
    plot_curves(axes[1], t, {"Cv": Cv_g, "Ce": Ce_g, "Cv+Ce": Cv_g + Ce_g,
                             "gkm_ct (reference)": Ct_g},
                title="A5b  GKM compartments", ylabel="[Gd] (mM)")
    fig.suptitle("A5  Compartment-sum regression", fontsize=11)
    savefig(fig, "A5_compartment_sum", SUBDIR)


# =============================================================================
# A6 -- guardrail rules
# =============================================================================

def test_a6_guardrails(log: TestLog):
    print("\n--- A6: _validate_wx_compatibility, all rules ---")
    from dce_mri.config import FittingConfig
    acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta, Cv, Ce = _setup()

    # rule 0: wx_model=None is a no-op
    try:
        _validate_wx_compatibility(None, "signal", TCXM, t, aif, None, None)
        log.record("A6a wx_model=None is a no-op (no raise)", True)
    except Exception as e:
        log.record("A6a wx_model=None is a no-op (no raise)", False, str(e)[:70])

    # rule 1: missing aif_cfg / relax_cfg
    log.check_raises("A6b missing aif_cfg raises",
                     lambda: _validate_wx_compatibility(
                         SSM3, "signal", TCXM, t, aif, None, relax_cfg))
    log.check_raises("A6c missing relax_cfg raises",
                     lambda: _validate_wx_compatibility(
                         SSM3, "signal", TCXM, t, aif, aif_cfg, None))

    # rule 2: fit_space must be 'signal'
    log.check_raises("A6d fit_space='concentration' raises",
                     lambda: _validate_wx_compatibility(
                         SSM3, "concentration", TCXM, t, aif, aif_cfg, relax_cfg))

    # rule 4: pool-count mismatch -- FXL, and a synthetic 2-pool spec
    log.check_raises("A6e FXL (n_pools=1) rejected",
                     lambda: _validate_wx_compatibility(
                         FXL, "signal", TCXM, t, aif, aif_cfg, relax_cfg))

    fake_2pool = WaterExchangeModelSpec(
        name="FAKE_2POOL", n_pools=2, n_params=1,
        param_names=("tau_i",), to_signal=SSM3.to_signal,
    )
    log.check_raises("A6f synthetic 2-pool wx_model rejected against TCXM",
                     lambda: _validate_wx_compatibility(
                         fake_2pool, "signal", TCXM, t, aif, aif_cfg, relax_cfg))
    log.check_raises("A6g synthetic 2-pool wx_model rejected against GKM",
                     lambda: _validate_wx_compatibility(
                         fake_2pool, "signal", GKM, t, aif, aif_cfg, relax_cfg))

    # rule 5: average_space must be 'signal' when supplied
    log.check_raises("A6h average_space='concentration' raises",
                     lambda: _validate_wx_compatibility(
                         SSM3, "signal", TCXM, t, aif, aif_cfg, relax_cfg,
                         average_space="concentration"))
    try:
        _validate_wx_compatibility(SSM3, "signal", TCXM, t, aif, aif_cfg,
                                   relax_cfg, average_space="signal")
        log.record("A6i average_space='signal' accepted", True)
    except Exception as e:
        log.record("A6i average_space='signal' accepted", False, str(e)[:70])

    # rule 3 depends on the GKM fn_compartments wiring being correct;
    # exercise it explicitly since it was a real bug earlier in this work
    try:
        _validate_wx_compatibility(SSM3, "signal", GKM, t, aif, aif_cfg, relax_cfg)
        log.record("A6j GKM+SSM3 accepted (fn_compartments wired correctly)", True)
    except Exception as e:
        log.record("A6j GKM+SSM3 accepted (fn_compartments wired correctly)",
                   False, f"{type(e).__name__}: {str(e)[:60]}")


# =============================================================================

def run_tier_a():
    log = TestLog("TIER A -- limits, bounds, conventions, guardrails")
    print("=" * 72)
    print("  TIER A  --  analytical limits, structural bounds, conventions")
    print("=" * 72)
    test_a1_fxl_limit(log)
    test_a2_sxl_limit(log)
    test_a3_structural_bounds(log)
    test_a4_conventions(log)
    test_a5_compartment_sum(log)
    test_a6_guardrails(log)
    return log.summary()


if __name__ == "__main__":
    run_tier_a()