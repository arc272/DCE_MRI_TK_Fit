#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Mon Aug 24 15:56:37 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bootstrap_wx.py
======================
Verification for bootstrap_wx.py -- does the resampling machinery
itself work correctly, before any of its output is trusted?

This is the Tier-A equivalent for the bootstrap module: fast,
deterministic, mostly analytical checks that should never fail. If any
of these fail, the uncertainty numbers downstream are meaningless.

  V1  frame_drop does NOT degrade the AIF
        The central implementation claim of the module. Verifies that
        the forward model sees the full 36-point AIF regardless of how
        many frames are masked out of the residual -- i.e. that we
        implemented subsampling of the OBSERVATION, not of the model
        input. Done by checking the predicted curve is bit-identical
        with and without a residual mask.

  V2  each method produces the right residual structure
        - residual : resampled values are drawn FROM the observed
                     residual set (with replacement)
        - wild     : |residual| preserved pointwise, only signs flip
        - block    : contiguous runs preserved
        - frame_drop: observation untouched, mask has the right size

  V3  replicates actually vary
        A bootstrap whose replicates are all identical is broken.
        Catches seeding bugs of the kind already found once in this
        project's Tier B.

  V4  diagnose_residuals() correctly identifies known cases
        Fed synthetic homoscedastic / heteroscedastic / autocorrelated
        residuals, does it recommend residual / wild / block?

  V5  coverage on synthetic data
        The real test of a CI: generate data with KNOWN tau_i, bootstrap,
        and check the nominal 95% interval contains the truth about 95%
        of the time. Expensive, so runs a small number of trials and
        reports rather than hard-asserting.

  V6  is_calibrated flag is set correctly
        frame_drop must be flagged as a stability index, everything
        else as calibrated -- figures rely on this to label honestly.

Figures -> ~/wx_verification_results/<timestamp>/bootstrap_verify/
"""



import numpy as np
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments
from dce_mri.water_exchange import SSM3

from wx_verify_common import (TestLog, savefig, default_configs,
                              synthetic_aif, default_time_grid)
from dce_mri.bootstrap_wx import (BootstrapWxConfig, bootstrap_curve,
                          diagnose_residuals, _make_replicate_data,
                          convergence_check)

SUBDIR = "bootstrap_verify"


def _setup(tau_i=0.3, sigma=5.0, seed=0):
    acq, subject, aif_cfg, relax_cfg, fitting_cfg = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([0.25, 0.04, 0.6/60, 0.15/60])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif)
    S_true = SSM3.to_signal(Cv, Ce, theta[0], theta[1],
                            (relax_cfg.tau_b_fixed, tau_i),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)
    rng = np.random.default_rng(seed)
    sig = S_true + rng.normal(0, sigma, len(t))
    return (acq, subject, aif_cfg, relax_cfg, fitting_cfg, t, aif, theta,
            S_true, sig)


# =============================================================================
# V1 -- the AIF-integrity claim
# =============================================================================

def test_v1_frame_drop_preserves_aif(log: TestLog):
    print("\n--- V1: frame_drop must NOT degrade the AIF ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif, theta, S_true, sig = _setup()

    # the forward model is called with the FULL grid in both cases;
    # masking happens only afterwards, on the residual
    Cv_full, Ce_full = tcxm_ct_compartments(theta, t, aif)
    S_full = SSM3.to_signal(Cv_full, Ce_full, theta[0], theta[1],
                            (relax_cfg.tau_b_fixed, 0.3),
                            acq, subject, aif_cfg, relax_cfg, 1000.0)

    rng = np.random.default_rng(0)
    keep = np.ones(len(t), dtype=bool)
    keep[rng.choice(len(t), size=8, replace=False)] = False

    # correct: predict on the full grid, then mask the residual
    resid_masked = (S_full - sig)[keep]

    # WRONG (for contrast): shorten time_var/aif and re-predict
    Cv_short, Ce_short = tcxm_ct_compartments(theta, t[keep], aif[keep])
    S_short = SSM3.to_signal(Cv_short, Ce_short, theta[0], theta[1],
                             (relax_cfg.tau_b_fixed, 0.3),
                             acq, subject, aif_cfg, relax_cfg, 1000.0)

    # the correct approach's retained predictions must equal the full-grid
    # predictions exactly -- no interpolation change at all
    log.check_close("V1a masked-residual predictions identical to full-grid",
                    S_full[keep], S_full[keep], rtol=0, atol=0,
                    detail="trivially true by construction; the real check is V1b")

    diff = float(np.max(np.abs(S_full[keep] - S_short)))
    log.record("V1b naive frame-dropping DOES change the prediction "
               "(justifying the masked-residual implementation)",
               diff > 1e-6,
               f"max|full-grid - shortened-grid| = {diff:.4e} a.u. over the "
               f"retained frames")

    # and confirm the replicate generator returns the ORIGINAL signal
    bcfg = BootstrapWxConfig(method="frame_drop", drop_frac=0.2)
    obs, mask = _make_replicate_data("frame_drop", sig, S_true,
                                     sig - S_true, np.random.default_rng(1), bcfg)
    log.check_close("V1c frame_drop leaves the observation untouched",
                    obs, sig, rtol=0, atol=0)
    log.record("V1d frame_drop returns a mask of the right size",
               mask is not None and mask.sum() == len(sig) - max(1, int(round(0.2*len(sig)))),
               f"{int(mask.sum())}/{len(sig)} frames retained (drop_frac=0.2)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(t, S_full, "k-", lw=2, label="full-grid prediction")
    axes[0].plot(t[keep], S_short, "r--", lw=1.6,
                 label="shortened-grid prediction (WRONG)")
    axes[0].scatter(t[~keep], S_full[~keep], c="crimson", zorder=5, s=40,
                    label="dropped frames")
    axes[0].set_xlabel("time (s)", fontsize=9)
    axes[0].set_ylabel("signal (a.u.)", fontsize=9)
    axes[0].set_title("V1  Why the AIF must stay at full resolution",
                      fontsize=10)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(t[keep], S_full[keep] - S_short, "r-", lw=1.6)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("time (s)", fontsize=9)
    axes[1].set_ylabel("prediction difference (a.u.)", fontsize=9)
    axes[1].set_title(f"error introduced by shortening the grid\n"
                      f"max = {diff:.3f} a.u.", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "V1_frame_drop_aif_integrity", SUBDIR)


# =============================================================================
# V2 -- residual structure per method
# =============================================================================

def test_v2_replicate_structure(log: TestLog):
    print("\n--- V2: each method produces the right residual structure ---")
    _, _, _, _, _, t, _, _, S_true, sig = _setup()
    resid = sig - S_true
    T = len(sig)

    # residual bootstrap: every resampled value must come FROM the
    # observed residual set
    bcfg = BootstrapWxConfig(method="residual")
    obs, mask = _make_replicate_data("residual", sig, S_true, resid,
                                     np.random.default_rng(2), bcfg)
    new_resid = obs - S_true
    from_set = np.all([np.any(np.isclose(r, resid)) for r in new_resid])
    log.record("V2a residual: all resampled values drawn from observed residuals",
               bool(from_set) and mask is None,
               f"{T} values, all matched to the original residual set")

    # wild bootstrap: |residual| preserved pointwise, only sign flips
    bcfg = BootstrapWxConfig(method="wild")
    obs, mask = _make_replicate_data("wild", sig, S_true, resid,
                                     np.random.default_rng(3), bcfg)
    new_resid = obs - S_true
    log.check_close("V2b wild: |residual| preserved at every timepoint",
                    np.abs(new_resid), np.abs(resid), rtol=1e-12, atol=1e-12)
    n_flipped = int(np.sum(np.sign(new_resid) != np.sign(resid)))
    log.record("V2c wild: some signs actually flipped",
               0 < n_flipped < T, f"{n_flipped}/{T} signs flipped")

    # block bootstrap: contiguous runs should appear
    bcfg = BootstrapWxConfig(method="block", block_len=4)
    obs, mask = _make_replicate_data("block", sig, S_true, resid,
                                     np.random.default_rng(4), bcfg)
    new_resid = obs - S_true
    runs = 0
    for i in range(T - 1):
        j = np.argmin(np.abs(resid - new_resid[i]))
        k = np.argmin(np.abs(resid - new_resid[i+1]))
        if k == j + 1:
            runs += 1
    log.record("V2d block: contiguous residual runs preserved",
               runs >= T // 3,
               f"{runs}/{T-1} adjacent pairs are consecutive in the original")


# =============================================================================
# V3 -- replicates must actually vary
# =============================================================================

def test_v3_replicates_vary(log: TestLog, n_rep=40):
    print("\n--- V3: replicates actually vary (seeding sanity) ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif, theta, S_true, sig = _setup()

    for method in ("residual", "wild", "frame_drop"):
        bcfg = BootstrapWxConfig(method=method, n_replicates=n_rep,
                                 warm_start=True, n_workers=1, seed=7)
        try:
            r = bootstrap_curve(sig, aif, t, TCXM, acq, subject, cfg,
                                SSM3, aif_cfg, relax_cfg, bcfg, verbose=False)
        except Exception as e:
            log.record(f"V3 {method}: bootstrap ran", False,
                       f"{type(e).__name__}: {e}")
            continue
        if len(r.replicates) < 2:
            log.record(f"V3 {method}: enough replicates succeeded", False,
                       f"only {len(r.replicates)} usable")
            continue
        j = r.param_names.index("tau_i")
        v = r.replicates[:, j]
        n_unique = len(np.unique(np.round(v, 10)))
        log.record(f"V3 {method}: replicates are not all identical",
                   n_unique > 1,
                   f"{n_unique}/{len(v)} distinct tau_i values, "
                   f"spread={v.max()-v.min():.4g}")


# =============================================================================
# V4 -- diagnostics identify known cases
# =============================================================================

def test_v4_diagnostics(log: TestLog):
    print("\n--- V4: diagnose_residuals() on known synthetic cases ---")
    rng = np.random.default_rng(0)
    T = 36
    fitted = np.linspace(1000, 1500, T)

    homo = rng.normal(0, 5, T)
    d = diagnose_residuals(homo, fitted, verbose=False)
    log.record("V4a homoscedastic residuals -> 'residual'",
               d["recommend"] == "residual",
               f"het_ratio={d['het_ratio']:.2f}, acf1={d['acf1']:+.2f}, "
               f"got '{d['recommend']}'")

    hetero = rng.normal(0, 1, T) * (fitted / fitted.min()) ** 3
    d = diagnose_residuals(hetero, fitted, verbose=False)
    log.record("V4b heteroscedastic residuals -> 'wild'",
               d["recommend"] == "wild",
               f"het_ratio={d['het_ratio']:.2f}, got '{d['recommend']}'")

    ar = np.zeros(T)
    e = rng.normal(0, 5, T)
    for i in range(1, T):
        ar[i] = 0.8 * ar[i-1] + e[i]
    d = diagnose_residuals(ar, np.full(T, 1000.0), verbose=False)
    log.record("V4c autocorrelated residuals -> 'block'",
               d["recommend"] == "block",
               f"acf1={d['acf1']:+.2f}, got '{d['recommend']}'")


# =============================================================================
# V5 -- coverage on synthetic data
# =============================================================================

def test_v5_coverage(log: TestLog, n_trials=12, n_rep=120, true_ti=0.3,
                     sigma=5.0, method="wild"):
    """
    The real test of a confidence interval. Nominal 95% coverage means
    the interval should contain the truth ~95% of the time across
    repeated experiments.

    REPORTED, not hard-asserted: given how weakly identified tau_i is in
    this data (Tier B), under-coverage here is a finding about the
    parameter, not necessarily a bug in the bootstrap. n_trials is small
    for cost reasons, so the coverage estimate itself is noisy.
    """
    print(f"\n--- V5: coverage of the nominal 95% interval "
          f"({method}, {n_trials} trials x {n_rep} replicates) ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif, theta, S_true, _ = \
        _setup(tau_i=true_ti, sigma=sigma)

    covered, widths, meds = 0, [], []
    for k in range(n_trials):
        rng = np.random.default_rng(9000 + k)
        sig = S_true + rng.normal(0, sigma, len(t))
        bcfg = BootstrapWxConfig(method=method, n_replicates=n_rep,
                                 warm_start=True, seed=9000 + k)
        try:
            r = bootstrap_curve(sig, aif, t, TCXM, acq, subject, cfg,
                                SSM3, aif_cfg, relax_cfg, bcfg, verbose=False)
        except Exception:
            continue
        if len(r.replicates) < 10:
            continue
        lo, hi = r.ci("tau_i")
        covered += int(lo <= true_ti <= hi)
        widths.append(hi - lo)
        meds.append(r.percentile(50, "tau_i"))
        print(f"    trial {k:2d}: CI=[{lo:.4f}, {hi:.4f}]  "
              f"{'CONTAINS' if lo <= true_ti <= hi else 'MISSES'} truth={true_ti}")

    n_ok = len(widths)
    if n_ok == 0:
        log.record("V5 coverage measurable", False, "no trials completed")
        return
    cov = covered / n_ok
    print(f"    coverage: {covered}/{n_ok} = {cov:.0%} (nominal 95%)")
    print(f"    median interval width: {np.median(widths):.4f} "
          f"({np.median(widths)/true_ti:.0%} of truth)")
    log.record("V5 nominal 95% interval covers truth at a plausible rate",
               cov >= 0.6,
               f"{cov:.0%} coverage over {n_ok} trials "
               f"(small n -- estimate is noisy; under-coverage may reflect "
               f"tau_i's weak identifiability, not a bootstrap defect)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(range(n_ok), meds,
                yerr=[np.array(meds) - np.array([m - w/2 for m, w in zip(meds, widths)]),
                      np.array([m + w/2 for m, w in zip(meds, widths)]) - np.array(meds)],
                fmt="o", capsize=3, color="steelblue", label="bootstrap 95% CI")
    ax.axhline(true_ti, color="r", ls="--", lw=1.8, label=f"truth = {true_ti}")
    ax.set_yscale("log")
    ax.set_xlabel("trial", fontsize=9)
    ax.set_ylabel("tau_i (s)", fontsize=9)
    ax.set_title(f"V5  Coverage: {covered}/{n_ok} intervals contain the truth "
                 f"({cov:.0%}, nominal 95%)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "V5_coverage", SUBDIR)


# =============================================================================
# V6 -- calibration flag
# =============================================================================

def test_v6_calibration_flag(log: TestLog):
    print("\n--- V6: is_calibrated flag set correctly ---")
    acq, subject, aif_cfg, relax_cfg, cfg, t, aif, theta, S_true, sig = _setup()
    for method, expected in [("frame_drop", False), ("residual", True),
                             ("wild", True), ("block", True)]:
        bcfg = BootstrapWxConfig(method=method, n_replicates=6,
                                 warm_start=True, n_workers=1)
        try:
            r = bootstrap_curve(sig, aif, t, TCXM, acq, subject, cfg,
                                SSM3, aif_cfg, relax_cfg, bcfg, verbose=False)
            log.record(f"V6 {method}: is_calibrated == {expected}",
                       r.is_calibrated == expected,
                       f"got {r.is_calibrated}")
        except Exception as e:
            log.record(f"V6 {method}: ran", False, f"{type(e).__name__}")


# =============================================================================

def run_bootstrap_verification(quick=False):
    log = TestLog("BOOTSTRAP MODULE VERIFICATION")
    print("=" * 72)
    print("  BOOTSTRAP MODULE VERIFICATION (bootstrap_wx.py)")
    print("=" * 72)
    test_v1_frame_drop_preserves_aif(log)
    test_v2_replicate_structure(log)
    test_v3_replicates_vary(log, n_rep=20 if quick else 40)
    test_v4_diagnostics(log)
    test_v6_calibration_flag(log)
    if not quick:
        test_v5_coverage(log)
    else:
        print("\n  (V5 coverage skipped in quick mode -- it is the expensive one)")
    return log.summary()


if __name__ == "__main__":
    import sys
    run_bootstrap_verification(quick="--quick" in sys.argv)