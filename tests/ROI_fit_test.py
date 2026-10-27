#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:31:07 2026

@author: allisonchen
"""

from dce_mri.config import (AcquisitionConfig, SubjectConfig,
                              FittingConfig, RoiFittingConfig)
from dce_mri.kinetic_models import TCXM, GKM, tcxm_ct_analytic, gkm_ct
from dce_mri.signal_models import concentration_to_signal, shift_aif
from dce_mri.fitting_roi import (fit_roi, fit_roi_with_delay_grid,
                                   fit_roi_with_delay_continuous)
import numpy as np

# ── Shared setup ──────────────────────────────────────────────────────
acq     = AcquisitionConfig()
subject = SubjectConfig("test", T10=1530.0, B1=1.0)
cfg     = FittingConfig(n_lhs=200, n_top=10)

t   = np.linspace(0, 300, 90)
t_b = t - 30.0
aif = np.clip(((t_b/15)**4) * np.exp(-4*(t_b/15 - 1)), 0, None)
aif[:9] = 0.0
aif /= aif.max()

def make_volume(Ct, S0=1000.0):
    sig = concentration_to_signal(Ct, acq, 1530.0, 1.0, S0)
    return np.tile(sig, (4, 4, 4, 1))

mask       = np.ones((4, 4, 4), dtype=bool)
tau_values = np.arange(-10, 31, 2.0)

roi_cfg_signal = RoiFittingConfig(average_space="signal")
roi_cfg_conc   = RoiFittingConfig(average_space="concentration")

def print_roi_result(label, result, true_params):
    print(f"  [{label}]")
    print(f"    success : {result.success}")
    print(f"    cost    : {result.cost:.4e}")
    print(f"    true    : {true_params}")
    print(f"    fitted  : {result.params}")
    print(f"    Ct_mean : {result.Ct_mean.shape}  "
          f"range [{result.Ct_mean.min():.4f}, {result.Ct_mean.max():.4f}]")
    print(f"    St_mean : {result.St_mean.shape}  "
          f"range [{result.St_mean.min():.1f}, {result.St_mean.max():.1f}]")

def print_delay_result(label, result, true_tau, true_params):
    print(f"  [{label}]")
    print(f"    success      : {result.success}")
    print(f"    cost         : {result.best_cost:.4e}")
    print(f"    true tau     : {true_tau}")
    print(f"    recovered tau: {result.best_tau:.2f}")
    print(f"    true params  : {true_params}")
    print(f"    fitted params: {result.best_params}")


# ══════════════════════════════════════════════════════════════════════
# TEST 1 — TCXM, no delay, signal vs concentration aggregation
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: TCXM no delay — signal vs concentration aggregation")
Ct_tcxm = tcxm_ct_analytic(TCXM.p0, t, aif)
sig_4d  = make_volume(Ct_tcxm)

r1_sig  = fit_roi(sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
                   roi_cfg_signal, seed=0)
r1_conc = fit_roi(sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
                   roi_cfg_conc, seed=0)

print_roi_result("signal aggregation",       r1_sig,  TCXM.p0)
print_roi_result("concentration aggregation", r1_conc, TCXM.p0)

assert r1_sig.Ct_mean.shape  == (len(t),)
assert r1_sig.St_mean.shape  == (len(t),)
assert r1_conc.Ct_mean.shape == (len(t),)
assert r1_conc.St_mean.shape == (len(t),)
print("  shape assertions passed")


# ══════════════════════════════════════════════════════════════════════
# TEST 2 — GKM, no delay, signal vs concentration aggregation
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 2: GKM no delay — signal vs concentration aggregation")
Ct_gkm = gkm_ct(GKM.p0, t, aif)
sig_4d = make_volume(Ct_gkm)

r2_sig  = fit_roi(sig_4d, mask, aif, t, GKM, acq, subject, cfg,
                   roi_cfg_signal, seed=0)
r2_conc = fit_roi(sig_4d, mask, aif, t, GKM, acq, subject, cfg,
                   roi_cfg_conc, seed=0)

print_roi_result("signal aggregation",        r2_sig,  GKM.p0)
print_roi_result("concentration aggregation", r2_conc, GKM.p0)


# ══════════════════════════════════════════════════════════════════════
# TEST 3 — TCXM, grid delay, signal vs concentration aggregation
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 3: TCXM grid delay (true tau=8s) — signal vs concentration")
true_tau    = 8.4
shifted_aif = shift_aif(aif, true_tau, t, acq)
sig_4d      = make_volume(tcxm_ct_analytic(TCXM.p0, t, shifted_aif))

r3_sig  = fit_roi_with_delay_grid(
    sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
    roi_cfg_signal, tau_values, seed=0
)
r3_conc = fit_roi_with_delay_grid(
    sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
    roi_cfg_conc, tau_values, seed=0
)

print_delay_result("signal aggregation",        r3_sig,  true_tau, TCXM.p0)
print_delay_result("concentration aggregation", r3_conc, true_tau, TCXM.p0)

assert abs(r3_sig.best_tau  - true_tau) <= 2.0, \
    f"signal agg tau error: {abs(r3_sig.best_tau - true_tau):.1f}s"
assert abs(r3_conc.best_tau - true_tau) <= 2.0, \
    f"conc agg tau error:   {abs(r3_conc.best_tau - true_tau):.1f}s"
print("  tau assertions passed")


# ══════════════════════════════════════════════════════════════════════
# TEST 4 — GKM, continuous delay, signal vs concentration aggregation
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 4: GKM continuous delay (true tau=6s) — signal vs concentration")
true_tau    = 8.4
shifted_aif = shift_aif(aif, true_tau, t, acq)
sig_4d      = make_volume(gkm_ct(GKM.p0, t, shifted_aif))

r4_sig  = fit_roi_with_delay_continuous(
    sig_4d, mask, aif, t, GKM, acq, subject, cfg,
    roi_cfg_signal, tau_lb=-10.0, tau_ub=30.0, tau_p0=0.0, seed=0
)
r4_conc = fit_roi_with_delay_continuous(
    sig_4d, mask, aif, t, GKM, acq, subject, cfg,
    roi_cfg_conc, tau_lb=-10.0, tau_ub=30.0, tau_p0=0.0, seed=0
)

print_delay_result("signal aggregation",        r4_sig,  true_tau, GKM.p0)
print_delay_result("concentration aggregation", r4_conc, true_tau, GKM.p0)

assert abs(r4_sig.best_tau  - true_tau) <= 3.0, \
    f"signal agg tau error: {abs(r4_sig.best_tau - true_tau):.1f}s"
assert abs(r4_conc.best_tau - true_tau) <= 3.0, \
    f"conc agg tau error:   {abs(r4_conc.best_tau - true_tau):.1f}s"
print("  tau assertions passed")


# ══════════════════════════════════════════════════════════════════════
# TEST 5 — cost comparison: no delay vs grid vs continuous (TCXM)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 5: cost comparison — no delay vs grid vs continuous (TCXM, tau=8s)")
true_tau    = 8.4
shifted_aif = shift_aif(aif, true_tau, t, acq)
sig_4d      = make_volume(tcxm_ct_analytic(TCXM.p0, t, shifted_aif))

r5_nodelay = fit_roi(sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
                      roi_cfg_signal, seed=0)
r5_grid    = fit_roi_with_delay_grid(
    sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
    roi_cfg_signal, tau_values, seed=0
)
r5_cont    = fit_roi_with_delay_continuous(
    sig_4d, mask, aif, t, TCXM, acq, subject, cfg,
    roi_cfg_signal, tau_lb=-10.0, tau_ub=30.0, tau_p0=0.0, seed=0
)

print(f"  no delay : cost={r5_nodelay.cost:.4e}  "
      f"params={r5_nodelay.params}")
print(f"  grid     : cost={r5_grid.best_cost:.4e}  "
      f"tau={r5_grid.best_tau:.1f}s  params={r5_grid.best_params}")
print(f"  continuous: cost={r5_cont.best_cost:.4e}  "
      f"tau={r5_cont.best_tau:.2f}s  params={r5_cont.best_params}")
print(f"  expected: grid and continuous cost << no delay cost")
assert r5_grid.best_cost < r5_nodelay.cost, \
    "grid delay should improve cost over no delay"
assert r5_cont.best_cost < r5_nodelay.cost, \
    "continuous delay should improve cost over no delay"
assert r5_cont.best_cost < r5_grid.best_cost, \
    "continuous should outperform grid when true tau is between grid points"
print("  continuous < grid cost assertion passed")
print("  cost assertions passed")

print("=" * 60)
print("All 5 tests complete.")