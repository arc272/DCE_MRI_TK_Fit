#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_verify_common.py
===================
Shared infrastructure for the water-exchange verification suite.

Provides:
  - RESULTS_DIR setup (timestamped, under ~/wx_verification_results)
  - savefig() helper so every figure lands in the same place
  - TestLog: accumulates pass/fail across scripts, prints a summary
  - default config builders (acquisition/subject/aif/relax/fitting)
  - INDEPENDENT reference implementations used to check SSM3 against

Design note on references
-------------------------
Tests run the REAL SSM3 from dce_mri.water_exchange -- never a
reimplementation of the thing under test. But the REFERENCE each test
compares against must be computed independently, or the test is
circular. The reference functions below (spgr_single_pool,
fxl_limit_reference, sxl_limit_reference) are deliberately written
from the physics rather than by calling into water_exchange.py.
"""

from __future__ import annotations

import os
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt


# =============================================================================
# Results directory
# =============================================================================

_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_ROOT = Path.home() / "wx_verification_results"
RESULTS_DIR  = RESULTS_ROOT / _STAMP


def init_results_dir(subdir: str | None = None) -> Path:
    """
    Create (and return) the results directory for this run. Timestamped
    so reruns don't clobber earlier output -- important if you're
    collecting figures for a progress update across several sessions.
    """
    d = RESULTS_DIR if subdir is None else RESULTS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, name: str, subdir: str | None = None, dpi: int = 150) -> Path:
    """Save a figure into the results dir and return its path."""
    d = init_results_dir(subdir)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [figure] {path}")
    return path


# =============================================================================
# Test result accumulation
# =============================================================================

class TestLog:
    """
    Minimal pass/fail accumulator. Deliberately does not raise on
    failure -- a failing check should not prevent later checks from
    running, since the whole point of the suite is to see the full
    picture in one pass.
    """

    def __init__(self, title: str):
        self.title = title
        self.rows = []

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(passed), detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
        return passed

    def check_close(self, name, a, b, rtol=1e-6, atol=1e-9, detail=""):
        a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
        ok = np.allclose(a_arr, b_arr, rtol=rtol, atol=atol)
        maxdiff = float(np.max(np.abs(a_arr - b_arr))) if a_arr.size else 0.0
        d = f"max|diff|={maxdiff:.3e}" + (f"; {detail}" if detail else "")
        return self.record(name, ok, d)

    def check_raises(self, name, fn, exc=ValueError, detail=""):
        try:
            fn()
        except exc as e:
            return self.record(name, True, f"raised: {str(e)[:70]}...")
        except Exception as e:
            return self.record(name, False,
                               f"raised {type(e).__name__}, expected {exc.__name__}")
        return self.record(name, False, "no exception raised")

    @property
    def n_pass(self):
        return sum(1 for _, p, _ in self.rows if p)

    @property
    def n_fail(self):
        return sum(1 for _, p, _ in self.rows if not p)

    def summary(self) -> bool:
        print()
        print("=" * 72)
        print(f"  {self.title}: {self.n_pass} passed, {self.n_fail} failed")
        if self.n_fail:
            print("  FAILED:")
            for name, passed, detail in self.rows:
                if not passed:
                    print(f"    - {name}  ({detail})")
        print("=" * 72)
        return self.n_fail == 0


# =============================================================================
# Default configs -- match the FMB056 acquisition used throughout
# =============================================================================

def default_configs(fit_tau_b: bool = False):
    """
    Returns (acq, subject, aif_cfg, relax_cfg, fitting_cfg) matching the
    real acquisition parameters used in this project, so verification
    results are relevant to the data rather than to arbitrary numbers.
    """
    from dce_mri.config import (AcquisitionConfig, SubjectConfig, AifConfig,
                                RelaxationConfig, FittingConfig)
    acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 11))
    subject = SubjectConfig(subject_id="synthetic", T10=1530.0, B1=1.0)
    aif_cfg = AifConfig(T10=1600.0, hematocrit=0.42, top_frac=1.0,
                        nbase=slice(0, 8))
    relax_cfg = RelaxationConfig(
        fit_tau_b=fit_tau_b, tau_b_fixed=0.005,
        tau_b_lb=1e-3, tau_b_ub=1e-2,
        tau_i_lb=1e-2, tau_i_ub=2.0,
    )
    fitting_cfg = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
    return acq, subject, aif_cfg, relax_cfg, fitting_cfg


def synthetic_aif(time_var, peak_mM=4.0, t_arrival=45.0, width=10.0):
    """
    Gamma-variate AIF on the given time grid. peak_mM defaults to a
    physiological 4 mM rather than the ~10 mM seen in the real FMB056
    extraction, so synthetic tests aren't run in the saturated regime.
    """
    tb = np.asarray(time_var, float) - t_arrival
    a = np.clip(((tb / width) ** 3) * np.exp(-3 * (tb / width - 1)), 0, None)
    a[np.asarray(time_var, float) < t_arrival * 0.2] = 0.0
    if a.max() > 0:
        a = a / a.max() * peak_mM
    return a


def default_time_grid(n_frames=36, dt=4.1667):
    """Matches FMB056: 36 frames at ~4.167 s (8 spokes/frame)."""
    return np.arange(n_frames) * dt


# =============================================================================
# INDEPENDENT reference implementations
# =============================================================================
# These are written from the physics, NOT by calling water_exchange.py.
# They are what SSM3 gets checked against.

def spgr_single_pool(R1, TR_s, FA_rad, M0=1.0):
    """
    Steady-state SPGR magnetisation for a single well-mixed pool.
    Scalar or array R1.
    """
    R1 = np.asarray(R1, float)
    E1 = np.exp(-TR_s * R1)
    return M0 * (1 - E1) * np.sin(FA_rad) / (1 - E1 * np.cos(FA_rad))


def fxl_limit_reference(Cv, Ce, ve, vb, acq, subject, aif_cfg, S0=1000.0):
    """
    Fast-exchange limit reference, computed independently.

    In the fast limit the three pools share one effective R1, which is
    the VOLUME-WEIGHTED average:

        R1_avg = vb*R1_b + ve*R1_e + vi*R1_i

    with the same per-pool densities SSM3 uses (Ce/ve, and Cv/vb with
    hematocrit dilution). Note this is NOT the same as plain
    concentration_to_signal, which uses a single tissue-only baseline --
    the two genuinely differ whenever vb>0, by a predictable amount.
    That difference is the whole reason this reference exists rather
    than reusing concentration_to_signal.
    """
    vi = 1.0 - vb - ve
    TR_s = acq.TR / 1000.0
    FA = np.deg2rad(acq.FA * subject.B1)
    R1_tissue = 1000.0 / subject.T10
    R1_blood  = 1000.0 / aif_cfg.T10
    plasma_frac = 1.0 - aif_cfg.hematocrit

    R1_e = R1_tissue + acq.r1 * (np.asarray(Ce, float) / ve)
    R1_b = R1_blood  + acq.r1 * (np.asarray(Cv, float) / vb) * plasma_frac
    R1_i = R1_tissue

    R1_avg      = vb * R1_b + ve * R1_e + vi * R1_i
    R1_avg_base = vb * R1_blood + (ve + vi) * R1_tissue

    M      = spgr_single_pool(R1_avg,      TR_s, FA)
    M_base = spgr_single_pool(R1_avg_base, TR_s, FA)
    return S0 * M / M_base


def sxl_limit_reference(Cv, Ce, ve, vb, acq, subject, aif_cfg, S0=1000.0):
    """
    Slow-exchange limit reference, computed independently.

    With no exchange at all, each pool relaxes on its own and the
    observed signal is the volume-weighted sum of three independent
    SPGR signals -- not a single pool with an averaged R1.
    """
    vi = 1.0 - vb - ve
    TR_s = acq.TR / 1000.0
    FA = np.deg2rad(acq.FA * subject.B1)
    R1_tissue = 1000.0 / subject.T10
    R1_blood  = 1000.0 / aif_cfg.T10
    plasma_frac = 1.0 - aif_cfg.hematocrit

    R1_e = R1_tissue + acq.r1 * (np.asarray(Ce, float) / ve)
    R1_b = R1_blood  + acq.r1 * (np.asarray(Cv, float) / vb) * plasma_frac

    M = (vb * spgr_single_pool(R1_b, TR_s, FA)
         + ve * spgr_single_pool(R1_e, TR_s, FA)
         + vi * spgr_single_pool(R1_tissue, TR_s, FA))
    M_base = (vb * spgr_single_pool(R1_blood,  TR_s, FA)
              + ve * spgr_single_pool(R1_tissue, TR_s, FA)
              + vi * spgr_single_pool(R1_tissue, TR_s, FA))
    return S0 * M / M_base


def ssm3_manual(Cv, Ce, ve, vb, tau_b, tau_i, acq, subject, aif_cfg,
                S0=1000.0, apply_density=True, apply_hematocrit=True):
    """
    Manual 3-pool SSM implementation with the density division and the
    hematocrit dilution each independently switchable.

    Used ONLY by the convention regression test (A4), to demonstrate
    what the signal looks like with and without each convention -- so
    the test can assert the real SSM3 matches the correct variant and
    differs from the incorrect ones, rather than just asserting a
    number it can't interpret.
    """
    from scipy.linalg import expm, pinv

    vi = 1.0 - vb - ve
    tau_e = ve / (vb / tau_b + vi / tau_i)
    TR_s = acq.TR / 1000.0
    FA = np.deg2rad(acq.FA * subject.B1)
    R1_tissue = 1000.0 / subject.T10
    R1_blood  = 1000.0 / aif_cfg.T10
    pf = (1.0 - aif_cfg.hematocrit) if apply_hematocrit else 1.0

    def pool_sum(R1_b, R1_e, R1_i):
        X = np.array([
            [-(R1_b + 1/tau_b),  1/tau_e - (vi/ve)/tau_i,  0.0],
            [ 1/tau_b,          -(R1_e + 1/tau_e),         1/tau_i],
            [ 0.0,               1/tau_e - (vb/ve)/tau_b, -(R1_i + 1/tau_i)],
        ])
        M0 = np.array([vb, ve, vi])
        I = np.eye(3)
        eX = expm(TR_s * X)
        return (pinv(I - eX * np.cos(FA)) @ (I - eX) @ (np.sin(FA) * M0)).sum()

    base = pool_sum(R1_blood, R1_tissue, R1_tissue)
    out = np.empty(len(Cv))
    for k in range(len(Cv)):
        ce_d = (Ce[k] / ve) if apply_density else Ce[k]
        cv_d = ((Cv[k] / vb) if apply_density else Cv[k]) * pf
        out[k] = S0 * pool_sum(R1_blood + acq.r1 * cv_d,
                               R1_tissue + acq.r1 * ce_d,
                               R1_tissue) / base
    return out


# =============================================================================
# Plot helpers
# =============================================================================

def plot_curves(ax, time_var, curves: dict, title="", ylabel="signal (a.u.)",
                xlabel="time (s)"):
    for label, y in curves.items():
        style = "--" if "reference" in label.lower() else "-"
        ax.plot(time_var, y, style, label=label, lw=1.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
