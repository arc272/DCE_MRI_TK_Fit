#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Wed Oct 21 12:47:14 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_taui_series.py
===================
Signal-intensity curves for ONE set of TCXM parameters across a range of
tau_i, so the size and shape of the water-exchange effect is visible, and so
it can be compared against the noise level the cohort actually has.

Goes through dce_mri.synthetic.simulate_curve, i.e. the same forward-model
path the fits use -- so this plots whatever version of water_exchange.py /
jax_forward.py is on disk, including the vascular-pool edit if you have
applied it.  The header printout states which convention it detected.

Panels
------
A  signal vs time, one curve per tau_i, with the no-water-exchange (FXL)
   curve dashed for reference.  Noise sd shown as a band on the baseline.
B  S(tau_i) - S(tau_i = fastest) in units of the noise sd -- the part of the
   curve that water exchange is actually responsible for.
C  summary: max |dS| against tau_i, versus both the fast-exchange end and the
   FXL curve, with the noise sd marked.  Where this flattens is where tau_i
   stops being identifiable.
D  pairwise distinguishability: max |S(tau_i) - S(tau_j)| / noise sd for every
   pair on the grid.  Cells below ~1 are pairs the data cannot separate at
   single-curve SNR.

Set AIF_SOURCE = "real" to use your extracted AIFs (the effect size scales
with AIF amplitude, so this matters for the absolute numbers).
"""


import glob
import os
import sys

import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
LIB_PATH = "/mnt/labspace/users/allison/DCE_MRI_TK_Fit/src"
OUT_DIR  = "/home/allisonchen/wx_verification_results/"

# TCXM parameters: ve, vp, fp, ps   (fp, ps in 1/s)
P0 = np.array([0.243, 0.196, 0.00635, 0.00515])

# tau_i values to draw (s).  Spread on a linear axis, spanning the prior.
TAU_I_LIST = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.25, 1.50]

DELAY_S = 0.0          # AIF shift applied before simulating, if you want one

AIF_SOURCE   = "parker"      # "real" | "parker"
AIF_DIR      = "/mnt/labspace/users/allison/fastMRI_data/aif_results_2"
AIF_SUBJECT  = None          # e.g. "FMB001"; None -> first CSV found
PARKER_ARRIVAL = 35.0

NOISE_SD = 15.27       # cohort noise sd on S0 = 1000, from the production run

if LIB_PATH:
    sys.path.insert(0, LIB_PATH)

from dce_mri.config import (AcquisitionConfig, AifConfig, FittingConfig,
                            RelaxationConfig, SubjectConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.signal_models import shift_aif
from dce_mri.synthetic import make_parker_aif, simulate_curve
from dce_mri.water_exchange import SSM3

# --- configs: COPIED from validate_wx_fixedtau_synthetic.py -------------------
NBASE          = slice(0, 7)
acq            = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)
T10_TISSUE, B1 = 1530.0, 1.0
cfg_fit        = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
relax_cfg      = RelaxationConfig(fit_tau_b=False, tau_b_fixed=0.005,
                                  tau_i_lb=0.01, tau_i_ub=1.5)
aif_cfg        = AifConfig(T10=1779.0, hematocrit=0.4)
S0_TRUE        = 1000.0
SPOKES_PER_FRAME, NSPOKE, SCANTIME = 8, 288, 150
N_FRAMES       = NSPOKE // SPOKES_PER_FRAME

subject = SubjectConfig(subject_id="TAUISERIES", T10=T10_TISSUE, B1=B1)


def get_aif():
    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    if AIF_SOURCE == "parker":
        t = np.arange(N_FRAMES) * dt
        return ("parker", t,
                make_parker_aif(t, t_arrival=PARKER_ARRIVAL,
                                hematocrit=aif_cfg.hematocrit))
    import pandas as pd
    p = (os.path.join(AIF_DIR, f"{AIF_SUBJECT}_AIF_plasma.csv") if AIF_SUBJECT
         else (sorted(glob.glob(os.path.join(AIF_DIR, "*_AIF_plasma.csv")))
               or [None])[0])
    if p is None or not os.path.exists(p):
        raise FileNotFoundError(
            f"no *_AIF_plasma.csv in {AIF_DIR}; set AIF_SOURCE='parker'")
    df = pd.read_csv(p)
    aif = df["aif_plasma_mM"].to_numpy(float)
    t = (df["time_s"].to_numpy(float) if "time_s" in df
         else np.arange(aif.size) * dt)
    return (os.path.basename(p).replace("_AIF_plasma.csv", ""), t, aif)


def curve(t, aif, tau_i=None):
    """Signal curve; tau_i=None gives the no-water-exchange (FXL) model."""
    a = shift_aif(np.asarray(aif, float), float(DELAY_S), t, acq) \
        if DELAY_S else np.asarray(aif, float)
    if tau_i is None:
        th = P0
        kw = dict(wx_model=None, aif_cfg=None, relax_cfg=None)
    else:
        th = np.concatenate([P0, [float(tau_i)]])
        kw = dict(wx_model=SSM3, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
    return np.asarray(simulate_curve(th, a, t, TCXM, acq, subject, cfg_fit,
                                     S0=S0_TRUE, **kw), float)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ve, vp, fp, ps = P0
    pf = 1.0 - aif_cfg.hematocrit
    label, t, aif = get_aif()

    print("=" * 72)
    print("plot_taui_series.py")
    print("=" * 72)
    print(f"p0            : ve={ve}  vp={vp}  fp={fp} 1/s ({fp*60:.4f} 1/min)"
          f"  ps={ps} 1/s ({ps*60:.4f} 1/min)")
    print(f"derived       : Ktrans={fp*ps/(fp+ps)*60:.4f} 1/min   "
          f"E={ps/(fp+ps):.4f}   vtot={ve+vp:.4f}")
    print(f"AIF           : {label}, {len(t)} frames, dt={t[1]-t[0]:.4f} s, "
          f"peak {aif.max():.3f} mM")
    print(f"delay applied : {DELAY_S} s")
    print(f"noise sd      : {NOISE_SD} on S0={S0_TRUE:g}  "
          f"({NOISE_SD/S0_TRUE*100:.3f}% of S0)")
    print(f"vascular pool : vp/(1-h) = {vp/pf:.4f} (edited convention) or "
          f"vp = {vp:.4f} (pre-edit)")
    print(f"                vi = {1-vp/pf-ve:+.4f} (edited) / "
          f"{1-vp-ve:+.4f} (pre-edit)")
    if 1 - vp / pf - ve < 0:
        print("  *** vi < 0 under the edited convention: this operating point "
              "is infeasible there")

    S_fxl = curve(t, aif, None)
    curves = {tau: curve(t, aif, tau) for tau in TAU_I_LIST}
    fast = TAU_I_LIST[0]

    print(f"\n  tau_i(s)   peak S    enh.    max|dS| vs tau={fast:g}s"
          f"    max|dS| vs FXL")
    print(f"             (a.u.)   (xS0)    (a.u.)  (x noise)    (a.u.)  "
          f"(x noise)")
    for tau in TAU_I_LIST:
        S = curves[tau]
        d_fast = float(np.max(np.abs(S - curves[fast])))
        d_fxl = float(np.max(np.abs(S - S_fxl)))
        print(f"  {tau:7.3f}  {S.max():8.1f}  {S.max()/S0_TRUE:5.3f}  "
              f"{d_fast:8.2f}  {d_fast/NOISE_SD:8.2f}   {d_fxl:8.2f}  "
              f"{d_fxl/NOISE_SD:8.2f}")
    print(f"  {'FXL':>7}  {S_fxl.max():8.1f}  {S_fxl.max()/S0_TRUE:5.3f}")

    # ---- figure ----
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(TAU_I_LIST), max(TAU_I_LIST))
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    a = ax[0, 0]
    for tau in TAU_I_LIST:
        a.plot(t, curves[tau], lw=1.9, color=cmap(norm(tau)),
               label=f"{tau:g}")
    a.plot(t, S_fxl, "k--", lw=2.2, label="no WX (FXL)")
    a.axhspan(S0_TRUE - NOISE_SD, S0_TRUE + NOISE_SD, color="grey", alpha=.25,
              zorder=0, label="±1 noise sd")
    a.set_xlabel("time (s)")
    a.set_ylabel("signal (a.u.)")
    a.set_title(f"A  signal vs tau_i    ve={ve} vp={vp} "
                f"fp={fp*60:.3f} ps={ps*60:.3f} 1/min")
    a.grid(alpha=.3)
    a.legend(title="tau_i (s)", fontsize=7, ncol=2, loc="upper left")

    a = ax[0, 1]
    for tau in TAU_I_LIST[1:]:
        a.plot(t, (curves[tau] - curves[fast]) / NOISE_SD, lw=1.9,
               color=cmap(norm(tau)))
    a.axhline(0, color="k", lw=.8)
    for s in (-1, 1):
        a.axhline(s, color="r", lw=1.0, ls=":")
    a.set_xlabel("time (s)")
    a.set_ylabel(f"[S(tau_i) - S(tau_i={fast:g}s)] / noise sd")
    a.set_title("B  the water-exchange contribution, in noise units\n"
                "(dotted red: ±1 noise sd on a single frame)")
    a.grid(alpha=.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=a, label="tau_i (s)")

    a = ax[1, 0]
    d_fast = [np.max(np.abs(curves[x] - curves[fast])) / NOISE_SD
              for x in TAU_I_LIST]
    d_fxl = [np.max(np.abs(curves[x] - S_fxl)) / NOISE_SD for x in TAU_I_LIST]
    a.plot(TAU_I_LIST, d_fast, "o-", lw=2, label=f"vs tau_i = {fast:g} s")
    a.plot(TAU_I_LIST, d_fxl, "s--", lw=2, label="vs no-WX (FXL)")
    a.axhline(1.0, color="r", ls=":", lw=1.2, label="1 noise sd")
    a.set_xlabel("tau_i (s)   [linear]")
    a.set_ylabel("max |dS| / noise sd")
    a.set_title("C  effect size against tau_i\n"
                "flat region = tau_i no longer changes the prediction")
    a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[1, 1]
    n = len(TAU_I_LIST)
    D = np.zeros((n, n))
    for i, ti in enumerate(TAU_I_LIST):
        for j, tj in enumerate(TAU_I_LIST):
            D[i, j] = np.max(np.abs(curves[ti] - curves[tj])) / NOISE_SD
    m = a.imshow(D, origin="lower", cmap="magma")
    a.set_xticks(range(n)); a.set_xticklabels([f"{x:g}" for x in TAU_I_LIST],
                                              rotation=90, fontsize=7)
    a.set_yticks(range(n)); a.set_yticklabels([f"{x:g}" for x in TAU_I_LIST],
                                              fontsize=7)
    a.set_xlabel("tau_i (s)"); a.set_ylabel("tau_i (s)")
    a.set_title("D  pairwise max|dS| / noise sd\n"
                "cells < 1 are pairs the data cannot separate")
    for i in range(n):
        for j in range(n):
            if D[i, j] >= 1.0 and i != j:
                a.text(j, i, f"{D[i,j]:.0f}", ha="center", va="center",
                       fontsize=6,
                       color="w" if D[i, j] < D.max() * .6 else "k")
    fig.colorbar(m, ax=a, label="max|dS| / noise sd")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "taui_series.png")
    fig.savefig(p, dpi=135)
    plt.close(fig)
    print(f"\nfigure -> {p}")


if __name__ == "__main__":
    main()