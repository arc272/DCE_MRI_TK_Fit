#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_fxl_vs_sxl_shape.py
===========================
Answers the question: "why do the FXL and SXL limits produce such
similar-looking curves, differing mainly in amplitude, when the TCXM
parameters are the same?"

Short answer: because water exchange is a STATIC (memoryless) pointwise
map from concentration to signal. It has no dynamics of its own on the
DCE timescale, so it cannot reshape the time-course -- only rescale it.
The shape is set entirely by the kinetic model.

Produces a 4-panel figure suitable for a progress update.

Run from the wx_verification directory (needs wx_verify_common.py).
"""



import numpy as np
import matplotlib.pyplot as plt

from dce_mri.kinetic_models import tcxm_ct_compartments
from wx_verify_common import (savefig, default_configs, synthetic_aif,
                              default_time_grid, fxl_limit_reference,
                              sxl_limit_reference)

SUBDIR = "explain"


def _norm(S, nbase=9):
    """Baseline-subtract and peak-normalise, isolating SHAPE from amplitude."""
    b = S[:nbase].mean()
    return (S - b) / (S.max() - b)


def main():
    acq, subject, aif_cfg, relax_cfg, _ = default_configs()
    t = default_time_grid()
    aif = synthetic_aif(t)
    theta = np.array([0.25, 0.04, 0.6/60, 0.15/60])
    Cv, Ce = tcxm_ct_compartments(theta, t, aif)
    ve, vb = theta[0], theta[1]

    S_fxl = fxl_limit_reference(Cv, Ce, ve, vb, acq, subject, aif_cfg, 1000.0)
    S_sxl = sxl_limit_reference(Cv, Ce, ve, vb, acq, subject, aif_cfg, 1000.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ---- (a) the raw curves: differ in amplitude ----------------------
    ax = axes[0, 0]
    ax.plot(t, S_fxl, "-", lw=2, color="steelblue", label="FXL (tau -> 0)")
    ax.plot(t, S_sxl, "-", lw=2, color="darkorange", label="SXL (tau -> inf)")
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("signal (a.u.)", fontsize=9)
    ax.set_title("(a) Raw curves: same shape, different amplitude\n"
                 f"peak enh FXL={S_fxl.max()/S_fxl[:9].mean():.3f}, "
                 f"SXL={S_sxl.max()/S_sxl[:9].mean():.3f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- (b) normalised: shapes nearly overlay ------------------------
    ax = axes[0, 1]
    n_fxl, n_sxl = _norm(S_fxl), _norm(S_sxl)
    ax.plot(t, n_fxl, "-", lw=2, color="steelblue", label="FXL, normalised")
    ax.plot(t, n_sxl, "--", lw=2, color="darkorange", label="SXL, normalised")
    ax.plot(t, n_fxl - n_sxl, ":", lw=1.5, color="crimson",
            label="difference")
    rms = np.sqrt(np.mean((n_fxl - n_sxl)**2))
    r = np.corrcoef(S_fxl, S_sxl)[0, 1]
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("normalised enhancement", fontsize=9)
    ax.set_title(f"(b) After removing amplitude: shapes nearly identical\n"
                 f"shape RMS diff = {rms:.4f},  Pearson r = {r:.5f}",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- (c) the static transfer function ----------------------------
    ax = axes[1, 0]
    Ct_probe = np.linspace(0, 1.2, 60)
    f_probe, s_probe = [], []
    for c in Ct_probe:
        cv = np.array([c * vb / (ve + vb)])
        ce = np.array([c * ve / (ve + vb)])
        f_probe.append(fxl_limit_reference(cv, ce, ve, vb, acq, subject,
                                           aif_cfg, 1000.0)[0])
        s_probe.append(sxl_limit_reference(cv, ce, ve, vb, acq, subject,
                                           aif_cfg, 1000.0)[0])
    ax.plot(Ct_probe, f_probe, "-", lw=2, color="steelblue", label="FXL map")
    ax.plot(Ct_probe, s_probe, "-", lw=2, color="darkorange", label="SXL map")
    ax.set_xlabel("total tissue [Gd] (mM)", fontsize=9)
    ax.set_ylabel("signal (a.u.)", fontsize=9)
    ax.set_title("(c) WHY: exchange is a STATIC pointwise map\n"
                 "both monotonic, no time dependence, no crossings",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- (d) counterfactual: what DOES change shape ------------------
    ax = axes[1, 1]
    cases = [
        ("ve=.25 vp=.04 fp=.6 ps=.15", [0.25, 0.04, 0.6/60, 0.15/60], "steelblue"),
        ("ve=.10 vp=.02 fp=.3 ps=.05", [0.10, 0.02, 0.3/60, 0.05/60], "seagreen"),
        ("ve=.40 vp=.10 fp=2.0 ps=.50", [0.40, 0.10, 2.0/60, 0.50/60], "purple"),
    ]
    for lab, th, col in cases:
        th = np.array(th)
        cv, ce = tcxm_ct_compartments(th, t, aif)
        f = fxl_limit_reference(cv, ce, th[0], th[1], acq, subject,
                                aif_cfg, 1000.0)
        s = sxl_limit_reference(cv, ce, th[0], th[1], acq, subject,
                                aif_cfg, 1000.0)
        ax.plot(t, _norm(f), "-", lw=1.8, color=col, label=f"{lab} (FXL)")
        ax.plot(t, _norm(s), "--", lw=1.2, color=col, alpha=0.7)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("normalised enhancement", fontsize=9)
    ax.set_title("(d) Counterfactual: changing TCXM params DOES change shape\n"
                 "(solid = FXL, dashed = SXL; colour pairs nearly overlay)",
                 fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Why FXL and SXL produce similar curve SHAPES:\n"
                 "water exchange rescales amplitude; the kinetic model sets the shape",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "FXL_vs_SXL_shape_explanation", SUBDIR)

    # ---- printed summary ---------------------------------------------
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    print(f"  shape RMS difference (normalised):  {rms:.4f}")
    print(f"  Pearson r between raw curves:       {r:.6f}")
    print(f"  amplitude difference at peak:       "
          f"{S_fxl.max()-S_sxl.max():.1f} a.u.")
    print()
    print("  Shape difference GROWS with enhancement (nonlinearity):")
    print(f"    {'AIF peak':>9} {'peak enh':>10} {'shape RMS':>11}")
    for pk in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        a = synthetic_aif(t, peak_mM=pk)
        cv, ce = tcxm_ct_compartments(theta, t, a)
        f = fxl_limit_reference(cv, ce, ve, vb, acq, subject, aif_cfg, 1000.0)
        s = sxl_limit_reference(cv, ce, ve, vb, acq, subject, aif_cfg, 1000.0)
        print(f"    {pk:>9.1f} {f.max()/f[:9].mean():>10.3f} "
              f"{np.sqrt(np.mean((_norm(f)-_norm(s))**2)):>11.4f}")
    print()
    print("  KEY POINT: that small residual shape difference IS the only")
    print("  information the fit has to estimate tau. Its smallness is")
    print("  exactly why tau_i is weakly identifiable (see Tier B).")
    print("=" * 68)


if __name__ == "__main__":
    main()