#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Fri Aug  7 10:48:27 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_sensitivity_check.py

Measures how much the SSM3 forward model's predicted signal ACTUALLY
moves when tau_b / tau_i are swept across their RelaxationConfig
bounds, and compares that to the noise floor estimated from each
voxel's own baseline frames.

The question this answers
------------------------
Before spending fitting time on tau_b/tau_i, are they identifiable
from YOUR data at YOUR acquisition? If sweeping a parameter across
its entire allowed range moves the predicted curve by less than the
noise, the optimizer cannot recover it -- it will return whatever the
starting point happened to favour, with no warning.

Secondary output: how the sensitivity is distributed ACROSS DECADES,
which is what determines whether log-uniform LHS sampling is spending
its budget where the information actually is.

Pipeline
--------
    load DCE + aorta ROI + lesion ROI
      -> extract plasma AIF from aorta ROI
      -> pick the N best-enhancing lesion voxels
      -> fit GKM (no water exchange) to each  ->  theta_kin
      -> Cv, Ce from gkm_ct_compartments(theta_kin)
      -> sweep tau_b / tau_i through SSM3.to_signal
      -> compare curve movement to that voxel's own noise floor
      -> report the DISTRIBUTION across voxels

Design notes
------------
- Runs against the real SSM3 from dce_mri.water_exchange -- not a
  reimplementation. Same philosophy as water_exchange_test.py.
- Takes the compartment function as an EXPLICIT argument rather than
  reading model.fn_compartments, so this script is unaffected by the
  GKM fn_compartments wiring bug (review item 1.1). If you have since
  fixed that bug, this still works unchanged.
- fit_single_voxel is called WITHOUT wx arguments, so this script runs
  against either the current fitting.py or the wx-integrated one.
- Noise is estimated per-voxel from the baseline frames, so this
  reflects real measured tSNR rather than an assumed value.

Caveat
------
The RMSE-vs-noise comparison here is a fast SCREEN, not a rigorous
identifiability test. It tells you whether a parameter is hopeless;
it does not prove one is well determined (parameters can be
individually sensitive but mutually degenerate -- tau_i trading
against ve, for instance). If this screen comes back MARGINAL or
better, follow up with the profile-likelihood machinery before
trusting any fitted tau.
"""

import numpy as np

from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    AifConfig,
    RelaxationConfig,
    FittingConfig,
)

from dce_mri.water_exchange import SSM3
from dce_mri.signal_models import compute_S0, extract_aif, shift_aif

# =============================================================================
# Core diagnostic  (unchanged from v1)
# =============================================================================

def wx_sensitivity(
    Cv:        np.ndarray,
    Ce:        np.ndarray,
    ve:        float,
    vb:        float,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    aif_cfg:   AifConfig,
    relax_cfg: RelaxationConfig,
    S0:        float,
    sigma:     float,
    tau_b_ref: float = 0.005,
    tau_i_ref: float = 0.3,
    n_grid:    int   = 15,
) -> dict:
    """
    Sweep tau_b and tau_i across their RelaxationConfig bounds, holding
    the other fixed at its reference value, and report how far the
    predicted signal curve moves.

    Returns a dict with, for each of 'tau_b' and 'tau_i':
        grid          : (n_grid,) parameter values swept
        max_rmse      : largest curve-to-curve RMSE across the sweep
        decade_share  : {(lo, hi): fraction of total variation}
    plus sigma, floor (= sigma/sqrt(T)), and T.
    """
    T = len(Cv)
    floor = sigma / np.sqrt(T)

    def curve(tau_b, tau_i):
        return SSM3.to_signal(Cv, Ce, ve, vb, (tau_b, tau_i),
                              acq, subject, aif_cfg, relax_cfg, S0)

    out = {"sigma": sigma, "floor": floor, "T": T}

    for pname, lb, ub, is_tau_b in [
        ("tau_b", relax_cfg.tau_b_lb, relax_cfg.tau_b_ub, True),
        ("tau_i", relax_cfg.tau_i_lb, relax_cfg.tau_i_ub, False),
    ]:
        grid = np.logspace(np.log10(lb), np.log10(ub), n_grid)
        curves = np.array([
            curve(g, tau_i_ref) if is_tau_b else curve(tau_b_ref, g)
            for g in grid
        ])

        # The honest "how much can this parameter move the curve at all"
        # number is the spread between the two most different curves in
        # the sweep, not the spread against an arbitrary reference.
        diffs = curves[:, None, :] - curves[None, :, :]
        rmse_mat = np.sqrt(np.mean(diffs ** 2, axis=2))
        max_rmse = float(rmse_mat.max())

        # Per-decade breakdown -- where along the range does the curve
        # actually change? This is what decides uniform vs log sampling.
        decade_share = {}
        edges = np.arange(np.floor(np.log10(lb)), np.ceil(np.log10(ub)) + 1)
        per_dec, total = [], 0.0
        for i in range(len(edges) - 1):
            d_lo = max(10.0 ** edges[i], lb)
            d_hi = min(10.0 ** edges[i + 1], ub)
            if d_hi <= d_lo:
                continue
            c_lo = curve(d_lo, tau_i_ref) if is_tau_b else curve(tau_b_ref, d_lo)
            c_hi = curve(d_hi, tau_i_ref) if is_tau_b else curve(tau_b_ref, d_hi)
            var = float(np.sqrt(np.mean((c_hi - c_lo) ** 2)))
            per_dec.append(((d_lo, d_hi), var))
            total += var
        for key, var in per_dec:
            decade_share[key] = var / total if total > 0 else 0.0

        out[pname] = {
            "grid":         grid,
            "max_rmse":     max_rmse,
            "decade_share": decade_share,
        }

    return out


def estimate_sigma(sig: np.ndarray, acq: AcquisitionConfig) -> float:
    """
    Per-timepoint noise sigma from the voxel's own pre-contrast frames.

    Uses the standard deviation across baseline frames, which for a
    stable pre-contrast period is dominated by thermal noise. If the
    baseline is drifting (motion, incomplete steady state) this will
    OVERestimate sigma, making the screen conservative -- it will call
    a parameter unidentifiable slightly too readily rather than too
    rarely. That's the safe direction.
    """
    return float(np.std(sig[acq.nbase]))


def verdict_for(ratio: float) -> str:
    if ratio >= 3.0:
        return "IDENTIFIABLE"
    if ratio >= 1.0:
        return "MARGINAL"
    return "NOT IDENTIFIABLE"


# =============================================================================
# Multi-voxel driver
# =============================================================================

def pick_enhancing_voxels(
    sig_4d:  np.ndarray,
    roi:     np.ndarray,
    acq:     AcquisitionConfig,
    subject: SubjectConfig,
    n_vox:   int   = 30,
    lo_frac: float = 1.3,
    hi_frac: float = 0.6,
) -> np.ndarray:

    """
    Return (n_vox, T) signal curves for well-conditioned enhancing
    voxels in roi.
    
    Excludes two failure modes:
      - non-enhancing voxels (ratio too close to 1), which carry no
        information about either kinetics or exchange
      - saturated voxels (ratio too close to the SPGR ceiling), where
        the enhancement-to-concentration inversion is asymptotic and
        small signal differences imply enormous, non-physiological
        concentration swings -- these are what railed every GKM fit
        in the first run of this screen
    
    hi_frac controls how close to the ceiling a voxel is allowed to
    get. 0.6 keeps peak concentrations in the low single-digit mM
    range; raise it if too few voxels survive, lower it if fits are
    still railing.
    """
    curves   = sig_4d[roi, :]                      # (N, T)
    baseline = curves[:, acq.nbase].mean(axis=1)
    peak     = curves.max(axis=1)
    ratio    = peak / (baseline + 1e-11)

    TR_s  = acq.TR / 1000.0
    T10_s = subject.T10 / 1000.0
    FA    = np.deg2rad(acq.FA * subject.B1)
    E10   = np.exp(-TR_s / T10_s)
    ceiling = (1 - E10 * np.cos(FA)) / (1 - E10)

    good = (
        np.isfinite(ratio) & (baseline > 0)
        & (ratio > lo_frac)
        & (ratio < hi_frac * ceiling)
    )
    curves, ratio = curves[good], ratio[good]
    
    print(f"  enhancement band: ({lo_frac:.2f}x, {hi_frac*ceiling:.2f}x]  "
      f"ceiling={ceiling:.2f}x  {len(curves)} voxels selected")
    
    if len(ratio) == 0:
        raise ValueError(
            f"No voxels survived the enhancement band "
            f"({lo_frac:.2f}x, {hi_frac*ceiling:.2f}x] out of ceiling "
            f"{ceiling:.2f}x. Widen lo_frac/hi_frac or check acq/subject."
        )

    # take a spread across the surviving band rather than the extreme
    # top, so the sample isn't biased toward whichever end is riskiest
    rng = np.random.default_rng(0)
    k = min(n_vox, len(ratio))
    idx = rng.choice(len(ratio), size=k, replace=False)
    idx = idx[np.argsort(ratio[idx])[::-1]]
    return curves[idx]

def run_cohort_screen(
    sig_4d:    np.ndarray,
    lesion_roi: np.ndarray,
    aif:       np.ndarray,
    time_var:  np.ndarray,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    aif_cfg:   AifConfig,
    relax_cfg: RelaxationConfig,
    cfg:       FittingConfig,
    n_vox:     int  = 30,
    verbose:   bool = False,
) -> dict:
    """
    Fit TCXM (with AIF delay) to the n_vox best-enhancing lesion
    voxels, then run the tau sensitivity screen on each.

    Every voxel that produces a fit gets a row, tagged by 'status'
    ("ok" / "railed" / "degenerate" / "bad_sigma"), carrying at
    minimum resid_ratio -- so the residual-vs-noise diagnostic can be
    compared ACROSS status groups, not just on survivors. That
    comparison is what separates "TCXM's known fp/ps degeneracy" from
    "something upstream is wrong".
    """
    from dce_mri.fitting import fit_with_delay_grid
    from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments

    curves = pick_enhancing_voxels(sig_4d, lesion_roi, acq, subject, n_vox)
    print(f"Screening {len(curves)} best-enhancing lesion voxels...\n")

    rows = []
    tau_values = np.arange(-10.0, 0, 2.0)
    for i, sig in enumerate(curves):
        fit = fit_with_delay_grid(sig, aif, time_var, TCXM, acq, subject, cfg,
                                  tau_values, seed=1234 + i)

        if not fit.success:
            if verbose:
                print(f"  voxel {i:3d}: TCXM fit failed, skipped")
            rows.append({"status": "fit_failed"})
            continue

        # ---- residual-vs-noise diagnostic, computed for EVERY voxel,
        # before any rail/degenerate filtering -----------------------
        best = fit.per_tau[fit.best_tau]
        resid_rms = float(np.sqrt(np.mean((best.S_pred - best.S_meas) ** 2)))
        sigma = estimate_sigma(sig, acq)
        resid_ratio = resid_rms / sigma if (np.isfinite(sigma) and sigma > 0) else np.nan

        theta_kin = fit.best_params
        ve = float(theta_kin[TCXM.param_names.index("ve")])
        vb = float(theta_kin[TCXM.param_names.index("vp")])

        base_row = {
            "ve": ve, "vb": vb,
            "fp_per_min": float(theta_kin[TCXM.param_names.index("fp")]) * 60.0,
            "ps_per_min": float(theta_kin[TCXM.param_names.index("ps")]) * 60.0,
            "best_tau":   float(fit.best_tau),
            "enh":        float(sig.max() / (sig[acq.nbase].mean() + 1e-11)),
            "sigma":      sigma,
            "resid_rms":  resid_rms,
            "resid_ratio": resid_ratio,
        }

        # flag any parameter within 1% of its bound
        railed = [nm for nm, v, u, l in
                 zip(TCXM.param_names, theta_kin, TCXM.ub, TCXM.lb)
                 if v > u * 0.99 or v < l * 1.01]
        if railed:
            if verbose:
                print(f"  voxel {i:3d}: railed on {railed}, "
                      f"resid/sigma={resid_ratio:.2f}, skipped")
            rows.append({**base_row, "status": "railed", "railed_on": railed})
            continue

        # Guard the degenerate corners SSM3 divides by. A voxel fitted
        # to ve~0 or vb~0 has no meaningful compartment to exchange
        # with, and its density division blows up -- exclude rather
        # than let it dominate the distribution.
        if ve < 1e-3 or vb < 1e-3 or (ve + vb) > 0.99:
            if verbose:
                print(f"  voxel {i:3d}: degenerate ve={ve:.4f} vb={vb:.4f}, "
                      f"resid/sigma={resid_ratio:.2f}, skipped")
            rows.append({**base_row, "status": "degenerate"})
            continue

        if not np.isfinite(sigma) or sigma <= 0:
            if verbose:
                print(f"  voxel {i:3d}: bad sigma, skipped")
            rows.append({**base_row, "status": "bad_sigma"})
            continue

        # ---- compartment curves, wx sensitivity screen ---------------
        # tcxm_ct_compartments passed explicitly, NOT via
        # TCXM.fn_compartments.
        aif_shifted = shift_aif(aif, fit.best_tau, time_var, acq, method="pchip")
        Cv, Ce = tcxm_ct_compartments(theta_kin, time_var, aif_shifted)

        S0 = float(compute_S0(sig[None, :], acq)[0])

        res = wx_sensitivity(Cv, Ce, ve, vb, acq, subject, aif_cfg,
                             relax_cfg, S0, sigma)

        row = {
            **base_row,
            "status":         "ok",
            "floor":          res["floor"],
            "tau_b_rmse":     res["tau_b"]["max_rmse"],
            "tau_i_rmse":     res["tau_i"]["max_rmse"],
            "tau_b_ratio":    res["tau_b"]["max_rmse"] / res["floor"],
            "tau_i_ratio":    res["tau_i"]["max_rmse"] / res["floor"],
            "tau_b_decades":  res["tau_b"]["decade_share"],
            "tau_i_decades":  res["tau_i"]["decade_share"],
        }
        rows.append(row)

        if verbose:
            print(f"  voxel {i:3d}: enh={row['enh']:.2f} tau={row['best_tau']:5.1f}s "
                  f"fp={row['fp_per_min']:.3f}/min ps={row['ps_per_min']:.3f}/min "
                  f"ve={ve:.3f} vb={vb:.3f} resid/sigma={resid_ratio:.2f} | "
                  f"tau_b {row['tau_b_ratio']:6.2f}x  tau_i {row['tau_i_ratio']:6.2f}x")

    return {"rows": rows}

def run_cohort_screen(
    sig_4d:    np.ndarray,
    lesion_roi: np.ndarray,
    aif:       np.ndarray,
    time_var:  np.ndarray,
    acq:       AcquisitionConfig,
    subject:   SubjectConfig,
    aif_cfg:   AifConfig,
    relax_cfg: RelaxationConfig,
    cfg:       FittingConfig,
    n_vox:     int  = 30,
    verbose:   bool = False,
) -> dict:
    """
    Fit TCXM (with AIF delay) to the n_vox best-enhancing lesion
    voxels, then run the tau sensitivity screen on each.

    Every voxel that produces a fit gets a row, tagged by 'status'
    ("ok" / "railed" / "degenerate" / "bad_sigma"), carrying at
    minimum resid_ratio -- so the residual-vs-noise diagnostic can be
    compared ACROSS status groups, not just on survivors. That
    comparison is what separates "TCXM's known fp/ps degeneracy" from
    "something upstream is wrong".
    """
    from dce_mri.fitting import fit_with_delay_grid
    from dce_mri.kinetic_models import TCXM, tcxm_ct_compartments

    curves = pick_enhancing_voxels(sig_4d, lesion_roi, acq, subject, n_vox)
    print(f"Screening {len(curves)} best-enhancing lesion voxels...\n")

    rows = []
    tau_values = np.arange(-10.0, 0, 2.0)
    for i, sig in enumerate(curves):
        fit = fit_with_delay_grid(sig, aif, time_var, TCXM, acq, subject, cfg,
                                  tau_values, seed=1234 + i)

        if not fit.success:
            if verbose:
                print(f"  voxel {i:3d}: TCXM fit failed, skipped")
            rows.append({"status": "fit_failed"})
            continue

        # ---- residual-vs-noise diagnostic, computed for EVERY voxel,
        # before any rail/degenerate filtering -----------------------
        best = fit.per_tau[fit.best_tau]
        resid_rms = float(np.sqrt(np.mean((best.S_pred - best.S_meas) ** 2)))
        sigma = estimate_sigma(sig, acq)
        resid_ratio = resid_rms / sigma if (np.isfinite(sigma) and sigma > 0) else np.nan

        theta_kin = fit.best_params
        ve = float(theta_kin[TCXM.param_names.index("ve")])
        vb = float(theta_kin[TCXM.param_names.index("vp")])

        base_row = {
            "ve": ve, "vb": vb,
            "fp_per_min": float(theta_kin[TCXM.param_names.index("fp")]) * 60.0,
            "ps_per_min": float(theta_kin[TCXM.param_names.index("ps")]) * 60.0,
            "best_tau":   float(fit.best_tau),
            "enh":        float(sig.max() / (sig[acq.nbase].mean() + 1e-11)),
            "sigma":      sigma,
            "resid_rms":  resid_rms,
            "resid_ratio": resid_ratio,
        }

        # flag any parameter within 1% of its bound
        railed = [nm for nm, v, u, l in
                 zip(TCXM.param_names, theta_kin, TCXM.ub, TCXM.lb)
                 if v > u * 0.99 or v < l * 1.01]
        if railed:
            if verbose:
                print(f"  voxel {i:3d}: railed on {railed}, "
                      f"resid/sigma={resid_ratio:.2f}, skipped")
            rows.append({**base_row, "status": "railed", "railed_on": railed})
            continue

        # Guard the degenerate corners SSM3 divides by. A voxel fitted
        # to ve~0 or vb~0 has no meaningful compartment to exchange
        # with, and its density division blows up -- exclude rather
        # than let it dominate the distribution.
        if ve < 1e-3 or vb < 1e-3 or (ve + vb) > 0.99:
            if verbose:
                print(f"  voxel {i:3d}: degenerate ve={ve:.4f} vb={vb:.4f}, "
                      f"resid/sigma={resid_ratio:.2f}, skipped")
            rows.append({**base_row, "status": "degenerate"})
            continue

        if not np.isfinite(sigma) or sigma <= 0:
            if verbose:
                print(f"  voxel {i:3d}: bad sigma, skipped")
            rows.append({**base_row, "status": "bad_sigma"})
            continue

        # ---- compartment curves, wx sensitivity screen ---------------
        # tcxm_ct_compartments passed explicitly, NOT via
        # TCXM.fn_compartments.
        aif_shifted = shift_aif(aif, fit.best_tau, time_var, acq, method="pchip")
        Cv, Ce = tcxm_ct_compartments(theta_kin, time_var, aif_shifted)

        S0 = float(compute_S0(sig[None, :], acq)[0])

        res = wx_sensitivity(Cv, Ce, ve, vb, acq, subject, aif_cfg,
                             relax_cfg, S0, sigma)

        row = {
            **base_row,
            "status":         "ok",
            "floor":          res["floor"],
            "tau_b_rmse":     res["tau_b"]["max_rmse"],
            "tau_i_rmse":     res["tau_i"]["max_rmse"],
            "tau_b_ratio":    res["tau_b"]["max_rmse"] / res["floor"],
            "tau_i_ratio":    res["tau_i"]["max_rmse"] / res["floor"],
            "tau_b_decades":  res["tau_b"]["decade_share"],
            "tau_i_decades":  res["tau_i"]["decade_share"],
        }
        rows.append(row)

        if verbose:
            print(f"  voxel {i:3d}: enh={row['enh']:.2f} tau={row['best_tau']:5.1f}s "
                  f"fp={row['fp_per_min']:.3f}/min ps={row['ps_per_min']:.3f}/min "
                  f"ve={ve:.3f} vb={vb:.3f} resid/sigma={resid_ratio:.2f} | "
                  f"tau_b {row['tau_b_ratio']:6.2f}x  tau_i {row['tau_i_ratio']:6.2f}x")

    return {"rows": rows}

def report_cohort(out: dict) -> None:
    rows = out["rows"]
    if not rows:
        print("No voxels survived fitting -- nothing to report.")
        return

    print()
    print("=" * 72)
    print(f"  WATER-EXCHANGE IDENTIFIABILITY SCREEN  ({len(rows)} voxels attempted)")
    print("=" * 72)

    # ---- residual-vs-noise, split by status, BEFORE any tau reporting.
    # This is the check that separates "TCXM's known fp/ps degeneracy"
    # from "something upstream is wrong" -- compare railed to ok.
    print("  Residual vs. noise floor, by fit outcome:")
    for status in ("ok", "railed", "degenerate"):
        grp = [r["resid_ratio"] for r in rows
              if r.get("status") == status and np.isfinite(r.get("resid_ratio", np.nan))]
        if grp:
            print(f"    resid/sigma, {status:>10} (n={len(grp):3d}): median {np.median(grp):.2f}")
    print()

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print("No voxels reached the wx sensitivity screen -- stopping here.")
        return

    enh = np.array([r["enh"] for r in ok_rows])
    print(f"  enhancement ratio   : median {np.median(enh):.2f} "
          f"[{enh.min():.2f} - {enh.max():.2f}]")
    fp = np.array([r["fp_per_min"] for r in ok_rows])
    ps = np.array([r["ps_per_min"] for r in ok_rows])
    print(f"  fp (/min)           : median {np.median(fp):.3f} "
          f"[{fp.min():.3f} - {fp.max():.3f}]")
    print(f"  ps (/min)           : median {np.median(ps):.3f} "
          f"[{ps.min():.3f} - {ps.max():.3f}]")
    print()

    for pname in ("tau_b", "tau_i"):
        ratios = np.array([r[f"{pname}_ratio"] for r in ok_rows])
        rmses  = np.array([r[f"{pname}_rmse"] for r in ok_rows])
        med    = float(np.median(ratios))

        n_id   = int(np.sum(ratios >= 3.0))
        n_marg = int(np.sum((ratios >= 1.0) & (ratios < 3.0)))
        n_no   = int(np.sum(ratios < 1.0))

        print(f"  {pname}")
        print(f"    curve RMSE across full bounds : median {np.median(rmses):.4f}")
        print(f"    ratio to detection floor      : median {med:6.2f}x   "
              f"[{ratios.min():.2f} - {ratios.max():.2f}]")
        print(f"    per-voxel verdicts            : "
              f"{n_id} identifiable / {n_marg} marginal / {n_no} not")
        print(f"    COHORT VERDICT                : {verdict_for(med)}")

        keys = list(ok_rows[0][f"{pname}_decades"].keys())
        print(f"    variation by decade (median across voxels):")
        for key in keys:
            share = np.median([r[f"{pname}_decades"].get(key, 0.0) for r in ok_rows])
            bar = "#" * int(round(share * 40))
            print(f"      [{key[0]:8.1e} , {key[1]:8.1e}]  {100*share:5.1f}%  {bar}")
        print()

    print("  How to read the resid/sigma block above")
    print("  ----------------------------------------")
    print("  ~1x on BOTH railed and ok   : railing is pure parameter")
    print("                                degeneracy -- fine to proceed.")
    print("  railed >> ok                : railed voxels are unfittable,")
    print("                                not just underdetermined --")
    print("                                look upstream (AIF, timing).")
    print()
    print("  How to read the tau verdicts")
    print("  -----------------------------")
    print("  ratio < 1x  : sweeping the parameter across its ENTIRE allowed")
    print("                range moves the curve less than the noise. The fit")
    print("                cannot recover it -- consider fixing it instead.")
    print("  1x - 3x     : marginal. Confirm with profile likelihood before")
    print("                trusting any fitted value.")
    print("  > 3x        : worth fitting.")
    print()
    print("  A decade holding a small share of the variation is a decade")
    print("  log-uniform LHS is over-sampling. If one decade dominates,")
    print("  narrow the bounds to it or switch that dimension to uniform.")
    print("=" * 72)
    
import numpy as np
from dce_mri.fitting import (fit_single_voxel, fit_with_delay_grid,
                             _lhs_starts, _top_finite_indices,
                             _validate_wx_compatibility)
from dce_mri.water_exchange import FXL, SSM3
from dce_mri.signal_models import compute_S0


def check_shapes(sig, aif, time_var, label=""):
    """Catch the sig_4d / stale-array class of bug before anything else runs."""
    print(f"--- shape check {label} ---")
    print(f"  sig:      {np.asarray(sig).shape}")
    print(f"  aif:      {np.asarray(aif).shape}")
    print(f"  time_var: {np.asarray(time_var).shape}")
    assert np.asarray(sig).ndim == 1, "sig is not 1-D -- did you pass sig_4d by mistake?"
    assert len(sig) == len(aif) == len(time_var), \
        f"length mismatch: sig={len(sig)} aif={len(aif)} time_var={len(time_var)}"
    print("  PASS\n")


def test_guardrail_rejects_fxl(sig, aif, time_var, model, acq, subject, cfg,
                                aif_cfg, relax_cfg):
    print("--- guardrail: FXL should be rejected ---")
    try:
        fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
                         wx_model=FXL, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
        print("  FAIL: no exception raised")
        return False
    except ValueError as e:
        print("  PASS:", e, "\n")
        return True


def test_guardrail_accepts_ssm3(sig, aif, time_var, model, acq, subject, cfg,
                                aif_cfg, relax_cfg):
    print("--- guardrail: SSM3 should be accepted ---")
    try:
        fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
                         wx_model=SSM3, aif_cfg=aif_cfg, relax_cfg=relax_cfg)
        print("  PASS: no exception\n")
        return True
    except ValueError as e:
        print("  FAIL:", e, "\n")
        return False


def test_wx_vs_no_wx(sig, aif, time_var, model, acq, subject, cfg,
                     aif_cfg, relax_cfg, seed=42):
    print("--- wx branch actually runs and differs from non-wx ---")
    r_no_wx = fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
                               seed=seed)
    r_wx    = fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
                               seed=seed, wx_model=SSM3,
                               aif_cfg=aif_cfg, relax_cfg=relax_cfg)

    n_kin = len(model.p0)
    print(f"  no-wx params (len={len(r_no_wx.params)}): {r_no_wx.params}")
    print(f"  wx    params (len={len(r_wx.params)}):    {r_wx.params}")
    print(f"  wx_params: {r_wx.wx_params}  wx_model_name: {r_wx.wx_model_name}")

    assert len(r_no_wx.params) == n_kin, "non-wx params length changed unexpectedly"
    assert len(r_wx.params) == n_kin, "wx params should still be kinetic-only length"
    assert r_no_wx.wx_params is None, "non-wx result should have wx_params=None"
    assert r_wx.wx_params is not None, "wx branch never populated wx_params"
    expected_n_wx = 2 if relax_cfg.fit_tau_b else 1
    assert len(r_wx.wx_params) == expected_n_wx, \
        f"expected {expected_n_wx} wx_params (fit_tau_b={relax_cfg.fit_tau_b}), got {len(r_wx.wx_params)}"
    assert r_wx.wx_model_name == "SSM3"

    same = np.allclose(r_no_wx.params, r_wx.params)
    print(f"  kinetic params identical between wx/non-wx fits: {same}"
          f"  {'(re-check residual_fn is really calling wx_model.to_signal)' if same else '(expected to differ)'}")
    print("  PASS\n")
    return r_no_wx, r_wx


import dataclasses

def test_fit_tau_b_toggle(sig, aif, time_var, model, acq, subject, cfg,
                          aif_cfg, relax_cfg, seed=42):
    print("--- fit_tau_b=False vs True produce different wx_params length ---")

    relax_cfg_fixed = dataclasses.replace(relax_cfg, fit_tau_b=False)
    relax_cfg_free  = dataclasses.replace(relax_cfg, fit_tau_b=True)

    r_fixed = fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
                               seed=seed, wx_model=SSM3,
                               aif_cfg=aif_cfg, relax_cfg=relax_cfg_fixed)
    r_free = fit_single_voxel(sig, aif, time_var, model, acq, subject, cfg,
                              seed=seed, wx_model=SSM3,
                              aif_cfg=aif_cfg, relax_cfg=relax_cfg_free)

    print(f"  fit_tau_b=False -> wx_params len {len(r_fixed.wx_params)}: {r_fixed.wx_params}")
    print(f"  fit_tau_b=True  -> wx_params len {len(r_free.wx_params)}:  {r_free.wx_params}")
    assert len(r_fixed.wx_params) == 1
    assert len(r_free.wx_params) == 2
    print("  PASS\n")


def test_delay_grid_wx_threading(sig, aif, time_var, model, acq, subject, cfg,
                                 aif_cfg, relax_cfg, tau_values, seed=42):
    """Only passes once fit_with_delay_grid's signature is patched per above."""
    print("--- fit_with_delay_grid threads wx_model through ---")
    try:
        result = fit_with_delay_grid(sig, aif, time_var, model, acq, subject,
                                     cfg, tau_values, seed=seed,
                                     wx_model=SSM3, aif_cfg=aif_cfg,
                                     relax_cfg=relax_cfg)
    except TypeError as e:
        print(f"  FAIL: {e}")
        print("  -> fit_with_delay_grid needs wx_model/aif_cfg/relax_cfg "
              "added to its signature and forwarded to fit_single_voxel.")
        return None

    best = result.per_tau[result.best_tau]
    print(f"  best_tau={result.best_tau}  wx_params={best.wx_params}"
          f"  wx_model_name={best.wx_model_name}")
    assert best.wx_params is not None, \
        "delay grid ran but the winning fit has no wx_params -- wx_model wasn't threaded through"
    print("  PASS\n")
    return result


def test_lhs_coverage(sig, aif, time_var, model, acq, subject, cfg,
                      wx_model, aif_cfg, relax_cfg, seed=42):
    """
    How much of the bounded parameter box does LHS cover, and how much
    of that survives into the top-n_top TRF candidates?
    """
    _validate_wx_compatibility(wx_model, cfg.fit_space, model, time_var,
                               aif, aif_cfg, relax_cfg)

    n_kin = len(model.p0)
    S0 = float(compute_S0(sig[None, :], acq)[0])
    target = sig
    ve_idx = model.param_names.index("ve")
    vp_idx = model.param_names.index("vp")

    if relax_cfg.fit_tau_b:
        lb_use = np.concatenate([model.lb, [relax_cfg.tau_b_lb, relax_cfg.tau_i_lb]])
        ub_use = np.concatenate([model.ub, [relax_cfg.tau_b_ub, relax_cfg.tau_i_ub]])
        log_dims_use = tuple(model.log_dims or ()) + (n_kin + 1,)
        names = list(model.param_names) + ["tau_b", "tau_i"]
    else:
        lb_use = np.concatenate([model.lb, [relax_cfg.tau_i_lb]])
        ub_use = np.concatenate([model.ub, [relax_cfg.tau_i_ub]])
        log_dims_use = tuple(model.log_dims or ()) + (n_kin,)
        names = list(model.param_names) + ["tau_i"]

    def residual_fn(p_aug):
        theta_kin = p_aug[:n_kin]
        if relax_cfg.fit_tau_b:
            tau_b, tau_i = p_aug[n_kin], p_aug[n_kin + 1]
        else:
            tau_b, tau_i = relax_cfg.tau_b_fixed, p_aug[n_kin]
        if model.constraints is not None and not model.constraints(theta_kin):
            return np.full_like(target, 1e6)
        try:
            Cv, Ce = model.fn_compartments(theta_kin, time_var, aif)
            ve, vb = theta_kin[ve_idx], theta_kin[vp_idx]
            S_pred = wx_model.to_signal(Cv, Ce, ve, vb, (tau_b, tau_i),
                                        acq, subject, aif_cfg, relax_cfg, S0)
        except Exception:
            return np.full_like(target, 1e6)
        if not np.all(np.isfinite(S_pred)):
            return np.full_like(target, 1e6)
        return S_pred - target

    samples = _lhs_starts(cfg.n_lhs, lb_use, ub_use,
                          model.constraints, log_dims_use, seed)

    # catch the still-open _lhs_starts clip bug (item 1.4) before it
    # confuses anything downstream
    out_of_bounds = np.any((samples < lb_use) | (samples > ub_use))
    if out_of_bounds:
        n_bad = np.sum(np.any((samples < lb_use) | (samples > ub_use), axis=1))
        print(f"  WARNING: {n_bad}/{cfg.n_lhs} LHS samples fall outside "
              f"[lb_use, ub_use] -- this is the known _lhs_starts repair-loop "
              f"bug (item 1.4, still unfixed). These will silently fail in "
              f"_trf_refine and be dropped.\n")

    costs   = np.array([np.sum(residual_fn(p) ** 2) for p in samples])
    top_idx = _top_finite_indices(costs, cfg.n_top)

    n_finite = np.sum(np.isfinite(costs))
    print(f"n_lhs={cfg.n_lhs}  n_finite_cost={n_finite}  n_top={cfg.n_top}  "
          f"n_top_actual={len(top_idx)}\n")

    if len(top_idx) == 0:
        print("No finite-cost candidates at all -- coverage is moot, "
              "every LHS draw failed the forward model or constraints.")
        return

    top_samples = samples[top_idx]

    print(f"{'param':>8} {'lb':>10} {'ub':>10} | "
          f"{'LHS min':>10} {'LHS max':>10} {'LHS frac':>9} | "
          f"{'top min':>10} {'top max':>10} {'top frac':>9}")
    for j, nm in enumerate(names):
        lo, hi = lb_use[j], ub_use[j]
        width = hi - lo
        lhs_frac = (samples[:, j].max() - samples[:, j].min()) / width
        top_frac = (top_samples[:, j].max() - top_samples[:, j].min()) / width
        print(f"{nm:>8} {lo:>10.4g} {hi:>10.4g} | "
              f"{samples[:,j].min():>10.4g} {samples[:,j].max():>10.4g} {lhs_frac:>8.1%} | "
              f"{top_samples[:,j].min():>10.4g} {top_samples[:,j].max():>10.4g} {top_frac:>8.1%}")

    print()
    print("Read this as:")
    print("  LHS frac ~1.0        : raw sampler covers the full box (expected)")
    print("  top frac << LHS frac : the top-n_top cut is COLLAPSING this")
    print("    dimension to a narrow slice before TRF gets to search it.")

    for j in log_dims_use:
        nm = names[j]
        lo, hi = lb_use[j], ub_use[j]
        edges = np.logspace(np.log10(lo), np.log10(hi),
                            int(np.ceil(np.log10(hi / lo))) + 1)
        print(f"\n  {nm} decade coverage (LHS n={cfg.n_lhs} vs top n={len(top_idx)}):")
        for k in range(len(edges) - 1):
            d_lo, d_hi = edges[k], edges[k + 1]
            n_lhs_in = np.sum((samples[:, j] >= d_lo) & (samples[:, j] < d_hi))
            n_top_in = np.sum((top_samples[:, j] >= d_lo) & (top_samples[:, j] < d_hi))
            print(f"    [{d_lo:8.1e}, {d_hi:8.1e})  LHS: {n_lhs_in:4d}  top: {n_top_in:3d}")


def run_all_verifications(sig, aif, time_var, model, acq, subject, cfg,
                          aif_cfg, relax_cfg, tau_values=None):
    check_shapes(sig, aif, time_var, label="voxel under test")

    ok = True
    ok &= test_guardrail_rejects_fxl(sig, aif, time_var, model, acq, subject,
                                     cfg, aif_cfg, relax_cfg)
    ok &= test_guardrail_accepts_ssm3(sig, aif, time_var, model, acq, subject,
                                      cfg, aif_cfg, relax_cfg)
    test_wx_vs_no_wx(sig, aif, time_var, model, acq, subject, cfg,
                     aif_cfg, relax_cfg)
    test_fit_tau_b_toggle(sig, aif, time_var, model, acq, subject, cfg,
                          aif_cfg, relax_cfg)
    if tau_values is not None:
        test_delay_grid_wx_threading(sig, aif, time_var, model, acq, subject,
                                     cfg, aif_cfg, relax_cfg, tau_values)
    test_lhs_coverage(sig, aif, time_var, model, acq, subject, cfg,
                      SSM3, aif_cfg, relax_cfg)

    print("=" * 60)
    print("ALL CHECKS COMPLETE" if ok else "SOME CHECKS FAILED -- see above")
    print("=" * 60)
# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from dce_mri.volume import load_nifti_canonical

    # ── acquisition / fitting parameters ─────────────────────────────
    spokes_per_frame = 8
    acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9,
                            nbase=slice(0, 11))   # changes with spokes_per_frame
    cfg = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")

    # AIF baseline window can differ from the tissue one -- the aorta
    # sees the bolus earlier, so its pre-contrast window is shorter.
    aif_cfg   = AifConfig(T10=1600.0, hematocrit=0.42,
                          top_frac=1.0, nbase=slice(0, 8))
    relax_cfg = RelaxationConfig()

    # ── paths ────────────────────────────────────────────────────────
    DCE_PATH  = ("/mnt/labspace/users/allison/fastMRI_data/tight_masks/"
                 "canonical/FMB056/MGRASP_temptv_FMB056_lambda_0_003.nii.gz")
    AIF_PATH  = ("/mnt/labspace/users/allison/fastMRI_data/tight_masks/"
                 "canonical/FMB056/56_Aorta_RotOrtho_6.nii")
    MASK_PATH = ("/mnt/labspace/users/allison/fastMRI_data/tight_masks/"
                 "canonical/FMB056/56_Lesion 3_RotOrtho_6_tight.nii")

    subject = SubjectConfig(
        subject_id         = "FMB056",
        T10                = 1530.0,
        B1                 = 1.0,
        dce_path_override  = DCE_PATH,
        aif_path_override  = AIF_PATH,
        mask_path_override = MASK_PATH,
    )

    # ── load ─────────────────────────────────────────────────────────
    sig_4d, _ = load_nifti_canonical(subject.dce_path)
    mask_aif, _ = load_nifti_canonical(subject.aif_path)
    mask_les, _ = load_nifti_canonical(subject.mask_path)
    aif_ROI    = mask_aif.astype(bool)
    lesion_ROI = mask_les.astype(bool)

    # ── time grid ────────────────────────────────────────────────────
    nspoke   = 288      # total spokes across the whole scan
    scantime = 150.0    # total scan duration, seconds
    dt       = scantime * spokes_per_frame / nspoke
    time_var = np.arange(sig_4d.shape[-1]) * dt

    print(f"Loaded {sig_4d.shape} | {lesion_ROI.sum()} lesion voxels | "
          f"{aif_ROI.sum()} aorta voxels")
    print(f"dt = {dt:.3f} s, {sig_4d.shape[-1]} frames, "
          f"scan = {time_var[-1]:.1f} s")

    # ── AIF ──────────────────────────────────────────────────────────
    aif, aif_signal = extract_aif(sig_4d, aif_ROI, acq, aif_cfg,
                                  B1=subject.B1)
    print(f"AIF: peak {aif.max():.3f} mM at frame {int(np.argmax(aif))}, "
          f"baseline mean {aif[aif_cfg.nbase].mean():+.4f} mM")

    # Sanity check -- a peak in the first couple of frames or a baseline
    # far from zero means the AIF nbase window is wrong, and every
    # number downstream inherits that error.
    if np.argmax(aif) < 3:
        print("  WARNING: AIF peaks almost immediately -- check aif_cfg.nbase")
    if abs(aif[aif_cfg.nbase].mean()) > 0.05:
        print("  WARNING: AIF baseline is not ~0 -- check aif_cfg.nbase/T10")

    # ── screen ───────────────────────────────────────────────────────
    out = run_cohort_screen(sig_4d, lesion_ROI, aif, time_var,
                            acq, subject, aif_cfg, relax_cfg, cfg,
                            n_vox=3, verbose=True)
    report_cohort(out)
    from dce_mri.kinetic_models import TCXM

    voxel_idx = np.argwhere(lesion_ROI)[0]
    sig = sig_4d[tuple(voxel_idx)]
    
    run_all_verifications(sig, aif, time_var, TCXM, acq, subject, cfg,
                          aif_cfg, relax_cfg,
                          tau_values=np.arange(-6.0, 4.0, 2.0))