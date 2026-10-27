#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:00:04 2026

@author: allisonchen
"""

"""
test_lhs_decade_stability.py

Runs the LHS -> top-n_top selection process many times (different
seeds) and tracks, per decade of each log-sampled parameter:

  - mean top share   : on average, what fraction of the n_top slots
                        land in this decade?
  - hole frequency    : in how many of the trials did this decade get
                        ZERO top slots, despite LHS having candidates
                        there? (systematic starvation, not just an
                        unlucky trial)
  - won-trial freq    : in how many trials did this decade capture the
                        SINGLE LARGEST share of top slots? A decade
                        that wins consistently across different seeds
                        is evidence of a genuine feature of the cost
                        surface (e.g. a degenerate ridge) -- not
                        sampling noise, which would spread wins evenly.

This answers two different questions the single-seed test_lhs_coverage
can't distinguish:
  "Am I falling into holes?"      -> look at hole frequency
  "Am I piling into one hole?"    -> look at won-trial frequency
                                      concentrated on one decade

Sanity-checked against a synthetic case with a ridge deliberately
placed in a known decade before being trusted on real data -- see the
chat writeup for that check's output.
"""

import numpy as np
from dce_mri.fitting import _lhs_starts, _top_finite_indices
from dce_mri.signal_models import compute_S0


from dce_mri.kinetic_models import TCXM
from dce_mri.config import (
    AcquisitionConfig,
    SubjectConfig,
    AifConfig,
    RelaxationConfig,
    FittingConfig,
)

from dce_mri.water_exchange import SSM3
from dce_mri.signal_models import compute_S0, extract_aif, shift_aif

def lhs_decade_stability(
    sig, aif, time_var, model, acq, subject, cfg,
    wx_model, aif_cfg, relax_cfg,
    n_trials=30,
    seed_base=0,
):
    """
    Returns a dict keyed by param name, each holding a per-decade
    table of {mean_top_share, hole_freq, won_freq, n_trials}.
    """
    n_kin = len(model.p0)
    S0 = float(compute_S0(sig[None, :], acq)[0])
    target = sig
    ve_idx = model.param_names.index("ve")
    vp_idx = model.param_names.index("vp")

    if relax_cfg.fit_tau_b:
        lb_use = np.concatenate([model.lb, [relax_cfg.tau_b_lb, relax_cfg.tau_i_lb]])
        ub_use = np.concatenate([model.ub, [relax_cfg.tau_b_ub, relax_cfg.tau_i_ub]])
        log_dims_use = tuple(model.log_dims or ()) + (n_kin, n_kin + 1)
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

    # ---- decade edges per log-sampled dimension --------------------
    decade_edges = {}
    for j in log_dims_use:
        lo, hi = lb_use[j], ub_use[j]
        edges = np.logspace(np.log10(lo), np.log10(hi),
                            int(np.ceil(np.log10(hi / lo))) + 1)
        decade_edges[j] = edges

    # ---- accumulate across trials ------------------------------------
    n_decades = {j: len(decade_edges[j]) - 1 for j in log_dims_use}
    top_share_sum = {j: np.zeros(n_decades[j]) for j in log_dims_use}
    hole_count    = {j: np.zeros(n_decades[j], dtype=int) for j in log_dims_use}
    lhs_empty     = {j: np.zeros(n_decades[j], dtype=int) for j in log_dims_use}
    won_count     = {j: np.zeros(n_decades[j], dtype=int) for j in log_dims_use}

    for trial in range(n_trials):
        seed = seed_base + trial
        samples = _lhs_starts(cfg.n_lhs, lb_use, ub_use,
                              model.constraints, log_dims_use, seed)
        costs   = np.array([np.sum(residual_fn(p) ** 2) for p in samples])
        top_idx = _top_finite_indices(costs, cfg.n_top)
        if len(top_idx) == 0:
            continue
        top = samples[top_idx]
        n_top_actual = len(top_idx)

        for j in log_dims_use:
            edges = decade_edges[j]
            counts_lhs = np.zeros(n_decades[j], dtype=int)
            counts_top = np.zeros(n_decades[j], dtype=int)
            for d in range(n_decades[j]):
                lo, hi = edges[d], edges[d + 1]
                counts_lhs[d] = np.sum((samples[:, j] >= lo) & (samples[:, j] < hi))
                counts_top[d] = np.sum((top[:, j] >= lo) & (top[:, j] < hi))

            for d in range(n_decades[j]):
                top_share_sum[j][d] += counts_top[d] / n_top_actual
                if counts_lhs[d] == 0:
                    lhs_empty[j][d] += 1          # LHS itself never reached here this trial
                elif counts_top[d] == 0:
                    hole_count[j][d] += 1          # LHS had candidates, top-n selected none
            winner = int(np.argmax(counts_top))
            won_count[j][winner] += 1

    results = {}
    for j in log_dims_use:
        nm = names[j]
        edges = decade_edges[j]
        table = []
        for d in range(n_decades[j]):
            table.append({
                "range":          (edges[d], edges[d + 1]),
                "mean_top_share": top_share_sum[j][d] / n_trials,
                "hole_freq":      hole_count[j][d],
                "lhs_empty_freq": lhs_empty[j][d],
                "won_freq":       won_count[j][d],
            })
        results[nm] = table

    return results, n_trials


def report_decade_stability(results, n_trials):
    print("=" * 78)
    print(f"  DECADE COVERAGE STABILITY  ({n_trials} independent trials, "
          f"different seeds)")
    print("=" * 78)
    print("  hole_freq      : LHS had candidates here, but top-n_top selected")
    print("                   NONE, this many times out of n_trials.")
    print("                   Near n_trials -> this region is SYSTEMATICALLY")
    print("                   starved, not just unlucky this run.")
    print("  won_freq       : this decade captured the single largest top-n")
    print("                   share, this many times out of n_trials.")
    print("                   Concentrated on one decade across most trials")
    print("                   -> genuine feature of the cost surface (e.g. a")
    print("                   degenerate ridge), not sampling noise.")
    print("  lhs_empty_freq : LHS itself never drew a point here (bounds/log")
    print("                   sampling issue, not a top-n selection issue).")
    print()

    for nm, table in results.items():
        print(f"  {nm}")
        print(f"    {'range':>22} {'mean top%':>10} {'hole_freq':>10} "
              f"{'won_freq':>9} {'lhs_empty':>10}")
        for row in table:
            lo, hi = row["range"]
            bar = "#" * int(round(row["mean_top_share"] * 30))
            print(f"    [{lo:8.1e},{hi:8.1e}) {row['mean_top_share']:>9.1%} "
                  f"{row['hole_freq']:>10d} {row['won_freq']:>9d} "
                  f"{row['lhs_empty_freq']:>10d}  {bar}")
        max_won = max(row["won_freq"] for row in table)
        max_hole = max(row["hole_freq"] for row in table)
        print()
        if max_won >= 0.7 * n_trials:
            winning_range = next(r["range"] for r in table
                                 if r["won_freq"] == max_won)
            print(f"    -> decade {winning_range} wins consistently "
                  f"({max_won}/{n_trials} trials): likely a genuine ridge, "
                  f"not noise.")
        if max_hole >= 0.7 * n_trials:
            hole_range = next(r["range"] for r in table
                              if r["hole_freq"] == max_hole)
            print(f"    -> decade {hole_range} is a persistent hole "
                  f"({max_hole}/{n_trials} trials): this region is never "
                  f"refined by TRF regardless of seed.")
        print()

    print("=" * 78)

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
    
    voxel_idx = np.argwhere(lesion_ROI)[0]
    sig = sig_4d[tuple(voxel_idx)]

    results, n_trials = lhs_decade_stability(
        sig, aif, time_var, TCXM, acq, subject, cfg,
        SSM3, aif_cfg, relax_cfg, n_trials=300
    )
    report_decade_stability(results, n_trials)
