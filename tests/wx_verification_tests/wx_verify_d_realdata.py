#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wx_verify_d_realdata.py
=======================
TIER D -- real-data sanity checks.

Everything in Tiers A-C runs on synthetic data with known ground truth.
This tier runs the finished pipeline on actual FMB data and asks
whether the numbers that come out are physiologically plausible --
WX-8 item 9.

  D1  Voxelwise wx fits on EVERY lesion voxel, fit BOTH plain (no AIF
      delay search) and delay-corrected (fit_with_delay_grid), paired
      per voxel. Produces:
        - parameter maps for every fitted parameter, opaquely
          overlaid on the anatomical slice (background transparent)
        - distributions for ALL parameters (ve, vp, fp, ps, tau_i,
          derived Ktrans, recovered delay), no-delay vs delay-
          corrected overlaid
        - a binary railed/clean spatial map (orange/cyan), showing
          WHERE in the tumour fits are trustworthy vs pinned at a
          bound -- not just how many
  D2  ROI-averaged wx fit, compared against the voxelwise distribution
  D3  Literature plausibility check for breast tau_i
  D4  FXL vs SSM3 model comparison on the same real curves

Why every voxel, not a screened subsample
------------------------------------------
Earlier versions of this tier screened to an enhancement band before
fitting, which is the right choice for a clean identifiability
estimate but throws away exactly the spatial information needed to
see WHERE in the tumour fits fail. D1 now fits every voxel in
lesion_ROI; each voxel's enhancement ratio and whether it falls
inside the previously-used screening band are still recorded, so
results can be stratified after the fact without having thrown
anything away up front.

Why paired no-delay / delay-corrected fitting
-----------------------------------------------
Earlier real-voxel screening in this project found voxels wanting a
nonzero (sometimes negative) AIF delay, and the C2 synthetic test
found a moderate correlation (r~0.61) between recovered delay and
recovered tau_i in joint fits. Fitting every voxel both ways, paired,
directly tests whether that matters for these real curves rather than
assuming it does or doesn't.

Runtime
-------
This is the most expensive script in the suite. Each voxel gets a
plain fit (1 LHS+TRF search) AND a delay-grid fit (1 search per tau in
TAU_VALUES, ~11 by default) -- roughly 12 fits/voxel. For ~450 lesion
voxels that is potentially thousands of fits. An adaptive ETA is
printed after a short warmup on the first few voxels, before
committing to the full run. Use --quick to cap the voxel count and
narrow the tau grid for fast iteration.

Interpretation
--------------
No ground truth here. Checks are plausibility/robustness flags, not
correctness assertions. A tau_i outside the literature range, or a
high railed fraction, is a finding to investigate, not proof of a bug.

Figures saved to ~/wx_verification_results/<timestamp>/tier_d/
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            RoiFittingConfig, AifConfig, RelaxationConfig)
from dce_mri.volume import load_nifti_canonical
from dce_mri.signal_models import extract_aif, compute_S0
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.fitting import fit_single_voxel, fit_with_delay_grid
from dce_mri.fitting_roi import fit_roi

from wx_verify_common import TestLog, savefig, plot_curves

SUBDIR = "tier_d"

# orientation range only -- see module docstring section below for sources
TAU_I_LIT_LO, TAU_I_LIT_HI = 0.05, 1.5

# ---- EDIT THESE for your subject -------------------------------------------
DCE_PATH  = ("/mnt/labspace/users/allison/fastMRI_data/tight_masks/"
             "canonical/FMB056/MGRASP_temptv_FMB056_lambda_0_003.nii.gz")
AIF_PATH  = ("/mnt/labspace/users/allison/fastMRI_data/tight_masks/"
             "canonical/FMB056/56_Aorta_RotOrtho_6.nii")
MASK_PATH = ("/mnt/labspace/users/allison/fastMRI_data/tight_masks/"
             "canonical/FMB056/56_Lesion 3_RotOrtho_6_tight.nii")
SUBJECT_ID = "FMB056"
SPOKES_PER_FRAME = 8
NSPOKE, SCANTIME = 288, 150.0

# delay grid, default (full run). Negative values included deliberately --
# earlier screening found real voxels preferring negative tau.
TAU_VALUES_FULL  = np.arange(-8.0, 12.1, 2.0)
TAU_VALUES_QUICK = np.arange(-6.0, 6.1, 3.0)

# literature source note:
# Intracellular water lifetime tau_i in tumour/breast tissue is generally
# reported ~0.1-1.0 s, with breast studies clustering in the low hundreds
# of ms. See Springer et al. (NMR Biomed 2014) for a shutter-speed tau_i
# review, and Huang et al. (Radiology 2016) for breast DCE-MRI specifically.
# Treat TAU_I_LIT_LO/HI as an orientation range, not a specification.
# ----------------------------------------------------------------------------


def load_real_case():
    acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 11))
    fitting_cfg = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
    roi_cfg = RoiFittingConfig(aggregation="mean", average_space="signal")
    aif_cfg = AifConfig(T10=1600.0, hematocrit=0.42, top_frac=1.0,
                        nbase=slice(0, 8))
    relax_cfg = RelaxationConfig(fit_tau_b=False, tau_b_fixed=0.005,
                                 tau_b_lb=1e-3, tau_b_ub=1e-2,
                                 tau_i_lb=1e-2, tau_i_ub=2.0)
    subject = SubjectConfig(subject_id=SUBJECT_ID, T10=1530.0, B1=1.0,
                            dce_path_override=DCE_PATH,
                            aif_path_override=AIF_PATH,
                            mask_path_override=MASK_PATH)

    sig_4d, _ = load_nifti_canonical(subject.dce_path)
    mask_aif, _ = load_nifti_canonical(subject.aif_path)
    mask_les, _ = load_nifti_canonical(subject.mask_path)
    aif_ROI, lesion_ROI = mask_aif.astype(bool), mask_les.astype(bool)

    dt = SCANTIME * SPOKES_PER_FRAME / NSPOKE
    time_var = np.arange(sig_4d.shape[-1]) * dt

    print(f"Loaded {sig_4d.shape} | {lesion_ROI.sum()} lesion voxels | "
          f"{aif_ROI.sum()} aorta voxels")
    print(f"dt = {dt:.3f} s, {sig_4d.shape[-1]} frames")

    aif, _ = extract_aif(sig_4d, aif_ROI, acq, aif_cfg, B1=subject.B1)
    print(f"AIF: peak {aif.max():.3f} mM at frame {int(np.argmax(aif))}")

    return dict(sig_4d=sig_4d, lesion_ROI=lesion_ROI, aif=aif,
                time_var=time_var, acq=acq, subject=subject,
                fitting_cfg=fitting_cfg, roi_cfg=roi_cfg,
                aif_cfg=aif_cfg, relax_cfg=relax_cfg)


def _enhancement_ceiling(acq, subject):
    TR_s, T10_s = acq.TR / 1000.0, subject.T10 / 1000.0
    FA = np.deg2rad(acq.FA * subject.B1)
    E10 = np.exp(-TR_s / T10_s)
    return (1 - E10 * np.cos(FA)) / (1 - E10)


def all_lesion_voxels(sig_4d, roi, acq, subject, lo_frac=1.3, hi_frac=0.6):
    """
    Returns EVERY voxel in roi -- no subsampling. Each voxel's
    enhancement ratio and whether it would have passed the
    (lo_frac, hi_frac*ceiling) screening band used earlier in this
    project are computed and returned alongside, so results can be
    stratified by data quality after the fact without having excluded
    anything up front.
    """
    coords = np.argwhere(roi)
    curves = sig_4d[roi, :]
    baseline = curves[:, acq.nbase].mean(axis=1)
    peak = curves.max(axis=1)
    ratio = peak / (baseline + 1e-11)
    ceiling = _enhancement_ceiling(acq, subject)
    in_band = (np.isfinite(ratio) & (baseline > 0)
              & (ratio > lo_frac) & (ratio < hi_frac * ceiling))
    print(f"  {len(coords)} total lesion voxels; "
          f"{in_band.sum()} ({100*in_band.mean():.0f}%) fall inside the "
          f"({lo_frac:.2f}x, {hi_frac*ceiling:.2f}x] screening band "
          f"(band shown for reference -- ALL voxels are still fitted)")
    return curves, coords, ratio, in_band, ceiling


def _is_railed(params, ub, ub_tol=0.99, lb=None, lb_tol=1.01):
    if lb is None:
        lb = np.zeros_like(ub)
    return any(v > u * ub_tol or v < l * lb_tol
              for v, u, l in zip(params, ub, lb))


# =============================================================================
# D1 -- fit EVERY voxel, both plain and delay-corrected
# =============================================================================

def run_d1_fits(data, tau_values, max_voxels=None, warmup=3, verbose_every=25):
    """
    Fits every voxel in lesion_ROI (or the first max_voxels, if
    capped) TWICE: plain wx (fit_single_voxel) and delay-grid wx
    (fit_with_delay_grid), paired.

    Prints an adaptive ETA after `warmup` voxels, based on measured
    per-voxel time, before the bulk of the run -- this is the most
    expensive step in the whole suite and the estimate is worth
    seeing before committing.
    """
    curves, coords, enh, in_band, ceiling = all_lesion_voxels(
        data["sig_4d"], data["lesion_ROI"], data["acq"], data["subject"])

    n_total = len(curves)
    if max_voxels is not None and max_voxels < n_total:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, size=max_voxels, replace=False)
        curves, coords, enh, in_band = curves[idx], coords[idx], enh[idx], in_band[idx]
        print(f"  capped to {max_voxels}/{n_total} voxels (random subset)")

    n = len(curves)
    print(f"\n  fitting {n} voxels, both plain and delay-corrected "
          f"({len(tau_values)} taus each) -- up to {1 + len(tau_values)} "
          f"fits/voxel\n")

    rows = []
    t_start = time.perf_counter()
    for i, (sig, xyz) in enumerate(zip(curves, coords)):
        row = dict(coord=tuple(int(v) for v in xyz), enh=float(enh[i]),
                  in_band=bool(in_band[i]))

#        try:
#            r0 = fit_single_voxel(sig, data["aif"], data["time_var"], TCXM,
#                                  data["acq"], data["subject"],
#                                  data["fitting_cfg"], seed=100 + i,
#                                  wx_model=SSM3, aif_cfg=data["aif_cfg"],
#                                  relax_cfg=data["relax_cfg"])
#            if r0.success and r0.wx_params is not None:
#                row.update(
#                    nod_ve=float(r0.params[0]), nod_vp=float(r0.params[1]),
#                    nod_fp=float(r0.params[2]), nod_ps=float(r0.params[3]),
#                    nod_tau_i=float(r0.wx_params[-1]), nod_cost=float(r0.cost),
#                    nod_railed=_is_railed(r0.params, TCXM.ub, lb=TCXM.lb),
#                )
#        except Exception:
#            pass

        try:
            rd = fit_with_delay_grid(sig, data["aif"], data["time_var"], TCXM,
                                     data["acq"], data["subject"],
                                     data["fitting_cfg"], tau_values,
                                     seed=100 + i, wx_model=SSM3,
                                     aif_cfg=data["aif_cfg"],
                                     relax_cfg=data["relax_cfg"])
            if rd.success and rd.wx_params is not None:
                row.update(
                    dly_ve=float(rd.best_params[0]), dly_vp=float(rd.best_params[1]),
                    dly_fp=float(rd.best_params[2]), dly_ps=float(rd.best_params[3]),
                    dly_tau_i=float(rd.wx_params[-1]), dly_cost=float(rd.best_cost),
                    best_tau=float(rd.best_tau),
                    dly_railed=_is_railed(rd.best_params, TCXM.ub, lb=TCXM.lb),
                )
        except Exception:
            pass

        if "nod_tau_i" in row or "dly_tau_i" in row:
            rows.append(row)

        if i + 1 == warmup:
            elapsed = time.perf_counter() - t_start
            per_voxel = elapsed / warmup
            eta_s = per_voxel * (n - warmup)
            print(f"  [ETA] {warmup} voxels took {elapsed:.1f}s "
                  f"({per_voxel:.2f}s/voxel) -> estimated remaining time: "
                  f"{eta_s/60:.1f} min ({eta_s/3600:.2f} hr) "
                  f"for the other {n - warmup} voxels\n")
        if (i + 1) % verbose_every == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  ... {i+1}/{n} voxels done "
                  f"({elapsed/60:.1f} min elapsed, "
                  f"{len(rows)} produced a usable fit)")

    total_time = time.perf_counter() - t_start
    print(f"\n  D1 fitting complete: {len(rows)}/{n} voxels usable, "
          f"total time {total_time/60:.1f} min")
    return rows, ceiling


# =============================================================================
# Figure -- parameter maps, opaque overlay on transparent background
# =============================================================================

def figure_parameter_maps(data, rows, prefix="dly", label="delay-corrected"):
    if not rows:
        print("  no rows -- skipping parameter maps")
        return

    shape3d = data["sig_4d"].shape[:3]
    fields = [
        (f"{prefix}_ve",    "ve",             None,  "viridis"),
        (f"{prefix}_vp",    "vp",             None,  "viridis"),
        (f"{prefix}_fp",    "fp (/min)",      60.0,  "plasma"),
        (f"{prefix}_ps",    "ps (/min)",      60.0,  "plasma"),
        (f"{prefix}_tau_i", "tau_i (s)",      None,  "magma"),
        ("best_tau",        "AIF delay (s)",  None,  "coolwarm"),
    ]

    z_counts = {}
    for r in rows:
        if f"{prefix}_tau_i" not in r:
            continue
        z = r["coord"][2]
        z_counts[z] = z_counts.get(z, 0) + 1
    if not z_counts:
        print("  no voxels with this prefix -- skipping parameter maps")
        return
    best_z = max(z_counts, key=z_counts.get)
    print(f"  displaying slice z={best_z} ({z_counts[best_z]} fitted voxels)")

    anat = data["sig_4d"][:, :, best_z, data["acq"].nbase].mean(axis=-1)

    panels = fields + [("__ktrans__", "Ktrans (/min)", None, "plasma")]
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, scale, cmap_name) in zip(axes, panels):
        pmap = np.full(shape3d, np.nan)
        for r in rows:
            if key == "__ktrans__":
                fp, ps = r.get(f"{prefix}_fp"), r.get(f"{prefix}_ps")
                val = (fp*ps/(fp+ps)*60.0
                      if (fp is not None and ps is not None and fp+ps > 0)
                      else None)
            else:
                val = r.get(key)
            if val is None or not np.isfinite(val):
                continue
            if scale is not None and key != "__ktrans__":
                val = val * scale
            pmap[r["coord"]] = val

        sl = pmap[:, :, best_z]
        masked = np.ma.masked_invalid(sl)
        n_vox_here = int(np.sum(~masked.mask))

        ax.imshow(anat.T, cmap="gray", origin="lower",
                  vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99))

        cmap = cm.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)
        if n_vox_here > 0:
            finite_vals = masked.compressed()
            vmin, vmax = np.percentile(finite_vals, [2, 98])
            if vmin == vmax:
                vmin, vmax = finite_vals.min(), finite_vals.max() + 1e-9
            im = ax.imshow(masked.T, cmap=cmap, origin="lower", alpha=1.0,
                           norm=Normalize(vmin=vmin, vmax=vmax),
                           interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\n({n_vox_here} voxels)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"D1  Parameter maps ({label}), {SUBJECT_ID}, slice z={best_z}\n"
                 f"overlay opaque where fitted; background transparent",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, f"D1_parameter_maps_{prefix}", SUBDIR)


# =============================================================================
# Figure -- railed vs clean, binary spatial map
# =============================================================================

def figure_rail_map(data, rows, prefix="dly", label="delay-corrected"):
    """
    Binary spatial map: orange = railed (a kinetic parameter within 1%
    of a bound), cyan = clean, transparent = not fitted. Shows WHERE
    in the tumour fits are trustworthy, not just what fraction.
    """
    key = f"{prefix}_railed"
    have = [r for r in rows if key in r]
    if not have:
        print("  no rows with rail status -- skipping rail map")
        return

    shape3d = data["sig_4d"].shape[:3]
    z_counts = {}
    for r in have:
        z = r["coord"][2]
        z_counts[z] = z_counts.get(z, 0) + 1
    best_z = max(z_counts, key=z_counts.get)

    # 0 = clean, 1 = railed, encoded as a masked array so "not fitted"
    # is transparent rather than a third colour
    code = np.full(shape3d, np.nan)
    for r in have:
        code[r["coord"]] = 1.0 if r[key] else 0.0

    anat = data["sig_4d"][:, :, best_z, data["acq"].nbase].mean(axis=-1)
    sl = code[:, :, best_z]
    masked = np.ma.masked_invalid(sl)
    n_here = int(np.sum(~masked.mask))
    n_railed = int(np.nansum(sl == 1.0))
    n_clean = int(np.nansum(sl == 0.0))

    cmap = ListedColormap(["cyan", "orange"])   # 0->cyan, 1->orange
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(anat.T, cmap="gray", origin="lower",
             vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99))
    if n_here > 0:
        ax.imshow(masked.T, cmap=cmap, origin="lower", alpha=1.0,
                  norm=Normalize(vmin=0, vmax=1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    frac_railed = n_railed / max(n_here, 1)
    ax.set_title(
        f"D1  Railed (orange) vs clean (cyan) -- {label}\n"
        f"{SUBJECT_ID}, slice z={best_z}: "
        f"{n_railed}/{n_here} railed ({frac_railed:.0%})",
        fontsize=11)

    # simple legend patches since this isn't a continuous colourbar
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="cyan", label="clean"),
                       Patch(facecolor="orange", label="railed"),
                       Patch(facecolor="none", label="not fitted "
                             "(transparent)")],
             loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    savefig(fig, f"D1_rail_map_{prefix}", SUBDIR)

    print(f"  rail map ({label}): {n_railed}/{n_here} railed "
          f"({frac_railed:.0%}) on slice z={best_z}")


def figure_rail_map_comparison(data, rows):
    """Side-by-side no-delay vs delay-corrected rail maps, same slice,
    so the effect of delay correction on railing is directly visible."""
    paired = [r for r in rows if "nod_railed" in r and "dly_railed" in r]
    if not paired:
        print("  no voxels with both rail statuses -- skipping comparison")
        return

    shape3d = data["sig_4d"].shape[:3]
    z_counts = {}
    for r in paired:
        z = r["coord"][2]
        z_counts[z] = z_counts.get(z, 0) + 1
    best_z = max(z_counts, key=z_counts.get)
    anat = data["sig_4d"][:, :, best_z, data["acq"].nbase].mean(axis=-1)

    cmap = ListedColormap(["cyan", "orange"])
    cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, prefix, label in zip(axes, ["nod", "dly"],
                                 ["no delay", "delay-corrected"]):
        code = np.full(shape3d, np.nan)
        for r in paired:
            code[r["coord"]] = 1.0 if r[f"{prefix}_railed"] else 0.0
        sl = code[:, :, best_z]
        masked = np.ma.masked_invalid(sl)
        n_here = int(np.sum(~masked.mask))
        n_railed = int(np.nansum(sl == 1.0))

        ax.imshow(anat.T, cmap="gray", origin="lower",
                 vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99))
        if n_here > 0:
            ax.imshow(masked.T, cmap=cmap, origin="lower", alpha=1.0,
                      norm=Normalize(vmin=0, vmax=1), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        frac = n_railed / max(n_here, 1)
        ax.set_title(f"{label}\n{n_railed}/{n_here} railed ({frac:.0%})",
                    fontsize=11)

    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor="cyan", label="clean"),
                            Patch(facecolor="orange", label="railed")],
                  loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"D1  Does AIF delay correction change WHERE fits rail?\n"
                 f"{SUBJECT_ID}, slice z={best_z}, same {len(paired)} voxels "
                 f"both ways", fontsize=12)
    fig.tight_layout()
    savefig(fig, "D1_rail_map_comparison", SUBDIR)


# =============================================================================
# Figure -- all parameter distributions, no-delay vs delay-corrected
# =============================================================================

def figure_all_parameter_distributions(data, rows):
    if not rows:
        print("  no rows -- skipping distributions")
        return

    relax = data["relax_cfg"]
    specs = [
        ("ve",    "ve",               1.0,  (TCXM.lb[0], TCXM.ub[0]), False, None),
        ("vp",    "vp",               1.0,  (TCXM.lb[1], TCXM.ub[1]), False, None),
        ("fp",    "fp (/min)",        60.0, (TCXM.lb[2]*60, TCXM.ub[2]*60), True, None),
        ("ps",    "ps (/min)",        60.0, (TCXM.lb[3]*60, TCXM.ub[3]*60), True, None),
        ("tau_i", "tau_i (s)",        1.0,  (relax.tau_i_lb, relax.tau_i_ub), True,
         (TAU_I_LIT_LO, TAU_I_LIT_HI)),
        ("cost",  "final cost (RSS)", 1.0,  None, True, None),
    ]

    ncols = 3
    nrows = int(np.ceil((len(specs) + 2) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 3.8*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, scale, bounds, logx, lit) in zip(axes, specs):
        nod = np.array([r[f"nod_{key}"] * scale for r in rows
                        if f"nod_{key}" in r and np.isfinite(r[f"nod_{key}"])])
        dly = np.array([r[f"dly_{key}"] * scale for r in rows
                        if f"dly_{key}" in r and np.isfinite(r[f"dly_{key}"])])
        allv = (np.concatenate([v for v in (nod, dly) if len(v)])
               if (len(nod) or len(dly)) else np.array([]))
        if len(allv) == 0:
            ax.axis("off"); continue

        if logx and allv.min() > 0:
            bins = np.logspace(np.log10(allv.min()*0.9),
                               np.log10(allv.max()*1.1), 25)
            ax.set_xscale("log")
        else:
            bins = np.linspace(allv.min(), allv.max(), 25)

        if len(nod):
            ax.hist(nod, bins=bins, alpha=0.6, color="steelblue",
                    edgecolor="k", label=f"no delay (n={len(nod)})")
        if len(dly):
            ax.hist(dly, bins=bins, alpha=0.6, color="darkorange",
                    edgecolor="k", label=f"delay-corrected (n={len(dly)})")
        if bounds is not None:
            for b in bounds:
                ax.axvline(b, color="gray", ls=":", lw=1.2)
        if lit is not None:
            ax.axvspan(lit[0], lit[1], alpha=0.15, color="green",
                      label="literature range")
        ax.set_xlabel(title, fontsize=9)
        ax.set_ylabel("voxel count", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[len(specs)]
    def _kt(prefix):
        out = []
        for r in rows:
            fp, ps = r.get(f"{prefix}_fp"), r.get(f"{prefix}_ps")
            if fp is not None and ps is not None and fp+ps > 0:
                out.append(fp*ps/(fp+ps)*60.0)
        return np.array(out)
    kt_n, kt_d = _kt("nod"), _kt("dly")
    allk = (np.concatenate([v for v in (kt_n, kt_d) if len(v)])
           if (len(kt_n) or len(kt_d)) else np.array([]))
    if len(allk):
        if allk.min() > 0:
            bins = np.logspace(np.log10(allk.min()*0.9),
                               np.log10(allk.max()*1.1), 25)
            ax.set_xscale("log")
        else:
            bins = np.linspace(allk.min(), allk.max(), 25)
        if len(kt_n):
            ax.hist(kt_n, bins=bins, alpha=0.6, color="steelblue",
                    edgecolor="k", label=f"no delay (n={len(kt_n)})")
        if len(kt_d):
            ax.hist(kt_d, bins=bins, alpha=0.6, color="darkorange",
                    edgecolor="k", label=f"delay-corrected (n={len(kt_d)})")
        ax.set_xlabel("Ktrans (/min)", fontsize=9)
        ax.set_ylabel("voxel count", fontsize=9)
        ax.set_title("Ktrans (derived: fp*ps/(fp+ps))", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    ax = axes[len(specs) + 1]
    bt = np.array([r["best_tau"] for r in rows
                   if "best_tau" in r and np.isfinite(r["best_tau"])])
    if len(bt):
        ax.hist(bt, bins=15, color="seagreen", edgecolor="k", alpha=0.85)
        ax.axvline(0.0, color="r", ls="--", lw=1.5, label="no delay")
        ax.set_xlabel("recovered AIF delay (s)", fontsize=9)
        ax.set_ylabel("voxel count", fontsize=9)
        ax.set_title("recovered delay", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    for ax in axes[len(specs)+2:]:
        ax.axis("off")

    fig.suptitle(f"D1  All fitted parameter distributions, "
                 f"no-delay vs delay-corrected ({SUBJECT_ID})", fontsize=12)
    fig.tight_layout()
    savefig(fig, "D1_all_parameter_distributions", SUBDIR)


# =============================================================================
# D1 summary / checks
# =============================================================================

def summarize_d1(log: TestLog, rows, tau_values):
    print("\n--- D1 summary ---")
    n_nod = sum("nod_tau_i" in r for r in rows)
    n_dly = sum("dly_tau_i" in r for r in rows)
    n_nod_railed = sum(r.get("nod_railed", False) for r in rows if "nod_tau_i" in r)
    n_dly_railed = sum(r.get("dly_railed", False) for r in rows if "dly_tau_i" in r)

    print(f"    voxels with a usable no-delay fit  : {n_nod}"
          f"  ({n_nod_railed} railed, {n_nod_railed/max(n_nod,1):.0%})")
    print(f"    voxels with a usable delay fit     : {n_dly}"
          f"  ({n_dly_railed} railed, {n_dly_railed/max(n_dly,1):.0%})")

    log.record("D1 voxelwise wx fits produced usable results",
               len(rows) > 0, f"{len(rows)} voxels with at least one usable fit")

    paired = [r for r in rows if "nod_tau_i" in r and "dly_tau_i" in r]
    if paired:
        nod = np.array([r["nod_tau_i"] for r in paired])
        dly = np.array([r["dly_tau_i"] for r in paired])
        c_n = np.array([r["nod_cost"] for r in paired])
        c_d = np.array([r["dly_cost"] for r in paired])
        bt = np.array([r["best_tau"] for r in paired])
        med_shift = float(np.median(dly) - np.median(nod))
        frac_better = float(np.mean(c_d < c_n))
        at_edge = float(np.mean((bt <= tau_values.min()+1e-9) |
                                (bt >= tau_values.max()-1e-9)))

        print(f"    paired voxels: {len(paired)}")
        print(f"    tau_i median: no-delay={np.median(nod):.4f}  "
              f"delay={np.median(dly):.4f}  shift={med_shift:+.4f}")
        print(f"    delay fit achieved lower cost on {frac_better:.0%} of "
              f"paired voxels")
        print(f"    recovered delay: median={np.median(bt):+.2f}s, "
              f"{at_edge:.0%} of voxels at a grid edge")

        log.record("D1 delay correction does not substantially shift the "
                   "tau_i distribution (no-delay conclusions robust)",
                   abs(med_shift) < 0.5 * max(np.median(nod), 1e-9) + 0.05,
                   f"median tau_i shift = {med_shift:+.4f}s")
        log.record("D1 delay grid is wide enough (few voxels at a grid edge)",
                   at_edge < 0.2,
                   f"{at_edge:.0%} of voxels at a grid edge")
    return paired


# =============================================================================
# D2 -- ROI vs voxelwise
# =============================================================================
from dce_mri.fitting_roi import fit_roi, fit_roi_with_delay_grid

def test_d2_roi_vs_voxelwise(log: TestLog, data, voxel_rows):
    print("\n--- D2: ROI-averaged wx fit vs the voxelwise distribution ---")
    TAU_VALUES_FULL  = np.arange(-8.0, 12.1, 2.0)
    try:
        r = fit_roi_with_delay_grid(data["sig_4d"], data["lesion_ROI"], data["aif"],
                                    data["time_var"], TCXM, data["acq"], data["subject"],
                                    data["fitting_cfg"], data["roi_cfg"], TAU_VALUES_FULL,
                                    seed=42, wx_model=SSM3, aif_cfg=data["aif_cfg"],
                                    relax_cfg=data["relax_cfg"])
#    try:
#        r = fit_roi(data["sig_4d"], data["lesion_ROI"], data["aif"],
#                   data["time_var"], TCXM, data["acq"], data["subject"],
#                   data["fitting_cfg"], data["roi_cfg"], seed=42,
#                   wx_model=SSM3, aif_cfg=data["aif_cfg"],
#                   relax_cfg=data["relax_cfg"])
    except Exception as e:
        log.record("D2 ROI wx fit runs", False, f"{type(e).__name__}: {e}")
        return None

    roi_tau_i = float(r.wx_params[-1]) if r.wx_params is not None else np.nan
    print(f"    ROI ({r.n_voxels} voxels): tau_i={roi_tau_i:.4f}s  "
          f"ve={r.params[0]:.4f}  vp={r.params[1]:.4f}")
    log.record("D2a ROI wx fit produces a tau_i", np.isfinite(roi_tau_i),
              f"tau_i={roi_tau_i:.4f}s from {r.n_voxels} voxels")

    vox_tau = np.array([x.get("dly_tau_i", x.get("nod_tau_i"))
                        for x in voxel_rows
                        if "dly_tau_i" in x or "nod_tau_i" in x])
    if len(vox_tau):
        lo, hi = np.percentile(vox_tau, [25, 75])
        inside = lo <= roi_tau_i <= hi
        log.record("D2b ROI tau_i falls inside the voxelwise IQR", inside,
                  f"ROI={roi_tau_i:.4f}, voxelwise IQR=[{lo:.4f}, {hi:.4f}]")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_curves(axes[0], data["time_var"],
               {"ROI mean signal": r.St_mean, "SSM3 fit": r.St_pred},
               title=f"D2  ROI mean curve and fit ({r.n_voxels} voxels)")
    if len(vox_tau):
        axes[1].hist(vox_tau, bins=25, color="steelblue", edgecolor="k",
                    alpha=0.85, label="voxelwise (best available per voxel)")
        axes[1].axvline(roi_tau_i, color="crimson", lw=2.5,
                        label=f"ROI fit = {roi_tau_i:.3f}s")
        axes[1].axvspan(TAU_I_LIT_LO, TAU_I_LIT_HI, alpha=0.15, color="green",
                        label="literature range")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("tau_i (s)", fontsize=9)
        axes[1].set_ylabel("voxel count", fontsize=9)
        axes[1].set_title("ROI estimate vs voxelwise distribution", fontsize=10)
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "D2_roi_vs_voxelwise", SUBDIR)
    return r


# =============================================================================
# D3 -- literature plausibility
# =============================================================================

def test_d3_literature(log: TestLog, rows, roi_result):
    print("\n--- D3: literature plausibility for breast tau_i ---")
    print(f"    orientation range: {TAU_I_LIT_LO}-{TAU_I_LIT_HI} s "
          f"(see module docstring for sources)")

    tau = np.array([r.get("dly_tau_i", r.get("nod_tau_i")) for r in rows
                   if "dly_tau_i" in r or "nod_tau_i" in r])
    if len(tau) == 0:
        log.record("D3 literature check", False, "no voxelwise results")
        return
    frac_in = float(np.mean((tau >= TAU_I_LIT_LO) & (tau <= TAU_I_LIT_HI)))
    med = float(np.median(tau))

    print(f"    voxelwise median tau_i = {med:.4f}s")
    print(f"    fraction within literature range = {frac_in:.1%}")
    if roi_result is not None and roi_result.wx_params is not None:
        roi_t = float(roi_result.wx_params[-1])
        print(f"    ROI tau_i = {roi_t:.4f}s  "
              f"({'inside' if TAU_I_LIT_LO <= roi_t <= TAU_I_LIT_HI else 'OUTSIDE'} "
              f"literature range)")

    log.record("D3 median tau_i is physiologically plausible",
              TAU_I_LIT_LO <= med <= TAU_I_LIT_HI,
              f"median={med:.4f}s vs range [{TAU_I_LIT_LO}, {TAU_I_LIT_HI}]")
    print(f"    [report] A median outside the range is a FINDING, not "
          f"necessarily a bug -- given the recovery scatter quantified in "
          f"Tier B, interpret alongside B1/B2.")


# =============================================================================
# D4 -- FXL vs SSM3
# =============================================================================

def test_d4_fxl_vs_ssm3(log: TestLog, data, n_vox=12):
    """
    Kept as a screened-subsample comparison (not all voxels) -- this
    test is about fit-quality comparison, not spatial coverage, and
    does not benefit from the same "every voxel" treatment as D1.
    """
    print("\n--- D4: FXL vs SSM3 on the same real curves ---")
    curves, coords, ratio, in_band, ceiling = all_lesion_voxels(
        data["sig_4d"], data["lesion_ROI"], data["acq"], data["subject"])
    band_curves = curves[in_band]
    if len(band_curves) == 0:
        log.record("D4 FXL/SSM3 comparison ran", False,
                  "no voxels in the screening band")
        return
    rng = np.random.default_rng(0)
    idx = rng.choice(len(band_curves), size=min(n_vox, len(band_curves)),
                     replace=False)
    curves = band_curves[idx]

    fxl_cost, wx_cost, fxl_ve, wx_ve = [], [], [], []
    for i, sig in enumerate(curves):
        try:
            r_fxl = fit_single_voxel(sig, data["aif"], data["time_var"], TCXM,
                                     data["acq"], data["subject"],
                                     data["fitting_cfg"], seed=200 + i)
            r_wx = fit_single_voxel(sig, data["aif"], data["time_var"], TCXM,
                                    data["acq"], data["subject"],
                                    data["fitting_cfg"], seed=200 + i,
                                    wx_model=SSM3, aif_cfg=data["aif_cfg"],
                                    relax_cfg=data["relax_cfg"])
        except Exception:
            continue
        fxl_cost.append(r_fxl.cost); wx_cost.append(r_wx.cost)
        fxl_ve.append(r_fxl.params[0]); wx_ve.append(r_wx.params[0])

    if not fxl_cost:
        log.record("D4 FXL/SSM3 comparison ran", False, "no successful fits")
        return

    fxl_cost, wx_cost = np.array(fxl_cost), np.array(wx_cost)
    better = int(np.sum(wx_cost < fxl_cost))
    print(f"    SSM3 achieved lower RSS on {better}/{len(wx_cost)} voxels")
    log.record("D4 SSM3 fits at least as well as FXL "
              "(expected: it nests FXL as tau->0)",
              better >= 0.5 * len(wx_cost),
              f"{better}/{len(wx_cost)} voxels improved")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(fxl_cost, wx_cost, c="steelblue", edgecolor="k", s=45)
    lim = [min(fxl_cost.min(), wx_cost.min()), max(fxl_cost.max(), wx_cost.max())]
    axes[0].plot(lim, lim, "k--", lw=1, label="equal fit quality")
    axes[0].set_xlabel("FXL cost (RSS)", fontsize=9)
    axes[0].set_ylabel("SSM3 cost (RSS)", fontsize=9)
    axes[0].set_title("fit quality: below the line = SSM3 better", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].scatter(fxl_ve, wx_ve, c="darkorange", edgecolor="k", s=45)
    lim2 = [0, max(max(fxl_ve), max(wx_ve)) * 1.1]
    axes[1].plot(lim2, lim2, "k--", lw=1, label="unchanged")
    axes[1].set_xlabel("ve from FXL fit", fontsize=9)
    axes[1].set_ylabel("ve from SSM3 fit", fontsize=9)
    axes[1].set_title("does accounting for exchange change ve?", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle(f"D4  FXL vs SSM3 on identical real curves, {SUBJECT_ID}",
                fontsize=11)
    fig.tight_layout()
    savefig(fig, "D4_fxl_vs_ssm3", SUBDIR)


# =============================================================================

def run_tier_d(quick=False, max_voxels=None):
    log = TestLog("TIER D -- real-data sanity")
    print("=" * 72)
    print("  TIER D  --  real-data sanity checks")
    print("  NOTE: no ground truth here. Checks are plausibility flags.")
    print("=" * 72)
    data = load_real_case()

    tau_values = TAU_VALUES_QUICK if quick else TAU_VALUES_FULL
    cap = (max_voxels if max_voxels is not None
          else (40 if quick else None))
    if cap is None:
        print("\n  *** fitting ALL lesion voxels, no cap -- this is the "
              "most expensive script in the suite, see ETA below ***")

    rows, ceiling = run_d1_fits(data, tau_values, max_voxels=cap)
    paired = summarize_d1(log, rows, tau_values)

    print("\n--- Figures: parameter maps ---")
    figure_parameter_maps(data, rows, prefix="dly", label="delay-corrected")
    figure_parameter_maps(data, rows, prefix="nod", label="no delay")

    print("\n--- Figures: rail maps ---")
    figure_rail_map(data, rows, prefix="dly", label="delay-corrected")
    figure_rail_map_comparison(data, rows)

    print("\n--- Figure: all parameter distributions ---")
    figure_all_parameter_distributions(data, rows)

    roi_r = test_d2_roi_vs_voxelwise(log, data, rows)
    test_d3_literature(log, rows, roi_r)
    test_d4_fxl_vs_ssm3(log, data, n_vox=6 if quick else 12)

    return log.summary()


if __name__ == "__main__":
    import sys
    run_tier_d(quick="--quick" in sys.argv)