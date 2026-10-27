#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
wx_figures.py
=============
Plug-and-play figure library for water-exchange DCE-MRI results.

Refactored out of the verification suite so the same figures can be
reused on any analysis run, not just the verification tests. Every
function is independent, takes a standard input, and returns the saved
path -- compose them however you like, or drive them by name through
make_figures().

STANDARD INPUT SCHEMA
---------------------
Most functions take `rows`: a list of per-voxel dicts. Only 'coord' is
required; everything else is optional and figures skip gracefully when
a field is missing.

    {
      'coord':    (i, j, k),         # required -- voxel index
      'enh':      2.4,               # enhancement ratio
      'in_band':  True,              # passed the enhancement screen
      've': ..., 'vp': ..., 'fp': ..., 'ps': ...,   # kinetic params
      'tau_i':    0.31,              # water-exchange param
      'best_tau': -2.0,              # AIF delay, if delay-fitted
      'cost':     812.4,
      'railed':   False,
      'boot':     BootstrapWxResult, # optional -- enables uncertainty figures
    }

A PREFIX may be used to hold two fit variants side by side (e.g. 'nod_ve'
and 'dly_ve'); pass prefix='dly' to the figure functions.

USAGE
-----
    from wx_figures import FigureContext, make_figures

    ctx = FigureContext(sig_4d=sig_4d, acq=acq, results_subdir="analysis")
    make_figures(rows, ctx, which=["param_maps", "rail_map",
                                    "distributions", "uncertainty_maps"])

or call individually:

    plot_param_maps(rows, ctx, prefix="dly")
    plot_bootstrap_ci(roi_boot, ctx, param="tau_i")
"""



from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch

from wx_verify_common import savefig


# =============================================================================
# Context -- everything the figures need that isn't in `rows`
# =============================================================================

@dataclass
class FigureContext:
    """
    sig_4d          : 4-D volume, for the anatomical background
    acq             : AcquisitionConfig, for nbase (baseline frames)
    time_var        : optional, for curve plots
    results_subdir  : subfolder under the timestamped results dir
    subject_id      : for figure titles
    tau_i_lit       : (lo, hi) literature orientation range, or None
    """
    sig_4d:         Optional[np.ndarray] = None
    acq:            object = None
    time_var:       Optional[np.ndarray] = None
    results_subdir: str = "figures"
    subject_id:     str = ""
    tau_i_lit:      Optional[tuple] = (0.05, 1.5)

    @property
    def shape3d(self):
        return None if self.sig_4d is None else self.sig_4d.shape[:3]

    def anatomy(self, z):
        """Pre-contrast mean at slice z, for the greyscale background."""
        if self.sig_4d is None or self.acq is None:
            return None
        return self.sig_4d[:, :, z, self.acq.nbase].mean(axis=-1)


# =============================================================================
# Shared helpers
# =============================================================================

def _get(row, key, prefix=None):
    """Fetch `key`, trying the prefixed name first if a prefix is given."""
    if prefix:
        v = row.get(f"{prefix}_{key}")
        if v is not None:
            return v
    return row.get(key)


def build_map(rows, key, shape3d, prefix=None, scale=None, transform=None):
    """
    Place a per-voxel scalar back into a 3-D array. Unfitted voxels stay
    NaN, which is what makes them transparent in the overlay figures.
    """
    m = np.full(shape3d, np.nan)
    for r in rows:
        v = transform(r) if transform is not None else _get(r, key, prefix)
        if v is None or not np.isfinite(v):
            continue
        m[r["coord"]] = v * scale if scale is not None else v
    return m


def pick_slice(rows, key=None, prefix=None):
    """Slice index holding the most voxels with a usable value."""
    counts = {}
    for r in rows:
        if key is not None and _get(r, key, prefix) is None:
            continue
        z = r["coord"][2]
        counts[z] = counts.get(z, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _overlay(ax, anat, sl, cmap_name, vmin=None, vmax=None, discrete=False):
    """
    Greyscale anatomy underneath, parameter overlay on top -- OPAQUE
    where a value exists, fully transparent where it doesn't.
    """
    if anat is not None:
        ax.imshow(anat.T, cmap="gray", origin="lower",
                  vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99))
    masked = np.ma.masked_invalid(sl)
    n = int(np.sum(~masked.mask))
    im = None
    if n > 0:
        cmap = (ListedColormap(cmap_name).copy() if discrete
                else cm.get_cmap(cmap_name).copy())
        cmap.set_bad(alpha=0.0)               # background transparent
        vals = masked.compressed()
        if vmin is None or vmax is None:
            vmin, vmax = np.percentile(vals, [2, 98])
            if vmin == vmax:
                vmin, vmax = vals.min(), vals.max() + 1e-9
        im = ax.imshow(masked.T, cmap=cmap, origin="lower", alpha=1.0,
                       norm=Normalize(vmin=vmin, vmax=vmax),
                       interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return im, n


def _ktrans(row, prefix=None):
    fp, ps = _get(row, "fp", prefix), _get(row, "ps", prefix)
    if fp is None or ps is None or (fp + ps) <= 0:
        return None
    return fp * ps / (fp + ps) * 60.0


# =============================================================================
# Figure: parameter maps
# =============================================================================

# (key, display title, unit scale, colormap)
DEFAULT_PARAM_PANELS = [
    ("ve",       "ve",            None, "viridis"),
    ("vp",       "vp",            None, "viridis"),
    ("fp",       "fp (/min)",     60.0, "plasma"),
    ("ps",       "ps (/min)",     60.0, "plasma"),
    ("tau_i",    "tau_i (s)",     None, "magma"),
    ("best_tau", "AIF delay (s)", None, "coolwarm"),
    ("cost",     "cost (RSS)",    None, "cividis"),
]


def plot_param_maps(rows, ctx: FigureContext, prefix=None, z=None,
                    panels=None, include_ktrans=True, name=None,
                    label=""):
    """Parameter maps, opaque overlay on transparent background."""
    if not rows or ctx.shape3d is None:
        print("  plot_param_maps: nothing to plot")
        return None
    panels = list(panels or DEFAULT_PARAM_PANELS)
    if include_ktrans:
        panels.append(("__ktrans__", "Ktrans (/min)", None, "plasma"))

    # keep only panels that actually have data
    panels = [p for p in panels
              if any((_ktrans(r, prefix) if p[0] == "__ktrans__"
                      else _get(r, p[0], prefix)) is not None for r in rows)]
    if not panels:
        print("  plot_param_maps: no populated fields")
        return None

    if z is None:
        z = pick_slice(rows, panels[0][0] if panels[0][0] != "__ktrans__" else "ve",
                       prefix)
    if z is None:
        return None
    anat = ctx.anatomy(z)

    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, scale, cmap_name) in zip(axes, panels):
        if key == "__ktrans__":
            sl = build_map(rows, None, ctx.shape3d,
                           transform=lambda r: _ktrans(r, prefix))[:, :, z]
        else:
            sl = build_map(rows, key, ctx.shape3d, prefix, scale)[:, :, z]
        im, n = _overlay(ax, anat, sl, cmap_name)
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\n({n} voxels)", fontsize=10)
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"Parameter maps{' -- ' + label if label else ''}"
                 f"  {ctx.subject_id}, slice z={z}\n"
                 f"overlay opaque where fitted; background transparent",
                 fontsize=12)
    fig.tight_layout()
    return savefig(fig, name or f"param_maps_{prefix or 'fit'}",
                   ctx.results_subdir)


# =============================================================================
# Figure: railed vs clean
# =============================================================================

def plot_rail_map(rows, ctx: FigureContext, prefix=None, z=None,
                  name=None, label=""):
    """Binary map: orange = railed, cyan = clean, transparent = not fitted."""
    have = [r for r in rows if _get(r, "railed", prefix) is not None]
    if not have or ctx.shape3d is None:
        print("  plot_rail_map: no rail status available")
        return None
    if z is None:
        z = pick_slice(have, "railed", prefix)
    anat = ctx.anatomy(z)

    sl = build_map(have, None, ctx.shape3d,
                   transform=lambda r: 1.0 if _get(r, "railed", prefix) else 0.0)[:, :, z]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    _overlay(ax, anat, sl, ["cyan", "orange"], vmin=0, vmax=1, discrete=True)

    n = int(np.sum(np.isfinite(sl)))
    n_rail = int(np.nansum(sl == 1.0))
    ax.set_title(f"Railed (orange) vs clean (cyan){' -- ' + label if label else ''}\n"
                 f"{ctx.subject_id}, slice z={z}: {n_rail}/{n} railed "
                 f"({n_rail/max(n,1):.0%})", fontsize=11)
    ax.legend(handles=[Patch(facecolor="cyan", label="clean"),
                       Patch(facecolor="orange", label="railed")],
              loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return savefig(fig, name or f"rail_map_{prefix or 'fit'}", ctx.results_subdir)


def plot_rail_map_comparison(rows, ctx: FigureContext, prefixes=("nod", "dly"),
                             labels=("no delay", "delay-corrected"), z=None):
    """Two rail maps side by side, same slice, same voxels."""
    paired = [r for r in rows
              if all(_get(r, "railed", p) is not None for p in prefixes)]
    if not paired or ctx.shape3d is None:
        print("  plot_rail_map_comparison: no paired rail status")
        return None
    if z is None:
        z = pick_slice(paired, "railed", prefixes[0])
    anat = ctx.anatomy(z)

    fig, axes = plt.subplots(1, len(prefixes), figsize=(6.5*len(prefixes), 6))
    axes = np.atleast_1d(axes)
    for ax, pfx, lab in zip(axes, prefixes, labels):
        sl = build_map(paired, None, ctx.shape3d,
                       transform=lambda r, p=pfx: 1.0 if _get(r, "railed", p) else 0.0)[:, :, z]
        _overlay(ax, anat, sl, ["cyan", "orange"], vmin=0, vmax=1, discrete=True)
        n = int(np.sum(np.isfinite(sl))); nr = int(np.nansum(sl == 1.0))
        ax.set_title(f"{lab}\n{nr}/{n} railed ({nr/max(n,1):.0%})", fontsize=11)
    axes[-1].legend(handles=[Patch(facecolor="cyan", label="clean"),
                             Patch(facecolor="orange", label="railed")],
                    loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Where do fits rail?  {ctx.subject_id}, slice z={z}, "
                 f"{len(paired)} voxels", fontsize=12)
    fig.tight_layout()
    return savefig(fig, "rail_map_comparison", ctx.results_subdir)


# =============================================================================
# Figure: parameter distributions
# =============================================================================

DEFAULT_DIST_SPECS = [
    ("ve",       "ve",             1.0,  False),
    ("vp",       "vp",             1.0,  False),
    ("fp",       "fp (/min)",      60.0, True),
    ("ps",       "ps (/min)",      60.0, True),
    ("tau_i",    "tau_i (s)",      1.0,  True),
    ("best_tau", "AIF delay (s)",  1.0,  False),
    ("cost",     "cost (RSS)",     1.0,  True),
]


def plot_distributions(rows, ctx: FigureContext, prefixes=(None,),
                       labels=("fit",), specs=None, include_ktrans=True,
                       name=None):
    """
    Histogram per parameter. Pass several prefixes to overlay variants
    (e.g. no-delay vs delay-corrected) on the same axes.
    """
    if not rows:
        return None
    specs = list(specs or DEFAULT_DIST_SPECS)
    if include_ktrans:
        specs.append(("__ktrans__", "Ktrans (/min)", 1.0, True))
    colors = ["steelblue", "darkorange", "seagreen", "purple"]

    populated = []
    for key, title, scale, logx in specs:
        vals = {}
        for pfx, lab in zip(prefixes, labels):
            if key == "__ktrans__":
                v = [_ktrans(r, pfx) for r in rows]
            else:
                v = [_get(r, key, pfx) for r in rows]
            v = np.array([x for x in v if x is not None and np.isfinite(x)])
            if len(v):
                vals[lab] = v * scale
        if vals:
            populated.append((key, title, logx, vals))
    if not populated:
        return None

    ncols = 3
    nrows = int(np.ceil(len(populated) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 3.8*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, title, logx, vals) in zip(axes, populated):
        allv = np.concatenate(list(vals.values()))
        if logx and allv.min() > 0:
            bins = np.logspace(np.log10(allv.min()*0.9),
                               np.log10(allv.max()*1.1), 25)
            ax.set_xscale("log")
        else:
            bins = np.linspace(allv.min(), allv.max(), 25)
        for (lab, v), col in zip(vals.items(), colors):
            ax.hist(v, bins=bins, alpha=0.6, color=col, edgecolor="k",
                    label=f"{lab} (n={len(v)})")
        if key == "tau_i" and ctx.tau_i_lit:
            ax.axvspan(*ctx.tau_i_lit, alpha=0.15, color="green",
                       label="literature range")
        ax.set_xlabel(title, fontsize=9)
        ax.set_ylabel("voxel count", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for ax in axes[len(populated):]:
        ax.axis("off")

    fig.suptitle(f"Fitted parameter distributions  {ctx.subject_id}",
                 fontsize=12)
    fig.tight_layout()
    return savefig(fig, name or "parameter_distributions", ctx.results_subdir)


# =============================================================================
# BOOTSTRAP FIGURES
# =============================================================================

def plot_bootstrap_ci(boot, ctx: FigureContext, param="tau_i", name=None,
                      label=""):
    """
    Single-curve bootstrap result: replicate histogram with the point
    estimate and percentile interval marked, plus a convergence trace.

    Respects boot.is_calibrated -- a frame_drop result is labelled a
    STABILITY INDEX, never a confidence interval.
    """
    if boot is None or len(boot.replicates) == 0:
        print("  plot_bootstrap_ci: no replicates")
        return None
    j = boot.param_names.index(param)
    v = boot.replicates[:, j]
    s = boot.summary(param)
    kind = "95% CI" if boot.is_calibrated else "central 95% (stability index)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.hist(v, bins=40, color="steelblue", edgecolor="k", alpha=0.85)
    ax.axvline(s["point"], color="crimson", lw=2.5,
               label=f"point estimate = {s['point']:.4f}")
    ax.axvline(s["ci_lo"], color="k", ls="--", lw=1.5, label=kind)
    ax.axvline(s["ci_hi"], color="k", ls="--", lw=1.5)
    if param == "tau_i" and ctx.tau_i_lit:
        ax.axvspan(*ctx.tau_i_lit, alpha=0.15, color="green",
                   label="literature range")
    ax.set_xlabel(f"{param} (bootstrap replicates)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(f"{boot.method} bootstrap, n={len(v)}\n"
                 f"median={s['median']:.4f}, IQR={s['iqr']:.4f}", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ns, iqrs, widths = [], [], []
    for n in (25, 50, 100, 200, 500, 1000, 2000):
        if n > len(v):
            break
        sub = v[:n]
        ns.append(n)
        iqrs.append(np.percentile(sub, 75) - np.percentile(sub, 25))
        widths.append(np.percentile(sub, 97.5) - np.percentile(sub, 2.5))
    if ns:
        ax.plot(ns, iqrs, "o-", color="steelblue", label="IQR")
        ax.plot(ns, widths, "s--", color="darkorange", label="95% width")
        ax.set_xscale("log")
        ax.set_xlabel("replicates used", fontsize=9)
        ax.set_ylabel(f"{param} interval width", fontsize=9)
        ax.set_title("convergence: has the interval stopped moving?\n"
                     "(flat = converged; still falling = need more)",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    warn = "" if boot.is_calibrated else \
        "\nSTABILITY INDEX (frame drop / subsampling) -- NOT a calibrated CI"
    fig.suptitle(f"Bootstrap uncertainty on {param}"
                 f"{' -- ' + label if label else ''}  {ctx.subject_id}{warn}",
                 fontsize=12)
    fig.tight_layout()
    return savefig(fig, name or f"bootstrap_ci_{param}", ctx.results_subdir)


def plot_uncertainty_maps(slice_result, ctx: FigureContext, param="tau_i",
                          name=None):
    """
    Spatial uncertainty from a tier-2 single-slice bootstrap.

    Four panels: point estimate, interval width, relative width
    (width/median -- the practically useful one), and which voxels have
    an interval tight enough to be worth reporting.
    """
    vox = [v for v in slice_result["voxels"] if v.get("boot") is not None
           and len(v["boot"].replicates) > 0]
    if not vox or ctx.shape3d is None:
        print("  plot_uncertainty_maps: no bootstrap voxels")
        return None
    z = slice_result["z"]
    anat = ctx.anatomy(z)
    calibrated = all(v["boot"].is_calibrated for v in vox)

    def _mk(fn):
        m = np.full(ctx.shape3d, np.nan)
        for v in vox:
            try:
                m[v["coord"]] = fn(v["boot"])
            except Exception:
                pass
        return m[:, :, z]

    med = _mk(lambda b: b.percentile(50, param))
    width = _mk(lambda b: b.percentile(97.5, param) - b.percentile(2.5, param))
    rel = _mk(lambda b: (b.percentile(97.5, param) - b.percentile(2.5, param))
              / max(b.percentile(50, param), 1e-12))
    usable = _mk(lambda b: 1.0 if ((b.percentile(97.5, param)
                                    - b.percentile(2.5, param))
                                   / max(b.percentile(50, param), 1e-12)) < 1.0
                 else 0.0)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, sl, title, cmap_name, discrete in [
            (axes[0], med,   f"{param} (bootstrap median)", "magma", False),
            (axes[1], width, f"{param} interval width",     "inferno", False),
            (axes[2], rel,   f"relative width\n(width / median)", "coolwarm", False),
            (axes[3], usable, "interval width < 100%\nof the estimate",
             ["orange", "cyan"], True)]:
        im, n = _overlay(ax, anat, sl, cmap_name,
                         vmin=0 if discrete else None,
                         vmax=1 if discrete else None, discrete=discrete)
        if im is not None and not discrete:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)
        ax.set_title(f"{title}\n({n} voxels)", fontsize=10)
    axes[3].legend(handles=[Patch(facecolor="cyan", label="usable"),
                            Patch(facecolor="orange", label="too uncertain")],
                   loc="upper right", fontsize=8, framealpha=0.9)

    kind = ("bootstrap 95% CI" if calibrated
            else "STABILITY INDEX -- not a calibrated CI")
    fig.suptitle(f"Spatial uncertainty in {param} ({kind})  "
                 f"{ctx.subject_id}, slice z={z}", fontsize=12)
    fig.tight_layout()
    return savefig(fig, name or f"uncertainty_maps_{param}", ctx.results_subdir)


def plot_ci_vs_enhancement(slice_result, ctx: FigureContext, param="tau_i",
                           name=None):
    """
    Does uncertainty track data quality? Interval width vs enhancement
    ratio, coloured by whether the voxel passed the enhancement screen.
    Directly tests the expectation that weakly-enhancing voxels give
    wider intervals regardless of noise.
    """
    vox = [v for v in slice_result["voxels"] if v.get("boot") is not None
           and len(v["boot"].replicates) > 0]
    if not vox:
        return None
    enh = np.array([v["enh"] for v in vox])
    band = np.array([v.get("in_band", True) for v in vox])
    med = np.array([v["boot"].percentile(50, param) for v in vox])
    wid = np.array([v["boot"].percentile(97.5, param)
                    - v["boot"].percentile(2.5, param) for v in vox])
    rel = wid / np.maximum(med, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, y, ylab, logy in [(axes[0], wid, f"{param} interval width", True),
                              (axes[1], rel, "relative width (width/median)", True)]:
        ax.scatter(enh[band], y[band], c="steelblue", edgecolor="k", s=45,
                   label="in enhancement band")
        ax.scatter(enh[~band], y[~band], c="crimson", edgecolor="k", s=45,
                   marker="^", label="outside band")
        if logy and np.all(y > 0):
            ax.set_yscale("log")
        ax.set_xlabel("enhancement ratio", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].axhline(1.0, color="r", ls="--", lw=1.2,
                    label="width = estimate")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Does {param} uncertainty track data quality?  "
                 f"{ctx.subject_id}, slice z={slice_result['z']}", fontsize=12)
    fig.tight_layout()
    return savefig(fig, name or f"ci_vs_enhancement_{param}",
                   ctx.results_subdir)


def plot_roi_vs_voxel_ci(roi_boot, slice_result, ctx: FigureContext,
                         param="tau_i", name=None):
    """
    Tier 1 vs tier 2: the ROI-averaged interval against the distribution
    of per-voxel intervals. Shows directly how much precision ROI
    pooling buys.
    """
    if roi_boot is None or len(roi_boot.replicates) == 0:
        return None
    vox = [v for v in slice_result["voxels"] if v.get("boot") is not None
           and len(v["boot"].replicates) > 0] if slice_result else []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    s = roi_boot.summary(param)

    ax = axes[0]
    ax.hist(roi_boot.replicates[:, roi_boot.param_names.index(param)],
            bins=40, color="crimson", alpha=0.7, edgecolor="k",
            label=f"ROI ({roi_boot.meta.get('n_voxels','?')} voxels pooled)")
    if vox:
        meds = [v["boot"].percentile(50, param) for v in vox]
        ax.hist(meds, bins=25, color="steelblue", alpha=0.55, edgecolor="k",
                label=f"per-voxel medians (n={len(meds)})")
    ax.set_xscale("log")
    ax.set_xlabel(param, fontsize=9); ax.set_ylabel("count", fontsize=9)
    ax.set_title("ROI bootstrap vs per-voxel estimates", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    roi_w = s["ci_hi"] - s["ci_lo"]
    if vox:
        vw = np.array([v["boot"].percentile(97.5, param)
                       - v["boot"].percentile(2.5, param) for v in vox])
        ax.hist(vw, bins=25, color="steelblue", edgecolor="k", alpha=0.85,
                label=f"per-voxel widths (median {np.median(vw):.3f})")
    ax.axvline(roi_w, color="crimson", lw=2.5,
               label=f"ROI width = {roi_w:.4f}")
    ax.set_xlabel(f"{param} 95% interval width", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title("how much precision does ROI pooling buy?", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"Tier 1 (ROI) vs tier 2 (per-voxel) uncertainty on {param}"
                 f"  {ctx.subject_id}", fontsize=12)
    fig.tight_layout()
    return savefig(fig, name or f"roi_vs_voxel_ci_{param}", ctx.results_subdir)


# =============================================================================
# Registry / driver
# =============================================================================

FIGURE_REGISTRY = {
    "param_maps":        lambda rows, ctx, **kw: plot_param_maps(rows, ctx, **kw),
    "rail_map":          lambda rows, ctx, **kw: plot_rail_map(rows, ctx, **kw),
    "rail_comparison":   lambda rows, ctx, **kw: plot_rail_map_comparison(rows, ctx, **kw),
    "distributions":     lambda rows, ctx, **kw: plot_distributions(rows, ctx, **kw),
}

BOOTSTRAP_FIGURE_REGISTRY = {
    "bootstrap_ci":       plot_bootstrap_ci,
    "uncertainty_maps":   plot_uncertainty_maps,
    "ci_vs_enhancement":  plot_ci_vs_enhancement,
    "roi_vs_voxel_ci":    plot_roi_vs_voxel_ci,
}


def make_figures(rows, ctx: FigureContext, which=None, **kwargs):
    """
    Drive the row-based figures by name.

        make_figures(rows, ctx, which=["param_maps", "rail_map"], prefix="dly")

    Unknown names raise, so a typo fails loudly rather than silently
    producing nothing. Per-figure kwargs can be passed as
    which={"param_maps": {"prefix": "dly"}, "rail_map": {}}.
    """
    if which is None:
        which = list(FIGURE_REGISTRY)
    paths = {}
    if isinstance(which, dict):
        items = which.items()
    else:
        items = [(n, {}) for n in which]
    for nm, kw in items:
        if nm not in FIGURE_REGISTRY:
            raise KeyError(f"unknown figure {nm!r}; "
                           f"available: {sorted(FIGURE_REGISTRY)}")
        merged = {**kwargs, **kw}
        try:
            paths[nm] = FIGURE_REGISTRY[nm](rows, ctx, **merged)
        except TypeError as e:
            raise TypeError(f"figure {nm!r} rejected an argument: {e}") from e
    return paths