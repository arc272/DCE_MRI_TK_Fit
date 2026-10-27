
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
bootstrap_wx.py
===============
Resampling-based uncertainty quantification for water-exchange (tau_i)
DCE-MRI fits.

NOTE ON PLACEMENT
-----------------
The package already contains a `bootstrap.py` (BootstrapConfig /
BootstrapResult exist in config.py). This module is written as a
SEPARATE file rather than edits to that one, because its current
contents weren't available when this was written. Before adopting,
check for overlap -- if bootstrap.py already implements case
resampling or a replicate driver, the two should be merged rather
than left as parallel implementations.

METHODS IMPLEMENTED
-------------------
Two methods proposed by the PI, plus one recommended upgrade:

  1. frame_drop   -- randomly drop a subset of the 36 frames and refit.
                     Formally subsampling / delete-d jackknife (Wu 1986;
                     Politis & Romano 1994), NOT the classical bootstrap.
                     Best read as a STABILITY INDEX, not a calibrated CI
                     (Politis, Romano & Wolf 1999; cf. the "stability
                     test" recommended by Grech-Sollars et al. 2019 for
                     shutter-speed tau_i specifically).

  2. residual     -- resample residuals with replacement, add back to the
                     best-fit curve, refit. The classical residual
                     bootstrap for nonlinear regression (Efron &
                     Tibshirani 1993), appropriate here because the
                     sampling times are fixed and non-random.

  3. wild         -- keep each residual at its own timepoint but multiply
                     by a random Rademacher (+/-1) weight. Generalises
                     method 2 to heteroscedastic noise (Wu 1986; Liu
                     1988; Mammen 1993). Applied to biophysical MRI
                     signal-model fitting by Gu et al. (2019) for exactly
                     this reason. Use when residual variance tracks the
                     bolus rather than being flat.

  4. block        -- resample contiguous blocks of residuals rather than
                     single frames, preserving short-range serial
                     correlation (Kuensch 1989). Only needed if the
                     residual ACF shows real autocorrelation -- check
                     with diagnose_residuals() first rather than
                     assuming.

WHY frame_drop NEEDS ITS OWN RESIDUAL FUNCTION
-----------------------------------------------
The obvious implementation -- shorten time_var/aif/sig together and call
fit_single_voxel -- is WRONG for this pipeline. tcxm_ct_compartments
builds its internal convolution grid from whatever time_var it receives
and interpolates the AIF onto it:

    t_fine  = np.linspace(t[0], t[-1], (len(t)-1)*upsample + 1)
    ca_fine = interp1d(t, ca, ...)(t_fine)

Handing it 28 points instead of 36 degrades the AIF itself, not just the
tissue observation. That conflates two different things: information
loss in the curve being fitted (intended) vs information loss in the
model's own input (not intended, and not covered by the subsampling
consistency results the method rests on).

The correct implementation keeps time_var and aif at full resolution,
computes the full 36-point prediction, and masks the RESIDUAL to the
retained frames. Least squares only minimises the residual vector it is
handed, so this realises "sum over whichever timepoints are present"
without touching AIF fidelity. It costs the same per evaluation as a
normal fit -- masking buys the intended statistical experiment, not
speed.

TIERED USAGE
------------
A full wx fit measures ~5-11 s, and the literature recommends 500-2000
replicates for stable percentile intervals. That makes uniform
voxelwise bootstrap infeasible (454 voxels x 500 reps ~ 500 hr serial).
Use the tiers:

  bootstrap_roi(...)          tier 1: ROI mean curve, full replicates.
                              ~4 min at 500 reps on 16 cores. This is
                              the number to report.
  bootstrap_slice(...)        tier 2: every voxel in ONE lesion slice,
                              full replicates -> a spatial uncertainty
                              map without paying for the whole volume.
                              ~3-6 hr at 500 reps on 16 cores.
  stability_index_volume(...) tier 3: frame_drop at low replicate count
                              over all voxels. Explicitly a stability
                              index, labelled as such in figures --
                              never presented as a calibrated CI.
"""



import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from dce_mri.kinetic_models import KineticModelSpec
from dce_mri.fitting import (fit_single_voxel, _lhs_starts, _trf_refine,
                             _top_finite_indices, _validate_wx_compatibility)
from dce_mri.signal_models import compute_S0


# =============================================================================
# Config / result containers
# =============================================================================

@dataclass(frozen=True)
class BootstrapWxConfig:
    """
    method        : 'frame_drop' | 'residual' | 'wild' | 'block'
    n_replicates  : number of resampling replicates. Efron & Tibshirani
                    recommend 500-2000 for percentile intervals; 100 is
                    documented as too few for stable IQR estimation.
    drop_frac     : frame_drop only -- fraction of frames removed per
                    replicate. m/n retention matters for subsampling
                    consistency (Politis/Romano/Wolf); 0.2 keeps 80%.
    block_len     : block only -- length of contiguous residual blocks.
    warm_start    : if True, each replicate is a single TRF refinement
                    from the original best fit (~10x faster). If False,
                    each replicate runs the full LHS+TRF global search.
                    False is the honest default: warm-starting can
                    understate spread by keeping replicates in the
                    original basin, which matters here because tau_i
                    rails frequently. Use compare_warm_start() to
                    measure the difference on your own data rather than
                    assuming it is negligible.
    n_workers     : processes for multiprocessing. None -> os.cpu_count().
    seed          : base seed; each replicate gets an independent child
                    stream via SeedSequence.
    """
    method:       str   = "residual"
    n_replicates: int   = 500
    drop_frac:    float = 0.2
    block_len:    int   = 4
    warm_start:   bool  = False
    n_workers:    Optional[int] = None
    seed:         int   = 0


@dataclass
class BootstrapWxResult:
    """
    param_names   : names for each column of `replicates`
    replicates    : (n_ok, n_params) recovered parameters per replicate
    point_estimate: the original (non-resampled) fit's parameters
    method        : which resampling method produced this
    n_requested   : replicates requested
    n_failed      : replicates that failed to fit
    is_calibrated : False for frame_drop -- a stability index, NOT a
                    calibrated confidence interval. Figures must respect
                    this flag when labelling.
    elapsed_s     : wall time
    """
    param_names:    tuple
    replicates:     np.ndarray
    point_estimate: np.ndarray
    method:         str
    n_requested:    int
    n_failed:       int
    is_calibrated:  bool
    elapsed_s:      float = 0.0
    meta:           dict = field(default_factory=dict)

    def percentile(self, q, param=None):
        """Percentile(s) across replicates. `param` may be a name or index."""
        if len(self.replicates) == 0:
            return np.nan
        if param is None:
            return np.percentile(self.replicates, q, axis=0)
        j = (self.param_names.index(param) if isinstance(param, str) else param)
        return np.percentile(self.replicates[:, j], q)

    def ci(self, param, lo=2.5, hi=97.5):
        return (self.percentile(lo, param), self.percentile(hi, param))

    def iqr(self, param):
        return self.percentile(75, param) - self.percentile(25, param)

    def summary(self, param):
        med = self.percentile(50, param)
        lo, hi = self.ci(param)
        j = (self.param_names.index(param) if isinstance(param, str) else param)
        return dict(point=float(self.point_estimate[j]), median=float(med),
                    ci_lo=float(lo), ci_hi=float(hi),
                    iqr=float(self.iqr(param)),
                    n_ok=len(self.replicates), n_failed=self.n_failed,
                    is_calibrated=self.is_calibrated)


# =============================================================================
# Residual diagnostics -- decide WHICH method is appropriate
# =============================================================================

def diagnose_residuals(residuals, fitted=None, verbose=True):
    """
    Checks the two assumptions that decide between residual / wild /
    block bootstrap. Run this before committing to a method rather than
    assuming -- the literature review flags both as empirical questions.

    Returns dict with:
      het_ratio  : var(residuals in top-third fitted values) /
                   var(residuals in bottom-third). ~1 means homoscedastic;
                   >>1 means residual size tracks signal level, which is
                   the documented DCE case (SNR tracks the bolus, Rician
                   floors are signal-dependent) -> prefer WILD.
      acf1       : lag-1 autocorrelation of the residuals. Near 0 means
                   frame-to-frame independence; substantially nonzero
                   suggests serial correlation (motion, drift) -> prefer
                   BLOCK.
      recommend  : suggested method string.
    """
    r = np.asarray(residuals, float)
    n = len(r)

    if fitted is not None and n >= 9:
        f = np.asarray(fitted, float)
        order = np.argsort(f)
        third = n // 3
        lo_var = np.var(r[order[:third]], ddof=1)
        hi_var = np.var(r[order[-third:]], ddof=1)
        het_ratio = float(hi_var / lo_var) if lo_var > 0 else np.inf
    else:
        het_ratio = np.nan

    rc = r - r.mean()
    denom = np.sum(rc * rc)
    acf1 = float(np.sum(rc[:-1] * rc[1:]) / denom) if denom > 0 else 0.0

    if np.isfinite(het_ratio) and het_ratio > 3.0:
        rec = "wild"
        why = f"residual variance {het_ratio:.1f}x larger at high signal"
    elif abs(acf1) > 0.3:
        rec = "block"
        why = f"lag-1 autocorrelation {acf1:+.2f} suggests serial correlation"
    else:
        rec = "residual"
        why = "residuals look roughly i.i.d."

    if verbose:
        print(f"    residual diagnostics: het_ratio={het_ratio:.2f}  "
              f"acf1={acf1:+.3f}  -> recommend '{rec}' ({why})")
        if np.isfinite(het_ratio) and 1.5 < het_ratio <= 3.0:
            print(f"      note: het_ratio in the ambiguous 1.5-3 range; "
                  f"wild bootstrap is the safer choice and costs nothing extra")

    return dict(het_ratio=het_ratio, acf1=acf1, recommend=rec, why=why)


# =============================================================================
# Replicate generation -- the four methods
# =============================================================================

def _make_replicate_data(method, sig, S_pred, resid, rng, cfg):
    """
    Returns (obs, keep_mask). For every method except frame_drop,
    keep_mask is None (all frames used) and `obs` is a synthetic
    observation. For frame_drop, `obs` is the ORIGINAL signal and
    keep_mask selects which residual entries count.
    """
    T = len(sig)

    if method == "frame_drop":
        n_drop = max(1, int(round(cfg.drop_frac * T)))
        drop_idx = rng.choice(T, size=n_drop, replace=False)
        keep = np.ones(T, dtype=bool)
        keep[drop_idx] = False
        return sig, keep                     # original data, masked objective

    if method == "residual":
        # classical nonparametric residual bootstrap: resample residuals
        # WITH replacement across timepoints, add to the fitted curve
        idx = rng.integers(0, T, size=T)
        return S_pred + resid[idx], None

    if method == "wild":
        # Rademacher weights: each residual stays at its own timepoint,
        # sign flipped at random. Preserves point-by-point noise
        # magnitude -> valid under heteroscedasticity.
        w = rng.choice([-1.0, 1.0], size=T)
        return S_pred + resid * w, None

    if method == "block":
        # Kuensch moving-block: resample contiguous runs of residuals
        L = max(1, int(cfg.block_len))
        n_blocks = int(np.ceil(T / L))
        starts = rng.integers(0, max(1, T - L + 1), size=n_blocks)
        boot = np.concatenate([resid[s:s+L] for s in starts])[:T]
        if len(boot) < T:                     # pad if the tail came up short
            boot = np.concatenate([boot, resid[:T-len(boot)]])
        return S_pred + boot, None

    raise ValueError(f"unknown bootstrap method {method!r}")


# =============================================================================
# Fitting one replicate
# =============================================================================

def _fit_replicate(args):
    """
    Module-level so it is picklable for multiprocessing.Pool.
    Returns the augmented parameter vector (kinetic + wx) or None.
    """
    (obs, keep_mask, aif, time_var, model, acq, subject, cfg_fit,
     wx_model, aif_cfg, relax_cfg, p0_warm, warm_start, lhs_seed) = args

    n_kin = len(model.p0)
    S0 = float(compute_S0(obs[None, :], acq)[0])

    if relax_cfg.fit_tau_b:
        lb = np.concatenate([model.lb, [relax_cfg.tau_b_lb, relax_cfg.tau_i_lb]])
        ub = np.concatenate([model.ub, [relax_cfg.tau_b_ub, relax_cfg.tau_i_ub]])
        log_dims = tuple(model.log_dims or ()) + (n_kin + 1,)
    else:
        lb = np.concatenate([model.lb, [relax_cfg.tau_i_lb]])
        ub = np.concatenate([model.ub, [relax_cfg.tau_i_ub]])
        log_dims = tuple(model.log_dims or ()) + (n_kin,)

    ve_idx = model.param_names.index("ve")
    vp_idx = model.param_names.index("vp")
    n_resid = int(keep_mask.sum()) if keep_mask is not None else len(obs)

    def residual_fn(p_aug):
        theta_kin = p_aug[:n_kin]
        if relax_cfg.fit_tau_b:
            tau_b, tau_i = p_aug[n_kin], p_aug[n_kin + 1]
        else:
            tau_b, tau_i = relax_cfg.tau_b_fixed, p_aug[n_kin]
        if model.constraints is not None and not model.constraints(theta_kin):
            return np.full(n_resid, 1e6)
        try:
            # NOTE: full time_var / full aif ALWAYS -- the forward model and
            # its AIF interpolation grid are never degraded. Only the
            # residual is masked. See module docstring.
            Cv, Ce = model.fn_compartments(theta_kin, time_var, aif)
            S = wx_model.to_signal(Cv, Ce, theta_kin[ve_idx], theta_kin[vp_idx],
                                   (tau_b, tau_i), acq, subject, aif_cfg,
                                   relax_cfg, S0)
        except Exception:
            return np.full(n_resid, 1e6)
        if not np.all(np.isfinite(S)):
            return np.full(n_resid, 1e6)
        r = S - obs
        return r[keep_mask] if keep_mask is not None else r

    try:
        if warm_start:
            p0 = np.clip(p0_warm, lb, ub)
            res = least_squares(residual_fn, p0, bounds=(lb, ub), method="trf",
                                jac="2-point", xtol=cfg_fit.xtol,
                                ftol=cfg_fit.ftol, gtol=cfg_fit.gtol,
                                max_nfev=cfg_fit.max_nfev)
            return res.x
        # full global search -- same LHS+TRF structure as the real fitter
        samples = _lhs_starts(cfg_fit.n_lhs, lb, ub, model.constraints,
                              log_dims, lhs_seed)
        costs = np.array([np.sum(residual_fn(p) ** 2) for p in samples])
        top = _top_finite_indices(costs, cfg_fit.n_top)
        if len(top) == 0:
            return None
        best, best_cost = None, np.inf
        for i in top:
            params, cost, _ = _trf_refine(samples[i], residual_fn, lb, ub, cfg_fit)
            if params is not None and cost < best_cost:
                best, best_cost = params, cost
        return best
    except Exception:
        return None


# =============================================================================
# Core driver
# =============================================================================

def bootstrap_curve(sig, aif, time_var, model, acq, subject, cfg_fit,
                    wx_model, aif_cfg, relax_cfg,
                    boot_cfg: BootstrapWxConfig,
                    point_fit=None, verbose=True):
    """
    Bootstrap ONE signal curve (a voxel, or an ROI mean curve).

    Runs the point fit first if not supplied, derives residuals from it,
    then generates and fits n_replicates resampled datasets in parallel.
    """
    _validate_wx_compatibility(wx_model, cfg_fit.fit_space, model, time_var,
                               aif, aif_cfg, relax_cfg)
    if wx_model is not None and wx_model.n_params == 0:
        raise NotImplementedError(
            f"bootstrap_wx is built around free tau parameters "
            f"(_fit_replicate always appends tau_i); wx_model="
            f"{wx_model.name!r} has none. Use bootstrap.py-style kinetic-only "
            f"replicates, or extend _fit_replicate's layout first.")

    if point_fit is None:
        point_fit = fit_single_voxel(sig, aif, time_var, model, acq, subject,
                                     cfg_fit, seed=boot_cfg.seed,
                                     wx_model=wx_model, aif_cfg=aif_cfg,
                                     relax_cfg=relax_cfg)
    if not point_fit.success or point_fit.wx_params is None:
        raise RuntimeError("point fit failed -- cannot bootstrap around it")

    p_point = np.concatenate([point_fit.params, point_fit.wx_params])
    S_pred = point_fit.S_pred
    resid = sig - S_pred

    n_kin = len(model.p0)
    wx_names = (("tau_b", "tau_i") if relax_cfg.fit_tau_b else ("tau_i",))
    param_names = tuple(model.param_names) + wx_names

    ss = np.random.SeedSequence(boot_cfg.seed)
    children = ss.spawn(boot_cfg.n_replicates)

    tasks = []
    for k in range(boot_cfg.n_replicates):
        rng = np.random.default_rng(children[k])
        obs, keep = _make_replicate_data(boot_cfg.method, sig, S_pred,
                                         resid, rng, boot_cfg)
        lhs_seed = int(rng.integers(0, 2**31 - 1))
        tasks.append((obs, keep, aif, time_var, model, acq, subject, cfg_fit,
                      wx_model, aif_cfg, relax_cfg, p_point,
                      boot_cfg.warm_start, lhs_seed))

    t0 = time.perf_counter()
    n_workers = boot_cfg.n_workers or os.cpu_count() or 1
    if verbose:
        print(f"    bootstrap: method={boot_cfg.method}, "
              f"n={boot_cfg.n_replicates}, warm_start={boot_cfg.warm_start}, "
              f"{n_workers} workers")

    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            out = pool.map(_fit_replicate, tasks)
    else:
        out = [_fit_replicate(a) for a in tasks]

    ok = [o for o in out if o is not None]
    reps = np.array(ok) if ok else np.empty((0, len(param_names)))
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"    -> {len(ok)}/{boot_cfg.n_replicates} replicates ok "
              f"in {elapsed/60:.1f} min "
              f"({elapsed/max(boot_cfg.n_replicates,1):.2f} s/replicate)")

    return BootstrapWxResult(
        param_names=param_names, replicates=reps, point_estimate=p_point,
        method=boot_cfg.method, n_requested=boot_cfg.n_replicates,
        n_failed=boot_cfg.n_replicates - len(ok),
        # frame_drop is subsampling, NOT a calibrated bootstrap CI
        is_calibrated=(boot_cfg.method != "frame_drop"),
        elapsed_s=elapsed,
        meta=dict(n_kin=n_kin, drop_frac=boot_cfg.drop_frac,
                  block_len=boot_cfg.block_len),
    )


# =============================================================================
# Convergence check -- is n_replicates enough?
# =============================================================================

def convergence_check(result: BootstrapWxResult, param="tau_i",
                      checkpoints=(50, 100, 200, 500, 1000), verbose=True):
    """
    Recomputes the IQR using only the first N replicates, for increasing
    N, to see whether the interval has stopped moving. Efron & Tibshirani
    note 100 replicates is too few for stable percentile estimation;
    this checks that empirically for YOUR data rather than assuming.
    """
    j = (result.param_names.index(param) if isinstance(param, str) else param)
    vals = result.replicates[:, j]
    out = []
    for n in checkpoints:
        if n > len(vals):
            break
        v = vals[:n]
        out.append(dict(n=n,
                        median=float(np.percentile(v, 50)),
                        iqr=float(np.percentile(v, 75) - np.percentile(v, 25)),
                        ci_lo=float(np.percentile(v, 2.5)),
                        ci_hi=float(np.percentile(v, 97.5))))
    if verbose and out:
        print(f"    convergence of {param}:")
        print(f"      {'n':>6} {'median':>10} {'IQR':>10} {'95% CI width':>14}")
        for r in out:
            print(f"      {r['n']:>6} {r['median']:>10.4f} {r['iqr']:>10.4f} "
                  f"{r['ci_hi']-r['ci_lo']:>14.4f}")
        if len(out) >= 2:
            drift = abs(out[-1]["iqr"] - out[-2]["iqr"]) / max(out[-1]["iqr"], 1e-12)
            verdict = ("converged" if drift < 0.1 else
                       "STILL DRIFTING -- use more replicates")
            print(f"      IQR change over the last doubling: {drift:.1%} "
                  f"-> {verdict}")
    return out


# =============================================================================
# Warm-start comparison -- measure the shortcut's cost in accuracy
# =============================================================================

def compare_warm_start(sig, aif, time_var, model, acq, subject, cfg_fit,
                       wx_model, aif_cfg, relax_cfg,
                       boot_cfg: BootstrapWxConfig, n_small=60, param="tau_i"):
    """
    Runs a small bootstrap BOTH ways (warm-started and full restart) and
    reports how much the interval differs. Warm-starting is ~10x faster
    but can understate spread by keeping replicates in the original
    basin -- which matters here because tau_i rails frequently. Run this
    once on a representative curve before trusting warm_start=True for a
    production run.
    """
    import dataclasses
    print("\n  warm-start vs full-restart comparison "
          f"({n_small} replicates each):")
    res = {}
    for warm in (False, True):
        c = dataclasses.replace(boot_cfg, n_replicates=n_small, warm_start=warm)
        r = bootstrap_curve(sig, aif, time_var, model, acq, subject, cfg_fit,
                            wx_model, aif_cfg, relax_cfg, c, verbose=False)
        res[warm] = r
        s = r.summary(param)
        print(f"    warm_start={str(warm):5s}: median={s['median']:.4f}  "
              f"IQR={s['iqr']:.4f}  95%CI=[{s['ci_lo']:.4f},{s['ci_hi']:.4f}]  "
              f"{r.elapsed_s:.0f}s")
    iqr_full, iqr_warm = res[False].iqr(param), res[True].iqr(param)
    ratio = iqr_warm / iqr_full if iqr_full > 0 else np.nan
    speedup = res[False].elapsed_s / max(res[True].elapsed_s, 1e-9)
    print(f"    warm-start IQR is {ratio:.2f}x the full-restart IQR "
          f"at {speedup:.1f}x the speed")
    if ratio < 0.8:
        print(f"    WARNING: warm-start understates spread by "
              f"{100*(1-ratio):.0f}% -- prefer warm_start=False")
    return res


# =============================================================================
# TIER 1 -- ROI mean curve
# =============================================================================

def bootstrap_roi(sig_4d, mask, aif, time_var, model, acq, subject,
                  cfg_fit, roi_cfg, wx_model, aif_cfg, relax_cfg,
                  boot_cfg: BootstrapWxConfig, verbose=True):
    """
    Tier 1: bootstrap the ROI-averaged curve. One curve, so the full
    500-2000 replicates are affordable (~4 min at 500 reps on 16 cores).
    This is the number to report.
    """
    from dce_mri.fitting_roi import _aggregate_roi_signal
    if verbose:
        print("\n--- TIER 1: ROI mean-curve bootstrap ---")
    Ct_mean, n_voxels, St_mean = _aggregate_roi_signal(sig_4d, mask, acq,
                                                       subject, roi_cfg)
    if verbose:
        print(f"    ROI: {n_voxels} voxels pooled")

    point = fit_single_voxel(St_mean, aif, time_var, model, acq, subject,
                             cfg_fit, seed=boot_cfg.seed, wx_model=wx_model,
                             aif_cfg=aif_cfg, relax_cfg=relax_cfg)
    diagnose_residuals(St_mean - point.S_pred, point.S_pred, verbose=verbose)

    res = bootstrap_curve(St_mean, aif, time_var, model, acq, subject,
                          cfg_fit, wx_model, aif_cfg, relax_cfg, boot_cfg,
                          point_fit=point, verbose=verbose)
    res.meta.update(n_voxels=n_voxels, St_mean=St_mean, S_pred=point.S_pred,
                    tier="roi")
    return res


# =============================================================================
# TIER 2 -- one lesion slice, every voxel
# =============================================================================

def bootstrap_slice(sig_4d, mask, aif, time_var, model, acq, subject,
                    cfg_fit, wx_model, aif_cfg, relax_cfg,
                    boot_cfg: BootstrapWxConfig, z=None,
                    enh_lo=1.3, enh_hi_frac=0.6, verbose=True):
    """
    Tier 2: bootstrap every voxel in ONE lesion slice, giving a spatial
    uncertainty map without paying for the whole volume. A parameter map
    doesn't need to be shown for every slice, so one representative
    slice is the efficient unit.

    z=None picks the slice containing the most lesion voxels.
    Voxels outside the enhancement band are still fitted but flagged, so
    the map shows the whole slice rather than a screened subset.
    """
    if verbose:
        print("\n--- TIER 2: single-slice voxelwise bootstrap ---")
    coords = np.argwhere(mask)
    if z is None:
        zs, counts = np.unique(coords[:, 2], return_counts=True)
        z = int(zs[np.argmax(counts)])
    sel = coords[coords[:, 2] == z]
    if verbose:
        print(f"    slice z={z}: {len(sel)} lesion voxels")

    TR_s, T10_s = acq.TR / 1000.0, subject.T10 / 1000.0
    FA = np.deg2rad(acq.FA * subject.B1)
    E10 = np.exp(-TR_s / T10_s)
    ceiling = (1 - E10 * np.cos(FA)) / (1 - E10)

    out = []
    t0 = time.perf_counter()
    for i, xyz in enumerate(sel):
        sig = sig_4d[tuple(xyz)]
        base = sig[acq.nbase].mean()
        enh = float(sig.max() / (base + 1e-11))
        in_band = bool(enh > enh_lo and enh < enh_hi_frac * ceiling)
        try:
            r = bootstrap_curve(sig, aif, time_var, model, acq, subject,
                                cfg_fit, wx_model, aif_cfg, relax_cfg,
                                boot_cfg, verbose=False)
            out.append(dict(coord=tuple(int(v) for v in xyz), enh=enh,
                            in_band=in_band, boot=r))
        except Exception:
            out.append(dict(coord=tuple(int(v) for v in xyz), enh=enh,
                            in_band=in_band, boot=None))
        if verbose and (i + 1) % 5 == 0:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(sel) - i - 1)
            print(f"    {i+1}/{len(sel)} voxels  ({el/60:.1f} min elapsed, "
                  f"~{eta/60:.0f} min remaining)")
    if verbose:
        ok = sum(1 for o in out if o["boot"] is not None)
        print(f"    slice bootstrap done: {ok}/{len(sel)} voxels, "
              f"{(time.perf_counter()-t0)/60:.1f} min total")
    return dict(z=z, voxels=out, ceiling=ceiling)


# =============================================================================
# TIER 3 -- cheap stability index over all voxels
# =============================================================================

def stability_index_volume(sig_4d, mask, aif, time_var, model, acq, subject,
                           cfg_fit, wx_model, aif_cfg, relax_cfg,
                           n_replicates=25, drop_frac=0.2, seed=0,
                           n_workers=None, verbose=True):
    """
    Tier 3: frame_drop at a LOW replicate count over every voxel.

    This is deliberately NOT a calibrated confidence interval -- the
    result is flagged is_calibrated=False and must be labelled as a
    stability index in any figure. It answers "how much does tau_i move
    when the curve is degraded like a real missed frame", which is the
    question the frame-drop method is actually suited to.
    """
    import dataclasses
    if verbose:
        print("\n--- TIER 3: whole-volume stability index (frame_drop) ---")
        print("    NOTE: stability index, NOT a calibrated CI")
    boot_cfg = BootstrapWxConfig(method="frame_drop", n_replicates=n_replicates,
                                 drop_frac=drop_frac, warm_start=True,
                                 n_workers=n_workers, seed=seed)
    coords = np.argwhere(mask)
    out = []
    t0 = time.perf_counter()
    for i, xyz in enumerate(coords):
        sig = sig_4d[tuple(xyz)]
        try:
            r = bootstrap_curve(sig, aif, time_var, model, acq, subject,
                                cfg_fit, wx_model, aif_cfg, relax_cfg,
                                boot_cfg, verbose=False)
            out.append(dict(coord=tuple(int(v) for v in xyz), boot=r))
        except Exception:
            out.append(dict(coord=tuple(int(v) for v in xyz), boot=None))
        if verbose and (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(coords) - i - 1)
            print(f"    {i+1}/{len(coords)} voxels ({el/60:.1f} min, "
                  f"~{eta/60:.0f} min left)")
    return out