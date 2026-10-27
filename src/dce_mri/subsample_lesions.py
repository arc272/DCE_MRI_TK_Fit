#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
subsample_lesions.py
====================
Size / SNR confound test. Large lesions (malignant by default) are refitted
on a RANDOM SUBSET of their voxels, sized like the benign lesions, with
exactly the model, prior, sampler and NLLS settings of fit_lesions_bayes.py
(imported from it -- one source of truth).

Logic: if a benign-vs-malignant difference is a size/SNR artifact (small
ROI -> noisier mean curve -> more prior-dominated / at-bound fits -> shrunk
toward the prior), the subsampled malignant estimates move toward the benign
ones and the AUC moves toward 0.5. A biological difference survives.

Random voxels, not a contiguous sub-ROI: a random subset keeps the lesion's
average tissue composition in expectation and changes only how many voxels
are averaged, so it isolates SNR. The reconstruction's regularisation makes
noise spatially correlated, so voxel count is not a perfect SNR proxy --
each subsampled curve's baseline CV is recorded, and group_stats_analysis.py
checks the SNR (not just the size) was matched to the benign group.

Target size (TARGET_MODE):
  "benign_rank"     DEFAULT. One benign size is drawn per lesion and the
                    draws are RANK-MATCHED to the lesions (largest draw to
                    the largest lesion), so the subsampled sizes follow the
                    benign distribution with as little truncation as
                    possible. The assignment is made once over the whole
                    malignant list, before sharding.
  "benign_sample_all"
                    per lesion and replicate, a size drawn at random from
                    ALL benign lesions; the lesion is fitted whole if the
                    draw is at or above its own size
  "benign_sample"   per lesion and replicate, a size drawn at random from
                    the benign lesions SMALLER than this lesion. NOTE: this
                    conditioning makes the subsampled malignant lesions
                    SYSTEMATICALLY SMALLER than the benign ones (a lesion at
                    the benign median can only draw from the lower half), so
                    it over-shrinks and is kept only for reproducing earlier
                    runs. Prefer "benign_rank".
  "benign_quantile" one size for all: the TARGET_QUANTILE of benign sizes
  "fixed"           TARGET_FIXED voxels for all
A lesion already at or below its target is fitted whole and recorded as
subsampled=False. Subsampling can only ever SHRINK a lesion, so malignant
lesions that are already benign-sized stay as they are and the match is
necessarily imperfect at the small end.

Whichever mode is used, the match is only checked afterwards, by
group_stats_analysis.py: n_vox_B / n_vox_Msub and snr_match_p_B_vs_Msub in
6_subsample_snr_check.csv. Matching voxel COUNT is not the same as matching
SNR -- the baseline CV is what decides that.

Outputs (RESULTS_ROOT/run_<timestamp>/ or RESUME_DIR):
  rows/{sid}_r{k}.csv          one row per lesion x replicate; same column
                               names as lesion_bayes_results.csv, plus the
                               subsampling bookkeeping
  {sid}_r{k}_draws.npz         chains (SAVE_DRAWS) -- for prior sensitivity
  subsample_results.csv        all rows (rebuilt at the end of every run)

CHANGING TARGET_MODE: start a FRESH run directory. A resumed run skips every
lesion that already has a rows/ file, so resuming after a mode change would
mix the two rules in one results file.

Run (resume / two GPUs exactly as with fit_lesions_bayes.py):
    PYTHONPATH=src python scripts/subsample_lesions.py
    CUDA_VISIBLE_DEVICES=0 RESUME_DIR=$RUN SHARD=0/2 python scripts/subsample_lesions.py
    CUDA_VISIBLE_DEVICES=1 RESUME_DIR=$RUN SHARD=1/2 python scripts/subsample_lesions.py
"""

import glob
import importlib.util
import os
import time
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# settings
# =============================================================================

FIT_SCRIPT   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fit_lesions_bayes.py")
BASE_RUN_DIR = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/run_20261006_185816_beta&logflat"
RESULTS_ROOT = "/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/run_20261006_185816_beta&logflat/lesion_subsample_results"
RESUME_DIR   = os.environ.get("RESUME_DIR") or None
SHARD        = os.environ.get("SHARD")          # "i/n"

SUBSAMPLE_DIAGNOSIS = "M"          # which group gets shrunk
BASE_SUBSET     = "all"            # lesions to take from the base run:
                                   # "all" (non-error) or "qc_pass". Note this
                                   # also sets which benign lesions form the
                                   # TARGET SIZE POOL -- use "qc_pass" to match
                                   # the benign group group_stats_analysis.py
                                   # compares against when its SUBSET is
                                   # "qc_pass".
TARGET_MODE     = "benign_rank"    # "benign_rank" | "benign_sample_all" |
                                   # "benign_sample" | "benign_quantile" | "fixed"
TARGET_QUANTILE = 0.5
TARGET_FIXED    = 50
MIN_VOXELS      = 5                # never subsample below this
N_REPLICATES    = 1                # independent random subsets per lesion
                                   # (3-5 averages out the draw-to-draw noise;
                                   # group_stats takes the median per lesion)
SEED            = 20261006
FITS            = ("bayes",)  # ("nlls",) for a fast first look
SAVE_DRAWS      = True
ONLY            = []               # e.g. ["FMB046"]
FIT_OVERRIDES   = {}               # setattr on the imported fit module, e.g.
                                   # {"N_DRAWS": 300} -- for TESTING only;
                                   # leave empty so subsampled fits use
                                   # exactly the base run's settings


# =============================================================================
# fit_lesions_bayes.py as a module
# =============================================================================

def load_fit_module(path=FIT_SCRIPT):
    spec = importlib.util.spec_from_file_location("fit_lesions_bayes", path)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    need = ["find_dce", "select_largest_lesion", "load_aif_csv", "summarise",
            "derived_draws",
            "PRIOR_CHOICES", "DELAY_BOUNDS", "N_DRAWS", "N_TUNE", "N_CHAINS",
            ]
    missing = [n for n in need if not hasattr(F, n)]
    if missing:
        raise AttributeError(f"{path} lacks {missing} ")
    return F


# =============================================================================
# subsampling
# =============================================================================

def lesion_rng(sid, rep, seed=SEED):
    """Deterministic per (lesion, replicate) -- independent of run order and
    sharding, so a resumed or split run draws the same subsets."""
    return np.random.default_rng([seed, zlib.crc32(sid.encode()), rep])


def assign_targets_rank(todo_sizes, benign_sizes, rep, seed=SEED,
                        min_voxels=None):
    """
    Rank-matched target sizes for a whole set of lesions at once.

    One benign size is drawn per lesion, the draws are sorted, and the largest
    draw goes to the largest lesion. Compared with drawing independently per
    lesion this (a) reproduces the benign size distribution with less sampling
    noise and (b) minimises how often the min() below binds, because the only
    lesions that cannot take their draw are the ones already smaller than it.

    Must be computed on the UNSHARDED lesion list: the assignment depends on
    the whole set, so two shards given different subsets would disagree. The
    seed does not involve the lesion id, so the same list + rep always gives
    the same assignment.

    Returns an int array of targets, aligned with todo_sizes.
    """
    min_voxels = MIN_VOXELS if min_voxels is None else min_voxels
    rng = np.random.default_rng([seed, 999, rep])
    n_full = np.asarray(todo_sizes, int)
    b = np.asarray(benign_sizes, int)
    if b.size == 0:
        raise ValueError("no benign lesions to draw target sizes from")
    drawn = np.sort(rng.choice(b, n_full.size, replace=True))
    k = np.empty(n_full.size, int)
    k[np.argsort(n_full, kind="stable")] = drawn   # smallest lesion <- smallest draw
    k = np.minimum(k, n_full)                      # subsampling can only shrink
    return np.maximum(k, np.minimum(min_voxels, n_full))


def choose_target(rng, n_full, benign_sizes, mode=None, quantile=None,
                  fixed=None, min_voxels=None):
    """Returns (k, note). k >= n_full means 'fit the whole lesion'.
    "benign_rank" is not handled here -- it is assigned for the whole cohort
    at once by assign_targets_rank()."""
    mode = mode or TARGET_MODE
    quantile = TARGET_QUANTILE if quantile is None else quantile
    fixed = TARGET_FIXED if fixed is None else fixed
    min_voxels = MIN_VOXELS if min_voxels is None else min_voxels
    b = np.asarray(benign_sizes, int)
    if mode == "benign_sample":
        # conditional draw: over-shrinks (see the module docstring)
        smaller = b[b < n_full]
        if smaller.size == 0:
            return n_full, "no benign lesion smaller"
        k = int(rng.choice(smaller))
    elif mode == "benign_sample_all":
        k = int(rng.choice(b))
    elif mode == "benign_quantile":
        k = int(round(np.quantile(b, quantile)))
    elif mode == "fixed":
        k = int(fixed)
    else:
        raise ValueError(f"unknown TARGET_MODE {mode!r}")
    k = max(k, min_voxels)
    return (n_full, "already benign-sized") if k >= n_full else (k, mode)


def subsample_mask(mask, k, rng):
    """Random k-voxel subset of a 3-D boolean mask (no replacement)."""
    idx = np.flatnonzero(mask)
    if not 0 < k <= idx.size:
        raise ValueError(f"k={k} must be in [1, {idx.size}]")
    pick = np.sort(rng.choice(idx, k, replace=False))
    sub = np.zeros(mask.size, bool)
    sub[pick] = True
    return sub.reshape(mask.shape), pick


def report_targets(sizes, targets, benign, label=""):
    """Print how the planned target sizes compare with the benign group."""
    k = np.asarray(targets, int)
    n = np.asarray(sizes, int)
    b = np.asarray(benign, int)
    whole = int((k >= n).sum())
    print(f"  planned sizes{label}: median {np.median(k):.0f} voxels "
          f"(benign {np.median(b):.0f}), IQR [{np.percentile(k, 25):.0f}, "
          f"{np.percentile(k, 75):.0f}] (benign [{np.percentile(b, 25):.0f}, "
          f"{np.percentile(b, 75):.0f}]); {whole}/{k.size} fitted whole")


# =============================================================================
# fits (mirror fit_lesions_bayes.fit_one -- keep in sync if fit_one changes)
# =============================================================================

def make_cfg_b(F):
    return F.BayesConfig(prior_choices=F.PRIOR_CHOICES, n_draws=F.N_DRAWS,
                         n_tune=F.N_TUNE, n_chains=F.N_CHAINS,
                         noise="gauss_marginalized_sigma", seed=F.SEED,
                         backend=F.BACKEND, step="nuts",
                         dense_mass_matrix=F.DENSE_MASS,
                         delay_mode="joint", delay_bounds=F.DELAY_BOUNDS)


def curve_qc(F, St):
    base = St[F.NBASE]
    return dict(baseline_cv=float(np.std(base) / np.mean(base)),
                baseline_sd=float(np.std(base, ddof=1)),
                peak_enhancement=float(St.max() / np.mean(base)))


def fit_bayes(F, St, aif, t, subject):
    cfg_b = make_cfg_b(F)
    t0 = time.time()
    res = F.fit_curve_bayes(St, aif, t, F.TCXM, F.acq, subject, F.cfg_fit, cfg_b)
    fit_s = time.time() - t0
    layout = F.build_param_layout(F.TCXM, cfg_b, aif=aif, time_var=t, acq=F.acq)
    noise = F.NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = F.assemble_prior(layout.model, None, delay_mode="joint",
                             n_noise=noise.n_params, noise_names=noise.param_names,
                             choices=F.PRIOR_CHOICES, delay_bounds=F.DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    ps_ = prior.sample(np.random.default_rng(F.PRIOR_SEED),
                       F.N_PRIOR_SAMPLES)[:, :layout.n_model]
    names_fit = list(layout.model.param_names)
    prior_d = {p: ps_[:, names_fit.index(p)] for p in F.PARAMS}
    prior_d.update(F.derived_draws(prior_d))
    summ, post = F.summarise(res, prior_d)

    theta_med = np.array([summ[f"{p}_median"] for p in names_fit])
    S0 = float(F.compute_S0(St[None, :], F.acq)[0])
    pred = F.simulate_curve(theta_med, aif, t, layout.model, F.acq, subject,
                            F.cfg_fit, S0=S0)
    resid = St - pred
    pr = resid[F.NBASE.stop:]
    base_sd = float(np.std(St[F.NBASE], ddof=1))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    rhat_max = max(summ[f"{p}_rhat"] for p in F.PARAMS)
    ess_min = min(summ[f"{p}_ess"] for p in F.PARAMS)
    n_div = int(res.n_divergences) if res.n_divergences is not None else -1
    disagree = any(summ[f"{p}_chains_disagree"] for p in F.PARAMS)
    row = dict(S0=S0, rmse=rmse,
               rmse_over_baseline_sd=rmse / base_sd if base_sd > 0 else np.nan,
               resid_lag1_autocorr=float(np.corrcoef(pr[:-1], pr[1:])[0, 1]),
               rhat_max=rhat_max, ess_min=ess_min, n_divergences=n_div,
               any_chains_disagree=disagree,
               qc_pass=bool(rhat_max <= F.RHAT_MAX and ess_min >= F.ESS_MIN
                            and n_div == 0 and not disagree),
               ps_flow_limited=bool(summ["E_median"] > F.E_FLOW_LIMITED),
               fit_seconds=fit_s)
    row.update(summ)
    return row, res, post, summ, layout


def fit_nlls(F, sig_4d, mask, aif, t, subject, St, layout, post=None, summ=None):
    t0 = time.time()
    nres = F.fit_roi_with_delay_continuous(
        sig_4d, mask, aif, t, F.TCXM, F.acq, subject, F.cfg_fit, F.roi_cfg,
        tau_lb=F.DELAY_BOUNDS[0], tau_ub=F.DELAY_BOUNDS[1],
        tau_p0=F.NLLS_TAU_P0, seed=F.NLLS_SEED, interp_method=F.NLLS_INTERP)
    secs = time.time() - t0
    if not np.allclose(nres.St_mean, St):
        raise ValueError("NLLS aggregated a different curve than the subset")
    est = F.nlls_estimates(nres)
    lb = np.asarray(layout.model.lb, float)
    ub = np.asarray(layout.model.ub, float)
    logm = set(layout.model.log_dims or ())
    names_fit = list(layout.model.param_names)
    at = {p: F.at_bound(est[p], lb[j], ub[j], j in logm)
          for j, p in enumerate(names_fit)}
    rm = float(np.sqrt(np.mean((St - nres.St_pred) ** 2)))
    row = dict(nlls_success=bool(nres.success), nlls_cost=float(nres.best_cost),
               nlls_rmse=rm, nlls_interp=F.NLLS_INTERP, nlls_seconds=secs,
               nlls_any_at_bound=any(at.values()), nlls_error="")
    row.update({f"nlls_{p}_at_bound": v for p, v in at.items()})
    if post is not None:
        row.update(F.compare_nlls(est, post, summ))
    else:
        row.update({f"nlls_{p}": est[p] for p in F.PARAMS + F.DERIVED})
    return row


# =============================================================================
# main
# =============================================================================

def read_base(base_dir, subset):
    df = pd.read_csv(os.path.join(base_dir, "lesion_bayes_results.csv"))
    err = df["error"].fillna("").astype(str).str.strip() if "error" in df else ""
    df = df[err == ""].copy()
    if subset == "qc_pass":
        df = df[df["qc_pass"].map(lambda v: str(v).strip().lower() == "true")]
    elif subset != "all":
        raise ValueError("BASE_SUBSET must be 'all' or 'qc_pass'")
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    return df


def main():
    F = load_fit_module()
    for k, v in FIT_OVERRIDES.items():
        if not hasattr(F, k):
            raise AttributeError(f"fit module has no setting {k!r}")
        print(f"  ! FIT_OVERRIDES: {k} = {v!r} (was {getattr(F, k)!r})")
        setattr(F, k, v)
    outdir = RESUME_DIR or os.path.join(
        RESULTS_ROOT, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    rowdir = os.path.join(outdir, "rows")
    os.makedirs(rowdir, exist_ok=True)
    print(f"results -> {outdir}")

    base = read_base(BASE_RUN_DIR, BASE_SUBSET)
    benign = base.loc[base["diagnosis"] == "B", "n_vox"].astype(int).to_numpy()
    todo = base[base["diagnosis"] == SUBSAMPLE_DIAGNOSIS]
    if ONLY:
        todo = todo[todo["subject_id"].isin(ONLY)]

    # --- target sizes ---------------------------------------------------------
    # "benign_rank" is assigned over the WHOLE malignant list, before sharding,
    # so every shard (and every resume) sees the same assignment.
    targets = None
    if TARGET_MODE == "benign_rank":
        all_sids = todo["subject_id"].astype(str).str.strip().to_numpy()
        all_sizes = todo["n_vox"].astype(int).to_numpy()
        targets = {}
        for rep in range(N_REPLICATES):
            k = assign_targets_rank(all_sizes, benign, rep)
            targets[rep] = dict(zip(all_sids, k))
            report_targets(all_sizes, k, benign,
                           f" (rep {rep})" if N_REPLICATES > 1 else "")

    if SHARD:
        i, n = (int(x) for x in SHARD.split("/"))
        todo = todo.iloc[i::n]
        print(f"shard {i}/{n}")
    print(f"{len(todo)} {SUBSAMPLE_DIAGNOSIS} lesion(s) x {N_REPLICATES} replicate(s); "
          f"target mode {TARGET_MODE}; benign sizes: n={benign.size}, "
          f"median {np.median(benign):.0f}, "
          f"IQR [{np.percentile(benign, 25):.0f}, {np.percentile(benign, 75):.0f}] voxels")
    print(f"fits: {FITS}; sampler {F.N_CHAINS} chains x {F.N_TUNE}+{F.N_DRAWS} "
          f"(from {os.path.basename(FIT_SCRIPT)})")

    dt = F.SCANTIME * F.SPOKES_PER_FRAME / F.NSPOKE
    for _, b in todo.iterrows():
        sid = str(b["subject_id"]).strip()
        for rep in range(N_REPLICATES):
            row_path = os.path.join(rowdir, f"{sid}_r{rep}.csv")
            if os.path.exists(row_path):
                print(f"{sid} r{rep}: already done, skipping")
                continue
            print(f'\n{"=" * 80}\n{sid} ({b["diagnosis"]}) replicate {rep}\n{"=" * 80}')
            t_all = time.time()
            row = dict(subject_id=sid, diagnosis=b["diagnosis"], replicate=rep)
            try:
                subj_dir = os.path.join(F.DATA_ROOT, sid)
                dce_path = F.find_dce(subj_dir, sid)
                sig_4d, _ = F.load_nifti_canonical(dce_path)
                t = np.arange(sig_4d.shape[-1]) * dt
                lesion, linfo = F.select_largest_lesion(subj_dir, sid, dce_path)
                if (linfo["lesion_file"] != b.get("lesion_file", linfo["lesion_file"])
                        or int(linfo["n_vox"]) != int(b["n_vox"])):
                    raise ValueError(
                        f"lesion selection differs from the base run: "
                        f"{linfo['lesion_file']} ({linfo['n_vox']}) vs "
                        f"{b.get('lesion_file')} ({b['n_vox']})")
                aif, aif_path = F.load_aif_csv(sid, t)
                subject = F.SubjectConfig(subject_id=sid, T10=F.T10_TISSUE, B1=F.B1)

                rng = lesion_rng(sid, rep)
                n_full = int(lesion.sum())
                if targets is not None:
                    k = int(targets[rep][sid])
                    note = TARGET_MODE if k < n_full else "already benign-sized"
                else:
                    k, note = choose_target(rng, n_full, benign)
                if k < n_full:
                    mask, pick = subsample_mask(lesion, k, rng)
                else:
                    mask, pick = lesion, np.flatnonzero(lesion)
                _, n_used, St = F._aggregate_roi_signal(sig_4d, mask, F.acq,
                                                        subject, F.roi_cfg)
                row.update(lesion_file=linfo["lesion_file"], n_vox_full=n_full,
                           n_vox=int(mask.sum()), n_vox_used=int(n_used),
                           subsampled=bool(k < n_full),
                           target_mode=TARGET_MODE, target_k=int(k),
                           target_note=note, seed=SEED,
                           voxel_crc=int(zlib.crc32(pick.astype(np.int64).tobytes())),
                           aif_file=os.path.basename(aif_path),
                           **curve_qc(F, St))
                print(f"  {n_full} -> {int(mask.sum())} voxels ({note}); "
                      f"baseline CV {row['baseline_cv']:.4f}")

                layout = post = summ = None
                if "bayes" in FITS:
                    brow, res, post, summ, layout = fit_bayes(F, St, aif, t, subject)
                    row.update(brow)
                    print(f"  Bayes: R-hat max {brow['rhat_max']:.3f}  ESS min "
                          f"{brow['ess_min']:.0f}  div {brow['n_divergences']}  "
                          f"QC {'PASS' if brow['qc_pass'] else 'FAIL'} "
                          f"({brow['fit_seconds']:.0f}s)")
                    if SAVE_DRAWS:
                        np.savez_compressed(
                            os.path.join(outdir, f"{sid}_r{rep}_draws.npz"),
                            chains=res.chains,
                            param_names=np.array(res.param_names, dtype=object),
                            St=St, aif=aif, t=t, lesion_nvox=int(mask.sum()))
                if "nlls" in FITS:
                    if layout is None:
                        layout = F.build_param_layout(F.TCXM, make_cfg_b(F), aif=aif,
                                                      time_var=t, acq=F.acq)
                    try:
                        row.update(fit_nlls(F, sig_4d, mask, aif, t, subject, St,
                                            layout, post, summ))
                    except Exception as e:
                        print(f"  !! NLLS FAILED: {e!r}")
                        row["nlls_error"] = repr(e)
                row["error"] = ""
            except Exception as e:
                print(f"  !! FAILED: {e!r}")
                row["error"] = repr(e)
            row["total_seconds"] = time.time() - t_all
            print(f"  total time {row['total_seconds'] / 60:.1f} min")
            pd.DataFrame([row]).to_csv(row_path, index=False)

    files = sorted(glob.glob(os.path.join(rowdir, "*.csv")))
    if files:
        allrows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        allrows.to_csv(os.path.join(outdir, "subsample_results.csv"), index=False)
        ok = allrows["error"].fillna("").astype(str).str.strip() == ""
        print(f"\n{int(ok.sum())}/{len(allrows)} lesion x replicate fits OK -> "
              f"{os.path.join(outdir, 'subsample_results.csv')}")
        done = allrows[ok]
        if "n_vox" in done and len(done):
            kk = done["n_vox"].astype(float).to_numpy()
            print(f"  achieved sizes: median {np.median(kk):.0f} voxels "
                  f"(benign {np.median(benign):.0f}), IQR "
                  f"[{np.percentile(kk, 25):.0f}, {np.percentile(kk, 75):.0f}] "
                  f"(benign [{np.percentile(benign, 25):.0f}, "
                  f"{np.percentile(benign, 75):.0f}])")
    return outdir


if __name__ == "__main__":
    main()
