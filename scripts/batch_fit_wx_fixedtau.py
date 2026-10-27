#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:00:54 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_fit_wx_fixedtau.py
========================
Finish the TCXM + FIXED delay + SSM3(tau_i) cohort fit: take the lesions an
earlier run did not reach, group them into batches, and run the batches on two
GPUs through the batched sampler.

It does NOT reimplement the fit. It imports your fitting script (SRC_SCRIPT)
and calls its own loaders, priors, summaries, plots and row builder, so every
number and figure is produced by the code you already validated. The only
thing replaced is the loop: fit_curve_bayes one lesion at a time becomes
fit_curve_bayes_batched over a batch.

WHAT IT DOES
  a) Reads DONE_RUN_DIR/rows/*.csv and drops those subjects. A row with a
     non-empty `error` counts as not done when RETRY_FAILED (default), because
     such a row records a crash, not a fit.
  b) Reads each remaining subject's run time from RUNTIME_RUN_DIR's cohort CSV,
     column RUNTIME_COL ("total_seconds").
  c) Groups the remainder into batches of BATCH_SIZE with similar run times.
  d) Assigns batches to GPUS, longest-first onto whichever GPU has least work
     queued (LPT), which balances better than alternating.
  e) Spawns one worker process per GPU, each pinned with CUDA_VISIBLE_DEVICES,
     and fits its batches. One row CSV is written per lesion as it finishes, so
     the script is interruptible and resumable; rerunning skips what is done.

WHY GROUP BY RUN TIME (your (c))
  A batch steps in lock-step: vmap runs one NUTS step for every chain at once,
  and the step is not finished until the LONGEST trajectory in the batch is.
  A batch therefore costs about as much per step as its slowest lesion, so
  mixing a fast lesion with a slow one wastes the fast one's lanes. Measured on
  this GPU (identical copies, so a best case): 8 copies cost 2.26x one copy per
  step, i.e. 3.5x throughput per lesion; 12 copies cost 4.6x, i.e. only 2.6x --
  past about 128 lanes the GPU stops having spare capacity. BATCH_SIZE = 8 at
  16 chains is 128 lanes, which is where that measurement peaked.

  Two caveats on the grouping. The run times come from RUNTIME_RUN_DIR, which
  is the FXL joint-delay run that supplied the fixed delays -- a different and
  cheaper model than the one fitted here. They are being used as a per-lesion
  DIFFICULTY proxy, which is reasonable, but they are not predictions of this
  fit's cost, so the projected times printed below are rough. And run time is
  only a proxy for trajectory length; grouping cannot remove the lock-step cost,
  only reduce it.

  Batches are also split by FRAME COUNT first, because one compiled model is
  shared across a batch: lesions with different numbers of frames cannot be
  batched together. Everything else the batch must share (TR, FA, r1, T10, B1,
  blood T1, hematocrit, priors, bounds) is a single global setting in your
  script, so it is shared by construction; fit_curve_bayes_batched re-checks
  and raises rather than fitting a mixed batch.

VERIFICATION THIS RESTS ON
  check_16  the batched path runs THIS model (SSM3, tau_i free, tau_b fixed,
            delay folded into the AIF, priors rates=uniform /
            volfrac=flat_masked / tau=flat): the batched log-posterior equals
            the sequential one to 6e-14, the NLLS anchors agree bitwise, and
            the draws agree to ~1e-15 when warmup is too short to amplify
            rounding.
  check_15  at production length, batched and sequential posteriors are
            indistinguishable from two runs of the same method with different
            seeds, with sensitivity to a shift of about 0.02-0.05 posterior SD.
            That was run on FXL with a joint delay, NOT on this model.
  So: the batched code computes this model correctly, and batching did not move
  the posterior for the model it was tested on. The equivalent distributional
  test for THIS model has not been run. VALIDATE_SIDS below does it cheaply.

RUN
    cd ~/DCE_MRI_TK_Fit
    PYTHONPATH=src python scripts/batch_fit_wx_fixedtau.py --dry-run   # plan only
    PYTHONPATH=src python scripts/batch_fit_wx_fixedtau.py             # fit

  --dry-run prints the batches, the GPU assignment and the projected time and
  fits nothing. Look at it first.

  Plots and the cohort CSV: this script writes rows/<sid>.csv, <sid>_draws.npz
  and the per-lesion figures, and concatenates the rows into
  lesion_bayes_results.csv. To get the cohort FIGURES, run your own script
  afterwards with RESUME_DIR set to OUT_DIR: every lesion already has a row, so
  it fits nothing and goes straight to the cohort table and group plots.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# =============================================================================
# settings
# =============================================================================
SRC_SCRIPT = os.path.expanduser(
    "~/DCE_MRI_TK_Fit/scripts/fit_lesions_bayes_wx_fixedtau.py")

# the run whose finished subjects are skipped, and where new results go.
# OUT_DIR = DONE_RUN_DIR resumes that run in place, which is what you want:
# the rows already there are the ones being skipped.
DONE_RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
                "run_20261021_160427")
OUT_DIR = DONE_RUN_DIR

# run times for the grouping. None -> the source script's TAU_FIXED_FROM_RUN,
# which is the FXL run that supplied the fixed delays.
RUNTIME_RUN_DIR = None
RUNTIME_COL = "total_seconds"

BATCH_SIZE = 8
GPUS = (0, 1)
RETRY_FAILED = True        # True: a row whose `error` is set is refitted
MAX_BATCHES = None         # cap the number of batches (testing); None = all
VALIDATE_SIDS = ()         # e.g. ("FMB003", "FMB010"): refit these lesions
                           # SEQUENTIALLY after the batches, into
                           # OUT_DIR/validate_sequential/, so batched and
                           # sequential posteriors can be compared for THIS
                           # model. Costs one sequential fit per lesion.
PROGRESS_EVERY_S = 60


# =============================================================================
# the source script
# =============================================================================
def load_src(path=SRC_SCRIPT):
    """Import the fitting script as a module without running its main()."""
    import importlib.util
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SRC_SCRIPT not found: {path}\nPoint SRC_SCRIPT at your "
            f"TCXM + fixed delay + SSM3 fitting script.")
    spec = importlib.util.spec_from_file_location("wx_src", path)
    M = importlib.util.module_from_spec(spec)
    sys.modules["wx_src"] = M
    spec.loader.exec_module(M)          # safe: its work is behind __main__
    for need in ("fit_one", "tau_for", "select_largest_lesion", "summarise",
                 "WX_MODEL", "relax_cfg", "PARAMS"):
        if not hasattr(M, need):
            raise AttributeError(f"{path} has no {need!r} -- is SRC_SCRIPT the "
                                 f"right script?")
    if M.WX_MODEL is None:
        raise ValueError(f"{path} has WX_MODEL = None, so it is not the "
                         f"water-exchange fit this driver is for")
    return M


# =============================================================================
# planning
# =============================================================================
def done_sids(outdir, retry_failed=RETRY_FAILED):
    """{sid: status} for every row already written in outdir/rows."""
    import pandas as pd
    out = {}
    for f in sorted(glob.glob(os.path.join(outdir, "rows", "*.csv"))):
        sid = os.path.basename(f)[:-4]
        try:
            d = pd.read_csv(f)
        except Exception:
            out[sid] = "unreadable"
            continue
        err = ""
        if "error" in d.columns and len(d):
            v = d["error"].iloc[0]
            err = "" if pd.isna(v) else str(v).strip()
        if err and retry_failed:
            out[sid] = "failed"          # refit
        else:
            out[sid] = "error" if err else "ok"
    return out


def load_runtimes(run_dir, col=RUNTIME_COL):
    """{sid: seconds} from a run's cohort CSV."""
    import numpy as np
    import pandas as pd
    path = os.path.join(run_dir, "lesion_bayes_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no lesion_bayes_results.csv in {run_dir}")
    df = pd.read_csv(path)
    use = col
    if use not in df.columns:
        use = df.columns[-1]
        print(f"  !! {col!r} not in {path}; using the last column {use!r}")
    out = {}
    for _, r in df.iterrows():
        v = r[use]
        if pd.notna(v):
            try:
                out[str(r["subject_id"]).strip()] = float(v)
            except (TypeError, ValueError):
                pass
    if not out:
        raise ValueError(f"no usable run times in {path} column {use!r}")
    return out, path, use


def n_frames_of(M, sid):
    """Frame count from the DCE header alone -- no voxel data is read.

    Canonical reorientation permutes the SPATIAL axes only, so the last axis is
    the time axis either way. The worker re-derives the frame count from the
    volume it actually loads and refuses a batch that disagrees with the plan.
    """
    import nibabel as nib
    return int(nib.load(M.find_dce(sid)).shape[-1])


def build_plan(M, verbose=True):
    """-> (batches, skipped, problems). Each batch is a dict ready for a worker."""
    import numpy as np
    import pandas as pd

    tumors = pd.read_csv(M.TUMOR_CSV)
    tumors.columns = [c.strip() for c in tumors.columns]
    tumors["Patient_ID"] = tumors["Patient_ID"].str.strip()
    tumors["Diagnosis"] = tumors["Diagnosis"].str.strip().str.upper()
    if M.ONLY:
        tumors = tumors[tumors["Patient_ID"].isin(M.ONLY)]
    cohort = [(r["Patient_ID"], r["Diagnosis"]) for _, r in tumors.iterrows()]

    status = done_sids(OUT_DIR)
    skipped = {s: v for s, v in status.items() if v in ("ok", "error")}
    todo = [(s, d) for s, d in cohort if status.get(s) not in ("ok", "error")]

    rt_dir = RUNTIME_RUN_DIR or M.TAU_FIXED_FROM_RUN
    rt, rt_path, rt_col = load_runtimes(rt_dir)
    med = float(np.median(list(rt.values())))

    if verbose:
        print(f"cohort            {len(cohort)} subjects"
              + (f"  (ONLY={M.ONLY})" if M.ONLY else ""))
        print(f"already in {os.path.basename(OUT_DIR)}: {len(skipped)} "
              f"({sum(v == 'ok' for v in skipped.values())} fitted, "
              f"{sum(v == 'error' for v in skipped.values())} error rows kept)")
        n_retry = sum(v == 'failed' for v in status.values())
        if n_retry:
            print(f"  refitting {n_retry} subject(s) whose row records an error "
                  f"(RETRY_FAILED=True)")
        print(f"to fit            {len(todo)}")
        print(f"run times from    {rt_path}")
        print(f"                  column {rt_col!r}, median {med / 60:.1f} min; "
              f"missing -> median")

    # frame count per lesion: the hard batching constraint
    problems, rows = [], []
    for sid, dx in todo:
        try:
            nf = n_frames_of(M, sid)
        except Exception as e:
            problems.append((sid, repr(e)))
            continue
        rows.append(dict(sid=sid, dx=dx, n_frames=nf,
                         runtime_s=rt.get(sid, med),
                         runtime_known=sid in rt))
    if problems and verbose:
        print(f"\n{len(problems)} subject(s) cannot be planned (no DCE / no mask):")
        for sid, e in problems:
            print(f"  {sid}: {e}")
        print("  these are left for your own script to run and record an error row")

    # group: frame count first, then similar run time
    batches = []
    for nf in sorted({r["n_frames"] for r in rows}):
        grp = sorted([r for r in rows if r["n_frames"] == nf],
                     key=lambda r: r["runtime_s"])
        for i in range(0, len(grp), BATCH_SIZE):
            chunk = grp[i:i + BATCH_SIZE]
            batches.append(dict(
                n_frames=nf,
                sids=[c["sid"] for c in chunk],
                dxs=[c["dx"] for c in chunk],
                runtimes=[c["runtime_s"] for c in chunk],
                # lock-step: a batch costs about as much per step as its
                # slowest member, so that is the scheduling weight
                cost_s=max(c["runtime_s"] for c in chunk)))
    batches.sort(key=lambda b: -b["cost_s"])          # longest first
    if MAX_BATCHES is not None:
        batches = batches[:MAX_BATCHES]

    # LPT: each batch to the least-loaded GPU
    load = {g: 0.0 for g in GPUS}
    for b in batches:
        g = min(load, key=load.get)
        b["gpu"] = g
        load[g] += b["cost_s"]
    for i, b in enumerate(batches):
        b["index"] = i

    if verbose and batches:
        print(f"\n{len(batches)} batch(es) of up to {BATCH_SIZE}, "
              f"{len({b['n_frames'] for b in batches})} distinct frame count(s)")
        hdr = f"{'#':>3} {'gpu':>3} {'frames':>6} {'n':>2} {'slowest':>9}  lesions"
        print(hdr)
        print("-" * max(len(hdr), 72))
        for b in sorted(batches, key=lambda b: (b["gpu"], -b["cost_s"])):
            print(f"{b['index']:>3} {b['gpu']:>3} {b['n_frames']:>6} "
                  f"{len(b['sids']):>2} {b['cost_s'] / 60:>7.1f}m  "
                  + " ".join(b["sids"]))
        print("\nper-GPU queued difficulty (sum of per-batch slowest run time, "
              "from the FXL run):")
        for g in GPUS:
            print(f"  GPU {g}: {load[g] / 60:6.1f} min over "
                  f"{sum(1 for b in batches if b['gpu'] == g)} batch(es)")
        seq = sum(sum(b["runtimes"]) for b in batches)
        print(f"\nSequential FXL-equivalent work: {seq / 3600:.1f} h. This model is "
              f"slower than that run,\nand batching + 2 GPUs cut it; the printed "
              f"numbers rank lesions, they do not predict\nthis fit's wall time. "
              f"Watch the first finished batch for a real rate.")
    return batches, skipped, problems


# =============================================================================
# one lesion: the two halves of the source script's fit_one
# =============================================================================
def prepare_inputs(M, sid, dx, outdir):
    """
    Everything fit_one does BEFORE fit_curve_bayes: volume, mask, lesion curve,
    baseline window, fixed delay, the NLLS arm, and the lesion figure.

    Mirrors fit_one step for step and calls the same functions, so the inputs
    the batched sampler sees are the inputs the sequential fit would have seen.
    The 4-D volume is released when this returns; only small arrays are kept.
    """
    import numpy as np
    from dataclasses import replace

    mask_dir = M.mask_dir_for(sid)
    dce_path = M.find_dce(sid)
    sig_4d, _ = M.load_nifti_canonical(dce_path)
    dt = M.SCANTIME * M.SPOKES_PER_FRAME / M.NSPOKE
    t = np.arange(sig_4d.shape[-1]) * dt

    lesion, linfo = M.select_largest_lesion(mask_dir, sid, dce_path)
    if lesion.shape != sig_4d.shape[:3]:
        raise ValueError(f"{sid}: lesion mask {lesion.shape} does not match the "
                         f"{M.RECON} volume {sig_4d.shape[:3]} "
                         f"({os.path.basename(dce_path)})")
    aif, aif_path = M.load_aif_csv(sid, t)
    subject = M.SubjectConfig(subject_id=sid, T10=M.T10_TISSUE, B1=M.B1)
    arr = M.aif_arrival(sid)

    _, n_vox, St = M._aggregate_roi_signal(sig_4d, lesion, M.acq, subject, M.roi_cfg)
    nb, binfo = M.choose_nbase(sid, St, arr)
    acq_s = replace(M.acq, nbase=nb)
    _, _, St_chk = M._aggregate_roi_signal(sig_4d, lesion, acq_s, subject, M.roi_cfg)
    if not np.allclose(St_chk, St):
        raise ValueError(f"{sid}: lesion mean signal changed with nbase -- "
                         f"roi_cfg.average_space must be 'signal'")
    M.plot_lesion(sig_4d, lesion, sid, outdir)

    base = St[nb]
    peak_enh = float(St.max() / np.mean(base))

    tau_fix, tinfo = M.tau_for(sid)
    aif_shift = M.shift_aif(aif, float(tau_fix), t, acq_s)
    aif_pad = float(np.mean(np.asarray(aif)[nb]))
    tinfo.update(aif_pad_mM=aif_pad,
                 aif_pad_flag=bool(abs(aif_pad) > M.AIF_PAD_FLAG))

    nres, est, nlls_row, nlls_s = None, None, {}, float("nan")
    if M.RUN_NLLS:
        try:
            t1 = time.time()
            nres = M.fit_roi(sig_4d, lesion, aif_shift, t, M.TCXM, acq_s, subject,
                             M.cfg_fit, M.roi_cfg, seed=M.NLLS_SEED,
                             wx_model=M.WX_MODEL,
                             aif_cfg=(M.aif_cfg if M.WX_MODEL else None),
                             relax_cfg=(M.relax_cfg if M.WX_MODEL else None))
            nlls_s = time.time() - t1
            if not np.allclose(nres.St_mean, St):
                raise ValueError("NLLS aggregated a different lesion curve than "
                                 "the Bayesian fit")
            est = M.nlls_estimates(nres)
        except Exception as e:
            print(f"  !! {sid}: NLLS FAILED: {e!r}", flush=True)
            nres, est, nlls_row = None, None, dict(nlls_error=repr(e))

    cfg_b = M.BayesConfig(
        prior_choices=M.PRIOR_CHOICES, n_draws=M.N_DRAWS, n_tune=M.N_TUNE,
        n_chains=M.N_CHAINS, noise="gauss_marginalized_sigma", seed=M.SEED,
        backend=M.BACKEND, step="nuts", dense_mass_matrix=M.DENSE_MASS,
        delay_mode=M.DELAY_MODE, delay_fixed=float(tau_fix),
        delay_bounds=M.DELAY_BOUNDS)

    return dict(sid=sid, dx=dx, t=t, St=St, aif=aif, aif_shift=aif_shift,
                acq_s=acq_s, subject=subject, nb=nb, binfo=binfo, linfo=linfo,
                tinfo=tinfo, arr=arr, n_vox=n_vox, dce_path=dce_path,
                aif_path=aif_path, tau_fix=float(tau_fix), cfg_b=cfg_b,
                base=base, peak_enh=peak_enh, nres=nres, est=est,
                nlls_row=nlls_row, nlls_s=nlls_s)


def finish_one(M, inp, res, fit_s, outdir, batch_meta):
    """
    Everything fit_one does AFTER fit_curve_bayes, for an already-fitted result:
    prior figure, summaries, posterior-median prediction, NLLS comparison,
    draws file and the result row. Same calls, same order, same columns.
    """
    import numpy as np

    sid, dx, t, St, aif = inp["sid"], inp["dx"], inp["t"], inp["St"], inp["aif"]
    acq_s, subject, nb = inp["acq_s"], inp["subject"], inp["nb"]
    cfg_b, est, nres = inp["cfg_b"], inp["est"], inp["nres"]

    layout = M.build_param_layout(M.TCXM, cfg_b, aif=aif, time_var=t, acq=acq_s,
                                  wx_model=M.WX_MODEL,
                                  relax_cfg=(M.relax_cfg if M.WX_MODEL else None))
    noise = M.NOISE_OPTIONS["gauss_marginalized_sigma"]
    prior = M.assemble_prior(layout.model, (M.relax_cfg if M.WX_MODEL else None),
                             wx_model=M.WX_MODEL, delay_mode=M.DELAY_MODE,
                             n_noise=noise.n_params, noise_names=noise.param_names,
                             choices=M.PRIOR_CHOICES, delay_bounds=M.DELAY_BOUNDS)
    prior.validate_against(layout.param_names)
    prior_samples = prior.sample(np.random.default_rng(M.PRIOR_SEED),
                                 M.N_PRIOR_SAMPLES)[:, :layout.n_model]
    names_fit = list(layout.param_names[:layout.n_model])
    lb_fit = np.asarray(layout.lb, float)[:layout.n_model]
    ub_fit = np.asarray(layout.ub, float)[:layout.n_model]
    prior_d = {p: prior_samples[:, names_fit.index(p)] for p in M.PARAMS}
    prior_d.update(M.derived_draws(prior_d))

    M.plot_prior_vs_posterior(res, names_fit, M.LOG_SCALE, lb_fit, ub_fit,
                              prior_samples, f"{sid} ({dx})",
                              os.path.join(outdir, f"{sid}_prior_post.png"),
                              nlls_vals=est)

    summ, post = M.summarise(res, prior_d)

    theta_med = np.array([summ[f"{p}_median"] for p in names_fit])
    S0 = float(M.compute_S0(St[None, :], acq_s)[0])
    pred = M.simulate_curve(theta_med, inp["aif_shift"], t, layout.model, acq_s,
                            subject, M.cfg_fit, wx_model=M.WX_MODEL,
                            aif_cfg=(M.aif_cfg if M.WX_MODEL else None),
                            relax_cfg=(M.relax_cfg if M.WX_MODEL else None), S0=S0)
    resid = St - pred
    post_resid = resid[nb.stop:]
    lag1 = float(np.corrcoef(post_resid[:-1], post_resid[1:])[0, 1])
    base_sd = float(np.std(inp["base"], ddof=1))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    M.plot_fit(t, St, pred, sid, dx, outdir, nb,
               pred_nlls=(nres.St_pred if nres is not None else None))

    nlls_row = dict(inp["nlls_row"])
    if nres is not None:
        at = {p: M.at_bound(est[p], lb_fit[j], ub_fit[j], p in M.LOG_SCALE)
              for j, p in enumerate(names_fit)}
        rmse_nlls = float(np.sqrt(np.mean((St - nres.St_pred) ** 2)))
        nlls_row = dict(nlls_success=bool(nres.success),
                        nlls_cost=M.nlls_cost(nres), nlls_rmse=rmse_nlls,
                        nlls_rmse_over_baseline_sd=(rmse_nlls / base_sd
                                                    if base_sd > 0 else np.nan),
                        nlls_interp="fixed_delay_preshift",
                        nlls_seconds=inp["nlls_s"],
                        nlls_any_at_bound=any(at.values()), nlls_error="")
        nlls_row.update({f"nlls_{p}_at_bound": v for p, v in at.items()})
        nlls_row.update(M.compare_nlls(est, post, summ))
        if res.point_estimate is not None:
            pe = np.asarray(res.point_estimate, float)[:len(names_fit)]
            th = np.array([est[p] for p in names_fit])
            nlls_row["nlls_vs_bayes_anchor_maxdiff"] = float(
                np.max(np.abs(th - pe) / (ub_fit - lb_fit)))

    np.savez_compressed(os.path.join(outdir, f"{sid}_draws.npz"),
                        chains=res.chains,
                        param_names=np.array(res.param_names, dtype=object),
                        St=St, aif=aif, aif_shift=inp["aif_shift"], t=t,
                        tau_delay_fixed=inp["tau_fix"], lesion_nvox=inp["n_vox"])

    rhat_max = max(summ[f"{p}_rhat"] for p in M.PARAMS)
    ess_min = min(summ[f"{p}_ess"] for p in M.PARAMS)
    mcse_rel = [summ[f"{p}_mcse_median_rel_ci"] for p in M.PARAMS]
    mcse_rel_max = float(np.nanmax(mcse_rel)) if len(mcse_rel) else np.nan
    n_div = int(res.n_divergences) if res.n_divergences is not None else -1
    any_disagree = any(summ[f"{p}_chains_disagree"] for p in M.PARAMS)
    mcse_ok = not (np.isfinite(mcse_rel_max) and mcse_rel_max > M.MCSE_REL_MAX)
    qc_pass = bool(rhat_max <= M.RHAT_MAX and ess_min >= M.ESS_MIN
                   and n_div == 0 and not any_disagree and mcse_ok)

    row = dict(subject_id=sid, diagnosis=dx, recon=M.RECON,
               dce_file=os.path.basename(inp["dce_path"]),
               dce_dir=os.path.dirname(inp["dce_path"]), **inp["tinfo"],
               aif_file=os.path.basename(inp["aif_path"]), **inp["linfo"],
               peak_enhancement=inp["peak_enh"],
               aif_arrival_frame=inp["arr"],
               baseline_overlaps_aif_arrival=bool(np.isfinite(inp["arr"])
                                                  and inp["arr"] < nb.stop),
               **inp["binfo"],
               S0=S0, rmse=rmse, baseline_sd=base_sd,
               rmse_over_baseline_sd=rmse / base_sd if base_sd > 0 else np.nan,
               resid_lag1_autocorr=lag1,
               rhat_max=rhat_max, ess_min=ess_min,
               mcse_median_rel_ci_max=mcse_rel_max, mcse_rel_flag=not mcse_ok,
               n_divergences=n_div,
               any_chains_disagree=any_disagree, qc_pass=qc_pass,
               ps_flow_limited=bool(summ["E_median"] > M.E_FLOW_LIMITED),
               model=M.MODEL_TAG, delay_mode=M.DELAY_MODE,
               wx_model=(M.WX_MODEL.name if M.WX_MODEL else ""),
               wx_fit_tau_b=(bool(M.relax_cfg.fit_tau_b) if M.WX_MODEL else False),
               wx_tau_b_fixed=(float(M.relax_cfg.tau_b_fixed) if M.WX_MODEL else np.nan),
               wx_tau_i_lb=(float(M.relax_cfg.tau_i_lb) if M.WX_MODEL else np.nan),
               wx_tau_i_ub=(float(M.relax_cfg.tau_i_ub) if M.WX_MODEL else np.nan),
               wx_blood_T10=(float(M.aif_cfg.T10) if M.WX_MODEL else np.nan),
               wx_hematocrit=(float(M.aif_cfg.hematocrit) if M.WX_MODEL else np.nan),
               vi_small_flag=bool(M.WX_MODEL and summ["vi_median"] < M.VI_SMALL),
               fit_seconds=fit_s, error="")
    row.update(summ)
    row.update(nlls_row)
    # provenance of the batched run. fit_seconds above is the batch's wall time
    # divided by its size -- a SHARE, not a sequential fit time, so it is not
    # comparable with a sequential run's fit_seconds and must not be reused as
    # a run-time input for grouping.
    row.update(batch_meta)
    return row


# =============================================================================
# worker
# =============================================================================
def run_worker(gpu, plan_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import numpy as np
    import pandas as pd
    import jax
    from dce_mri.bayes import fit_curve_bayes
    from dce_mri import bayes_batched as bb

    with open(plan_path) as fh:
        plan = json.load(fh)
    outdir = plan["out_dir"]
    rowdir = os.path.join(outdir, "rows")
    os.makedirs(rowdir, exist_ok=True)
    mine = [b for b in plan["batches"] if b["gpu"] == gpu]

    M = load_src(plan["src_script"])
    print(f"[gpu {gpu}] devices: {jax.devices()}  x64: {jax.config.jax_enable_x64}")
    print(f"[gpu {gpu}] {len(mine)} batch(es), "
          f"{sum(len(b['sids']) for b in mine)} lesion(s)", flush=True)

    for b in mine:
        # Skip a lesion only by the SAME rule the planner used, re-read now so
        # an interrupted earlier run of this driver is resumed rather than
        # repeated. Testing only whether the row FILE exists would silently
        # un-do RETRY_FAILED: a lesion whose row records a crash has a file,
        # and the planner put it in this batch precisely to refit it.
        status = done_sids(outdir, plan.get("retry_failed", RETRY_FAILED))
        todo = [(s, d) for s, d in zip(b["sids"], b["dxs"])
                if status.get(s) not in ("ok", "error")]
        if not todo:
            print(f"[gpu {gpu}] batch {b['index']}: already fitted, skipping",
                  flush=True)
            continue
        print(f"\n[gpu {gpu}] {'=' * 64}\n[gpu {gpu}] batch {b['index']}: "
              f"{len(todo)} lesion(s), {b['n_frames']} frames: "
              f"{' '.join(s for s, _ in todo)}\n[gpu {gpu}] {'=' * 64}", flush=True)

        # --- CPU preparation -------------------------------------------------
        t_prep = time.time()
        inputs, prepared = [], []
        for sid, dx in todo:
            try:
                inp = prepare_inputs(M, sid, dx, outdir)
                if inp["t"].size != b["n_frames"]:
                    raise ValueError(
                        f"{inp['t'].size} frames, plan says {b['n_frames']}")
                p = bb.prepare_lesion(
                    inp["St"], inp["aif"], inp["t"], M.TCXM, inp["acq_s"],
                    inp["subject"], M.cfg_fit, inp["cfg_b"],
                    wx_model=M.WX_MODEL, aif_cfg=M.aif_cfg,
                    relax_cfg=M.relax_cfg, point_fit=None)
                inputs.append(inp)
                prepared.append(p)
                print(f"[gpu {gpu}]   {sid}: prepared "
                      f"(tau_delay {inp['tau_fix']:.2f} s, {inp['n_vox']} voxels)",
                      flush=True)
            except Exception as e:
                print(f"[gpu {gpu}]   !! {sid} FAILED in preparation: {e!r}",
                      flush=True)
                pd.DataFrame([dict(subject_id=sid, diagnosis=dx, recon=M.RECON,
                                   model=M.MODEL_TAG, delay_mode=M.DELAY_MODE,
                                   error=repr(e), total_seconds=np.nan,
                                   batch_gpu=gpu, batch_index=b["index"])]
                             ).to_csv(os.path.join(rowdir, f"{sid}.csv"), index=False)
        prep_s = time.time() - t_prep
        if not prepared:
            continue

        # --- one batched fit --------------------------------------------------
        t0 = time.time()
        results, how = None, "batched"
        try:
            results = bb.fit_curve_bayes_batched(
                None, None, inputs[0]["t"], M.TCXM,
                [i["acq_s"] for i in inputs], [i["subject"] for i in inputs],
                M.cfg_fit, inputs[0]["cfg_b"], prepared=prepared,
                wx_model=M.WX_MODEL, aif_cfg=M.aif_cfg, relax_cfg=M.relax_cfg,
                verbose=True)
        except Exception as e:
            print(f"[gpu {gpu}]   !! BATCH FIT FAILED: {e!r}\n"
                  f"[gpu {gpu}]   falling back to one lesion at a time",
                  flush=True)
            how, results = "sequential_fallback", []
            for inp in inputs:
                try:
                    results.append(fit_curve_bayes(
                        inp["St"], inp["aif"], inp["t"], M.TCXM, inp["acq_s"],
                        inp["subject"], M.cfg_fit, inp["cfg_b"],
                        wx_model=M.WX_MODEL, aif_cfg=M.aif_cfg,
                        relax_cfg=M.relax_cfg))
                except Exception as e2:
                    print(f"[gpu {gpu}]   !! {inp['sid']} FAILED: {e2!r}", flush=True)
                    results.append(None)
        batch_s = time.time() - t0

        # --- per-lesion post-processing --------------------------------------
        for inp, res in zip(inputs, results):
            sid = inp["sid"]
            row_path = os.path.join(rowdir, f"{sid}.csv")
            t1 = time.time()
            if res is None:
                pd.DataFrame([dict(subject_id=sid, diagnosis=inp["dx"],
                                   recon=M.RECON, model=M.MODEL_TAG,
                                   delay_mode=M.DELAY_MODE,
                                   error="fit failed in batch and in fallback",
                                   total_seconds=np.nan, batch_gpu=gpu,
                                   batch_index=b["index"])]
                             ).to_csv(row_path, index=False)
                continue
            bmeta = dict(batch_gpu=gpu, batch_index=b["index"],
                         batch_size=len(inputs), batch_how=how,
                         batch_wall_s=batch_s, batch_prep_s=prep_s,
                         batch_frames=b["n_frames"],
                         batch_runtime_proxy_s=inp.get("runtime_s", np.nan))
            bm = res.meta.get("batch") if isinstance(res.meta, dict) else None
            if bm:
                bmeta.update(batch_compile_s=bm.get("compile_s"),
                             batch_sample_s=bm.get("sample_s"),
                             batch_mean_leapfrog=bm.get("mean_leapfrog"),
                             batch_mean_of_batch_max=bm.get("mean_of_batch_max"),
                             batch_mean_of_lesion_max=bm.get("mean_of_lesion_max"))
            try:
                row = finish_one(M, inp, res, float(res.elapsed_s or np.nan),
                                 outdir, bmeta)
                print(f"[gpu {gpu}]   {sid}: R-hat {row['rhat_max']:.3f}  "
                      f"ESS {row['ess_min']:.0f}  div {row['n_divergences']}  "
                      f"QC {'PASS' if row['qc_pass'] else 'FAIL'}  "
                      f"tau_i={row['tau_i_median']:.3g}"
                      f"{'*' if row['tau_i_prior_dominated'] else ''}  "
                      f"vi={row['vi_median']:.3f}"
                      f"{'  [vi_small]' if row['vi_small_flag'] else ''}",
                      flush=True)
            except Exception as e:
                print(f"[gpu {gpu}]   !! {sid} FAILED after fitting: {e!r}",
                      flush=True)
                row = dict(subject_id=sid, diagnosis=inp["dx"], recon=M.RECON,
                           model=M.MODEL_TAG, delay_mode=M.DELAY_MODE,
                           error=repr(e), **bmeta)
            row["finish_seconds"] = time.time() - t1
            # total_seconds keeps the source script's meaning as closely as a
            # batched run allows: this lesion's own CPU work plus its share of
            # the batch's GPU time.
            row["total_seconds"] = (prep_s / len(inputs) + batch_s / len(inputs)
                                    + row["finish_seconds"])
            pd.DataFrame([row]).to_csv(row_path, index=False)
        print(f"[gpu {gpu}] batch {b['index']} done: prep {prep_s / 60:.1f} min, "
              f"{how} {batch_s / 60:.1f} min "
              f"({batch_s / 60 / max(len(inputs), 1):.1f} min per lesion)",
              flush=True)

    print(f"[gpu {gpu}] all batches done", flush=True)


# =============================================================================
# orchestrator
# =============================================================================
def validate_sequential(M, sids, outdir):
    """Refit the named lesions ONE AT A TIME into outdir/validate_sequential.

    Gives a sequential counterpart to the batched posterior for this model, so
    the two can be compared the way check_15 compared them for FXL. Uses the
    same seed, so the comparison is of the two code paths rather than of two
    seeds -- the draws will still differ (see check_16 Part 4), so compare
    medians, SDs and R-hat, not draw by draw.
    """
    import numpy as np
    import pandas as pd
    from dce_mri.bayes import fit_curve_bayes
    vdir = os.path.join(outdir, "validate_sequential")
    os.makedirs(os.path.join(vdir, "rows"), exist_ok=True)
    for sid in sids:
        row_path = os.path.join(vdir, "rows", f"{sid}.csv")
        if os.path.exists(row_path):
            print(f"  {sid}: already validated, skipping", flush=True)
            continue
        print(f"\n  sequential validation fit: {sid}", flush=True)
        t0 = time.time()
        try:
            inp = prepare_inputs(M, sid, "?", vdir)
            res = fit_curve_bayes(inp["St"], inp["aif"], inp["t"], M.TCXM,
                                  inp["acq_s"], inp["subject"], M.cfg_fit,
                                  inp["cfg_b"], wx_model=M.WX_MODEL,
                                  aif_cfg=M.aif_cfg, relax_cfg=M.relax_cfg)
            row = finish_one(M, inp, res, time.time() - t0, vdir,
                             dict(batch_how="sequential_validation"))
        except Exception as e:
            print(f"  !! {sid} validation FAILED: {e!r}", flush=True)
            row = dict(subject_id=sid, error=repr(e))
        row["total_seconds"] = time.time() - t0
        pd.DataFrame([row]).to_csv(row_path, index=False)
    print(f"\nsequential validation fits -> {vdir}\n"
          f"Compare each lesion's {sid}_draws.npz there with the batched one in\n"
          f"{outdir} (medians, SDs, R-hat -- not draw by draw).", flush=True)


def cohort_table(outdir):
    import pandas as pd
    rows = [pd.read_csv(f) for f in
            sorted(glob.glob(os.path.join(outdir, "rows", "*.csv")))]
    if not rows:
        print("no rows to concatenate")
        return None
    df = pd.concat(rows, ignore_index=True)
    out = os.path.join(outdir, "lesion_bayes_results.csv")
    df.to_csv(out, index=False)
    err = (df["error"].notna() & (df["error"].astype(str).str.strip() != "")
           if "error" in df.columns else pd.Series(False, index=df.index))
    ok = df[~err]
    qc = (ok["qc_pass"].map(lambda v: str(v).strip().lower() == "true").sum()
          if "qc_pass" in ok.columns and len(ok) else 0)
    print(f"\ncohort table -> {out}")
    print(f"{len(ok)}/{len(df)} subjects fitted, {int(qc)} pass QC, "
          f"{int(err.sum())} error row(s)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="plan", choices=("plan", "worker"))
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--plan")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.role == "worker":
        return run_worker(a.gpu, a.plan)

    # the orchestrator must not take a GPU: both are for the workers
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 78)
    print("batch_fit_wx_fixedtau -- planning")
    print("=" * 78)
    M = load_src()
    print(f"source script     {SRC_SCRIPT}")
    print(f"model             {M.MODEL_TAG}   sampled {M.PARAMS}")
    print(f"sampler           {M.N_TUNE} tune + {M.N_DRAWS} draws, "
          f"{M.N_CHAINS} chains, seed {M.SEED}")
    print(f"water exchange    {M.WX_MODEL.name}, fit_tau_b="
          f"{M.relax_cfg.fit_tau_b}, tau_b fixed {M.relax_cfg.tau_b_fixed} s, "
          f"tau_i in [{M.relax_cfg.tau_i_lb}, {M.relax_cfg.tau_i_ub}] s")
    print(f"priors            {M.PRIOR_CHOICES}")
    print(f"out               {OUT_DIR}"
          + ("   (resuming that run in place)" if OUT_DIR == DONE_RUN_DIR else ""))
    print(f"batching          {BATCH_SIZE} lesions x {M.N_CHAINS} chains = "
          f"{BATCH_SIZE * M.N_CHAINS} lanes, GPUs {list(GPUS)}\n")

    batches, skipped, problems = build_plan(M)
    if not batches:
        print("\nnothing to fit.")
        cohort_table(OUT_DIR)
        return

    os.makedirs(os.path.join(OUT_DIR, "rows"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_path = os.path.join(OUT_DIR, f"batch_plan_{stamp}.json")
    with open(plan_path, "w") as fh:
        json.dump(dict(created=stamp, src_script=SRC_SCRIPT, out_dir=OUT_DIR,
                       batch_size=BATCH_SIZE, gpus=list(GPUS),
                       retry_failed=bool(RETRY_FAILED),
                       runtime_run_dir=RUNTIME_RUN_DIR or M.TAU_FIXED_FROM_RUN,
                       batches=batches, skipped=skipped,
                       problems=problems), fh, indent=1)
    print(f"\nplan -> {plan_path}")

    if a.dry_run:
        print("\n--dry-run: nothing fitted. Check the batches above, then rerun "
              "without --dry-run.")
        return

    n_target = sum(len(b["sids"]) for b in batches)
    done0 = len(glob.glob(os.path.join(OUT_DIR, "rows", "*.csv")))
    procs, logs = {}, {}
    for g in GPUS:
        if not any(b["gpu"] == g for b in batches):
            continue
        log_path = os.path.join(OUT_DIR, f"batch_gpu{g}_{stamp}.log")
        logs[g] = log_path
        fh = open(log_path, "w")
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)      # the worker sets its own
        env.pop("JAX_PLATFORMS", None)
        procs[g] = (subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--role", "worker",
             "--gpu", str(g), "--plan", plan_path],
            stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
        print(f"GPU {g}: pid {procs[g][0].pid}, log {log_path}")

    print(f"\nfitting {n_target} lesion(s). Follow a worker with:\n"
          + "\n".join(f"  tail -f {p}" for p in logs.values()))
    t0 = time.time()
    try:
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(PROGRESS_EVERY_S)
            n = len(glob.glob(os.path.join(OUT_DIR, "rows", "*.csv"))) - done0
            el = (time.time() - t0) / 60
            rate = f", {el / n:.1f} min/lesion" if n else ""
            alive = [g for g, (p, _) in procs.items() if p.poll() is None]
            print(f"  [{el:6.1f} min] {n}/{n_target} new rows{rate}; "
                  f"GPU(s) running: {alive}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating workers. Finished lesions keep their "
              "rows; rerunning resumes.")
        for p, _ in procs.values():
            p.terminate()
    for g, (p, fh) in procs.items():
        p.wait()
        fh.close()
        print(f"GPU {g}: exit {p.returncode}"
              + ("" if p.returncode == 0 else f"  -- see {logs[g]}"))

    print(f"\ntotal wall time {(time.time() - t0) / 60:.1f} min")
    cohort_table(OUT_DIR)
    if VALIDATE_SIDS:
        print("\n" + "=" * 78)
        print(f"sequential validation fits for {list(VALIDATE_SIDS)}")
        print("=" * 78)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUS[0])
        validate_sequential(M, VALIDATE_SIDS, OUT_DIR)
    print(f"\nFor the cohort FIGURES, run your own script with "
          f"RESUME_DIR = {OUT_DIR!r}:\nevery lesion has a row, so it fits "
          f"nothing and goes straight to the group plots.")


if __name__ == "__main__":
    main()