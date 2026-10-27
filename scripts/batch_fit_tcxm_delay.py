#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:56:01 2026

@author: allisonchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_fit_tcxm_delay.py
=======================
Batched, two-GPU driver for the TCXM + JOINT-delay fit (fit_lesions_bayes.py).
The twin of batch_fit_wx_fixedtau.py: same planning, same batching, same
workers, same resume rules. What differs is the model, and HOW the driver
reaches the fitting script.

It does NOT reimplement the fit -- and, unlike batch_fit_wx_fixedtau.py, it
does not even mirror it. fit_lesions_bayes.py now exposes the two halves of
its own fit_one:

    fit_one = prepare_inputs  ->  fit_curve_bayes          ->  finish_one
    batched = prepare_inputs  ->  fit_curve_bayes_batched  ->  finish_one

and this driver calls those two functions directly. So the sequential and
batched paths share every line of pre- and post-processing; the only thing
replaced is the sampler call. batch_fit_wx_fixedtau.py instead copies its
script's fit_one step by step, which is how two copies drift apart.

WHAT IT DOES
  a) Reads OUT_DIR/rows/*.csv and drops those subjects. A row with a
     non-empty `error` counts as not done when RETRY_FAILED (default), because
     such a row records a crash, not a fit.
  b) Reads each remaining subject's run time from RUNTIME_RUN_DIR's cohort CSV
     (column RUNTIME_COL). With no such file every lesion gets the same weight.
  c) Groups the remainder into batches of BATCH_SIZE with similar run times,
     split first by FRAME COUNT (one compiled model per batch).
  d) Assigns batches to GPUS, longest-first onto the least-loaded GPU (LPT).
  e) Spawns one worker process per GPU, each pinned with CUDA_VISIBLE_DEVICES.
     One row CSV is written per lesion as it finishes, so the script is
     interruptible and resumable; rerunning skips what is done.

JOINT DELAY IN THE BATCHED SAMPLER
  bayes_batched handles delay_mode="joint" natively: prepare_lesion leaves the
  AIF unshifted, and the compiled forward model shifts it by each draw's
  tau_delay (_forward_kwargs sets delay=True). Every lesion's data carries its
  own RAW AIF. This is the configuration check_15 tested (FXL, i.e. plain TCXM,
  with a joint delay): at production length batched and sequential posteriors
  were indistinguishable from two seeds of the same method.

RUN
    cd ~/DCE_MRI_TK_Fit
    PYTHONPATH=src python scripts/batch_fit_tcxm_delay.py --dry-run   # plan only
    PYTHONPATH=src python scripts/batch_fit_tcxm_delay.py             # fit

  For the cohort FIGURES afterwards, run fit_lesions_bayes.py with
  RESUME_DIR = OUT_DIR: every lesion already has a row, so it fits nothing and
  goes straight to the cohort table and group plots.
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
SRC_SCRIPT = os.path.expanduser("~/DCE_MRI_TK_Fit/scripts/fit_lesions_bayes_vs_NLLS.py")

# the run whose finished subjects are skipped, and where new results go.
# OUT_DIR = DONE_RUN_DIR resumes that run in place.
DONE_RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
                "run_grasp_8spf_20261013_180711_allflatprior")
OUT_DIR = DONE_RUN_DIR

# run times for the grouping: a per-lesion DIFFICULTY proxy, not a prediction.
# None -> DONE_RUN_DIR's own cohort CSV. If that file does not exist (a fresh
# run), every lesion gets the same weight and grouping is by frame count only.
RUNTIME_RUN_DIR = None
RUNTIME_COL = "total_seconds"

BATCH_SIZE = 8
GPUS = (0, 1)
RETRY_FAILED = True        # True: a row whose `error` is set is refitted
MAX_BATCHES = None         # cap the number of batches (testing); None = all
VALIDATE_SIDS = ()         # e.g. ("FMB003", "FMB010"): refit these lesions
                           # SEQUENTIALLY after the batches, into
                           # OUT_DIR/validate_sequential/
PROGRESS_EVERY_S = 60


# =============================================================================
# the source script
# =============================================================================
REQUIRED = ("prepare_inputs", "finish_one", "fit_one", "make_bayes_config",
            "select_largest_lesion", "find_dce", "PARAMS", "MODEL_TAG",
            "DELAY_MODE", "DELAY_BOUNDS", "TCXM", "cfg_fit", "TUMOR_CSV", "ONLY",
            "N_TUNE", "N_DRAWS", "N_CHAINS", "SEED", "PRIOR_CHOICES", "RECON")


def load_src(path=SRC_SCRIPT):
    """Import the fitting script as a module without running its main()."""
    import importlib.util
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SRC_SCRIPT not found: {path}\nPoint SRC_SCRIPT at your "
            f"TCXM + joint-delay fitting script (fit_lesions_bayes.py).")
    spec = importlib.util.spec_from_file_location("tcxm_delay_src", path)
    M = importlib.util.module_from_spec(spec)
    sys.modules["tcxm_delay_src"] = M
    spec.loader.exec_module(M)          # safe: its work is behind __main__
    missing = [n for n in REQUIRED if not hasattr(M, n)]
    if missing:
        raise AttributeError(
            f"{path} has no {missing} -- it is either not the TCXM + delay "
            f"script, or the version from before prepare_inputs / finish_one "
            f"were split out of fit_one")
    if M.DELAY_MODE != "joint":
        raise ValueError(f"{path} has DELAY_MODE = {M.DELAY_MODE!r}; this driver "
                         f"is for the joint-delay fit")
    if getattr(M, "WX_MODEL", None) is not None:
        raise ValueError(f"{path} has WX_MODEL = {M.WX_MODEL!r}; this driver is "
                         f"for plain TCXM + delay")
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
    """{sid: seconds} from a run's cohort CSV. Raises if there is none."""
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


def runtimes_or_flat(run_dir, col=RUNTIME_COL):
    """load_runtimes, or ({}, None, None) when the run has no cohort CSV yet --
    a fresh run has no history to rank lesions by, which is not an error."""
    try:
        return load_runtimes(run_dir, col)
    except (FileNotFoundError, ValueError) as e:
        print(f"  no run times ({e}); every lesion gets the same weight")
        return {}, None, None


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

    rt, rt_path, rt_col = runtimes_or_flat(RUNTIME_RUN_DIR or DONE_RUN_DIR)
    med = float(np.median(list(rt.values()))) if rt else 1.0

    if verbose:
        print(f"cohort            {len(cohort)} subjects"
              + (f"  (ONLY={M.ONLY})" if M.ONLY else ""))
        print(f"already in {os.path.basename(OUT_DIR.rstrip('/'))}: {len(skipped)} "
              f"({sum(v == 'ok' for v in skipped.values())} fitted, "
              f"{sum(v == 'error' for v in skipped.values())} error rows kept)")
        n_retry = sum(v == 'failed' for v in status.values())
        if n_retry:
            print(f"  refitting {n_retry} subject(s) whose row records an error "
                  f"(RETRY_FAILED=True)")
        print(f"to fit            {len(todo)}")
        if rt_path:
            print(f"run times from    {rt_path}")
            print(f"                  column {rt_col!r}, median {med / 60:.1f} min; "
                  f"missing -> median")

    problems, rows = [], []
    for sid, dx in todo:
        try:
            nf = n_frames_of(M, sid)
        except Exception as e:
            problems.append((sid, repr(e)))
            continue
        rows.append(dict(sid=sid, dx=dx, n_frames=nf,
                         runtime_s=rt.get(sid, med), runtime_known=sid in rt))
    if problems and verbose:
        print(f"\n{len(problems)} subject(s) cannot be planned (no DCE / no mask):")
        for sid, e in problems:
            print(f"  {sid}: {e}")
        print("  these are left for the sequential script to run and record an "
              "error row")

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
                cost_s=max(c["runtime_s"] for c in chunk)))
    batches.sort(key=lambda b: -b["cost_s"])          # longest first
    if MAX_BATCHES is not None:
        batches = batches[:MAX_BATCHES]

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
        if rt:
            print("\nper-GPU queued difficulty (sum of per-batch slowest run time):")
            for g in GPUS:
                print(f"  GPU {g}: {load[g] / 60:6.1f} min over "
                      f"{sum(1 for b in batches if b['gpu'] == g)} batch(es)")
            print("These rank lesions; they do not predict this run's wall time. "
                  "Watch the first\nfinished batch for a real rate.")
    return batches, skipped, problems


# =============================================================================
# one lesion: the fitting script's own two halves
# =============================================================================
def prepare_inputs(M, sid, dx, outdir):
    """The fitting script's prepare_inputs, unchanged."""
    return M.prepare_inputs(sid, dx, outdir)


def finish_one(M, inp, res, fit_s, outdir, batch_meta):
    """The fitting script's finish_one, plus the batch's provenance columns.

    fit_seconds in a batched row is the batch's wall time divided by its size
    -- a SHARE, not a sequential fit time -- so it is not comparable with a
    sequential run's fit_seconds and must not be reused as a run-time input
    for grouping.
    """
    row = M.finish_one(inp, res, fit_s, outdir)
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
    print(f"[gpu {gpu}] model {M.MODEL_TAG}: {M.N_TUNE} tune + {M.N_DRAWS} draws, "
          f"{M.N_CHAINS} chains, seed {M.SEED}")
    print(f"[gpu {gpu}] {len(mine)} batch(es), "
          f"{sum(len(b['sids']) for b in mine)} lesion(s)", flush=True)

    for b in mine:
        # Skip a lesion only by the SAME rule the planner used, re-read now so
        # an interrupted earlier run is resumed rather than repeated.
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
                # RAW aif: with delay_mode="joint" prepare_lesion does not
                # shift it; the compiled model shifts it by each draw's tau
                p = bb.prepare_lesion(
                    inp["St"], inp["aif"], inp["t"], M.TCXM, inp["acq_s"],
                    inp["subject"], M.cfg_fit, inp["cfg_b"],
                    wx_model=None, aif_cfg=None, relax_cfg=None, point_fit=None)
                inputs.append(inp)
                prepared.append(p)
                print(f"[gpu {gpu}]   {sid}: prepared ({inp['n_vox']} voxels, "
                      f"baseline 0-{inp['nb'].stop - 1})", flush=True)
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
                wx_model=None, aif_cfg=None, relax_cfg=None, verbose=True)
        except Exception as e:
            print(f"[gpu {gpu}]   !! BATCH FIT FAILED: {e!r}\n"
                  f"[gpu {gpu}]   falling back to one lesion at a time",
                  flush=True)
            how, results = "sequential_fallback", []
            for inp in inputs:
                try:
                    # the same call the fitting script's fit_one makes
                    results.append(fit_curve_bayes(
                        inp["St"], inp["aif"], inp["t"], M.TCXM, inp["acq_s"],
                        inp["subject"], M.cfg_fit, inp["cfg_b"]))
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
                         batch_frames=b["n_frames"])
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
                      f"ESS {row['ess_min']:.0f}  "
                      f"MCSE/CrI {row['mcse_median_rel_ci_max']:.3f}  "
                      f"div {row['n_divergences']}  "
                      f"QC {'PASS' if row['qc_pass'] else 'FAIL'}  "
                      f"tau_delay={row['tau_delay_median']:.2f} s  "
                      f"ve={row['ve_median']:.3f}  "
                      f"Ktrans={row['Ktrans_median'] * 60:.3f}/min",
                      flush=True)
            except Exception as e:
                print(f"[gpu {gpu}]   !! {sid} FAILED after fitting: {e!r}",
                      flush=True)
                row = dict(subject_id=sid, diagnosis=inp["dx"], recon=M.RECON,
                           model=M.MODEL_TAG, delay_mode=M.DELAY_MODE,
                           error=repr(e), **bmeta)
            row["finish_seconds"] = time.time() - t1
            # this lesion's own CPU work plus its share of the batch's GPU time
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
    """Refit the named lesions ONE AT A TIME into outdir/validate_sequential,
    through the fitting script's own fit_one -- the reference the batched
    posterior is compared with. Same seed, so the comparison is of the two
    code paths; compare medians, SDs and R-hat, not draw by draw."""
    import pandas as pd
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
            row = M.fit_one(sid, "?", vdir)
            row["batch_how"] = "sequential_validation"
        except Exception as e:
            print(f"  !! {sid} validation FAILED: {e!r}", flush=True)
            row = dict(subject_id=sid, error=repr(e))
        row["total_seconds"] = time.time() - t0
        pd.DataFrame([row]).to_csv(row_path, index=False)
    print(f"\nsequential validation fits -> {vdir}", flush=True)


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
    print("batch_fit_tcxm_delay -- planning")
    print("=" * 78)
    M = load_src()
    print(f"source script     {SRC_SCRIPT}")
    print(f"model             {M.MODEL_TAG}   sampled {M.PARAMS}")
    print(f"sampler           {M.N_TUNE} tune + {M.N_DRAWS} draws, "
          f"{M.N_CHAINS} chains, seed {M.SEED}")
    print(f"delay             joint, uniform on {M.DELAY_BOUNDS} s")
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
                       runtime_run_dir=RUNTIME_RUN_DIR or DONE_RUN_DIR,
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
        os.environ.pop("JAX_PLATFORMS", None)
        validate_sequential(M, VALIDATE_SIDS, OUT_DIR)
    print(f"\nFor the cohort FIGURES, run fit_lesions_bayes.py with "
          f"RESUME_DIR = {OUT_DIR!r}:\nevery lesion has a row, so it fits "
          f"nothing and goes straight to the group plots.")


if __name__ == "__main__":
    main()