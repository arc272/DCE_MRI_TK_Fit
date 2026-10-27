#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refit_problem_lesions.py
========================
Refit, at a LONGER chain length, only the lesions of a finished stage-1 run
that are worth refitting:

  a) never fitted        -- no row in DONE_RUN_DIR/rows
  b) stage-1 error       -- the row records a crash rather than a fit
  c) stage-1 QC failure  -- qc_pass False (R-hat / ESS / divergences / MCSE)
  d) weight-starved      -- stage-2 Kish ESS below KISH_MIN, i.e. the lesion's
                            saved draws cannot represent its contribution to
                            the hierarchical fit however long stage 2 runs

(a)-(c) come from DONE_RUN_DIR/rows/*.csv; (d) from a stage-2 output's
stage2_lesions.csv (column kish_ess_fitted). A lesion can be selected for
several reasons and all of them are recorded.

It does NOT reimplement anything. It imports batch_fit_wx_fixedtau.py
(DRIVER_SCRIPT) and reuses its prepare_inputs / finish_one / run_worker /
cohort_table, which in turn import your fitting script. The only things
replaced are WHICH lesions are planned and the sampler length.

WHY MORE DRAWS FIXES (d) AND NOT MUCH ELSE
  Stage 2 proposes theta_i uniformly from lesion i's SAVED draws and corrects
  by p(theta|B,Sigma)/p_1(theta). Kish ESS counts how many of those draws
  carry weight once the population model has reweighted them. It is a property
  of the saved draws and the fitted population, NOT of how long stage 2 runs:
  lengthening stage 2 raises n_distinct and leaves Kish ESS where it was. The
  only lever is more (or better-mixed) stage-1 draws, which is what this does.

  Measured on run_20261021_160427, median Kish ESS grew as K^0.88 (93.5 at
  K=1500, 196.3 at K=3000, 505.0 at K=10000), and the count below 30 fell
  19 -> 12 -> 7. So K is already the dominant lever; doubling each lesion's
  chain length raises the K that lesion can supply and de-correlates it.

WHAT THIS CHANGES ABOUT YOUR COHORT, AND WHY IT NEEDS SAYING IN THE WRITE-UP
  1. Lesions will no longer have equal stage-1 effort. Refit lesions get
     8000+8000 x 16; the rest keep what they had. Stage 2 thins everyone to
     the same K, so this is not a bias in the arithmetic, but it IS an
     unequal-effort design and belongs in the methods.
  2. Selecting on Kish ESS is data-dependent. A lesion is weight-starved
     largely BECAUSE its posterior sits far from the population mean, so this
     preferentially buys resolution for the lesions that disagree with the
     population. That direction is toward the exact posterior, not away from
     it -- finite K is an approximation error and this reduces it where it is
     largest -- but it is not symmetric effort. The control is to rerun with
     every lesion at 8000+8000 and check the population numbers do not move;
     set SELECT_ALL = True below to do exactly that.

SAFETY: THIS DOES NOT OVERWRITE YOUR RUN
  OUT_DIR defaults to DONE_RUN_DIR/refit_<timestamp>/, a NEW directory. The
  original rows, draws and figures are untouched. When you are satisfied:

      python scripts/refit_problem_lesions.py --merge --refit-dir <that dir>

  which moves each refit lesion's ORIGINAL row/npz/figures into
  DONE_RUN_DIR/preserved_before_refit_<stamp>/, copies the new ones in, and
  rebuilds lesion_bayes_results.csv. Add --dry-run to see the moves first.
  Nothing is deleted, so the merge is reversible by hand.

RUN
    cd ~/DCE_MRI_TK_Fit
    PYTHONPATH=src python scripts/refit_problem_lesions.py --dry-run   # plan
    PYTHONPATH=src python scripts/refit_problem_lesions.py             # fit
    PYTHONPATH=src python scripts/refit_problem_lesions.py --merge --refit-dir ... --dry-run
    PYTHONPATH=src python scripts/refit_problem_lesions.py --merge --refit-dir ...

  Then rerun stage 2 with RUN_DIR = DONE_RUN_DIR.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# =============================================================================
# settings
# =============================================================================
DRIVER_SCRIPT = os.path.expanduser(
    "~/DCE_MRI_TK_Fit/scripts/batch_fit_wx_fixedtau.py")

DONE_RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
                "run_20261021_160427")
OUT_DIR = None                 # None -> DONE_RUN_DIR/refit_<timestamp>

# stage-2 output supplying Kish ESS. None -> newest stage2_* under
# DONE_RUN_DIR that contains stage2_lesions.csv.
STAGE2_DIR = None
KISH_MIN = 50.0                # None disables the Kish rule entirely

# which rules to apply
SELECT_NO_ROW = True           # (a) never fitted
SELECT_ERROR = True            # (b) row records a crash
SELECT_QC_FAIL = True          # (c) qc_pass False
SELECT_LOW_KISH = True         # (d) stage-2 Kish ESS < KISH_MIN
SELECT_ALL = False             # True -> refit the WHOLE cohort at the new
                               # length (the equal-effort control described
                               # above). Overrides (a)-(d).
EXTRA_SIDS = ()                # always include these, whatever the rules say
SKIP_SIDS = ()                 # never include these

# --- sampler, overriding the source fitting script -------------------------
# 6 lesions x 16 chains = 96 lanes, down from the driver's 8 x 16 = 128,
# because each lane now holds twice the draws. If a batch runs out of GPU
# memory, BATCH_SIZE is the knob to turn, not N_CHAINS.
N_TUNE = 10000
N_DRAWS = 10000
N_CHAINS = 16
SEED_OVERRIDE = None           # None keeps the source script's seed, so the
                               # refit differs from the original only by
                               # chain length. An int makes it a fresh draw.

BATCH_SIZE = 6
GPUS = (0, 1)
MAX_BATCHES = None
PROGRESS_EVERY_S = 120


# =============================================================================
# the driver script (which itself imports your fitting script)
# =============================================================================
def load_driver(path=DRIVER_SCRIPT):
    import importlib.util
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DRIVER_SCRIPT not found: {path}\nPoint DRIVER_SCRIPT at your "
            f"batch_fit_wx_fixedtau.py.")
    spec = importlib.util.spec_from_file_location("wx_driver", path)
    D = importlib.util.module_from_spec(spec)
    sys.modules["wx_driver"] = D
    spec.loader.exec_module(D)        # safe: its work is behind __main__
    for need in ("load_src", "prepare_inputs", "finish_one", "run_worker",
                 "cohort_table", "n_frames_of", "load_runtimes"):
        if not hasattr(D, need):
            raise AttributeError(f"{path} has no {need!r} -- is DRIVER_SCRIPT "
                                 f"the right script?")
    return D


def overrides():
    ov = dict(N_TUNE=int(N_TUNE), N_DRAWS=int(N_DRAWS), N_CHAINS=int(N_CHAINS))
    if SEED_OVERRIDE is not None:
        ov["SEED"] = int(SEED_OVERRIDE)
    return ov


def apply_overrides(M, ov, verbose=True):
    """Set the sampler length on the imported fitting script.

    prepare_inputs builds BayesConfig from these module attributes, so setting
    them here is what makes the refit longer. Anything not listed -- priors,
    bounds, tau_b, the QC thresholds -- is deliberately left alone, so a refit
    lesion is judged by the same rules as the rest of the cohort.
    """
    for k, v in ov.items():
        if not hasattr(M, k):
            raise AttributeError(f"the fitting script has no {k!r} to override")
        old = getattr(M, k)
        setattr(M, k, v)
        if verbose and old != v:
            print(f"    override {k}: {old} -> {v}")
    return M


# =============================================================================
# selection
# =============================================================================
def stage1_status(run_dir):
    """{sid: dict(error, qc_pass, ess_min, rhat_max, has_npz)} from rows/*.csv."""
    import numpy as np
    import pandas as pd
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "rows", "*.csv"))):
        sid = os.path.basename(f)[:-4]
        try:
            d = pd.read_csv(f)
        except Exception as e:
            out[sid] = dict(error=f"unreadable row ({e!r})", qc_pass=False,
                            ess_min=np.nan, rhat_max=np.nan, has_npz=False)
            continue

        def col(name, default=np.nan):
            if name in d.columns and len(d):
                v = d[name].iloc[0]
                return default if pd.isna(v) else v
            return default

        err = col("error", "")
        err = "" if not isinstance(err, str) else err.strip()
        qc = str(col("qc_pass", "")).strip().lower() == "true"
        out[sid] = dict(error=err, qc_pass=qc,
                        ess_min=float(col("ess_min", np.nan)),
                        rhat_max=float(col("rhat_max", np.nan)),
                        has_npz=os.path.exists(
                            os.path.join(run_dir, f"{sid}_draws.npz")))
    return out


def find_stage2_dir(done_dir, explicit=STAGE2_DIR):
    if explicit:
        p = os.path.expanduser(explicit)
        if not os.path.exists(os.path.join(p, "stage2_lesions.csv")):
            raise FileNotFoundError(f"no stage2_lesions.csv in {p}")
        return p
    cands = [d for d in sorted(glob.glob(os.path.join(done_dir, "stage2_*")))
             if os.path.exists(os.path.join(d, "stage2_lesions.csv"))]
    if not cands:
        return None
    return cands[-1]              # names are stage2_<YYYYmmdd_HHMMSS>, so sorted = oldest first


def read_kish(stage2_dir):
    """{sid: kish_ess_fitted} from a stage-2 run."""
    import pandas as pd
    path = os.path.join(stage2_dir, "stage2_lesions.csv")
    d = pd.read_csv(path)
    if "kish_ess_fitted" not in d.columns:
        raise ValueError(f"{path} has no kish_ess_fitted column")
    return {str(r["subject_id"]).strip(): float(r["kish_ess_fitted"])
            for _, r in d.iterrows() if pd.notna(r["kish_ess_fitted"])}, path


def select_lesions(D, M, done_dir, out_dir, verbose=True):
    """-> (list of dicts with sid/dx/reasons, context dict for printing)."""
    import numpy as np
    import pandas as pd

    tumors = pd.read_csv(M.TUMOR_CSV)
    tumors.columns = [c.strip() for c in tumors.columns]
    tumors["Patient_ID"] = tumors["Patient_ID"].str.strip()
    tumors["Diagnosis"] = tumors["Diagnosis"].str.strip().str.upper()
    if M.ONLY:
        tumors = tumors[tumors["Patient_ID"].isin(M.ONLY)]
    cohort = [(r["Patient_ID"], r["Diagnosis"]) for _, r in tumors.iterrows()]

    st = stage1_status(done_dir)
    s2dir = find_stage2_dir(done_dir)
    kish, kish_path = ({}, None)
    if SELECT_LOW_KISH and KISH_MIN is not None:
        if s2dir is None:
            print("    !! SELECT_LOW_KISH is on but no stage-2 output was found "
                  "under\n       " + done_dir +
                  "\n       -> the Kish rule is NOT applied. Set STAGE2_DIR, or "
                  "run stage 2 first.")
        else:
            kish, kish_path = read_kish(s2dir)

    # already refit in this output directory? then leave it alone
    already = stage1_status(out_dir) if os.path.isdir(out_dir) else {}

    picked = []
    for sid, dx in cohort:
        if sid in SKIP_SIDS:
            continue
        prev = already.get(sid)
        if prev is not None and not prev["error"]:
            continue                       # resume: already refit successfully

        reasons = []
        if SELECT_ALL:
            reasons.append("SELECT_ALL")
        else:
            row = st.get(sid)
            if row is None:
                if SELECT_NO_ROW:
                    reasons.append("never fitted")
            elif row["error"]:
                if SELECT_ERROR:
                    reasons.append("stage-1 error")
            else:
                if SELECT_QC_FAIL and not row["qc_pass"]:
                    bits = []
                    if np.isfinite(row["ess_min"]):
                        bits.append(f"ESS {row['ess_min']:.0f}")
                    if np.isfinite(row["rhat_max"]):
                        bits.append(f"R-hat {row['rhat_max']:.3f}")
                    reasons.append("qc_pass False"
                                   + (f" ({', '.join(bits)})" if bits else ""))
                if row["has_npz"] is False and SELECT_NO_ROW:
                    reasons.append("row but no _draws.npz")
            if sid in kish and kish[sid] < KISH_MIN:
                reasons.append(f"Kish {kish[sid]:.1f} < {KISH_MIN:g}")
        if sid in EXTRA_SIDS and not reasons:
            reasons.append("EXTRA_SIDS")
        if reasons:
            picked.append(dict(sid=sid, dx=dx, reasons=reasons,
                               kish=kish.get(sid, float("nan"))))

    ctx = dict(n_cohort=len(cohort), n_rows=len(st), stage2_dir=s2dir,
               kish_path=kish_path, n_kish=len(kish), n_already=len(already))
    if verbose:
        print(f"    cohort                      {len(cohort)}")
        print(f"    rows in the finished run    {len(st)}")
        if kish_path:
            print(f"    Kish ESS from               {kish_path}")
            print(f"                                {len(kish)} lesions, "
                  f"{sum(v < KISH_MIN for v in kish.values())} below "
                  f"{KISH_MIN:g}")
        if already:
            n_ok = sum(1 for v in already.values() if not v["error"])
            print(f"    already refit in OUT_DIR    {n_ok} (skipped)")
        print(f"    selected for refit          {len(picked)}")
    return picked, ctx


# =============================================================================
# planning
# =============================================================================
def build_plan(D, M, picked, verbose=True):
    import numpy as np

    rt_dir = M.TAU_FIXED_FROM_RUN
    rt, rt_path, rt_col = D.load_runtimes(rt_dir)
    med = float(np.median(list(rt.values())))

    problems, rows = [], []
    for p in picked:
        try:
            nf = D.n_frames_of(M, p["sid"])
        except Exception as e:
            problems.append((p["sid"], repr(e)))
            continue
        rows.append(dict(sid=p["sid"], dx=p["dx"], n_frames=nf,
                         reasons=p["reasons"], kish=p["kish"],
                         runtime_s=rt.get(p["sid"], med)))
    if problems and verbose:
        print(f"\n    {len(problems)} lesion(s) cannot be planned "
              f"(no DCE / no mask):")
        for sid, e in problems:
            print(f"      {sid}: {e}")

    # frame count first (one compiled model per batch), then similar difficulty
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
    batches.sort(key=lambda b: -b["cost_s"])
    if MAX_BATCHES is not None:
        batches = batches[:MAX_BATCHES]

    load = {g: 0.0 for g in GPUS}
    for b in batches:
        g = min(load, key=load.get)
        b["gpu"] = g
        load[g] += b["cost_s"]
    for i, b in enumerate(batches):
        b["index"] = i

    if verbose and rows:
        print(f"\n    why each lesion was selected:")
        print(f"      {'lesion':<10}{'dx':<3}{'Kish':>8}  reasons")
        for r in sorted(rows, key=lambda r: (np.nan_to_num(r["kish"], nan=1e9))):
            k = "-" if not np.isfinite(r["kish"]) else f"{r['kish']:.1f}"
            print(f"      {r['sid']:<10}{r['dx']:<3}{k:>8}  "
                  f"{'; '.join(r['reasons'])}")
    if verbose and batches:
        print(f"\n    {len(batches)} batch(es) of up to {BATCH_SIZE} "
              f"({BATCH_SIZE} x {N_CHAINS} = {BATCH_SIZE * N_CHAINS} lanes), "
              f"{len({b['n_frames'] for b in batches})} frame count(s)")
        hdr = f"      {'#':>3} {'gpu':>3} {'frames':>6} {'n':>2} {'slowest':>9}  lesions"
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))
        for b in sorted(batches, key=lambda b: (b["gpu"], -b["cost_s"])):
            print(f"      {b['index']:>3} {b['gpu']:>3} {b['n_frames']:>6} "
                  f"{len(b['sids']):>2} {b['cost_s'] / 60:>7.1f}m  "
                  + " ".join(b["sids"]))
        print(f"\n    per-GPU queued difficulty (FXL proxy, ranks lesions only):")
        for g in GPUS:
            print(f"      GPU {g}: {load[g] / 60:6.1f} min over "
                  f"{sum(1 for b in batches if b['gpu'] == g)} batch(es)")
        print(f"    This refit is {N_DRAWS + N_TUNE} iterations against the "
              f"original run's, so expect\n    roughly "
              f"{(N_DRAWS + N_TUNE) / (4000 + 4000):.1f}x that run's per-lesion "
              f"GPU time at the same batch size,\n    and a further "
              f"{8 / BATCH_SIZE:.2f}x because the batch is {BATCH_SIZE} rather "
              f"than 8. Watch the first batch.")
    return batches, problems


# =============================================================================
# merge
# =============================================================================
def merge_refit(D, done_dir, refit_dir, dry_run=False):
    """Move originals aside, copy the refit in, rebuild the cohort table."""
    import pandas as pd
    st = stage1_status(refit_dir)
    good = sorted(s for s, v in st.items() if not v["error"])
    bad = sorted(s for s, v in st.items() if v["error"])
    if not good:
        print("nothing to merge: no error-free rows in " + refit_dir)
        return
    stamp = os.path.basename(refit_dir.rstrip("/")).replace("refit_", "")
    keep = os.path.join(done_dir, f"preserved_before_refit_{stamp}")
    print(f"merging {len(good)} refit lesion(s) into {done_dir}")
    if bad:
        print(f"  leaving {len(bad)} error row(s) behind: {bad}")
    print(f"  originals -> {keep}")
    if not dry_run:
        os.makedirs(os.path.join(keep, "rows"), exist_ok=True)

    n_moved = n_copied = 0
    for sid in good:
        # 1. move the original row and any per-lesion file out of the way
        orig_row = os.path.join(done_dir, "rows", f"{sid}.csv")
        if os.path.exists(orig_row):
            print(f"    move  rows/{sid}.csv")
            if not dry_run:
                shutil.move(orig_row, os.path.join(keep, "rows", f"{sid}.csv"))
            n_moved += 1
        for src in sorted(glob.glob(os.path.join(done_dir, f"{sid}*"))):
            if os.path.isdir(src):
                continue
            print(f"    move  {os.path.basename(src)}")
            if not dry_run:
                shutil.move(src, os.path.join(keep, os.path.basename(src)))
            n_moved += 1
        # 2. copy the refit in
        new_row = os.path.join(refit_dir, "rows", f"{sid}.csv")
        if os.path.exists(new_row):
            if not dry_run:
                shutil.copy2(new_row, os.path.join(done_dir, "rows", f"{sid}.csv"))
            n_copied += 1
        for src in sorted(glob.glob(os.path.join(refit_dir, f"{sid}*"))):
            if os.path.isdir(src):
                continue
            if not dry_run:
                shutil.copy2(src, os.path.join(done_dir, os.path.basename(src)))
            n_copied += 1

    print(f"  {n_moved} file(s) moved aside, {n_copied} copied in")
    if dry_run:
        print("\n--dry-run: nothing changed. Rerun without --dry-run to merge.")
        return
    D.cohort_table(done_dir)
    print(f"\nMerged. The originals are in {keep} -- nothing was deleted.\n"
          f"Now rerun stage 2 with RUN_DIR = {done_dir!r}.\n"
          f"Note for the methods section: {len(good)} lesion(s) now have "
          f"{N_TUNE}+{N_DRAWS} x {N_CHAINS}\nwhile the rest have the original "
          f"chain length.")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="plan", choices=("plan", "worker"))
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--plan")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="merge a finished refit directory into DONE_RUN_DIR")
    ap.add_argument("--refit-dir", help="the refit directory to merge")
    a = ap.parse_args()

    # ---- worker: reuse the driver's loop, with the sampler overridden -------
    if a.role == "worker":
        D = load_driver()
        with open(a.plan) as fh:
            ov = json.load(fh).get("overrides", {})
        _orig_load_src = D.load_src

        def load_src_with_overrides(path=None, *args, **kw):
            M = _orig_load_src(path) if path else _orig_load_src()
            apply_overrides(M, ov, verbose=False)
            return M

        D.load_src = load_src_with_overrides       # run_worker calls this
        return D.run_worker(a.gpu, a.plan)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    D = load_driver()

    if a.merge:
        refit = a.refit_dir
        if not refit:
            cands = sorted(glob.glob(os.path.join(DONE_RUN_DIR, "refit_*")))
            if not cands:
                raise SystemExit("--merge needs --refit-dir (none found "
                                 f"under {DONE_RUN_DIR})")
            refit = cands[-1]
            print(f"--refit-dir not given; using the newest: {refit}\n")
        return merge_refit(D, DONE_RUN_DIR, os.path.expanduser(refit), a.dry_run)

    out_dir = OUT_DIR or os.path.join(
        DONE_RUN_DIR, "refit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    print("=" * 78)
    print("refit_problem_lesions -- planning")
    print("=" * 78)
    M = D.load_src()
    print(f"driver script     {DRIVER_SCRIPT}")
    print(f"fitting script    {D.SRC_SCRIPT}")
    print(f"model             {M.MODEL_TAG}   sampled {M.PARAMS}")
    print(f"finished run      {DONE_RUN_DIR}")
    print(f"refit output      {out_dir}   (NEW -- the finished run is untouched)")
    print(f"sampler           originally {M.N_TUNE} tune + {M.N_DRAWS} draws, "
          f"{M.N_CHAINS} chains, seed {M.SEED}")
    apply_overrides(M, overrides())
    print(f"                  now {M.N_TUNE} tune + {M.N_DRAWS} draws, "
          f"{M.N_CHAINS} chains, seed {M.SEED}")
    print(f"water exchange    {M.WX_MODEL.name}, tau_b fixed "
          f"{M.relax_cfg.tau_b_fixed} s, tau_i in "
          f"[{M.relax_cfg.tau_i_lb}, {M.relax_cfg.tau_i_ub}] s")
    print(f"priors            {M.PRIOR_CHOICES}   (unchanged)")
    print(f"batching          {BATCH_SIZE} x {M.N_CHAINS} = "
          f"{BATCH_SIZE * M.N_CHAINS} lanes, GPUs {list(GPUS)}\n")

    print("SELECTION")
    picked, ctx = select_lesions(D, M, DONE_RUN_DIR, out_dir)
    if not picked:
        print("\nnothing selected -- nothing to refit.")
        return
    batches, problems = build_plan(D, M, picked)
    if not batches:
        print("\nnothing plannable.")
        return

    os.makedirs(os.path.join(out_dir, "rows"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_path = os.path.join(out_dir, f"refit_plan_{stamp}.json")
    with open(plan_path, "w") as fh:
        json.dump(dict(created=stamp, kind="refit_problem_lesions",
                       src_script=D.SRC_SCRIPT, driver_script=DRIVER_SCRIPT,
                       out_dir=out_dir, done_run_dir=DONE_RUN_DIR,
                       stage2_dir=ctx["stage2_dir"], kish_min=KISH_MIN,
                       select=dict(no_row=SELECT_NO_ROW, error=SELECT_ERROR,
                                   qc_fail=SELECT_QC_FAIL,
                                   low_kish=SELECT_LOW_KISH, all=SELECT_ALL),
                       overrides=overrides(), batch_size=BATCH_SIZE,
                       gpus=list(GPUS), retry_failed=True,
                       reasons={p["sid"]: p["reasons"] for p in picked},
                       batches=batches, problems=problems), fh, indent=1)
    print(f"\nplan -> {plan_path}")

    if a.dry_run:
        print("\n--dry-run: nothing fitted. Check the selection and the "
              "batches above,\nthen rerun without --dry-run.")
        return

    n_target = sum(len(b["sids"]) for b in batches)
    done0 = len(glob.glob(os.path.join(out_dir, "rows", "*.csv")))
    procs, logs = {}, {}
    for g in GPUS:
        if not any(b["gpu"] == g for b in batches):
            continue
        log_path = os.path.join(out_dir, f"refit_gpu{g}_{stamp}.log")
        logs[g] = log_path
        fh = open(log_path, "w")
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("JAX_PLATFORMS", None)
        procs[g] = (subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--role", "worker",
             "--gpu", str(g), "--plan", plan_path],
            stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
        print(f"GPU {g}: pid {procs[g][0].pid}, log {log_path}")

    print(f"\nrefitting {n_target} lesion(s). Follow a worker with:\n"
          + "\n".join(f"  tail -f {p}" for p in logs.values()))
    t0 = time.time()
    try:
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(PROGRESS_EVERY_S)
            n = len(glob.glob(os.path.join(out_dir, "rows", "*.csv"))) - done0
            el = (time.time() - t0) / 60
            rate = f", {el / n:.1f} min/lesion" if n else ""
            alive = [g for g, (p, _) in procs.items() if p.poll() is None]
            print(f"  [{el:6.1f} min] {n}/{n_target} rows{rate}; "
                  f"GPU(s) running: {alive}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating workers. Finished lesions keep "
              "their rows; rerunning resumes.")
        for p, _ in procs.values():
            p.terminate()
    for g, (p, fh) in procs.items():
        p.wait()
        fh.close()
        print(f"GPU {g}: exit {p.returncode}"
              + ("" if p.returncode == 0 else f"  -- see {logs[g]}"))

    print(f"\ntotal wall time {(time.time() - t0) / 60:.1f} min")
    D.cohort_table(out_dir)
    print(f"\nThe refit lives in {out_dir} and the finished run is unchanged.")
    print("Check the new QC and the refit figures, then merge with:\n"
          f"  python {os.path.basename(__file__)} --merge --refit-dir "
          f"{out_dir} --dry-run\n"
          f"  python {os.path.basename(__file__)} --merge --refit-dir {out_dir}")


if __name__ == "__main__":
    main()