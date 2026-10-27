"""
bench_batched_gpu.py  --  STEP 3 of the batched-fitting plan
============================================================
How much faster per lesion is a batched NUTS fit on this GPU, and what
limits it?

For each batch size B in BATCH_SIZES, fits B lesions together with
bayes_batched.fit_curve_bayes_batched (short chains -- this measures speed,
not posteriors) and reports:

  compile_s          XLA compile for that batch shape (paid once per shape)
  ms_per_step        wall time of one NUTS step for the whole batch
  ms_per_step_lesion the same divided by B -- the number that should fall
  speedup            ms_per_step_lesion at B=1 / at this B
  leapfrog mean      mean leapfrog steps per lane per NUTS step
  leapfrog batchmax  mean over steps of the batch's LONGEST trajectory --
                     what every lane actually waits for
  lockstep_cost      batchmax / (mean over lesions of that lesion's own
                     per-step max over its chains) -- the extra waiting
                     caused by putting different lesions in one batch
                     (1.0 = none)
  projected_h        hours for N_COHORT lesions at the production chain
                     length, from ms_per_step (excludes CPU post-processing)

Lesions come from an existing fit_lesions_bayes.py run directory: each
<sid>_draws.npz there holds St, aif and t. MODE='NXL' fixes each lesion's
delay at the median of its FXL tau_delay posterior from the same file,
i.e. the NXL-at-FXL-delay setup.

Run on ONE GPU, e.g.
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 PYTHONPATH=src \
        python scripts/bench_batched_gpu.py
"""
import os
import glob
import time
os.environ.setdefault("JAX_ENABLE_X64", "1")
from datetime import datetime

import numpy as np
import pandas as pd
import jax

from dce_mri.config import AcquisitionConfig, SubjectConfig, FittingConfig, AifConfig
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL
from dce_mri.bayes import BayesConfig
from dce_mri import bayes_batched as bb

# =============================================================================
# settings
# =============================================================================
RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
           "run_grasp_8spf_2026100_beta&logflat")      # has <sid>_draws.npz
OUT_DIR = os.path.expanduser("~/bench_batched_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

MODE = "FXL"                    # "FXL" (joint delay) | "NXL" (fixed delay from FXL)
BATCH_SIZES = (1, 4, 16, 64)
N_TUNE = N_DRAWS = 300          # short: speed only. Steps per batch = 600
N_CHAINS = 16                   # as in production
PROD_STEPS = 3000 + 3000        # production n_tune + n_draws, for the projection
N_COHORT = 174

NBASE = slice(0, 7)
acq = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=NBASE)
subject = SubjectConfig(subject_id="bench", T10=1530.0, B1=1.0)
cfg_fit = FittingConfig(n_lhs=200, n_top=10, fit_space="signal")
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)        # NXL only
PRIOR_CHOICES = {"rates": "loguniform", "volfrac": "beta"}
DELAY_BOUNDS = (0.0, 25.0)

# =============================================================================


def load_lesions(run_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*_draws.npz"))):
        z = np.load(f, allow_pickle=True)
        names = list(z["param_names"])
        delay = float(np.median(z["chains"][..., names.index("tau_delay")]))
        out.append(dict(sid=os.path.basename(f).split("_draws")[0],
                        St=np.asarray(z["St"], float), aif=np.asarray(z["aif"], float),
                        t=np.asarray(z["t"], float), delay=delay))
    if not out:
        raise FileNotFoundError(f"no *_draws.npz in {run_dir}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"devices: {jax.devices()}   x64: {jax.config.jax_enable_x64}")
    lesions = load_lesions(RUN_DIR)
    t = lesions[0]["t"]
    if not all(np.allclose(L["t"], t) for L in lesions):
        raise ValueError("lesions have different time grids")
    print(f"{len(lesions)} lesions from {RUN_DIR}; MODE={MODE}")

    cb = BayesConfig(prior_choices=PRIOR_CHOICES, n_draws=N_DRAWS, n_tune=N_TUNE,
                     n_chains=N_CHAINS, noise="gauss_marginalized_sigma", seed=1,
                     backend="blackjax", step="nuts", dense_mass_matrix=True,
                     delay_mode=("joint" if MODE == "FXL" else "fixed"),
                     delay_bounds=DELAY_BOUNDS, progressbar=False)
    wx_kw = dict(wx_model=NXL, aif_cfg=aif_cfg) if MODE == "NXL" else {}

    # CPU preparation (NLLS anchors) once, for the largest batch; reused below
    n_max = max(BATCH_SIZES)
    pick = [lesions[i % len(lesions)] for i in range(n_max)]   # cycles if < n_max
    t0 = time.time()
    prepared = [bb.prepare_lesion(L["St"], L["aif"], t, TCXM, acq, subject, cfg_fit,
                                  BayesConfig(**{**cb.__dict__, "delay_fixed": L["delay"]}),
                                  **wx_kw) for L in pick]
    print(f"prepared {n_max} lesions on CPU in {time.time() - t0:.0f} s")

    rows = []
    base = None
    for B in BATCH_SIZES:
        print(f"\n--- batch size {B} ---", flush=True)
        res = bb.fit_curve_bayes_batched(None, None, t, TCXM, acq, subject, cfg_fit, cb,
                                         prepared=prepared[:B], **wx_kw)
        m = res[0].meta["batch"]
        steps = N_TUNE + N_DRAWS
        ms_step = 1e3 * m["sample_s"] / steps
        ms_lesion = ms_step / B
        base = base or ms_lesion
        n_batches = int(np.ceil(N_COHORT / B))
        proj_h = (ms_step / 1e3 * PROD_STEPS + m["compile_s"]) * n_batches / 3600
        rhat_max = max(max(r.rhat.values()) for r in res)
        rows.append(dict(
            batch=B, compile_s=m["compile_s"], sample_s=m["sample_s"],
            ms_per_step=ms_step, ms_per_step_lesion=ms_lesion,
            speedup=base / ms_lesion,
            leapfrog_mean=m["mean_leapfrog"],
            leapfrog_batchmax=m["mean_of_batch_max"],
            lockstep_cost=m["mean_of_batch_max"] / m["mean_of_lesion_max"],
            projected_h=proj_h, rhat_max_short_run=rhat_max))
        print(pd.DataFrame([rows[-1]]).round(3).to_string(index=False), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"bench_{MODE}.csv"), index=False)
    print("\n" + "=" * 100)
    print(df.round(3).to_string(index=False))
    print("=" * 100)
    print(f"projected_h = hours for {N_COHORT} lesions at {PROD_STEPS} steps, "
          f"GPU sampling + compile only.  Saved: {OUT_DIR}")
    print(f"(rhat_max_short_run is from {N_TUNE}+{N_DRAWS}-step chains: expected to "
          f"be poor, not a convergence check)")


if __name__ == "__main__":
    main()
