#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:32:02 2026

@author: allisonchen
"""

"""
bench_batched_gpu_same_lesion.py   (v2: warmup and sampling timed separately)
=============================================================================
Why does a batched NUTS step get slower as the batch grows -- longer
trajectories, or a slower GPU iteration -- and what does that mean for run time?

B INDEPENDENT COPIES OF ONE LESION are batched (same data, different seeds
-> different starts and random streams). Nothing differs between copies
except chance, so what is left is exactly the part no re-grouping of lesions
can remove.

Per NUTS step the batch runs as many lock-step leapfrog iterations as its
longest trajectory needs, and every iteration costs some wall time:

    time per step  =  (iterations per step)  x  (time per iteration)

Each factor, at batch size B relative to B = 1, is one cause:

  traj  = mean [longest trajectory in the batch]
          / mean [longest trajectory within one copy's chains]
          The waiting that comes only from taking a maximum over more chains.
  hw    = (ms per lock-step iteration at B) / (the same at B = 1)
          How much slower one iteration gets as lanes are added. ~1 = the
          GPU has spare capacity; > 1 = it is filling up.
  slow  = ms per NUTS step at B / at B = 1  =  traj x hw   (exact)
  speedup_per_copy = B / slow

HOW IT IS TIMED (v1 differenced two runs and gave a negative time; do not
do that)
  * Warmup and sampling are two separate compiled functions, so each is
    timed on its own and its leapfrog counts are its own.
  * Each compiled function is executed once UNTIMED first (one-off kernel
    loading lands there), then the sampling phase is timed 3x on identical
    inputs; the median is used and the spread is reported (ms_spread). A
    large spread means something else is using the GPU.
  * The B = 1 reference is the mean of N_REF independent single copies, so
    it does not hinge on one random trajectory.
  * Warmup is timed on its second execution.

Caveat: "iterations per step" is taken as the longest trajectory in the
batch (BlackJAX num_integration_steps). The real GPU while-loop can run a
little more, because it waits at every tree doubling, so ms/iteration is an
upper bound and hw a slight overestimate. The split is a diagnostic, not an
exact accounting. The run-time projection assumes warmup and sampling
scale linearly with their step counts.

Run on ONE GPU:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 PYTHONPATH=src \
        python scripts/bench_batched_gpu_same_lesion.py
"""
import os
import glob
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("JAX_ENABLE_X64", "1")
from datetime import datetime

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import blackjax

from dce_mri.config import AcquisitionConfig, SubjectConfig, FittingConfig, AifConfig
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL
from dce_mri.bayes import BayesConfig
from dce_mri.bayes_blackjax import build_transform
from dce_mri import bayes_batched as bb

# =============================================================================
# settings
# =============================================================================
RUN_DIR = ("/mnt/labspace/users/allison/fastMRI_data/lesion_bayes_results/"
           "run_grasp_8spf_2026100_beta&logflat")      # has <sid>_draws.npz
OUT_DIR = os.path.expanduser("~/bench_same_lesion_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

# Name at least one lesion that sampled FAST and one that sampled SLOW in your
# earlier runs (fit_seconds in lesion_bayes_results.csv); the split can differ
# a lot between easy and hard posteriors.  [] -> the first lesion in RUN_DIR.
LESION_SIDS = []

MODE = "FXL"                    # "FXL" (joint delay) | "NXL" (fixed delay from FXL)
BATCH_SIZES = (1, 4, 8, 12)     # add larger sizes for a longer curve
N_TUNE = 300                    # warmup steps in the benchmark
N_DRAWS = 300                   # sampling steps in the benchmark
N_CHAINS = 16                   # chains PER COPY, as in production
N_REF = 3                       # independent single copies averaged for B = 1
N_TIMED = 3                     # timed repeats of the sampling phase
SEED0 = 1000

# for the run-time projection (your production settings)
PROD_TUNE, PROD_DRAWS, N_COHORT = 3000, 3000, 174

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


def cv(x):
    x = np.asarray(x, float).ravel()
    return float(np.std(x) / np.mean(x)) if len(x) > 1 and np.mean(x) != 0 else np.nan


class BatchRunner:
    """
    B copies of one prepared lesion as two compiled phases, warmup and
    sampling. Both are compiled once (on the first run()) and reused, so the
    N_REF repeats at B = 1 cost only their execution.
    """

    def __init__(self, p0, B, t, cb):
        self.p0, self.B, self.cb = p0, B, cb
        self.C = cb.n_chains
        self.D = p0.layout.n_total
        self.data = bb.stack_lesion_data([p0.data] * B)
        transform = build_transform(p0.prior, self.D)
        self.transform = transform
        fk = dict(p0.fwd_kwargs)
        predict = bb.make_predict_data(fk.pop("model_name"), t, p0.ctx, **fk)
        log_lik = bb.make_log_lik_data(predict, noise=bb._NOISE_MAP[cb.noise])
        logdensity = bb.make_unconstrained_logdensity_data(log_lik, transform)
        target_accept = 0.8 if cb.target_accept == 0.234 else cb.target_accept
        dense = getattr(cb, "dense_mass_matrix", False)
        C, n_tune, n_draws = self.C, cb.n_tune, cb.n_draws

        def warm_chain(wk, z0, data):
            ld = lambda z: logdensity(z, data)
            warm = blackjax.window_adaptation(
                blackjax.nuts, ld, is_mass_matrix_diagonal=not dense,
                target_acceptance_rate=target_accept,
                adaptation_info_fn=lambda s, i, a: i.num_integration_steps)
            (state, params), nlf = warm.run(wk, z0, num_steps=n_tune)
            return state, params, nlf

        def sample_chain(state, params, run_keys, data):
            ld = lambda z: logdensity(z, data)
            kernel = blackjax.nuts(ld, step_size=params["step_size"],
                                   inverse_mass_matrix=params["inverse_mass_matrix"])

            def step(st, k):
                st, info = kernel.step(k, st)
                return st, info.num_integration_steps
            _, nlf = jax.lax.scan(step, state, run_keys)
            return nlf

        def warm_lesion(wks, z0s, data):
            return jax.vmap(warm_chain, in_axes=(0, 0, None))(wks, z0s, data)

        def sample_lesion(states, params, k_run, data):
            per_draw = jax.random.split(k_run, n_draws)
            run_keys = jnp.swapaxes(
                jax.vmap(lambda k: jax.random.split(k, C))(per_draw), 0, 1)
            return jax.vmap(sample_chain, in_axes=(0, 0, 0, None))(
                states, params, run_keys, data)

        self._warm_jit = jax.jit(jax.vmap(warm_lesion))
        self._sample_jit = jax.jit(jax.vmap(sample_lesion))
        self._warm_c = self._sample_c = None
        self.compile_s = 0.0

    def run(self, seeds):
        p0, C, D = self.p0, self.C, self.D
        inits = [bb._init_and_keys(self.transform, p0.prior, p0.init_point, C, D, s)
                 for s in seeds]
        z_init = jnp.stack([a[0] for a in inits])
        wk = jnp.stack([a[1] for a in inits])
        kr = jnp.stack([a[2] for a in inits])

        if self._warm_c is None:
            t0 = time.time()
            self._warm_c = self._warm_jit.lower(wk, z_init, self.data).compile()
            self.compile_s += time.time() - t0
            jax.block_until_ready(self._warm_c(wk, z_init, self.data))  # untimed first run
        t0 = time.time()
        out = jax.block_until_ready(self._warm_c(wk, z_init, self.data))
        warm_s = time.time() - t0
        states, params, wnlf = out

        if self._sample_c is None:
            t0 = time.time()
            self._sample_c = self._sample_jit.lower(states, params, kr, self.data).compile()
            self.compile_s += time.time() - t0
            jax.block_until_ready(self._sample_c(states, params, kr, self.data))  # untimed
        times = []
        for _ in range(N_TIMED):
            t0 = time.time()
            nlf = jax.block_until_ready(self._sample_c(states, params, kr, self.data))
            times.append(time.time() - t0)

        return dict(warm_s=warm_s, sample_s=float(np.median(times)),
                    spread=(max(times) - min(times)) / float(np.median(times)),
                    wnlf=np.asarray(wnlf), nlf=np.asarray(nlf),
                    step_size=np.asarray(params["step_size"]))


def lock_stats(nlf):
    """nlf (L, C, N): (batch max, per-lesion max, mean lane), each averaged over steps."""
    return (float(nlf.max(axis=(0, 1)).mean()), float(nlf.max(axis=1).mean()),
            float(nlf.mean()))


def bench_lesion(L, t, cb, wx_kw):
    print(f"\n{'=' * 100}\n{L['sid']}  (MODE={MODE}, {N_CHAINS} chains per copy, "
          f"{N_TUNE} tune + {N_DRAWS} draws)\n{'=' * 100}")
    p0 = bb.prepare_lesion(L["St"], L["aif"], t, TCXM, acq, subject, cfg_fit,
                           BayesConfig(**{**cb.__dict__, "delay_fixed": L["delay"]}),
                           **wx_kw)
    rows, ref = [], None
    for B in BATCH_SIZES:
        print(f"--- {B} copies x {N_CHAINS} chains = {B * N_CHAINS} lanes ---", flush=True)
        t_block = time.time()
        runner = BatchRunner(p0, B, t, cb)
        n_rep = N_REF if B == 1 else 1
        reps = [runner.run([SEED0 + 1000 * r + k for k in range(B)]) for r in range(n_rep)]

        ms_step = float(np.mean([1e3 * r["sample_s"] / N_DRAWS for r in reps]))
        stats = [lock_stats(r["nlf"]) for r in reps]
        iters = float(np.mean([s[0] for s in stats]))
        lesion_max = float(np.mean([s[1] for s in stats]))
        mean_lane = float(np.mean([s[2] for s in stats]))
        ms_iter = float(np.mean([1e3 * r["sample_s"] / N_DRAWS / s[0]
                                 for r, s in zip(reps, stats)]))
        wst = [lock_stats(r["wnlf"]) for r in reps]
        warm_ms = float(np.mean([1e3 * r["warm_s"] / N_TUNE for r in reps]))
        if ref is None:
            ref = dict(ms_step=ms_step, ms_iter=ms_iter)
        slow, hw = ms_step / ref["ms_step"], ms_iter / ref["ms_iter"]
        traj = iters / lesion_max

        nlf = reps[-1]["nlf"]
        ss = reps[-1]["step_size"]
        lf_copy = nlf.mean(axis=(1, 2))
        ss_copy = ss.mean(axis=1)
        # whole-batch run times
        #   batch_run_s       one complete batch (warmup + sampling) at the benchmark
        #                     length, compile excluded; for B = 1 the mean of the
        #                     N_REF single copies
        #   bench_elapsed_s   wall clock for everything done at this batch size
        #                     (compile, untimed first runs, all repeats) -- how long
        #                     this block of the benchmark actually took you
        #   proj_batch_min    one batch at production length (PROD_TUNE + PROD_DRAWS),
        #                     from the measured per-step times; compile excluded
        batch_run_s = float(np.mean([r["warm_s"] + r["sample_s"] for r in reps]))
        bench_elapsed_s = time.time() - t_block
        proj_batch_min = (warm_ms * PROD_TUNE + ms_step * PROD_DRAWS) / 6e4
        proj_min = proj_batch_min / B
        row = dict(
            copies=B, lanes=B * N_CHAINS,
            ms_step=ms_step, slow=slow, traj=traj, hw=hw,
            speedup_per_copy=B / slow, ideal=B, ms_per_iter=ms_iter,
            iters_per_step=iters, lesion_max=lesion_max, leapfrog_mean=mean_lane,
            warm_ms_step=warm_ms, warm_traj=float(np.mean([s[0] / s[1] for s in wst])),
            ms_spread=float(np.mean([r["spread"] for r in reps])),
            ref_cv=(cv([1e3 * r["sample_s"] / N_DRAWS for r in reps]) if B == 1 else np.nan),
            step_size_mean=float(ss.mean()), step_size_cv_chains=cv(ss),
            leapfrog_cv_copies=cv(lf_copy),
            corr_stepsize_leapfrog=(float(np.corrcoef(ss_copy, lf_copy)[0, 1])
                                    if B >= 4 else np.nan),
            compile_s=runner.compile_s,
            batch_run_s=batch_run_s, bench_elapsed_s=bench_elapsed_s,
            proj_batch_min=proj_batch_min,
            proj_min_per_lesion=proj_min, proj_h_cohort=proj_min * N_COHORT / 60)
        rows.append(row)
        print(pd.DataFrame([row]).round(3).T.to_string(header=False), flush=True)
        del runner

    df = pd.DataFrame(rows)
    df.insert(0, "sid", L["sid"])
    return df


def plot(df_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sids = list(df_all["sid"].unique())
    fig, axes = plt.subplots(1, len(sids), figsize=(5.2 * len(sids), 4.2), squeeze=False)
    for ax, sid in zip(axes[0], sids):
        d = df_all[df_all["sid"] == sid]
        ax.plot(d["copies"], d["copies"], "k:", label="ideal (slowdown = B)")
        ax.plot(d["copies"], d["slow"], "o-", color="#1B4F72", lw=2,
                label="measured slowdown per NUTS step")
        ax.plot(d["copies"], d["traj"], "s--", color="#B22234", label="trajectory part (traj)")
        ax.plot(d["copies"], d["hw"], "^--", color="#E67E22", label="GPU-iteration part (hw)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("copies of the same lesion in the batch")
        ax.set_ylabel("factor relative to 1 copy")
        ax.set_title(f"{sid}:  slowdown = traj x hw", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"devices: {jax.devices()}   x64: {jax.config.jax_enable_x64}")
    lesions = {L["sid"]: L for L in load_lesions(RUN_DIR)}
    sids = LESION_SIDS or [next(iter(lesions))]
    missing = [s for s in sids if s not in lesions]
    if missing:
        raise KeyError(f"not in {RUN_DIR}: {missing}.  Available: {sorted(lesions)[:10]} ...")
    t = lesions[sids[0]]["t"]
    print(f"lesions: {sids}")

    cb = BayesConfig(prior_choices=PRIOR_CHOICES, n_draws=N_DRAWS, n_tune=N_TUNE,
                     n_chains=N_CHAINS, noise="gauss_marginalized_sigma", seed=SEED0,
                     backend="blackjax", step="nuts", dense_mass_matrix=True,
                     delay_mode=("joint" if MODE == "FXL" else "fixed"),
                     delay_bounds=DELAY_BOUNDS, progressbar=False)
    wx_kw = dict(wx_model=NXL, aif_cfg=aif_cfg) if MODE == "NXL" else {}

    dfs = []
    for sid in sids:
        if not np.allclose(lesions[sid]["t"], t):
            raise ValueError(f"{sid}: different time grid")
        dfs.append(bench_lesion(lesions[sid], t, cb, wx_kw))
        pd.concat(dfs).to_csv(os.path.join(OUT_DIR, f"same_lesion_{MODE}.csv"), index=False)

    df_all = pd.concat(dfs, ignore_index=True)
    show = ["sid", "copies", "lanes", "ms_step", "slow", "traj", "hw", "speedup_per_copy",
            "ms_per_iter", "iters_per_step", "lesion_max", "warm_ms_step", "ms_spread",
            "step_size_cv_chains"]
    times = ["sid", "copies", "batch_run_s", "compile_s", "bench_elapsed_s",
             "proj_batch_min", "proj_min_per_lesion", "proj_h_cohort"]
    print("\n" + "=" * 130)
    print(df_all[show].round(3).to_string(index=False))
    print("=" * 130)
    print("RUN TIMES")
    print(df_all[times].round(2).to_string(index=False))
    print("=" * 130)
    print("slow = traj x hw.  ms_step = SAMPLING phase, median of "
          f"{N_TIMED} timed repeats.  ms_spread = (max-min)/median of those repeats: "
          "large means the GPU is shared.\nB=1 reference = mean of "
          f"{N_REF} independent copies (ref_cv in the CSV is their spread).\n"
          f"batch_run_s = one batch, warmup + sampling at {N_TUNE}+{N_DRAWS} steps, "
          "compile excluded.  bench_elapsed_s = wall clock for this batch size in this "
          "script.\n"
          f"proj_batch_min = one batch at {PROD_TUNE}+{PROD_DRAWS} steps; "
          f"proj_h_cohort = hours for {N_COHORT} lesions if every batch looked like "
          "this one (identical copies = best case; excludes compile and CPU work).\n"
          "Saved: " + OUT_DIR)
    plot(df_all, os.path.join(OUT_DIR, f"same_lesion_{MODE}.png"))


if __name__ == "__main__":
    main()