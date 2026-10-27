"""
check_13_batched_sampler.py  --  STEP 2 of the batched-fitting plan
===================================================================
Does fit_curve_bayes_batched give each lesion the same posterior as
fitting it alone with fit_curve_bayes (backend='blackjax')?

A  one lesion, FXL + joint delay: batch of 1 vs fit_curve_bayes, same seed.
   Keys and starting points are derived identically, so the draws should
   coincide up to floating-point differences between the two compiled
   programs. Reported: how many draws agree, and where chains first part.
B  three lesions, NXL + fixed delay (a different delay and baseline window
   per lesion -- the NXL cohort setup): one batch vs three separate fits.
   Same comparison per lesion.

If draws do part ways (NUTS amplifies 1e-15 differences over enough steps),
the fallback test is statistical: posterior quantiles must agree within
their Monte Carlo error, |dq| / sqrt(mcse_seq^2 + mcse_bat^2) < 3 for every
parameter of every lesion.

CPU here, so short chains (N_TUNE = N_DRAWS = 300, 4 chains). This tests
correctness, not speed.
"""
import os, sys, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
from dataclasses import replace
import numpy as np

from dce_mri.config import AcquisitionConfig, SubjectConfig, FittingConfig, AifConfig
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL
from dce_mri.synthetic import make_parker_aif, simulate_curve, add_noise
from dce_mri.signal_models import shift_aif
from dce_mri.bayes import BayesConfig, fit_curve_bayes
from dce_mri import bayes_batched as bb

N_TUNE = N_DRAWS = int(os.environ.get("N_STEPS", 300))
N_CHAINS = 4
acq0 = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
subj = SubjectConfig(subject_id="S", T10=1530.0, B1=1.0)
cfg_fit = FittingConfig(n_lhs=50, n_top=3, fit_space="signal")
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)
t = np.arange(36) * 150 * 8 / 288
rng = np.random.default_rng(13)


def lesion(i):
    aif = make_parker_aif(t, t_arrival=rng.uniform(25, 40), hematocrit=0.4,
                          scale=rng.uniform(0.8, 1.2))
    acq = replace(acq0, nbase=slice(0, int(rng.integers(5, 9))))
    theta = np.array([rng.uniform(0.15, 0.45), rng.uniform(0.02, 0.12),
                      rng.uniform(0.005, 0.03), rng.uniform(0.002, 0.012)])
    d = rng.uniform(2, 8)
    S = add_noise(simulate_curve(theta, shift_aif(aif, d, t, acq), t, TCXM, acq,
                                 subj, cfg_fit, S0=rng.uniform(600, 1500)),
                  rng.uniform(3, 10), rng=np.random.default_rng(200 + i))
    return dict(sig=S, aif=aif, acq=acq, delay=d)


def compare(tag, r_seq, r_bat):
    names = list(r_seq.param_names)
    a, b = r_seq.chains, r_bat.chains                      # (C, N, D)
    sd = a.reshape(-1, a.shape[-1]).std(0)
    diff = np.abs(a - b) / sd                              # in posterior SDs
    same = np.all(diff < 1e-8, axis=-1)                    # (C, N)
    first_split = [int(np.argmin(s)) if not s.all() else None for s in same]
    worst_z = 0.0
    print(f"\n{tag}")
    print(f"   draws identical (< 1e-8 posterior SD): {same.mean():.1%}   "
          f"first draw where each chain parts: {first_split}")
    print(f"   {'param':10s} {'median seq':>11s} {'median bat':>11s} "
          f"{'|dq50|/mcse':>12s} {'|dq05|/mcse':>12s} {'|dq95|/mcse':>12s}  "
          f"{'R-hat seq/bat':>14s} {'ESS seq/bat':>12s}")
    for j, p in enumerate(names):
        xa, xb = a[..., j].ravel(), b[..., j].ravel()
        m = np.hypot(r_seq.mcse_median[p], r_bat.mcse_median[p])
        z = [abs(np.quantile(xa, q) - np.quantile(xb, q)) / m for q in (0.5, 0.05, 0.95)]
        worst_z = max(worst_z, *z)
        print(f"   {p:10s} {np.median(xa):11.4g} {np.median(xb):11.4g} "
              f"{z[0]:12.2f} {z[1]:12.2f} {z[2]:12.2f}  "
              f"{r_seq.rhat[p]:6.3f}/{r_bat.rhat[p]:6.3f} "
              f"{r_seq.ess[p]:5.0f}/{r_bat.ess[p]:5.0f}")
    print(f"   divergences seq/bat: {r_seq.n_divergences}/{r_bat.n_divergences}")
    return same.mean(), worst_z


# ---- A -----------------------------------------------------------------------
cb = BayesConfig(prior_choices={"rates": "loguniform", "volfrac": "beta"},
                 n_draws=N_DRAWS, n_tune=N_TUNE, n_chains=N_CHAINS,
                 noise="gauss_marginalized_sigma", seed=1, backend="blackjax",
                 step="nuts", dense_mass_matrix=True, delay_mode="joint",
                 delay_bounds=(0.0, 25.0), progressbar=False)
L0 = lesion(0)
t0 = time.time()
r_seq = fit_curve_bayes(L0["sig"], L0["aif"], t, TCXM, L0["acq"], subj, cfg_fit, cb)
print(f"A sequential: {time.time() - t0:.0f} s", flush=True)
r_bat = bb.fit_curve_bayes_batched([L0["sig"]], [L0["aif"]], t, TCXM, [L0["acq"]],
                                   subj, cfg_fit, cb)[0]
res = {"A": compare("A  FXL + joint delay, batch of 1 vs fit_curve_bayes", r_seq, r_bat)}

# ---- B -----------------------------------------------------------------------
cbn = BayesConfig(**{**cb.__dict__, "delay_mode": "fixed"})
Ls = [lesion(i) for i in (1, 2, 3)]
seq = []
t0 = time.time()
for Li in Ls:
    c = BayesConfig(**{**cbn.__dict__, "delay_fixed": Li["delay"]})
    seq.append(fit_curve_bayes(Li["sig"], Li["aif"], t, TCXM, Li["acq"], subj,
                               cfg_fit, c, wx_model=NXL, aif_cfg=aif_cfg))
print(f"\nB sequential, 3 fits: {time.time() - t0:.0f} s", flush=True)
bat = bb.fit_curve_bayes_batched([Li["sig"] for Li in Ls], [Li["aif"] for Li in Ls],
                                 t, TCXM, [Li["acq"] for Li in Ls], subj, cfg_fit, cbn,
                                 delay_fixed=[Li["delay"] for Li in Ls],
                                 wx_model=NXL, aif_cfg=aif_cfg)
for k, (rs, rb) in enumerate(zip(seq, bat)):
    res[f"B{k}"] = compare(f"B{k} NXL + fixed delay {Ls[k]['delay']:.2f} s, "
                           f"nbase {Ls[k]['acq'].nbase}: batch of 3 vs alone", rs, rb)
print(f"\nlock-step: mean leapfrog steps per lane {bat[0].meta['batch']['mean_leapfrog']:.1f}, "
      f"mean of per-step batch max {bat[0].meta['batch']['mean_of_batch_max']:.1f}")

print("\n" + "=" * 78)
for k, (frac, z) in res.items():
    print(f"{k:3s} identical draws {frac:6.1%}   worst quantile |d|/mcse {z:5.2f}   "
          f"{'PASS' if (frac == 1.0 or z < 3) else 'FAIL'}")
print("=" * 78)
