"""
check_13b_batched_divergence.py
===============================
Follow-up to check_13, which found (i) NO draw identical between a lesion
fitted alone and in a batch, even at draw 0, and (ii) lesion B1 failing the
quantile test (|dq95(ps)| = 68 MCSE) -- while its SEQUENTIAL run had
R-hat 1.05-1.11 and ESS 25-75, i.e. had not mixed.

Two questions:

1. Do the two programs run the SAME algorithm on the same random stream?
   Very short runs (n_tune=10). Same NLLS anchor? Same start? If draws
   agree to ~1e-12 at first and the gap then grows geometrically, the
   only difference is floating-point rounding amplified by NUTS (chaotic,
   like any long MCMC trajectory). If they differ by O(1) at the first
   draw, the algorithms differ.

2. Is B1's disagreement larger than two SEQUENTIAL runs disagree with each
   other? Null comparison: sequential seed 1 vs sequential seed 2 at the
   same 300/300 length (same statistic). Then batched vs sequential at
   1500/1500, where both should have mixed.
"""
import os, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp

from dce_mri.config import AcquisitionConfig, SubjectConfig, FittingConfig, AifConfig
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL
from dce_mri.synthetic import make_parker_aif, simulate_curve, add_noise
from dce_mri.signal_models import shift_aif
from dce_mri.bayes import BayesConfig, fit_curve_bayes
from dce_mri.bayes_blackjax import build_transform
from dce_mri import bayes_batched as bb

acq0 = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
subj = SubjectConfig(subject_id="S", T10=1530.0, B1=1.0)
cfg_fit = FittingConfig(n_lhs=50, n_top=3, fit_space="signal")
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)
t = np.arange(36) * 150 * 8 / 288

# regenerate check_13's lesions with the same rng sequence
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
    return dict(sig=S, aif=aif, acq=acq, delay=d, theta=theta)
Ls = [lesion(i) for i in range(4)]
B1 = Ls[2]                                         # check_13's B1
print(f"B1 truth: ve {B1['theta'][0]:.3f} vp {B1['theta'][1]:.3f} "
      f"fp {B1['theta'][2]:.4f} ps {B1['theta'][3]:.4f}  (simulated with FXL, fitted with NXL)")

def cfg(n_tune, n_draws, seed=1, chains=4):
    return BayesConfig(prior_choices={"rates": "loguniform", "volfrac": "beta"},
                       n_draws=n_draws, n_tune=n_tune, n_chains=chains,
                       noise="gauss_marginalized_sigma", seed=seed, backend="blackjax",
                       step="nuts", dense_mass_matrix=True, delay_mode="fixed",
                       delay_fixed=B1["delay"], delay_bounds=(0.0, 25.0),
                       progressbar=False)

def seq(c):
    return fit_curve_bayes(B1["sig"], B1["aif"], t, TCXM, B1["acq"], subj, cfg_fit, c,
                           wx_model=NXL, aif_cfg=aif_cfg)

def bat(c):
    return bb.fit_curve_bayes_batched([B1["sig"]], [B1["aif"]], t, TCXM, [B1["acq"]],
                                      subj, cfg_fit, c, delay_fixed=[B1["delay"]],
                                      wx_model=NXL, aif_cfg=aif_cfg, verbose=False)[0]

# ---- 1. same algorithm, same stream? ------------------------------------------
print("\n1. very short runs (n_tune=10, n_draws=40), B1, batch of 1 vs alone")
rs, rb = seq(cfg(10, 40)), bat(cfg(10, 40))
print(f"   NLLS anchor identical: {np.array_equal(rs.point_estimate, rb.point_estimate)}"
      f"   (max |diff| {np.max(np.abs(rs.point_estimate - rb.point_estimate)):.1e})")
sd = rs.chains.reshape(-1, rs.chains.shape[-1]).std(0)
gap = np.max(np.abs(rs.chains - rb.chains) / sd, axis=(0, 2))     # per draw, worst chain/param
for i in (0, 1, 2, 5, 10, 20, 39):
    print(f"   draw {i:2d}: max |seq - bat| = {gap[i]:.1e} posterior SDs")

# ---- 2. null: two sequential seeds at the check_13 length ----------------------
def zstats(a, b):
    out = {}
    for j, p in enumerate(a.param_names):
        m = np.hypot(a.mcse_median[p], b.mcse_median[p])
        xa, xb = a.chains[..., j].ravel(), b.chains[..., j].ravel()
        out[p] = max(abs(np.quantile(xa, q) - np.quantile(xb, q)) / m
                     for q in (0.05, 0.5, 0.95))
    return out

def show(tag, a, b):
    z = zstats(a, b)
    ra = max(a.rhat.values()); rb_ = max(b.rhat.values())
    ea = min(a.ess.values()); eb = min(b.ess.values())
    print(f"   {tag:44s} worst |dq|/mcse " +
          "  ".join(f"{p} {v:5.1f}" for p, v in z.items()) +
          f"   R-hat max {ra:.3f}/{rb_:.3f}  ESS min {ea:.0f}/{eb:.0f}")

print("\n2. B1 quantile disagreement, worst over q05/q50/q95, in MCSE units")
t0 = time.time()
s1, s2 = seq(cfg(300, 300, seed=1)), seq(cfg(300, 300, seed=2))
show("300/300   seq seed 1 vs seq seed 2 (null)", s1, s2)
b1 = bat(cfg(300, 300, seed=1))
show("300/300   seq seed 1 vs batched seed 1", s1, b1)
print(f"   ({time.time() - t0:.0f} s)", flush=True)
t0 = time.time()
S1, B1r = seq(cfg(1500, 1500, seed=1)), bat(cfg(1500, 1500, seed=1))
show("1500/1500 seq seed 1 vs batched seed 1", S1, B1r)
S2 = seq(cfg(1500, 1500, seed=2))
show("1500/1500 seq seed 1 vs seq seed 2 (null)", S1, S2)
print(f"   ({time.time() - t0:.0f} s)")

m = B1r.meta["batch"]
print(f"\nleapfrog steps per NUTS step, B1 alone (1500/1500): mean {m['mean_leapfrog']:.1f}, "
      f"mean of per-step max over its 4 chains {m['mean_of_lesion_max']:.1f}")
