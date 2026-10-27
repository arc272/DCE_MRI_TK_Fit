# -*- coding: utf-8 -*-
"""
test_jax_equivalence.py
=======================
Asserts that dce_mri.jax_forward reproduces the numpy reference to within
floating-point tolerance, then reports the speedup.

The numpy implementation stays the reference. This file is the thing that
makes the JAX path trustworthy: any future edit to either side that moves
a curve shows up here rather than in a posterior three weeks later.

Run:
    JAX_ENABLE_X64=1 python tests/test_jax_equivalence.py

float64 is REQUIRED. JAX defaults to float32; at float32 the SSM3 matrix
exponential and the baseline normalisation disagree with numpy in the 4th
significant figure, which is large enough to shift a tau_i posterior.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dce_mri.synthetic import default_test_setup
from dce_mri.kinetic_models import TCXM, GKM, tcxm_ct_compartments, gkm_ct_compartments
from dce_mri.water_exchange import SSM3, _ssm3_to_signal
from dce_mri.signal_models import concentration_to_signal as np_c2s, shift_aif as np_shift
from dce_mri.fitting import build_residual

from dce_mri import jax_forward as jf


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def relerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def check(label, a, b, tol=1e-9):
    e = relerr(a, b)
    flag = "PASS" if e < tol else "FAIL"
    print(f"  [{flag}] {label:<42s} max rel err = {e:.3e}   (tol {tol:.0e})")
    return e < tol


def bench(fn, n=100):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6      # microseconds


# -----------------------------------------------------------------------------
# setup
# -----------------------------------------------------------------------------

s = default_test_setup()
t_np   = s["time_var"]
aif_np = s["aif"]
acq, subject, aif_cfg = s["acq"], s["subject"], s["aif_cfg"]
cfg_fit, relax = s["cfg"], s["relax_cfg"]
S0 = s["S0"]

ctx = jf.static_ctx_from_configs(acq, subject, aif_cfg)
t_j   = jnp.asarray(t_np)
aif_j = jnp.asarray(aif_np)

theta_tcxm = np.asarray(TCXM.p0, float)          # [ve, vp, fp, ps]
theta_gkm  = np.asarray(GKM.p0,  float)          # [Ktrans, ve, vp]
tau_i, tau_b = 0.5, 0.05

ok = True
print(f"\nsetup: {len(t_np)} frames, dt={t_np[1]-t_np[0]:.1f}s, "
      f"upsample={acq.upsample}, x64={jnp.zeros(1).dtype}")


# -----------------------------------------------------------------------------
# 1. kinetic models
# -----------------------------------------------------------------------------
print("\n1. kinetic forward models")

Cv_np, Ce_np = tcxm_ct_compartments(theta_tcxm, t_np, aif_np, acq.upsample)
Cv_j,  Ce_j  = jf.tcxm_compartments(jnp.asarray(theta_tcxm), t_j, aif_j, acq.upsample)
ok &= check("TCXM  Cv", Cv_np, Cv_j)
ok &= check("TCXM  Ce", Ce_np, Ce_j)

Cvg_np, Ceg_np = gkm_ct_compartments(theta_gkm, t_np, aif_np, acq.upsample)
Cvg_j,  Ceg_j  = jf.gkm_compartments(jnp.asarray(theta_gkm), t_j, aif_j, acq.upsample)
ok &= check("GKM   Cv", Cvg_np, Cvg_j)
ok &= check("GKM   Ce", Ceg_np, Ceg_j)


# -----------------------------------------------------------------------------
# 2. signal models
# -----------------------------------------------------------------------------
print("\n2. signal models")

Ct_np = Cv_np + Ce_np
ok &= check("FXL concentration_to_signal",
            np_c2s(Ct_np, acq, subject.T10, subject.B1, S0),
            jf.concentration_to_signal(jnp.asarray(Ct_np), ctx, S0))

ok &= check("SSM3 to_signal",
            _ssm3_to_signal(Cv_np, Ce_np, theta_tcxm[0], theta_tcxm[1],
                            (tau_b, tau_i), acq, subject, aif_cfg, relax, S0),
            jf.ssm3_to_signal(jnp.asarray(Cv_np), jnp.asarray(Ce_np),
                              theta_tcxm[0], theta_tcxm[1], tau_b, tau_i, ctx, S0))

baseline_val = float(np.asarray(aif_np)[acq.nbase].mean())
for tau in (-4.0, 0.7, 12.0):
    ok &= check(f"shift_aif (tau={tau:+.1f}s)",
                np_shift(aif_np, tau, t_np, acq),
                jf.shift_aif(aif_j, tau, t_j, baseline_val))


# -----------------------------------------------------------------------------
# 3. full residual, wx path -- the thing samplers actually call
# -----------------------------------------------------------------------------
print("\n3. full residual (TCXM + SSM3, fit_tau_b=False)")

rng = np.random.default_rng(0)
sig = _ssm3_to_signal(Cv_np, Ce_np, theta_tcxm[0], theta_tcxm[1],
                      (relax.tau_b_fixed, tau_i), acq, subject, aif_cfg, relax, S0)
sig = sig + rng.normal(0, 0.01 * S0, sig.shape)

bundle = build_residual(sig, aif_np, t_np, TCXM, acq, subject, cfg_fit,
                        wx_model=SSM3, aif_cfg=aif_cfg, relax_cfg=relax,
                        on_failure="nan")

# build_residual derives S0 from the MEASURED baseline (compute_S0), not from
# the nominal S0 the curve was simulated with. With noise on the curve those
# differ by ~sigma/sqrt(nbase) -- 0.4% here -- which is exactly the size of
# the residual mismatch this line removes. The JAX path must be handed the
# same number, not the nominal one.
from dce_mri.signal_models import compute_S0
S0_fit = float(compute_S0(sig[None, :], acq)[0])
print(f"  (S0 nominal {S0:.2f} vs measured-baseline {S0_fit:.2f})")

predict_j = jf.make_predict("TCXM", t_np, aif_np, ctx, S0_fit, wx=True,
                            fit_tau_b=relax.fit_tau_b,
                            tau_b_fixed=relax.tau_b_fixed,
                            upsample=acq.upsample)
resid_j = jax.jit(jf.make_residual(predict_j, sig))

# sweep a few parameter vectors, not just p0 -- a single point can agree by
# accident where a whole region does not
for i in range(5):
    p = np.concatenate([
        TCXM.lb + rng.uniform(0.1, 0.9, 4) * (TCXM.ub - TCXM.lb),
        [rng.uniform(relax.tau_i_lb, min(relax.tau_i_ub, 2.0))],
    ])
    if p[0] + p[1] > 1.0:
        continue
    ok &= check(f"residual_fn  draw {i}",
                bundle.residual_fn(p), resid_j(jnp.asarray(p)), tol=1e-8)


# -----------------------------------------------------------------------------
# 4. gradient sanity -- the reason for the whole exercise
# -----------------------------------------------------------------------------
print("\n4. autodiff vs central differences (log-posterior)")

lb = np.concatenate([TCXM.lb, [relax.tau_i_lb]])
ub = np.concatenate([TCXM.ub, [min(relax.tau_i_ub, 2.0)]])
lb_j, ub_j = jnp.asarray(lb), jnp.asarray(ub)


def log_prior(p):
    inside = jnp.all((p >= lb_j) & (p <= ub_j))
    return jnp.where(inside, 0.0, -1e30)


log_post_j = jax.jit(jf.make_log_posterior(resid_j, log_prior, noise="marginalized"))
grad_j = jax.jit(jax.grad(log_post_j))

p_test = jnp.asarray(np.concatenate([theta_tcxm, [tau_i]]))
g_ad = np.asarray(grad_j(p_test))

eps_fd = 1e-6
g_fd = np.empty_like(g_ad)
for d in range(len(g_ad)):
    step = eps_fd * max(abs(float(p_test[d])), 1e-3)
    pp = np.array(p_test); pp[d] += step
    pm = np.array(p_test); pm[d] -= step
    g_fd[d] = (float(log_post_j(jnp.asarray(pp)))
               - float(log_post_j(jnp.asarray(pm)))) / (2 * step)

print("      param      autodiff          central-diff")
for nm, a, b in zip(("ve", "vp", "fp", "ps", "tau_i"), g_ad, g_fd):
    print(f"      {nm:<7s} {a:+16.6e}  {b:+16.6e}")
ok &= check("grad(log_post)", g_fd, g_ad, tol=1e-4)
assert np.all(np.isfinite(g_ad)), "non-finite gradient -- check _safe_div usage"


# -----------------------------------------------------------------------------
# 5. speed
# -----------------------------------------------------------------------------
print("\n5. timing (per call, microseconds)")

p_np = np.concatenate([theta_tcxm, [tau_i]])
t_numpy = bench(lambda: bundle.residual_fn(p_np))

resid_j(p_test).block_until_ready()
t_jax = bench(lambda: resid_j(p_test).block_until_ready())

log_post_j(p_test).block_until_ready()
t_lp = bench(lambda: log_post_j(p_test).block_until_ready())
t_gr = bench(lambda: grad_j(p_test).block_until_ready())

# batched: what a vectorised ensemble sampler would actually issue
for B in (32, 256, 1024):
    P = jnp.asarray(np.tile(np.array(p_test), (B, 1)))
    vlp = jax.jit(jax.vmap(log_post_j))
    vlp(P).block_until_ready()
    tb = bench(lambda: vlp(P).block_until_ready(), n=30)
    print(f"  vmap log_post  B={B:<5d} {tb:9.1f} us total "
          f"= {tb/B:7.2f} us/curve  ({t_numpy/(tb/B):6.1f}x vs numpy)")

print(f"\n  numpy residual_fn          {t_numpy:9.1f} us")
print(f"  jax   residual_fn (jit)    {t_jax:9.1f} us   ({t_numpy/t_jax:5.1f}x)")
print(f"  jax   log_post    (jit)    {t_lp:9.1f} us")
print(f"  jax   grad(log_post)       {t_gr:9.1f} us   "
      f"(= {t_gr/t_lp:.2f} log_post evals; finite differences would cost "
      f"{2*len(g_ad)})")

print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
sys.exit(0 if ok else 1)
