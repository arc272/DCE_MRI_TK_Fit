"""
check_12_batched_loglik.py  --  STEP 1 of the batched-fitting plan
==================================================================
Is bayes_batched.make_log_lik_data(p, data) the same function as the
log-likelihood fit_curve_bayes hands to BlackJAX today?

Reference = the exact construction in fit_curve_bayes's blackjax branch:
    static_ctx_from_configs -> make_predict(aif/S0/baseline as closure
    constants) -> make_residual -> make_log_lik_jax
Candidate = prepare_lesion -> make_predict_data -> make_log_lik_data, with
all L lesions evaluated in ONE vmapped call (the way the sampler uses it).

Lesions differ in everything that is per-lesion data: AIF arrival and
scale, S0, noise, true delay, the fixed delay, and acq.nbase (so S0 and the
AIF baseline are computed over different windows).

Configurations: FXL + joint delay, NXL + fixed delay, SSM3 (tau_i, tau_b
fixed) + fixed delay, and FXL + joint delay with free sigma. For each:
value and gradient at 40 prior draws per lesion, plus the full
unconstrained log-density (likelihood + prior + Jacobian) at the same points.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp

from dce_mri.config import (AcquisitionConfig, SubjectConfig, FittingConfig,
                            AifConfig, RelaxationConfig)
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import NXL, SSM3
from dce_mri.synthetic import make_parker_aif, simulate_curve, add_noise
from dce_mri.signal_models import shift_aif, compute_S0
from dce_mri.jax_forward import static_ctx_from_configs, make_predict, make_residual
from dce_mri.bayes import BayesConfig
from dce_mri.bayes_blackjax import (make_log_lik_jax, build_transform,
                                    make_unconstrained_logdensity)
from dce_mri import bayes_batched as bb

acq0 = AcquisitionConfig(TR=4.87, FA=10.0, r1=3.9, nbase=slice(0, 7))
subj = SubjectConfig(subject_id="S", T10=1530.0, B1=1.0)
cfg_fit = FittingConfig(n_lhs=50, n_top=3, fit_space="signal")
aif_cfg = AifConfig(T10=1779.0, hematocrit=0.4)
relax_cfg = RelaxationConfig(fit_tau_b=False, tau_b_fixed=0.005,
                             tau_i_lb=0.01, tau_i_ub=1.5)
t = np.arange(36) * 150 * 8 / 288
L, N_PTS = 4, 40
NOISE_MAP = bb._NOISE_MAP

rng = np.random.default_rng(12)
lesions = []
for i in range(L):
    aif = make_parker_aif(t, t_arrival=rng.uniform(25, 40), hematocrit=0.4,
                          scale=rng.uniform(0.8, 1.2))
    acq = replace(acq0, nbase=slice(0, int(rng.integers(5, 9))))
    theta = np.array([rng.uniform(0.15, 0.45), rng.uniform(0.02, 0.12),
                      rng.uniform(0.005, 0.03), rng.uniform(0.002, 0.012)])
    d_true = rng.uniform(2, 8)
    S = add_noise(simulate_curve(theta, shift_aif(aif, d_true, t, acq), t, TCXM,
                                 acq, subj, cfg_fit, S0=rng.uniform(600, 1500)),
                  rng.uniform(3, 10), rng=np.random.default_rng(100 + i))
    lesions.append(dict(sig=S, aif=aif, acq=acq,
                        d_fix=d_true + rng.uniform(-1, 1)))

CONFIGS = [
    ("FXL  + joint delay",  None, "joint", "gauss_marginalized_sigma",
     {"rates": "loguniform", "volfrac": "beta"}),
    ("NXL  + fixed delay",  NXL,  "fixed", "gauss_marginalized_sigma",
     {"rates": "loguniform", "volfrac": "beta"}),
    ("SSM3 + fixed delay",  SSM3, "fixed", "gauss_marginalized_sigma",
     {"rates": "uniform", "volfrac": "flat_masked", "tau": "flat"}),
    # free sigma is not tested: bayes_blackjax.build_transform has no
    # transform for its noise prior (Jeffreys), so sample_blackjax cannot run
    # it either. The likelihood branch exists for when that is added.
]

print(f"{L} lesions, {N_PTS} prior draws each; max |difference| over all points\n")
print(f"{'config':26s} {'log-lik rel':>12s} {'gradient rel':>13s} "
      f"{'log-density rel':>16s}")
worst = 0.0
for name, wx, dmode, noise, choices in CONFIGS:
    cb = BayesConfig(prior_choices=choices, noise=noise, backend="blackjax",
                     step="nuts", delay_mode=dmode, delay_bounds=(0.0, 25.0),
                     n_chains=4, seed=1)
    wx_kw = dict(wx_model=wx, aif_cfg=aif_cfg if wx else None,
                 relax_cfg=relax_cfg if wx else None)

    preps, refs = [], []
    for les in lesions:
        cbl = BayesConfig(**{**cb.__dict__, "delay_fixed": les["d_fix"]})
        p = bb.prepare_lesion(les["sig"], les["aif"], t, TCXM, les["acq"], subj,
                              cfg_fit, cbl, **wx_kw)
        preps.append(p)

        # ---- reference: fit_curve_bayes's blackjax branch, verbatim -------
        acq = les["acq"]
        ctx = static_ctx_from_configs(acq, subj, wx_kw["aif_cfg"])
        S0 = float(compute_S0(les["sig"][None, :], acq)[0])
        aif_use = les["aif"]
        if dmode == "fixed":
            aif_use = shift_aif(les["aif"], float(les["d_fix"]), t, acq)
        has_tau = wx is not None and wx.n_params > 0
        uses_tb = has_tau or (wx is not None and getattr(wx, "uses_tau_b_fixed", False))
        predict = make_predict(
            TCXM.name, t, aif_use, ctx, S0,
            wx=wx is not None, wx_name=wx.name if wx is not None else "SSM3",
            fit_tau_b=relax_cfg.fit_tau_b if has_tau else False,
            tau_b_fixed=float(relax_cfg.tau_b_fixed) if uses_tb else 0.0,
            delay=(dmode == "joint"),
            baseline_val=float(np.mean(np.asarray(aif_use)[acq.nbase])),
            upsample=acq.upsample)
        refs.append(make_log_lik_jax(make_residual(predict, les["sig"]),
                                     noise=NOISE_MAP[noise]))

    p0 = preps[0]
    D = p0.layout.n_total
    transform = build_transform(p0.prior, D)
    fk = dict(p0.fwd_kwargs)
    cand = bb.make_log_lik_data(bb.make_predict_data(fk.pop("model_name"), t,
                                                     p0.ctx, **fk),
                                noise=NOISE_MAP[noise])
    cand_ld = bb.make_unconstrained_logdensity_data(cand, transform)
    data = bb.stack_lesion_data([p.data for p in preps])

    # same points for every lesion: prior draws (constrained), and their z
    X = jnp.asarray(p0.prior.sample(np.random.default_rng(7), N_PTS))
    Z = jax.vmap(transform.unconstrain)(X)

    # candidate: ONE call, vmapped over lesions and points
    over_pts = lambda f: jax.vmap(f, in_axes=(0, None))
    over_les = lambda f: jax.vmap(f, in_axes=(None, 0))
    c_ll = jax.jit(over_les(over_pts(cand)))(X, data)                 # (L, N)
    c_g = jax.jit(over_les(over_pts(jax.grad(cand))))(X, data)        # (L, N, D)
    c_ld = jax.jit(over_les(over_pts(cand_ld)))(Z, data)

    r_ll = np.stack([np.asarray(jax.jit(jax.vmap(f))(X)) for f in refs])
    r_g = np.stack([np.asarray(jax.jit(jax.vmap(jax.grad(f)))(X)) for f in refs])
    r_ld = np.stack([np.asarray(jax.jit(jax.vmap(
        make_unconstrained_logdensity(f, transform)))(Z)) for f in refs])

    rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - b) /
                                    np.maximum(np.abs(b), 1e-300)))
    # gradient: relative to each point's gradient norm (single components
    # can be ~0)
    gn = np.linalg.norm(r_g, axis=-1, keepdims=True)
    g_rel = float(np.max(np.abs(np.asarray(c_g) - r_g) / np.maximum(gn, 1e-300)))
    e = (rel(c_ll, r_ll), g_rel, rel(c_ld, r_ld))
    worst = max(worst, *e)
    finite = np.isfinite(r_ll).mean()
    print(f"{name:26s} {e[0]:12.2e} {e[1]:13.2e} {e[2]:16.2e}"
          f"   ({finite:.0%} of reference values finite, D={D})")

print(f"\nworst relative difference: {worst:.2e}  ->  "
      f"{'PASS' if worst < 1e-10 else 'FAIL'} (tolerance 1e-10)")
