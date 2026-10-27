"""
Decisive check: does the JAX target density (likelihood via make_log_lik_jax
+ prior) agree with the trusted NUMPY log-posterior from bayes.py, at the
SAME parameter points? This is deterministic -- no sampler, no Monte Carlo
noise. If these agree up to an additive constant, the BlackJAX wiring is
targeting the correct posterior and any mixing trouble is geometry, not code.
"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from dce_mri.synthetic import default_test_setup, simulate_curve, add_noise
from dce_mri.kinetic_models import TCXM
from dce_mri.water_exchange import SSM3
from dce_mri.config import RelaxationConfig
from dce_mri.bayes import BayesConfig, build_param_layout, make_log_posterior as np_make_log_post
from dce_mri.bayes_priors import assemble_prior
from dce_mri.bayes_blackjax import make_log_lik_jax, build_transform, make_unconstrained_logdensity
from dce_mri.jax_forward import make_predict, make_residual, static_ctx_from_configs

SIGMA_TRUE, TAU_I_TRUE, seed = 20.0, 0.5, 1
setup = default_test_setup()
acq, subject, cfg_fit, aif_cfg = setup["acq"], setup["subject"], setup["cfg"], setup["aif_cfg"]
relax = RelaxationConfig(fit_tau_b=False)
cfg_b = BayesConfig(n_draws=10, n_tune=10, n_chains=1, noise="gauss_marginalized_sigma",
                    delay_mode="none", seed=seed, backend="pymc", step="nuts")

layout = build_param_layout(TCXM, cfg_b, aif=setup["aif"], time_var=setup["time_var"],
                            acq=acq, wx_model=SSM3, relax_cfg=relax)
truth = np.array(list(np.asarray(TCXM.p0, float)) + [TAU_I_TRUE])
S = simulate_curve(truth, setup["aif"], setup["time_var"], layout.model, acq, subject,
                   cfg_fit, SSM3, aif_cfg, relax, S0=setup["S0"])
sig = add_noise(S, SIGMA_TRUE, "gaussian", np.random.default_rng(seed))

# --- numpy reference -------------------------------------------------
np_log_post, layout2, prior, ctx_np = np_make_log_post(
    sig, setup["aif"], setup["time_var"], TCXM, acq, subject, cfg_fit, cfg_b,
    wx_model=SSM3, aif_cfg=aif_cfg, relax_cfg=relax)

# --- jax side --------------------------------------------------------
ctx = static_ctx_from_configs(acq, subject, aif_cfg)
from dce_mri.signal_models import compute_S0
S0 = float(compute_S0(sig[None, :], acq)[0])   # measured baseline, as build_residual does
predict = make_predict("TCXM", setup["time_var"], setup["aif"], ctx, S0=S0,
                       wx=True, fit_tau_b=relax.fit_tau_b,
                       tau_b_fixed=relax.tau_b_fixed,   # 0.005, NOT make_predict's 0.05 default
                       upsample=acq.upsample)
residual = make_residual(predict, sig)
log_lik_jax = make_log_lik_jax(residual, noise="marginalized")

print("S0 used (jax):", S0)
rng = np.random.default_rng(7)
lo = np.array([0.05, 0.02, 0.002, 0.001, 0.1])
hi = np.array([0.45, 0.45, 0.02,  0.02,  1.5])
pts = rng.uniform(lo, hi, size=(300, 5))

ref = np.array([np_log_post(p) for p in pts])
new = np.array([float(log_lik_jax(jnp.asarray(p))) + float(prior.log_pdf(p)) for p in pts])
ok = np.isfinite(ref) & np.isfinite(new)
d = ref[ok] - new[ok]
print(f"n finite = {ok.sum()}/{len(pts)}")
print(f"offset mean = {d.mean():+.6f}   spread (max-min) = {d.max()-d.min():.3e}")
print("PASS" if (d.max()-d.min()) < 1e-6 else "FAIL -- densities differ by more than a constant")
