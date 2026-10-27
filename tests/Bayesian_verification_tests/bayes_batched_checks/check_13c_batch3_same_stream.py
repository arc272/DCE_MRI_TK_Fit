"""
check_13c: same-stream test for a batch of 3 (check_13b part 1 covered a
batch of 1). Three NXL fixed-delay lesions, each with its own delay and
nbase; n_tune=10, n_draws=40. Each lesion's draws in the batch vs the
same lesion fitted alone with the same seed.
"""
import os
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
Ls = [lesion(i) for i in range(4)][1:]            # check_13's B0, B1, B2
cb = BayesConfig(prior_choices={"rates": "loguniform", "volfrac": "beta"},
                 n_draws=40, n_tune=10, n_chains=4, noise="gauss_marginalized_sigma",
                 seed=1, backend="blackjax", step="nuts", dense_mass_matrix=True,
                 delay_mode="fixed", delay_bounds=(0.0, 25.0), progressbar=False)
bat = bb.fit_curve_bayes_batched([L["sig"] for L in Ls], [L["aif"] for L in Ls], t, TCXM,
                                 [L["acq"] for L in Ls], subj, cfg_fit, cb,
                                 delay_fixed=[L["delay"] for L in Ls],
                                 wx_model=NXL, aif_cfg=aif_cfg, verbose=False)
for k, L in enumerate(Ls):
    c = BayesConfig(**{**cb.__dict__, "delay_fixed": L["delay"]})
    rs = fit_curve_bayes(L["sig"], L["aif"], t, TCXM, L["acq"], subj, cfg_fit, c,
                         wx_model=NXL, aif_cfg=aif_cfg)
    sd = rs.chains.reshape(-1, rs.chains.shape[-1]).std(0)
    gap = np.max(np.abs(rs.chains - bat[k].chains) / sd, axis=(0, 2))
    print(f"B{k}: draw 0 {gap[0]:.1e}  draw 5 {gap[5]:.1e}  draw 10 {gap[10]:.1e}  "
          f"draw 20 {gap[20]:.1e}  draw 39 {gap[39]:.1e}  (posterior SDs)", flush=True)
